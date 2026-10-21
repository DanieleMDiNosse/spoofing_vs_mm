"""Joint, descriptive candidate-posture episodes; never independent intent labels.

Clusters sharing a candidate order lifecycle form a connected component inside
one actor/partition/day/side. A mixed-anchor episode is one observation, with
separate passive and aggressive quantities. Cluster and withdrawal membership
remain explicit; branch participation counts must not be added across branches.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

import polars as pl

EPISODE_VERSION = "candidate_posture_overlap_quantity_primary_v3_supported_cancellation"
IDENTITY = {
    "actor_key": pl.String, "actor_id": pl.String, "identity_level": pl.String,
    "identity_source": pl.String, "identity_fallback_flag": pl.Boolean,
}
BASE = {"partition_id": pl.String, "event_date": pl.Date, **IDENTITY}
EPISODE_SCHEMA = {
    "episode_id": pl.String, **BASE, "execution_side": pl.String,
    "episode_start_ts": pl.Datetime('us'), "episode_end_ts": pl.Datetime('us'),
    "cluster_count": pl.Int64, "passive_cluster_count": pl.Int64,
    "aggressive_cluster_count": pl.Int64, "mixed_anchor": pl.Boolean,
    "total_execution_quantity": pl.Float64, "passive_execution_quantity": pl.Float64,
    "aggressive_execution_quantity": pl.Float64, "execution_vwap": pl.Float64,
    "candidate_order_count": pl.Int64,
    "candidate_lifecycle_max_quantity_sum": pl.Float64,
    "unique_withdrawal_count": pl.Int64, "withdrawn_quantity": pl.Float64,
    "withdrawal_to_execution_ratio": pl.Float64,
    "primary_lifecycle_union_exposure_quantity": pl.Float64,
    "primary_raw_unique_cancellation_quantity": pl.Float64,
    "primary_supported_unique_cancellation_quantity": pl.Float64,
    "primary_withdrawal_to_lifecycle_exposure_ratio": pl.Float64,
    "primary_quantity_only_wmsci": pl.Float64,
    "fpm_observed_cluster_count": pl.Int64, "reversion_observed_withdrawal_count": pl.Int64,
    "favorable_mid_move": pl.Float64, "post_cancel_mid_reversion": pl.Float64,
    "price_path_observed": pl.Boolean, "gate_joint_smallness": pl.Boolean,
    "spoofing_compatible_episode": pl.Boolean,
}
MEMBER_SCHEMA = {
    "episode_id": pl.String, **BASE, "execution_cluster_id": pl.String,
    "execution_anchor_mode": pl.String, "execution_quantity": pl.Float64,
}
WITHDRAWAL_SCHEMA = {
    "episode_id": pl.String, **BASE, "execution_cluster_id": pl.String,
    "execution_anchor_mode": pl.String, "candidate_order_id": pl.String,
    "cancel_sort_index": pl.Int64, "cancel_event_ts": pl.Datetime('us'),
    "attributed_cancel_visible_qty": pl.Float64,
    "raw_cancel_visible_qty": pl.Float64,
    "supported_cancel_visible_qty": pl.Float64,
    "legacy_attributed_cancel_visible_qty": pl.Float64,
    "cancel_reversion_weight": pl.Float64, "post_cancel_mid_reversion": pl.Float64,
}
LIFECYCLE_SCHEMA = {
    "episode_id": pl.String, **BASE, "execution_side": pl.String,
    "candidate_order_id": pl.String, "lifecycle_first_seen_sort_index": pl.Int64,
    "lifecycle_first_seen_replay_ns": pl.Int64,
    "candidate_snapshot_count": pl.Int64,
    "admissible_pre_cancel_snapshot_count": pl.Int64,
    "lifecycle_union_exposure_quantity": pl.Float64,
    "physical_cancellation_count": pl.Int64,
    "raw_unique_cancellation_quantity": pl.Float64,
    "supported_unique_cancellation_quantity": pl.Float64,
    "legacy_attributed_cancellation_quantity": pl.Float64,
}
ANCHOR_SCHEMA = {
    "partition_id": pl.String, "event_date": pl.Date, "identity_level": pl.String,
    "execution_anchor_mode": pl.String, "episode_participation_count": pl.Int64,
    "strict_episode_participation_count": pl.Int64,
    "mixed_episode_participation_count": pl.Int64,
    "cluster_count": pl.Int64, "execution_quantity": pl.Float64,
}
DAY_SCHEMA = {
    **BASE, "episode_count": pl.Int64, "matched_episode_count": pl.Int64,
    "strict_episode_count": pl.Int64, "cluster_count": pl.Int64,
    "total_execution_quantity": pl.Float64, "withdrawn_quantity": pl.Float64,
}


@dataclass(frozen=True)
class EpisodeResult:
    episodes: pl.DataFrame
    members: pl.DataFrame
    withdrawals: pl.DataFrame
    candidate_lifecycles: pl.DataFrame
    anchor_summary: pl.DataFrame
    actor_day_summary: pl.DataFrame


def _frame(rows: list[dict], schema: dict) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema, strict=True)


def _time(value: Any) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return result.astimezone(timezone.utc).replace(tzinfo=None) if result.tzinfo else result


def _replay_ns(row: dict, field: str) -> int | None:
    value = row.get(field)
    return int(value) if value is not None else None


def _replay_date(value: int) -> date:
    return datetime.fromtimestamp(value // 1_000_000_000, timezone.utc).date()


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _positive(value: Any) -> float:
    number = _finite(value)
    if number is None or number <= 0:
        raise ValueError("episode quantities must be positive and finite")
    return number


def _nonnegative(value: Any) -> float | None:
    number = _finite(value)
    return number if number is not None and number >= 0 else None


def quantity_only_episode_wmsci(
    lifecycle_union_exposure_quantity: Any,
    total_execution_quantity: Any,
    raw_unique_cancellation_quantity: Any,
) -> float | None:
    """Primary quantity-only episode score with explicit null/zero semantics.

    P is the sum of one maximum pre-cancel profile snapshot per candidate
    lifecycle, Q is the sum of every member-cluster execution quantity, and U
    is the sum of unique physical cancellation quantities, each bounded by the
    maximum admissible pre-cancel snapshot of its lifecycle. Missing/non-finite
    inputs and Q <= 0 are undefined; observed P == 0 or U == 0 gives zero.
    """
    exposure = _nonnegative(lifecycle_union_exposure_quantity)
    execution = _finite(total_execution_quantity)
    cancellation = _nonnegative(raw_unique_cancellation_quantity)
    if exposure is None or cancellation is None or execution is None or execution <= 0:
        return None
    if exposure == 0 or cancellation == 0:
        return 0.0
    return (
        math.log1p(exposure / execution)
        * math.log1p(cancellation / execution)
        * (cancellation / exposure)
    )


def _raw_cancel_quantity(link: dict) -> float:
    """Use a raw pre-removal display, never a winner-level attribution cap."""
    for field in ("cancel_visible_qty", "visible_qty_pre_cancel"):
        if link.get(field) is not None:
            value = _nonnegative(link.get(field))
            if value is None:
                raise ValueError(f"invalid raw physical cancellation quantity: {field}")
            return value
    raise ValueError("assigned withdrawal lacks raw physical cancellation quantity")


def _key(row: dict) -> tuple:
    return row['partition_id'], row['execution_cluster_id']


def _provenance(row: dict, execution: dict) -> None:
    for field in (*IDENTITY, 'partition_id', 'execution_anchor_mode', 'execution_side'):
        if row.get(field) != execution.get(field):
            raise ValueError(f"episode provenance mismatch: {field}")
    expected_side = 'bid' if execution['execution_side'] == 'ask' else 'ask'
    if row.get('deceptive_side') != expected_side:
        raise ValueError("episode provenance mismatch: deceptive_side")


def build_candidate_episodes(
    executions: pl.DataFrame,
    candidate_orders: pl.DataFrame,
    cancellation_links: pl.DataFrame,
    *,
    require_v3_clocks: bool = False,
) -> EpisodeResult:
    """Aggregate all connected clusters, not merely cancellation winners.

    Outcomes are retrospective, with complete price-path coverage required.
    The existing cluster assignment is retained solely as unique withdrawal
    bookkeeping: changing the winning member inside an episode does not multiply
    its withdrawn quantity. No calibrated probability or significance is emitted.
    """
    if (require_v3_clocks and not candidate_orders.is_empty()
        and 'deceptive_order_first_seen_replay_ns' not in candidate_orders.columns):
        raise ValueError('missing v3 candidate clock column: deceptive_order_first_seen_replay_ns')
    lookup, scopes = {}, {}
    for row in executions.iter_rows(named=True):
        key = _key(row)
        if key in lookup:
            raise ValueError('duplicate execution cluster key')
        actor_key = row.get('actor_key') or ''
        level, separator, actor_id = actor_key.partition(':')
        if (separator != ':' or level not in {'client_original','firm'} or not actor_id
            or row.get('actor_id') != actor_id or row.get('identity_level') != level
            or row.get('identity_fallback_flag') != (level == 'firm')
            or row.get('identity_source') != ('FIRMID' if level == 'firm' else 'NMSC_ORIGINALCLIENTIDSHORTCODE')):
            raise ValueError('invalid episode actor provenance')
        if row['execution_anchor_mode'] not in {'passive','aggressive'} or row['execution_side'] not in {'bid','ask'}:
            raise ValueError('invalid episode execution provenance')
        start, end = _time(row['cluster_start_ts']), _time(row['cluster_end_ts'])
        start_replay_ns = _replay_ns(row, 'cluster_start_replay_time_ns')
        end_replay_ns = _replay_ns(row, 'cluster_end_replay_time_ns')
        if start_replay_ns is not None and end_replay_ns is not None:
            if _replay_date(start_replay_ns) != _replay_date(end_replay_ns) or end_replay_ns < start_replay_ns:
                raise ValueError('execution spans trading days or has reversed timestamps')
            scope_day = _replay_date(start_replay_ns)
        elif start_replay_ns is None and end_replay_ns is None:
            if start.date() != end.date() or end < start:
                raise ValueError('execution spans trading days or has reversed timestamps')
            scope_day = start.date()
        else:
            raise ValueError('incomplete v3 execution replay clock')
        if int(row['cluster_first_sort_index']) > int(row['cluster_last_sort_index']):
            raise ValueError('execution has reversed source indices')
        _positive(row['execution_quantity'])
        lookup[key] = row
        scopes[key] = (row['partition_id'], actor_key, scope_day, row['execution_side'])

    parent, owners = {}, {}
    candidates_by_cluster = defaultdict(list)

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    candidate_keys = set()
    for candidate in candidate_orders.iter_rows(named=True):
        key = _key(candidate)
        if key not in lookup:
            raise ValueError('candidate provenance references unknown cluster')
        execution = lookup[key]
        _provenance(candidate, execution)
        first_index = int(candidate['deceptive_order_first_seen_sort_index'])
        first_ts = _time(candidate['deceptive_order_first_seen_ts'])
        first_replay_ns = _replay_ns(candidate, 'deceptive_order_first_seen_replay_ns')
        cluster_start_replay_ns = _replay_ns(execution, 'cluster_start_replay_time_ns')
        if require_v3_clocks and (first_replay_ns is None or cluster_start_replay_ns is None):
            # Do not let a derived lifecycle enter a v3 episode via the legacy
            # display-time branch when its required sequence clock is absent.
            continue
        candidate_invalid = first_index >= int(execution['cluster_first_sort_index'])
        if first_replay_ns is not None and cluster_start_replay_ns is not None:
            v3_candidate = True
            candidate_invalid |= (first_replay_ns >= cluster_start_replay_ns
                                  or _replay_date(first_replay_ns) != scopes[key][2])
        else:
            v3_candidate = False
            candidate_invalid |= (first_ts > _time(execution['cluster_start_ts'])
                                  or first_ts.date() != scopes[key][2])
        if candidate_invalid:
            raise ValueError(
                "candidate provenance is not prior and same-day: "
                f"partition_id={candidate['partition_id']!r}, "
                f"execution_cluster_id={candidate['execution_cluster_id']!r}, "
                f"deceptive_order_id={candidate['deceptive_order_id']!r}, "
                f"candidate_first_sort_index={first_index}, "
                f"cluster_first_sort_index={execution['cluster_first_sort_index']}, "
                f"v3_candidate={v3_candidate}, "
                f"candidate_first_replay_ns={first_replay_ns!r}, "
                f"cluster_start_replay_ns={cluster_start_replay_ns!r}, "
                f"candidate_first_ts={first_ts.isoformat()}, "
                f"cluster_start_ts={_time(execution['cluster_start_ts']).isoformat()}, "
                f"scope_day={scopes[key][2].isoformat()}"
            )
        lifecycle = (str(candidate['deceptive_order_id']),
                     first_replay_ns if v3_candidate else first_index)
        if (key,lifecycle) in candidate_keys:
            raise ValueError('duplicate candidate lifecycle within cluster')
        candidate_keys.add((key,lifecycle))
        _positive(candidate['deceptive_order_visible_qty_pre'])
        candidates_by_cluster[key].append((lifecycle,candidate))
        parent.setdefault(key,key)
        link_key = (scopes[key], lifecycle)
        if link_key in owners:
            left,right=root(key),root(owners[link_key])
            # Deterministic union; final IDs also include sorted member keys.
            parent[max(left,right)] = min(left,right)
        else:
            owners[link_key]=key

    groups = defaultdict(list)
    for key in parent:
        groups[root(key)].append(key)
    assignments = defaultdict(list)
    assignments_by_lifecycle = defaultdict(lambda: defaultdict(list))
    physical_assignments = set()
    for link in cancellation_links.iter_rows(named=True):
        key = _key(link)
        if key not in lookup:
            raise ValueError('withdrawal provenance references unknown execution cluster')
        if key not in parent:
            if require_v3_clocks:
                # The linked candidate was excluded because its mandatory v3
                # lifecycle clock was unavailable; no withdrawal can revive it.
                continue
            raise ValueError('withdrawal provenance references unknown candidate cluster')
        execution=lookup[key]
        _provenance(link, execution)
        if str(link['candidate_order_id']) not in {life[0] for life,_ in candidates_by_cluster[key]}:
            raise ValueError('withdrawal provenance references noncandidate order')
        timestamp=_time(link['cancel_event_ts'])
        cancellation_removal_ns = _replay_ns(link, 'cancellation_removal_ns')
        cluster_end_fill_anchor_ns = _replay_ns(execution, 'cluster_end_fill_anchor_ns')
        cancellation_invalid = int(link['cancel_sort_index']) <= int(execution['cluster_last_sort_index'])
        if cancellation_removal_ns is not None and cluster_end_fill_anchor_ns is not None:
            cancellation_invalid |= (_replay_date(cancellation_removal_ns) != scopes[key][2]
                                   or cancellation_removal_ns - cluster_end_fill_anchor_ns < 0)
        else:
            cancellation_invalid |= (timestamp.date() != scopes[key][2]
                                   or timestamp < _time(execution['cluster_end_ts']))
        if cancellation_invalid:
            raise ValueError('withdrawal provenance is not causally after execution on same day')
        if not link['assigned_flag']:
            continue
        # A physical cancellation is one raw message inside one canonical
        # actor/day/side scope. Winner assignment only selects the unique
        # message; primary support is derived below from the full lifecycle,
        # never from a winner-level attributed quantity.
        physical = (
            link['partition_id'], link['actor_key'], link['deceptive_side'], scopes[key][2],
            int(link['cancel_sort_index']), str(link['candidate_order_id']),
        )
        if physical in physical_assignments:
            raise ValueError('physical cancellation assigned more than once')
        physical_assignments.add(physical)
        lifecycle_matches = [
            life for life, _ in candidates_by_cluster[key]
            if life[0] == str(link['candidate_order_id'])
        ]
        if len(lifecycle_matches) != 1:
            raise ValueError('withdrawal provenance has ambiguous candidate lifecycle')
        assigned_link = {**link, 'raw_cancel_visible_qty': _raw_cancel_quantity(link)}
        assignments[root(key)].append(assigned_link)
        assignments_by_lifecycle[root(key)][lifecycle_matches[0]].append(assigned_link)

    episodes, members, withdrawals, lifecycles = [], [], [], []
    for representative, keys in sorted(groups.items()):
        keys=sorted(keys)
        rows=[lookup[key] for key in keys]
        first=rows[0]
        base={field:first[field] for field in IDENTITY}
        base.update(partition_id=first['partition_id'],event_date=scopes[keys[0]][2])
        episode_id='EP-'+hashlib.sha256(json.dumps([EPISODE_VERSION,scopes[keys[0]],keys],default=str).encode()).hexdigest()[:24]
        cluster_qty=math.fsum(float(row['execution_quantity']) for row in rows)
        execution_vwap = math.fsum(
            _positive(row['execution_vwap']) * float(row['execution_quantity']) for row in rows
        ) / cluster_qty
        assigned=sorted(assignments[representative],key=lambda r:(r['cancel_sort_index'],r['candidate_order_id']))
        snapshots_by_lifecycle=defaultdict(list)
        for key in keys:
            for life,candidate in candidates_by_cluster[key]:
                snapshots_by_lifecycle[life].append((key,candidate))
        lifecycle_rows=[]
        for life,snapshots in sorted(snapshots_by_lifecycle.items()):
            physical=sorted(
                assignments_by_lifecycle[representative].get(life,[]),
                key=lambda link: (link['cancel_sort_index'], link['execution_cluster_id']),
            )
            first_cancel_index=min((int(link['cancel_sort_index']) for link in physical),default=None)
            admissible=[candidate for key,candidate in snapshots if first_cancel_index is None or int(lookup[key]['cluster_first_sort_index']) < first_cancel_index]
            exposure=max((float(candidate['deceptive_order_visible_qty_pre']) for candidate in admissible),default=0.0)
            raw_quantity=math.fsum(float(link['raw_cancel_visible_qty']) for link in physical)
            # The raw cancellation message can include replacement or other
            # increase not observed in the candidate's pre-cancel profile.
            # Allocate support in physical-message order, once per lifecycle,
            # so U_lifecycle <= P_lifecycle and hence episode A=U/P <= 1.
            remaining_support=exposure
            for link in physical:
                supported=min(float(link['raw_cancel_visible_qty']), remaining_support)
                link['supported_cancel_visible_qty']=supported
                remaining_support-=supported
            supported_quantity=math.fsum(float(link['supported_cancel_visible_qty']) for link in physical)
            legacy_quantity=math.fsum(float(link.get('attributed_cancel_visible_qty') or 0.0) for link in physical)
            example=snapshots[0][1]
            lifecycle_rows.append((life,dict(
                episode_id=episode_id,**base,execution_side=first['execution_side'],candidate_order_id=life[0],
                lifecycle_first_seen_sort_index=int(example['deceptive_order_first_seen_sort_index']),
                lifecycle_first_seen_replay_ns=_replay_ns(example,'deceptive_order_first_seen_replay_ns'),
                candidate_snapshot_count=len(snapshots),admissible_pre_cancel_snapshot_count=len(admissible),
                lifecycle_union_exposure_quantity=exposure,physical_cancellation_count=len(physical),
                raw_unique_cancellation_quantity=raw_quantity,
                supported_unique_cancellation_quantity=supported_quantity,
                legacy_attributed_cancellation_quantity=legacy_quantity,
                )))
        primary_exposure=math.fsum(row['lifecycle_union_exposure_quantity'] for _,row in lifecycle_rows)
        raw_withdrawn=math.fsum(float(row['raw_cancel_visible_qty']) for row in assigned)
        supported_withdrawn=math.fsum(float(row['supported_cancel_visible_qty']) for row in assigned)
        score=quantity_only_episode_wmsci(primary_exposure,cluster_qty,supported_withdrawn)
        fpm_rows=[r for r in rows if _finite(r.get('favorable_mid_move_pre_fill')) is not None]
        supported_withdrawals=[r for r in assigned if float(r['supported_cancel_visible_qty']) > 0]
        rev_rows=[r for r in supported_withdrawals if r.get('has_cancel_reversion_state') and
                  _finite(r.get('post_cancel_mid_reversion')) is not None and
                  float(r['supported_cancel_visible_qty']) > 0]
        fpm=(math.fsum(float(r['favorable_mid_move_pre_fill'])*float(r['execution_quantity']) for r in fpm_rows)/cluster_qty
             if len(fpm_rows)==len(rows) else None)
        reversion=(math.fsum(float(r['post_cancel_mid_reversion'])*float(r['supported_cancel_visible_qty']) for r in rev_rows)
                   /math.fsum(float(r['supported_cancel_visible_qty']) for r in rev_rows)
                   if supported_withdrawals and len(rev_rows)==len(supported_withdrawals) else None)
        quantities={a:math.fsum(float(r['execution_quantity']) for r in rows if r['execution_anchor_mode']==a)
                    for a in ('passive','aggressive')}
        counts={a:sum(r['execution_anchor_mode']==a for r in rows) for a in quantities}
        observed=fpm is not None and reversion is not None
        episode=dict(episode_id=episode_id,**base,execution_side=first['execution_side'],
            episode_start_ts=min(_time(c['deceptive_order_first_seen_ts']) for key in keys for _,c in candidates_by_cluster[key]),
            episode_end_ts=max([_time(r['cluster_end_ts']) for r in rows]+[_time(r['cancel_event_ts']) for r in assigned]),
            cluster_count=len(rows),passive_cluster_count=counts['passive'],aggressive_cluster_count=counts['aggressive'],
            mixed_anchor=all(counts.values()),total_execution_quantity=cluster_qty,
            passive_execution_quantity=quantities['passive'],aggressive_execution_quantity=quantities['aggressive'],
            execution_vwap=execution_vwap,
            candidate_order_count=len(lifecycle_rows),candidate_lifecycle_max_quantity_sum=primary_exposure,
            unique_withdrawal_count=len(assigned),withdrawn_quantity=supported_withdrawn,withdrawal_to_execution_ratio=supported_withdrawn/cluster_qty,
            primary_lifecycle_union_exposure_quantity=primary_exposure,
            primary_raw_unique_cancellation_quantity=raw_withdrawn,
            primary_supported_unique_cancellation_quantity=supported_withdrawn,
            primary_withdrawal_to_lifecycle_exposure_ratio=(supported_withdrawn/primary_exposure if primary_exposure>0 else None),
            primary_quantity_only_wmsci=score,
            fpm_observed_cluster_count=len(fpm_rows),reversion_observed_withdrawal_count=len(rev_rows),
            favorable_mid_move=fpm,post_cancel_mid_reversion=reversion,price_path_observed=observed,
            gate_joint_smallness=supported_withdrawn>cluster_qty,
            spoofing_compatible_episode=bool(supported_withdrawals and supported_withdrawn>cluster_qty and observed and fpm>0 and reversion>0))
        episodes.append(episode)
        lifecycles.extend(row for _,row in lifecycle_rows)
        members.extend(dict(episode_id=episode_id,**base,execution_cluster_id=r['execution_cluster_id'],
                            execution_anchor_mode=r['execution_anchor_mode'],execution_quantity=r['execution_quantity']) for r in rows)
        withdrawals.extend(dict(episode_id=episode_id,**base,**{k:r[k] for k in (
            'execution_cluster_id','execution_anchor_mode','candidate_order_id','cancel_sort_index',
            'attributed_cancel_visible_qty')},cancel_event_ts=_time(r['cancel_event_ts']),
            raw_cancel_visible_qty=float(r['raw_cancel_visible_qty']),
            supported_cancel_visible_qty=float(r['supported_cancel_visible_qty']),
            legacy_attributed_cancel_visible_qty=_finite(r.get('attributed_cancel_visible_qty')),
            cancel_reversion_weight=float(r['supported_cancel_visible_qty']),
            post_cancel_mid_reversion=_finite(r.get('post_cancel_mid_reversion'))) for r in assigned)

    anchor_groups, day_groups = defaultdict(list), defaultdict(list)
    for row in episodes:
        day_groups[tuple(row[k] for k in BASE)].append(row)
        for anchor in ('passive','aggressive'):
            if row[f'{anchor}_cluster_count']:
                anchor_groups[(row['partition_id'],row['event_date'],row['identity_level'],anchor)].append(row)
    anchors=[]
    for (partition,day,level,anchor),rows in sorted(anchor_groups.items()):
        anchors.append(dict(partition_id=partition,event_date=day,identity_level=level,execution_anchor_mode=anchor,
            episode_participation_count=len(rows),strict_episode_participation_count=sum(r['spoofing_compatible_episode'] for r in rows),
            mixed_episode_participation_count=sum(r['mixed_anchor'] for r in rows),
            cluster_count=sum(r[f'{anchor}_cluster_count'] for r in rows),
            execution_quantity=math.fsum(r[f'{anchor}_execution_quantity'] for r in rows)))
    days=[]
    for key,rows in sorted(day_groups.items()):
        days.append(dict(zip(BASE,key),episode_count=len(rows),
            matched_episode_count=sum(r['unique_withdrawal_count']>0 for r in rows),
            strict_episode_count=sum(r['spoofing_compatible_episode'] for r in rows),
            cluster_count=sum(r['cluster_count'] for r in rows),
            total_execution_quantity=math.fsum(r['total_execution_quantity'] for r in rows),
            withdrawn_quantity=math.fsum(r['withdrawn_quantity'] for r in rows)))
    return EpisodeResult(_frame(episodes,EPISODE_SCHEMA),_frame(members,MEMBER_SCHEMA),
                         _frame(withdrawals,WITHDRAWAL_SCHEMA),_frame(lifecycles,LIFECYCLE_SCHEMA),
                         _frame(anchors,ANCHOR_SCHEMA),_frame(days,DAY_SCHEMA))
