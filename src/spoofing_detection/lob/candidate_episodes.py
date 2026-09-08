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
from datetime import datetime, timezone
from typing import Any

import polars as pl

EPISODE_VERSION = "candidate_posture_overlap_v1"
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
    "cancel_reversion_weight": pl.Float64, "post_cancel_mid_reversion": pl.Float64,
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
    anchor_summary: pl.DataFrame
    actor_day_summary: pl.DataFrame


def _frame(rows: list[dict], schema: dict) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=schema, strict=True)


def _time(value: Any) -> datetime:
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value))
    return result.astimezone(timezone.utc).replace(tzinfo=None) if result.tzinfo else result


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
) -> EpisodeResult:
    """Aggregate all connected clusters, not merely cancellation winners.

    Outcomes are retrospective, with complete price-path coverage required.
    The existing cluster assignment is retained solely as unique withdrawal
    bookkeeping: changing the winning member inside an episode does not multiply
    its withdrawn quantity. No calibrated probability or significance is emitted.
    """
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
        if start.date() != end.date() or end < start:
            raise ValueError('execution spans trading days or has reversed timestamps')
        if int(row['cluster_first_sort_index']) > int(row['cluster_last_sort_index']):
            raise ValueError('execution has reversed source indices')
        _positive(row['execution_quantity'])
        lookup[key] = row
        scopes[key] = (row['partition_id'], actor_key, start.date(), row['execution_side'])

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
        if (first_index >= int(execution['cluster_first_sort_index'])
            or first_ts > _time(execution['cluster_start_ts'])
            or first_ts.date() != scopes[key][2]):
            raise ValueError('candidate provenance is not prior and same-day')
        lifecycle = (str(candidate['deceptive_order_id']), first_index)
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
    physical_assignments = set()
    for link in cancellation_links.iter_rows(named=True):
        key = _key(link)
        if key not in lookup or key not in parent:
            raise ValueError('withdrawal provenance references unknown candidate cluster')
        execution=lookup[key]
        _provenance(link, execution)
        if str(link['candidate_order_id']) not in {life[0] for life,_ in candidates_by_cluster[key]}:
            raise ValueError('withdrawal provenance references noncandidate order')
        timestamp=_time(link['cancel_event_ts'])
        if (timestamp.date() != scopes[key][2]
            or timestamp < _time(execution['cluster_end_ts'])
            or int(link['cancel_sort_index']) <= int(execution['cluster_last_sort_index'])):
            raise ValueError('withdrawal provenance is not causally after execution on same day')
        if not link['assigned_flag']:
            continue
        physical=(link['partition_id'],link['actor_key'],int(link['cancel_sort_index']),str(link['candidate_order_id']))
        if physical in physical_assignments:
            raise ValueError('physical cancellation assigned more than once')
        physical_assignments.add(physical)
        _positive(link['attributed_cancel_visible_qty'])
        assignments[root(key)].append(link)

    episodes, members, withdrawals = [], [], []
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
        lifecycle_qty={}
        for key in keys:
            for life,candidate in candidates_by_cluster[key]:
                lifecycle_qty[life]=max(lifecycle_qty.get(life,0.),float(candidate['deceptive_order_visible_qty_pre']))
        assigned=sorted(assignments[representative],key=lambda r:(r['cancel_sort_index'],r['candidate_order_id']))
        withdrawn=math.fsum(float(row['attributed_cancel_visible_qty']) for row in assigned)
        fpm_rows=[r for r in rows if _finite(r.get('favorable_mid_move_pre_fill')) is not None]
        rev_rows=[r for r in assigned if r.get('has_cancel_reversion_state') and
                  _finite(r.get('post_cancel_mid_reversion')) is not None and
                  (_finite(r.get('cancel_reversion_weight')) or 0.) > 0]
        fpm=(math.fsum(float(r['favorable_mid_move_pre_fill'])*float(r['execution_quantity']) for r in fpm_rows)/cluster_qty
             if len(fpm_rows)==len(rows) else None)
        reversion=(math.fsum(float(r['post_cancel_mid_reversion'])*float(r['cancel_reversion_weight']) for r in rev_rows)
                   /math.fsum(float(r['cancel_reversion_weight']) for r in rev_rows)
                   if assigned and len(rev_rows)==len(assigned) else None)
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
            candidate_order_count=len(lifecycle_qty),candidate_lifecycle_max_quantity_sum=math.fsum(lifecycle_qty.values()),
            unique_withdrawal_count=len(assigned),withdrawn_quantity=withdrawn,withdrawal_to_execution_ratio=withdrawn/cluster_qty,
            fpm_observed_cluster_count=len(fpm_rows),reversion_observed_withdrawal_count=len(rev_rows),
            favorable_mid_move=fpm,post_cancel_mid_reversion=reversion,price_path_observed=observed,
            gate_joint_smallness=withdrawn>cluster_qty,
            spoofing_compatible_episode=bool(assigned and withdrawn>cluster_qty and observed and fpm>0 and reversion>0))
        episodes.append(episode)
        members.extend(dict(episode_id=episode_id,**base,execution_cluster_id=r['execution_cluster_id'],
                            execution_anchor_mode=r['execution_anchor_mode'],execution_quantity=r['execution_quantity']) for r in rows)
        withdrawals.extend(dict(episode_id=episode_id,**base,**{k:r[k] for k in (
            'execution_cluster_id','execution_anchor_mode','candidate_order_id','cancel_sort_index',
            'attributed_cancel_visible_qty')},cancel_event_ts=_time(r['cancel_event_ts']),
            cancel_reversion_weight=_finite(r.get('cancel_reversion_weight')),
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
                         _frame(withdrawals,WITHDRAWAL_SCHEMA),_frame(anchors,ANCHOR_SCHEMA),_frame(days,DAY_SCHEMA))
