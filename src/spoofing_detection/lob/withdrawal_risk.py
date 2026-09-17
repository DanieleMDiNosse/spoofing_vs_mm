"""Pre-event top-of-book time-at-risk accounting for physical order lifecycles.

This module deliberately does not select spoofing candidates or execution windows.
It turns immutable ``ReplayObservation`` boundaries into auditable time-at-risk
spells and explicit competing withdrawal transitions.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import math
from numbers import Integral, Real
from typing import Any

import polars as pl

from .actor_identity import ActorIdentity, actor_identity_from_order
from .replay_observation import ReplayObservation, ReplayOrder, replay_lob

CLOCK_REGRESSION_POLICY = "quarantine_until_prior_accepted_high_watermark_never_reorder_or_clamp"

ACTOR_SCHEMA: dict[str, pl.DataType] = {
    "actor_key": pl.String,
    "actor_id": pl.String,
    "identity_level": pl.String,
    "identity_source": pl.String,
    "identity_fallback_flag": pl.Boolean,
}
SPELL_SCHEMA: dict[str, pl.DataType] = {
    "risk_spell_id": pl.String,
    "partition_id": pl.String,
    "event_date": pl.Date,
    **ACTOR_SCHEMA,
    "side": pl.String,
    "start_ts": pl.Datetime("us"),
    "end_ts": pl.Datetime("us"),
    "duration_seconds": pl.Float64,
    "interval_count": pl.Int64,
    "end_reason": pl.String,
}
INTERVAL_SCHEMA: dict[str, pl.DataType] = {
    "risk_interval_id": pl.String,
    "risk_spell_id": pl.String,
    "partition_id": pl.String,
    "event_date": pl.Date,
    **ACTOR_SCHEMA,
    "side": pl.String,
    "start_ts": pl.Datetime("us"),
    # Canonical feed boundary establishing this outgoing interval.  Synthetic
    # age-expiry boundaries deliberately have no feed sort order.
    "start_sort_index": pl.Int64,
    "end_ts": pl.Datetime("us"),
    "duration_seconds": pl.Float64,
    "member_count": pl.Int64,
    "eligible_visible_qty": pl.Float64,
    "mean_member_age_seconds": pl.Float64,
    "oldest_member_age_seconds": pl.Float64,
    "exposure_quantity": pl.Float64,
    "membership_signature": pl.String,
}
MEMBERSHIP_SCHEMA: dict[str, pl.DataType] = {
    "risk_membership_id": pl.String,
    "physical_order_key": pl.String,
    "partition_id": pl.String,
    "event_date": pl.Date,
    "order_id": pl.String,
    "first_seen_sort_index": pl.Int64,
    **ACTOR_SCHEMA,
    "side": pl.String,
    "eligible_start_ts": pl.Datetime("us"),
    "eligible_end_ts": pl.Datetime("us"),
    "duration_seconds": pl.Float64,
    "start_reason": pl.String,
    "end_reason": pl.String,
}
WITHDRAWAL_SCHEMA: dict[str, pl.DataType] = {
    "withdrawal_event_id": pl.String,
    "physical_order_key": pl.String,
    "partition_id": pl.String,
    "event_date": pl.Date,
    "sort_index": pl.Int64,
    "event_ts": pl.Datetime("us"),
    **ACTOR_SCHEMA,
    "side": pl.String,
    "order_id": pl.String,
    "first_seen_sort_index": pl.Int64,
    "visible_qty_removed": pl.Float64,
    "transition_reason": pl.String,
}
TRANSITION_SCHEMA: dict[str, pl.DataType] = {
    "transition_id": pl.String,
    "physical_order_key": pl.String,
    "partition_id": pl.String,
    "event_date": pl.Date,
    "sort_index": pl.Int64,
    "event_ts": pl.Datetime("us"),
    **ACTOR_SCHEMA,
    "side": pl.String,
    "order_id": pl.String,
    "first_seen_sort_index": pl.Int64,
    "visible_qty_pre": pl.Float64,
    "transition_reason": pl.String,
}
DIAGNOSTIC_SCHEMA: dict[str, pl.DataType] = {
    "partition_id": pl.String,
    "sort_index": pl.Int64,
    "event_ts": pl.Datetime("us"),
    "reason": pl.String,
    "detail": pl.String,
}
ACTOR_SUMMARY_SCHEMA: dict[str, pl.DataType] = {
    "partition_id": pl.String,
    "event_date": pl.Date,
    **ACTOR_SCHEMA,
    "side": pl.String,
    "time_at_risk_seconds": pl.Float64,
    "withdrawal_count": pl.Int64,
    "withdrawal_visible_qty": pl.Float64,
    "withdrawal_intensity": pl.Float64,
}
EXPOSURE_INTERVAL_SCHEMA: dict[str, pl.DataType] = {
    "exposure_interval_id": pl.String,
    "risk_interval_id": pl.String,
    "risk_spell_id": pl.String,
    "withdrawal_event_id": pl.String,
    "partition_id": pl.String,
    "event_date": pl.Date,
    **ACTOR_SCHEMA,
    "side": pl.String,
    "start_ts": pl.Datetime("us"),
    "end_ts": pl.Datetime("us"),
    "duration_seconds": pl.Float64,
    "own_passive": pl.Boolean,
    "own_aggressive": pl.Boolean,
    "other_passive": pl.Boolean,
    "other_aggressive": pl.Boolean,
    "identity_ambiguous_cluster_count": pl.Int64,
    "exposure_mask": pl.String,
    "execution_cluster_ids": pl.String,
    "execution_anchor_mode": pl.String,
    "execution_quantity": pl.Float64,
    "contrast_label": pl.String,
}
EXPOSURE_CONTRAST_SCHEMA: dict[str, pl.DataType] = {
    "risk_interval_id": pl.String,
    "withdrawal_event_id": pl.String,
    "partition_id": pl.String,
    "event_date": pl.Date,
    **ACTOR_SCHEMA,
    "side": pl.String,
    "start_ts": pl.Datetime("us"),
    "end_ts": pl.Datetime("us"),
    "duration_seconds": pl.Float64,
    "contrast_label": pl.String,
    "execution_anchor_mode": pl.String,
    "execution_quantity": pl.Float64,
    "identity_ambiguous_cluster_count": pl.Int64,
}
EXPOSURE_STRATUM_SCHEMA: dict[str, pl.DataType] = {
    "contrast_label": pl.String,
    "exposure_mask": pl.String,
    "interval_count": pl.Int64,
    "duration_seconds": pl.Float64,
    "withdrawal_count": pl.Int64,
}
_FINALIZE = object()


@dataclass(frozen=True)
class WithdrawalRiskResult:
    spells: pl.DataFrame
    intervals: pl.DataFrame
    membership: pl.DataFrame
    withdrawal_events: pl.DataFrame
    transitions: pl.DataFrame
    diagnostics: pl.DataFrame
    actor_summary: pl.DataFrame
    exposure_intervals: pl.DataFrame
    exposure_contrasts: pl.DataFrame
    exposure_strata: pl.DataFrame


@dataclass(frozen=True)
class _Eligible:
    physical_key: str
    partition_id: str | None
    event_date: date
    order_id: str
    first_seen_sort_index: int
    actor: ActorIdentity
    side: str
    visible_qty: float
    first_seen_ts: datetime

    @property
    def group(self) -> tuple[str | None, date, str, str]:
        return (self.partition_id, self.event_date, self.actor.actor_key, self.side)


def _empty(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _frame(rows: list[dict[str, Any]], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return _empty(schema)
    frame = pl.DataFrame(rows, infer_schema_length=None)
    return frame.select(pl.col(column).cast(dtype, strict=False) for column, dtype in schema.items())


def _event_date(observation: ReplayObservation) -> date:
    raw = observation.event.get("TRADEDATE")
    if raw is not None:
        try:
            return date.fromisoformat(str(raw)[:10])
        except ValueError:
            pass
    if observation.event_ts is not None:
        return observation.event_ts.date()
    # This value is only used for diagnostic-safe records with no usable clock.
    return date.min


def _order_numeric_issue(order: ReplayOrder) -> str | None:
    """Return the rejected eligibility field, without coercing malformed data."""
    for field in ("price", "leaves_qty", "displayed_qty"):
        value = getattr(order, field)
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            return f"order_id={order.order_id!r} field={field} value={value!r}"
    return None


def _valid_orders(
    observation: ReplayObservation,
    orders: Mapping[str, ReplayOrder],
    *,
    phase: str,
    diagnostics: list[dict[str, Any]],
) -> dict[str, ReplayOrder]:
    """Quarantine orders whose numeric book fields cannot safely define risk."""
    valid: dict[str, ReplayOrder] = {}
    for order_id, order in sorted(orders.items(), key=lambda item: str(item[0])):
        issue = _order_numeric_issue(order)
        if issue is None:
            valid[order_id] = order
            continue
        diagnostics.append({
            "partition_id": observation.partition_id,
            "sort_index": observation.sort_index,
            "event_ts": observation.event_ts,
            "reason": "invalid_order_numeric",
            "detail": f"phase={phase} {issue}",
        })
    return valid


def _visible_qty(order: ReplayOrder) -> float:
    if order.leaves_qty <= 0 or order.displayed_qty <= 0:
        return 0.0
    return float(order.displayed_qty)


def _physical_key(partition_id: str | None, event_date: date, order: ReplayOrder) -> str:
    return "|".join((str(partition_id or ""), event_date.isoformat(), str(order.order_id), str(order.first_seen_sort_index)))


def _identity_fields(actor: ActorIdentity) -> dict[str, Any]:
    return {
        "actor_key": actor.actor_key,
        "actor_id": actor.actor_id,
        "identity_level": actor.identity_level,
        "identity_source": actor.identity_source,
        "identity_fallback_flag": actor.identity_fallback_flag,
    }


def _eligible_orders(
    orders: Mapping[str, ReplayOrder], *, partition_id: str | None, event_date: date,
    at_ts: datetime, first_seen_ts: Mapping[str, datetime], top_n: int, max_age: timedelta,
) -> dict[str, _Eligible]:
    """Return pre/post boundary eligible orders using canonical visible top-N ranks."""
    prices: dict[str, set[float]] = {"bid": set(), "ask": set()}
    for _, order in sorted(orders.items(), key=lambda item: str(item[0])):
        if order.side in prices and _visible_qty(order) > 0:
            prices[order.side].add(float(order.price))
    ranked = {
        "bid": set(sorted(prices["bid"], reverse=True)[:top_n]),
        "ask": set(sorted(prices["ask"])[:top_n]),
    }
    result: dict[str, _Eligible] = {}
    for _, order in sorted(orders.items(), key=lambda item: str(item[0])):
        actor = actor_identity_from_order(order.to_active_order())
        if actor is None or order.side not in ranked or _visible_qty(order) <= 0:
            continue
        if float(order.price) not in ranked[order.side]:
            continue
        key = _physical_key(partition_id, event_date, order)
        created = first_seen_ts.get(key)
        # Eligibility is half-open in time: an order expires at the exact
        # ``first_seen + max_age`` boundary rather than after it.
        if created is None or at_ts < created or at_ts - created >= max_age:
            continue
        result[key] = _Eligible(
            physical_key=key, partition_id=partition_id, event_date=event_date,
            order_id=order.order_id, first_seen_sort_index=order.first_seen_sort_index,
            actor=actor, side=order.side, visible_qty=_visible_qty(order), first_seen_ts=created,
        )
    return result


def _signature(members: Mapping[str, _Eligible]) -> str:
    return ";".join(f"{key}:{members[key].visible_qty:.12g}" for key in sorted(members))


def _coverage_end_for(value: datetime | Mapping[str, datetime] | None, partition_id: str | None) -> datetime | None:
    if value is None or isinstance(value, datetime):
        return value
    return value.get(str(partition_id)) if partition_id is not None else None


def _withdrawal_risk_observer(
    *, top_n: int, max_order_age_seconds: float,
    coverage_end: datetime | Mapping[str, datetime] | None = None,
) -> Any:
    """Build disjoint actor-side time-at-risk spells from canonical observations.

    Observations are consumed in supplied canonical order. A timestamp regression is
    quarantined in diagnostics rather than silently sorted into a different event
    sequence. Cancellation outcomes are emitted only from an explicit ``cancel``
    event whose *pre-event* physical order was eligible.
    """
    if type(top_n) is not int or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    if isinstance(max_order_age_seconds, bool) or not isinstance(max_order_age_seconds, Real) or (
        not math.isfinite(max_order_age_seconds)
    ) or max_order_age_seconds < 0:
        raise ValueError("max_order_age_seconds must be a finite non-negative real")
    max_age = timedelta(seconds=max_order_age_seconds)
    first_seen_ts: dict[str, datetime] = {}
    interval_rows: list[dict[str, Any]] = []
    spell_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    withdrawal_rows: list[dict[str, Any]] = []
    emitted_withdrawal_keys: set[str] = set()
    transition_rows: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    state: dict[str, _Eligible] = {}
    membership_open: dict[str, tuple[_Eligible, datetime, str]] = {}
    spell_open: dict[tuple[str | None, date, str, str], tuple[str, datetime, ActorIdentity]] = {}
    current_partition: str | None = None
    current_date: date | None = None
    last_ts: datetime | None = None
    clock_high_watermark: datetime | None = None
    last_boundary_sort_index: int | None = None
    interval_counter = 0
    spell_counter = 0
    membership_counter = 0

    def add_interval(start: datetime, end: datetime, *, start_sort_index: int | None) -> None:
        nonlocal interval_counter
        if end <= start:
            return
        grouped: dict[tuple[str | None, date, str, str], dict[str, _Eligible]] = defaultdict(dict)
        for key, item in sorted(state.items()):
            grouped[item.group][key] = item
        for group in sorted(grouped, key=str):
            members = grouped[group]
            if not members:
                continue
            spell_id, _, actor = spell_open[group]
            member_ages = [
                max(0.0, (start - item.first_seen_ts).total_seconds())
                for _, item in sorted(members.items())
            ]
            interval_counter += 1
            interval_rows.append({
                "risk_interval_id": f"risk-interval-{interval_counter}", "risk_spell_id": spell_id,
                "partition_id": group[0], "event_date": group[1], **_identity_fields(actor), "side": group[3],
                "start_ts": start, "start_sort_index": start_sort_index,
                "end_ts": end, "duration_seconds": (end - start).total_seconds(),
                "member_count": len(members), "eligible_visible_qty": sum(item.visible_qty for _, item in sorted(members.items())),
                "mean_member_age_seconds": sum(member_ages) / len(member_ages),
                "oldest_member_age_seconds": max(member_ages),
                "exposure_quantity": None, "membership_signature": _signature(members),
            })

    def open_spell(item: _Eligible, at: datetime) -> None:
        nonlocal spell_counter
        group = item.group
        if group not in spell_open:
            spell_counter += 1
            spell_open[group] = (f"risk-spell-{spell_counter}", at, item.actor)

    def close_spell(group: tuple[str | None, date, str, str], at: datetime, reason: str) -> None:
        open_value = spell_open.pop(group, None)
        if open_value is None:
            return
        spell_id, start, actor = open_value
        duration = (at - start).total_seconds()
        if duration <= 0:
            return
        spell_rows.append({
            "risk_spell_id": spell_id, "partition_id": group[0], "event_date": group[1],
            **_identity_fields(actor), "side": group[3], "start_ts": start, "end_ts": at,
            "duration_seconds": duration, "interval_count": 0, "end_reason": reason,
        })

    def close_membership(key: str, at: datetime, reason: str) -> None:
        nonlocal membership_counter
        open_value = membership_open.pop(key, None)
        if open_value is None:
            return
        item, start, start_reason = open_value
        duration = (at - start).total_seconds()
        if duration <= 0:
            return
        membership_counter += 1
        membership_rows.append({
            "risk_membership_id": f"risk-membership-{membership_counter}", "physical_order_key": key,
            "partition_id": item.partition_id, "event_date": item.event_date, "order_id": item.order_id,
            "first_seen_sort_index": item.first_seen_sort_index, **_identity_fields(item.actor), "side": item.side,
            "eligible_start_ts": start, "eligible_end_ts": at, "duration_seconds": duration,
            "start_reason": start_reason, "end_reason": reason,
        })

    def set_state(next_state: dict[str, _Eligible], at: datetime, *, transition_reason: str | None = None,
                  event_sort_index: int | None = None, event_order_id: str | None = None,
                  outgoing_reason: str = "loss_of_eligibility") -> None:
        nonlocal state
        before_groups = {item.group for item in state.values()}
        after_groups = {item.group for item in next_state.values()}
        removed = sorted(set(state) - set(next_state))
        added = sorted(set(next_state) - set(state))
        identity_changes = [
            (key, state[key], next_state[key])
            for key in sorted(set(state) & set(next_state))
            if state[key].actor != next_state[key].actor or state[key].side != next_state[key].side
        ]
        for key in removed:
            item = state[key]
            # A market event can alter another order's top-N rank. Only the
            # event's own physical order gets fill/modify/cancel attribution;
            # all displaced orders have a plain loss-of-eligibility transition.
            reason = (transition_reason or outgoing_reason) if item.order_id == event_order_id else outgoing_reason
            close_membership(key, at, reason)
            if event_sort_index is not None:
                transition_rows.append({
                    "transition_id": f"transition-{len(transition_rows) + 1}", "physical_order_key": key,
                    "partition_id": item.partition_id, "event_date": item.event_date, "sort_index": event_sort_index,
                    "event_ts": at, **_identity_fields(item.actor), "side": item.side, "order_id": item.order_id,
                    "first_seen_sort_index": item.first_seen_sort_index, "visible_qty_pre": item.visible_qty,
                    "transition_reason": reason,
                })
        for key in added:
            membership_open[key] = (next_state[key], at, "eligible_entry")
        # A physical lifecycle can be retained across a feed correction while
        # its canonical actor or side changes. Its membership is actor-side
        # specific, so never let the prior identity accrue across that boundary.
        for key, previous, current in identity_changes:
            close_membership(key, at, "identity_change")
            membership_open[key] = (current, at, "identity_change")
            if event_sort_index is not None:
                transition_rows.append({
                    "transition_id": f"transition-{len(transition_rows) + 1}", "physical_order_key": key,
                    "partition_id": previous.partition_id, "event_date": previous.event_date,
                    "sort_index": event_sort_index, "event_ts": at, **_identity_fields(previous.actor),
                    "side": previous.side, "order_id": previous.order_id,
                    "first_seen_sort_index": previous.first_seen_sort_index,
                    "visible_qty_pre": previous.visible_qty, "transition_reason": "identity_change",
                })
        # A retained physical member whose displayed quantity changes must split
        # the interval, but it does not end its spell or become a withdrawal.
        # Retain an explicit competing transition for partial fills/modifications.
        if event_sort_index is not None and transition_reason in {"fill", "modify"}:
            for key in sorted(set(state) & set(next_state)):
                previous, current = state[key], next_state[key]
                if previous.order_id == event_order_id and previous.visible_qty != current.visible_qty:
                    transition_rows.append({
                        "transition_id": f"transition-{len(transition_rows) + 1}", "physical_order_key": key,
                        "partition_id": previous.partition_id, "event_date": previous.event_date,
                        "sort_index": event_sort_index, "event_ts": at, **_identity_fields(previous.actor),
                        "side": previous.side, "order_id": previous.order_id,
                        "first_seen_sort_index": previous.first_seen_sort_index,
                        "visible_qty_pre": previous.visible_qty, "transition_reason": transition_reason,
                    })
        for group in sorted(after_groups - before_groups, key=str):
            item = next(item for _, item in sorted(next_state.items()) if item.group == group)
            open_spell(item, at)
        for group in sorted(before_groups - after_groups, key=str):
            reason = "identity_change" if any(previous.group == group for _, previous, _ in identity_changes) else (
                transition_reason or outgoing_reason
            )
            close_spell(group, at, reason)
        state = next_state

    def close_all(at: datetime, reason: str) -> None:
        nonlocal state
        add_interval(
            last_ts if last_ts is not None else at,
            at,
            start_sort_index=last_boundary_sort_index,
        )
        groups = {item.group for item in state.values()}
        for key in sorted(state):
            close_membership(key, at, reason)
        for group in sorted(groups, key=str):
            close_spell(group, at, reason)
        state = {}

    def advance_time_to(target: datetime, *, event_sort_index: int | None = None) -> None:
        """Emit every age-expiry boundary through ``target`` before another boundary."""
        nonlocal last_ts, last_boundary_sort_index
        while last_ts is not None:
            expiries = [item.first_seen_ts + max_age for _, item in sorted(state.items())]
            if not expiries:
                return
            expiry = min(expiries)
            if expiry > target:
                return
            if expiry < last_ts:
                raise AssertionError("eligible order expired before the current risk boundary")
            add_interval(last_ts, expiry, start_sort_index=last_boundary_sort_index)
            next_state = {
                key: item for key, item in sorted(state.items())
                if item.first_seen_ts + max_age > expiry
            }
            set_state(
                next_state,
                expiry,
                transition_reason="age_expiry",
                event_sort_index=event_sort_index,
                outgoing_reason="age_expiry",
            )
            last_ts = expiry
            # Synthetic age boundaries have no canonical feed position.
            last_boundary_sort_index = None

    while (observation := (yield)) is not _FINALIZE:
        event_ts = observation.event_ts
        event_date = _event_date(observation)
        partition_changed = current_partition is not None and (
            observation.partition_id != current_partition or event_date != current_date
        )
        if partition_changed:
            clock_high_watermark = None
        if event_ts is None:
            if partition_changed and last_ts is not None:
                # A partition identity is observable even when its first clock
                # is not. End the preceding lifecycle at its last valid clock;
                # never borrow a later timestamp or carry its state forward.
                close_all(last_ts, "partition_boundary")
                last_ts = None
                last_boundary_sort_index = None
                first_seen_ts = {}
            elif last_ts is not None:
                # The LOB boundary itself may have changed, but without a clock
                # its effect is unobservable. Censor at the last valid boundary
                # and restart later coverage from the next valid pre/post state.
                close_all(last_ts, "clock_ambiguous")
                last_ts = None
                last_boundary_sort_index = None
                first_seen_ts = {}
            diagnostics.append({"partition_id": observation.partition_id, "sort_index": observation.sort_index,
                                "event_ts": None, "reason": "missing_event_timestamp", "detail": "observation quarantined"})
            current_partition, current_date = observation.partition_id, event_date
            continue
        if clock_high_watermark is not None and event_ts < clock_high_watermark:
            # Quarantine until the clock catches up to the last accepted event.
            # Clearing last_ts alone would let the next regressed event reopen
            # risk over already-accounted clock time. Never sort or clamp clocks.
            previous_ts = clock_high_watermark
            if last_ts is not None:
                close_all(last_ts, "clock_ambiguous")
            last_ts = None
            last_boundary_sort_index = None
            first_seen_ts = {}
            diagnostics.append({"partition_id": observation.partition_id, "sort_index": observation.sort_index,
                                "event_ts": event_ts, "reason": "clock_regression", "detail": f"previous={previous_ts.isoformat()}"})
            current_partition, current_date = observation.partition_id, event_date
            continue
        if partition_changed and last_ts is not None:
            # The first timestamp in the next partition is the observable end
            # of the prior partition's coverage. A scalar applies only at the
            # terminal partition; a mapped end is capped at this next clock.
            configured_end = _coverage_end_for(coverage_end, current_partition)
            if isinstance(coverage_end, Mapping) and configured_end is not None:
                end = configured_end
                if end < last_ts:
                    raise ValueError("coverage_end precedes the final observed timestamp")
                end = min(end, event_ts)
            else:
                end = event_ts
            if end < last_ts:
                raise ValueError("coverage_end precedes the final observed timestamp")
            advance_time_to(end, event_sort_index=observation.sort_index)
            close_all(end, "partition_boundary")
            last_ts = None
            last_boundary_sort_index = None
            first_seen_ts = {}
        current_partition, current_date = observation.partition_id, event_date
        pre_orders = _valid_orders(observation, observation.pre_active_orders, phase="pre", diagnostics=diagnostics)
        post_orders = _valid_orders(observation, observation.post_active_orders, phase="post", diagnostics=diagnostics)
        # Populate physical lifecycle clocks from the pre-state if a caller starts midstream.
        for _, order in sorted(pre_orders.items(), key=lambda item: str(item[0])):
            first_seen_ts.setdefault(_physical_key(current_partition, current_date, order), event_ts)
        # Age expiry is a real boundary even where there is no feed row.
        advance_time_to(event_ts, event_sort_index=observation.sort_index)
        if last_ts is not None:
            add_interval(last_ts, event_ts, start_sort_index=last_boundary_sort_index)
        # The explicit cancellation numerator uses the *pre-event* eligible map.
        if observation.event.get("event_class") == "cancel":
            order = pre_orders.get(str(observation.event.get("ORDERID")))
            if order is not None:
                key = _physical_key(current_partition, current_date, order)
                eligible = state.get(key)
                if eligible is not None and key not in emitted_withdrawal_keys:
                    withdrawal_rows.append({
                        "withdrawal_event_id": f"withdrawal-{len(withdrawal_rows) + 1}", "physical_order_key": key,
                        "partition_id": eligible.partition_id, "event_date": eligible.event_date,
                        "sort_index": observation.sort_index, "event_ts": event_ts, **_identity_fields(eligible.actor),
                        "side": eligible.side, "order_id": eligible.order_id,
                        "first_seen_sort_index": eligible.first_seen_sort_index,
                        "visible_qty_removed": eligible.visible_qty, "transition_reason": "cancellation",
                    })
                    emitted_withdrawal_keys.add(key)
        # Post-event state starts the next outgoing time segment. New physical orders
        # receive this event's timestamp as their lifecycle origin.
        for _, order in sorted(post_orders.items(), key=lambda item: str(item[0])):
            key = _physical_key(current_partition, current_date, order)
            first_seen_ts.setdefault(key, event_ts)
        post_state = _eligible_orders(post_orders, partition_id=current_partition,
                                      event_date=current_date, at_ts=event_ts, first_seen_ts=first_seen_ts,
                                      top_n=top_n, max_age=max_age)
        event_class = str(observation.event.get("event_class") or "")
        reason = {"cancel": "cancellation", "fill": "fill", "modify_order": "modify"}.get(event_class, "loss_of_eligibility")
        set_state(post_state, event_ts, transition_reason=reason, event_sort_index=observation.sort_index,
                  event_order_id=str(observation.event.get("ORDERID")) if observation.event.get("ORDERID") is not None else None)
        live_physical_keys = {
            _physical_key(current_partition, current_date, order)
            for order in post_orders.values()
        }
        first_seen_ts = {
            key: first_seen
            for key, first_seen in first_seen_ts.items()
            if key in live_physical_keys
        }
        last_ts = event_ts
        clock_high_watermark = event_ts
        last_boundary_sort_index = observation.sort_index

    if current_partition is not None and last_ts is not None:
        end = _coverage_end_for(coverage_end, current_partition) or last_ts
        if end < last_ts:
            raise ValueError("coverage_end precedes the final observed timestamp")
        advance_time_to(end)
        close_all(end, "coverage_end")

    # interval_count is defined after all emitted intervals are known.
    counts: dict[str, int] = defaultdict(int)
    for row in interval_rows:
        counts[row["risk_spell_id"]] += 1
    for row in spell_rows:
        row["interval_count"] = counts[row["risk_spell_id"]]
    intervals = _frame(interval_rows, INTERVAL_SCHEMA)
    spells = _frame(spell_rows, SPELL_SCHEMA)
    membership = _frame(membership_rows, MEMBERSHIP_SCHEMA)
    withdrawals = _frame(withdrawal_rows, WITHDRAWAL_SCHEMA)
    transitions = _frame(transition_rows, TRANSITION_SCHEMA)
    _assert_invariants(spells, intervals, withdrawals)
    summary_rows: list[dict[str, Any]] = []
    time_by_group: dict[tuple[Any, ...], float] = defaultdict(float)
    identity_by_group: dict[tuple[Any, ...], dict[str, Any]] = {}
    for spell in spells.iter_rows(named=True):
        group = (spell["partition_id"], spell["event_date"], spell["actor_key"], spell["side"])
        time_by_group[group] += float(spell["duration_seconds"])
        identity_by_group[group] = spell
    withdrawal_by_group: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in withdrawal_rows:
        withdrawal_by_group[(row["partition_id"], row["event_date"], row["actor_key"], row["side"])].append(row)
    for group in sorted(set(time_by_group) | set(withdrawal_by_group), key=str):
        denom = time_by_group.get(group, 0.0)
        nums = withdrawal_by_group.get(group, [])
        if denom == 0 and nums:
            raise AssertionError("positive withdrawal numerator with zero time-at-risk denominator")
        source = identity_by_group.get(group) or nums[0]
        summary_rows.append({
            "partition_id": group[0], "event_date": group[1], **{key: source[key] for key in ACTOR_SCHEMA}, "side": group[3],
            "time_at_risk_seconds": denom, "withdrawal_count": len(nums),
            "withdrawal_visible_qty": sum(float(row["visible_qty_removed"]) for row in nums),
            "withdrawal_intensity": (len(nums) / denom) if denom > 0 else None,
        })
    return WithdrawalRiskResult(
        spells, intervals, membership, withdrawals, transitions,
        _frame(diagnostics, DIAGNOSTIC_SCHEMA), _frame(summary_rows, ACTOR_SUMMARY_SCHEMA),
        _empty(EXPOSURE_INTERVAL_SCHEMA), _empty(EXPOSURE_CONTRAST_SCHEMA), _empty(EXPOSURE_STRATUM_SCHEMA),
    )


class _WithdrawalRiskAccumulator:
    """Incrementally account replay boundaries; retained state is O(active risk)."""

    def __init__(
        self,
        *,
        top_n: int,
        max_order_age_seconds: float,
        coverage_end: datetime | Mapping[str, datetime] | None = None,
    ) -> None:
        self._worker = _withdrawal_risk_observer(
            top_n=top_n,
            max_order_age_seconds=max_order_age_seconds,
            coverage_end=coverage_end,
        )
        next(self._worker)
        self._result: WithdrawalRiskResult | None = None

    def observe(self, observation: ReplayObservation) -> None:
        if self._result is not None:
            raise RuntimeError("withdrawal risk accumulator is already finalized")
        self._worker.send(observation)

    def finalize(self) -> WithdrawalRiskResult:
        if self._result is None:
            try:
                self._worker.send(_FINALIZE)
            except StopIteration as completed:
                self._result = completed.value
        if self._result is None:
            raise AssertionError("withdrawal risk accumulator finalized without a result")
        return self._result


def canonical_execution_cluster_rows(
    execution_clusters: Iterable[Mapping[str, Any]] | pl.DataFrame,
) -> list[dict[str, Any]]:
    """Copy-normalize canonical cluster rows and reject child-fill-shaped input.

    Exposure is deliberately defined on aggregate cluster rows.  Child fills are
    provenance, not independent exposure anchors, so accepting them would inflate
    actor-side time when a trade has two reported rows or a sweep has many fills.
    Aware anchors are converted to naive UTC copies, matching
    :func:`choose_event_timestamp`; supplied mappings are never mutated.
    """
    raw_rows = execution_clusters.to_dicts() if isinstance(execution_clusters, pl.DataFrame) else execution_clusters
    valid: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_rows):
        if not isinstance(raw, Mapping):
            raise ValueError(f"invalid execution cluster row {index}: row must be a mapping")
        row = dict(raw)

        def require_nonempty_text(field: str) -> str:
            value = row.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"invalid execution cluster row {index}: {field} must be nonempty text")
            return value

        cluster_id = require_nonempty_text("execution_cluster_id")
        partition_id = require_nonempty_text("partition_id")
        actor_key = require_nonempty_text("actor_key")
        actor_id = require_nonempty_text("actor_id")
        identity_level = row.get("identity_level")
        if identity_level not in {"client_original", "firm"}:
            raise ValueError(f"invalid execution cluster row {index}: identity_level is invalid")
        expected_identity_source = (
            "NMSC_ORIGINALCLIENTIDSHORTCODE" if identity_level == "client_original" else "FIRMID"
        )
        expected_fallback = identity_level == "firm"
        if (
            actor_key != f"{identity_level}:{actor_id}"
            or row.get("identity_source") != expected_identity_source
            or type(row.get("identity_fallback_flag")) is not bool
            or row.get("identity_fallback_flag") is not expected_fallback
        ):
            raise ValueError(
                f"invalid execution cluster row {index}: canonical identity is inconsistent"
            )
        mode = row.get("execution_anchor_mode")
        if mode not in {"passive", "aggressive"}:
            raise ValueError(f"invalid execution cluster row {index}: execution_anchor_mode is invalid")
        side = row.get("event_side")
        if side not in {"bid", "ask"}:
            raise ValueError(f"invalid execution cluster row {index}: event_side is invalid")
        end_ts = row.get("cluster_end_ts")
        if not isinstance(end_ts, datetime):
            raise ValueError(f"invalid execution cluster row {index}: cluster_end_ts must be a datetime")
        if end_ts.tzinfo is not None and end_ts.utcoffset() is not None:
            end_ts = end_ts.astimezone(timezone.utc).replace(tzinfo=None)
        last_sort_index = row.get("cluster_last_sort_index")
        if isinstance(last_sort_index, bool) or not isinstance(last_sort_index, Integral) or last_sort_index <= 0:
            raise ValueError(
                f"invalid execution cluster row {index}: cluster_last_sort_index must be a positive integer"
            )
        quantity = row.get("execution_quantity")
        if (
            isinstance(quantity, bool)
            or not isinstance(quantity, Real)
            or not math.isfinite(quantity)
            or quantity < 0
        ):
            raise ValueError(
                f"invalid execution cluster row {index}: execution_quantity must be a finite non-negative real"
            )
        row.update({
            "execution_cluster_id": cluster_id,
            "partition_id": partition_id,
            "actor_key": actor_key,
            "actor_id": actor_id,
            "cluster_end_ts": end_ts,
            "cluster_last_sort_index": int(last_sort_index),
            "execution_quantity": float(quantity),
            "event_side": side,
        })
        valid.append(row)

    canonical_by_id: dict[str, dict[str, Any]] = {}
    unique_ids: list[dict[str, Any]] = []
    for row in valid:
        cluster_id = row["execution_cluster_id"]
        previous = canonical_by_id.get(cluster_id)
        if previous is None:
            canonical_by_id[cluster_id] = row
            unique_ids.append(row)
        elif row != previous:
            raise ValueError(f"conflicting duplicate execution_cluster_id: {cluster_id}")

    # A canonical cluster id is authoritative.  For feeds that repeat a
    # canonical sweep/trade row under a different presentation id, its stable
    # actor/role/side/anchor membership is the deduplication key instead.
    selected: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in sorted(unique_ids, key=lambda item: (
        item["cluster_end_ts"], item["cluster_last_sort_index"], item["execution_cluster_id"],
    )):
        membership = row.get("execution_sweep_id") or row.get("TRADEUNIQUEIDENTIFIER") or row.get("trade_uid")
        key = (
            row["partition_id"], row["actor_key"], row["identity_level"], row["execution_anchor_mode"],
            row["event_side"], row["cluster_end_ts"], row["cluster_last_sort_index"],
            ("membership", str(membership)) if membership is not None else ("cluster", row["execution_cluster_id"]),
        )
        selected.setdefault(key, row)
    return list(selected.values())


# Backward-compatible private spelling for callers that predate the public API.
_execution_cluster_rows = canonical_execution_cluster_rows


def _exposure_stratum(row: Mapping[str, Any], *, side_field: str) -> tuple[str, date, str]:
    return (str(row["partition_id"]), row["event_date"], str(row[side_field]))


def _window_stratum(window: Mapping[str, Any]) -> tuple[str, date, str]:
    cluster = window["cluster"]
    return (str(cluster["partition_id"]), cluster["cluster_end_ts"].date(), str(cluster["event_side"]))


def _window_overlaps_interval(window: Mapping[str, Any], interval: Mapping[str, Any]) -> bool:
    """Keep boundary-only candidates for canonical-order point outcomes."""
    return window["start_ts"] <= interval["end_ts"] and window["end_ts"] >= interval["start_ts"]


def _exposure_relation(interval: Mapping[str, Any], cluster: Mapping[str, Any]) -> str:
    """Classify a cluster relative to an actor-side risk interval.

    Cross-level client/firm comparisons intentionally remain ambiguous: a firm
    fallback does not establish that an execution belongs to a different client.
    """
    if cluster["identity_level"] != interval["identity_level"]:
        return "identity_ambiguous"
    return "own" if cluster["actor_key"] == interval["actor_key"] else "other"


def _mask_label(flags: Mapping[str, bool]) -> str:
    return "|".join(name for name in ("own_passive", "own_aggressive", "other_passive", "other_aggressive") if flags[name]) or "none"


def _contrast_label(flags: Mapping[str, bool]) -> str:
    own = flags["own_passive"] or flags["own_aggressive"]
    other = flags["other_passive"] or flags["other_aggressive"]
    if own and other:
        return "mixed"
    if own:
        return "own_only"
    if other:
        return "other_only"
    return "no_qualifying_fill"


def intersect_observed_execution_exposure(
    risk_result: WithdrawalRiskResult,
    execution_clusters: Iterable[Mapping[str, Any]] | pl.DataFrame,
    *,
    withdrawal_window_seconds: float,
) -> WithdrawalRiskResult:
    """Intersect fixed canonical-cluster horizons with risk intervals.

    Every cluster contributes the outcome-independent fixed span
    ``[cluster_end_ts, cluster_end_ts + withdrawal_window_seconds]``.  Its
    duration is half-open for time accounting, while point outcomes include the
    horizon endpoint.  Contrast rows partition every positive-duration risk
    interval; withdrawals are assigned to one segment by event clock.
    """
    if (
        isinstance(withdrawal_window_seconds, bool)
        or not isinstance(withdrawal_window_seconds, Real)
        or not math.isfinite(withdrawal_window_seconds)
        or withdrawal_window_seconds <= 0
    ):
        raise ValueError("withdrawal_window_seconds must be a finite positive real")

    clusters = canonical_execution_cluster_rows(execution_clusters)
    withdrawals = risk_result.withdrawal_events.to_dicts()
    horizon = timedelta(seconds=float(withdrawal_window_seconds))
    windows = [
        {
            "cluster": cluster,
            "start_ts": cluster["cluster_end_ts"],
            "end_ts": cluster["cluster_end_ts"] + horizon,
        }
        for cluster in clusters
    ]

    intervals = risk_result.intervals.to_dicts()
    windows_by_stratum: dict[tuple[str, date, str], list[dict[str, Any]]] = defaultdict(list)
    for window in windows:
        windows_by_stratum[_window_stratum(window)].append(window)
    window_times_by_stratum: dict[tuple[str, date, str], tuple[list[datetime], list[datetime]]] = {}
    for stratum_key, candidates in windows_by_stratum.items():
        candidates.sort(key=lambda window: (
            window["start_ts"], window["end_ts"], window["cluster"]["execution_cluster_id"],
        ))
        window_times_by_stratum[stratum_key] = (
            [window["start_ts"] for window in candidates],
            [window["end_ts"] for window in candidates],
        )

    intervals_by_stratum: dict[
        tuple[str, date, str], dict[tuple[str, str], list[dict[str, Any]]]
    ] = defaultdict(lambda: defaultdict(list))
    for interval in intervals:
        intervals_by_stratum[_exposure_stratum(interval, side_field="side")][
            (str(interval["actor_key"]), str(interval["identity_level"]))
        ].append(interval)
    interval_starts_by_stratum: dict[tuple[str, date, str], dict[tuple[str, str], list[datetime]]] = {}
    for stratum_key, actor_intervals in intervals_by_stratum.items():
        starts_by_actor: dict[tuple[str, str], list[datetime]] = {}
        for candidates in actor_intervals.values():
            candidates.sort(key=lambda interval: (interval["start_ts"], interval["end_ts"]))
        for actor_key, candidates in actor_intervals.items():
            starts_by_actor[actor_key] = [interval["start_ts"] for interval in candidates]
        interval_starts_by_stratum[stratum_key] = starts_by_actor

    withdrawals_by_stratum: dict[tuple[str, date, str], list[dict[str, Any]]] = defaultdict(list)
    for withdrawal in withdrawals:
        withdrawals_by_stratum[_exposure_stratum(withdrawal, side_field="side")].append(withdrawal)
    withdrawals_by_interval: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for stratum, stratum_withdrawals in withdrawals_by_stratum.items():
        actor_intervals = intervals_by_stratum.get(stratum, {})
        for withdrawal in stratum_withdrawals:
            candidates = actor_intervals.get(
                (str(withdrawal["actor_key"]), str(withdrawal["identity_level"])), []
            )
            starts = interval_starts_by_stratum.get(stratum, {}).get(
                (str(withdrawal["actor_key"]), str(withdrawal["identity_level"])), []
            )
            position = bisect_left(starts, withdrawal["event_ts"]) - 1
            if position < 0 or withdrawal["event_ts"] > candidates[position]["end_ts"]:
                raise AssertionError("each withdrawal must map to exactly one risk interval")
            withdrawals_by_interval[candidates[position]["risk_interval_id"]].append(withdrawal)

    exposure_rows: list[dict[str, Any]] = []
    contrast_rows: list[dict[str, Any]] = []
    contrast_masks: list[str] = []

    def active_windows_at(
        candidates: Iterable[Mapping[str, Any]], at: datetime,
    ) -> list[Mapping[str, Any]]:
        return [
            window for window in candidates
            if window["start_ts"] <= at < window["end_ts"]
        ]

    def outcome_active_windows_at(
        candidates: Iterable[Mapping[str, Any]], withdrawal: Mapping[str, Any],
    ) -> list[Mapping[str, Any]]:
        """Return windows active at a cancellation point under canonical order."""
        at = withdrawal["event_ts"]
        active: list[Mapping[str, Any]] = []
        for window in candidates:
            cluster = window["cluster"]
            if at < window["start_ts"] or at > window["end_ts"]:
                continue
            if at == window["start_ts"] and int(withdrawal["sort_index"]) <= cluster["cluster_last_sort_index"]:
                continue
            active.append(window)
        return active

    def mask_fields(
        interval: Mapping[str, Any], active: Iterable[Mapping[str, Any]],
    ) -> tuple[dict[str, bool], list[str], str | None, float | None]:
        flags = {name: False for name in ("own_passive", "own_aggressive", "other_passive", "other_aggressive")}
        by_cluster: dict[str, Mapping[str, Any]] = {}
        for window in active:
            cluster = window["cluster"]
            relation = _exposure_relation(interval, cluster)
            if relation == "identity_ambiguous":
                continue
            flags[f"{relation}_{cluster['execution_anchor_mode']}"] = True
            by_cluster[cluster["execution_cluster_id"]] = cluster
        ids = sorted(by_cluster)
        modes = sorted({cluster["execution_anchor_mode"] for cluster in by_cluster.values()})
        quantity = sum(float(cluster["execution_quantity"]) for cluster in by_cluster.values())
        return flags, ids, (modes[0] if len(modes) == 1 else "mixed" if modes else None), (quantity if ids else None)

    for interval in intervals:
        interval_stratum = _exposure_stratum(interval, side_field="side")
        stratum_windows = windows_by_stratum.get(interval_stratum, [])
        window_starts, window_ends = window_times_by_stratum.get(interval_stratum, ([], []))
        first = bisect_left(window_ends, interval["start_ts"])
        last = bisect_right(window_starts, interval["end_ts"])
        relevant = [
            window for window in stratum_windows[first:last]
            if _window_overlaps_interval(window, interval)
        ]
        points = {interval["start_ts"], interval["end_ts"]}
        for window in relevant:
            if interval["start_ts"] < window["start_ts"] < interval["end_ts"]:
                points.add(window["start_ts"])
            if interval["start_ts"] < window["end_ts"] < interval["end_ts"]:
                points.add(window["end_ts"])
        segment_withdrawals: dict[int, list[dict[str, Any]]] = defaultdict(list)
        segments = list(zip(sorted(points), sorted(points)[1:]))
        for withdrawal in withdrawals_by_interval[interval["risk_interval_id"]]:
            matching_segments = [
                index for index, (start, end) in enumerate(segments)
                if start < withdrawal["event_ts"] <= end
            ]
            if len(matching_segments) != 1:
                raise AssertionError("each withdrawal must map to exactly one contrast segment")
            segment_withdrawals[matching_segments[0]].append(withdrawal)

        for index, (start, end) in enumerate(segments):
            active = active_windows_at(relevant, start)
            temporal_flags, temporal_ids, temporal_mode, temporal_quantity = mask_fields(interval, active)
            ambiguous_count = len({
                window["cluster"]["execution_cluster_id"]
                for window in active
                if _exposure_relation(interval, window["cluster"]) == "identity_ambiguous"
            })
            outcome_active = list(active)
            for withdrawal in segment_withdrawals[index]:
                outcome_active.extend(outcome_active_windows_at(relevant, withdrawal))
            contrast_flags, _, contrast_mode, contrast_quantity = mask_fields(interval, outcome_active)
            label = _contrast_label(contrast_flags)
            withdrawal_ids = sorted({row["withdrawal_event_id"] for row in segment_withdrawals[index]})
            # Exposure intervals record positive temporal overlap only.  A
            # point outcome can change its contrast stratum at a boundary, but
            # must not manufacture a zero-duration exposure row.
            if temporal_ids:
                exposure_rows.append({
                    "exposure_interval_id": f"exposure-interval-{len(exposure_rows) + 1}",
                    "risk_interval_id": interval["risk_interval_id"], "risk_spell_id": interval["risk_spell_id"],
                    "withdrawal_event_id": "|".join(withdrawal_ids) if withdrawal_ids else None,
                    "partition_id": interval["partition_id"], "event_date": interval["event_date"],
                    **{key: interval[key] for key in ACTOR_SCHEMA}, "side": interval["side"],
                    "start_ts": start, "end_ts": end, "duration_seconds": (end - start).total_seconds(),
                    **temporal_flags, "identity_ambiguous_cluster_count": ambiguous_count,
                    "exposure_mask": _mask_label(temporal_flags), "execution_cluster_ids": "|".join(temporal_ids),
                    "execution_anchor_mode": temporal_mode, "execution_quantity": temporal_quantity,
                    "contrast_label": _contrast_label(temporal_flags),
                })
            contrast_rows.append({
                "risk_interval_id": interval["risk_interval_id"],
                "withdrawal_event_id": "|".join(withdrawal_ids) if withdrawal_ids else None,
                "partition_id": interval["partition_id"], "event_date": interval["event_date"],
                **{key: interval[key] for key in ACTOR_SCHEMA}, "side": interval["side"],
                "start_ts": start, "end_ts": end, "duration_seconds": (end - start).total_seconds(),
                "contrast_label": label,
                "execution_anchor_mode": contrast_mode,
                "execution_quantity": contrast_quantity,
                "identity_ambiguous_cluster_count": ambiguous_count,
            })
            contrast_masks.append(_mask_label(contrast_flags))

    strata_by_mask: dict[tuple[str, str], dict[str, Any]] = {}
    for row, mask in zip(contrast_rows, contrast_masks, strict=True):
        key = (row["contrast_label"], mask)
        value = strata_by_mask.setdefault(key, {"contrast_label": key[0], "exposure_mask": key[1], "interval_count": 0, "duration_seconds": 0.0, "withdrawal_ids": set()})
        value["interval_count"] += 1
        value["duration_seconds"] += row["duration_seconds"]
        if row["withdrawal_event_id"]:
            value["withdrawal_ids"].update(row["withdrawal_event_id"].split("|"))
    strata_rows = [{
        "contrast_label": value["contrast_label"], "exposure_mask": value["exposure_mask"],
        "interval_count": value["interval_count"], "duration_seconds": value["duration_seconds"],
        "withdrawal_count": len(value["withdrawal_ids"]),
    } for _, value in sorted(strata_by_mask.items())]
    _assert_exposure_invariants(risk_result.intervals, contrast_rows, withdrawals)
    return replace(
        risk_result,
        exposure_intervals=_frame(exposure_rows, EXPOSURE_INTERVAL_SCHEMA),
        exposure_contrasts=_frame(contrast_rows, EXPOSURE_CONTRAST_SCHEMA),
        exposure_strata=_frame(strata_rows, EXPOSURE_STRATUM_SCHEMA),
    )


def build_withdrawal_risk(
    observations: Iterable[ReplayObservation], *, top_n: int, max_order_age_seconds: float,
    coverage_end: datetime | Mapping[str, datetime] | None = None,
    execution_clusters: Iterable[Mapping[str, Any]] | pl.DataFrame | None = None,
    withdrawal_window_seconds: float | None = None,
) -> WithdrawalRiskResult:
    """Build risk from a single-pass iterable without retaining its snapshots."""
    accumulator = _WithdrawalRiskAccumulator(
        top_n=top_n,
        max_order_age_seconds=max_order_age_seconds,
        coverage_end=coverage_end,
    )
    for observation in observations:
        accumulator.observe(observation)
    result = accumulator.finalize()
    if execution_clusters is None:
        return result
    if withdrawal_window_seconds is None:
        raise ValueError("withdrawal_window_seconds is required when execution_clusters are supplied")
    return intersect_observed_execution_exposure(
        result, execution_clusters, withdrawal_window_seconds=withdrawal_window_seconds,
    )


def _assert_invariants(spells: pl.DataFrame, intervals: pl.DataFrame, withdrawals: pl.DataFrame) -> None:
    if not intervals.is_empty() and (intervals.get_column("duration_seconds") <= 0).any():
        raise AssertionError("risk intervals must have positive duration")
    for row in intervals.iter_rows(named=True):
        for field in ("mean_member_age_seconds", "oldest_member_age_seconds"):
            value = row[field]
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise AssertionError("lifecycle-age covariates must be finite and non-negative")
    if not withdrawals.is_empty() and withdrawals.get_column("physical_order_key").is_duplicated().any():
        raise AssertionError("physical cancellation keys must be unique")
    by_spell: dict[str, float] = defaultdict(float)
    for row in intervals.iter_rows(named=True):
        by_spell[str(row["risk_spell_id"])] += float(row["duration_seconds"])
    for spell in spells.iter_rows(named=True):
        if abs(by_spell[str(spell["risk_spell_id"])] - float(spell["duration_seconds"])) > 1e-9:
            raise AssertionError("spell duration does not equal interval duration sum")


def _assert_exposure_invariants(
    intervals: pl.DataFrame,
    contrast_rows: Iterable[Mapping[str, Any]],
    withdrawals: Iterable[Mapping[str, Any]],
) -> None:
    """Check that contrasts are a complete risk partition and outcomes are unique."""
    duration_by_interval: dict[str, float] = defaultdict(float)
    outcome_ids: list[str] = []
    for row in contrast_rows:
        duration = float(row["duration_seconds"])
        if duration <= 0:
            raise AssertionError("exposure contrasts must have positive duration")
        duration_by_interval[str(row["risk_interval_id"])] += duration
        if row["withdrawal_event_id"]:
            outcome_ids.extend(str(row["withdrawal_event_id"]).split("|"))
    for interval in intervals.iter_rows(named=True):
        risk_interval_id = str(interval["risk_interval_id"])
        if abs(duration_by_interval[risk_interval_id] - float(interval["duration_seconds"])) > 1e-9:
            raise AssertionError("contrast duration does not equal risk interval duration")
    expected_ids = [str(row["withdrawal_event_id"]) for row in withdrawals]
    if sorted(outcome_ids) != sorted(expected_ids) or len(outcome_ids) != len(set(outcome_ids)):
        raise AssertionError("each withdrawal must occur in exactly one contrast")


def compute_withdrawal_risk(raw_events: pl.DataFrame, *, top_n: int, max_order_age_seconds: float,
                            coverage_end: datetime | Mapping[str, datetime] | None = None,
                            execution_clusters: Iterable[Mapping[str, Any]] | pl.DataFrame | None = None,
                            withdrawal_window_seconds: float | None = None) -> WithdrawalRiskResult:
    """Replay raw events into incremental risk accounting without snapshot retention."""
    accumulator = _WithdrawalRiskAccumulator(
        top_n=top_n,
        max_order_age_seconds=max_order_age_seconds,
        coverage_end=coverage_end,
    )
    replay_lob(raw_events, observer=accumulator.observe, retain_events=False)
    result = accumulator.finalize()
    if execution_clusters is None:
        return result
    if withdrawal_window_seconds is None:
        raise ValueError("withdrawal_window_seconds is required when execution_clusters are supplied")
    return intersect_observed_execution_exposure(
        result, execution_clusters, withdrawal_window_seconds=withdrawal_window_seconds,
    )


# Readable aliases for downstream code while Task 4 extends execution exposure.
build_time_at_risk = build_withdrawal_risk
analyze_withdrawal_risk = build_withdrawal_risk
