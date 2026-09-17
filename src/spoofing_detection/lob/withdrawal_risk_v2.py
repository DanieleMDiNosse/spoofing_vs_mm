"""Compact callback-only top-N withdrawal-risk observer.

``compute_compact_risk`` consumes the canonical replay loop directly.  It keeps
only order-origin clocks and the current eligible projection; it never captures
pre/post copies of the full book.
"""
from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import heapq
import math
from numbers import Real
from typing import Any

import polars as pl

from .actor_identity import ActorIdentity, actor_identity_from_order
from .config import LOBConfig
from .models import ActiveOrder
from .panel import ReplayHooks, replay_events
from .replay_observation import choose_event_timestamp

ACTOR_SCHEMA: dict[str, pl.DataType] = {
    "actor_key": pl.String, "actor_id": pl.String, "identity_level": pl.String,
    "identity_source": pl.String, "identity_fallback_flag": pl.Boolean,
}
INTERVAL_SCHEMA: dict[str, pl.DataType] = {
    "risk_interval_id": pl.String, "instrument": pl.String, "partition_id": pl.String,
    "event_date": pl.Date, **ACTOR_SCHEMA, "side": pl.String, "coverage_epoch_id": pl.String,
    "start_ts": pl.Datetime("us"), "end_ts": pl.Datetime("us"), "duration_seconds": pl.Float64,
    "member_count": pl.Int64, "eligible_visible_qty": pl.Float64,
    "mean_member_age_seconds": pl.Float64, "oldest_member_age_seconds": pl.Float64,
    "membership_signature": pl.String,
}
MEMBERSHIP_SCHEMA: dict[str, pl.DataType] = {
    "risk_membership_id": pl.String, "instrument": pl.String, "physical_order_key": pl.String,
    "partition_id": pl.String, "event_date": pl.Date, "order_id": pl.String,
    "first_seen_sort_index": pl.Int64, **ACTOR_SCHEMA, "side": pl.String,
    "coverage_epoch_id": pl.String, "eligible_start_ts": pl.Datetime("us"),
    "eligible_end_ts": pl.Datetime("us"), "duration_seconds": pl.Float64,
    "start_reason": pl.String, "end_reason": pl.String,
}
WITHDRAWAL_SCHEMA: dict[str, pl.DataType] = {
    "withdrawal_event_id": pl.String, "instrument": pl.String, "physical_order_key": pl.String,
    "partition_id": pl.String, "event_date": pl.Date, "sort_index": pl.Int64,
    "event_ts": pl.Datetime("us"), **ACTOR_SCHEMA, "side": pl.String, "coverage_epoch_id": pl.String,
    "order_id": pl.String,
    "first_seen_sort_index": pl.Int64, "visible_qty_removed": pl.Float64,
    "transition_reason": pl.String,
}
TRANSITION_SCHEMA: dict[str, pl.DataType] = {
    "transition_id": pl.String, "instrument": pl.String, "physical_order_key": pl.String,
    "partition_id": pl.String, "event_date": pl.Date, "sort_index": pl.Int64,
    "event_ts": pl.Datetime("us"), **ACTOR_SCHEMA, "side": pl.String, "order_id": pl.String,
    "first_seen_sort_index": pl.Int64, "visible_qty_pre": pl.Float64,
    "visible_qty_post": pl.Float64, "transition_reason": pl.String,
}
COVERAGE_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.String, "partition_id": pl.String, "event_date": pl.Date,
    "coverage_epoch_id": pl.String, "start_ts": pl.Datetime("us"), "end_ts": pl.Datetime("us"),
    "end_reason": pl.String,
}
DIAGNOSTIC_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.String, "partition_id": pl.String, "sort_index": pl.Int64,
    "event_ts": pl.Datetime("us"), "reason": pl.String, "detail": pl.String,
}
MARKET_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.String, "partition_id": pl.String, "event_date": pl.Date,
    "coverage_epoch_id": pl.String, "start_ts": pl.Datetime("us"), "end_ts": pl.Datetime("us"),
    "spread": pl.Float64, "depth": pl.Float64, "prior_event_count_60s": pl.Int64,
}
_TABLE_SCHEMAS = {
    "intervals": INTERVAL_SCHEMA, "membership": MEMBERSHIP_SCHEMA,
    "withdrawal_events": WITHDRAWAL_SCHEMA, "transitions": TRANSITION_SCHEMA,
    "coverage_epochs": COVERAGE_SCHEMA, "diagnostics": DIAGNOSTIC_SCHEMA,
    "market_intervals": MARKET_SCHEMA,
}
# Public schemas for sink writers.  Keys are result attribute/table names.
SCHEMAS: Mapping[str, Mapping[str, pl.DataType]] = _TABLE_SCHEMAS


@dataclass(frozen=True)
class CompactRiskResult:
    intervals: pl.DataFrame
    membership: pl.DataFrame
    withdrawal_events: pl.DataFrame
    transitions: pl.DataFrame
    coverage_epochs: pl.DataFrame
    diagnostics: pl.DataFrame
    market_intervals: pl.DataFrame
    counters: Mapping[str, int]


@dataclass(frozen=True)
class _Member:
    physical_key: str
    partition_id: str
    event_date: date
    order_id: str
    first_seen_sort_index: int
    actor: ActorIdentity
    side: str
    visible_qty: float
    origin_ts: datetime
    coverage_epoch_id: str

    @property
    def group(self) -> tuple[str, date, str, str, str]:
        return (self.partition_id, self.event_date, self.actor.actor_key, self.side, self.coverage_epoch_id)


class _Rows:
    """Bounded per-table writer; streaming mode retains no completed output."""

    def __init__(self, sink: Callable[[str, list[dict]], None] | None, buffer_rows: int) -> None:
        if type(buffer_rows) is not int or buffer_rows <= 0:
            raise ValueError("buffer_rows must be a positive integer")
        self.sink, self.buffer_rows = sink, buffer_rows
        self.rows: dict[str, list[dict[str, Any]]] = {name: [] for name in _TABLE_SCHEMAS}
        self.counters: dict[str, int] = {name: 0 for name in _TABLE_SCHEMAS}

    def emit(self, name: str, row: dict[str, Any]) -> None:
        self.rows[name].append(row)
        self.counters[name] += 1
        if self.sink is not None and len(self.rows[name]) >= self.buffer_rows:
            self.flush(name)

    def flush(self, name: str) -> None:
        rows = self.rows[name]
        if rows and self.sink is not None:
            self.sink(name, rows)
            self.rows[name] = []

    def finish(self) -> dict[str, pl.DataFrame]:
        if self.sink is not None:
            for name in _TABLE_SCHEMAS:
                self.flush(name)
            return {name: pl.DataFrame(schema=schema) for name, schema in _TABLE_SCHEMAS.items()}
        return {name: _frame(self.rows[name], schema) for name, schema in _TABLE_SCHEMAS.items()}


def _frame(rows: list[dict[str, Any]], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, infer_schema_length=None).select(
        pl.col(name).cast(dtype, strict=False) for name, dtype in schema.items()
    )


def _identity(actor: ActorIdentity) -> dict[str, Any]:
    return {"actor_key": actor.actor_key, "actor_id": actor.actor_id,
            "identity_level": actor.identity_level, "identity_source": actor.identity_source,
            "identity_fallback_flag": actor.identity_fallback_flag}


def _event_date(event: Mapping[str, Any], timestamp: datetime | None) -> date:
    value = event.get("TRADEDATE")
    if value is not None:
        try:
            return date.fromisoformat(str(value)[:10])
        except ValueError:
            pass
    return timestamp.date() if timestamp is not None else date.min


def _partition(event: Mapping[str, Any]) -> str:
    return "|".join("" if event.get(key) is None else str(event.get(key)) for key in
                    ("TRADEDATE", "MIC", "MARKETCODE", "SYMBOLINDEX", "EMM (*)"))


def _visible(order: ActiveOrder) -> float:
    if order.leaves_qty <= 0 or order.displayed_qty <= 0:
        return 0.0
    return float(order.displayed_qty)


def _valid_order(order: ActiveOrder) -> bool:
    for value in (order.price, order.leaves_qty, order.displayed_qty):
        if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
            return False
    return True


def _validate_args(top_n: int, max_order_age_seconds: float) -> None:
    if type(top_n) is not int or top_n <= 0:
        raise ValueError("top_n must be a positive integer")
    if (isinstance(max_order_age_seconds, bool) or not isinstance(max_order_age_seconds, Real)
            or not math.isfinite(max_order_age_seconds) or max_order_age_seconds < 0):
        raise ValueError("max_order_age_seconds must be a finite non-negative real")


class _Observer:
    def __init__(self, *, instrument: str, top_n: int, max_age: float,
                 original_sort_indices: Mapping[int, int] | None, writer: _Rows) -> None:
        self.instrument, self.top_n, self.max_age = instrument, top_n, timedelta(seconds=max_age)
        self.original = original_sort_indices
        self.writer = writer
        self.partition_id: str | None = None
        self.event_date: date | None = None
        self.last_ts: datetime | None = None
        self.high_water: datetime | None = None
        self.quarantine = False
        self.epoch: str | None = None
        self.epoch_start: datetime | None = None
        self.epoch_counter = 0
        self.origins: dict[str, datetime] = {}
        self.state: dict[str, _Member] = {}
        self.membership_open: dict[str, tuple[_Member, datetime, str]] = {}
        self.group_open: dict[tuple[str, date, str, str, str], tuple[datetime, dict[str, _Member]]] = {}
        self.expiries: list[tuple[datetime, str]] = []
        self.scheduled_expiries: set[str] = set()
        self.emitted_withdrawal_messages: set[tuple[Any, ...]] = set()
        self.point_member: _Member | None = None
        self.activity: deque[datetime] = deque()
        self.market_open: tuple[datetime, tuple[Any, ...]] | None = None
        self.interval_id = self.membership_id = self.withdrawal_id = self.transition_id = self.market_id = 0

    def source_index(self, local: int) -> int:
        if self.original is None:
            value = local
        else:
            if local not in self.original:
                raise ValueError(f"original_sort_indices missing local sort_index {local}")
            value = self.original[local]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError("original_sort_indices values must be positive integers")
        return value

    def diagnostic(self, event: Mapping[str, Any], ts: datetime | None, reason: str, detail: str = "") -> None:
        self.writer.emit("diagnostics", {"instrument": self.instrument, "partition_id": _partition(event),
            "sort_index": self.source_index(int(event["sort_index"])), "event_ts": ts,
            "reason": reason, "detail": detail})

    def close_epoch(self, reason: str) -> None:
        if self.epoch is None or self.last_ts is None or self.epoch_start is None:
            self.clear_risk()
            return
        self._close_groups(self.last_ts, reason, all_groups=True)
        self._close_market(self.last_ts)
        self.writer.emit("coverage_epochs", {"instrument": self.instrument, "partition_id": self.partition_id,
            "event_date": self.event_date, "coverage_epoch_id": self.epoch, "start_ts": self.epoch_start,
            "end_ts": self.last_ts, "end_reason": reason})
        self.clear_risk()

    def clear_risk(self) -> None:
        self.origins.clear(); self.state.clear(); self.membership_open.clear(); self.group_open.clear(); self.expiries.clear()
        self.scheduled_expiries.clear()
        self.activity.clear(); self.market_open = None; self.epoch = None; self.epoch_start = None; self.last_ts = None

    def open_epoch(self, ts: datetime) -> None:
        self.epoch_counter += 1
        self.epoch = f"coverage-{self.epoch_counter}"
        self.epoch_start = self.last_ts = ts
        self.activity.clear(); self.market_open = None

    def _physical_key(self, order: ActiveOrder) -> str:
        assert self.partition_id is not None and self.event_date is not None
        first = self.source_index(int(order.first_seen_sort_index))
        return "|".join((self.partition_id, self.event_date.isoformat(), str(order.order_id), str(first)))

    def _eligible(self, active: Mapping[str, ActiveOrder], ts: datetime, event: Mapping[str, Any], *, inclusive_age: bool = False) -> dict[str, _Member]:
        prices: dict[str, set[float]] = {"bid": set(), "ask": set()}
        valid: list[ActiveOrder] = []
        for order in active.values():
            if not _valid_order(order):
                self.diagnostic(event, ts, "invalid_order_numeric", f"order_id={order.order_id!r}")
                continue
            valid.append(order)
            if order.side in prices and _visible(order) > 0:
                prices[order.side].add(float(order.price))
        top = {"bid": set(sorted(prices["bid"], reverse=True)[:self.top_n]),
               "ask": set(sorted(prices["ask"])[:self.top_n])}
        out: dict[str, _Member] = {}
        for order in valid:
            if order.side not in top or _visible(order) <= 0 or float(order.price) not in top[order.side]:
                continue
            actor = actor_identity_from_order(order)
            if actor is None:
                self.diagnostic(event, ts, "unknown_identity", f"order_id={order.order_id!r}")
                continue
            key = self._physical_key(order)
            origin = self.origins.get(key)
            if origin is None or ts < origin or ts - origin > self.max_age or (not inclusive_age and ts - origin == self.max_age):
                continue
            assert self.event_date is not None and self.epoch is not None and self.partition_id is not None
            out[key] = _Member(key, self.partition_id, self.event_date, order.order_id,
                self.source_index(int(order.first_seen_sort_index)), actor, order.side, _visible(order), origin, self.epoch)
        return out

    def _signature(self, members: Mapping[str, _Member]) -> tuple[tuple[str, float], ...]:
        return tuple((key, members[key].visible_qty) for key in sorted(members))

    def _emit_interval(self, group: tuple[str, date, str, str, str], start: datetime,
                       end: datetime, members: Mapping[str, _Member]) -> None:
        if end <= start or not members:
            return
        self.interval_id += 1
        first = next(iter(members.values()))
        ages = [(start - item.origin_ts).total_seconds() for item in members.values()]
        self.writer.emit("intervals", {"risk_interval_id": f"compact-risk-{self.interval_id}",
            "instrument": self.instrument, "partition_id": group[0], "event_date": group[1], **_identity(first.actor),
            "side": group[3], "coverage_epoch_id": group[4], "start_ts": start, "end_ts": end,
            "duration_seconds": (end - start).total_seconds(), "member_count": len(members),
            "eligible_visible_qty": sum(item.visible_qty for item in members.values()),
            "mean_member_age_seconds": sum(ages) / len(ages), "oldest_member_age_seconds": max(ages),
            "membership_signature": ";".join(f"{key}:{members[key].visible_qty:.12g}" for key in sorted(members))})

    def _close_groups(self, at: datetime, reason: str, *, all_groups: bool = False,
                      changed: set[tuple[str, date, str, str, str]] | None = None) -> None:
        for group, (start, members) in list(self.group_open.items()):
            if all_groups or (changed is not None and group in changed):
                self._emit_interval(group, start, at, members)
                self.group_open.pop(group, None)
        if all_groups:
            for key in list(self.membership_open):
                self._close_membership(key, at, reason)

    def _close_membership(self, key: str, at: datetime, reason: str) -> None:
        opened = self.membership_open.pop(key, None)
        if opened is None:
            return
        item, start, start_reason = opened
        if at <= start:
            return
        self.membership_id += 1
        self.writer.emit("membership", {"risk_membership_id": f"compact-member-{self.membership_id}",
            "instrument": self.instrument, "physical_order_key": key, "partition_id": item.partition_id,
            "event_date": item.event_date, "order_id": item.order_id, "first_seen_sort_index": item.first_seen_sort_index,
            **_identity(item.actor), "side": item.side, "coverage_epoch_id": item.coverage_epoch_id,
            "eligible_start_ts": start, "eligible_end_ts": at, "duration_seconds": (at-start).total_seconds(),
            "start_reason": start_reason, "end_reason": reason})

    def _transition(self, item: _Member, ts: datetime, local: int, reason: str, post: float | None = None) -> None:
        self.transition_id += 1
        self.writer.emit("transitions", {"transition_id": f"compact-transition-{self.transition_id}",
            "instrument": self.instrument, "physical_order_key": item.physical_key, "partition_id": item.partition_id,
            "event_date": item.event_date, "sort_index": self.source_index(local), "event_ts": ts,
            **_identity(item.actor), "side": item.side, "order_id": item.order_id,
            "first_seen_sort_index": item.first_seen_sort_index, "visible_qty_pre": item.visible_qty,
            "visible_qty_post": post, "transition_reason": reason})

    def _replace_state(self, next_state: dict[str, _Member], ts: datetime, local: int, reason: str,
                       event_order_id: str | None = None) -> None:
        old, old_keys, new_keys = self.state, set(self.state), set(next_state)
        changed_keys = {key for key in old_keys & new_keys if old[key] != next_state[key]}
        removed = old_keys - new_keys
        group_changed = {old[key].group for key in removed | changed_keys} | {next_state[key].group for key in changed_keys | (new_keys-old_keys)}
        self._close_groups(ts, reason, changed=group_changed)
        for key in sorted(removed | changed_keys):
            item = old[key]
            if key in removed:
                why = reason if reason == "age_expiry" or item.order_id == event_order_id else "loss_of_eligibility"
                self._close_membership(key, ts, why)
                self._transition(item, ts, local, why)
            else:
                self._close_membership(key, ts, "membership_change")
                self._transition(item, ts, local, reason, next_state[key].visible_qty)
        for key in sorted((new_keys-old_keys) | changed_keys):
            item = next_state[key]
            self.membership_open[key] = (item, ts, "eligible_entry" if key not in changed_keys else "membership_change")
        groups: dict[tuple[str, date, str, str, str], dict[str, _Member]] = defaultdict(dict)
        for key, item in next_state.items():
            groups[item.group][key] = item
        for group in group_changed:
            members = groups.get(group, {})
            if members:
                self.group_open[group] = (ts, members)
        self.state = next_state

    def _market_values(self, active: Mapping[str, ActiveOrder], ts: datetime) -> tuple[Any, ...]:
        levels: dict[str, dict[float, float]] = {"bid": defaultdict(float), "ask": defaultdict(float)}
        for order in active.values():
            if _valid_order(order) and order.side in levels and _visible(order) > 0:
                levels[order.side][float(order.price)] += _visible(order)
        bid = max(levels["bid"], default=None); ask = min(levels["ask"], default=None)
        history_complete = self.epoch_start is not None and ts - self.epoch_start >= timedelta(seconds=60)
        return (None if bid is None or ask is None else ask-bid,
                sum(levels["bid"].values()) + sum(levels["ask"].values()),
                len(self.activity) if history_complete else None)

    def _close_market(self, at: datetime) -> None:
        if self.market_open is None or at <= self.market_open[0]:
            self.market_open = None
            return
        start, values = self.market_open
        assert self.partition_id is not None and self.event_date is not None and self.epoch is not None
        self.writer.emit("market_intervals", {"instrument": self.instrument,
            "partition_id": self.partition_id, "event_date": self.event_date, "coverage_epoch_id": self.epoch,
            "start_ts": start, "end_ts": at, "spread": values[0], "depth": values[1],
            "prior_event_count_60s": values[2]})
        self.market_open = None

    def _set_market(self, active: Mapping[str, ActiveOrder], ts: datetime) -> None:
        values = self._market_values(active, ts)
        if self.market_open is not None and self.market_open[1] == values:
            return
        self._close_market(ts)
        self.market_open = (ts, values)

    def _expire_to(self, target: datetime, active: Mapping[str, ActiveOrder], local: int) -> None:
        while self.expiries and self.expiries[0][0] <= target:
            expiry, key = heapq.heappop(self.expiries)
            self.scheduled_expiries.discard(key)
            item = self.state.get(key)
            if item is not None and item.origin_ts + self.max_age == expiry:
                next_state = dict(self.state); next_state.pop(key, None)
                self._replace_state(next_state, expiry, local, "age_expiry")
                self.last_ts = expiry
        while self.activity and self.activity[0] + timedelta(seconds=60) <= target:
            expiry = self.activity[0] + timedelta(seconds=60)
            self._close_market(expiry)
            while self.activity and self.activity[0] + timedelta(seconds=60) <= expiry:
                self.activity.popleft()
            self._set_market(active, expiry)
            self.last_ts = expiry

    def pre(self, event: dict[str, Any], active: dict[str, ActiveOrder], partition_id: str | None) -> None:
        self.point_member = None
        local = int(event["sort_index"])
        # Validate every canonical event, including events that never produce an
        # output row.  A supplied map must be a total local-to-original mapping.
        self.source_index(local)
        ts = choose_event_timestamp(event)
        if (self.epoch is None or self.quarantine or ts is None or self.high_water is None
                or ts < self.high_water or _partition(event) != self.partition_id):
            return
        # Expiry boundaries are immediately before this event's mutation.  This
        # preserves the preceding book state instead of assigning the current
        # event's post-mutation depth to an earlier market interval.
        self._expire_to(ts, active, local)
        # Point eligibility is inclusive at the age limit and does not require
        # positive elapsed membership time. Snapshot values, not live orders.
        if event.get("event_class") == "cancel":
            eligible = self._eligible(active, ts, event, inclusive_age=True)
            self.point_member = next((item for item in eligible.values()
                                      if item.order_id == str(event.get("ORDERID"))), None)

    def post(self, event: dict[str, Any], active: dict[str, ActiveOrder], partition_id: str | None, flags: list[str]) -> None:
        ts = choose_event_timestamp(event); current_partition = _partition(event); current_date = _event_date(event, ts)
        local = int(event["sort_index"])
        if self.partition_id is not None and current_partition != self.partition_id:
            self.close_epoch("partition_boundary")
            self.high_water = None; self.quarantine = False
        self.partition_id, self.event_date = current_partition, current_date
        if ts is None:
            self.diagnostic(event, None, "missing_event_timestamp", "coverage interrupted")
            if self.last_ts is not None:
                self.close_epoch("clock_ambiguous")
            self.quarantine = True
            return
        if self.high_water is not None and ts < self.high_water:
            self.diagnostic(event, ts, "clock_regression", f"previous={self.high_water.isoformat()}")
            if self.last_ts is not None:
                self.close_epoch("clock_ambiguous")
            self.quarantine = True
            return
        if self.quarantine and self.high_water is not None and ts < self.high_water:
            return
        if self.epoch is None:
            self.open_epoch(ts)
        # Record only origins observed at this exact canonical new-order boundary.
        for order in active.values():
            key = self._physical_key(order)
            if int(order.first_seen_sort_index) == local:
                self.origins.setdefault(key, ts)
        self._expire_to(ts, active, local)
        event_order = str(event.get("ORDERID")) if event.get("ORDERID") is not None else None
        if event.get("event_class") == "cancel":
            pre = self.point_member
            if pre is not None:
                message_key: tuple[Any, ...] = (pre.physical_key, event.get("EVENTID"), event.get("event_class"),
                    event.get("LEAVESQTY"), event.get("DISPLAYEDQTY"), ts)
            else:
                message_key = ()
            if pre is not None and message_key not in self.emitted_withdrawal_messages:
                self.withdrawal_id += 1; self.emitted_withdrawal_messages.add(message_key)
                self.writer.emit("withdrawal_events", {"withdrawal_event_id": f"compact-withdrawal-{self.withdrawal_id}",
                    "instrument": self.instrument, "physical_order_key": pre.physical_key, "partition_id": pre.partition_id,
                    "event_date": pre.event_date, "sort_index": self.source_index(local), "event_ts": ts, **_identity(pre.actor),
                    "side": pre.side, "coverage_epoch_id": pre.coverage_epoch_id, "order_id": pre.order_id,
                    "first_seen_sort_index": pre.first_seen_sort_index,
                    "visible_qty_removed": pre.visible_qty, "transition_reason": "cancellation"})
        values = self._market_values(active, ts)
        if values[0] is not None and values[0] <= 0:
            self.diagnostic(event, ts, "crossed_or_locked_book", f"spread={values[0]}")
        next_state = self._eligible(active, ts, event)
        reason = {"cancel": "cancellation", "fill": "fill", "modify_order": "modify"}.get(str(event.get("event_class")), "loss_of_eligibility")
        self._replace_state(next_state, ts, local, reason, event_order)
        live = {self._physical_key(order) for order in active.values()}
        self.origins = {key: value for key, value in self.origins.items() if key in live}
        for key, item in next_state.items():
            if key not in self.scheduled_expiries:
                heapq.heappush(self.expiries, (item.origin_ts + self.max_age, key))
                self.scheduled_expiries.add(key)
        # The compact covariate is explicitly prior-event activity, so the
        # current event becomes available only at a later boundary.
        self._set_market(active, ts)
        self.activity.append(ts)
        self.last_ts, self.high_water, self.quarantine = ts, ts, False

    def partition_end(self, event: dict[str, Any], active: dict[str, ActiveOrder], partition_id: str | None) -> None:
        # The driver invokes this both at partition rollover and end-of-input.
        if self.last_ts is not None:
            self.close_epoch("end_of_partition")
        self.high_water = None; self.quarantine = False

    def result(self) -> CompactRiskResult:
        frames = self.writer.finish()
        return CompactRiskResult(**frames, counters=dict(self.writer.counters))


def compute_compact_risk(
    raw_events: pl.DataFrame,
    *,
    instrument: str,
    top_n: int = 10,
    max_order_age_seconds: float = 90,
    config: LOBConfig | None = None,
    original_sort_indices: Mapping[int, int] | None = None,
    sink: Callable[[str, list[dict]], None] | None = None,
    buffer_rows: int = 10_000,
) -> CompactRiskResult:
    """Compute compact actor-side top-N risk using canonical ``ReplayHooks``.

    ``sort_index`` and ``first_seen_sort_index`` are mapped through
    ``original_sort_indices`` (canonical local one-based index -> original
    one-based index).  With ``sink`` set, completed chunks are flushed under
    table names matching result attributes and returned frames are empty.
    """
    if not isinstance(instrument, str) or not instrument:
        raise ValueError("instrument must be nonempty text")
    _validate_args(top_n, max_order_age_seconds)
    writer = _Rows(sink, buffer_rows)
    observer = _Observer(instrument=instrument, top_n=top_n, max_age=max_order_age_seconds,
                         original_sort_indices=original_sort_indices, writer=writer)
    replay_events(raw_events, config=config or LOBConfig(), hooks=ReplayHooks(
        on_pre_event=observer.pre, on_post_event=observer.post, on_partition_end=observer.partition_end,
    ), retain_events=False)
    return observer.result()
