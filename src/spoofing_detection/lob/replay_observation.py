from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import datetime, timezone
from types import MappingProxyType
from typing import Any

import polars as pl

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.models import ActiveOrder
from spoofing_detection.lob.panel import (
    ReplayHooks,
    replay_events,
)

EVENT_TIMESTAMP_PRECEDENCE = ("TRADETIME", "BOOKOUTTIME", "BOOKIN", "SEQUENCETIME")


@dataclass(frozen=True)
class ReplayOrder:
    """An immutable active-order value captured at one replay boundary."""

    order_id: str
    side: str
    price: float
    leaves_qty: float
    displayed_qty: float
    order_qty: float | None
    order_priority: str | None
    order_type_code: int | None
    order_type_label: str | None
    time_in_force_code: int | None
    firm_id: str | None
    client_original_id: str | None
    first_seen_sort_index: int
    last_update_sort_index: int
    last_event_class: str

    @classmethod
    def from_active_order(cls, order: ActiveOrder) -> ReplayOrder:
        return cls(
            order_id=order.order_id,
            side=order.side,
            price=order.price,
            leaves_qty=order.leaves_qty,
            displayed_qty=order.displayed_qty,
            order_qty=order.order_qty,
            order_priority=order.order_priority,
            order_type_code=order.order_type_code,
            order_type_label=order.order_type_label,
            time_in_force_code=order.time_in_force_code,
            firm_id=order.firm_id,
            client_original_id=order.client_original_id,
            first_seen_sort_index=order.first_seen_sort_index,
            last_update_sort_index=order.last_update_sort_index,
            last_event_class=order.last_event_class,
        )

    def to_active_order(self) -> ActiveOrder:
        """Return a fresh mutable order for consumers of a captured boundary."""
        return ActiveOrder(
            order_id=self.order_id,
            side=self.side,
            price=self.price,
            leaves_qty=self.leaves_qty,
            displayed_qty=self.displayed_qty,
            order_qty=self.order_qty,
            order_priority=self.order_priority,
            order_type_code=self.order_type_code,
            order_type_label=self.order_type_label,
            time_in_force_code=self.time_in_force_code,
            firm_id=self.firm_id,
            client_original_id=self.client_original_id,
            first_seen_sort_index=self.first_seen_sort_index,
            last_update_sort_index=self.last_update_sort_index,
            last_event_class=self.last_event_class,
        )


@dataclass(frozen=True)
class ReplayObservation:
    """Immutable pre/post active-order boundaries for one canonical event."""

    partition_id: str | None
    sort_index: int
    event_ts: datetime | None
    event: Mapping[str, Any]
    pre_active_orders: Mapping[str, ReplayOrder]
    post_active_orders: Mapping[str, ReplayOrder]


ReplayObserver = Callable[[ReplayObservation], None]


def parse_timestamp(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text or text.lower() in {"nan", "none", "null"}:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc)
    return parsed.replace(tzinfo=None)


def choose_event_timestamp(event: Mapping[str, Any]) -> datetime | None:
    for key in EVENT_TIMESTAMP_PRECEDENCE:
        parsed = parse_timestamp(event.get(key))
        if parsed is not None:
            return parsed
    return None


def _immutable_active_orders(active_orders: Mapping[str, ActiveOrder]) -> Mapping[str, ReplayOrder]:
    return MappingProxyType(
        {
            order_id: ReplayOrder.from_active_order(order)
            for order_id, order in active_orders.items()
        }
    )


def replay_lob(
    raw_events: pl.DataFrame,
    *,
    config: LOBConfig | None = None,
    max_rows: int | None = None,
    observer: ReplayObserver | None = None,
    retain_events: bool = True,
) -> tuple[dict[str, Any], ...]:
    """Replay canonical LOB state and optionally observe immutable event boundaries.

    The event order, normalization, partition resets, event application, and
    aggressive-residual flushing are the same primitives used by reconstruction.
    Observations are emitted after the event's post-flush boundary; their pre and
    post order maps contain frozen value copies, never the live active-order map.
    """
    if observer is None:
        return replay_events(
            raw_events,
            config=config,
            max_rows=max_rows,
            retain_events=retain_events,
        )

    pre_active_orders: Mapping[str, ReplayOrder] = MappingProxyType({})

    def on_pre_event(
        event: dict[str, Any], active_orders: dict[str, ActiveOrder], partition_id: str | None
    ) -> None:
        nonlocal pre_active_orders
        pre_active_orders = _immutable_active_orders(active_orders)

    def on_post_event(
        event: dict[str, Any], active_orders: dict[str, ActiveOrder], partition_id: str | None, mutation_flags: list[str]
    ) -> None:
        observer(
            ReplayObservation(
                partition_id=partition_id,
                sort_index=int(event["sort_index"]),
                event_ts=choose_event_timestamp(event),
                event=MappingProxyType(dict(event)),
                pre_active_orders=pre_active_orders,
                post_active_orders=_immutable_active_orders(active_orders),
            )
        )

    return replay_events(
        raw_events,
        config=config,
        max_rows=max_rows,
        hooks=ReplayHooks(on_pre_event=on_pre_event, on_post_event=on_post_event),
        retain_events=retain_events,
    )
