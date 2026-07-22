"""Deterministic, pure clustering of raw passive execution messages.

The builder deliberately knows nothing about the LOB reconstruction.  Callers pass
an already ordered stream containing eligible passive fills plus lifecycle rows.
It returns analytical cluster rows and one provenance row for every child fill.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


LIFECYCLE_BREAK_CLASSES = {
    "cancel",
    "modify_order",
    "new_order",
    "session_reload",
    "iceberg_refill",
    "move_dark_to_cob",
}


@dataclass
class PendingExecutionCluster:
    partition_id: str | None
    order_id: str
    client_original_id: str
    side: str
    price: float
    first_sort_index: int
    last_sort_index: int
    start_ts: datetime
    end_ts: datetime
    first_event_id: Any = None
    last_event_id: Any = None
    weighted_price_notional: float = 0.0
    fill_qty: float = 0.0
    child_fill_count: int = 0
    first_metric_row: dict[str, Any] = field(default_factory=dict)
    members: list[dict[str, Any]] = field(default_factory=list)


def _as_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            value = value.astimezone(timezone.utc)
        return value.replace(tzinfo=None)
    return None


def _finite_positive(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) and numeric > 0 else None


def _fill_identity(fill: Mapping[str, Any]) -> tuple[str | None, str, str, str, float] | None:
    order_id = fill.get("ORDERID")
    client_id = fill.get("client_original_id")
    side = fill.get("side_label")
    price = _finite_positive(fill.get("event_price", fill.get("ORDERPX")))
    if order_id is None or client_id is None or side not in {"bid", "ask"} or price is None:
        return None
    return fill.get("partition_id"), str(order_id), str(client_id), str(side), price


def is_valid_passive_fill(fill: Mapping[str, Any]) -> bool:
    """Return whether a record has the minimum identity and quantity for a cluster."""
    passive = fill.get("is_passive_fill")
    if passive is None:
        passive = (
            fill.get("event_class") == "fill"
            and str(fill.get("AGGRESSIVEORDER") or "").upper() != "Y"
        )
    return bool(passive) and _fill_identity(fill) is not None and _finite_positive(fill.get("fill_qty")) is not None and _as_datetime(fill.get("event_ts")) is not None and fill.get("sort_index") is not None


def can_extend_execution_cluster(
    cluster: PendingExecutionCluster,
    fill: Mapping[str, Any],
    *,
    max_gap_ms: int,
    intervening_lifecycle_break: bool = False,
) -> bool:
    """Whether a valid child fill can join ``cluster`` under the operational rule."""
    if max_gap_ms < 0 or intervening_lifecycle_break or not is_valid_passive_fill(fill):
        return False
    identity = _fill_identity(fill)
    if identity != (cluster.partition_id, cluster.order_id, cluster.client_original_id, cluster.side, cluster.price):
        return False
    timestamp = _as_datetime(fill.get("event_ts"))
    assert timestamp is not None
    gap_ms = (timestamp - cluster.end_ts).total_seconds() * 1_000.0
    return 0 <= gap_ms <= max_gap_ms


def _member_row(cluster_id: str, fill: Mapping[str, Any], *, price: float, qty: float, timestamp: datetime) -> dict[str, Any]:
    return {
        "execution_cluster_id": cluster_id,
        "partition_id": fill.get("partition_id"),
        "child_order_id": str(fill.get("ORDERID")) if fill.get("ORDERID") is not None else None,
        "child_event_id": fill.get("EVENTID"),
        "child_sort_index": int(fill["sort_index"]),
        "child_execution_id": fill.get("EXECUTIONID"),
        "child_trade_uid": fill.get("TRADEUNIQUEIDENTIFIER"),
        "child_fill_qty": qty,
        "child_fill_price": price,
        "child_event_ts": timestamp,
    }


def _cluster_id(first_sort_index: int, last_sort_index: int) -> str:
    return f"EC{first_sort_index:09d}-{last_sort_index:09d}"


def _start_cluster(fill: Mapping[str, Any]) -> PendingExecutionCluster:
    identity = _fill_identity(fill)
    timestamp = _as_datetime(fill.get("event_ts"))
    qty = _finite_positive(fill.get("fill_qty"))
    assert identity is not None and timestamp is not None and qty is not None
    partition_id, order_id, client_id, side, price = identity
    return PendingExecutionCluster(
        partition_id=partition_id,
        order_id=order_id,
        client_original_id=client_id,
        side=side,
        price=price,
        first_sort_index=int(fill["sort_index"]),
        last_sort_index=int(fill["sort_index"]),
        start_ts=timestamp,
        end_ts=timestamp,
        first_event_id=fill.get("EVENTID"),
        last_event_id=fill.get("EVENTID"),
        weighted_price_notional=qty * price,
        fill_qty=qty,
        child_fill_count=1,
        first_metric_row=dict(fill.get("metric_row") or {}),
        members=[],
    )


def _append_child(cluster: PendingExecutionCluster, fill: Mapping[str, Any]) -> None:
    timestamp = _as_datetime(fill.get("event_ts"))
    qty = _finite_positive(fill.get("fill_qty"))
    price = _finite_positive(fill.get("event_price", fill.get("ORDERPX")))
    assert timestamp is not None and qty is not None and price is not None
    cluster.last_sort_index = int(fill["sort_index"])
    cluster.end_ts = timestamp
    cluster.last_event_id = fill.get("EVENTID")
    cluster.weighted_price_notional += qty * price
    cluster.fill_qty += qty
    cluster.child_fill_count += 1


def finalize_execution_cluster(cluster: PendingExecutionCluster, *, gap_ms: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Materialize one cluster row and its child-fill provenance rows."""
    cluster_id = _cluster_id(cluster.first_sort_index, cluster.last_sort_index)
    price = cluster.weighted_price_notional / cluster.fill_qty
    cluster_row = {
        **cluster.first_metric_row,
        "execution_cluster_id": cluster_id,
        "partition_id": cluster.partition_id,
        "cluster_first_event_id": cluster.first_event_id,
        "cluster_last_event_id": cluster.last_event_id,
        "cluster_first_sort_index": cluster.first_sort_index,
        "cluster_last_sort_index": cluster.last_sort_index,
        "cluster_start_ts": cluster.start_ts,
        "cluster_end_ts": cluster.end_ts,
        "child_fill_count": cluster.child_fill_count,
        "event_order_id": cluster.order_id,
        "event_client_original_id": cluster.client_original_id,
        "event_side": cluster.side,
        "event_price": price,
        "fill_qty": cluster.fill_qty,
        "execution_cluster_gap_ms": gap_ms,
        "execution_cluster_quality_flags": "",
        # Backward-compatible aliases; cluster time semantics are explicit above.
        "sort_index": cluster.first_sort_index,
        "event_ts": cluster.start_ts,
        "ORDERID": cluster.order_id,
        "client_id": cluster.client_original_id,
        "execution_side": cluster.side,
    }
    member_rows = [
        _member_row(cluster_id, member, price=_finite_positive(member.get("event_price", member.get("ORDERPX"))) or 0.0, qty=_finite_positive(member.get("fill_qty")) or 0.0, timestamp=_as_datetime(member.get("event_ts")) or cluster.start_ts)
        for member in cluster.members
    ]
    return cluster_row, member_rows


def _is_terminal_fill(fill: Mapping[str, Any]) -> bool:
    leaves = fill.get("LEAVESQTY")
    try:
        return leaves is not None and float(leaves) <= 0
    except (TypeError, ValueError):
        return False


def _lifecycle_break_key(event: Mapping[str, Any]) -> tuple[str | None, str] | None:
    if event.get("event_class") not in LIFECYCLE_BREAK_CLASSES:
        return None
    order_id = event.get("ORDERID")
    return (event.get("partition_id"), str(order_id)) if order_id is not None else None


def cluster_passive_execution_fills(
    events: Iterable[Mapping[str, Any]], *, max_gap_ms: int = 100
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Cluster ordered passive fill messages while retaining one raw-member row each.

    Non-fill aggressive legs and unrelated new orders remain in the stream but do
    not close a pending cluster.  Lifecycle events for the same partition/order do.
    """
    if max_gap_ms < 0:
        raise ValueError("max_gap_ms must be non-negative")
    pending: dict[tuple[str | None, str, str, str, float], PendingExecutionCluster] = {}
    clusters: list[dict[str, Any]] = []
    members: list[dict[str, Any]] = []

    def finalize_key(key: tuple[str | None, str, str, str, float]) -> None:
        cluster = pending.pop(key, None)
        if cluster is not None:
            row, member_rows = finalize_execution_cluster(cluster, gap_ms=max_gap_ms)
            clusters.append(row)
            members.extend(member_rows)

    for event in events:
        lifecycle_key = _lifecycle_break_key(event)
        if lifecycle_key is not None:
            for key in [key for key in pending if key[:2] == lifecycle_key]:
                finalize_key(key)
            continue
        if not is_valid_passive_fill(event):
            continue
        key = _fill_identity(event)
        assert key is not None
        existing = pending.get(key)
        if existing is not None and not can_extend_execution_cluster(existing, event, max_gap_ms=max_gap_ms):
            finalize_key(key)
            existing = None
        if existing is None:
            existing = _start_cluster(event)
            pending[key] = existing
        else:
            _append_child(existing, event)
        existing.members.append(dict(event))
        if _is_terminal_fill(event):
            finalize_key(key)

    for key in sorted(pending, key=lambda item: (str(item[0]), item[1], item[2], item[3], item[4])):
        finalize_key(key)
    clusters.sort(key=lambda row: (int(row["cluster_first_sort_index"]), int(row["cluster_last_sort_index"])))
    members.sort(key=lambda row: int(row["child_sort_index"]))
    return clusters, members
