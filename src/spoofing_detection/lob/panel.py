from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import polars as pl

from .config import LOBConfig
from .models import ActiveOrder, ReconstructionResult
from .normalize import is_visible_resting_event, join_flags, normalize_event

SORT_COLUMNS = [
    "TRADEDATE",
    "MIC",
    "MARKETCODE",
    "SYMBOLINDEX",
    "EMM (*)",
    "SEQUENCETIME",
    "HDR_APPLKEYSEQUENCENUMBER",
    "HDR_HWMSEQUENCENUMBER",
    "HDR_OFFSETID",
    "BOOKIN",
    "BOOKOUTTIME",
    "TRADETIME",
    "ROW_NUMBER",
]

MUTATING_UPDATE_CLASSES = {
    "new_order",
    "session_reload",
    "modify_order",
    "iceberg_refill",
    "move_dark_to_cob",
}

NON_VISIBLE_RESTING_ORDER_TYPES = {
    "market",
    "stop_market_or_stop_market_on_quote",
    "stop_limit_or_stop_limit_on_quote",
    "mid_point_peg",
}


def sort_events(df: pl.DataFrame) -> pl.DataFrame:
    available = [col for col in SORT_COLUMNS if col in df.columns]
    if not available:
        return df
    return df.sort(available, nulls_last=True)


def _make_active_order(event: dict[str, Any], *, first_seen_sort_index: int | None = None) -> ActiveOrder:
    return ActiveOrder(
        order_id=event["ORDERID"],
        side=event["side_label"],
        price=float(event["ORDERPX"]),
        leaves_qty=float(event["LEAVESQTY"] or 0),
        displayed_qty=float(event["DISPLAYEDQTY"] or 0),
        order_qty=event["ORDERQTY"],
        order_priority=event["ORDERPRIORITY"],
        order_type_code=event["event_order_type_code"],
        order_type_label=event["event_order_type_label"],
        time_in_force_code=event["time_in_force_code"],
        firm_id=event["firm_id"],
        client_original_id=event["client_original_id"],
        first_seen_sort_index=first_seen_sort_index or event["sort_index"],
        last_update_sort_index=event["sort_index"],
        last_event_class=event["event_class"],
    )


def _visible_qty(order: ActiveOrder) -> float:
    if order.leaves_qty <= 0 or order.displayed_qty <= 0:
        return 0.0
    return order.displayed_qty


def book_summary(active_orders: dict[str, ActiveOrder], *, prefix: str, top_n: int) -> dict[str, Any]:
    levels: dict[str, dict[float, dict[str, float]]] = {"bid": {}, "ask": {}}
    totals = {
        "bid_visible_qty_total": 0.0,
        "ask_visible_qty_total": 0.0,
        "bid_order_count_total": 0,
        "ask_order_count_total": 0,
    }
    for order in active_orders.values():
        qty = _visible_qty(order)
        if qty <= 0 or order.side not in {"bid", "ask"}:
            continue
        side_levels = levels[order.side]
        level = side_levels.setdefault(order.price, {"visible_qty": 0.0, "order_count": 0})
        level["visible_qty"] += qty
        level["order_count"] += 1
        totals[f"{order.side}_visible_qty_total"] += qty
        totals[f"{order.side}_order_count_total"] += 1

    bid_prices = sorted(levels["bid"], reverse=True)
    ask_prices = sorted(levels["ask"])
    best_bid = bid_prices[0] if bid_prices else None
    best_ask = ask_prices[0] if ask_prices else None
    mid = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
    spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None

    out: dict[str, Any] = {
        f"{prefix}_best_bid": best_bid,
        f"{prefix}_best_ask": best_ask,
        f"{prefix}_mid": mid,
        f"{prefix}_spread": spread,
        f"{prefix}_bid_visible_qty_total": totals["bid_visible_qty_total"],
        f"{prefix}_ask_visible_qty_total": totals["ask_visible_qty_total"],
        f"{prefix}_bid_order_count_total": totals["bid_order_count_total"],
        f"{prefix}_ask_order_count_total": totals["ask_order_count_total"],
        f"book_locked_{prefix}_flag": bool(spread == 0) if spread is not None else False,
        f"book_crossed_{prefix}_flag": bool(spread < 0) if spread is not None else False,
    }
    for side, prices in (("bid", bid_prices), ("ask", ask_prices)):
        for idx in range(1, top_n + 1):
            if idx <= len(prices):
                price = prices[idx - 1]
                level = levels[side][price]
                out[f"{prefix}_{side}_level_{idx}_price"] = price
                out[f"{prefix}_{side}_level_{idx}_visible_qty"] = level["visible_qty"]
                out[f"{prefix}_{side}_level_{idx}_order_count"] = int(level["order_count"])
            else:
                out[f"{prefix}_{side}_level_{idx}_price"] = None
                out[f"{prefix}_{side}_level_{idx}_visible_qty"] = 0.0
                out[f"{prefix}_{side}_level_{idx}_order_count"] = 0
    return out


def agent_aggregates(
    active_orders: dict[str, ActiveOrder],
    *,
    dimension: str,
    agent_id: str | None,
    event_side: str | None,
    prefix: str,
) -> dict[str, Any]:
    out = {
        f"{prefix}_{dimension}_active_bid_visible_qty": 0.0,
        f"{prefix}_{dimension}_active_ask_visible_qty": 0.0,
        f"{prefix}_{dimension}_active_bid_leaves_qty": 0.0,
        f"{prefix}_{dimension}_active_ask_leaves_qty": 0.0,
        f"{prefix}_{dimension}_active_bid_order_count": 0,
        f"{prefix}_{dimension}_active_ask_order_count": 0,
    }
    if agent_id is not None:
        for order in active_orders.values():
            order_agent = order.firm_id if dimension == "firm" else order.client_original_id
            if order_agent != agent_id or order.side not in {"bid", "ask"}:
                continue
            out[f"{prefix}_{dimension}_active_{order.side}_visible_qty"] += _visible_qty(order)
            out[f"{prefix}_{dimension}_active_{order.side}_leaves_qty"] += max(order.leaves_qty, 0.0)
            out[f"{prefix}_{dimension}_active_{order.side}_order_count"] += 1
    if event_side in {"bid", "ask"}:
        opposite = "ask" if event_side == "bid" else "bid"
        out[f"{prefix}_{dimension}_active_same_side_visible_qty"] = out[
            f"{prefix}_{dimension}_active_{event_side}_visible_qty"
        ]
        out[f"{prefix}_{dimension}_active_opposite_side_visible_qty"] = out[
            f"{prefix}_{dimension}_active_{opposite}_visible_qty"
        ]
    else:
        out[f"{prefix}_{dimension}_active_same_side_visible_qty"] = 0.0
        out[f"{prefix}_{dimension}_active_opposite_side_visible_qty"] = 0.0
    return out


def _side_code(side: str | None) -> int | None:
    if side == "bid":
        return 1
    if side == "ask":
        return 2
    return None


def _partition_id(event: dict[str, Any] | None) -> str | None:
    if event is None:
        return None
    parts = [event.get("TRADEDATE"), event.get("MIC"), event.get("MARKETCODE"), event.get("SYMBOLINDEX"), event.get("EMM (*)")]
    return "|".join("" if part is None else str(part) for part in parts)


def depth_snapshot_rows(
    active_orders: dict[str, ActiveOrder],
    *,
    event: dict[str, Any] | None,
    snapshot_sort_index: int | None,
    snapshot_reason: str,
) -> list[dict[str, Any]]:
    levels: dict[tuple[str, float], dict[str, float]] = {}
    for order in active_orders.values():
        qty = _visible_qty(order)
        if qty <= 0 or order.side not in {"bid", "ask"}:
            continue
        key = (order.side, order.price)
        level = levels.setdefault(key, {"visible_qty": 0.0, "leaves_qty": 0.0, "order_count": 0})
        level["visible_qty"] += qty
        level["leaves_qty"] += max(order.leaves_qty, 0.0)
        level["order_count"] += 1

    rows: list[dict[str, Any]] = []
    for side in ("bid", "ask"):
        prices = sorted(
            [price for level_side, price in levels if level_side == side],
            reverse=(side == "bid"),
        )
        for rank, price in enumerate(prices, start=1):
            level = levels[(side, price)]
            rows.append(
                {
                    "partition_id": _partition_id(event),
                    "snapshot_sort_index": snapshot_sort_index,
                    "snapshot_reason": snapshot_reason,
                    "side": side,
                    "side_code": _side_code(side),
                    "price": price,
                    "rank": rank,
                    "visible_qty": level["visible_qty"],
                    "leaves_qty": level["leaves_qty"],
                    "order_count": int(level["order_count"]),
                }
            )
    return rows


def active_order_snapshot_rows(
    active_orders: dict[str, ActiveOrder],
    *,
    event: dict[str, Any] | None,
    snapshot_sort_index: int | None,
    snapshot_reason: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for order in sorted(active_orders.values(), key=lambda order: (order.side, order.price, order.order_id)):
        rows.append(
            {
                "partition_id": _partition_id(event),
                "snapshot_sort_index": snapshot_sort_index,
                "snapshot_reason": snapshot_reason,
                "ORDERID": order.order_id,
                "side": order.side,
                "side_code": _side_code(order.side),
                "price": order.price,
                "leaves_qty": order.leaves_qty,
                "displayed_qty": order.displayed_qty,
                "original_order_qty": order.order_qty,
                "order_type_code": order.order_type_code,
                "time_in_force_code": order.time_in_force_code,
                "ORDERPRIORITY": order.order_priority,
                "firm_id": order.firm_id,
                "client_original_id": order.client_original_id,
                "client_original_id_missing_flag": order.client_original_id is None,
                "FIRMID": order.firm_id,
                "NMSC_ORIGINALCLIENTIDSHORTCODE": order.client_original_id,
                "first_seen_time": order.first_seen_sort_index,
                "last_update_time": order.last_update_sort_index,
                "last_event_type": order.last_event_class,
                "issue_flags": None,
            }
        )
    return rows


def agent_long_rows(
    event: dict[str, Any],
    *,
    pre_firm: dict[str, Any],
    post_firm: dict[str, Any],
    pre_client: dict[str, Any],
    post_client: dict[str, Any],
    pre_distance: dict[str, Any],
    post_distance: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    specs = [
        ("firm", event["firm_id"], "FIRMID", event["firm_id"] is None, pre_firm, post_firm),
        (
            "client_original",
            event["client_original_id"],
            "NMSC_ORIGINALCLIENTIDSHORTCODE",
            event["client_original_id"] is None,
            pre_client,
            post_client,
        ),
    ]
    for dimension, agent_id, source, missing, pre_values, post_values in specs:
        rows.append(
            {
                "partition_id": _partition_id(event),
                "sort_index": event["sort_index"],
                "agent_dimension": dimension,
                "agent_id": agent_id,
                "agent_id_source": source,
                "agent_id_missing_flag": missing,
                "event_ORDERID": event["ORDERID"],
                "event_side_code": event["side_code"],
                "event_side": event["side_label"],
                "event_class": event["event_class"],
                "pre_agent_bid_visible_qty": pre_values[f"pre_{dimension}_active_bid_visible_qty"],
                "pre_agent_ask_visible_qty": pre_values[f"pre_{dimension}_active_ask_visible_qty"],
                "pre_agent_bid_leaves_qty": pre_values[f"pre_{dimension}_active_bid_leaves_qty"],
                "pre_agent_ask_leaves_qty": pre_values[f"pre_{dimension}_active_ask_leaves_qty"],
                "pre_agent_bid_order_count": pre_values[f"pre_{dimension}_active_bid_order_count"],
                "pre_agent_ask_order_count": pre_values[f"pre_{dimension}_active_ask_order_count"],
                "post_agent_bid_visible_qty": post_values[f"post_{dimension}_active_bid_visible_qty"],
                "post_agent_ask_visible_qty": post_values[f"post_{dimension}_active_ask_visible_qty"],
                "post_agent_bid_leaves_qty": post_values[f"post_{dimension}_active_bid_leaves_qty"],
                "post_agent_ask_leaves_qty": post_values[f"post_{dimension}_active_ask_leaves_qty"],
                "post_agent_bid_order_count": post_values[f"post_{dimension}_active_bid_order_count"],
                "post_agent_ask_order_count": post_values[f"post_{dimension}_active_ask_order_count"],
                "pre_agent_same_side_visible_qty": pre_values[f"pre_{dimension}_active_same_side_visible_qty"],
                "pre_agent_opposite_side_visible_qty": pre_values[f"pre_{dimension}_active_opposite_side_visible_qty"],
                "post_agent_same_side_visible_qty": post_values[f"post_{dimension}_active_same_side_visible_qty"],
                "post_agent_opposite_side_visible_qty": post_values[f"post_{dimension}_active_opposite_side_visible_qty"],
                "pre_event_order_same_side_distance_price": pre_distance[
                    "pre_event_order_same_side_distance_price"
                ],
                "pre_event_order_same_side_distance_bps": pre_distance[
                    "pre_event_order_same_side_distance_bps"
                ],
                "post_event_order_same_side_distance_price": post_distance[
                    "post_event_order_same_side_distance_price"
                ],
                "post_event_order_same_side_distance_bps": post_distance[
                    "post_event_order_same_side_distance_bps"
                ],
                "agent_issue_flags": "missing_agent_id" if missing else None,
            }
        )
    return rows


def event_distance(event: dict[str, Any], summary: dict[str, Any], *, prefix: str) -> dict[str, Any]:
    side = event["side_label"]
    price = event["ORDERPX"]
    distance = None
    if price is not None and side == "bid" and summary[f"{prefix}_best_bid"] is not None:
        distance = summary[f"{prefix}_best_bid"] - price
    elif price is not None and side == "ask" and summary[f"{prefix}_best_ask"] is not None:
        distance = price - summary[f"{prefix}_best_ask"]
    mid = summary[f"{prefix}_mid"]
    distance_bps = 10000.0 * distance / mid if distance is not None and mid not in (None, 0) else None
    return {
        f"{prefix}_event_order_same_side_distance_price": distance,
        f"{prefix}_event_order_same_side_distance_bps": distance_bps,
    }


def _requires_prior_visible_order(event: dict[str, Any]) -> bool:
    """Whether an unseen cancel/fill should be treated as a reconstruction issue.

    Events that cannot seed visible resting state should not later look like
    missing active orders when they are filled or cancelled.
    """
    if event["event_order_type_label"] in NON_VISIBLE_RESTING_ORDER_TYPES:
        return False
    if event["ORDERPX"] is None:
        return False
    return True


def _is_aggressive_event(event: dict[str, Any]) -> bool:
    return str(event.get("AGGRESSIVEORDER") or "").upper() == "Y"


def _would_lock_or_cross_book(active_orders: dict[str, ActiveOrder], event: dict[str, Any]) -> bool:
    price = event.get("ORDERPX")
    side = event.get("side_label")
    if price is None or side not in {"bid", "ask"}:
        return False
    summary = book_summary(active_orders, prefix="pre", top_n=1)
    if side == "bid" and summary["pre_best_ask"] is not None:
        return price >= summary["pre_best_ask"]
    if side == "ask" and summary["pre_best_bid"] is not None:
        return price <= summary["pre_best_bid"]
    return False


def _fill_group_key(event: dict[str, Any]) -> tuple[Any, ...] | None:
    if event["event_class"] != "fill":
        return None
    execution_id = event.get("EXECUTIONID")
    trade_uid = event.get("TRADEUNIQUEIDENTIFIER")
    trade_time = event.get("TRADETIME")
    if execution_id in (None, 0, "0") and trade_uid is None and trade_time is None:
        return None
    return (
        event.get("TRADEDATE"),
        event.get("MIC"),
        event.get("MARKETCODE"),
        event.get("SYMBOLINDEX"),
        event.get("EMM (*)"),
        execution_id,
        trade_uid,
        trade_time,
        event.get("LASTTRADEDPX"),
    )


def _flush_pending_aggressive_residuals(
    active_orders: dict[str, ActiveOrder],
    pending_aggressive_residuals: dict[str, tuple[dict[str, Any], tuple[Any, ...] | None]],
    *,
    keep_group: tuple[Any, ...] | None,
) -> list[str]:
    flags: list[str] = []
    for order_id, (event, group_key) in list(pending_aggressive_residuals.items()):
        if keep_group is not None and group_key == keep_group:
            continue
        if not is_visible_resting_event(event):
            pending_aggressive_residuals.pop(order_id, None)
            continue
        if _would_lock_or_cross_book(active_orders, event):
            # The residual is still marketable against remaining opposite-side
            # liquidity. Keep it pending until later fills/cancels make it safe
            # to rest, or until a terminal event removes it.
            continue
        first_seen = active_orders[order_id].first_seen_sort_index if order_id in active_orders else None
        active_orders[order_id] = _make_active_order(event, first_seen_sort_index=first_seen)
        pending_aggressive_residuals.pop(order_id, None)
    return flags


def _apply_event(
    active_orders: dict[str, ActiveOrder],
    event: dict[str, Any],
    *,
    pending_aggressive_residuals: dict[str, tuple[dict[str, Any], tuple[Any, ...] | None]],
    non_resting_order_ids: set[str],
) -> list[str]:
    flags: list[str] = []
    order_id = event["ORDERID"]
    event_class = event["event_class"]
    if order_id is None:
        return flags

    if event_class == "cancel":
        was_pending = pending_aggressive_residuals.pop(order_id, None) is not None
        if (
            order_id not in active_orders
            and not was_pending
            and order_id not in non_resting_order_ids
            and _requires_prior_visible_order(event)
        ):
            flags.append("cancel_for_unseen_order")
        active_orders.pop(order_id, None)
        non_resting_order_ids.discard(order_id)
        return flags

    if event_class == "fill":
        if (event["LEAVESQTY"] or 0) <= 0:
            was_pending = pending_aggressive_residuals.pop(order_id, None) is not None
            if (
                order_id not in active_orders
                and not was_pending
                and order_id not in non_resting_order_ids
                and not _is_aggressive_event(event)
                and _requires_prior_visible_order(event)
            ):
                flags.append("full_fill_for_unseen_order")
            active_orders.pop(order_id, None)
            non_resting_order_ids.discard(order_id)
            return flags
        if is_visible_resting_event(event):
            if _is_aggressive_event(event) or _would_lock_or_cross_book(active_orders, event):
                active_orders.pop(order_id, None)
                pending_aggressive_residuals[order_id] = (event, _fill_group_key(event))
                non_resting_order_ids.add(order_id)
                return flags
            first_seen = active_orders[order_id].first_seen_sort_index if order_id in active_orders else None
            if order_id not in active_orders and order_id not in non_resting_order_ids:
                flags.append("partial_fill_for_unseen_order")
            active_orders[order_id] = _make_active_order(event, first_seen_sort_index=first_seen)
            non_resting_order_ids.discard(order_id)
        return flags

    if event_class in MUTATING_UPDATE_CLASSES:
        if is_visible_resting_event(event):
            if _would_lock_or_cross_book(active_orders, event):
                active_orders.pop(order_id, None)
                pending_aggressive_residuals.pop(order_id, None)
                non_resting_order_ids.add(order_id)
                flags.append("marketable_order_not_resting")
                return flags
            first_seen = active_orders[order_id].first_seen_sort_index if order_id in active_orders else None
            if event_class == "modify_order" and order_id not in active_orders and order_id not in non_resting_order_ids:
                flags.append("modify_for_unseen_order")
            active_orders[order_id] = _make_active_order(event, first_seen_sort_index=first_seen)
            non_resting_order_ids.discard(order_id)
        elif event_class == "modify_order" and order_id in active_orders and (event["LEAVESQTY"] or 0) <= 0:
            active_orders.pop(order_id, None)
            non_resting_order_ids.discard(order_id)
        elif event_class in {"new_order", "session_reload"}:
            non_resting_order_ids.add(order_id)
        return flags

    # Trigger/VFA and other special rows are retained in the panel but do not
    # mutate visible depth unless later documented.
    return flags


def _event_identity(event: dict[str, Any]) -> dict[str, Any]:
    return {
        "sort_index": event["sort_index"],
        "TRADEDATE": event["TRADEDATE"],
        "MIC": event["MIC"],
        "MARKETCODE": event["MARKETCODE"],
        "SYMBOLINDEX": event["SYMBOLINDEX"],
        "EMM (*)": event["EMM (*)"],
        "ISIN": event["ISIN"],
        "EVENTID": event["EVENTID"],
        "ORDERID": event["ORDERID"],
        "ORDERPRIORITY": event["ORDERPRIORITY"],
        "event_class": event["event_class"],
        "event_type_code": event["event_type_code"],
        "event_type_label_observed": event["event_type_label_observed"],
        "event_side": event["side_label"],
        "event_price": event["ORDERPX"],
        "event_order_qty": event["ORDERQTY"],
        "event_displayed_qty": event["DISPLAYEDQTY"],
        "event_leaves_qty": event["LEAVESQTY"],
        "event_last_shares": event["LASTSHARES"],
        "event_last_traded_price": event["LASTTRADEDPX"],
        "event_order_type_code": event["event_order_type_code"],
        "event_order_type_label": event["event_order_type_label"],
        "event_firm_id": event["firm_id"],
        "event_client_original_id": event["client_original_id"],
        "firm_id": event["firm_id"],
        "client_original_id": event["client_original_id"],
        "client_original_id_missing_flag": event["client_original_id_missing_flag"],
        "normalization_issue_flags": event["normalization_issue_flags"],
    }


def reconstruct_dataframe(
    df: pl.DataFrame, *, config: LOBConfig | None = None, max_rows: int | None = None
) -> ReconstructionResult:
    config = config or LOBConfig()
    sorted_df = sort_events(df)
    if max_rows is not None:
        sorted_df = sorted_df.head(max_rows)
    events = [
        normalize_event(raw_row, sort_index=idx, config=config)
        for idx, raw_row in enumerate(sorted_df.iter_rows(named=True), start=1)
    ]
    active_orders: dict[str, ActiveOrder] = {}
    pending_aggressive_residuals: dict[str, tuple[dict[str, Any], tuple[Any, ...] | None]] = {}
    non_resting_order_ids: set[str] = set()
    normalized_rows: list[dict[str, Any]] = []
    panel_rows: list[dict[str, Any]] = []
    agent_rows: list[dict[str, Any]] = []
    active_snapshot_rows: list[dict[str, Any]] = []
    price_depth_rows: list[dict[str, Any]] = []
    issue_counts: dict[str, int] = {}
    last_event: dict[str, Any] | None = None
    current_partition_id: str | None = None
    partitions_processed = 0
    active_orders_end_total = 0

    for event_index, event in enumerate(events):
        event_partition_id = _partition_id(event)
        if current_partition_id is None:
            current_partition_id = event_partition_id
            partitions_processed = 1
        elif event_partition_id != current_partition_id:
            _flush_pending_aggressive_residuals(
                active_orders,
                pending_aggressive_residuals,
                keep_group=None,
            )
            if config.snapshot_mode == "end_of_partition" and last_event is not None:
                active_snapshot_rows.extend(
                    active_order_snapshot_rows(
                        active_orders,
                        event=last_event,
                        snapshot_sort_index=last_event["sort_index"],
                        snapshot_reason="end_of_partition",
                    )
                )
                price_depth_rows.extend(
                    depth_snapshot_rows(
                        active_orders,
                        event=last_event,
                        snapshot_sort_index=last_event["sort_index"],
                        snapshot_reason="end_of_partition",
                    )
                )
            active_orders_end_total += len(active_orders)
            active_orders = {}
            pending_aggressive_residuals = {}
            non_resting_order_ids = set()
            current_partition_id = event_partition_id
            partitions_processed += 1
        last_event = event
        normalized_rows.append(event)

        pre_summary = book_summary(active_orders, prefix="pre", top_n=config.top_n)
        pre_firm = agent_aggregates(
            active_orders,
            dimension="firm",
            agent_id=event["firm_id"],
            event_side=event["side_label"],
            prefix="pre",
        )
        pre_client = agent_aggregates(
            active_orders,
            dimension="client_original",
            agent_id=event["client_original_id"],
            event_side=event["side_label"],
            prefix="pre",
        )
        pre_distance = event_distance(event, pre_summary, prefix="pre")

        mutation_flags = _apply_event(
            active_orders,
            event,
            pending_aggressive_residuals=pending_aggressive_residuals,
            non_resting_order_ids=non_resting_order_ids,
        )
        next_event = events[event_index + 1] if event_index + 1 < len(events) else None
        next_group = _fill_group_key(next_event) if next_event is not None else None
        mutation_flags.extend(
            _flush_pending_aggressive_residuals(
                active_orders,
                pending_aggressive_residuals,
                keep_group=next_group,
            )
        )

        post_summary = book_summary(active_orders, prefix="post", top_n=config.top_n)
        post_firm = agent_aggregates(
            active_orders,
            dimension="firm",
            agent_id=event["firm_id"],
            event_side=event["side_label"],
            prefix="post",
        )
        post_client = agent_aggregates(
            active_orders,
            dimension="client_original",
            agent_id=event["client_original_id"],
            event_side=event["side_label"],
            prefix="post",
        )
        post_distance = event_distance(event, post_summary, prefix="post")

        all_flags = []
        if event["normalization_issue_flags"]:
            all_flags.extend(str(event["normalization_issue_flags"]).split(";"))
        all_flags.extend(mutation_flags)
        for flag in all_flags:
            issue_counts[flag] = issue_counts.get(flag, 0) + 1

        panel_rows.append(
            {
                **_event_identity(event),
                **pre_summary,
                **post_summary,
                **pre_firm,
                **post_firm,
                **pre_client,
                **post_client,
                **pre_distance,
                **post_distance,
                "lob_issue_flags": join_flags(mutation_flags),
            }
        )
        agent_rows.extend(
            agent_long_rows(
                event,
                pre_firm=pre_firm,
                post_firm=post_firm,
                pre_client=pre_client,
                post_client=post_client,
                pre_distance=pre_distance,
                post_distance=post_distance,
            )
        )
        if config.snapshot_mode == "every_event_for_sample" or (
            config.snapshot_mode == "issue_rows_only" and mutation_flags
        ):
            reason = "post_event" if config.snapshot_mode == "every_event_for_sample" else "issue_row"
            active_snapshot_rows.extend(
                active_order_snapshot_rows(
                    active_orders,
                    event=event,
                    snapshot_sort_index=event["sort_index"],
                    snapshot_reason=reason,
                )
            )
            price_depth_rows.extend(
                depth_snapshot_rows(
                    active_orders,
                    event=event,
                    snapshot_sort_index=event["sort_index"],
                    snapshot_reason=reason,
                )
            )

    if last_event is not None:
        _flush_pending_aggressive_residuals(
            active_orders,
            pending_aggressive_residuals,
            keep_group=None,
        )

    if config.snapshot_mode == "end_of_partition" and last_event is not None:
        active_snapshot_rows.extend(
            active_order_snapshot_rows(
                active_orders,
                event=last_event,
                snapshot_sort_index=last_event["sort_index"],
                snapshot_reason="end_of_partition",
            )
        )
        price_depth_rows.extend(
            depth_snapshot_rows(
                active_orders,
                event=last_event,
                snapshot_sort_index=last_event["sort_index"],
                snapshot_reason="end_of_partition",
            )
        )
    if last_event is not None:
        active_orders_end_total += len(active_orders)

    panel = pl.DataFrame(panel_rows, infer_schema_length=None) if panel_rows else pl.DataFrame()
    normalized = pl.DataFrame(normalized_rows, infer_schema_length=None) if normalized_rows else pl.DataFrame()
    agent_panel = pl.DataFrame(agent_rows, infer_schema_length=None) if agent_rows else pl.DataFrame()
    active_snapshots = (
        pl.DataFrame(active_snapshot_rows, infer_schema_length=None) if active_snapshot_rows else pl.DataFrame()
    )
    price_depth_snapshots = (
        pl.DataFrame(price_depth_rows, infer_schema_length=None) if price_depth_rows else pl.DataFrame()
    )
    validation = {
        "events_processed": len(panel_rows),
        "partitions_processed": partitions_processed,
        "active_orders_end": active_orders_end_total,
        "agent_event_state_rows": len(agent_rows),
        "active_order_snapshot_rows": len(active_snapshot_rows),
        "price_level_depth_snapshot_rows": len(price_depth_rows),
        "snapshot_mode": config.snapshot_mode,
        "issue_counts": issue_counts,
        "top_n": config.top_n,
        "agent_dimensions": list(config.agent_dimensions),
    }
    return ReconstructionResult(
        panel=panel,
        normalized_events=normalized,
        agent_event_state_panel=agent_panel,
        active_order_snapshots=active_snapshots,
        price_level_depth_snapshots=price_depth_snapshots,
        validation=validation,
    )
