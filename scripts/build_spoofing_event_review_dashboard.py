#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from html import escape
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.actor_identity import actor_identity_from_event, actor_identity_from_order
from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.enums import TRADING_CAPACITY_BY_CODE
from spoofing_detection.lob.models import ActiveOrder
from spoofing_detection.lob.normalize import normalize_event
from spoofing_detection.lob.panel import (
    _apply_event,
    _fill_group_key,
    _flush_pending_aggressive_residuals,
    _partition_id,
    sort_events,
)
from spoofing_detection.lob.spoofing_config import (
    DEFAULT_SPOOFING_CONFIG_PATH,
    load_spoofing_config_defaults,
    reject_parameter_overrides,
    spoofing_config_provenance,
)
from spoofing_detection.lob.spoofing_metrics import (
    MSCI_DEFINITION,
    MSCI_RANGE,
    RATIO_ZERO_DENOMINATOR_POLICY,
    _parse_ts,
    choose_event_timestamp,
)


_CONFIGURABLE_DEFAULT_KEYS = {
    "top_n",
    "pre_window_seconds",
    "post_window_seconds",
    "max_events",
    "queue_snapshot_mode",
}

_CONFIG_PARAMETER_OPTIONS = {
    "--top-n",
    "--pre-window-seconds",
    "--post-window-seconds",
    "--max-events",
    "--queue-snapshot-mode",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_SPOOFING_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    try:
        config_defaults = load_spoofing_config_defaults(
            config_path=config_args.config,
            section="event_review",
            allowed_keys=_CONFIGURABLE_DEFAULT_KEYS,
        )
    except (OSError, ValueError) as exc:
        config_parser.error(str(exc))

    parser = argparse.ArgumentParser(
        description="Build an interactive review dashboard and exact queue parquet for matched spoofing-like events.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_args.config,
        help="Authoritative JSON file containing all event-review parameters",
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw input parquet event file")
    parser.add_argument("--execution-metrics", type=Path, required=True, help="execution_metrics.parquet from spoofing run")
    parser.add_argument(
        "--candidate-deceptive-orders",
        type=Path,
        required=True,
        help="candidate_deceptive_orders.parquet from spoofing run",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--top-n", type=int, help="Top book levels to include in queue snapshots")
    parser.add_argument("--pre-window-seconds", type=float, help="Seconds before execution to show")
    parser.add_argument("--post-window-seconds", type=float, help="Seconds after execution to show")
    parser.add_argument("--max-events", type=int, help="Optional cap on matched events for smoke runs")
    parser.add_argument(
        "--queue-snapshot-mode",
        choices=("all", "key-events"),
        help="Use 'key-events' to write queue snapshots only for nearest pre/execution/post events, reducing dashboard memory.",
    )
    parser.add_argument("--parameter-grid-root", type=Path, default=None, help="Optional root containing kappa/lambda sensitivity runs")
    parser.add_argument("--actor-session-alerts", type=Path, default=None, help="Optional actor/anchor-session alert parquet")
    parser.add_argument(
        "--client-session-alerts",
        type=Path,
        default=None,
        help="Deprecated legacy client-session alert parquet",
    )
    parser.add_argument("--execution-cluster-members", type=Path, default=None, help="Optional raw child-fill provenance parquet")
    parser.add_argument("--execution-cancel-candidates", type=Path, default=None, help="Optional assigned and competing cancellation links parquet")
    parser.set_defaults(**config_defaults)
    reject_parameter_overrides(
        parser,
        argv,
        parameter_options=_CONFIG_PARAMETER_OPTIONS,
    )
    return parser.parse_args(argv)


def _split_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    return {part for part in str(value).split(";") if part}


def _priority_key(value: Any, fallback: int) -> tuple[int, float | str, int]:
    if value is None:
        return (1, 0.0, fallback)
    text = str(value)
    try:
        return (0, float(text), fallback)
    except ValueError:
        return (0, text, fallback)


def _visible_qty(order: ActiveOrder) -> float:
    if order.leaves_qty <= 0 or order.displayed_qty <= 0:
        return 0.0
    return float(order.displayed_qty)


def _market_levels(active_orders: dict[str, ActiveOrder], *, side: str, top_n: int) -> list[float]:
    by_price: dict[float, float] = defaultdict(float)
    for order in active_orders.values():
        qty = _visible_qty(order)
        if qty > 0 and order.side == side:
            by_price[float(order.price)] += qty
    return sorted(by_price, reverse=(side == "bid"))[:top_n]


def _same_side_book_position(
    active_orders: dict[str, ActiveOrder], *, side: Any, price: Any
) -> tuple[int | None, bool]:
    if side not in {"bid", "ask"}:
        return None, False
    try:
        event_price = float(price)
    except (TypeError, ValueError):
        return None, False
    if not math.isfinite(event_price):
        return None, False

    visible_prices: set[float] = set()
    for order in active_orders.values():
        if order.side != side or _visible_qty(order) <= 0:
            continue
        try:
            order_price = float(order.price)
        except (TypeError, ValueError):
            continue
        if math.isfinite(order_price):
            visible_prices.add(order_price)

    if side == "bid":
        better_price_count = sum(1 for order_price in visible_prices if order_price > event_price)
    else:
        better_price_count = sum(1 for order_price in visible_prices if order_price < event_price)
    return better_price_count + 1, event_price in visible_prices


def _same_side_book_level(active_orders: dict[str, ActiveOrder], *, side: Any, price: Any) -> int | None:
    return _same_side_book_position(active_orders, side=side, price=price)[0]


def _actor_queue_dict(level_orders: list[ActiveOrder]) -> str:
    total = sum(_visible_qty(order) for order in level_orders)
    by_actor: dict[str, dict[str, Any]] = {}
    for position, order in enumerate(level_orders, start=1):
        identity = actor_identity_from_order(order)
        actor_key = identity.actor_key if identity is not None else "<missing_actor>"
        qty = _visible_qty(order)
        item = by_actor.setdefault(
            actor_key,
            {
                "perc_vol": 0.0,
                "priority": position,
                "visible_qty": 0.0,
                "order_count": 0,
            },
        )
        item["visible_qty"] += qty
        item["order_count"] += 1
        item["priority"] = min(int(item["priority"]), position)
    if total > 0:
        for item in by_actor.values():
            item["perc_vol"] = item["visible_qty"] / total
    return json.dumps(by_actor, sort_keys=True)


def _review_actor_fields(review: dict[str, Any]) -> tuple[str | None, str | None, str | None]:
    actor_key = review.get("actor_key")
    actor_id = review.get("actor_id")
    identity_level = review.get("identity_level")
    if actor_key is not None:
        return (
            str(actor_key),
            str(actor_id) if actor_id is not None else None,
            str(identity_level) if identity_level is not None else None,
        )
    legacy_client_id = review.get("client_id")
    if legacy_client_id is None:
        return None, None, None
    return f"client_original:{legacy_client_id}", str(legacy_client_id), "client_original"


def _queue_rows_for_snapshot(
    *,
    review_event_id: str,
    review_client_id: str | None = None,
    review_actor_key: str | None = None,
    execution_sort_index: int,
    execution_ts: datetime | None,
    snapshot_event: dict[str, Any],
    snapshot_ts: datetime | None,
    snapshot_phase: str,
    active_orders: dict[str, ActiveOrder],
    candidate_order_ids: set[str],
    matched_order_ids: set[str],
    top_n: int,
) -> list[dict[str, Any]]:
    if review_actor_key is None and review_client_id is not None:
        review_actor_key = f"client_original:{review_client_id}"
    review_identity_level = review_actor_key.partition(":")[0] if review_actor_key else None
    review_actor_id = review_actor_key.partition(":")[2] if review_actor_key else None
    rows: list[dict[str, Any]] = []
    for side in ("bid", "ask"):
        prices = _market_levels(active_orders, side=side, top_n=top_n)
        for level, price in enumerate(prices, start=1):
            level_orders = [
                order
                for order in active_orders.values()
                if order.side == side and float(order.price) == float(price) and _visible_qty(order) > 0
            ]
            level_orders.sort(key=lambda order: _priority_key(order.order_priority, order.first_seen_sort_index))
            level_qty = sum(_visible_qty(order) for order in level_orders)
            actor_dict = _actor_queue_dict(level_orders)
            for queue_position, order in enumerate(level_orders, start=1):
                qty = _visible_qty(order)
                order_identity = actor_identity_from_order(order)
                is_review_actor = order_identity is not None and order_identity.actor_key == review_actor_key
                rows.append(
                    {
                        "review_event_id": review_event_id,
                        "execution_sort_index": execution_sort_index,
                        "execution_ts": execution_ts,
                        "review_actor_key": review_actor_key,
                        "snapshot_sort_index": snapshot_event.get("sort_index"),
                        "snapshot_ts": snapshot_ts,
                        "snapshot_phase": snapshot_phase,
                        "snapshot_event_class": snapshot_event.get("event_class"),
                        "snapshot_ORDERID": snapshot_event.get("ORDERID"),
                        "side": side,
                        "price": price,
                        "level": level,
                        "level_visible_qty": level_qty,
                        "queue_position": queue_position,
                        "ORDERID": order.order_id,
                        "ORDERPRIORITY": order.order_priority,
                        "order_client_original_id": order.client_original_id,
                        "order_firm_id": order.firm_id,
                        "displayed_qty": order.displayed_qty,
                        "leaves_qty": order.leaves_qty,
                        "visible_qty": qty,
                        "perc_level_volume": qty / level_qty if level_qty > 0 else None,
                        "actor_queue_dict": actor_dict,
                        "is_review_actor": is_review_actor,
                        "is_review_client": (
                            is_review_actor
                            and review_identity_level == "client_original"
                            and order.client_original_id == review_actor_id
                        ),
                        "is_candidate_deceptive_order": order.order_id in candidate_order_ids,
                        "is_matched_deceptive_cancel_order": order.order_id in matched_order_ids,
                    }
                )
    return rows


def _trading_capacity(event: dict[str, Any]) -> tuple[int | None, str | None]:
    code = event.get("order_trading_capacity_code")
    observed_label = event.get("order_trading_capacity_label")
    label = TRADING_CAPACITY_BY_CODE.get(code, "") if code is not None else ""
    if not label and observed_label is not None:
        label = str(observed_label).strip()
        if code is not None and ":" in label:
            prefix, remainder = label.split(":", maxsplit=1)
            if prefix.strip() == str(code):
                label = remainder.strip()
    label = label.replace("_", " ").strip()
    label = label[:1].upper() + label[1:]
    return code, label or None


def _event_log_row(
    review_event: dict[str, Any],
    event: dict[str, Any],
    ts: datetime | None,
    *,
    book_level: int | None,
    book_level_is_resting: bool,
) -> dict[str, Any]:
    trading_capacity_code, trading_capacity_label = _trading_capacity(event)
    review_actor_key, review_actor_id, review_identity_level = _review_actor_fields(review_event)
    event_identity = actor_identity_from_event(event)
    is_review_actor = event_identity is not None and event_identity.actor_key == review_actor_key
    return {
        "review_event_id": review_event["review_event_id"],
        "execution_sort_index": review_event["sort_index"],
        "sort_index": event.get("sort_index"),
        "event_ts": ts,
        "event_class": event.get("event_class"),
        "ORDERID": event.get("ORDERID"),
        "ORDERPRIORITY": event.get("ORDERPRIORITY"),
        "side": event.get("side_label"),
        "price": event.get("ORDERPX"),
        "book_level": book_level,
        "book_level_is_resting": book_level_is_resting,
        "displayed_qty": event.get("DISPLAYEDQTY"),
        "leaves_qty": event.get("LEAVESQTY"),
        "last_shares": event.get("LASTSHARES"),
        "event_client_original_id": event.get("client_original_id"),
        "event_firm_id": event.get("firm_id"),
        "actor_key": event_identity.actor_key if event_identity is not None else None,
        "actor_id": event_identity.actor_id if event_identity is not None else None,
        "identity_level": event_identity.identity_level if event_identity is not None else None,
        "identity_source": event_identity.identity_source if event_identity is not None else None,
        "identity_fallback_flag": (
            event_identity.identity_fallback_flag if event_identity is not None else None
        ),
        "execution_anchor_mode": str(review_event.get("execution_anchor_mode") or "passive"),
        "trading_capacity_code": trading_capacity_code,
        "trading_capacity_label": trading_capacity_label,
        "is_review_actor": is_review_actor,
        "is_review_client": (
            is_review_actor
            and review_identity_level == "client_original"
            and event.get("client_original_id") == review_actor_id
        ),
        "is_execution_order": (
            int(event.get("sort_index", -1)) in review_event.get("child_fill_sort_indexes", {int(review_event["sort_index"])})
        ),
        "is_candidate_deceptive_order": str(event.get("ORDERID")) in review_event["candidate_order_ids"],
        "is_matched_deceptive_cancel_order": str(event.get("ORDERID")) in review_event["matched_order_ids"],
    }


def _prepare_review_events(
    execution_metrics: pl.DataFrame,
    max_events: int | None,
    *,
    cluster_members: pl.DataFrame | None = None,
) -> list[dict[str, Any]]:
    if execution_metrics.is_empty():
        return []
    required = {"has_matched_deceptive_cancel_window", "execution_anchor_mode"}
    missing = sorted(required - set(execution_metrics.columns))
    if missing:
        raise ValueError(f"execution metrics missing required columns: {missing}")
    anchors = set(
        execution_metrics.get_column("execution_anchor_mode")
        .drop_nulls()
        .cast(pl.String)
        .to_list()
    )
    if execution_metrics.get_column("execution_anchor_mode").null_count() or not anchors <= {
        "passive",
        "aggressive",
    }:
        raise ValueError("execution_anchor_mode must be passive or aggressive for every row")
    matched = execution_metrics.filter(pl.col("has_matched_deceptive_cancel_window"))
    sort_key = "cluster_first_sort_index" if "execution_cluster_id" in matched.columns and "cluster_first_sort_index" in matched.columns else "sort_index"
    ranking_column = next(
        (
            column
            for column in (
                "withdrawal_profile_scale_event",
                "WMSCI_event",
            )
            if column in matched.columns
        ),
        None,
    )
    if ranking_column is None:
        raise ValueError(
            "execution metrics must include withdrawal_profile_scale_event or WMSCI_event "
            "for WMSCI-ranked review"
        )
    sort_columns = [ranking_column, sort_key]
    if sort_columns:
        matched = matched.sort(
            sort_columns,
            descending=[column != sort_key for column in sort_columns],
            nulls_last=True,
        )
    if max_events is not None:
        matched = matched.head(max_events)
    out = []
    for row in matched.iter_rows(named=True):
        actor_key, actor_id, identity_level = _review_actor_fields(row)
        candidate_ids = _split_ids(row.get("candidate_deceptive_order_ids_pre"))
        matched_ids = _split_ids(row.get("matched_deceptive_cancel_order_ids_window"))
        sort_index = row.get("cluster_first_sort_index", row.get("sort_index"))
        if sort_index is None:
            raise ValueError("matched execution cluster is missing cluster_first_sort_index")
        cluster_id = row.get("execution_cluster_id")
        review_id = str(cluster_id) if cluster_id is not None else f"S{int(sort_index)}"
        child_indexes = {
            int(value)
            for value in str(row.get("child_fill_sort_indices") or "").split(";")
            if value.isdigit()
        }
        if cluster_id is not None and cluster_members is not None and not cluster_members.is_empty():
            child_indexes = set(
                cluster_members.filter(
                    pl.col("execution_cluster_id").cast(pl.String) == str(cluster_id)
                )["child_sort_index"].cast(pl.Int64).to_list()
            )
        if cluster_id is not None and not child_indexes:
            raise ValueError(f"execution cluster {cluster_id} has no child-member provenance")
        if cluster_id is None:
            child_indexes = {int(sort_index)}
        out.append(
            {
                **row,
                "actor_key": actor_key,
                "actor_id": actor_id,
                "identity_level": identity_level,
                "identity_source": row.get("identity_source") or (
                    "legacy_client_id" if identity_level == "client_original" else None
                ),
                "identity_fallback_flag": bool(row.get("identity_fallback_flag", identity_level == "firm")),
                "execution_anchor_mode": row["execution_anchor_mode"],
                "sort_index": int(sort_index),
                "review_event_id": review_id,
                "event_ts": row.get("cluster_start_ts", row.get("event_ts")),
                "event_ts_parsed": _parse_ts(row.get("cluster_start_ts", row.get("event_ts"))),
                "candidate_order_ids": candidate_ids,
                "matched_order_ids": matched_ids,
                "child_fill_sort_indexes": child_indexes,
            }
        )
    return out


def _review_population_summary(
    execution_metrics: pl.DataFrame,
    visible_review_events: pl.DataFrame,
) -> dict[str, int]:
    required = {
        "has_matched_deceptive_cancel_window",
        "spoofing_compatible_sequence",
    }
    missing = sorted(required - set(execution_metrics.columns))
    if missing:
        raise ValueError(f"execution metrics missing review-status columns: {missing}")
    visible_missing = sorted(required - set(visible_review_events.columns))
    if visible_missing:
        raise ValueError(f"visible review events missing review-status columns: {visible_missing}")

    return {
        "reconstructed_clusters": execution_metrics.height,
        "review_candidates": int(
            execution_metrics.get_column("has_matched_deceptive_cancel_window").fill_null(False).sum()
        ),
        "compatible_sequences": int(
            execution_metrics.get_column("spoofing_compatible_sequence").fill_null(False).sum()
        ),
        "displayed_candidates": visible_review_events.height,
        "displayed_compatible_sequences": int(
            visible_review_events.get_column("spoofing_compatible_sequence").fill_null(False).sum()
        ),
    }


def reconstruct_review_windows(
    raw_events: pl.DataFrame,
    review_events: list[dict[str, Any]],
    *,
    top_n: int,
    pre_window_seconds: float,
    post_window_seconds: float,
    queue_snapshot_mode: str = "all",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    config = LOBConfig(top_n=max(top_n, 1), snapshot_mode="none")
    sorted_events = sort_events(raw_events)
    events = [normalize_event(raw_row, sort_index=idx, config=config) for idx, raw_row in enumerate(sorted_events.iter_rows(named=True), start=1)]

    by_sort_index = {int(event["sort_index"]): event for event in review_events}
    sorted_review = sorted(review_events, key=lambda row: int(row["sort_index"]))
    windows = []
    for row in sorted_review:
        if row["event_ts_parsed"] is None:
            continue
        windows.append(
            {
                "review": row,
                "start": row["event_ts_parsed"] - timedelta(seconds=pre_window_seconds),
                "end": row["event_ts_parsed"] + timedelta(seconds=post_window_seconds),
                "queue_sort_indexes": set(),
            }
        )

    if queue_snapshot_mode not in {"all", "key-events"}:
        raise ValueError(f"unknown queue snapshot mode: {queue_snapshot_mode}")
    event_times = [(int(event["sort_index"]), choose_event_timestamp(event)) for event in events]
    post_cancel_sort_index_by_review_id: dict[str, int | None] = {}
    for window in windows:
        review = window["review"]
        review_sort_index = int(review["sort_index"])
        cluster_last_sort_index = int(review.get("cluster_last_sort_index", review_sort_index))
        matched_order_ids = review["matched_order_ids"]
        matched_cancels = [
            int(event["sort_index"])
            for event in events
            if int(event["sort_index"]) > cluster_last_sort_index
            and event["event_class"] == "cancel"
            and str(event.get("ORDERID")) in matched_order_ids
            and (event_ts := choose_event_timestamp(event)) is not None
            and event_ts <= window["end"]
        ]
        post_cancel_sort_index = max(matched_cancels) if matched_cancels else None
        post_cancel_sort_index_by_review_id[str(review["review_event_id"])] = post_cancel_sort_index
        if queue_snapshot_mode == "key-events":
            pre = [
                idx
                for idx, ts in event_times
                if ts is not None and idx < review_sort_index and window["start"] <= ts
            ]
            selected = {review_sort_index}
            if pre:
                selected.add(max(pre))
            if post_cancel_sort_index is not None:
                selected.add(post_cancel_sort_index)
            window["queue_sort_indexes"] = selected

    active_orders: dict[str, ActiveOrder] = {}
    pending_aggressive_residuals: dict[str, tuple[dict[str, Any], tuple[Any, ...] | None]] = {}
    non_resting_order_ids: set[str] = set()
    current_partition_id: str | None = None
    queue_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    execution_sort_indices = {
        child_index
        for window in windows
        for child_index in window["review"].get("child_fill_sort_indexes", {int(window["review"]["sort_index"])})
    }
    review_summary_rows: list[dict[str, Any]] = []

    for event_index, event in enumerate(events):
        partition_id = _partition_id(event)
        if current_partition_id is None:
            current_partition_id = partition_id
        elif partition_id != current_partition_id:
            _flush_pending_aggressive_residuals(active_orders, pending_aggressive_residuals, keep_group=None)
            active_orders = {}
            pending_aggressive_residuals = {}
            non_resting_order_ids = set()
            current_partition_id = partition_id

        event_ts = choose_event_timestamp(event)
        pre_event_active_orders = (
            dict(active_orders) if int(event["sort_index"]) in execution_sort_indices else None
        )
        _apply_event(
            active_orders,
            event,
            pending_aggressive_residuals=pending_aggressive_residuals,
            non_resting_order_ids=non_resting_order_ids,
        )
        next_event = events[event_index + 1] if event_index + 1 < len(events) else None
        next_group = _fill_group_key(next_event) if next_event is not None else None
        _flush_pending_aggressive_residuals(active_orders, pending_aggressive_residuals, keep_group=next_group)

        if event_ts is not None:
            for window in windows:
                review = window["review"]
                if window["start"] <= event_ts <= window["end"]:
                    book_level, book_level_is_resting = _same_side_book_position(
                        active_orders,
                        side=event.get("side_label"),
                        price=event.get("ORDERPX"),
                    )
                    event_rows.append(
                        _event_log_row(
                            review,
                            event,
                            event_ts,
                            book_level=book_level,
                            book_level_is_resting=book_level_is_resting,
                        )
                    )
                    child_indexes = review.get("child_fill_sort_indexes", {int(review["sort_index"])})
                    if int(event["sort_index"]) in child_indexes:
                        phase = "execution"
                    elif int(event["sort_index"]) < min(child_indexes):
                        phase = "pre"
                    else:
                        phase = "post"
                    if queue_snapshot_mode == "all" or int(event["sort_index"]) in window["queue_sort_indexes"]:
                        snapshot_active_orders = (
                            pre_event_active_orders
                            if phase == "execution" and pre_event_active_orders is not None
                            else active_orders
                        )
                        queue_rows.extend(
                            _queue_rows_for_snapshot(
                                review_event_id=review["review_event_id"],
                                review_client_id=review.get("client_id"),
                                review_actor_key=_review_actor_fields(review)[0],
                                execution_sort_index=int(review["sort_index"]),
                                execution_ts=review["event_ts_parsed"],
                                snapshot_event=event,
                                snapshot_ts=event_ts,
                                snapshot_phase=phase,
                                active_orders=snapshot_active_orders,
                                candidate_order_ids=review["candidate_order_ids"],
                                matched_order_ids=review["matched_order_ids"],
                                top_n=top_n,
                            )
                        )

        if int(event["sort_index"]) in by_sort_index:
            review = by_sort_index[int(event["sort_index"])]
            trading_capacity_code, trading_capacity_label = _trading_capacity(event)
            review_summary_rows.append(
                {
                    "review_event_id": review["review_event_id"],
                    "execution_cluster_id": review.get("execution_cluster_id"),
                    "cluster_first_sort_index": review.get("cluster_first_sort_index", review["sort_index"]),
                    "cluster_last_sort_index": review.get("cluster_last_sort_index", review["sort_index"]),
                    "cluster_start_ts": review.get("cluster_start_ts", review.get("event_ts")),
                    "cluster_end_ts": review.get("cluster_end_ts", review.get("event_ts")),
                    "child_fill_count": review.get("child_fill_count", 1),
                    "event_ts": review.get("event_ts"),
                    "actor_key": _review_actor_fields(review)[0],
                    "actor_id": _review_actor_fields(review)[1],
                    "identity_level": _review_actor_fields(review)[2],
                    "identity_source": review.get("identity_source"),
                    "identity_fallback_flag": review.get("identity_fallback_flag"),
                    "execution_anchor_mode": review.get("execution_anchor_mode", "passive"),
                    "event_client_original_id": review.get(
                        "event_client_original_id", review.get("client_id")
                    ),
                    "event_firm_id": review.get("event_firm_id", review.get("firm_id")),
                    "trading_capacity_code": trading_capacity_code,
                    "trading_capacity_label": trading_capacity_label,
                    "execution_side": review.get("execution_side"),
                    "deceptive_side": review.get("deceptive_side"),
                    "execution_quantity": review.get("execution_quantity", review.get("fill_qty")),
                    "resting_executed_quantity": review.get("resting_executed_quantity"),
                    "aggressive_executed_quantity": review.get("aggressive_executed_quantity"),
                    "fill_qty": review.get("fill_qty"),
                    "MSCI_resting_profile": review.get(
                        "MSCI_resting_profile", review.get("MSCI")
                    ),
                    "MSCI": review.get("MSCI"),
                    "SCI": review.get("SCI"),
                    "collapse_opposite_side": review.get("collapse_opposite_side"),
                    "collapse_same_side": review.get("collapse_same_side"),
                    "candidate_deceptive_visible_qty_pre": review.get("candidate_deceptive_visible_qty_pre"),
                    "matched_deceptive_cancel_visible_qty_window": review.get("matched_deceptive_cancel_visible_qty_window"),
                    "matched_deceptive_cancel_fraction_window": review.get("matched_deceptive_cancel_fraction_window"),
                    "matched_deceptive_cancel_min_delay_seconds": review.get("matched_deceptive_cancel_min_delay_seconds"),
                    "matched_deceptive_cancel_max_delay_seconds": review.get("matched_deceptive_cancel_max_delay_seconds"),
                    "has_matched_deceptive_cancel_window": review.get(
                        "has_matched_deceptive_cancel_window"
                    ),
                    "gate_rapid_matched_withdrawal": review.get(
                        "gate_rapid_matched_withdrawal"
                    ),
                    "gate_small_fill_relative_to_withdrawal": review.get(
                        "gate_small_fill_relative_to_withdrawal"
                    ),
                    "gate_favorable_pre_fill_move": review.get(
                        "gate_favorable_pre_fill_move"
                    ),
                    "gate_cancel_anchored_reversion": review.get(
                        "gate_cancel_anchored_reversion"
                    ),
                    "spoofing_compatible_sequence": review.get(
                        "spoofing_compatible_sequence"
                    ),
                    "post_cancel_sort_index": post_cancel_sort_index_by_review_id.get(
                        str(review["review_event_id"])
                    ),
                    "weighted_net_withdrawal_qty_window": review.get("weighted_net_withdrawal_qty_window"),
                    "withdrawal_to_fill_ratio": review.get("withdrawal_to_fill_ratio"),
                    "weighted_withdrawal_to_fill_ratio": review.get("weighted_withdrawal_to_fill_ratio"),
                    "withdrawal_profile_scale_event": review.get(
                        "withdrawal_profile_scale_event", review.get("WMSCI_event")
                    ),
                    "WMSCI_passive": review.get("WMSCI_passive"),
                    "WMSCI_aggressive": review.get("WMSCI_aggressive"),
                    "WMSCI_event": review.get("WMSCI_event"),
                    "favorable_mid_move_pre_fill": review.get("favorable_mid_move_pre_fill"),
                    "post_cancel_mid_reversion": review.get("post_cancel_mid_reversion"),
                    "execution_price_advantage_vs_posture_mid": review.get("execution_price_advantage_vs_posture_mid"),
                    "candidate_deceptive_order_ids_pre": review.get("candidate_deceptive_order_ids_pre"),
                    "matched_deceptive_cancel_order_ids_window": review.get("matched_deceptive_cancel_order_ids_window"),
                }
            )

    return (
        pl.DataFrame(review_summary_rows, infer_schema_length=None) if review_summary_rows else pl.DataFrame(),
        pl.DataFrame(event_rows, infer_schema_length=None) if event_rows else pl.DataFrame(),
        pl.DataFrame(queue_rows, infer_schema_length=None) if queue_rows else pl.DataFrame(),
    )


def _json_for_script(value: Any) -> str:
    return (
        json.dumps(value, default=str)
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def _json_records(df: pl.DataFrame) -> str:
    return _json_for_script(df.to_dicts())


def _json_records_with_string_columns(df: pl.DataFrame, columns: tuple[str, ...]) -> str:
    string_columns = [column for column in columns if column in df.columns]
    if not string_columns:
        return _json_records(df)
    return _json_records(df.with_columns(pl.col(column).cast(pl.String) for column in string_columns))


def _load_parameter_review_events(root: Path, max_events: int | None) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for metadata_path in sorted(root.glob("kappa_*_lambda_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        execution_path = metadata_path.parent / "execution_metrics.parquet"
        if not execution_path.exists():
            continue
        _validate_metric_metadata(metadata, source=metadata_path)
        members_path = metadata_path.parent / "execution_cluster_members.parquet"
        members = pl.read_parquet(members_path) if members_path.exists() else None
        events = _prepare_review_events(
            pl.read_parquet(execution_path),
            max_events,
            cluster_members=members,
        )
        for event in events:
            event.pop("event_ts_parsed", None)
            event.pop("candidate_order_ids", None)
            event.pop("matched_order_ids", None)
        runs.append(
            {
                "kappa": metadata.get("kappa"),
                "lambda": metadata.get("lambda_"),
                "label": f"kappa={metadata.get('kappa')}, lambda={metadata.get('lambda_')}",
                "events": events,
            }
        )
    return runs


def _format_number(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _format_seconds(value: Any) -> str:
    if value is None:
        return "NA"
    return f"{_format_number(value)} seconds"


def _format_gamma_grid(value: Any) -> str:
    if isinstance(value, list):
        return ", ".join(_format_number(item) for item in value)
    if value is None:
        return "NA"
    return str(value)


def _parameter_table_html(
    *,
    review_top_n: int,
    pre_window_seconds: float,
    post_window_seconds: float,
    metric_metadata: dict[str, Any],
) -> str:
    kernel_mode = str(metric_metadata.get("kernel_mode") or "unspecified")
    kernel_rows = [
        (
            "Depth-kernel mode",
            kernel_mode,
            "Operational metrics require instrument- and side-specific empirical rank weights.",
        ),
        (
            "Empirical-kernel artifact",
            str(metric_metadata.get("empirical_depth_kernel") or "NA"),
            "Calibration artifact supplying the operational rank weights for this metric run.",
        ),
    ]
    rows = [
        (
            "LOB review top-N depth",
            _format_number(review_top_n),
            "Number of bid and ask price levels reconstructed in the manual event-review chart.",
        ),
        (
            "Metric-run top-N depth",
            _format_number(metric_metadata.get("top_n")),
            "Depth used by the MSCI/MCPS run that produced the selected candidate events.",
        ),
        (
            "Candidate-order age window",
            _format_seconds(metric_metadata.get("max_deceptive_order_age_seconds")),
            "Maximum time between first seeing the opposite-side candidate order and the small execution.",
        ),
        (
            "Local review window",
            f"{_format_seconds(pre_window_seconds)} before, {_format_seconds(post_window_seconds)} after",
            "Event-log and queue-reconstruction interval shown around each selected execution.",
        ),
        (
            "Post-execution matched-withdrawal window",
            _format_seconds(metric_metadata.get("withdrawal_window_seconds")),
            "Window after the execution cluster in which an attributed candidate-order cancellation is matched.",
        ),
        *kernel_rows,
        (
            "Zero-denominator policy",
            str(metric_metadata.get("ratio_zero_denominator_policy") or "NA"),
            "Exact piecewise ratios: zero only where absence of the phenomenon is defined; undefined ratios remain null.",
        ),
        (
            "MCPS gamma grid",
            _format_gamma_grid(metric_metadata.get("gamma_grid")),
            "Thresholds used to aggregate repeated high-MSCI executions into actor/anchor-stratified MCPS scores.",
        ),
    ]
    row_html = "\n".join(
        "      <tr><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            escape(str(name), quote=True),
            escape(str(value), quote=True),
            escape(str(meaning), quote=True),
        )
        for name, value, meaning in rows
    )
    return f"""  <table class=\"parameter-table\">
    <thead><tr><th>Parameter</th><th>Value</th><th>Meaning</th></tr></thead>
    <tbody>
{row_html}
    </tbody>
  </table>"""


def _validate_metric_metadata(metadata: dict[str, Any], *, source: Path) -> None:
    if metadata.get("msci_definition") != MSCI_DEFINITION:
        raise ValueError(
            f"incompatible MSCI definition in {source}: expected {MSCI_DEFINITION!r}, "
            f"found {metadata.get('msci_definition')!r}"
        )
    if metadata.get("msci_range") != list(MSCI_RANGE):
        raise ValueError(
            f"incompatible MSCI range in {source}: expected {list(MSCI_RANGE)!r}, "
            f"found {metadata.get('msci_range')!r}"
        )
    if metadata.get("ratio_zero_denominator_policy") != RATIO_ZERO_DENOMINATOR_POLICY:
        raise ValueError(
            f"incompatible ratio zero-denominator policy in {source}: "
            f"expected {RATIO_ZERO_DENOMINATOR_POLICY!r}, "
            f"found {metadata.get('ratio_zero_denominator_policy')!r}"
        )


def _load_metric_metadata(execution_metrics_path: Path) -> dict[str, Any]:
    metadata_path = execution_metrics_path.parent / "metadata.json"
    if not metadata_path.exists():
        raise ValueError(f"metric metadata missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text())
    _validate_metric_metadata(metadata, source=metadata_path)
    return metadata


def _metric_run_parameters(metric_metadata: dict[str, Any]) -> dict[str, Any]:
    return {
        "top_n": metric_metadata.get("top_n"),
        "window_seconds": metric_metadata.get("window_seconds"),
        "max_deceptive_order_age_seconds": metric_metadata.get("max_deceptive_order_age_seconds"),
        "kappa": metric_metadata.get("kappa"),
        "lambda_": metric_metadata.get("lambda_"),
        "msci_definition": metric_metadata["msci_definition"],
        "msci_range": metric_metadata["msci_range"],
        "ratio_zero_denominator_policy": metric_metadata["ratio_zero_denominator_policy"],
        "execution_anchor_modes": metric_metadata.get("execution_anchor_modes", []),
        "observed_execution_anchor_modes": metric_metadata.get(
            "observed_execution_anchor_modes", []
        ),
        "actor_identity_mode": metric_metadata.get("actor_identity_mode"),
        "actor_identity_schema": metric_metadata.get("actor_identity_schema"),
        "execution_anchor_schema": metric_metadata.get("execution_anchor_schema"),
        "gamma_grid": metric_metadata.get("gamma_grid"),
        "normalization": metric_metadata.get("normalization"),
    }


def _load_optional_parquet(path: Path | None) -> pl.DataFrame:
    if path is None or not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def write_review_artifacts(*, output_dir: Path, event_log: pl.DataFrame, queue: pl.DataFrame) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = output_dir / "matched_spoofing_event_log.parquet"
    queue_path = output_dir / "matched_spoofing_lob_queue.parquet"
    event_log.write_parquet(event_log_path)
    queue.write_parquet(queue_path)
    return {"event_log": event_log_path, "queue": queue_path}


def write_dashboard(
    path: Path,
    *,
    review_events: pl.DataFrame,
    event_log: pl.DataFrame,
    queue: pl.DataFrame,
    review_top_n: int = 10,
    pre_window_seconds: float = 30.0,
    post_window_seconds: float = 5.0,
    metric_metadata: dict[str, Any] | None = None,
    parameter_review_events: list[dict[str, Any]] | None = None,
    actor_session_alerts: pl.DataFrame | None = None,
    client_session_alerts: pl.DataFrame | None = None,
    child_members: pl.DataFrame | None = None,
    cancel_candidates: pl.DataFrame | None = None,
    dashboard_refreshed_at_utc: str | None = None,
    population_summary: dict[str, int | None] | None = None,
) -> None:
    if actor_session_alerts is not None and client_session_alerts is not None:
        raise ValueError("provide actor_session_alerts or legacy client_session_alerts, not both")
    session_alerts = actor_session_alerts if actor_session_alerts is not None else client_session_alerts
    path.parent.mkdir(parents=True, exist_ok=True)
    parameter_table = _parameter_table_html(
        review_top_n=review_top_n,
        pre_window_seconds=pre_window_seconds,
        post_window_seconds=post_window_seconds,
        metric_metadata=metric_metadata or {},
    )
    if population_summary is None:
        population_summary = {
            "reconstructed_clusters": None,
            "review_candidates": None,
            "compatible_sequences": None,
            "displayed_candidates": review_events.height,
            "displayed_compatible_sequences": int(
                review_events.get_column("spoofing_compatible_sequence").fill_null(False).sum()
            )
            if "spoofing_compatible_sequence" in review_events.columns
            else None,
        }
    html = f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>Spoofing event review dashboard</title>
<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
<style>
body {{ font-family: Inter, Arial, sans-serif; margin: 0; background: #f7f8fb; color: #18202f; }}
.page {{ max-width: 1420px; margin: 24px auto; padding: 0 24px; }}
h1 {{ margin-bottom: 0.2rem; text-align: center; }}
.note {{ color: #4d5a6d; max-width: 1100px; margin-left: auto; margin-right: auto; }}
.intro {{ color: #364152; max-width: 1180px; line-height: 1.45; }}
.intro ul {{ margin: 0.5rem 0 0.2rem 1.2rem; padding: 0; }}
.intro li {{ margin: 0.25rem 0; }}
.metric-guide {{ color: #273449; line-height: 1.45; }}
.metric-guide h2 {{ margin-bottom: 0.4rem; }}
.metric-guide-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
.metric-guide-item {{ border: 1px solid #dbe3ef; border-radius: 8px; padding: 12px; background: #f8fafc; }}
.metric-guide-item h3 {{ margin: 0 0 6px; font-size: 1.02rem; }}
.metric-example {{ margin: 12px 0 0; padding: 11px 12px; border-left: 5px solid #7c3aed; background: #f5f3ff; border-radius: 6px; }}
.metric-caution {{ margin: 8px 0 0; color: #59667a; font-size: 0.9rem; }}
.parameter-table {{ margin-top: 0.7rem; width: 100%; max-width: 100%; table-layout: fixed; }}
.parameter-table th {{ position: static; }}
.parameter-table td {{ overflow-wrap: anywhere; vertical-align: top; }}
.parameter-table td:first-child {{ width: 24%; font-weight: 600; color: #273449; }}
.parameter-table td:nth-child(2) {{ width: 36%; }}
.parameter-table td:nth-child(3) {{ width: 40%; }}
.controls {{ display: grid; grid-template-columns: minmax(210px, .8fr) minmax(230px, .9fr) minmax(360px, 2fr); gap: 14px; margin: 18px auto; align-items: start; }}
.control-group {{ display: flex; min-width: 0; flex-direction: column; gap: 6px; }}
.control-group label {{ line-height: 1.25; }}
.control-group select {{ box-sizing: border-box; width: 100%; min-width: 0; padding: 8px; }}
.card {{ background: white; border: 1px solid #dfe5ef; border-radius: 10px; padding: 14px; margin: 14px auto; box-shadow: 0 1px 2px rgba(20,30,50,0.05); }}
#summary {{ font-size: 0.95rem; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
th, td {{ border-bottom: 1px solid #e8edf5; padding: 6px; text-align: left; }}
th {{ background: #f1f4f9; position: sticky; top: 0; }}
.badge {{ display: inline-block; padding: 2px 6px; border-radius: 6px; background: #fee2e2; color: #991b1b; font-weight: 600; }}
.event-row-actor {{ background: #fff7ed; }}
.event-row-execution {{ background: #dcfce7; font-weight: 600; }}
.event-row-candidate {{ box-shadow: inset 4px 0 0 #f97316; }}
.event-row-matched-cancel {{ background: #fee2e2; font-weight: 600; }}
.event-legend {{ display: flex; gap: 10px; flex-wrap: wrap; margin: 0 0 8px 0; color: #4d5a6d; font-size: 0.84rem; }}
.event-legend span {{ padding: 3px 7px; border-radius: 6px; border: 1px solid #e5e7eb; }}
.population-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 10px; margin: 12px 0; }}
.population-item {{ border: 1px solid #dbe3ef; border-radius: 8px; padding: 12px; background: #f8fafc; }}
.population-value {{ display: block; font-size: 1.55rem; line-height: 1.15; font-weight: 750; color: #172033; }}
.population-label {{ display: block; margin-top: 4px; font-weight: 650; }}
.population-help {{ display: block; margin-top: 4px; color: #59667a; font-size: 0.82rem; line-height: 1.35; }}
.evidence-levels {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 10px; margin-top: 12px; }}
.evidence-level {{ border-left: 5px solid #94a3b8; padding: 10px 12px; background: #f8fafc; border-radius: 6px; }}
.evidence-level.candidate {{ border-left-color: #d97706; background: #fffbeb; }}
.evidence-level.complete {{ border-left-color: #15803d; background: #f0fdf4; }}
.surveillance-note {{ padding: 10px 12px; border-left: 5px solid #1d4ed8; background: #eff6ff; border-radius: 6px; line-height: 1.45; }}
.selection-reason {{ padding: 10px 12px; border-left: 5px solid #d97706; background: #fffbeb; border-radius: 6px; }}
.sequence-badge {{ display: inline-block; margin-left: 6px; padding: 4px 8px; border-radius: 999px; font-weight: 700; font-size: 0.82rem; vertical-align: middle; }}
.sequence-complete {{ color: #166534; background: #dcfce7; border: 1px solid #86efac; }}
.sequence-review {{ color: #92400e; background: #fef3c7; border: 1px solid #fcd34d; }}
.sequence-unavailable {{ color: #475569; background: #f1f5f9; border: 1px solid #cbd5e1; }}
.control-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(270px, 1fr)); gap: 10px; margin: 10px 0; }}
.control-card {{ border: 1px solid #dbe3ef; border-radius: 8px; padding: 11px; background: #fff; }}
.control-card h4 {{ margin: 0 0 6px; font-size: 0.95rem; }}
.control-status {{ display: inline-block; min-width: 92px; text-align: center; padding: 3px 7px; border-radius: 999px; font-weight: 750; }}
.control-yes {{ color: #166534; background: #dcfce7; }}
.control-no {{ color: #991b1b; background: #fee2e2; }}
.control-na {{ color: #475569; background: #e2e8f0; }}
.control-detail {{ display: block; margin-top: 7px; color: #4d5a6d; font-size: 0.86rem; line-height: 1.35; }}
.key-facts {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 8px; margin: 10px 0; }}
.key-fact {{ padding: 9px; background: #f8fafc; border-radius: 6px; }}
details {{ margin-top: 12px; border-top: 1px solid #e5e7eb; padding-top: 10px; }}
details summary {{ cursor: pointer; font-weight: 650; color: #334155; }}
.filter-count {{ color: #4d5a6d; font-size: 0.9rem; }}
.empty-filter {{ padding: 18px; text-align: center; color: #64748b; }}
@media (max-width: 900px) {{
  .controls {{ grid-template-columns: 1fr; }}
}}
</style>
</head>
<body>
<div class=\"page\">
<h1>Revisione degli eventi compatibili con spoofing</h1>
<p class="note">Questa dashboard ordina i candidati, spiega perché ogni evento è stato selezionato e collega le metriche ai messaggi e alle quantità osservate nel book.</p>
<div class="card metric-guide">
  <h2>Come leggere WMSCI e MSCI</h2>
  <div class="metric-guide-grid">
    <div class="metric-guide-item">
      <h3>WMSCI — intensità del ritiro attribuito</h3>
      <p><b>WMSCI misura l’intensità del ritiro attribuito</b> allo stesso soggetto dopo l’esecuzione. Cresce quando l’ordine opposto visibile e il ritiro rapido attribuito sono grandi rispetto alla quantità eseguita e quando una quota maggiore dell’ordine candidato viene ritirata.</p>
      <p><b>Come interpretarlo:</b> a parità di configurazione, un valore più alto porta il candidato più in alto nell’ordinamento. Parte da zero e non ha un limite superiore prefissato.</p>
    </div>
    <div class="metric-guide-item">
      <h3>MSCI — cambiamento relativo della forma del book</h3>
      <p><b>MSCI descrive come cambia la forma relativa del book</b> attorno all’esecuzione; è un contrasto con segno compreso tra −1 e 2. Valori positivi indicano un contrasto complessivo verso il lato opposto, valori vicini a zero non mostrano un contrasto netto e valori negativi indicano che la riduzione sullo stesso lato prevale anche sulla componente di cambiamento della forma.</p>
      <p><b>Come interpretarlo:</b> è un contesto secondario sulla forma del book, non una versione normalizzata di WMSCI.</p>
    </div>
  </div>
  <p class="metric-example"><b>Mini esempio illustrativo.</b> Un evento con <b>WMSCI = 3,2</b> e <b>MSCI = 0,8</b> mostra un ritiro attribuito relativamente intenso e un contrasto della forma del book positivo. Un evento con <b>WMSCI = 0,4</b> e <b>MSCI = −0,3</b> ha un segnale di ritiro più debole e una riduzione che prevale sullo stesso lato. Il primo viene ordinato prima per WMSCI.</p>
  <p class="metric-caution">I due valori hanno scale diverse e vanno letti separatamente: WMSCI = 3,2 non significa “3,2 volte più sospetto”. Sono indicatori di screening, non probabilità né prove di intento.</p>
</div>
<div class=\"card intro\">
  <h2>Cosa contiene questa dashboard</h2>
  <div id="populationSummary" class="population-grid"></div>
  <p>I risultati sono organizzati in tre livelli distinti. Il passaggio a un livello successivo richiede evidenze aggiuntive:</p>
  <div class="evidence-levels">
    <div class="evidence-level"><b>1. Cluster ricostruito</b><br><span class="population-help">Una o più esecuzioni collegate ricostruite dai messaggi del mercato. Non è ancora un candidato mostrato nella dashboard.</span></div>
    <div class="evidence-level candidate"><b>2. Candidato da revisionare</b><br><span class="population-help">Dopo l'esecuzione è stato attribuito allo stesso soggetto il ritiro rapido di un ordine sul lato opposto. Tutti gli eventi selezionabili in questa dashboard appartengono almeno a questo livello.</span></div>
    <div class="evidence-level complete"><b>3. Sequenza completa</b><br><span class="population-help">Il candidato supera contemporaneamente i quattro controlli mostrati nella scheda evento.</span></div>
  </div>
  <p class="surveillance-note"><b>Interpretazione prudente.</b> Un candidato che non supera tutti i controlli non è automaticamente un falso positivo: può mancare una misura, oppure il caso può essere rilevante per altri elementi. La decisione finale richiede la lettura del book, degli ordini, delle esecuzioni e del contesto del soggetto.</p>
  <details>
    <summary>Dettagli di calcolo, parametri e provenienza</summary>
    <p><b>Trading capacity</b>: <b>1</b> = negoziazione per conto proprio (<span lang="en">Dealing on own account</span>); <b>2</b> = matched principal (<span lang="en">Matched principal</span>); <b>3</b> = altra capacità (<span lang="en">Any other capacity</span>). <b>Firm ID</b> è usato solo quando manca il cliente originale e può riunire più clienti sottostanti.</p>
    <p><b>Candidate deceptive orders</b> indica gli ordini recenti dello stesso soggetto, sul lato opposto, visibili prima dell'esecuzione. <b>Price-response diagnostics</b> comprende il favorable pre-fill mid move e la reversione successiva al ritiro. <b>DWI e SCI</b> sono componenti tecniche del profilo a riposo conservate per l'approfondimento.</p>
    <p>Le tre fasi usano l'ordine di elaborazione del matching engine (<span lang="en">Matching-engine sort order defines the three stages</span>); nella fase di esecuzione la profondità totale è quella immediatamente precedente al fill (<span lang="en">execution stage uses immediately pre-fill total depth</span>). I timestamp originali sono mostrati senza correzioni e possono non essere monotoni a precisione sub-millisecondo.</p>
    {parameter_table}
  </details>
</div>
<div class="card"><h2>Segnalazioni aggregate per soggetto e sessione <small>(Actor-session alerts)</small></h2><p>Questa sezione segnala concentrazioni ripetute nella stessa sessione. È un livello aggregato: non coincide con la valutazione dei quattro controlli del singolo evento e non implica che ogni evento del soggetto sia una sequenza completa.</p><div id="actorSessionAlerts"></div></div>
<div class=\"card\"><div id=\"overview\"></div></div>
<div class=\"controls\"><div class="control-group"><label for=\"parameterSelect\"><b>Elaborazione da consultare <span hidden>Choose metric-run variant</span></b></label><select id=\"parameterSelect\"></select></div><div class="control-group"><label for=\"evidenceFilter\"><b>Filtra per esito</b></label><select id=\"evidenceFilter\"><option value=\"all\">Tutti i candidati da revisionare</option><option value=\"complete\">Solo sequenze complete</option><option value=\"review\">Solo candidati da approfondire</option></select></div><div class="control-group"><label for=\"eventSelect\"><b>Evento da esaminare</b></label><select id=\"eventSelect\"></select><span id="filterResultCount" class="filter-count"></span></div></div>
<div class=\"card\" id=\"summary\"></div>
<div class=\"card\"><div id=\"lob\"></div></div>
<div class=\"card\"><h2>Esecuzioni elementari del cluster <small>(Raw child fills)</small></h2><p>Mostra le singole esecuzioni riunite nel cluster selezionato, così da verificare quantità, prezzo, ordine e identificativi originali.</p><div id=\"childFills\"></div></div>
<div class=\"card\"><h2>Ritiri attribuiti e possibili alternative <small>(Canonical assigned vs competing cancellations)</small></h2><p>Distingue il ritiro assegnato a questo cluster dalle cancellazioni concorrenti che avrebbero potuto essere collegate a più cluster.</p><div id=\"cancelCandidates\"></div></div>
<div class=\"card\"><h2>Sequenza completa dei messaggi nella finestra <small>(Actual events in zoom window)</small></h2><p>Consente di verificare l'ordine temporale e gli identificativi grezzi di immissioni, modifiche, esecuzioni e cancellazioni.</p><div id=\"eventTable\"></div></div>
<script>
const baseReviewEvents = {_json_records(review_events)};
const parameterRuns = {_json_for_script(parameter_review_events or [])};
const dashboardPopulation = {_json_for_script(population_summary)};
const baseReviewById = new Map(baseReviewEvents.map(event => [event.review_event_id, event]));
function withBaseReviewContext(events) {{
  return events.map(event => ({{...(baseReviewById.get(event.review_event_id) || {{}}), ...event}}));
}}
let currentRunEvents = parameterRuns.length ? withBaseReviewContext(parameterRuns[0].events) : baseReviewEvents;
let reviewEvents = currentRunEvents;
const eventLog = {_json_records(event_log)};
const queueRows = {_json_records(queue)};
const actorSessionAlerts = {_json_records(session_alerts if session_alerts is not None else pl.DataFrame())};
const childMembers = {_json_records_with_string_columns(child_members if child_members is not None else pl.DataFrame(), ("child_event_id", "child_execution_id"))};
const cancelCandidates = {_json_records(cancel_candidates if cancel_candidates is not None else pl.DataFrame())};
const dashboardRefreshedAtUtc = {_json_for_script(dashboard_refreshed_at_utc)};
const reviewTopN = {int(review_top_n)};
function byEvent(id, rows) {{ return rows.filter(r => r.review_event_id === id); }}
function byCluster(id, rows) {{ return rows.filter(r => r.execution_cluster_id === id); }}
function finiteNumber(value) {{ const number = Number(value); return Number.isFinite(number) ? number : null; }}
function formatZoomLevel(value, isResting) {{
  const number = finiteNumber(value);
  if (number === null || number < 1) return '';
  const rank = Math.trunc(number);
  const zoomLabel = rank <= reviewTopN ? `L${{rank}}` : `oltre i primi ${{reviewTopN}} livelli (posizione ${{rank}})`;
  return isResting === false ? `non presente nel book; si collocherebbe a ${{zoomLabel}}` : zoomLabel;
}}
function metricText(value, digits=6) {{ const number = finiteNumber(value); return number === null ? 'NA' : number.toFixed(digits); }}
function capacityText(row) {{
  const hasCode = row.trading_capacity_code !== null && row.trading_capacity_code !== undefined && row.trading_capacity_code !== '';
  const capacityLabel = row.trading_capacity_label ? String(row.trading_capacity_label) : '';
  if (!hasCode && !capacityLabel) return 'NA';
  if (!hasCode) return capacityLabel;
  return capacityLabel ? `${{row.trading_capacity_code}} — ${{capacityLabel}}` : String(row.trading_capacity_code);
}}
function withdrawalProfileScaleValue(ev) {{ return finiteNumber(ev.withdrawal_profile_scale_event ?? ev.WMSCI_event); }}
function msciRestingProfileValue(ev) {{ return finiteNumber(ev.MSCI_resting_profile ?? ev.MSCI); }}
function executionQuantityValue(ev) {{ return finiteNumber(ev.execution_quantity ?? ev.fill_qty); }}
function actorText(ev) {{ return ev.actor_key || (ev.client_id ? `client_original:${{ev.client_id}}` : 'unattributable'); }}
function identityLevelText(ev) {{ return ev.identity_level || (ev.client_id ? 'client_original' : 'missing'); }}
function executionAnchorText(ev) {{ return ev.execution_anchor_mode || 'missing'; }}
function identityScopeText(ev) {{
  if (ev.identity_scope_warning) return ev.identity_scope_warning;
  return ev.identity_fallback_flag || identityLevelText(ev) === 'firm'
    ? 'firm_fallback_may_aggregate_multiple_clients'
    : 'client_original_identity';
}}
function booleanValue(value) {{
  if (value === true || value === 1 || value === 'true') return true;
  if (value === false || value === 0 || value === 'false') return false;
  return null;
}}
function countText(value) {{
  if (value === null || value === undefined || !Number.isFinite(Number(value))) return 'Non disponibile';
  return Number(value).toLocaleString('it-IT');
}}
function identityPlainText(ev) {{
  const level = identityLevelText(ev);
  if (level === 'client_original') return 'cliente originale';
  if (level === 'firm') return 'firm ID (può riunire più clienti)';
  return 'identificazione non disponibile';
}}
function executionAnchorPlainText(ev) {{
  const anchor = executionAnchorText(ev);
  if (anchor === 'passive') return 'esecuzione passiva: l’ordine era già presente nel book';
  if (anchor === 'aggressive') return 'esecuzione aggressiva: l’ordine ha colpito liquidità già presente';
  return 'tipo di esecuzione non disponibile';
}}
function sidePlainText(value) {{
  if (value === 'bid') return 'acquisto (bid)';
  if (value === 'ask') return 'vendita (ask)';
  return value ?? 'Non disponibile';
}}
function actorPlainText(ev) {{
  const actor = actorText(ev);
  if (actor.startsWith('client_original:')) return `Cliente ${{actor.slice('client_original:'.length)}}`;
  if (actor.startsWith('firm:')) return `Firm ${{actor.slice('firm:'.length)}}`;
  return actor === 'unattributable' ? 'Soggetto non attribuibile' : actor;
}}
function executionAnchorShortText(ev) {{
  const anchor = executionAnchorText(ev);
  if (anchor === 'passive') return 'passiva';
  if (anchor === 'aggressive') return 'aggressiva';
  return 'non disponibile';
}}
function capacityPlainText(row) {{
  const code = row.trading_capacity_code;
  if (code === 1 || code === '1') return '1 — conto proprio';
  if (code === 2 || code === '2') return '2 — matched principal';
  if (code === 3 || code === '3') return '3 — altra capacità';
  return code === null || code === undefined ? 'Non disponibile' : String(code);
}}
function eventClassPlainText(value) {{
  const labels = {{new_order:'nuovo ordine', modify_order:'modifica ordine', cancel:'cancellazione', fill:'esecuzione'}};
  return labels[value] || value || 'Non disponibile';
}}
function recommendedActionPlainText(value) {{
  return value === 'human_review' ? 'revisione manuale' : (value || 'Non disponibile');
}}
function assignmentRulePlainText(value) {{
  const labels = {{latest_prior_cluster_end:'cluster precedente più vicino', single_prior_cluster:'unico cluster precedente'}};
  return labels[value] || String(value || 'Non disponibile').replaceAll('_', ' ');
}}
function gateStatus(explicitValue, available, fallbackValue) {{
  if (!available) return 'na';
  const explicit = booleanValue(explicitValue);
  if (explicit !== null) return explicit ? 'yes' : 'no';
  return fallbackValue ? 'yes' : 'no';
}}
function behavioralControls(ev) {{
  const executionQty = executionQuantityValue(ev);
  const withdrawnQty = finiteNumber(ev.matched_deceptive_cancel_visible_qty_window);
  const delayMin = finiteNumber(ev.matched_deceptive_cancel_min_delay_seconds);
  const delayMax = finiteNumber(ev.matched_deceptive_cancel_max_delay_seconds);
  const favorableMove = finiteNumber(ev.favorable_mid_move_pre_fill);
  const reversion = finiteNumber(ev.post_cancel_mid_reversion);
  const rapidExplicit = booleanValue(ev.gate_rapid_matched_withdrawal ?? ev.has_matched_deceptive_cancel_window);
  const rapidStatus = rapidExplicit === null ? 'na' : (rapidExplicit ? 'yes' : 'no');
  const qtyAvailable = executionQty !== null && withdrawnQty !== null;
  const moveAvailable = favorableMove !== null;
  const reversionAvailable = reversion !== null;
  return [
    {{
      label: 'Ritiro rapido attribuito allo stesso soggetto',
      status: rapidStatus,
      detail: rapidStatus === 'yes'
        ? `Ritiro collegato allo stesso soggetto; intervallo osservato ${{metricText(delayMin)}}–${{metricText(delayMax)}} secondi.`
        : (rapidStatus === 'no' ? 'Non è stato osservato un ritiro rapido attribuito dopo l’esecuzione.' : 'Tempo o attribuzione del ritiro non disponibili.'),
    }},
    {{
      label: 'Quantità eseguita inferiore alla quantità ritirata',
      status: gateStatus(
        ev.gate_small_fill_relative_to_withdrawal,
        qtyAvailable,
        executionQty > 0 && withdrawnQty > 0 && executionQty < withdrawnQty,
      ),
      detail: qtyAvailable
        ? `Quantità eseguita: ${{metricText(executionQty)}}; quantità ritirata: ${{metricText(withdrawnQty)}}.`
        : 'Una delle due quantità necessarie al confronto non è disponibile.',
    }},
    {{
      label: 'Movimento del prezzo favorevole prima dell’esecuzione',
      status: gateStatus(ev.gate_favorable_pre_fill_move, moveAvailable, favorableMove > 0),
      detail: moveAvailable
        ? `Variazione misurata: ${{metricText(favorableMove)}}. È favorevole solo se positiva.`
        : 'Non è disponibile una misura valida del prezzo prima dell’esecuzione.',
    }},
    {{
      label: 'Inversione del prezzo dopo il ritiro',
      status: gateStatus(ev.gate_cancel_anchored_reversion, reversionAvailable, reversion > 0),
      detail: reversionAvailable
        ? `Inversione misurata: ${{metricText(reversion)}}. Il controllo è superato solo se positiva.`
        : 'Non è disponibile una misura valida del prezzo dopo il ritiro.',
    }},
  ];
}}
function sequenceState(ev) {{
  const explicit = booleanValue(ev.spoofing_compatible_sequence);
  if (explicit !== null) return explicit ? 'complete' : 'review';
  const controls = behavioralControls(ev);
  if (controls.every(control => control.status === 'yes')) return 'complete';
  if (controls.some(control => control.status === 'no')) return 'review';
  return 'unavailable';
}}
function sequenceLabel(ev) {{
  const state = sequenceState(ev);
  if (state === 'complete') return 'Sequenza completa: 4 controlli su 4 superati';
  if (state === 'review') return 'Candidato da approfondire: controlli non tutti superati';
  return 'Valutazione incompleta: controlli non disponibili';
}}
function sequenceClass(ev) {{
  const state = sequenceState(ev);
  return state === 'complete' ? 'sequence-complete' : (state === 'review' ? 'sequence-review' : 'sequence-unavailable');
}}
function renderPopulation() {{
  const items = [
    ['reconstructed_clusters', 'Cluster ricostruiti', 'Tutti i cluster di esecuzione ricostruiti nell’elaborazione principale.'],
    ['review_candidates', 'Candidati da revisionare', 'Cluster con ritiro rapido attribuito allo stesso soggetto; costituiscono la popolazione della dashboard.'],
    ['compatible_sequences', 'Sequenze complete', 'Candidati che superano contemporaneamente tutti i quattro controlli.'],
    ['displayed_candidates', 'Candidati mostrati in questo file', 'Eventi effettivamente caricati; una dashboard top-N può mostrare solo i primi candidati ordinati per intensità del ritiro.'],
  ];
  document.getElementById('populationSummary').innerHTML = items.map(([key, title, help]) =>
    `<div class="population-item"><span class="population-value">${{countText(dashboardPopulation[key])}}</span><span class="population-label">${{title}}</span><span class="population-help">${{help}}</span></div>`
  ).join('');
}}
function label(ev) {{
  const status = sequenceState(ev) === 'complete' ? 'SEQUENZA COMPLETA' : 'DA APPROFONDIRE';
  return `[${{status}}] ${{ev.review_event_id}} | ${{ev.event_ts}} | soggetto=${{actorPlainText(ev)}} | identificazione=${{identityPlainText(ev)}} | esecuzione=${{executionAnchorShortText(ev)}} | capacità di negoziazione=${{capacityPlainText(ev)}} | quantità ritirata=${{metricText(ev.matched_deceptive_cancel_visible_qty_window, 2)}}`;
}}
function renderOverview() {{
  const x = reviewEvents.map(r => r.event_ts);
  const y = reviewEvents.map(withdrawalProfileScaleValue);
  const text = reviewEvents.map(label);
  Plotly.newPlot('overview', [{{x, y, text, mode:'markers', type:'scattergl', marker:{{color:'#d62728', size:9}}, hovertemplate:'%{{text}}<extra></extra>'}}],
    {{title:'Candidati mostrati, ordinati per WMSCI decrescente', xaxis:{{title:'orario dell’esecuzione'}}, yaxis:{{title:'WMSCI — intensità del ritiro attribuito'}}, margin:{{t:55}}}}, {{responsive:true}});
}}
function renderActorSessionAlerts() {{
  const el = document.getElementById('actorSessionAlerts');
  if (!actorSessionAlerts.length) {{ el.innerHTML = '<p>Nessuna segnalazione aggregata per soggetto/sessione è stata caricata.</p>'; return; }}
  let html = ['<table><thead><tr><th>Soggetto<br><small>Actor</small></th><th>Identificazione usata<br><small>Identity level</small></th><th>Tipo di esecuzione<br><small>Execution anchor</small></th><th>Limite dell’identificazione</th><th>Massima intensità del ritiro<br><small>Max withdrawal profile scale</small></th><th>Intensità media del ritiro<br><small>Mean withdrawal profile scale</small></th><th>Massimo indicatore MSCI<br><small>Max MSCI resting profile</small></th><th>MSCI medio<br><small>Mean MSCI resting profile</small></th><th>Eventi con ritiro attribuito</th><th>Cluster totali</th><th>Azione suggerita</th></tr></thead><tbody>'];
  for (const row of actorSessionAlerts) {{
    html.push(`<tr><td>${{escapeHtml(actorPlainText(row))}}</td><td>${{escapeHtml(identityPlainText(row))}}</td><td>${{escapeHtml(executionAnchorPlainText(row))}}</td><td>${{escapeHtml(identityScopeText(row) === 'firm_fallback_may_aggregate_multiple_clients' ? 'Il firm ID può riunire più clienti' : 'Cliente originale')}} </td><td>${{metricText(row.max_withdrawal_profile_scale_event ?? row.max_WMSCI_event, 4)}}</td><td>${{metricText(row.mean_withdrawal_profile_scale_event ?? row.mean_WMSCI_event, 4)}}</td><td>${{metricText(row.max_MSCI_resting_profile ?? row.max_MSCI, 4)}}</td><td>${{metricText(row.mean_MSCI_resting_profile ?? row.mean_MSCI, 4)}}</td><td>${{row.matched_event_count ?? ''}}</td><td>${{row.event_count ?? ''}}</td><td>${{escapeHtml(recommendedActionPlainText(row.recommended_action))}}</td></tr>`);
  }}
  html.push('</tbody></table>');
  el.innerHTML = html.join('');
}}
function renderSummary(ev) {{
  const controls = behavioralControls(ev);
  const statusText = status => status === 'yes' ? 'Sì' : (status === 'no' ? 'No' : 'Non disponibile');
  const statusClass = status => status === 'yes' ? 'control-yes' : (status === 'no' ? 'control-no' : 'control-na');
  const controlCards = controls.map(control => `<div class="control-card"><h4>${{escapeHtml(control.label)}}</h4><span class="control-status ${{statusClass(control.status)}}">${{statusText(control.status)}}</span><span class="control-detail">${{escapeHtml(control.detail)}}</span></div>`).join('');
  const complete = sequenceState(ev) === 'complete';
  const conclusion = complete
    ? 'Tutti i quattro controlli risultano soddisfatti: la sequenza è compatibile con lo schema comportamentale esaminato.'
    : 'Non tutti i controlli risultano soddisfatti o misurabili. Il caso resta visibile come candidato da approfondire.';
  document.getElementById('summary').innerHTML = `<h2>Evento ${{escapeHtml(ev.review_event_id ?? '')}} <span class="sequence-badge ${{sequenceClass(ev)}}">${{escapeHtml(sequenceLabel(ev))}}</span></h2>
  <p class="selection-reason"><b>Perché questo evento è presente.</b> È stato collegato allo stesso soggetto un ritiro rapido, successivo all’esecuzione, di uno o più ordini sul lato opposto. Identificativi degli ordini ritirati: ${{escapeHtml(ev.matched_deceptive_cancel_order_ids_window ?? 'Non disponibile')}}.</p>
  <p><b>Conclusione operativa:</b> ${{escapeHtml(conclusion)}}</p>
  <h3>Esito dei quattro controlli</h3>
  <div class="control-grid">${{controlCards}}</div>
  <h3>Informazioni essenziali</h3>
  <div class="key-facts">
    <div class="key-fact"><b>Orario dell’esecuzione</b><br>${{escapeHtml(ev.event_ts ?? 'Non disponibile')}}</div>
    <div class="key-fact"><b>Soggetto</b><br>${{escapeHtml(actorPlainText(ev))}}<br><small>${{escapeHtml(identityPlainText(ev))}}</small></div>
    <div class="key-fact"><b>Tipo di esecuzione</b><br>${{escapeHtml(executionAnchorPlainText(ev))}}</div>
    <div class="key-fact"><b>Lato eseguito / lato ritirato</b><br>${{escapeHtml(sidePlainText(ev.execution_side))}} / ${{escapeHtml(sidePlainText(ev.deceptive_side))}}</div>
    <div class="key-fact"><b>Quantità eseguita</b><br>${{metricText(executionQuantityValue(ev))}}</div>
    <div class="key-fact"><b>Quantità ritirata attribuita</b><br>${{metricText(ev.matched_deceptive_cancel_visible_qty_window)}}</div>
    <div class="key-fact"><b>WMSCI — intensità del ritiro attribuito</b><br>${{metricText(withdrawalProfileScaleValue(ev), 6)}}</div>
    <div class="key-fact"><b>MSCI del profilo a riposo</b><br>${{metricText(msciRestingProfileValue(ev), 6)}}</div>
  </div>
  <details><summary>Dettagli tecnici e dati originali</summary>
  <p><span class="badge">matched deceptive-order cancellation</span></p>
  <b>actor:</b> ${{escapeHtml(actorText(ev))}} &nbsp; <b>identity level:</b> ${{escapeHtml(identityLevelText(ev))}} &nbsp; <b>identity scope:</b> ${{escapeHtml(identityScopeText(ev))}} &nbsp; <b>execution anchor:</b> ${{escapeHtml(executionAnchorText(ev))}}<br>
  <b>raw event client:</b> ${{escapeHtml(ev.event_client_original_id ?? 'NA')}} &nbsp; <b>raw event firm:</b> ${{escapeHtml(ev.event_firm_id ?? 'NA')}} &nbsp; <b>trading capacity:</b> ${{escapeHtml(capacityText(ev))}}<br>
  <b>campi metrici:</b> <code>withdrawal_profile_scale_event</code> (alias <code>WMSCI_event</code>); <code>MSCI_resting_profile</code> (alias <code>MSCI</code>) &nbsp; <b>SCI:</b> ${{metricText(ev.SCI, 6)}}<br>
  <b>favorable pre-fill mid move:</b> ${{metricText(ev.favorable_mid_move_pre_fill)}} &nbsp; <b>post-cancel mid reversion:</b> ${{metricText(ev.post_cancel_mid_reversion)}} &nbsp; <b>execution advantage vs posture mid:</b> ${{metricText(ev.execution_price_advantage_vs_posture_mid)}}<br>
  <b>candidate visible qty pre:</b> ${{metricText(ev.candidate_deceptive_visible_qty_pre)}} &nbsp; <b>matched cancel qty:</b> ${{metricText(ev.matched_deceptive_cancel_visible_qty_window)}} &nbsp; <b>matched fraction:</b> ${{metricText(ev.matched_deceptive_cancel_fraction_window)}}<br>
  <b>withdrawal/fill:</b> ${{metricText(ev.withdrawal_to_fill_ratio)}} &nbsp; <b>cancel delay:</b> ${{metricText(ev.matched_deceptive_cancel_min_delay_seconds)}}–${{metricText(ev.matched_deceptive_cancel_max_delay_seconds)}}s<br>
  <b>candidate order ids:</b> ${{escapeHtml(ev.candidate_deceptive_order_ids_pre ?? 'NA')}}<br>
  <b>matched cancelled ids:</b> ${{escapeHtml(ev.matched_deceptive_cancel_order_ids_window ?? 'NA')}}
  </details>`;
}}
function renderLOB(ev) {{
  const rows = byEvent(ev.review_event_id, queueRows);
  const maxLevel = Math.max(...rows.map(r => Number(r.level) || 0), 1);
  const yLabels = [];
  for (let level = maxLevel; level >= 1; level--) yLabels.push(`bid ${{level}}`);
  for (let level = 1; level <= maxLevel; level++) yLabels.push(`ask ${{level}}`);

  function chooseSnapshot(phase) {{
    const phaseRows = rows.filter(r => r.snapshot_phase === phase);
    if (!phaseRows.length) return null;
    let sortIndex;
    if (phase === 'pre') sortIndex = Math.max(...phaseRows.map(r => Number(r.snapshot_sort_index)));
    else if (phase === 'execution') sortIndex = Number(ev.cluster_first_sort_index);
    else sortIndex = Number(ev.post_cancel_sort_index);
    if (!Number.isFinite(sortIndex)) return null;
    if (!phaseRows.some(r => Number(r.snapshot_sort_index) === sortIndex)) {{
      if (phase === 'post') return null;
      let best = phaseRows[0];
      for (const r of phaseRows) {{
        if (Math.abs(Number(r.snapshot_sort_index) - sortIndex) < Math.abs(Number(best.snapshot_sort_index) - sortIndex)) best = r;
      }}
      sortIndex = Number(best.snapshot_sort_index);
    }}
    return phaseRows.filter(r => Number(r.snapshot_sort_index) === sortIndex);
  }}

  function bestQuotesForStage(phase) {{
    const chosen = chooseSnapshot(phase) || [];
    let bestBid = null;
    let bestAsk = null;
    for (const r of chosen) {{
      const price = Number(r.price);
      if (!Number.isFinite(price)) continue;
      if (r.side === 'bid' && (bestBid === null || price > bestBid)) bestBid = price;
      if (r.side === 'ask' && (bestAsk === null || price < bestAsk)) bestAsk = price;
    }}
    return {{bestBid, bestAsk}};
  }}

  function formatPrice(value) {{
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return 'NA';
    return Number(value).toFixed(4).replace(/0+$/, '').replace(/\\.$/, '');
  }}

  function formatQuantity(value) {{
    const number = Number(value);
    if (!Number.isFinite(number)) return 'NA';
    return number.toLocaleString(undefined, {{maximumFractionDigits:6}});
  }}

  function stageLabel(phase) {{
    const chosen = chooseSnapshot(phase) || [];
    if (!chosen.length) return 'istantanea non disponibile';
    const row = chosen[0];
    const ts = row.snapshot_ts || '';
    const sortIndex = row.snapshot_sort_index;
    const quotes = bestQuotesForStage(phase);
    const quoteText = `miglior acquisto ${{formatPrice(quotes.bestBid)}}<br>miglior vendita ${{formatPrice(quotes.bestAsk)}}`;
    if (phase === 'execution') return `${{ts}}<br>sequenza ${{sortIndex}}<br>${{quoteText}}<br>quantità eseguita ${{ev.fill_qty}}`;
    return `${{ts}}<br>sequenza ${{sortIndex}}<br>${{quoteText}}`;
  }}

  function stageArrays(phase) {{
    const chosen = chooseSnapshot(phase) || [];
    const byLevel = new Map();
    for (const r of chosen) {{
      const key = `${{r.side}}|${{r.level}}`;
      const item = byLevel.get(key) || {{total: Number(r.level_visible_qty) || 0, candidate: 0, executed: 0, price: r.price, queue: r.actor_queue_dict ?? r.client_queue_dict}};
      if (r.is_candidate_deceptive_order || r.is_matched_deceptive_cancel_order) item.candidate += Number(r.visible_qty) || 0;
      byLevel.set(key, item);
    }}

    if (phase === 'execution') {{
      const eventRows = byEvent(ev.review_event_id, eventLog);
      const execEvent = eventRows.find(r => r.is_execution_order && Number(r.sort_index) === Number(ev.cluster_first_sort_index)) ||
        eventRows.find(r => r.is_execution_order && r.event_class === 'fill' && Number(r.last_shares || 0) > 0) ||
        eventRows.find(r => r.is_execution_order);
      if (execEvent) {{
        const execSide = execEvent.side;
        const execPrice = Number(execEvent.price);
        const execQty = Number(execEvent.last_shares || ev.fill_qty || 0);
        const sidePrices = Array.from(new Set(chosen.filter(r => r.side === execSide).map(r => Number(r.price))));
        sidePrices.sort((a, b) => execSide === 'bid' ? b - a : a - b);
        let execLevel = sidePrices.findIndex(p => Math.abs(p - execPrice) < 1e-12) + 1;
        if (execLevel <= 0) {{
          execLevel = sidePrices.filter(p => execSide === 'bid' ? p > execPrice : p < execPrice).length + 1;
        }}
        if (execLevel >= 1 && execLevel <= maxLevel && execQty > 0) {{
          const key = `${{execSide}}|${{execLevel}}`;
          const item = byLevel.get(key) || {{total: 0, candidate: 0, executed: 0, price: execPrice, queue: ''}};
          item.executed += execQty;
          item.price = item.price || execPrice;
          byLevel.set(key, item);
        }}
      }}
    }}

    const totalBid = [], totalAsk = [], candidateBid = [], candidateAsk = [], executedBid = [], executedAsk = [], hover = [];
    for (const label of yLabels) {{
      const [side, levelText] = label.split(' ');
      const item = byLevel.get(`${{side}}|${{levelText}}`) || {{total: 0, candidate: 0, executed: 0, price: '', queue: ''}};
      totalBid.push(side === 'bid' ? item.total : 0);
      totalAsk.push(side === 'ask' ? item.total : 0);
      candidateBid.push(side === 'bid' ? item.candidate : 0);
      candidateAsk.push(side === 'ask' ? item.candidate : 0);
      executedBid.push(side === 'bid' ? item.executed : 0);
      executedAsk.push(side === 'ask' ? item.executed : 0);
      hover.push(`${{label}} @ ${{item.price}}<br>quantità totale=${{item.total}}<br>quantità del soggetto candidato=${{item.candidate}}<br>quantità eseguita=${{item.executed}}<br>coda del soggetto=${{item.queue}}`);
    }}
    return {{totalBid, totalAsk, candidateBid, candidateAsk, executedBid, executedAsk, hover}};
  }}

  const stages = [
    ['pre', '1. prima dell’esecuzione'],
    ['execution', '2. esecuzione selezionata'],
    ['post', '3. dopo il ritiro']
  ];
  const traces = [];
  const xaxes = ['x', 'x2', 'x3'];
  for (let i = 0; i < stages.length; i++) {{
    const [phase, title] = stages[i];
    const vals = stageArrays(phase);
    const common = {{y: yLabels, type:'bar', orientation:'h', hovertext: vals.hover, hovertemplate:'%{{hovertext}}<extra></extra>', xaxis: xaxes[i], yaxis:'y'}};
    const executionMarker = {{
      y: yLabels, type:'scatter', mode:'markers+text', hovertext: vals.hover,
      hovertemplate:'%{{hovertext}}<extra></extra>', xaxis: xaxes[i], yaxis:'y',
      textposition:'middle right', cliponaxis:false,
    }};
    const executionX = values => values.map(value => Number(value) > 0 ? Number(value) : null);
    const executionText = values => values.map(value => Number(value) > 0 ? `eseguito ${{formatQuantity(value)}}` : '');
    traces.push({{...common, x: vals.totalBid, name:'totale acquisti (bid)', legendgroup:'bid total', showlegend:i===0, marker:{{color:'rgba(44,160,44,0.24)'}}}});
    traces.push({{...common, x: vals.totalAsk, name:'totale vendite (ask)', legendgroup:'ask total', showlegend:i===0, marker:{{color:'rgba(214,39,40,0.22)'}}}});
    traces.push({{...common, x: vals.candidateBid, name:'soggetto candidato — acquisti', legendgroup:'candidate bid', showlegend:i===0, marker:{{color:'rgba(0,100,0,0.95)'}}, width:0.48}});
    traces.push({{...common, x: vals.candidateAsk, name:'soggetto candidato — vendite', legendgroup:'candidate ask', showlegend:i===0, marker:{{color:'rgba(150,0,0,0.95)'}}, width:0.48}});
    traces.push({{...executionMarker, x: executionX(vals.executedBid), text: executionText(vals.executedBid), name:'quantità eseguita — acquisto', legendgroup:'executed bid', showlegend:i===1, marker:{{color:'rgba(0,55,0,1.0)', line:{{color:'#111', width:1}}, symbol:'diamond', size:10}}}});
    traces.push({{...executionMarker, x: executionX(vals.executedAsk), text: executionText(vals.executedAsk), name:'quantità eseguita — vendita', legendgroup:'executed ask', showlegend:i===1, marker:{{color:'rgba(110,0,0,1.0)', line:{{color:'#111', width:1}}, symbol:'diamond', size:10}}}});
  }}
  Plotly.newPlot('lob', traces,
    {{
      title:'Profondità del book nelle tre fasi: quantità totale, quantità del soggetto candidato e quantità eseguita',
      barmode:'overlay',
      bargap:0.18,
      yaxis:{{categoryorder:'array', categoryarray:yLabels, autorange:'reversed'}},
      xaxis:{{domain:[0.00,0.30], title:{{text:'1. prima dell’esecuzione', standoff:10}}, rangemode:'tozero'}},
      xaxis2:{{domain:[0.35,0.65], title:{{text:'2. esecuzione selezionata', standoff:10}}, rangemode:'tozero'}},
      xaxis3:{{domain:[0.70,1.00], title:{{text:'3. dopo il ritiro', standoff:10}}, rangemode:'tozero'}},
      annotations:[
        {{text:'acquisti (bid)', xref:'paper', yref:'paper', x:-0.055, y:0.77, textangle:-90, showarrow:false, font:{{color:'#1b7f1b', size:12}}}},
        {{text:'vendite (ask)', xref:'paper', yref:'paper', x:-0.055, y:0.27, textangle:-90, showarrow:false, font:{{color:'#b22222', size:12}}}},
        {{text:stageLabel('pre'), xref:'paper', yref:'paper', x:0.15, y:1.08, showarrow:false, align:'center', bgcolor:'rgba(255,255,255,0.86)', bordercolor:'#d7dde8', borderpad:4, font:{{size:11, color:'#364152'}}}},
        {{text:stageLabel('execution'), xref:'paper', yref:'paper', x:0.50, y:1.08, showarrow:false, align:'center', bgcolor:'rgba(255,255,255,0.86)', bordercolor:'#d7dde8', borderpad:4, font:{{size:11, color:'#364152'}}}},
        {{text:stageLabel('post'), xref:'paper', yref:'paper', x:0.85, y:1.08, showarrow:false, align:'center', bgcolor:'rgba(255,255,255,0.86)', bordercolor:'#d7dde8', borderpad:4, font:{{size:11, color:'#364152'}}}}
      ],
      legend:{{orientation:'h', y:-0.16}},
      margin:{{l:100,t:125,b:95}}
    }}, {{responsive:true}});
}}
function escapeHtml(text) {{ return String(text).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function renderChildFills(ev) {{
  const clusterId = ev.execution_cluster_id || ev.review_event_id;
  const rows = byCluster(clusterId, childMembers).sort((a,b) => Number(a.child_sort_index)-Number(b.child_sort_index));
  const el = document.getElementById('childFills');
  if (!rows.length) {{ el.innerHTML = '<p>Nessuna esecuzione elementare è stata caricata per questo cluster.</p>'; return; }}
  const html = ['<table><thead><tr><th>Sequenza</th><th>Orario</th><th>Tipo di esecuzione</th><th>ID ordine</th><th>ID evento</th><th>ID esecuzione</th><th>UID operazione</th><th>Quantità</th><th>Prezzo</th><th>Cliente originale</th><th>Firm originale</th></tr></thead><tbody>'];
  for (const r of rows) {{
    const anchor = r.child_execution_anchor_mode ?? r.execution_anchor_mode;
    const anchorLabel = anchor === 'passive' ? 'passiva' : (anchor === 'aggressive' ? 'aggressiva' : (anchor || 'Non disponibile'));
    html.push(`<tr><td>${{escapeHtml(r.child_sort_index ?? '')}}</td><td>${{escapeHtml(r.child_event_ts ?? '')}}</td><td>${{escapeHtml(anchorLabel)}}</td><td>${{escapeHtml(r.child_order_id ?? '')}}</td><td>${{escapeHtml(r.child_event_id ?? '')}}</td><td>${{escapeHtml(r.child_execution_id ?? '')}}</td><td>${{escapeHtml(r.child_trade_uid ?? '')}}</td><td>${{escapeHtml(r.child_fill_qty ?? '')}}</td><td>${{escapeHtml(r.child_fill_price ?? '')}}</td><td>${{escapeHtml(r.child_event_client_original_id ?? '')}}</td><td>${{escapeHtml(r.child_event_firm_id ?? '')}}</td></tr>`);
  }}
  html.push('</tbody></table>');
  el.innerHTML = html.join('');
}}
function renderCancelCandidates(ev) {{
  const clusterId = ev.execution_cluster_id || ev.review_event_id;
  const rows = byCluster(clusterId, cancelCandidates).sort((a,b) => Number(a.cancel_sort_index)-Number(b.cancel_sort_index));
  const el = document.getElementById('cancelCandidates');
  if (!rows.length) {{ el.innerHTML = '<p>Nessun collegamento a ritiri attribuiti o alternativi è stato caricato per questo cluster.</p>'; return; }}
  const html = ['<table><thead><tr><th>Esito</th><th>Sequenza del ritiro</th><th>Orario del ritiro</th><th>Ordine candidato</th><th>Lato</th><th>Quantità visibile</th><th>Regola di assegnazione</th><th>Cluster concorrenti</th><th>Misura dopo il ritiro</th><th>Inversione del prezzo</th></tr></thead><tbody>'];
  for (const r of rows) {{
    const status = r.assigned_flag === true ? 'attribuito a questo cluster' : 'collegamento alternativo';
    html.push(`<tr><td>${{status}}</td><td>${{escapeHtml(r.cancel_sort_index ?? '')}}</td><td>${{escapeHtml(r.cancel_event_ts ?? r.event_ts ?? '')}}</td><td>${{escapeHtml(r.candidate_order_id ?? r.ORDERID ?? '')}}</td><td>${{escapeHtml(sidePlainText(r.deceptive_side))}}</td><td>${{escapeHtml(r.cancel_visible_qty ?? r.visible_qty_pre_cancel ?? '')}}</td><td>${{escapeHtml(assignmentRulePlainText(r.assignment_rule))}}</td><td>${{escapeHtml(r.competing_cluster_count ?? '')}}</td><td>${{r.has_cancel_reversion_state === true ? 'disponibile' : 'non disponibile'}}</td><td>${{metricText(r.post_cancel_mid_reversion)}}</td></tr>`);
  }}
  html.push('</tbody></table>');
  el.innerHTML = html.join('');
}}
function renderEventTable(ev) {{
  const rows = byEvent(ev.review_event_id, eventLog).sort((a,b) => a.sort_index-b.sort_index);
  const html = ['<div class="event-legend"><span class="event-row-actor">soggetto in revisione</span><span class="event-row-execution">esecuzione selezionata</span><span class="event-row-candidate">ordine candidato</span><span class="event-row-matched-cancel">ritiro attribuito</span></div><table><thead><tr><th>Sequenza</th><th>Orario</th><th>Tipo di messaggio</th><th>Lato</th><th>Prezzo</th><th title="Posizione effettiva sullo stesso lato dopo il messaggio; per un prezzo non presente nel book è mostrata la posizione ipotetica.">Livello nel book</th><th>ID ordine</th><th>Cliente originale</th><th>Firm originale</th><th>Capacità di negoziazione</th><th>Quantità residua</th><th>Quantità mostrata</th><th>Quantità eseguita</th><th>Indicatori</th></tr></thead><tbody>'];
  for (const r of rows) {{
    const isReviewActor = r.is_review_actor ?? r.is_review_client ?? false;
    const flags = [r.is_execution_order?'esecuzione selezionata':'', r.is_candidate_deceptive_order?'ordine candidato':'', r.is_matched_deceptive_cancel_order?'ritiro attribuito':'', isReviewActor?'soggetto in revisione':''].filter(Boolean).join(', ');
    const rowClasses = [
      isReviewActor ? 'event-row-actor' : '',
      r.is_execution_order ? 'event-row-execution' : '',
      r.is_candidate_deceptive_order ? 'event-row-candidate' : '',
      r.is_matched_deceptive_cancel_order ? 'event-row-matched-cancel' : ''
    ].filter(Boolean).join(' ');
    html.push(`<tr class="${{rowClasses}}"><td>${{escapeHtml(r.sort_index ?? '')}}</td><td>${{escapeHtml(r.event_ts ?? '')}}</td><td>${{escapeHtml(eventClassPlainText(r.event_class))}}</td><td>${{escapeHtml(sidePlainText(r.side))}}</td><td>${{escapeHtml(r.price ?? '')}}</td><td>${{escapeHtml(formatZoomLevel(r.book_level, r.book_level_is_resting))}}</td><td>${{escapeHtml(r.ORDERID ?? '')}}</td><td>${{escapeHtml(r.event_client_original_id ?? r.client_id ?? '')}}</td><td>${{escapeHtml(r.event_firm_id ?? r.firm_id ?? '')}}</td><td>${{escapeHtml(capacityPlainText(r))}}</td><td>${{escapeHtml(r.leaves_qty ?? '')}}</td><td>${{escapeHtml(r.displayed_qty ?? '')}}</td><td>${{escapeHtml(r.last_shares ?? '')}}</td><td>${{escapeHtml(flags)}}</td></tr>`);
  }}
  html.push('</tbody></table>');
  document.getElementById('eventTable').innerHTML = html.join('');
}}
function update(id) {{
  const ev = reviewEvents.find(r => r.review_event_id === id);
  if (!ev) return;
  renderSummary(ev); renderLOB(ev); renderChildFills(ev); renderCancelCandidates(ev); renderEventTable(ev);
}}
const select = document.getElementById('eventSelect');
const parameterSelect = document.getElementById('parameterSelect');
const evidenceFilter = document.getElementById('evidenceFilter');
function populateEvents() {{
  select.innerHTML = '';
  for (const ev of reviewEvents) {{ const opt = document.createElement('option'); opt.value = ev.review_event_id; opt.textContent = label(ev); select.appendChild(opt); }}
  select.disabled = reviewEvents.length === 0;
  document.getElementById('filterResultCount').textContent = `${{reviewEvents.length.toLocaleString('it-IT')}} eventi mostrati su ${{currentRunEvents.length.toLocaleString('it-IT')}} caricati`;
}}
function clearEventDetail() {{
  document.getElementById('summary').innerHTML = '<div class="empty-filter"><b>Nessun evento corrisponde al filtro selezionato.</b><br>Modifica il filtro per tornare ai candidati disponibili.</div>';
  document.getElementById('lob').innerHTML = '';
  document.getElementById('childFills').innerHTML = '';
  document.getElementById('cancelCandidates').innerHTML = '';
  document.getElementById('eventTable').innerHTML = '';
}}
function applyEvidenceFilter() {{
  const value = evidenceFilter.value;
  if (value === 'complete') reviewEvents = currentRunEvents.filter(event => sequenceState(event) === 'complete');
  else if (value === 'review') reviewEvents = currentRunEvents.filter(event => sequenceState(event) !== 'complete');
  else reviewEvents = currentRunEvents;
  populateEvents();
  renderOverview();
  if (reviewEvents.length) update(reviewEvents[0].review_event_id);
  else clearEventDetail();
}}
if (parameterRuns.length) {{
  for (let i = 0; i < parameterRuns.length; i++) {{ const opt = document.createElement('option'); opt.value = String(i); opt.textContent = parameterRuns[i].label; parameterSelect.appendChild(opt); }}
}} else {{
  const opt = document.createElement('option'); opt.value = 'base'; opt.textContent = 'elaborazione principale (current metric run)'; parameterSelect.appendChild(opt);
  parameterSelect.disabled = true;
}}
parameterSelect.addEventListener('change', e => {{
  if (parameterRuns.length) currentRunEvents = withBaseReviewContext(parameterRuns[Number(e.target.value)].events);
  applyEvidenceFilter();
}});
evidenceFilter.addEventListener('change', applyEvidenceFilter);
select.addEventListener('change', e => update(e.target.value));
renderPopulation();
renderActorSessionAlerts();
applyEvidenceFilter();
</script>
</div>
</body>
</html>
"""
    path.write_text(html)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.actor_session_alerts is not None and args.client_session_alerts is not None:
        raise ValueError("provide --actor-session-alerts or deprecated --client-session-alerts, not both")
    raw_events = pl.read_parquet(args.input)
    execution_metrics = pl.read_parquet(args.execution_metrics)
    # Candidate file is loaded to verify provenance and fail early if absent/corrupt.
    pl.read_parquet(args.candidate_deceptive_orders)
    child_members = _load_optional_parquet(args.execution_cluster_members)
    review_events = _prepare_review_events(
        execution_metrics,
        args.max_events,
        cluster_members=child_members,
    )
    parameter_review_events = _load_parameter_review_events(args.parameter_grid_root, args.max_events) if args.parameter_grid_root else []
    if parameter_review_events:
        by_id = {event["review_event_id"]: event for event in review_events}
        for run in parameter_review_events:
            for event in run["events"]:
                if event["review_event_id"] not in by_id:
                    clone = dict(event)
                    clone["event_ts_parsed"] = _parse_ts(clone.get("event_ts"))
                    clone["candidate_order_ids"] = _split_ids(clone.get("candidate_deceptive_order_ids_pre"))
                    clone["matched_order_ids"] = _split_ids(clone.get("matched_deceptive_cancel_order_ids_window"))
                    by_id[clone["review_event_id"]] = clone
        review_events = sorted(by_id.values(), key=lambda row: int(row["sort_index"]))
    if not review_events:
        raise ValueError("no matched deceptive-order cancellation events found")
    review_df, event_log_df, queue_df = reconstruct_review_windows(
        raw_events,
        review_events,
        top_n=args.top_n,
        pre_window_seconds=args.pre_window_seconds,
        post_window_seconds=args.post_window_seconds,
        queue_snapshot_mode=args.queue_snapshot_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.output_dir / "matched_spoofing_events.parquet"
    event_log_path = args.output_dir / "matched_spoofing_event_log.parquet"
    queue_path = args.output_dir / "matched_spoofing_lob_queue.parquet"
    dashboard_path = args.output_dir / "matched_spoofing_event_review_dashboard.html"
    metadata_path = args.output_dir / "metadata.json"
    review_df.write_parquet(review_path)
    artifact_paths = write_review_artifacts(output_dir=args.output_dir, event_log=event_log_df, queue=queue_df)
    event_log_path = artifact_paths["event_log"]
    queue_path = artifact_paths["queue"]
    metric_metadata = _load_metric_metadata(args.execution_metrics)
    cancel_candidates = _load_optional_parquet(args.execution_cancel_candidates)
    if child_members is not None:
        child_members.write_parquet(args.output_dir / "execution_cluster_members.parquet")
    if cancel_candidates is not None:
        cancel_candidates.write_parquet(args.output_dir / "execution_cancel_candidates.parquet")
    dashboard_refreshed_at_utc = datetime.now(timezone.utc).isoformat()
    population_summary = _review_population_summary(execution_metrics, review_df)
    write_dashboard(
        dashboard_path,
        review_events=review_df,
        event_log=event_log_df,
        queue=queue_df,
        review_top_n=args.top_n,
        pre_window_seconds=args.pre_window_seconds,
        post_window_seconds=args.post_window_seconds,
        metric_metadata=metric_metadata,
        parameter_review_events=parameter_review_events,
        actor_session_alerts=(
            _load_optional_parquet(args.actor_session_alerts) if args.actor_session_alerts is not None else None
        ),
        client_session_alerts=(
            _load_optional_parquet(args.client_session_alerts) if args.client_session_alerts is not None else None
        ),
        child_members=child_members,
        cancel_candidates=cancel_candidates,
        dashboard_refreshed_at_utc=dashboard_refreshed_at_utc,
        population_summary=population_summary,
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **spoofing_config_provenance(args.config, section="event_review"),
        "input": str(args.input),
        "execution_metrics": str(args.execution_metrics),
        "candidate_deceptive_orders": str(args.candidate_deceptive_orders),
        "output_dir": str(args.output_dir),
        "top_n": args.top_n,
        "pre_window_seconds": args.pre_window_seconds,
        "post_window_seconds": args.post_window_seconds,
        "max_events": args.max_events,
        "queue_snapshot_mode": args.queue_snapshot_mode,
        "execution_cluster_members": str(args.execution_cluster_members) if args.execution_cluster_members else None,
        "execution_cancel_candidates": str(args.execution_cancel_candidates) if args.execution_cancel_candidates else None,
        "actor_session_alerts": str(args.actor_session_alerts) if args.actor_session_alerts else None,
        "legacy_client_session_alerts": str(args.client_session_alerts) if args.client_session_alerts else None,
        "dashboard_refreshed_at_utc": dashboard_refreshed_at_utc,
        "metric_run_parameters": _metric_run_parameters(metric_metadata),
        "parameter_grid_root": str(args.parameter_grid_root) if args.parameter_grid_root else None,
        "parameter_run_count": len(parameter_review_events),
        "review_event_count": review_df.height,
        "review_population_summary": population_summary,
        "event_log_rows": event_log_df.height,
        "queue_rows": queue_df.height,
        "paths": {
            "matched_spoofing_events": str(review_path),
            "matched_spoofing_event_log": str(event_log_path),
            "matched_spoofing_lob_queue": str(queue_path),
            "dashboard": str(dashboard_path),
        },
        "command": sys.argv,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    print(f"matched_events: {review_df.height}")
    print(f"event_log_rows: {event_log_df.height}")
    print(f"queue_rows: {queue_df.height}")
    print(f"queue_parquet: {queue_path}")
    print(f"dashboard: {dashboard_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
