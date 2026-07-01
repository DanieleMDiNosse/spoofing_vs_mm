from __future__ import annotations

import bisect
import math
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

import polars as pl

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.models import ActiveOrder
from spoofing_detection.lob.normalize import normalize_event
from spoofing_detection.lob.panel import (
    _apply_event,
    _fill_group_key,
    _flush_pending_aggressive_residuals,
    _partition_id,
    sort_events,
)

BEST_QUOTE_COLUMNS = ("post_best_bid", "post_best_ask")
VISIBLE_LIMIT_ORDER_TYPES = {"limit", "iceberg"}


@dataclass(frozen=True)
class ExploratoryMetricsConfig:
    top_n: int = 3
    kappa: float = 1.0
    lambda_: float = 1.0
    epsilon: float = 1e-12
    window_seconds: float = 1.0


@dataclass(frozen=True)
class ExploratoryMetricsResult:
    state_time_series: pl.DataFrame
    execution_metrics: pl.DataFrame
    candidate_deceptive_orders: pl.DataFrame
    direct_cancellations: pl.DataFrame
    rejected_executions: pl.DataFrame


def infer_tick_size_from_best_quotes(panel: pl.DataFrame) -> float:
    missing = [column for column in BEST_QUOTE_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(f"cannot infer tick size; missing columns: {', '.join(missing)}")

    positive_diffs: list[float] = []
    for column in BEST_QUOTE_COLUMNS:
        prices: list[float] = []
        for value in panel.get_column(column).drop_nulls().to_list():
            numeric = float(value)
            if math.isfinite(numeric):
                prices.append(numeric)
        unique = sorted(set(prices))
        positive_diffs.extend(b - a for a, b in zip(unique, unique[1:]) if b > a)
    if not positive_diffs:
        raise ValueError("cannot infer tick size from best quotes; no positive price changes")
    return min(positive_diffs)


def _parse_ts(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=None)
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def choose_event_timestamp(event: dict[str, Any]) -> datetime | None:
    for key in ("TRADETIME", "BOOKOUTTIME", "BOOKIN", "SEQUENCETIME"):
        parsed = _parse_ts(event.get(key))
        if parsed is not None:
            return parsed
    return None


def _visible_qty(order: ActiveOrder) -> float:
    if order.leaves_qty <= 0 or order.displayed_qty <= 0:
        return 0.0
    return float(order.displayed_qty)


def _market_levels(
    active_orders: Mapping[str, ActiveOrder], *, side: str, top_n: int
) -> list[tuple[float, float]]:
    by_price: dict[float, float] = defaultdict(float)
    for order in active_orders.values():
        qty = _visible_qty(order)
        if qty > 0 and order.side == side:
            by_price[float(order.price)] += qty
    prices = sorted(by_price, reverse=(side == "bid"))[:top_n]
    return [(price, by_price[price]) for price in prices]


def _distance_ticks(side: str, price: float, best_price: float, tick_size: float) -> float:
    raw = (best_price - price) / tick_size if side == "bid" else (price - best_price) / tick_size
    if abs(raw) < 1e-9:
        return 0.0
    return max(raw, 0.0)


def shifted_depth_distance_ticks(side: str, price: float, best_price: float, tick_size: float) -> float:
    """Paper-aligned same-side tick distance: level 1 has strictly positive distance."""
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    return _distance_ticks(side, price, best_price, tick_size) + 1.0


def depth_kernel_weights(distances: list[float], *, kappa: float, lambda_: float) -> list[float]:
    """Normalized depth kernel from the top-n DWI/MSCI paper formulation."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive")
    raw = [math.exp(-lambda_ * d) * (1.0 - math.exp(-kappa * d)) for d in distances]
    total = sum(raw)
    if total <= 0:
        return [0.0 for _ in distances]
    return [value / total for value in raw]


def _side_depth_metadata(
    levels: list[tuple[float, float]],
    *,
    side: str,
    tick_size: float,
    kappa: float,
    lambda_: float,
) -> dict[float, dict[str, float]]:
    if not levels:
        return {}
    best_price = levels[0][0]
    distances = [shifted_depth_distance_ticks(side, price, best_price, tick_size) for price, _ in levels]
    weights = depth_kernel_weights(distances, kappa=kappa, lambda_=lambda_)
    return {
        price: {
            "delta_ticks": _distance_ticks(side, price, best_price, tick_size),
            "depth_distance_ticks": distances[idx],
            "kernel_weight": weights[idx],
        }
        for idx, (price, _) in enumerate(levels)
    }


def compute_client_top_n_exposures(
    active_orders: Mapping[str, ActiveOrder],
    *,
    top_n: int,
    tick_size: float,
    kappa: float,
    lambda_: float,
    partition_id: str | None,
    sort_index: int,
    event_ts: datetime | None,
    include_level_columns: bool = True,
    client_ids: set[str] | None = None,
) -> list[dict[str, Any]]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive")

    levels = {
        "bid": _market_levels(active_orders, side="bid", top_n=top_n),
        "ask": _market_levels(active_orders, side="ask", top_n=top_n),
    }
    market_total = {side: sum(qty for _, qty in side_levels) for side, side_levels in levels.items()}
    best_bid = levels["bid"][0][0] if levels["bid"] else None
    best_ask = levels["ask"][0][0] if levels["ask"] else None
    best_bid_qty = levels["bid"][0][1] if levels["bid"] else 0.0
    best_ask_qty = levels["ask"][0][1] if levels["ask"] else 0.0
    market_mid = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else None
    market_microprice = (
        (best_ask * best_bid_qty + best_bid * best_ask_qty) / (best_bid_qty + best_ask_qty)
        if best_bid is not None and best_ask is not None and best_bid_qty + best_ask_qty > 0
        else None
    )
    price_to_rank = {
        side: {price: rank for rank, (price, _) in enumerate(side_levels, start=1)}
        for side, side_levels in levels.items()
    }
    side_meta = {
        side: _side_depth_metadata(side_levels, side=side, tick_size=tick_size, kappa=kappa, lambda_=lambda_)
        for side, side_levels in levels.items()
    }

    client_level_qty: dict[tuple[str, str, int], float] = defaultdict(float)
    active_client_ids: set[str] = set()
    for order in active_orders.values():
        client_id = str(order.client_original_id) if order.client_original_id is not None else None
        qty = _visible_qty(order)
        if client_id is None or qty <= 0 or order.side not in {"bid", "ask"}:
            continue
        if client_ids is not None and client_id not in client_ids:
            continue
        rank = price_to_rank[order.side].get(float(order.price))
        if rank is None:
            continue
        active_client_ids.add(client_id)
        client_level_qty[(client_id, order.side, rank)] += qty

    rows: list[dict[str, Any]] = []
    for client_id in sorted(active_client_ids):
        row: dict[str, Any] = {
            "partition_id": partition_id,
            "sort_index": sort_index,
            "event_ts": event_ts,
            "client_id": client_id,
            "top_n": top_n,
            "tick_size": tick_size,
            "kappa": kappa,
            "lambda_": lambda_,
            "market_best_bid": best_bid,
            "market_best_ask": best_ask,
            "market_mid": market_mid,
            "market_microprice": market_microprice,
        }
        liquidity: dict[str, float] = {}
        for side in ("bid", "ask"):
            client_side_qty = 0.0
            side_liquidity = 0.0
            for rank in range(1, top_n + 1):
                if rank <= len(levels[side]):
                    price, market_qty = levels[side][rank - 1]
                    client_qty = client_level_qty[(client_id, side, rank)]
                    meta = side_meta[side][price]
                    relative_depth = client_qty / market_qty if market_qty > 0 else 0.0
                    contribution = meta["kernel_weight"] * relative_depth
                    delta_ticks = meta["delta_ticks"]
                    depth_distance = meta["depth_distance_ticks"]
                    kernel_weight = meta["kernel_weight"]
                else:
                    price = None
                    market_qty = 0.0
                    client_qty = 0.0
                    relative_depth = 0.0
                    contribution = 0.0
                    delta_ticks = None
                    depth_distance = None
                    kernel_weight = 0.0
                client_side_qty += client_qty
                side_liquidity += contribution
                if include_level_columns:
                    row[f"{side}_level_{rank}_price"] = price
                    row[f"{side}_level_{rank}_market_visible_qty"] = market_qty
                    row[f"{side}_level_{rank}_client_visible_qty"] = client_qty
                    row[f"{side}_level_{rank}_client_fraction"] = client_qty / market_qty if market_qty > 0 else 0.0
                    row[f"{side}_level_{rank}_client_relative_depth"] = relative_depth
                    row[f"{side}_level_{rank}_delta_ticks"] = delta_ticks
                    row[f"{side}_level_{rank}_depth_distance_ticks"] = depth_distance
                    row[f"{side}_level_{rank}_kernel_weight"] = kernel_weight
                    row[f"{side}_level_{rank}_weighted_liquidity_contribution"] = contribution
            liquidity[side] = side_liquidity
            row[f"client_{side}_qty_topN"] = client_side_qty
            row[f"market_{side}_qty_topN"] = market_total[side]
            row[f"raw_{side}_fraction_topN"] = client_side_qty / market_total[side] if market_total[side] > 0 else 0.0
            row[f"L_{side}_topN"] = side_liquidity
        denom = liquidity["ask"] + liquidity["bid"]
        row["DWI_denominator"] = denom
        row["DWI"] = (liquidity["ask"] - liquidity["bid"]) / denom if denom > 0 else None
        rows.append(row)
    return rows


def _same_level_visible_qty(
    active_orders: Mapping[str, ActiveOrder], *, side: str, price: float, client_id: str | None = None
) -> float:
    total = 0.0
    for order in active_orders.values():
        if order.side != side or float(order.price) != float(price):
            continue
        if client_id is not None and order.client_original_id != client_id:
            continue
        total += _visible_qty(order)
    return total


def _fill_qty(event: dict[str, Any], active_order: ActiveOrder) -> float:
    for key in ("LASTSHARES", "event_last_shares"):
        value = event.get(key)
        if value is not None and float(value) > 0:
            return float(value)
    order_qty = event.get("ORDERQTY")
    if order_qty is not None and float(order_qty) > 0:
        return float(order_qty)
    return _visible_qty(active_order)


def _opposite_side(side: str) -> str:
    return "ask" if side == "bid" else "bid"


def _is_aggressive(event: dict[str, Any]) -> bool:
    return str(event.get("AGGRESSIVEORDER") or "").upper() == "Y"


def _execution_candidate_or_rejection(
    event: dict[str, Any],
    active_orders: Mapping[str, ActiveOrder],
    *,
    event_ts: datetime | None,
    partition_id: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if event["event_class"] != "fill":
        return None, None

    base = {
        "partition_id": partition_id,
        "sort_index": event["sort_index"],
        "event_ts": event_ts,
        "ORDERID": event["ORDERID"],
        "client_id": event["client_original_id"],
        "execution_side": event["side_label"],
    }

    def rejected(reason: str) -> tuple[None, dict[str, Any]]:
        return None, {**base, "reject_reason": reason}

    if event_ts is None:
        return rejected("missing_event_timestamp")
    if event["client_original_id"] is None:
        return rejected("missing_client_id")
    if event["side_label"] not in {"bid", "ask"}:
        return rejected("missing_or_invalid_side")
    if event["event_order_type_label"] not in VISIBLE_LIMIT_ORDER_TYPES:
        return rejected("non_limit_order_type")
    if _is_aggressive(event):
        return rejected("aggressive_execution")

    order_id = event["ORDERID"]
    active_order = active_orders.get(order_id)
    if active_order is None:
        return rejected("fill_order_not_active_before_execution")
    if active_order.client_original_id != event["client_original_id"]:
        return rejected("active_order_client_mismatch")
    if active_order.side != event["side_label"]:
        return rejected("active_order_side_mismatch")
    if active_order.order_type_label not in VISIBLE_LIMIT_ORDER_TYPES:
        return rejected("active_order_not_visible_limit")

    fill_qty = _fill_qty(event, active_order)
    market_qty = _same_level_visible_qty(active_orders, side=active_order.side, price=active_order.price)
    client_qty = _same_level_visible_qty(
        active_orders,
        side=active_order.side,
        price=active_order.price,
        client_id=active_order.client_original_id,
    )
    return {
        **base,
        "client_id": active_order.client_original_id,
        "execution_side": active_order.side,
        "deceptive_side": _opposite_side(active_order.side),
        "event_price": active_order.price,
        "fill_qty": fill_qty,
        "same_level_market_visible_qty_pre": market_qty,
        "same_level_client_visible_qty_pre": client_qty,
        "smallness_fraction_market_level": fill_qty / market_qty if market_qty > 0 else None,
        "smallness_fraction_client_level": fill_qty / client_qty if client_qty > 0 else None,
    }, None


def _candidate_deceptive_order_rows(
    execution: dict[str, Any],
    active_orders: Mapping[str, ActiveOrder],
    *,
    top_n: int,
    tick_size: float,
    kappa: float,
    lambda_: float,
    order_first_seen_ts: Mapping[str, datetime | None],
    max_deceptive_order_age_seconds: float = 600.0,
) -> list[dict[str, Any]]:
    deceptive_side = execution["deceptive_side"]
    levels = _market_levels(active_orders, side=deceptive_side, top_n=top_n)
    if not levels:
        return []
    price_to_rank = {price: rank for rank, (price, _) in enumerate(levels, start=1)}
    price_to_market_qty = {price: market_qty for price, market_qty in levels}
    side_meta = _side_depth_metadata(
        levels,
        side=deceptive_side,
        tick_size=tick_size,
        kappa=kappa,
        lambda_=lambda_,
    )

    rows: list[dict[str, Any]] = []
    execution_ts = _parse_ts(execution.get("event_ts"))
    for order in sorted(active_orders.values(), key=lambda item: (item.side, item.price, item.order_id)):
        qty = _visible_qty(order)
        if qty <= 0 or order.client_original_id != execution["client_id"] or order.side != deceptive_side:
            continue
        price = float(order.price)
        rank = price_to_rank.get(price)
        if rank is None:
            continue
        market_qty = price_to_market_qty[price]
        relative_depth = qty / market_qty if market_qty > 0 else 0.0
        meta = side_meta[price]
        first_seen_ts = order_first_seen_ts.get(order.order_id)
        age_seconds = (
            (execution_ts - first_seen_ts).total_seconds()
            if execution_ts is not None and first_seen_ts is not None
            else None
        )
        if age_seconds is None or age_seconds < 0 or age_seconds > max_deceptive_order_age_seconds:
            continue
        rows.append(
            {
                "partition_id": execution["partition_id"],
                "execution_sort_index": execution["sort_index"],
                "execution_ts": execution["event_ts"],
                "client_id": execution["client_id"],
                "execution_order_id": execution["ORDERID"],
                "execution_side": execution["execution_side"],
                "deceptive_side": deceptive_side,
                "top_n": top_n,
                "kappa": kappa,
                "lambda_": lambda_,
                "deceptive_order_id": order.order_id,
                "deceptive_order_price": price,
                "deceptive_order_level": rank,
                "deceptive_order_delta_ticks": meta["delta_ticks"],
                "deceptive_order_depth_distance_ticks": meta["depth_distance_ticks"],
                "deceptive_order_kernel_weight": meta["kernel_weight"],
                "deceptive_order_visible_qty_pre": qty,
                "deceptive_order_level_market_qty_pre": market_qty,
                "deceptive_order_relative_depth_pre": relative_depth,
                "deceptive_order_weighted_liquidity_contribution_pre": meta["kernel_weight"] * relative_depth,
                "deceptive_order_first_seen_sort_index": order.first_seen_sort_index,
                "deceptive_order_first_seen_ts": first_seen_ts,
                "deceptive_order_age_seconds_pre": age_seconds,
            }
        )
    return rows


def _candidate_deceptive_order_summary(candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not candidate_rows:
        return {
            "candidate_deceptive_order_count_pre": 0,
            "candidate_deceptive_visible_qty_pre": 0.0,
            "candidate_deceptive_weighted_liquidity_pre": 0.0,
            "candidate_deceptive_max_order_qty_pre": 0.0,
            "candidate_deceptive_min_delta_ticks_pre": None,
            "candidate_deceptive_mean_delta_ticks_pre": None,
            "candidate_deceptive_min_depth_distance_ticks_pre": None,
            "candidate_deceptive_mean_depth_distance_ticks_pre": None,
            "candidate_deceptive_qty_weighted_depth_distance_ticks_pre": None,
            "candidate_deceptive_max_relative_depth_pre": 0.0,
            "candidate_deceptive_min_age_seconds_pre": None,
            "candidate_deceptive_first_seen_sort_index_min": None,
            "candidate_deceptive_order_ids_pre": "",
        }

    total_qty = sum(float(row["deceptive_order_visible_qty_pre"] or 0.0) for row in candidate_rows)
    weighted_liquidity = sum(
        float(row["deceptive_order_weighted_liquidity_contribution_pre"] or 0.0) for row in candidate_rows
    )
    deltas = [float(row["deceptive_order_delta_ticks"]) for row in candidate_rows if row["deceptive_order_delta_ticks"] is not None]
    distances = [
        float(row["deceptive_order_depth_distance_ticks"])
        for row in candidate_rows
        if row["deceptive_order_depth_distance_ticks"] is not None
    ]
    ages = [
        float(row["deceptive_order_age_seconds_pre"])
        for row in candidate_rows
        if row["deceptive_order_age_seconds_pre"] is not None
    ]
    return {
        "candidate_deceptive_order_count_pre": len(candidate_rows),
        "candidate_deceptive_visible_qty_pre": total_qty,
        "candidate_deceptive_weighted_liquidity_pre": weighted_liquidity,
        "candidate_deceptive_max_order_qty_pre": max(
            float(row["deceptive_order_visible_qty_pre"] or 0.0) for row in candidate_rows
        ),
        "candidate_deceptive_min_delta_ticks_pre": min(deltas) if deltas else None,
        "candidate_deceptive_mean_delta_ticks_pre": sum(deltas) / len(deltas) if deltas else None,
        "candidate_deceptive_min_depth_distance_ticks_pre": min(distances) if distances else None,
        "candidate_deceptive_mean_depth_distance_ticks_pre": sum(distances) / len(distances) if distances else None,
        "candidate_deceptive_qty_weighted_depth_distance_ticks_pre": (
            sum(
                float(row["deceptive_order_visible_qty_pre"] or 0.0)
                * float(row["deceptive_order_depth_distance_ticks"] or 0.0)
                for row in candidate_rows
            )
            / total_qty
            if total_qty > 0
            else None
        ),
        "candidate_deceptive_max_relative_depth_pre": max(
            float(row["deceptive_order_relative_depth_pre"] or 0.0) for row in candidate_rows
        ),
        "candidate_deceptive_min_age_seconds_pre": min(ages) if ages else None,
        "candidate_deceptive_first_seen_sort_index_min": min(
            int(row["deceptive_order_first_seen_sort_index"])
            for row in candidate_rows
            if row["deceptive_order_first_seen_sort_index"] is not None
        ),
        "candidate_deceptive_order_ids_pre": ";".join(str(row["deceptive_order_id"]) for row in candidate_rows),
    }


def _direct_cancel_row(
    event: dict[str, Any],
    active_orders: Mapping[str, ActiveOrder],
    *,
    event_ts: datetime | None,
    partition_id: str | None,
) -> dict[str, Any] | None:
    if event["event_class"] != "cancel" or event_ts is None:
        return None
    order_id = event["ORDERID"]
    active_order = active_orders.get(order_id)
    if active_order is None or active_order.client_original_id is None or active_order.side not in {"bid", "ask"}:
        return None
    qty = _visible_qty(active_order)
    if qty <= 0:
        return None
    return {
        "partition_id": partition_id,
        "sort_index": event["sort_index"],
        "event_ts": event_ts,
        "client_id": active_order.client_original_id,
        "side": active_order.side,
        "ORDERID": order_id,
        "visible_qty_pre_cancel": qty,
    }


def _empty_frame() -> pl.DataFrame:
    return pl.DataFrame()


def _sync_order_first_seen_timestamps(
    active_orders: Mapping[str, ActiveOrder],
    order_first_seen_ts: dict[str, datetime | None],
    event_ts: datetime | None,
) -> None:
    active_ids = set(active_orders)
    for order_id in list(order_first_seen_ts):
        if order_id not in active_ids:
            order_first_seen_ts.pop(order_id, None)
    for order_id in active_ids:
        order_first_seen_ts.setdefault(order_id, event_ts)


def _stream_metric_inputs(
    raw_events: pl.DataFrame,
    *,
    top_n: int,
    tick_size: float,
    kappa: float,
    lambda_: float,
    max_rows: int | None = None,
    include_level_columns: bool = True,
    max_deceptive_order_age_seconds: float = 600.0,
    state_client_ids: set[str] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    config = LOBConfig(top_n=max(top_n, 1), snapshot_mode="none")
    sorted_events = sort_events(raw_events)
    if max_rows is not None:
        sorted_events = sorted_events.head(max_rows)
    events = [
        normalize_event(raw_row, sort_index=idx, config=config)
        for idx, raw_row in enumerate(sorted_events.iter_rows(named=True), start=1)
    ]

    active_orders: dict[str, ActiveOrder] = {}
    order_first_seen_ts: dict[str, datetime | None] = {}
    pending_aggressive_residuals: dict[str, tuple[dict[str, Any], tuple[Any, ...] | None]] = {}
    non_resting_order_ids: set[str] = set()
    current_partition_id: str | None = None
    state_rows: list[dict[str, Any]] = []
    execution_rows: list[dict[str, Any]] = []
    candidate_deceptive_rows: list[dict[str, Any]] = []
    direct_cancel_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []

    for event_index, event in enumerate(events):
        partition_id = _partition_id(event)
        if current_partition_id is None:
            current_partition_id = partition_id
        elif partition_id != current_partition_id:
            _flush_pending_aggressive_residuals(active_orders, pending_aggressive_residuals, keep_group=None)
            active_orders = {}
            order_first_seen_ts = {}
            pending_aggressive_residuals = {}
            non_resting_order_ids = set()
            current_partition_id = partition_id

        event_ts = choose_event_timestamp(event)
        execution, rejection = _execution_candidate_or_rejection(
            event,
            active_orders,
            event_ts=event_ts,
            partition_id=partition_id,
        )
        if execution is not None:
            execution.update({"top_n": top_n, "kappa": kappa, "lambda_": lambda_})
            candidates = _candidate_deceptive_order_rows(
                execution,
                active_orders,
                top_n=top_n,
                tick_size=tick_size,
                kappa=kappa,
                lambda_=lambda_,
                order_first_seen_ts=order_first_seen_ts,
                max_deceptive_order_age_seconds=max_deceptive_order_age_seconds,
            )
            execution.update(_candidate_deceptive_order_summary(candidates))
            execution_rows.append(execution)
            candidate_deceptive_rows.extend(candidates)
        elif rejection is not None:
            rejected_rows.append(rejection)

        direct_cancel = _direct_cancel_row(
            event,
            active_orders,
            event_ts=event_ts,
            partition_id=partition_id,
        )
        if direct_cancel is not None:
            direct_cancel_rows.append(direct_cancel)

        _apply_event(
            active_orders,
            event,
            pending_aggressive_residuals=pending_aggressive_residuals,
            non_resting_order_ids=non_resting_order_ids,
        )
        next_event = events[event_index + 1] if event_index + 1 < len(events) else None
        next_group = _fill_group_key(next_event) if next_event is not None else None
        _flush_pending_aggressive_residuals(
            active_orders,
            pending_aggressive_residuals,
            keep_group=next_group,
        )
        _sync_order_first_seen_timestamps(active_orders, order_first_seen_ts, event_ts)
        state_rows.extend(
            compute_client_top_n_exposures(
                active_orders,
                top_n=top_n,
                tick_size=tick_size,
                kappa=kappa,
                lambda_=lambda_,
                partition_id=partition_id,
                sort_index=event["sort_index"],
                event_ts=event_ts,
                include_level_columns=include_level_columns,
                client_ids=state_client_ids,
            )
        )

    state_df = pl.DataFrame(state_rows, infer_schema_length=None) if state_rows else _empty_frame()
    execution_df = pl.DataFrame(execution_rows, infer_schema_length=None) if execution_rows else _empty_frame()
    candidate_df = pl.DataFrame(candidate_deceptive_rows, infer_schema_length=None) if candidate_deceptive_rows else _empty_frame()
    cancel_df = pl.DataFrame(direct_cancel_rows, infer_schema_length=None) if direct_cancel_rows else _empty_frame()
    rejected_df = pl.DataFrame(rejected_rows, infer_schema_length=None) if rejected_rows else _empty_frame()
    return state_df, execution_df, candidate_df, cancel_df, rejected_df


def compute_client_metric_time_series(
    raw_events: pl.DataFrame,
    *,
    top_n: int,
    tick_size: float,
    kappa: float,
    lambda_: float = 1.0,
    max_rows: int | None = None,
    include_level_columns: bool = True,
    state_client_ids: set[str] | None = None,
) -> pl.DataFrame:
    state_df, _, _, _, _ = _stream_metric_inputs(
        raw_events,
        top_n=top_n,
        tick_size=tick_size,
        kappa=kappa,
        lambda_=lambda_,
        max_rows=max_rows,
        include_level_columns=include_level_columns,
        state_client_ids=state_client_ids,
    )
    return state_df


def _group_state_rows(states: pl.DataFrame) -> dict[tuple[Any, str], list[dict[str, Any]]]:
    groups: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    if states.is_empty():
        return groups
    required = {"partition_id", "client_id", "event_ts", "DWI", "L_bid_topN", "L_ask_topN"}
    if not required.issubset(states.columns):
        return groups
    optional = [
        col
        for col in ("sort_index", "market_best_bid", "market_best_ask", "market_mid", "market_microprice")
        if col in states.columns
    ]
    selected = list(required) + optional
    for row in states.select(selected).iter_rows(named=True):
        ts = _parse_ts(row["event_ts"])
        if ts is None:
            continue
        row = {**row, "sort_index": row.get("sort_index"), "_ts": ts}
        groups[(row["partition_id"], row["client_id"])].append(row)
    for values in groups.values():
        values.sort(key=lambda item: (float("inf") if item.get("sort_index") is None else int(item["sort_index"]), item["_ts"]))
    return groups


def _lookup_pre_state(values: list[dict[str, Any]], event_ts: datetime, sort_index: int | None) -> dict[str, Any] | None:
    if not values:
        return None
    if sort_index is not None:
        values_with_index = [item for item in values if item.get("sort_index") is not None]
        indexes = [int(item["sort_index"]) for item in values_with_index]
        idx = bisect.bisect_left(indexes, int(sort_index)) - 1
        return values_with_index[idx] if idx >= 0 else None
    eligible = [item for item in values if item["_ts"] < event_ts]
    return eligible[-1] if eligible else None


def _lookup_post_state(values: list[dict[str, Any]], end_ts: datetime, sort_index: int | None) -> dict[str, Any] | None:
    if not values:
        return None
    eligible = [item for item in values if item["_ts"] <= end_ts]
    if sort_index is not None:
        eligible = [item for item in eligible if item.get("sort_index") is not None and int(item["sort_index"]) > int(sort_index)]
    return eligible[-1] if eligible else None


def _lookup_state_at_or_after_index(values: list[dict[str, Any]], sort_index: int | None) -> dict[str, Any] | None:
    if sort_index is None:
        return None
    values_with_index = [item for item in values if item.get("sort_index") is not None]
    indexes = [int(item["sort_index"]) for item in values_with_index]
    idx = bisect.bisect_left(indexes, int(sort_index))
    return values_with_index[idx] if idx < len(values_with_index) else None


def _execution_price_direction(execution_side: str | None) -> float | None:
    if execution_side == "ask":
        return 1.0
    if execution_side == "bid":
        return -1.0
    return None


def _signed_change(direction: float | None, start: float | None, end: float | None) -> float | None:
    if direction is None or start is None or end is None:
        return None
    values = [float(direction), float(start), float(end)]
    if not all(math.isfinite(value) for value in values):
        return None
    return values[0] * (values[2] - values[1])


def _execution_price_advantage(direction: float | None, benchmark: float | None, event_price: float | None) -> float | None:
    if direction is None or benchmark is None or event_price is None:
        return None
    values = [float(direction), float(benchmark), float(event_price)]
    if not all(math.isfinite(value) for value in values):
        return None
    return values[0] * (values[2] - values[1])


def _collapse(pre: float | None, post: float | None, *, epsilon: float) -> float | None:
    if pre is None or post is None:
        return None
    pre_value = float(pre)
    post_value = float(post)
    if not math.isfinite(pre_value) or not math.isfinite(post_value) or pre_value + epsilon <= 0:
        return None
    return max(pre_value - post_value, 0.0) / (pre_value + epsilon)


def _finite_product_msci(sci: float | None, c_opposite: float | None, c_same: float | None) -> float | None:
    if sci is None or c_opposite is None or c_same is None:
        return None
    values = [float(sci), float(c_opposite), float(c_same)]
    if not all(math.isfinite(value) for value in values):
        return None
    return values[0] * values[1] * max(values[1] - values[2], 0.0)


def _finite_wmsci(
    *,
    candidate_qty: float,
    weighted_withdrawal_qty: float,
    fill_qty: float,
    matched_fraction: float | None,
) -> float | None:
    if fill_qty <= 0 or candidate_qty <= 0 or weighted_withdrawal_qty <= 0:
        return 0.0
    values = [candidate_qty, weighted_withdrawal_qty, fill_qty]
    if not all(math.isfinite(value) for value in values):
        return None
    fraction = max(float(matched_fraction or 0.0), 0.0)
    return math.log1p(candidate_qty / fill_qty) * math.log1p(weighted_withdrawal_qty / fill_qty) * fraction


def attach_sci_window_metrics(
    executions: pl.DataFrame,
    states: pl.DataFrame,
    *,
    window_seconds: float,
    epsilon: float = 1e-12,
) -> pl.DataFrame:
    if executions.is_empty():
        return executions
    grouped_states = _group_state_rows(states)
    rows: list[dict[str, Any]] = []
    window = timedelta(seconds=window_seconds)
    for row in executions.iter_rows(named=True):
        event_ts = _parse_ts(row.get("event_ts"))
        sort_index = row.get("sort_index")
        pre_state = None
        post_state = None
        posture_state = None
        post_target = None
        if event_ts is not None:
            post_target = event_ts + window
            values = grouped_states.get((row.get("partition_id"), row.get("client_id")), [])
            posture_state = _lookup_state_at_or_after_index(
                values,
                int(row["candidate_deceptive_first_seen_sort_index_min"])
                if row.get("candidate_deceptive_first_seen_sort_index_min") is not None
                else None,
            )
            pre_state = _lookup_pre_state(values, event_ts, int(sort_index) if sort_index is not None else None)
            post_state = _lookup_post_state(values, post_target, int(sort_index) if sort_index is not None else None)
        pre_dwi = pre_state.get("DWI") if pre_state is not None else None
        post_dwi = post_state.get("DWI") if post_state is not None else (0.0 if pre_state is not None else None)
        sci = abs(float(pre_dwi) - float(post_dwi)) if pre_dwi is not None and post_dwi is not None else None
        l_bid_pre = pre_state.get("L_bid_topN") if pre_state is not None else None
        l_bid_post = post_state.get("L_bid_topN") if post_state is not None else (0.0 if pre_state is not None else None)
        l_ask_pre = pre_state.get("L_ask_topN") if pre_state is not None else None
        l_ask_post = post_state.get("L_ask_topN") if post_state is not None else (0.0 if pre_state is not None else None)
        collapse_bid = _collapse(l_bid_pre, l_bid_post, epsilon=epsilon)
        collapse_ask = _collapse(l_ask_pre, l_ask_post, epsilon=epsilon)
        if row.get("deceptive_side") == "bid":
            collapse_opposite = collapse_bid
            collapse_same = collapse_ask
        elif row.get("deceptive_side") == "ask":
            collapse_opposite = collapse_ask
            collapse_same = collapse_bid
        else:
            collapse_opposite = None
            collapse_same = None
        price_direction = _execution_price_direction(row.get("execution_side"))
        posture_mid = posture_state.get("market_mid") if posture_state is not None else None
        pre_mid = pre_state.get("market_mid") if pre_state is not None else None
        post_mid = post_state.get("market_mid") if post_state is not None else None
        posture_microprice = posture_state.get("market_microprice") if posture_state is not None else None
        pre_microprice = pre_state.get("market_microprice") if pre_state is not None else None
        post_microprice = post_state.get("market_microprice") if post_state is not None else None
        rows.append(
            {
                **row,
                "price_response_direction": price_direction,
                "posture_state_sort_index": posture_state.get("sort_index") if posture_state is not None else None,
                "pre_state_sort_index": pre_state.get("sort_index") if pre_state is not None else None,
                "post_state_sort_index": post_state.get("sort_index") if post_state is not None else None,
                "post_target_ts": post_target,
                "market_mid_posture": posture_mid,
                "market_mid_pre_window": pre_mid,
                "market_mid_post_window": post_mid,
                "market_microprice_posture": posture_microprice,
                "market_microprice_pre_window": pre_microprice,
                "market_microprice_post_window": post_microprice,
                "favorable_mid_move_pre_fill": _signed_change(price_direction, posture_mid, pre_mid),
                "favorable_microprice_move_pre_fill": _signed_change(
                    price_direction, posture_microprice, pre_microprice
                ),
                "post_cancel_mid_reversion": _signed_change(price_direction, post_mid, pre_mid),
                "post_cancel_microprice_reversion": _signed_change(price_direction, post_microprice, pre_microprice),
                "execution_price_advantage_vs_posture_mid": _execution_price_advantage(
                    price_direction,
                    posture_mid,
                    row.get("event_price"),
                ),
                "execution_price_advantage_vs_posture_microprice": _execution_price_advantage(
                    price_direction,
                    posture_microprice,
                    row.get("event_price"),
                ),
                "DWI_pre_window": pre_dwi,
                "DWI_post_window": post_dwi,
                "SCI": sci,
                "L_bid_pre_window": l_bid_pre,
                "L_bid_post_window": l_bid_post,
                "L_ask_pre_window": l_ask_pre,
                "L_ask_post_window": l_ask_post,
                "collapse_bid": collapse_bid,
                "collapse_ask": collapse_ask,
                "collapse_opposite_side": collapse_opposite,
                "collapse_same_side": collapse_same,
                "MSCI": _finite_product_msci(sci, collapse_opposite, collapse_same),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _attach_direct_cancellation_window(
    executions: pl.DataFrame,
    cancellations: pl.DataFrame,
    *,
    window_seconds: float,
    withdrawal_decay_seconds: float = 10.0,
) -> pl.DataFrame:
    if executions.is_empty():
        return executions
    cancel_groups: dict[tuple[Any, str, str], list[dict[str, Any]]] = defaultdict(list)
    if not cancellations.is_empty():
        for row in cancellations.iter_rows(named=True):
            cancel_groups[(row["partition_id"], row["client_id"], row["side"])].append(row)
    window = timedelta(seconds=window_seconds)
    rows: list[dict[str, Any]] = []
    for row in executions.iter_rows(named=True):
        event_ts = _parse_ts(row.get("event_ts"))
        matches: list[dict[str, Any]] = []
        if event_ts is not None:
            end = event_ts + window
            candidates = cancel_groups.get((row.get("partition_id"), row.get("client_id"), row.get("deceptive_side")), [])
            matches = [
                cancel
                for cancel in candidates
                if (cancel_ts := _parse_ts(cancel.get("event_ts"))) is not None
                and event_ts < cancel_ts <= end
            ]
        candidate_order_ids = {
            order_id
            for order_id in str(row.get("candidate_deceptive_order_ids_pre") or "").split(";")
            if order_id
        }
        matched = [cancel for cancel in matches if str(cancel["ORDERID"]) in candidate_order_ids]
        order_ids = ";".join(str(cancel["ORDERID"]) for cancel in matches)
        total_qty = sum(float(cancel["visible_qty_pre_cancel"] or 0.0) for cancel in matches)
        matched_order_ids = ";".join(str(cancel["ORDERID"]) for cancel in matched)
        matched_qty = sum(float(cancel["visible_qty_pre_cancel"] or 0.0) for cancel in matched)
        candidate_qty = float(row.get("candidate_deceptive_visible_qty_pre") or 0.0)
        fill_qty = float(row.get("fill_qty") or 0.0)
        matched_fraction = matched_qty / candidate_qty if candidate_qty > 0 else None
        matched_delays: list[float] = []
        weighted_withdrawal_qty = 0.0
        if event_ts is not None:
            for cancel in matched:
                cancel_ts = _parse_ts(cancel.get("event_ts"))
                if cancel_ts is None:
                    continue
                delay = max((cancel_ts - event_ts).total_seconds(), 0.0)
                matched_delays.append(delay)
                weighted_withdrawal_qty += float(cancel["visible_qty_pre_cancel"] or 0.0) * math.exp(
                    -delay / withdrawal_decay_seconds
                )
        rows.append(
            {
                **row,
                "direct_opposite_cancel_count_window": len(matches),
                "direct_opposite_cancel_visible_qty_window": total_qty,
                "direct_opposite_cancel_order_ids_window": order_ids,
                "has_direct_opposite_cancel_window": bool(matches),
                "matched_deceptive_cancel_count_window": len(matched),
                "matched_deceptive_cancel_visible_qty_window": matched_qty,
                "matched_deceptive_cancel_order_ids_window": matched_order_ids,
                "has_matched_deceptive_cancel_window": bool(matched),
                "matched_deceptive_cancel_fraction_window": matched_fraction,
                "matched_deceptive_cancel_min_delay_seconds": min(matched_delays) if matched_delays else None,
                "matched_deceptive_cancel_max_delay_seconds": max(matched_delays) if matched_delays else None,
                "weighted_net_withdrawal_qty_window": weighted_withdrawal_qty,
                "withdrawal_to_fill_ratio": matched_qty / fill_qty if fill_qty > 0 else None,
                "weighted_withdrawal_to_fill_ratio": weighted_withdrawal_qty / fill_qty if fill_qty > 0 else None,
                "WMSCI_event": _finite_wmsci(
                    candidate_qty=candidate_qty,
                    weighted_withdrawal_qty=weighted_withdrawal_qty,
                    fill_qty=fill_qty,
                    matched_fraction=matched_fraction,
                ),
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    mid = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[mid]
    return (sorted_values[mid - 1] + sorted_values[mid]) / 2.0


def _bool_share(rows: list[dict[str, Any]], column: str) -> float | None:
    if not rows or column not in rows[0]:
        return None
    return sum(1.0 for row in rows if bool(row.get(column))) / len(rows)


def compute_mcps_scores(execution_metrics: pl.DataFrame, *, gamma_grid: list[float]) -> pl.DataFrame:
    if execution_metrics.is_empty():
        return pl.DataFrame()
    if not gamma_grid:
        raise ValueError("gamma_grid must contain at least one threshold")
    group_cols = [col for col in ("partition_id", "client_id", "top_n", "kappa", "lambda_") if col in execution_metrics.columns]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in execution_metrics.to_dicts():
        grouped[tuple(row.get(col) for col in group_cols)].append(row)

    out_rows: list[dict[str, Any]] = []
    for gamma in gamma_grid:
        for key, rows in grouped.items():
            finite_msci = [float(row["MSCI"]) for row in rows if row.get("MSCI") is not None]
            finite_sci = [float(row["SCI"]) for row in rows if row.get("SCI") is not None]
            collapse_opposite = [
                float(row["collapse_opposite_side"]) for row in rows if row.get("collapse_opposite_side") is not None
            ]
            collapse_same = [
                float(row["collapse_same_side"]) for row in rows if row.get("collapse_same_side") is not None
            ]
            favorable_mid_moves = [
                float(row["favorable_mid_move_pre_fill"])
                for row in rows
                if row.get("favorable_mid_move_pre_fill") is not None
            ]
            favorable_microprice_moves = [
                float(row["favorable_microprice_move_pre_fill"])
                for row in rows
                if row.get("favorable_microprice_move_pre_fill") is not None
            ]
            post_cancel_mid_reversions = [
                float(row["post_cancel_mid_reversion"])
                for row in rows
                if row.get("post_cancel_mid_reversion") is not None
            ]
            execution_advantages = [
                float(row["execution_price_advantage_vs_posture_mid"])
                for row in rows
                if row.get("execution_price_advantage_vs_posture_mid") is not None
            ]
            above = sum(1 for value in finite_msci if value > gamma)
            out = {col: key[idx] for idx, col in enumerate(group_cols)}
            out.update(
                {
                    "gamma": gamma,
                    "executions": len(rows),
                    "finite_msci_executions": len(finite_msci),
                    "msci_above_gamma_count": above,
                    "MCPS": above / len(rows) if rows else None,
                    "median_MSCI": _median(finite_msci),
                    "max_MSCI": max(finite_msci) if finite_msci else None,
                    "mean_MSCI": _mean(finite_msci),
                    "mean_SCI": _mean(finite_sci),
                    "mean_collapse_opposite_side": _mean(collapse_opposite),
                    "mean_collapse_same_side": _mean(collapse_same),
                    "mean_favorable_mid_move_pre_fill": _mean(favorable_mid_moves),
                    "mean_favorable_microprice_move_pre_fill": _mean(favorable_microprice_moves),
                    "mean_post_cancel_mid_reversion": _mean(post_cancel_mid_reversions),
                    "mean_execution_price_advantage_vs_posture_mid": _mean(execution_advantages),
                    "matched_deceptive_cancel_share": _bool_share(rows, "has_matched_deceptive_cancel_window"),
                    "direct_opposite_cancel_share": _bool_share(rows, "has_direct_opposite_cancel_window"),
                    "candidate_profile_share": sum(
                        1.0 for row in rows if float(row.get("candidate_deceptive_order_count_pre") or 0.0) > 0
                    )
                    / len(rows),
                }
            )
            out_rows.append(out)
    return pl.DataFrame(out_rows, infer_schema_length=None)


def compute_exploratory_metrics(
    raw_events: pl.DataFrame,
    *,
    top_n: int,
    tick_size: float,
    kappa: float,
    window_seconds: float,
    lambda_: float = 1.0,
    epsilon: float = 1e-12,
    max_rows: int | None = None,
    include_level_columns: bool = True,
    max_deceptive_order_age_seconds: float = 600.0,
    state_client_ids: set[str] | None = None,
) -> ExploratoryMetricsResult:
    state_df, execution_df, candidate_df, cancel_df, rejected_df = _stream_metric_inputs(
        raw_events,
        top_n=top_n,
        tick_size=tick_size,
        kappa=kappa,
        lambda_=lambda_,
        max_rows=max_rows,
        include_level_columns=include_level_columns,
        max_deceptive_order_age_seconds=max_deceptive_order_age_seconds,
        state_client_ids=state_client_ids,
    )
    execution_df = attach_sci_window_metrics(
        execution_df,
        state_df,
        window_seconds=window_seconds,
        epsilon=epsilon,
    )
    execution_df = _attach_direct_cancellation_window(
        execution_df,
        cancel_df,
        window_seconds=window_seconds,
    )
    return ExploratoryMetricsResult(
        state_time_series=state_df,
        execution_metrics=execution_df,
        candidate_deceptive_orders=candidate_df,
        direct_cancellations=cancel_df,
        rejected_executions=rejected_df,
    )