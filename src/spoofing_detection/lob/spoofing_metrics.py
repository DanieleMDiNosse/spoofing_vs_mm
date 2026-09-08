from __future__ import annotations

import bisect
import math
from collections import defaultdict
from collections.abc import Collection, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

import polars as pl

from spoofing_detection.lob.actor_identity import (
    ActorIdentity,
    actor_identity_from_event,
    actor_identity_from_order,
    same_actor,
)
from spoofing_detection.lob.behavioral_gate import attach_spoofing_compatible_sequence_gate
from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.execution_clusters import (
    classify_execution_anchor,
    cluster_execution_fills,
)
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
MSCI_DEFINITION = "SCI / 2 + C_opposite - C_same"
MSCI_RESTING_PROFILE_DEFINITION = MSCI_DEFINITION
WITHDRAWAL_PROFILE_SCALE_DEFINITION = (
    "log1p(candidate_deceptive_visible_qty_pre / execution_quantity) * "
    "log1p(weighted_net_withdrawal_qty_window / execution_quantity) * "
    "matched_deceptive_cancel_fraction_window"
)
MSCI_RANGE = (-1.0, 2.0)
RATIO_ZERO_DENOMINATOR_POLICY = (
    "exact_piecewise_v1: collapse=0 when pre_liquidity=0; "
    "DWI=0 only for explicit zero-profile states; other undefined ratios=null"
)

ACTOR_IDENTITY_SCHEMA: dict[str, pl.DataType] = {
    "actor_key": pl.String,
    "actor_id": pl.String,
    "identity_level": pl.String,
    "identity_source": pl.String,
    "identity_fallback_flag": pl.Boolean,
}
EXECUTION_EMPTY_SCHEMA: dict[str, pl.DataType] = {
    "partition_id": pl.String,
    "sort_index": pl.Int64,
    "event_ts": pl.Datetime("us"),
    **ACTOR_IDENTITY_SCHEMA,
    "execution_anchor_mode": pl.String,
    "execution_cluster_id": pl.String,
    "execution_side": pl.String,
    "fill_qty": pl.Float64,
    "execution_quantity": pl.Float64,
    "execution_vwap": pl.Float64,
    "passive_execution_diagnostics_applicable": pl.Boolean,
    "aggressive_execution_diagnostics_applicable": pl.Boolean,
    "passive_execution_quantity": pl.Float64,
    "aggressive_execution_quantity": pl.Float64,
    "MSCI": pl.Float64,
    "MSCI_resting_profile": pl.Float64,
    "withdrawal_profile_scale_event": pl.Float64,
    "WMSCI_passive": pl.Float64,
    "WMSCI_aggressive": pl.Float64,
}
CANDIDATE_DECEPTIVE_ORDER_EMPTY_SCHEMA: dict[str, pl.DataType] = {
    "partition_id": pl.String,
    "execution_sort_index": pl.Int64,
    "execution_ts": pl.Datetime("us"),
    **ACTOR_IDENTITY_SCHEMA,
    "execution_anchor_mode": pl.String,
    "execution_cluster_id": pl.String,
    "deceptive_order_id": pl.String,
}
EXECUTION_CLUSTER_MEMBER_EMPTY_SCHEMA: dict[str, pl.DataType] = {
    "execution_cluster_id": pl.String,
    "partition_id": pl.String,
    **ACTOR_IDENTITY_SCHEMA,
    "execution_anchor_mode": pl.String,
    "child_order_id": pl.String,
    "child_sort_index": pl.Int64,
    "child_fill_qty": pl.Float64,
    "child_fill_price": pl.Float64,
}
DIRECT_CANCELLATION_EMPTY_SCHEMA: dict[str, pl.DataType] = {
    "partition_id": pl.String,
    "sort_index": pl.Int64,
    "event_ts": pl.Datetime("us"),
    **ACTOR_IDENTITY_SCHEMA,
    "ORDERID": pl.String,
    "visible_qty_pre_cancel": pl.Float64,
}

MCPS_SCORE_SCHEMA: dict[str, pl.DataType] = {
    "partition_id": pl.String,
    "actor_key": pl.String,
    "actor_id": pl.String,
    "identity_level": pl.String,
    "identity_source": pl.String,
    "identity_fallback_flag": pl.Boolean,
    "execution_anchor_mode": pl.String,
    "top_n": pl.Int64,
    "gamma": pl.Float64,
    "executions": pl.Int64,
    "finite_msci_executions": pl.Int64,
    "msci_above_gamma_count": pl.Int64,
    "MCPS": pl.Float64,
    "MCPS_resting_profile": pl.Float64,
    "median_MSCI": pl.Float64,
    "max_MSCI": pl.Float64,
    "mean_MSCI": pl.Float64,
    "median_MSCI_resting_profile": pl.Float64,
    "max_MSCI_resting_profile": pl.Float64,
    "mean_MSCI_resting_profile": pl.Float64,
    "mean_SCI": pl.Float64,
    "mean_collapse_opposite_side": pl.Float64,
    "mean_collapse_same_side": pl.Float64,
    "mean_favorable_mid_move_pre_fill": pl.Float64,
    "mean_favorable_microprice_move_pre_fill": pl.Float64,
    "mean_post_cancel_mid_reversion": pl.Float64,
    "mean_execution_price_advantage_vs_posture_mid": pl.Float64,
    "matched_deceptive_cancel_share": pl.Float64,
    "direct_opposite_cancel_share": pl.Float64,
    "candidate_profile_share": pl.Float64,
}


def _state_metric_empty_schema(*, top_n: int, include_level_columns: bool) -> dict[str, pl.DataType]:
    schema: dict[str, pl.DataType] = {
        "partition_id": pl.String,
        "sort_index": pl.Int64,
        "event_ts": pl.Datetime("us"),
        **ACTOR_IDENTITY_SCHEMA,
        "has_active_top_n_profile": pl.Boolean,
        "top_n": pl.Int64,
        "tick_size": pl.Float64,
        "market_best_bid": pl.Float64,
        "market_best_ask": pl.Float64,
        "market_mid": pl.Float64,
        "market_microprice": pl.Float64,
    }
    for side in ("bid", "ask"):
        if include_level_columns:
            for rank in range(1, top_n + 1):
                prefix = f"{side}_level_{rank}"
                schema.update(
                    {
                        f"{prefix}_price": pl.Float64,
                        f"{prefix}_market_visible_qty": pl.Float64,
                        f"{prefix}_actor_visible_qty": pl.Float64,
                        f"{prefix}_actor_fraction": pl.Float64,
                        f"{prefix}_actor_relative_depth": pl.Float64,
                        f"{prefix}_delta_ticks": pl.Float64,
                        f"{prefix}_depth_distance_ticks": pl.Float64,
                        f"{prefix}_kernel_weight": pl.Float64,
                        f"{prefix}_weighted_liquidity_contribution": pl.Float64,
                    }
                )
        schema.update(
            {
                f"actor_{side}_qty_topN": pl.Float64,
                f"market_{side}_qty_topN": pl.Float64,
                f"raw_{side}_fraction_topN": pl.Float64,
                f"L_{side}_topN": pl.Float64,
            }
        )
    schema.update({"DWI_denominator": pl.Float64, "DWI": pl.Float64})
    return schema


_STATE_ROW_CHUNK_SIZE = 50_000


REJECTED_EXECUTION_SCHEMA: dict[str, pl.DataType] = {
    "partition_id": pl.String,
    "sort_index": pl.Int64,
    "event_ts": pl.Datetime("us"),
    "ORDERID": pl.String,
    "client_original_id": pl.String,
    "firm_id": pl.String,
    "execution_side": pl.String,
    "actor_key": pl.String,
    "actor_id": pl.String,
    "identity_level": pl.String,
    "identity_source": pl.String,
    "identity_fallback_flag": pl.Boolean,
    "execution_anchor_mode": pl.String,
    "reject_reason": pl.String,
}

EXECUTION_CANCEL_CANDIDATE_SCHEMA: dict[str, pl.DataType] = {
    "partition_id": pl.String,
    "cancel_sort_index": pl.Int64,
    "candidate_order_id": pl.String,
    "execution_cluster_id": pl.String,
    "actor_key": pl.String,
    "actor_id": pl.String,
    "identity_level": pl.String,
    "identity_source": pl.String,
    "identity_fallback_flag": pl.Boolean,
    "execution_anchor_mode": pl.String,
    "execution_side": pl.String,
    "deceptive_side": pl.String,
    "cluster_end_ts": pl.Datetime("us"),
    "cluster_first_sort_index": pl.Int64,
    "cluster_last_sort_index": pl.Int64,
    "cancel_event_ts": pl.Datetime("us"),
    "cancel_visible_qty": pl.Float64,
    "candidate_visible_qty_pre": pl.Float64,
    "attributed_cancel_visible_qty": pl.Float64,
    "ORDERID": pl.String,
    "event_ts": pl.Datetime("us"),
    "visible_qty_pre_cancel": pl.Float64,
    "assigned_flag": pl.Boolean,
    "assignment_rule": pl.String,
    "competing_cluster_count": pl.Int64,
    "cancel_reversion_target_ts": pl.Datetime("us"),
    "cancel_pre_state_sort_index": pl.Int64,
    "cancel_post_state_sort_index": pl.Int64,
    "cancel_mid_pre": pl.Float64,
    "cancel_mid_post_horizon": pl.Float64,
    "cancel_microprice_pre": pl.Float64,
    "cancel_microprice_post_horizon": pl.Float64,
    "post_cancel_mid_reversion": pl.Float64,
    "post_cancel_microprice_reversion": pl.Float64,
    "cancel_reversion_weight": pl.Float64,
    "has_cancel_reversion_state": pl.Boolean,
}


@dataclass(frozen=True)
class ExploratoryMetricsResult:
    state_time_series: pl.DataFrame
    execution_metrics: pl.DataFrame
    candidate_deceptive_orders: pl.DataFrame
    direct_cancellations: pl.DataFrame
    rejected_executions: pl.DataFrame
    execution_cluster_members: pl.DataFrame
    execution_cancel_candidates: pl.DataFrame
    spoofing_compatible_events: pl.DataFrame


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


def _validate_empirical_kernel_weights(
    empirical_kernel_weights: Mapping[str, Mapping[int, float]] | None,
    *,
    top_n: int,
) -> None:
    if empirical_kernel_weights is None:
        raise ValueError("empirical depth kernel weights are required")
    missing_sides = {"bid", "ask"} - set(empirical_kernel_weights)
    if missing_sides:
        raise ValueError(
            "empirical depth kernel must provide bid and ask weights; "
            f"missing: {', '.join(sorted(missing_sides))}"
        )
    required_ranks = set(range(1, top_n + 1))
    for side in ("bid", "ask"):
        side_weights = empirical_kernel_weights[side]
        missing_ranks = sorted(required_ranks - set(side_weights))
        if missing_ranks:
            raise ValueError(
                f"empirical depth kernel for side {side!r} is missing ranks: "
                f"{', '.join(map(str, missing_ranks))}"
            )
        values = [float(side_weights[rank]) for rank in range(1, top_n + 1)]
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError("empirical depth kernel weights must be finite and non-negative")
        if sum(values) <= 0:
            raise ValueError(f"empirical depth kernel has no positive weights for side {side!r}")


def _side_depth_metadata(
    levels: list[tuple[float, float]],
    *,
    side: str,
    tick_size: float,
    empirical_weights_by_rank: Mapping[int, float],
) -> dict[float, dict[str, float]]:
    if not levels:
        return {}
    best_price = levels[0][0]
    distances = [shifted_depth_distance_ticks(side, price, best_price, tick_size) for price, _ in levels]
    weights = [float(empirical_weights_by_rank.get(rank, 0.0)) for rank in range(1, len(levels) + 1)]
    if any(not math.isfinite(value) or value < 0 for value in weights):
        raise ValueError("empirical depth kernel weights must be finite and non-negative")
    total = sum(weights)
    if total <= 0:
        raise ValueError(f"empirical depth kernel has no positive weights for side {side!r}")
    weights = [value / total for value in weights]
    return {
        price: {
            "delta_ticks": _distance_ticks(side, price, best_price, tick_size),
            "depth_distance_ticks": distances[idx],
            "kernel_weight": weights[idx],
        }
        for idx, (price, _) in enumerate(levels)
    }


def compute_actor_top_n_exposures(
    active_orders: Mapping[str, ActiveOrder],
    *,
    top_n: int,
    tick_size: float,
    partition_id: str | None,
    sort_index: int,
    event_ts: datetime | None,
    include_level_columns: bool = True,
    actor_keys: set[str] | None = None,
    include_zero_actor_keys: set[str] | None = None,
    empirical_kernel_weights: Mapping[str, Mapping[int, float]] | None = None,
) -> list[dict[str, Any]]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")
    _validate_empirical_kernel_weights(empirical_kernel_weights, top_n=top_n)
    assert empirical_kernel_weights is not None

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
        side: _side_depth_metadata(
            side_levels,
            side=side,
            tick_size=tick_size,
            empirical_weights_by_rank=empirical_kernel_weights[side],
        )
        for side, side_levels in levels.items()
    }

    actor_level_qty: dict[tuple[str, str, int], float] = defaultdict(float)
    active_identities: dict[str, ActorIdentity] = {}
    for order in active_orders.values():
        identity = actor_identity_from_order(order)
        qty = _visible_qty(order)
        if identity is None or qty <= 0 or order.side not in {"bid", "ask"}:
            continue
        if actor_keys is not None and identity.actor_key not in actor_keys:
            continue
        rank = price_to_rank[order.side].get(float(order.price))
        if rank is None:
            continue
        active_identities[identity.actor_key] = identity
        actor_level_qty[(identity.actor_key, order.side, rank)] += qty

    profile_actor_keys = set(active_identities)
    zero_actor_keys = set(include_zero_actor_keys or ()) - profile_actor_keys
    if actor_keys is not None:
        zero_actor_keys.intersection_update(actor_keys)

    rows: list[dict[str, Any]] = []
    for actor_key in sorted(profile_actor_keys | zero_actor_keys):
        identity = active_identities.get(actor_key) or _actor_identity_from_key(actor_key)
        if identity is None:
            continue
        row: dict[str, Any] = {
            "partition_id": partition_id,
            "sort_index": sort_index,
            "event_ts": event_ts,
            "actor_key": identity.actor_key,
            "actor_id": identity.actor_id,
            "identity_level": identity.identity_level,
            "identity_source": identity.identity_source,
            "identity_fallback_flag": identity.identity_fallback_flag,
            "has_active_top_n_profile": actor_key in profile_actor_keys,
            "top_n": top_n,
            "tick_size": tick_size,
            "market_best_bid": best_bid,
            "market_best_ask": best_ask,
            "market_mid": market_mid,
            "market_microprice": market_microprice,
        }
        liquidity: dict[str, float] = {}
        for side in ("bid", "ask"):
            actor_side_qty = 0.0
            side_liquidity = 0.0
            for rank in range(1, top_n + 1):
                if rank <= len(levels[side]):
                    price, market_qty = levels[side][rank - 1]
                    actor_qty = actor_level_qty[(actor_key, side, rank)]
                    meta = side_meta[side][price]
                    relative_depth = actor_qty / market_qty if market_qty > 0 else 0.0
                    contribution = meta["kernel_weight"] * relative_depth
                    delta_ticks = meta["delta_ticks"]
                    depth_distance = meta["depth_distance_ticks"]
                    kernel_weight = meta["kernel_weight"]
                else:
                    price = None
                    market_qty = 0.0
                    actor_qty = 0.0
                    relative_depth = 0.0
                    contribution = 0.0
                    delta_ticks = None
                    depth_distance = None
                    kernel_weight = 0.0
                actor_side_qty += actor_qty
                side_liquidity += contribution
                if include_level_columns:
                    row[f"{side}_level_{rank}_price"] = price
                    row[f"{side}_level_{rank}_market_visible_qty"] = market_qty
                    row[f"{side}_level_{rank}_actor_visible_qty"] = actor_qty
                    row[f"{side}_level_{rank}_actor_fraction"] = actor_qty / market_qty if market_qty > 0 else 0.0
                    row[f"{side}_level_{rank}_actor_relative_depth"] = relative_depth
                    row[f"{side}_level_{rank}_delta_ticks"] = delta_ticks
                    row[f"{side}_level_{rank}_depth_distance_ticks"] = depth_distance
                    row[f"{side}_level_{rank}_kernel_weight"] = kernel_weight
                    row[f"{side}_level_{rank}_weighted_liquidity_contribution"] = contribution
            liquidity[side] = side_liquidity
            row[f"actor_{side}_qty_topN"] = actor_side_qty
            row[f"market_{side}_qty_topN"] = market_total[side]
            row[f"raw_{side}_fraction_topN"] = actor_side_qty / market_total[side] if market_total[side] > 0 else 0.0
            row[f"L_{side}_topN"] = side_liquidity
        denom = liquidity["ask"] + liquidity["bid"]
        row["DWI_denominator"] = denom
        row["DWI"] = (
            (liquidity["ask"] - liquidity["bid"]) / denom
            if denom > 0
            else (0.0 if actor_key in zero_actor_keys else None)
        )
        rows.append(row)
    return rows


def _actor_identity_from_key(actor_key: str) -> ActorIdentity | None:
    prefix, separator, actor_id = str(actor_key).partition(":")
    if not separator or not actor_id:
        return None
    if prefix == "client_original":
        return ActorIdentity(
            actor_key=actor_key,
            actor_id=actor_id,
            identity_level="client_original",
            identity_source="NMSC_ORIGINALCLIENTIDSHORTCODE",
            identity_fallback_flag=False,
        )
    if prefix == "firm":
        return ActorIdentity(
            actor_key=actor_key,
            actor_id=actor_id,
            identity_level="firm",
            identity_source="FIRMID",
            identity_fallback_flag=True,
        )
    return None


def compute_client_top_n_exposures(
    active_orders: Mapping[str, ActiveOrder],
    *,
    top_n: int,
    tick_size: float,
    partition_id: str | None,
    sort_index: int,
    event_ts: datetime | None,
    include_level_columns: bool = True,
    client_ids: set[str] | None = None,
    include_zero_client_ids: set[str] | None = None,
    empirical_kernel_weights: Mapping[str, Mapping[int, float]] | None = None,
) -> list[dict[str, Any]]:
    """Compatibility adapter for client-only callers; official outputs are actor-aware."""
    available_client_keys = {
        identity.actor_key
        for order in active_orders.values()
        if (identity := actor_identity_from_order(order)) is not None
        and identity.identity_level == "client_original"
    }
    requested_keys = (
        {f"client_original:{client_id}" for client_id in client_ids}
        if client_ids is not None
        else available_client_keys
    )
    zero_keys = {f"client_original:{client_id}" for client_id in include_zero_client_ids or ()}
    actor_rows = compute_actor_top_n_exposures(
        active_orders,
        top_n=top_n,
        tick_size=tick_size,
        partition_id=partition_id,
        sort_index=sort_index,
        event_ts=event_ts,
        include_level_columns=include_level_columns,
        actor_keys=requested_keys,
        include_zero_actor_keys=zero_keys,
        empirical_kernel_weights=empirical_kernel_weights,
    )
    rows: list[dict[str, Any]] = []
    for actor_row in actor_rows:
        row: dict[str, Any] = {}
        for key, value in actor_row.items():
            if key in {
                "actor_key",
                "actor_id",
                "identity_level",
                "identity_source",
                "identity_fallback_flag",
            }:
                continue
            row[key.replace("_actor_", "_client_").replace("actor_", "client_", 1)] = value
        row["client_id"] = actor_row["actor_id"]
        rows.append(row)
    return rows


def _same_level_visible_qty(
    active_orders: Mapping[str, ActiveOrder], *, side: str, price: float, actor_key: str | None = None
) -> float:
    total = 0.0
    for order in active_orders.values():
        if order.side != side or float(order.price) != float(price):
            continue
        identity = actor_identity_from_order(order)
        if actor_key is not None and (identity is None or identity.actor_key != actor_key):
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


def _positive_trade_value(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) and parsed > 0 else None


def _execution_candidate_or_rejection(
    event: dict[str, Any],
    active_orders: Mapping[str, ActiveOrder],
    *,
    event_ts: datetime | None,
    partition_id: str | None,
    allowed_anchor_modes: Collection[str] = ("passive", "aggressive"),
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    if event["event_class"] != "fill":
        return None, None

    actor = actor_identity_from_event(event)
    base: dict[str, Any] = {
        "partition_id": partition_id,
        "sort_index": event["sort_index"],
        "event_ts": event_ts,
        "ORDERID": event["ORDERID"],
        "client_original_id": event.get("client_original_id"),
        "firm_id": event.get("firm_id"),
        "execution_side": event["side_label"],
    }
    if actor is not None:
        base.update(
            {
                "actor_key": actor.actor_key,
                "actor_id": actor.actor_id,
                "identity_level": actor.identity_level,
                "identity_source": actor.identity_source,
                "identity_fallback_flag": actor.identity_fallback_flag,
            }
        )

    def rejected(reason: str) -> tuple[None, dict[str, Any]]:
        return None, {**base, "reject_reason": reason}

    if event_ts is None:
        return rejected("missing_event_timestamp")
    if actor is None:
        return rejected("missing_actor_identity")
    if event["side_label"] not in {"bid", "ask"}:
        return rejected("missing_or_invalid_side")

    allowed_modes = {str(mode).strip().lower() for mode in allowed_anchor_modes}
    unknown_modes = allowed_modes - {"passive", "aggressive"}
    if unknown_modes:
        raise ValueError(f"unsupported execution anchor modes: {sorted(unknown_modes)}")
    anchor_mode = classify_execution_anchor(event)
    if anchor_mode is None:
        passive_flag = str(event.get("PASSIVEORDER") or "").strip().upper() == "Y"
        aggressive_flag = str(event.get("AGGRESSIVEORDER") or "").strip().upper() == "Y"
        reason = "ambiguous_execution_role" if passive_flag and aggressive_flag else "missing_execution_role"
        return rejected(reason)
    if anchor_mode not in allowed_modes:
        return rejected(f"execution_anchor_mode_disabled:{anchor_mode}")
    base["execution_anchor_mode"] = anchor_mode

    if anchor_mode == "aggressive":
        fill_qty = _positive_trade_value(event.get("LASTSHARES"))
        if fill_qty is None:
            return rejected("missing_or_invalid_last_shares")
        execution_price = _positive_trade_value(event.get("LASTTRADEDPX"))
        if execution_price is None:
            return rejected("missing_or_invalid_last_traded_price")
        return {
            **base,
            "deceptive_side": _opposite_side(event["side_label"]),
            "event_price": execution_price,
            "fill_qty": fill_qty,
            "execution_price_source": "LASTTRADEDPX",
            "same_level_market_visible_qty_pre": None,
            "same_level_actor_visible_qty_pre": None,
            "smallness_fraction_market_level": None,
            "smallness_fraction_actor_level": None,
        }, None

    if event["event_order_type_label"] not in VISIBLE_LIMIT_ORDER_TYPES:
        return rejected("non_limit_order_type")

    order_id = event["ORDERID"]
    active_order = active_orders.get(order_id)
    if active_order is None:
        return rejected("fill_order_not_active_before_execution")
    active_actor = actor_identity_from_order(active_order)
    if not same_actor(actor, active_actor):
        return rejected("active_order_actor_mismatch")
    if active_order.side != event["side_label"]:
        return rejected("active_order_side_mismatch")
    if active_order.order_type_label not in VISIBLE_LIMIT_ORDER_TYPES:
        return rejected("active_order_not_visible_limit")

    fill_qty = _fill_qty(event, active_order)
    market_qty = _same_level_visible_qty(active_orders, side=active_order.side, price=active_order.price)
    actor_qty = _same_level_visible_qty(
        active_orders,
        side=active_order.side,
        price=active_order.price,
        actor_key=actor.actor_key,
    )
    return {
        **base,
        "execution_side": active_order.side,
        "deceptive_side": _opposite_side(active_order.side),
        "event_price": active_order.price,
        "fill_qty": fill_qty,
        "execution_price_source": "active_order_price",
        "same_level_market_visible_qty_pre": market_qty,
        "same_level_actor_visible_qty_pre": actor_qty,
        "smallness_fraction_market_level": fill_qty / market_qty if market_qty > 0 else None,
        "smallness_fraction_actor_level": fill_qty / actor_qty if actor_qty > 0 else None,
    }, None


def _candidate_deceptive_order_rows(
    execution: dict[str, Any],
    active_orders: Mapping[str, ActiveOrder],
    *,
    top_n: int,
    tick_size: float,
    order_first_seen_ts: Mapping[str, datetime | None],
    max_deceptive_order_age_seconds: float = 600.0,
    empirical_kernel_weights: Mapping[str, Mapping[int, float]] | None = None,
) -> list[dict[str, Any]]:
    if empirical_kernel_weights is None:
        raise ValueError("empirical depth kernel weights are required")
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
        empirical_weights_by_rank=empirical_kernel_weights[deceptive_side],
    )

    rows: list[dict[str, Any]] = []
    execution_ts = _parse_ts(execution.get("event_ts"))
    for order in sorted(active_orders.values(), key=lambda item: (item.side, item.price, item.order_id)):
        qty = _visible_qty(order)
        order_actor = actor_identity_from_order(order)
        if (
            qty <= 0
            or order_actor is None
            or order_actor.actor_key != execution.get("actor_key")
            or order.side != deceptive_side
        ):
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
                "actor_key": execution["actor_key"],
                "actor_id": execution["actor_id"],
                "identity_level": execution["identity_level"],
                "identity_source": execution["identity_source"],
                "identity_fallback_flag": execution["identity_fallback_flag"],
                "execution_anchor_mode": execution["execution_anchor_mode"],
                "client_original_id": order.client_original_id,
                "firm_id": order.firm_id,
                "execution_order_id": execution["ORDERID"],
                "execution_side": execution["execution_side"],
                "deceptive_side": deceptive_side,
                "top_n": top_n,
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
    if active_order is None or active_order.side not in {"bid", "ask"}:
        return None
    actor = actor_identity_from_order(active_order)
    if actor is None:
        return None
    qty = _visible_qty(active_order)
    if qty <= 0:
        return None
    return {
        "partition_id": partition_id,
        "sort_index": event["sort_index"],
        "event_ts": event_ts,
        "actor_key": actor.actor_key,
        "actor_id": actor.actor_id,
        "identity_level": actor.identity_level,
        "identity_source": actor.identity_source,
        "identity_fallback_flag": actor.identity_fallback_flag,
        "client_original_id": active_order.client_original_id,
        "firm_id": active_order.firm_id,
        "side": active_order.side,
        "ORDERID": order_id,
        "visible_qty_pre_cancel": qty,
    }


def _coerce_frame_schema(frame: pl.DataFrame, schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=schema)
    missing = [
        pl.lit(None).cast(dtype).alias(column)
        for column, dtype in schema.items()
        if column not in frame.columns
    ]
    if missing:
        frame = frame.with_columns(missing)
    return frame.select(
        pl.col(column).cast(dtype, strict=False)
        for column, dtype in schema.items()
    )


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
    execution_cluster_max_gap_ms: int = 100,
    max_rows: int | None = None,
    include_level_columns: bool = True,
    max_deceptive_order_age_seconds: float = 600.0,
    state_actor_keys: set[str] | None = None,
    state_client_ids: set[str] | None = None,
    execution_anchor_modes: Collection[str] = ("passive", "aggressive"),
    empirical_kernel_weights: Mapping[str, Mapping[int, float]] | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
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
    if state_actor_keys is not None and state_client_ids is not None:
        raise ValueError("state_actor_keys and state_client_ids are mutually exclusive")
    selected_actor_keys = (
        state_actor_keys
        if state_actor_keys is not None
        else (
            {f"client_original:{client_id}" for client_id in state_client_ids}
            if state_client_ids is not None
            else None
        )
    )
    state_rows: list[dict[str, Any]] = []
    state_chunks: list[pl.DataFrame] = []
    state_schema = _state_metric_empty_schema(
        top_n=top_n,
        include_level_columns=include_level_columns,
    )
    previous_profile_actor_keys: set[str] = set()
    execution_rows: list[dict[str, Any]] = []
    candidate_deceptive_rows: list[dict[str, Any]] = []
    candidate_rows_by_execution_sort_index: dict[int, list[dict[str, Any]]] = {}
    direct_cancel_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    cluster_source_rows: list[dict[str, Any]] = []

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
            previous_profile_actor_keys = set()
            current_partition_id = partition_id

        event_ts = choose_event_timestamp(event)
        execution, rejection = _execution_candidate_or_rejection(
            event,
            active_orders,
            event_ts=event_ts,
            partition_id=partition_id,
            allowed_anchor_modes=execution_anchor_modes,
        )
        if execution is not None:
            execution.update({"top_n": top_n})
            candidates = _candidate_deceptive_order_rows(
                execution,
                active_orders,
                top_n=top_n,
                tick_size=tick_size,
                order_first_seen_ts=order_first_seen_ts,
                max_deceptive_order_age_seconds=max_deceptive_order_age_seconds,
                empirical_kernel_weights=empirical_kernel_weights,
            )
            execution.update(_candidate_deceptive_order_summary(candidates))
            execution_rows.append(execution)
            candidate_deceptive_rows.extend(candidates)
            candidate_rows_by_execution_sort_index[int(execution["sort_index"])] = candidates
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

        cluster_source = {
            **event,
            "partition_id": partition_id,
            "event_ts": event_ts,
            "is_passive_fill": execution is not None and execution.get("execution_anchor_mode") == "passive",
            "is_execution_fill": execution is not None,
            "event_price": execution.get("event_price") if execution is not None else event.get("ORDERPX"),
            "fill_qty": execution.get("fill_qty") if execution is not None else None,
            "metric_row": execution,
        }
        if execution is not None:
            cluster_source.update(execution)
        cluster_source_rows.append(cluster_source)

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
        current_execution_actor_keys = {
            str(execution["actor_key"])
            for execution in (execution,)
            if execution is not None
            and execution.get("actor_key") is not None
            and (selected_actor_keys is None or str(execution["actor_key"]) in selected_actor_keys)
        }
        exposure_rows = compute_actor_top_n_exposures(
            active_orders,
            top_n=top_n,
            tick_size=tick_size,
            partition_id=partition_id,
            sort_index=event["sort_index"],
            event_ts=event_ts,
            include_level_columns=include_level_columns,
            actor_keys=selected_actor_keys,
            include_zero_actor_keys=previous_profile_actor_keys | current_execution_actor_keys,
            empirical_kernel_weights=empirical_kernel_weights,
        )
        state_rows.extend(exposure_rows)
        if len(state_rows) >= _STATE_ROW_CHUNK_SIZE:
            state_chunks.append(pl.DataFrame(state_rows, schema=state_schema, strict=False))
            state_rows = []
        previous_profile_actor_keys = {
            str(row["actor_key"]) for row in exposure_rows if row["has_active_top_n_profile"]
        }

    cluster_rows, member_rows = cluster_execution_fills(
        cluster_source_rows,
        max_gap_ms=execution_cluster_max_gap_ms,
        allowed_anchor_modes=execution_anchor_modes,
    )
    clustered_candidate_rows: list[dict[str, Any]] = []
    for cluster in cluster_rows:
        first_sort_index = int(cluster["cluster_first_sort_index"])
        for candidate in candidate_rows_by_execution_sort_index.get(first_sort_index, []):
            clustered_candidate_rows.append(
                {
                    **candidate,
                    "execution_cluster_id": cluster["execution_cluster_id"],
                    "cluster_first_sort_index": cluster["cluster_first_sort_index"],
                    "cluster_last_sort_index": cluster["cluster_last_sort_index"],
                }
            )

    if state_rows:
        state_chunks.append(pl.DataFrame(state_rows, schema=state_schema, strict=False))
    state_df = (
        pl.concat(state_chunks, how="vertical", rechunk=False)
        if state_chunks
        else pl.DataFrame(schema=state_schema)
    )
    execution_df = (
        pl.DataFrame(cluster_rows, infer_schema_length=None)
        if cluster_rows
        else pl.DataFrame(schema=EXECUTION_EMPTY_SCHEMA)
    )
    candidate_df = (
        pl.DataFrame(clustered_candidate_rows, infer_schema_length=None)
        if clustered_candidate_rows
        else pl.DataFrame(schema=CANDIDATE_DECEPTIVE_ORDER_EMPTY_SCHEMA)
    )
    cancel_df = (
        pl.DataFrame(direct_cancel_rows, infer_schema_length=None)
        if direct_cancel_rows
        else pl.DataFrame(schema=DIRECT_CANCELLATION_EMPTY_SCHEMA)
    )
    rejected_df = _coerce_frame_schema(
        pl.DataFrame(rejected_rows, infer_schema_length=None) if rejected_rows else pl.DataFrame(),
        REJECTED_EXECUTION_SCHEMA,
    )
    member_df = (
        pl.DataFrame(member_rows, infer_schema_length=None)
        if member_rows
        else pl.DataFrame(schema=EXECUTION_CLUSTER_MEMBER_EMPTY_SCHEMA)
    )
    return state_df, execution_df, candidate_df, cancel_df, rejected_df, member_df


def compute_actor_metric_time_series(
    raw_events: pl.DataFrame,
    *,
    top_n: int,
    tick_size: float,
    max_rows: int | None = None,
    include_level_columns: bool = True,
    state_actor_keys: set[str] | None = None,
    execution_anchor_modes: Collection[str] = ("passive", "aggressive"),
    empirical_kernel_weights: Mapping[str, Mapping[int, float]] | None = None,
) -> pl.DataFrame:
    state_df, _, _, _, _, _ = _stream_metric_inputs(
        raw_events,
        top_n=top_n,
        tick_size=tick_size,
        max_rows=max_rows,
        include_level_columns=include_level_columns,
        state_actor_keys=state_actor_keys,
        execution_anchor_modes=execution_anchor_modes,
        empirical_kernel_weights=empirical_kernel_weights,
    )
    return state_df


def compute_client_metric_time_series(
    raw_events: pl.DataFrame,
    *,
    top_n: int,
    tick_size: float,
    max_rows: int | None = None,
    include_level_columns: bool = True,
    state_client_ids: set[str] | None = None,
    empirical_kernel_weights: Mapping[str, Mapping[int, float]] | None = None,
) -> pl.DataFrame:
    """Compatibility adapter for legacy client-only time-series consumers."""
    state = compute_actor_metric_time_series(
        raw_events,
        top_n=top_n,
        tick_size=tick_size,
        max_rows=max_rows,
        include_level_columns=include_level_columns,
        state_actor_keys=(
            {f"client_original:{client_id}" for client_id in state_client_ids}
            if state_client_ids is not None
            else None
        ),
        execution_anchor_modes=("passive",),
        empirical_kernel_weights=empirical_kernel_weights,
    )
    if state.is_empty() or "identity_level" not in state.columns:
        return state
    state = state.filter(pl.col("identity_level") == "client_original")
    if state.is_empty():
        return state.drop(
            [
                column
                for column in ("actor_key", "actor_id", "identity_level", "identity_source", "identity_fallback_flag")
                if column in state.columns
            ]
        ).with_columns(pl.lit(None, dtype=pl.String).alias("client_id"))
    rename = {
        column: column.replace("_actor_", "_client_").replace("actor_", "client_", 1)
        for column in state.columns
        if "actor_" in column and column not in {"actor_key", "actor_id"}
    }
    state = state.rename(rename).with_columns(pl.col("actor_id").alias("client_id"))
    return state.drop(
        [
            column
            for column in ("actor_key", "actor_id", "identity_level", "identity_source", "identity_fallback_flag")
            if column in state.columns
        ]
    )


def _group_state_rows(states: pl.DataFrame) -> dict[tuple[Any, str], list[dict[str, Any]]]:
    groups: dict[tuple[Any, str], list[dict[str, Any]]] = defaultdict(list)
    if states.is_empty():
        return groups
    required = {"partition_id", "actor_key", "event_ts", "DWI", "L_bid_topN", "L_ask_topN"}
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
        groups[(row["partition_id"], row["actor_key"])].append(row)
    for values in groups.values():
        values.sort(key=lambda item: (float("inf") if item.get("sort_index") is None else int(item["sort_index"]), item["_ts"]))
    return groups


def _state_group_cache(
    values: list[dict[str, Any]],
) -> tuple[
    list[dict[str, Any]],
    list[int],
    list[dict[str, Any]],
    list[datetime],
    list[datetime],
]:
    if not values:
        return [], [], [], [], []
    cache = values[0].get("_lookup_cache")
    if cache is None:
        indexed = [item for item in values if item.get("sort_index") is not None]
        indexes = [int(item["sort_index"]) for item in indexed]
        time_sorted = sorted(indexed, key=lambda item: (item["_ts"], int(item["sort_index"])))
        times = [item["_ts"] for item in time_sorted]
        suffix_max_times = [item["_ts"] for item in indexed]
        for idx in range(len(suffix_max_times) - 2, -1, -1):
            suffix_max_times[idx] = max(suffix_max_times[idx], suffix_max_times[idx + 1])
        # Preserve both causal source order and event-time order.  They are not
        # interchangeable when source timestamps move backwards.
        cache = (indexed, indexes, time_sorted, times, suffix_max_times)
        values[0]["_lookup_cache"] = cache
    return cache


def _lookup_pre_state(values: list[dict[str, Any]], event_ts: datetime, sort_index: int | None) -> dict[str, Any] | None:
    if not values:
        return None
    indexed, indexes, time_sorted, times, _ = _state_group_cache(values)
    if sort_index is not None and indexes:
        idx = bisect.bisect_left(indexes, int(sort_index)) - 1
        return indexed[idx] if idx >= 0 else None
    idx = bisect.bisect_left(times, event_ts) - 1
    return time_sorted[idx] if idx >= 0 else None


def _lookup_post_state(values: list[dict[str, Any]], end_ts: datetime, sort_index: int | None) -> dict[str, Any] | None:
    if not values:
        return None
    indexed, indexes, time_sorted, times, _ = _state_group_cache(values)
    if not time_sorted:
        return None
    hi = bisect.bisect_right(times, end_ts)
    if sort_index is not None and indexes:
        # State rows are recorded after applying their event.  The state at the
        # cluster's final sort index is therefore already post-cluster and may
        # be carried forward to the window target.
        lower_index = int(sort_index)
        for item in reversed(time_sorted[:hi]):
            if int(item["sort_index"]) >= lower_index:
                return item
        return None
    return time_sorted[hi - 1] if hi > 0 else None


def _has_post_target_coverage(
    values: list[dict[str, Any]],
    target_ts: datetime,
    sort_index: int | None,
) -> bool:
    if not values:
        return False
    indexed, indexes, time_sorted, times, suffix_max_times = _state_group_cache(values)
    if sort_index is None:
        return bool(time_sorted) and times[-1] >= target_ts
    lo = bisect.bisect_left(indexes, int(sort_index))
    return lo < len(indexed) and suffix_max_times[lo] >= target_ts


def _lookup_state_at_or_after_index(values: list[dict[str, Any]], sort_index: int | None) -> dict[str, Any] | None:
    if sort_index is None:
        return None
    indexed, indexes, _, _, _ = _state_group_cache(values)
    idx = bisect.bisect_left(indexes, int(sort_index))
    return indexed[idx] if idx < len(indexed) else None


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


def _collapse(pre: float | None, post: float | None) -> float | None:
    if pre is None or post is None:
        return None
    pre_value = float(pre)
    post_value = float(post)
    if not math.isfinite(pre_value) or not math.isfinite(post_value):
        return None
    if pre_value < 0 or post_value < 0:
        return None
    if pre_value == 0:
        return 0.0
    return max(pre_value - post_value, 0.0) / pre_value


def _finite_signed_msci(sci: float | None, c_opposite: float | None, c_same: float | None) -> float | None:
    if sci is None or c_opposite is None or c_same is None:
        return None
    values = [float(sci), float(c_opposite), float(c_same)]
    if not all(math.isfinite(value) for value in values):
        return None
    return values[0] / 2.0 + values[1] - values[2]


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
) -> pl.DataFrame:
    if executions.is_empty():
        return executions
    grouped_states = _group_state_rows(states)
    rows: list[dict[str, Any]] = []
    window = timedelta(seconds=window_seconds)
    for row in executions.iter_rows(named=True):
        event_ts = _parse_ts(row.get("cluster_start_ts") or row.get("event_ts"))
        cluster_end_ts = _parse_ts(row.get("cluster_end_ts") or row.get("event_ts"))
        sort_index = row.get("cluster_first_sort_index") or row.get("sort_index")
        last_sort_index = row.get("cluster_last_sort_index") or row.get("sort_index")
        pre_state = None
        post_state = None
        posture_state = None
        post_target = None
        if event_ts is not None and cluster_end_ts is not None:
            post_target = cluster_end_ts + window
            values = grouped_states.get((row.get("partition_id"), row.get("actor_key")), [])
            posture_state = _lookup_state_at_or_after_index(
                values,
                int(row["candidate_deceptive_first_seen_sort_index_min"])
                if row.get("candidate_deceptive_first_seen_sort_index_min") is not None
                else None,
            )
            pre_state = _lookup_pre_state(values, event_ts, int(sort_index) if sort_index is not None else None)
            post_state = _lookup_post_state(
                values,
                post_target,
                int(last_sort_index) if last_sort_index is not None else None,
            )
        pre_dwi = pre_state.get("DWI") if pre_state is not None else None
        post_dwi = post_state.get("DWI") if post_state is not None else None
        sci = abs(float(pre_dwi) - float(post_dwi)) if pre_dwi is not None and post_dwi is not None else None
        l_bid_pre = pre_state.get("L_bid_topN") if pre_state is not None else None
        l_bid_post = post_state.get("L_bid_topN") if post_state is not None else None
        l_ask_pre = pre_state.get("L_ask_topN") if pre_state is not None else None
        l_ask_post = post_state.get("L_ask_topN") if post_state is not None else None
        collapse_bid = _collapse(l_bid_pre, l_bid_post)
        collapse_ask = _collapse(l_ask_pre, l_ask_post)
        if row.get("deceptive_side") == "bid":
            collapse_opposite = collapse_bid
            collapse_same = collapse_ask
        elif row.get("deceptive_side") == "ask":
            collapse_opposite = collapse_ask
            collapse_same = collapse_bid
        else:
            collapse_opposite = None
            collapse_same = None
        msci_resting_profile = _finite_signed_msci(sci, collapse_opposite, collapse_same)
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
                "has_post_window_state": post_state is not None,
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
                "post_fill_mid_reversal_from_pre_fill": _signed_change(price_direction, post_mid, pre_mid),
                "post_fill_microprice_reversal_from_pre_fill": _signed_change(
                    price_direction,
                    post_microprice,
                    pre_microprice,
                ),
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
                # MSCI is derived only from the common resting profile.
                # ``MSCI`` remains as a compatibility alias.
                "MSCI_resting_profile": msci_resting_profile,
                "MSCI": msci_resting_profile,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def assign_cancellations_to_clusters(candidate_links: pl.DataFrame) -> pl.DataFrame:
    """Assign each physical cancellation candidate to exactly one prior cluster."""
    if candidate_links.is_empty():
        return candidate_links
    required = {
        "partition_id",
        "cancel_sort_index",
        "candidate_order_id",
        "execution_cluster_id",
        "cluster_end_ts",
        "cluster_first_sort_index",
    }
    missing = sorted(required.difference(candidate_links.columns))
    if missing:
        raise ValueError(f"candidate links missing required columns: {', '.join(missing)}")

    rows = candidate_links.to_dicts()
    partition_columns = [
        column
        for column in ("actor_key",)
        if column in candidate_links.columns
    ]
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for index, row in enumerate(rows):
        key = (
            row.get("partition_id"),
            *(row.get(column) for column in partition_columns),
            int(row["cancel_sort_index"]),
            str(row["candidate_order_id"]),
        )
        groups[key].append(index)

    for indexes in groups.values():
        winner = min(
            indexes,
            key=lambda index: (
                -(_parse_ts(rows[index].get("cluster_end_ts")) or datetime.min).timestamp(),
                int(rows[index]["cluster_first_sort_index"]),
                str(rows[index]["execution_cluster_id"]),
            ),
        )
        for index in indexes:
            rows[index]["assigned_flag"] = index == winner
            rows[index]["assignment_rule"] = "latest_prior_cluster_end"
            rows[index]["competing_cluster_count"] = len(indexes)
    return pl.DataFrame(rows, infer_schema_length=None)


def _coerce_execution_cancel_candidate_schema(frame: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=EXECUTION_CANCEL_CANDIDATE_SCHEMA)
    missing = [
        pl.lit(None).cast(dtype).alias(column)
        for column, dtype in EXECUTION_CANCEL_CANDIDATE_SCHEMA.items()
        if column not in frame.columns
    ]
    if missing:
        frame = frame.with_columns(missing)
    return frame.select(
        pl.col(column).cast(dtype, strict=False)
        for column, dtype in EXECUTION_CANCEL_CANDIDATE_SCHEMA.items()
    )


def _build_execution_cancel_candidates(
    executions: pl.DataFrame,
    cancellations: pl.DataFrame,
    candidate_orders: pl.DataFrame | None = None,
    *,
    window_seconds: float,
) -> pl.DataFrame:
    if executions.is_empty() or cancellations.is_empty():
        return _coerce_execution_cancel_candidate_schema(pl.DataFrame())
    cancel_groups: dict[tuple[Any, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for cancel in cancellations.iter_rows(named=True):
        cancel_groups[
            (
                cancel["partition_id"],
                cancel["actor_key"],
                cancel["side"],
                str(cancel.get("ORDERID")),
            )
        ].append(cancel)

    candidate_qty_by_cluster_order = (
        {
            (str(row["execution_cluster_id"]), str(row["deceptive_order_id"])): max(
                float(row["deceptive_order_visible_qty_pre"] or 0.0),
                0.0,
            )
            for row in candidate_orders.iter_rows(named=True)
        }
        if candidate_orders is not None
        else {}
    )

    links: list[dict[str, Any]] = []
    window = timedelta(seconds=window_seconds)
    for execution in executions.iter_rows(named=True):
        cluster_end_ts = _parse_ts(execution.get("cluster_end_ts") or execution.get("event_ts"))
        if cluster_end_ts is None:
            continue
        cluster_last_sort_index_value = execution.get("cluster_last_sort_index")
        if cluster_last_sort_index_value is None:
            cluster_last_sort_index_value = execution.get("cluster_first_sort_index")
        if cluster_last_sort_index_value is None:
            cluster_last_sort_index_value = execution.get("sort_index")
        cluster_last_sort_index = int(cluster_last_sort_index_value)
        candidate_order_ids = {
            order_id
            for order_id in str(execution.get("candidate_deceptive_order_ids_pre") or "").split(";")
            if order_id
        }
        if not candidate_order_ids:
            continue
        end = cluster_end_ts + window
        for order_id in sorted(candidate_order_ids):
            candidates = cancel_groups.get(
                (
                    execution.get("partition_id"),
                    execution.get("actor_key"),
                    execution.get("deceptive_side"),
                    order_id,
                ),
                [],
            )
            for cancel in candidates:
                cancel_ts = _parse_ts(cancel.get("event_ts"))
                cancel_sort_index = int(cancel["sort_index"])
                if (
                    cancel_ts is None
                    or cancel_sort_index <= cluster_last_sort_index
                    or not (cluster_end_ts <= cancel_ts <= end)
                ):
                    continue
                cancel_visible_qty = max(float(cancel.get("visible_qty_pre_cancel") or 0.0), 0.0)
                candidate_visible_qty = candidate_qty_by_cluster_order.get(
                    (str(execution.get("execution_cluster_id")), order_id)
                )
                attributed_cancel_visible_qty = (
                    min(cancel_visible_qty, candidate_visible_qty)
                    if candidate_visible_qty is not None
                    else cancel_visible_qty
                )
                cluster_first_sort_index = execution.get("cluster_first_sort_index")
                if cluster_first_sort_index is None:
                    cluster_first_sort_index = execution.get("sort_index")
                links.append(
                    {
                        "partition_id": execution.get("partition_id"),
                        "cancel_sort_index": cancel_sort_index,
                        "candidate_order_id": order_id,
                        "execution_cluster_id": execution.get("execution_cluster_id"),
                        "actor_key": execution.get("actor_key"),
                        "actor_id": execution.get("actor_id"),
                        "identity_level": execution.get("identity_level"),
                        "identity_source": execution.get("identity_source"),
                        "identity_fallback_flag": execution.get("identity_fallback_flag"),
                        "execution_anchor_mode": execution.get("execution_anchor_mode"),
                        "execution_side": execution.get("execution_side"),
                        "deceptive_side": execution.get("deceptive_side"),
                        "cluster_end_ts": cluster_end_ts,
                        "cluster_first_sort_index": int(cluster_first_sort_index),
                        "cluster_last_sort_index": cluster_last_sort_index,
                        "cancel_event_ts": cancel_ts,
                        "cancel_visible_qty": cancel_visible_qty,
                        "candidate_visible_qty_pre": candidate_visible_qty,
                        "attributed_cancel_visible_qty": attributed_cancel_visible_qty,
                        "ORDERID": order_id,
                        "event_ts": cancel_ts,
                        "visible_qty_pre_cancel": cancel_visible_qty,
                    }
                )
    if not links:
        return _coerce_execution_cancel_candidate_schema(pl.DataFrame())
    assigned = assign_cancellations_to_clusters(pl.DataFrame(links, infer_schema_length=None))
    return _coerce_execution_cancel_candidate_schema(assigned)


def _weighted_finite_mean(values: list[tuple[float | None, float]]) -> float | None:
    finite = [
        (float(value), float(weight))
        for value, weight in values
        if value is not None
        and math.isfinite(float(value))
        and math.isfinite(float(weight))
        and float(weight) > 0
    ]
    if not finite:
        return None
    total_weight = sum(weight for _, weight in finite)
    return sum(value * weight for value, weight in finite) / total_weight


def _attributed_cancel_qty(candidate: dict[str, Any]) -> float:
    value = candidate.get("attributed_cancel_visible_qty")
    if value is None:
        value = candidate.get("cancel_visible_qty")
    if value is None:
        value = candidate.get("visible_qty_pre_cancel")
    return max(float(value or 0.0), 0.0)


def attach_cancel_anchored_reversion(
    executions: pl.DataFrame,
    states: pl.DataFrame,
    cancel_candidates: pl.DataFrame,
    *,
    reversion_horizon_seconds: float,
    withdrawal_decay_seconds: float = 10.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Measure price reversal from each assigned cancellation's actual timestamp.

    State rows are post-event observations.  The baseline is therefore the last
    state strictly before ``cancel_sort_index`` and the horizon state is the last
    state at or before ``cancel_event_ts + horizon`` with an index at or after the
    cancellation.  Physical cancellations remain candidate-level audit rows;
    execution-cluster metrics are quantity-and-delay weighted means.
    """
    if reversion_horizon_seconds <= 0:
        raise ValueError("reversion_horizon_seconds must be positive")
    if withdrawal_decay_seconds <= 0:
        raise ValueError("withdrawal_decay_seconds must be positive")
    if executions.is_empty():
        return executions, cancel_candidates

    grouped_states = _group_state_rows(states)
    enriched_candidates: list[dict[str, Any]] = []
    for candidate in cancel_candidates.iter_rows(named=True):
        enriched = {
            **candidate,
            "cancel_reversion_target_ts": None,
            "cancel_pre_state_sort_index": None,
            "cancel_post_state_sort_index": None,
            "cancel_mid_pre": None,
            "cancel_mid_post_horizon": None,
            "cancel_microprice_pre": None,
            "cancel_microprice_post_horizon": None,
            "post_cancel_mid_reversion": None,
            "post_cancel_microprice_reversion": None,
            "cancel_reversion_weight": None,
            "has_cancel_reversion_state": False,
        }
        if not bool(candidate.get("assigned_flag")):
            enriched_candidates.append(enriched)
            continue

        cancel_ts = _parse_ts(candidate.get("cancel_event_ts") or candidate.get("event_ts"))
        cluster_end_ts = _parse_ts(candidate.get("cluster_end_ts"))
        cancel_sort_index = candidate.get("cancel_sort_index")
        if cancel_ts is None or cluster_end_ts is None or cancel_sort_index is None:
            enriched_candidates.append(enriched)
            continue

        target_ts = cancel_ts + timedelta(seconds=reversion_horizon_seconds)
        values = grouped_states.get((candidate.get("partition_id"), candidate.get("actor_key")), [])
        pre_state = _lookup_pre_state(values, cancel_ts, int(cancel_sort_index))
        has_target_coverage = _has_post_target_coverage(
            values,
            target_ts,
            int(cancel_sort_index),
        )
        post_state = (
            _lookup_post_state(values, target_ts, int(cancel_sort_index))
            if has_target_coverage
            else None
        )
        direction = _execution_price_direction(candidate.get("execution_side"))
        delay_seconds = max((cancel_ts - cluster_end_ts).total_seconds(), 0.0)
        cancel_qty = _attributed_cancel_qty(candidate)
        weight = cancel_qty * math.exp(-delay_seconds / withdrawal_decay_seconds)
        mid_pre = pre_state.get("market_mid") if pre_state is not None else None
        mid_post = post_state.get("market_mid") if post_state is not None else None
        microprice_pre = pre_state.get("market_microprice") if pre_state is not None else None
        microprice_post = post_state.get("market_microprice") if post_state is not None else None
        enriched.update(
            {
                "cancel_reversion_target_ts": target_ts,
                "cancel_pre_state_sort_index": pre_state.get("sort_index") if pre_state is not None else None,
                "cancel_post_state_sort_index": post_state.get("sort_index") if post_state is not None else None,
                "cancel_mid_pre": mid_pre,
                "cancel_mid_post_horizon": mid_post,
                "cancel_microprice_pre": microprice_pre,
                "cancel_microprice_post_horizon": microprice_post,
                # Positive means reversal against the pre-fill favorable direction.
                "post_cancel_mid_reversion": _signed_change(direction, mid_post, mid_pre),
                "post_cancel_microprice_reversion": _signed_change(
                    direction,
                    microprice_post,
                    microprice_pre,
                ),
                "cancel_reversion_weight": weight if weight > 0 else None,
                "has_cancel_reversion_state": pre_state is not None and post_state is not None,
            }
        )
        enriched_candidates.append(enriched)

    enriched_df = _coerce_execution_cancel_candidate_schema(
        pl.DataFrame(enriched_candidates, infer_schema_length=None)
        if enriched_candidates
        else cancel_candidates
    )
    assigned_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in enriched_candidates:
        if bool(candidate.get("assigned_flag")):
            assigned_by_cluster[str(candidate.get("execution_cluster_id"))].append(candidate)

    execution_rows: list[dict[str, Any]] = []
    for execution in executions.iter_rows(named=True):
        assigned = assigned_by_cluster.get(str(execution.get("execution_cluster_id")), [])
        cancel_times = [
            cancel_ts
            for candidate in assigned
            if (cancel_ts := _parse_ts(candidate.get("cancel_event_ts") or candidate.get("event_ts")))
            is not None
        ]
        mid_values = [
            (candidate.get("post_cancel_mid_reversion"), float(candidate.get("cancel_reversion_weight") or 0.0))
            for candidate in assigned
        ]
        microprice_values = [
            (
                candidate.get("post_cancel_microprice_reversion"),
                float(candidate.get("cancel_reversion_weight") or 0.0),
            )
            for candidate in assigned
        ]
        execution_rows.append(
            {
                **execution,
                "post_cancel_mid_reversion": _weighted_finite_mean(mid_values),
                "post_cancel_microprice_reversion": _weighted_finite_mean(microprice_values),
                "cancel_reversion_observation_count": sum(
                    1 for value, weight in mid_values if value is not None and weight > 0
                ),
                "has_cancel_reversion_state": any(
                    bool(candidate.get("has_cancel_reversion_state")) for candidate in assigned
                ),
                "first_matched_cancel_ts": min(cancel_times) if cancel_times else None,
                "last_matched_cancel_ts": max(cancel_times) if cancel_times else None,
                "reversion_horizon_seconds": reversion_horizon_seconds,
            }
        )
    return pl.DataFrame(execution_rows, infer_schema_length=None), enriched_df


def _attach_direct_cancellation_window(
    executions: pl.DataFrame,
    cancellations: pl.DataFrame,
    *,
    window_seconds: float,
    withdrawal_decay_seconds: float = 10.0,
    assigned_candidates: pl.DataFrame | None = None,
) -> pl.DataFrame:
    if executions.is_empty():
        return executions
    cancel_groups: dict[tuple[Any, str, str], list[dict[str, Any]]] = defaultdict(list)
    if not cancellations.is_empty():
        for row in cancellations.iter_rows(named=True):
            cancel_groups[(row["partition_id"], row["actor_key"], row["side"])].append(row)
    assigned_by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    if assigned_candidates is not None and not assigned_candidates.is_empty():
        for candidate in assigned_candidates.filter(pl.col("assigned_flag")).iter_rows(named=True):
            assigned_by_cluster[str(candidate["execution_cluster_id"])].append(candidate)
    window = timedelta(seconds=window_seconds)
    rows: list[dict[str, Any]] = []
    for row in executions.iter_rows(named=True):
        event_ts = _parse_ts(row.get("cluster_end_ts") or row.get("event_ts"))
        matches: list[dict[str, Any]] = []
        if event_ts is not None:
            end = event_ts + window
            candidates = cancel_groups.get((row.get("partition_id"), row.get("actor_key"), row.get("deceptive_side")), [])
            matches = [
                cancel
                for cancel in candidates
                if (cancel_ts := _parse_ts(cancel.get("event_ts"))) is not None
                and event_ts < cancel_ts <= end
            ]
        matched = assigned_by_cluster.get(str(row.get("execution_cluster_id")), [])
        order_ids = ";".join(str(cancel["ORDERID"]) for cancel in matches)
        total_qty = sum(float(cancel["visible_qty_pre_cancel"] or 0.0) for cancel in matches)
        matched_order_ids = ";".join(str(cancel["ORDERID"]) for cancel in matched)
        matched_qty = sum(_attributed_cancel_qty(cancel) for cancel in matched)
        candidate_qty = float(row.get("candidate_deceptive_visible_qty_pre") or 0.0)
        execution_quantity = float(row.get("execution_quantity") or row.get("fill_qty") or 0.0)
        anchor_mode = row.get("execution_anchor_mode")
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
                withdrawal_qty = _attributed_cancel_qty(cancel)
                weighted_withdrawal_qty += withdrawal_qty * math.exp(-delay / withdrawal_decay_seconds)
        withdrawal_profile_scale = _finite_wmsci(
            candidate_qty=candidate_qty,
            weighted_withdrawal_qty=weighted_withdrawal_qty,
            fill_qty=execution_quantity,
            matched_fraction=matched_fraction,
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
                "withdrawal_to_execution_ratio": (
                    matched_qty / execution_quantity if execution_quantity > 0 else None
                ),
                "weighted_withdrawal_to_execution_ratio": (
                    weighted_withdrawal_qty / execution_quantity if execution_quantity > 0 else None
                ),
                # Compatibility aliases retained for existing artifacts.
                "withdrawal_to_fill_ratio": (
                    matched_qty / execution_quantity if execution_quantity > 0 else None
                ),
                "weighted_withdrawal_to_fill_ratio": (
                    weighted_withdrawal_qty / execution_quantity if execution_quantity > 0 else None
                ),
                "withdrawal_profile_scale_denominator_mode": (
                    f"{anchor_mode}_execution_quantity"
                    if anchor_mode in {"passive", "aggressive"}
                    else None
                ),
                "withdrawal_profile_scale_event": withdrawal_profile_scale,
                "WMSCI_passive": withdrawal_profile_scale if anchor_mode == "passive" else None,
                "WMSCI_aggressive": withdrawal_profile_scale if anchor_mode == "aggressive" else None,
                "WMSCI_event": withdrawal_profile_scale,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _finite_column_values(rows: list[dict[str, Any]], column: str) -> list[float]:
    return [
        numeric
        for row in rows
        if (numeric := _finite_float_or_none(row.get(column))) is not None
    ]


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
    if not gamma_grid:
        raise ValueError("gamma_grid must contain at least one threshold")
    try:
        finite_gammas = [float(gamma) for gamma in gamma_grid]
    except (TypeError, ValueError) as exc:
        raise ValueError("gamma thresholds must be finite numeric values") from exc
    if not all(math.isfinite(gamma) for gamma in finite_gammas):
        raise ValueError("gamma thresholds must be finite numeric values")
    if execution_metrics.is_empty():
        return pl.DataFrame(schema=MCPS_SCORE_SCHEMA)
    msci_column = (
        "MSCI_resting_profile"
        if "MSCI_resting_profile" in execution_metrics.columns
        else "MSCI"
    )
    group_cols = [
        col
        for col in (
            "partition_id",
            "actor_key",
            "actor_id",
            "identity_level",
            "identity_source",
            "identity_fallback_flag",
            "execution_anchor_mode",
            "top_n",
        )
        if col in execution_metrics.columns
    ]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in execution_metrics.to_dicts():
        actor_key = str(row.get("actor_key") or "").strip()
        anchor_mode = str(row.get("execution_anchor_mode") or "").strip().lower()
        if not actor_key.startswith(("client_original:", "firm:")):
            continue
        if anchor_mode not in {"passive", "aggressive"}:
            continue
        grouped[tuple(row.get(col) for col in group_cols)].append(row)

    out_rows: list[dict[str, Any]] = []
    for gamma in finite_gammas:
        for key, rows in grouped.items():
            finite_msci = _finite_column_values(rows, msci_column)
            finite_sci = _finite_column_values(rows, "SCI")
            collapse_opposite = _finite_column_values(rows, "collapse_opposite_side")
            collapse_same = _finite_column_values(rows, "collapse_same_side")
            favorable_mid_moves = _finite_column_values(rows, "favorable_mid_move_pre_fill")
            favorable_microprice_moves = _finite_column_values(
                rows, "favorable_microprice_move_pre_fill"
            )
            post_cancel_mid_reversions = _finite_column_values(
                rows, "post_cancel_mid_reversion"
            )
            execution_advantages = _finite_column_values(
                rows, "execution_price_advantage_vs_posture_mid"
            )
            above = sum(1 for value in finite_msci if value > gamma)
            out = {col: key[idx] for idx, col in enumerate(group_cols)}
            out.update(
                {
                    "gamma": gamma,
                    "executions": len(rows),
                    "finite_msci_executions": len(finite_msci),
                    "msci_above_gamma_count": above,
                    "MCPS": above / len(rows) if rows else None,
                    "MCPS_resting_profile": above / len(rows) if rows else None,
                    "median_MSCI": _median(finite_msci),
                    "max_MSCI": max(finite_msci) if finite_msci else None,
                    "mean_MSCI": _mean(finite_msci),
                    "median_MSCI_resting_profile": _median(finite_msci),
                    "max_MSCI_resting_profile": max(finite_msci) if finite_msci else None,
                    "mean_MSCI_resting_profile": _mean(finite_msci),
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
                        1.0
                        for row in rows
                        if (
                            _finite_float_or_none(
                                row.get("candidate_deceptive_order_count_pre")
                            )
                            or 0.0
                        )
                        > 0
                    )
                    / len(rows),
                }
            )
            out_rows.append(out)
    if not out_rows:
        return pl.DataFrame(schema=MCPS_SCORE_SCHEMA)
    return pl.DataFrame(out_rows, infer_schema_length=None)


def compute_exploratory_metrics(
    raw_events: pl.DataFrame,
    *,
    top_n: int,
    tick_size: float,
    window_seconds: float,
    withdrawal_window_seconds: float = 2.0,
    reversion_horizon_seconds: float = 2.0,
    max_rows: int | None = None,
    include_level_columns: bool = True,
    max_deceptive_order_age_seconds: float = 600.0,
    execution_cluster_max_gap_ms: int = 100,
    state_actor_keys: set[str] | None = None,
    state_client_ids: set[str] | None = None,
    execution_anchor_modes: Collection[str] = ("passive", "aggressive"),
    empirical_kernel_weights: Mapping[str, Mapping[int, float]] | None = None,
) -> ExploratoryMetricsResult:
    (
        state_df,
        execution_df,
        candidate_df,
        cancel_df,
        rejected_df,
        member_df,
    ) = _stream_metric_inputs(
        raw_events,
        top_n=top_n,
        tick_size=tick_size,
        max_rows=max_rows,
        include_level_columns=include_level_columns,
        max_deceptive_order_age_seconds=max_deceptive_order_age_seconds,
        execution_cluster_max_gap_ms=execution_cluster_max_gap_ms,
        state_actor_keys=state_actor_keys,
        state_client_ids=state_client_ids,
        execution_anchor_modes=execution_anchor_modes,
        empirical_kernel_weights=empirical_kernel_weights,
    )
    execution_df = attach_sci_window_metrics(
        execution_df,
        state_df,
        window_seconds=window_seconds,
    )
    cancel_candidate_df = _build_execution_cancel_candidates(
        execution_df,
        cancel_df,
        candidate_df,
        window_seconds=withdrawal_window_seconds,
    )
    execution_df = _attach_direct_cancellation_window(
        execution_df,
        cancel_df,
        window_seconds=withdrawal_window_seconds,
        assigned_candidates=cancel_candidate_df,
    )
    execution_df, cancel_candidate_df = attach_cancel_anchored_reversion(
        execution_df,
        state_df,
        cancel_candidate_df,
        reversion_horizon_seconds=reversion_horizon_seconds,
    )
    execution_df, spoofing_compatible_events = attach_spoofing_compatible_sequence_gate(execution_df)
    return ExploratoryMetricsResult(
        state_time_series=state_df,
        execution_metrics=execution_df,
        candidate_deceptive_orders=candidate_df,
        direct_cancellations=cancel_df,
        rejected_executions=rejected_df,
        execution_cluster_members=member_df,
        execution_cancel_candidates=cancel_candidate_df,
        spoofing_compatible_events=spoofing_compatible_events,
    )