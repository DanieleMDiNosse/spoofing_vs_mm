from __future__ import annotations

import bisect
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
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
from spoofing_detection.lob.spoofing_metrics import choose_event_timestamp, shifted_depth_distance_ticks


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must have positive total")
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total


def fit_log_decay_slope(
    distances: Sequence[float],
    values: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    value_floor: float = 1e-12,
) -> float | None:
    if weights is not None and len(weights) != len(values):
        raise ValueError("weights must have the same length as values")
    rows = []
    for distance, value, weight in zip(
        distances,
        values,
        weights if weights is not None else [1.0] * len(values),
        strict=True,
    ):
        distance_f = float(distance)
        value_f = float(value)
        weight_f = float(weight)
        if math.isfinite(distance_f) and math.isfinite(value_f) and weight_f > 0:
            rows.append((distance_f, math.log(max(value_f, value_floor)), weight_f))
    if len(rows) < 2:
        return None
    xs, ys, ws = zip(*rows, strict=True)
    xbar = _weighted_mean(xs, ws)
    ybar = _weighted_mean(ys, ws)
    denom = sum(weight * (x - xbar) ** 2 for x, weight in zip(xs, ws, strict=True))
    if denom <= 0:
        return None
    beta = sum(weight * (x - xbar) * (y - ybar) for x, y, weight in rows) / denom
    return max(-beta, 0.0)


def _empty_profile() -> pl.DataFrame:
    return pl.DataFrame()


def build_empirical_depth_kernel(
    profile: pl.DataFrame,
    *,
    protection_floor: float = 0.0,
    visibility_floor: float = 0.0,
) -> pl.DataFrame:
    required = {
        "instrument_id",
        "side",
        "rank",
        "depth_distance_ticks",
        "exposure_count",
        "hit_count",
        "hit_probability",
        "visibility_covariance",
    }
    missing = sorted(required - set(profile.columns))
    if missing:
        raise ValueError(f"missing empirical kernel profile columns: {', '.join(missing)}")
    if profile.is_empty():
        return profile

    with_components = profile.with_columns(
        protection_component=(1.0 - pl.col("hit_probability")).clip(lower_bound=protection_floor),
        visibility_component=pl.col("visibility_covariance").abs().clip(lower_bound=visibility_floor),
    ).with_columns(raw_weight=pl.col("protection_component") * pl.col("visibility_component"))

    totals = with_components.group_by(["instrument_id", "side"]).agg(
        pl.col("raw_weight").sum().alias("raw_weight_total")
    )
    out = with_components.join(totals, on=["instrument_id", "side"], how="left").with_columns(
        kernel_weight=pl.when(pl.col("raw_weight_total") > 0)
        .then(pl.col("raw_weight") / pl.col("raw_weight_total"))
        .otherwise(0.0)
    )
    return out.drop("raw_weight_total").sort(["instrument_id", "side", "rank"])


def _to_seconds(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.timestamp()
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        try:
            return datetime.fromisoformat(str(value).replace("Z", "+00:00")).replace(tzinfo=None).timestamp()
        except ValueError:
            return None
    return numeric if math.isfinite(numeric) else None


def estimate_hit_probability_profile(snapshots: pl.DataFrame, fills: pl.DataFrame, *, horizon_seconds: float) -> pl.DataFrame:
    if snapshots.is_empty():
        return _empty_profile()
    required_snapshots = {"instrument_id", "side", "rank", "depth_distance_ticks", "price", "visible_qty", "event_ts"}
    required_fills = {"instrument_id", "side", "price", "event_ts"}
    missing_snapshots = sorted(required_snapshots - set(snapshots.columns))
    missing_fills = sorted(required_fills - set(fills.columns))
    if missing_snapshots:
        raise ValueError(f"missing snapshot columns: {', '.join(missing_snapshots)}")
    if missing_fills:
        raise ValueError(f"missing fill columns: {', '.join(missing_fills)}")

    fills_by_group: dict[tuple[Any, Any, str], list[tuple[float, float]]] = defaultdict(list)
    for row in fills.iter_rows(named=True):
        ts = _to_seconds(row.get("event_ts"))
        if ts is None or row.get("side") not in {"bid", "ask"} or row.get("price") is None:
            continue
        fills_by_group[(row.get("instrument_id"), row.get("partition_id"), str(row.get("side")))].append(
            (ts, float(row.get("price")))
        )
    fill_indexes: dict[tuple[Any, Any, str], tuple[list[float], list[float]]] = {}
    for key, values in fills_by_group.items():
        values.sort(key=lambda item: item[0])
        fill_indexes[key] = ([ts for ts, _ in values], [price for _, price in values])

    def range_reaches_level(key: tuple[Any, Any, str], *, start_ts: float, end_ts: float, side: str, price: float) -> bool:
        index = fill_indexes.get(key)
        if index is None:
            # Some synthetic/unit inputs omit partition_id on fill rows.
            index = fill_indexes.get((key[0], None, key[2]))
        if index is None:
            return False
        times, prices = index
        lo = bisect.bisect_right(times, start_ts)
        hi = bisect.bisect_right(times, end_ts)
        if lo >= hi:
            return False
        if side == "ask":
            return max(prices[lo:hi]) >= price
        return min(prices[lo:hi]) <= price

    grouped: dict[tuple[Any, str, int, float], dict[str, Any]] = {}
    for row in snapshots.iter_rows(named=True):
        ts = _to_seconds(row.get("event_ts"))
        price = row.get("price")
        visible_qty = float(row.get("visible_qty") or 0.0)
        side = row.get("side")
        if ts is None or price is None or visible_qty <= 0 or side not in {"bid", "ask"}:
            continue
        key = (row.get("instrument_id"), str(side), int(row.get("rank")), float(row.get("depth_distance_ticks")))
        bucket = grouped.setdefault(
            key,
            {
                "instrument_id": row.get("instrument_id"),
                "side": str(side),
                "rank": int(row.get("rank")),
                "depth_distance_ticks": float(row.get("depth_distance_ticks")),
                "exposure_count": 0,
                "hit_count": 0,
            },
        )
        bucket["exposure_count"] += 1
        partition_id = row.get("partition_id")
        snapshot_price = float(price)
        hit = range_reaches_level(
            (row.get("instrument_id"), partition_id, str(side)),
            start_ts=ts,
            end_ts=ts + float(horizon_seconds),
            side=str(side),
            price=snapshot_price,
        )
        if hit:
            bucket["hit_count"] += 1

    rows = []
    for bucket in grouped.values():
        exposure = bucket["exposure_count"]
        rows.append({**bucket, "hit_probability": bucket["hit_count"] / exposure if exposure else None})
    return pl.DataFrame(rows, infer_schema_length=None).sort(["instrument_id", "side", "rank"]) if rows else _empty_profile()


def estimate_visibility_covariance_profile(snapshots: pl.DataFrame) -> pl.DataFrame:
    if snapshots.is_empty():
        return _empty_profile()
    required = {
        "instrument_id",
        "snapshot_id",
        "side",
        "rank",
        "depth_distance_ticks",
        "visible_qty",
        "mid",
        "future_mid",
    }
    missing = sorted(required - set(snapshots.columns))
    if missing:
        raise ValueError(f"missing visibility snapshot columns: {', '.join(missing)}")

    totals_by_snapshot: dict[tuple[Any, Any], float] = defaultdict(float)
    rows = []
    for row in snapshots.iter_rows(named=True):
        qty = float(row.get("visible_qty") or 0.0)
        if qty > 0:
            totals_by_snapshot[(row.get("instrument_id"), row.get("snapshot_id"))] += qty
    for row in snapshots.iter_rows(named=True):
        qty = float(row.get("visible_qty") or 0.0)
        total = totals_by_snapshot[(row.get("instrument_id"), row.get("snapshot_id"))]
        side = row.get("side")
        mid = row.get("mid")
        future_mid = row.get("future_mid")
        if qty <= 0 or total <= 0 or side not in {"bid", "ask"} or mid is None or future_mid is None:
            continue
        delta_mid = float(future_mid) - float(mid)
        if not math.isfinite(delta_mid):
            continue
        sign = 1.0 if side == "ask" else -1.0
        rows.append(
            {
                "instrument_id": row.get("instrument_id"),
                "side": str(side),
                "rank": int(row.get("rank")),
                "depth_distance_ticks": float(row.get("depth_distance_ticks")),
                "level_contribution": sign * qty / total,
                "delta_mid": delta_mid,
            }
        )

    grouped: dict[tuple[Any, str, int, float], list[tuple[float, float]]] = defaultdict(list)
    for row in rows:
        grouped[(row["instrument_id"], row["side"], row["rank"], row["depth_distance_ticks"])].append(
            (row["level_contribution"], row["delta_mid"])
        )

    out = []
    for (instrument_id, side, rank, distance), values in grouped.items():
        xs = [x for x, _ in values]
        ys = [y for _, y in values]
        if len(values) < 2:
            covariance = 0.0
        else:
            xbar = sum(xs) / len(xs)
            ybar = sum(ys) / len(ys)
            covariance = sum((x - xbar) * (y - ybar) for x, y in values) / len(values)
        out.append(
            {
                "instrument_id": instrument_id,
                "side": side,
                "rank": rank,
                "depth_distance_ticks": distance,
                "visibility_observation_count": len(values),
                "visibility_covariance": abs(covariance),
            }
        )
    return pl.DataFrame(out, infer_schema_length=None).sort(["instrument_id", "side", "rank"]) if out else _empty_profile()


def _market_levels(active_orders: Mapping[str, ActiveOrder], *, side: str, top_n: int) -> list[tuple[float, float]]:
    by_price: dict[float, float] = defaultdict(float)
    for order in active_orders.values():
        if order.side != side or order.leaves_qty <= 0 or order.displayed_qty <= 0:
            continue
        by_price[float(order.price)] += float(order.displayed_qty)
    prices = sorted(by_price, reverse=(side == "bid"))[:top_n]
    return [(price, by_price[price]) for price in prices]


def _snapshot_rows(
    active_orders: Mapping[str, ActiveOrder],
    *,
    instrument_id: str,
    partition_id: str | None,
    snapshot_id: int,
    sort_index: int,
    event_ts: datetime | None,
    tick_size: float,
    top_n: int,
) -> list[dict[str, Any]]:
    levels = {side: _market_levels(active_orders, side=side, top_n=top_n) for side in ("bid", "ask")}
    best_bid = levels["bid"][0][0] if levels["bid"] else None
    best_ask = levels["ask"][0][0] if levels["ask"] else None
    mid = (best_bid + best_ask) / 2.0 if best_bid is not None and best_ask is not None else None
    rows = []
    for side in ("bid", "ask"):
        if not levels[side]:
            continue
        best_price = levels[side][0][0]
        for rank, (price, qty) in enumerate(levels[side], start=1):
            rows.append(
                {
                    "instrument_id": instrument_id,
                    "partition_id": partition_id,
                    "snapshot_id": snapshot_id,
                    "sort_index": sort_index,
                    "event_ts": event_ts,
                    "side": side,
                    "rank": rank,
                    "price": price,
                    "visible_qty": qty,
                    "mid": mid,
                    "depth_distance_ticks": shifted_depth_distance_ticks(side, price, best_price, tick_size),
                }
            )
    return rows


def _fill_row(event: dict[str, Any], *, instrument_id: str, event_ts: datetime | None, partition_id: str | None) -> dict[str, Any] | None:
    if event.get("event_class") != "fill" or event.get("side_label") not in {"bid", "ask"} or event.get("ORDERPX") is None:
        return None
    last_shares = event.get("LASTSHARES")
    if last_shares is None or float(last_shares or 0.0) <= 0:
        return None
    return {
        "instrument_id": instrument_id,
        "partition_id": partition_id,
        "sort_index": event.get("sort_index"),
        "event_ts": event_ts,
        "side": event.get("side_label"),
        "price": float(event.get("ORDERPX")),
        "last_shares": float(last_shares),
    }


def _attach_future_mid(snapshots: pl.DataFrame, *, horizon_seconds: float) -> pl.DataFrame:
    if snapshots.is_empty():
        return snapshots
    by_snapshot: dict[tuple[Any, int], dict[str, Any]] = {}
    for row in snapshots.iter_rows(named=True):
        key = (row.get("partition_id"), int(row["snapshot_id"]))
        if key not in by_snapshot:
            by_snapshot[key] = {
                "partition_id": row.get("partition_id"),
                "snapshot_id": int(row["snapshot_id"]),
                "sort_index": int(row["sort_index"]),
                "event_ts": row.get("event_ts"),
                "ts_seconds": _to_seconds(row.get("event_ts")),
                "mid": row.get("mid"),
            }

    by_partition: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for row in by_snapshot.values():
        by_partition[row.get("partition_id")].append(row)
    for rows in by_partition.values():
        rows.sort(key=lambda row: int(row["sort_index"]))

    future_mid_by_key: dict[tuple[Any, int], float | None] = {}
    finite_horizon = math.isfinite(float(horizon_seconds))
    for partition_id, rows in by_partition.items():
        for idx, snapshot in enumerate(rows):
            future_mid = None
            snapshot_ts = snapshot.get("ts_seconds")
            for candidate in rows[idx + 1 :]:
                if finite_horizon and snapshot_ts is not None and candidate.get("ts_seconds") is not None:
                    if float(candidate["ts_seconds"]) > float(snapshot_ts) + float(horizon_seconds):
                        break
                if candidate.get("mid") is not None:
                    future_mid = float(candidate["mid"])
            future_mid_by_key[(partition_id, int(snapshot["snapshot_id"]))] = future_mid

    return snapshots.with_columns(
        pl.struct(["partition_id", "snapshot_id"])
        .map_elements(lambda s: future_mid_by_key.get((s["partition_id"], int(s["snapshot_id"]))), return_dtype=pl.Float64)
        .alias("future_mid")
    )


def build_calibration_snapshot_tables(
    raw_events: pl.DataFrame,
    *,
    instrument_id: str,
    top_n: int,
    tick_size: float,
    max_rows: int | None = None,
    horizon_seconds: float | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    config = LOBConfig(top_n=max(top_n, 1), snapshot_mode="none")
    sorted_events = sort_events(raw_events)
    if max_rows is not None:
        sorted_events = sorted_events.head(max_rows)
    events = [normalize_event(raw_row, sort_index=idx, config=config) for idx, raw_row in enumerate(sorted_events.iter_rows(named=True), start=1)]

    active_orders: dict[str, ActiveOrder] = {}
    pending_aggressive_residuals: dict[str, tuple[dict[str, Any], tuple[Any, ...] | None]] = {}
    non_resting_order_ids: set[str] = set()
    current_partition_id: str | None = None
    snapshot_rows: list[dict[str, Any]] = []
    fill_rows: list[dict[str, Any]] = []
    snapshot_id = 0

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
        fill = _fill_row(event, instrument_id=instrument_id, event_ts=event_ts, partition_id=partition_id)
        if fill is not None:
            fill_rows.append(fill)
        _apply_event(
            active_orders,
            event,
            pending_aggressive_residuals=pending_aggressive_residuals,
            non_resting_order_ids=non_resting_order_ids,
        )
        next_event = events[event_index + 1] if event_index + 1 < len(events) else None
        next_group = _fill_group_key(next_event) if next_event is not None else None
        _flush_pending_aggressive_residuals(active_orders, pending_aggressive_residuals, keep_group=next_group)
        snapshot_id += 1
        snapshot_rows.extend(
            _snapshot_rows(
                active_orders,
                instrument_id=instrument_id,
                partition_id=partition_id,
                snapshot_id=snapshot_id,
                sort_index=int(event["sort_index"]),
                event_ts=event_ts,
                tick_size=tick_size,
                top_n=top_n,
            )
        )

    snapshots = pl.DataFrame(snapshot_rows, infer_schema_length=None) if snapshot_rows else pl.DataFrame()
    if horizon_seconds is not None:
        snapshots = _attach_future_mid(snapshots, horizon_seconds=horizon_seconds)
    else:
        snapshots = _attach_future_mid(snapshots, horizon_seconds=float("inf"))
    fills = pl.DataFrame(fill_rows, infer_schema_length=None) if fill_rows else pl.DataFrame()
    return snapshots, fills


def _attach_decay_summaries(kernel: pl.DataFrame) -> pl.DataFrame:
    if kernel.is_empty():
        return kernel
    summary_rows = []
    for (instrument_id, side), group in kernel.group_by(["instrument_id", "side"]):
        distances = group["depth_distance_ticks"].to_list()
        hit_prob = group["hit_probability"].to_list()
        visibility = group["visibility_covariance"].to_list()
        exposure_weights = group["exposure_count"].to_list()
        visibility_weights = group.get_column("visibility_observation_count").to_list() if "visibility_observation_count" in group.columns else exposure_weights
        summary_rows.append(
            {
                "instrument_id": instrument_id,
                "side": side,
                "kappa_hat_side": fit_log_decay_slope(distances, hit_prob, weights=exposure_weights),
                "lambda_hat_side": fit_log_decay_slope(distances, visibility, weights=visibility_weights),
            }
        )
    side_summary = pl.DataFrame(summary_rows, infer_schema_length=None)
    out = kernel.join(side_summary, on=["instrument_id", "side"], how="left")
    instrument_summary = summarise_instrument_kernel(out)
    return out.join(instrument_summary, on="instrument_id", how="left")


def summarise_instrument_kernel(kernel: pl.DataFrame) -> pl.DataFrame:
    if kernel.is_empty():
        return pl.DataFrame()
    rows = []
    for instrument_id, group in kernel.group_by("instrument_id"):
        side_rows = []
        for side, side_group in group.group_by("side"):
            exposure_weight = float(side_group["exposure_count"].sum()) if "exposure_count" in side_group.columns else 0.0
            visibility_weight = (
                float(side_group["visibility_observation_count"].sum())
                if "visibility_observation_count" in side_group.columns
                else exposure_weight
            )
            kappa_values = [v for v in side_group["kappa_hat_side"].drop_nulls().unique().to_list()]
            lambda_values = [v for v in side_group["lambda_hat_side"].drop_nulls().unique().to_list()]
            side_rows.append(
                {
                    "side": side,
                    "kappa": float(kappa_values[0]) if kappa_values else None,
                    "lambda": float(lambda_values[0]) if lambda_values else None,
                    "exposure_weight": exposure_weight,
                    "visibility_weight": visibility_weight,
                }
            )
        k_num = sum(row["kappa"] * row["exposure_weight"] for row in side_rows if row["kappa"] is not None)
        k_den = sum(row["exposure_weight"] for row in side_rows if row["kappa"] is not None)
        l_num = sum(row["lambda"] * row["visibility_weight"] for row in side_rows if row["lambda"] is not None)
        l_den = sum(row["visibility_weight"] for row in side_rows if row["lambda"] is not None)
        rows.append(
            {
                "instrument_id": instrument_id[0] if isinstance(instrument_id, tuple) else instrument_id,
                "kappa_hat_instrument": k_num / k_den if k_den > 0 else None,
                "lambda_hat_instrument": l_num / l_den if l_den > 0 else None,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def calibrate_empirical_depth_kernel(
    raw_events: pl.DataFrame,
    *,
    instrument_id: str,
    top_n: int,
    tick_size: float,
    horizon_seconds: float,
    protection_floor: float = 0.0,
    visibility_floor: float = 0.0,
    max_rows: int | None = None,
) -> pl.DataFrame:
    snapshots, fills = build_calibration_snapshot_tables(
        raw_events,
        instrument_id=instrument_id,
        top_n=top_n,
        tick_size=tick_size,
        max_rows=max_rows,
        horizon_seconds=horizon_seconds,
    )
    hit_profile = estimate_hit_probability_profile(snapshots, fills, horizon_seconds=horizon_seconds)
    visibility_profile = estimate_visibility_covariance_profile(snapshots)
    if hit_profile.is_empty():
        return hit_profile
    profile = hit_profile.join(
        visibility_profile,
        on=["instrument_id", "side", "rank", "depth_distance_ticks"],
        how="left",
    ).with_columns(pl.col("visibility_covariance").fill_null(0.0))
    if "visibility_observation_count" not in profile.columns:
        profile = profile.with_columns(pl.lit(0).alias("visibility_observation_count"))
    profile = profile.with_columns(pl.col("visibility_observation_count").fill_null(0))
    kernel = build_empirical_depth_kernel(profile, protection_floor=protection_floor, visibility_floor=visibility_floor)
    return _attach_decay_summaries(kernel)


def load_empirical_kernel_weights(path: Path | str, *, instrument_id: str | None = None) -> dict[str, dict[int, float]]:
    path = Path(path)
    df = pl.read_parquet(path) if path.suffix == ".parquet" else pl.read_csv(path)
    if instrument_id is not None and "instrument_id" in df.columns:
        df = df.filter(pl.col("instrument_id") == instrument_id)
    required = {"side", "rank", "kernel_weight"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing empirical kernel columns: {', '.join(missing)}")
    weights: dict[str, dict[int, float]] = {}
    for side, group in df.group_by("side"):
        side_name = side[0] if isinstance(side, tuple) else side
        raw = {int(row["rank"]): float(row["kernel_weight"]) for row in group.iter_rows(named=True)}
        total = sum(raw.values())
        weights[str(side_name)] = {rank: value / total for rank, value in raw.items()} if total > 0 else raw
    return weights
