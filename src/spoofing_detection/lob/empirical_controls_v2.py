"""Pure interval accounting for the empirical-controls v2 contract.

This module consumes already-certified risk intervals, withdrawals, clusters and
coverage.  It neither replays the book nor selects detector candidates.
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import math
from typing import Any, Iterable, Mapping, Sequence

import polars as pl

HORIZON = timedelta(seconds=2)
SHIFT_OFFSETS = (-60, -30, 0, 30, 60)
STATES = ("no_own_observed", "own_passive_only", "own_aggressive_only", "own_mixed")
IDENTITY = ("actor_key", "actor_id", "identity_level", "identity_source", "identity_fallback_flag")
GROUP_FIELDS = ("instrument", "partition_id", "event_date", *IDENTITY, "side", "time_band_id")

STATE_SCHEMA = {
    **{x: pl.String for x in ("instrument", "partition_id", "actor_key", "actor_id", "identity_level", "identity_source", "side", "time_band_id")},
    "event_date": pl.Date, "identity_fallback_flag": pl.Boolean,
    "analysis_kind": pl.String, "offset_seconds": pl.Int64, "state": pl.String,
    "time_at_risk_seconds": pl.Float64, "withdrawal_count": pl.Int64,
    "withdrawal_intensity": pl.Float64, "null_reason": pl.String,
    "support_anomaly": pl.Boolean, "opening_equality_count": pl.Int64,
}
CONTRAST_SCHEMA = {
    **{x: STATE_SCHEMA[x] for x in GROUP_FIELDS},
    "analysis_kind": pl.String, "offset_seconds": pl.Int64, "comparison_mode": pl.String,
    "mode_time_seconds": pl.Float64, "reference_time_seconds": pl.Float64,
    "mode_withdrawal_count": pl.Int64, "reference_withdrawal_count": pl.Int64,
    "mode_intensity": pl.Float64, "reference_intensity": pl.Float64,
    "intensity_difference": pl.Float64, "weight_seconds": pl.Float64, "comparison_status": pl.String,
}
SUMMARY_SCHEMA = {
    "instrument": pl.String, "identity_level": pl.String, "analysis_kind": pl.String,
    "offset_seconds": pl.Int64, "comparison_mode": pl.String, "group_count": pl.Int64,
    "weight_seconds": pl.Float64, "weighted_intensity_difference": pl.Float64, "summary_status": pl.String,
}
COVERAGE_SCHEMA = {
    **{x: STATE_SCHEMA[x] for x in GROUP_FIELDS},
    "coverage_block_count": pl.Int64, "total_risk_seconds": pl.Float64,
    "observed_domain_seconds": pl.Float64, "shift_common_domain_seconds": pl.Float64,
    "excluded_initial_history_seconds": pl.Float64, "excluded_shift_margin_seconds": pl.Float64,
    "observed_exclusion_reason": pl.String, "shift_exclusion_reason": pl.String,
}
BALANCE_SCHEMA = {
    **{x: STATE_SCHEMA[x] for x in GROUP_FIELDS},
    "analysis_kind": pl.String, "offset_seconds": pl.Int64, "state": pl.String,
    "time_at_risk_seconds": pl.Float64,
    "member_count_time_seconds": pl.Float64, "member_count_missing_time_seconds": pl.Float64, "mean_member_count": pl.Float64,
    "eligible_visible_qty_time_seconds": pl.Float64, "eligible_visible_qty_missing_time_seconds": pl.Float64, "mean_eligible_visible_qty": pl.Float64,
    "mean_member_age_time_seconds": pl.Float64, "mean_member_age_missing_time_seconds": pl.Float64, "analytical_mean_member_age_seconds": pl.Float64,
    "oldest_member_age_time_seconds": pl.Float64, "oldest_member_age_missing_time_seconds": pl.Float64, "analytical_oldest_member_age_seconds": pl.Float64,
    "spread_time_seconds": pl.Float64, "spread_missing_time_seconds": pl.Float64, "mean_spread": pl.Float64,
    "depth_time_seconds": pl.Float64, "depth_missing_time_seconds": pl.Float64, "mean_depth": pl.Float64,
    "prior_event_count_60s_time_seconds": pl.Float64, "prior_event_count_60s_missing_time_seconds": pl.Float64, "mean_prior_event_count_60s": pl.Float64,
    "market_covariate_support": pl.String,
}
SHIFT_SUPPORT_SCHEMA = {
    **{x: STATE_SCHEMA[x] for x in GROUP_FIELDS},
    "comparison_mode": pl.String, "supported_all_offsets": pl.Boolean,
    "unsupported_offset_count": pl.Int64, "zero_weight_seconds": pl.Float64, "support_reason": pl.String,
}

INTERVAL_REQUIRED = {"instrument", "partition_id", "event_date", *IDENTITY, "side", "coverage_epoch_id", "start_ts", "end_ts", "member_count", "eligible_visible_qty", "mean_member_age_seconds", "oldest_member_age_seconds"}
WITHDRAWAL_REQUIRED = {"instrument", "partition_id", "event_date", *IDENTITY, "side", "coverage_epoch_id", "withdrawal_event_id", "sort_index", "event_ts", "physical_order_key", "visible_qty_removed"}
CLUSTER_REQUIRED = {"instrument", "partition_id", "event_date", *IDENTITY, "event_side", "execution_anchor_mode", "cluster_end_ts", "cluster_last_sort_index", "execution_cluster_id", "execution_quantity"}
COVERAGE_REQUIRED = {"instrument", "partition_id", "event_date", "coverage_epoch_id", "start_ts", "end_ts", "end_reason"}
MARKET_REQUIRED = {"instrument", "partition_id", "event_date", "coverage_epoch_id", "start_ts", "end_ts", "spread", "depth", "prior_event_count_60s"}
MARKET_FIELDS = ("spread", "depth", "prior_event_count_60s")
WindowRow = tuple[datetime, datetime, int]
WindowIndex = tuple[Sequence[datetime], Sequence[WindowRow]]
MarketIndex = tuple[Sequence[datetime], Sequence[dict[str, Any]]]


@dataclass(frozen=True)
class EmpiricalControlsV2Result:
    state_statistics: pl.DataFrame
    contrast_statistics: pl.DataFrame
    summary: pl.DataFrame
    coverage: pl.DataFrame
    balance: pl.DataFrame
    shift_support: pl.DataFrame


def _frame(rows: list[dict[str, Any]], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, infer_schema_length=None).select(
        pl.col(name).cast(dtype, strict=False) for name, dtype in schema.items()
    )


def _require(frame: pl.DataFrame, fields: set[str], name: str) -> list[dict[str, Any]]:
    missing = sorted(fields - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")
    return frame.to_dicts()


def _band_start(value: datetime) -> datetime:
    return value.replace(minute=(value.minute // 30) * 30, second=0, microsecond=0)


def _band_id(value: datetime) -> str:
    return _band_start(value).strftime("%H:%M")


def _opposite(side: str) -> str:
    if side == "bid":
        return "ask"
    if side == "ask":
        return "bid"
    raise ValueError("interval side must be bid or ask")


def _identity_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in IDENTITY)


def _domain_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (row["instrument"], row["partition_id"], row["event_date"], *_identity_key(row), row["side"])


def _group_key(row: Mapping[str, Any], band_id: str) -> tuple[Any, ...]:
    return (*_domain_key(row), band_id)


def _row_group(row: Mapping[str, Any], key: tuple[Any, ...]) -> dict[str, Any]:
    return dict(zip(GROUP_FIELDS, key, strict=True))


def _clip(start: datetime, end: datetime, lower: datetime, upper: datetime) -> tuple[datetime, datetime] | None:
    left, right = max(start, lower), min(end, upper)
    return (left, right) if right > left else None


def _contains_piece(pieces: list[dict[str, Any]], point: datetime) -> bool:
    # Intervals are indexed/sorted; a point belongs to a positive half-open
    # interval, or to the explicitly retained zero-duration risk boundary.
    starts = [piece["start"] for piece in pieces]
    i = bisect_right(starts, point) - 1
    for candidate in (i, i + 1):
        if 0 <= candidate < len(pieces):
            piece = pieces[candidate]
            if piece["start"] <= point <= piece["end"]:
                return True
    return False


def _state(passive: bool, aggressive: bool) -> str:
    if passive and aggressive:
        return "own_mixed"
    if passive:
        return "own_passive_only"
    if aggressive:
        return "own_aggressive_only"
    return "no_own_observed"


def _mode_at_time(windows: Mapping[str, WindowIndex], point: datetime, *, observed: bool, sort_index: int | None) -> tuple[bool, bool, bool]:
    """Return passive/aggressive membership and opening-equality audit flag.

    The bisect start bounds make point attribution O(log W + local overlaps), not
    a global events-by-windows product.  Point inclusion intentionally differs
    from duration inclusion at both boundaries according to the v2 contract.
    """
    active: list[bool] = []
    equality = False
    for mode in ("passive", "aggressive"):
        starts, rows = windows[mode]
        lo = bisect_left(starts, point - HORIZON)
        hi = bisect_right(starts, point)
        present = False
        for index in range(lo, hi):
            anchor, _, anchor_sort = rows[index]
            if anchor == point and (not observed or sort_index is None or sort_index <= anchor_sort):
                equality = True
            if observed:
                if anchor <= point <= anchor + HORIZON and sort_index is not None and sort_index > anchor_sort:
                    present = True
            elif anchor < point <= anchor + HORIZON:
                present = True
        active.append(present)
    return active[0], active[1], equality


def _segments(piece: dict[str, Any], lower: datetime, upper: datetime, windows: Mapping[str, WindowIndex]) -> Iterable[tuple[datetime, datetime, str]]:
    clipped = _clip(piece["start"], piece["end"], lower, upper)
    if clipped is None:
        return ()
    start, end = clipped
    boundaries = {start, end}
    for starts, rows in windows.values():
        first = bisect_left(starts, start - HORIZON)
        last = bisect_left(starts, end)
        for index in range(first, last):
            anchor, _, _ = rows[index]
            if start < anchor < end:
                boundaries.add(anchor)
            finish = anchor + HORIZON
            if start < finish < end:
                boundaries.add(finish)
    ordered = sorted(boundaries)
    result = []
    for left, right in zip(ordered, ordered[1:]):
        midpoint = left + (right - left) / 2
        passive, aggressive, _ = _mode_at_time(windows, midpoint, observed=False, sort_index=None)
        result.append((left, right, _state(passive, aggressive)))
    return result


def _finite(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _shifted_windows(schedule: Mapping[str, WindowIndex], offset_seconds: int) -> dict[str, WindowIndex]:
    if offset_seconds == 0:
        return dict(schedule)
    shift = timedelta(seconds=offset_seconds)
    return {
        mode: (
            [start + shift for start in starts],
            [(anchor + shift, end + shift, anchor_sort) for anchor, end, anchor_sort in rows],
        )
        for mode, (starts, rows) in schedule.items()
    }


def _market_values(index: MarketIndex, start: datetime, end: datetime) -> Iterable[tuple[float, dict[str, float]]]:
    """Yield finite market covariate overlaps from one sorted epoch index."""
    starts, rows = index
    first = max(0, bisect_right(starts, start) - 1)
    last = bisect_left(starts, end)
    for row_index in range(first, last):
        row = rows[row_index]
        left, right = max(start, row["start_ts"]), min(end, row["end_ts"])
        if right <= left:
            continue
        finite = {field: value for field in MARKET_FIELDS if (value := _finite(row[field])) is not None}
        if finite:
            yield (right - left).total_seconds(), finite


def analyze_controls_v2(
    intervals: pl.DataFrame,
    withdrawals: pl.DataFrame,
    clusters: pl.DataFrame,
    coverage: pl.DataFrame,
    *,
    market_intervals: pl.DataFrame | None = None,
) -> EmpiricalControlsV2Result:
    """Account observed and fixed-shift empirical controls without replaying data.

    Risk intervals remain immutable.  Coverage is split at source-clock 30-minute
    boundaries; all subsequent interval operations use sorted sweep boundaries or
    bisection over per-group schedules.
    """
    interval_rows = _require(intervals, INTERVAL_REQUIRED, "intervals")
    withdrawal_rows = _require(withdrawals, WITHDRAWAL_REQUIRED, "withdrawals")
    cluster_rows = _require(clusters, CLUSTER_REQUIRED, "clusters")
    coverage_rows = _require(coverage, COVERAGE_REQUIRED, "coverage")
    market_by_epoch: dict[tuple[Any, ...], tuple[list[datetime], list[dict[str, Any]]]] = {}
    market_present = market_intervals is not None
    if market_intervals is not None:
        raw_market = _require(market_intervals, MARKET_REQUIRED, "market_intervals")
        grouped_market: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for row in raw_market:
            if not isinstance(row["start_ts"], datetime) or not isinstance(row["end_ts"], datetime) or row["end_ts"] <= row["start_ts"]:
                raise ValueError("market_intervals rows must have positive datetime intervals")
            grouped_market[(row["instrument"], row["partition_id"], row["event_date"], row["coverage_epoch_id"])].append(row)
        for epoch, rows in grouped_market.items():
            rows.sort(key=lambda row: (row["start_ts"], row["end_ts"]))
            if any(right["start_ts"] < left["end_ts"] for left, right in zip(rows, rows[1:])):
                raise ValueError("market_intervals overlap within an epoch")
            market_by_epoch[epoch] = ([row["start_ts"] for row in rows], rows)
    coverage_by_epoch: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    coverage_by_clock: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in coverage_rows:
        if not isinstance(row["start_ts"], datetime) or not isinstance(row["end_ts"], datetime) or row["end_ts"] < row["start_ts"]:
            raise ValueError("coverage rows must have non-negative datetime intervals")
        coverage_by_epoch[(row["instrument"], row["partition_id"], row["event_date"], row["coverage_epoch_id"])].append(row)
        coverage_by_clock[(row["instrument"], row["partition_id"], row["event_date"])].append(row)
    for values in coverage_by_epoch.values():
        values.sort(key=lambda row: (row["start_ts"], row["end_ts"]))
        if any(right["start_ts"] < left["end_ts"] for left, right in zip(values, values[1:])):
            raise ValueError("coverage rows overlap within an epoch")
    for values in coverage_by_clock.values():
        values.sort(key=lambda row: (row["start_ts"], row["end_ts"]))
        if any(right["start_ts"] < left["end_ts"] for left, right in zip(values, values[1:])):
            raise ValueError("coverage rows overlap across epochs")

    risk_by_domain: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for raw in interval_rows:
        if raw["side"] not in {"bid", "ask"}:
            raise ValueError("interval side must be bid or ask")
        start, end = raw["start_ts"], raw["end_ts"]
        if not isinstance(start, datetime) or not isinstance(end, datetime) or end < start:
            raise ValueError("interval rows must have non-negative datetime intervals")
        risk_by_domain[_domain_key(raw)].append(raw)
    for values in risk_by_domain.values():
        values.sort(key=lambda row: (row["start_ts"], row["end_ts"]))
        if any(right["start_ts"] < left["end_ts"] for left, right in zip(values, values[1:])):
            raise ValueError("risk intervals overlap within a group")
    groups: dict[tuple[Any, ...], dict[str, Any]] = {}
    for raw in interval_rows:
        if raw["side"] not in {"bid", "ask"}:
            raise ValueError("interval side must be bid or ask")
        start, end = raw["start_ts"], raw["end_ts"]
        if not isinstance(start, datetime) or not isinstance(end, datetime) or end < start:
            raise ValueError("interval rows must have non-negative datetime intervals")
        cov_key = (raw["instrument"], raw["partition_id"], raw["event_date"], raw["coverage_epoch_id"])
        for cov in coverage_by_epoch.get(cov_key, []):
            cursor = _band_start(max(start, cov["start_ts"]))
            while cursor < min(end, cov["end_ts"]) or (start == end and cursor <= start < cursor + timedelta(minutes=30) and cov["start_ts"] <= start < cov["end_ts"]):
                band_end = cursor + timedelta(minutes=30)
                left, right = max(start, cov["start_ts"], cursor), min(end, cov["end_ts"], band_end)
                if right > left or (start == end == left and left < cov["end_ts"] and left < band_end):
                    key = _group_key(raw, _band_id(cursor))
                    group = groups.setdefault(key, {"source": _row_group(raw, key), "blocks": {}, "pieces": []})
                    block_key = (max(cov["start_ts"], cursor), min(cov["end_ts"], band_end))
                    group["blocks"][block_key] = block_key
                    group["pieces"].append({"start": left, "end": right, "raw": raw, "block": block_key})
                if start == end:
                    break
                cursor = band_end

    clusters_by_domain: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in cluster_rows:
        if row["event_side"] not in {"bid", "ask"} or row["execution_anchor_mode"] not in {"passive", "aggressive"}:
            raise ValueError("clusters require bid/ask side and passive/aggressive mode")
        clusters_by_domain[_domain_key({**row, "side": row["event_side"]})].append(row)
    withdrawal_by_domain: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in withdrawal_rows:
        if row["side"] not in {"bid", "ask"}:
            raise ValueError("withdrawals require bid/ask side")
        withdrawal_by_domain[_domain_key(row)].append(row)
    for rows in withdrawal_by_domain.values():
        rows.sort(key=lambda row: (row["event_ts"], row["sort_index"], row["withdrawal_event_id"]))
    # Withdrawals are canonical eligibility decisions from the observer.  Retain
    # an eligible point in the observed coverage domain even when its equal-clock
    # lifecycle emitted no positive-duration interval.  The epoch equality keeps
    # a point from crossing an observer coverage gap into another epoch's risk.
    for row in withdrawal_rows:
        point = row["event_ts"]
        if not isinstance(point, datetime):
            continue
        cov_key = (row["instrument"], row["partition_id"], row["event_date"], row["coverage_epoch_id"])
        for cov in coverage_by_epoch.get(cov_key, []):
            if not cov["start_ts"] + HORIZON <= point < cov["end_ts"]:
                continue
            band_start = _band_start(point)
            block = (max(cov["start_ts"], band_start), min(cov["end_ts"], band_start + timedelta(minutes=30)))
            key = _group_key(row, _band_id(band_start))
            group = groups.setdefault(key, {"source": _row_group(row, key), "blocks": {}, "pieces": []})
            group["blocks"][block] = block
            group["pieces"].append({"start": point, "end": point, "raw": row, "block": block})

    state_rows: list[dict[str, Any]] = []
    balance_rows: list[dict[str, Any]] = []
    coverage_out: list[dict[str, Any]] = []
    internal: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}
    for key in sorted(groups, key=str):
        group = groups[key]
        source, pieces = group["source"], sorted(group["pieces"], key=lambda item: (item["start"], item["end"]))
        blocks = sorted(group["blocks"].values())
        domain = key[:-1]
        own = [row for row in clusters_by_domain.get(domain[:-1] + (_opposite(source["side"]),), []) if _identity_key(row) == tuple(source[f] for f in IDENTITY)]
        schedules: dict[tuple[datetime, datetime], dict[str, WindowIndex]] = {}
        for block in blocks:
            schedule_rows: dict[str, list[WindowRow]] = {mode: [] for mode in ("passive", "aggressive")}
            for row in own:
                anchor = row["cluster_end_ts"]
                if block[0] <= anchor < block[1]:
                    schedule_rows[row["execution_anchor_mode"]].append((anchor, anchor + HORIZON, int(row["cluster_last_sort_index"])))
            for rows in schedule_rows.values():
                rows.sort()
            schedules[block] = {
                mode: ([row[0] for row in rows], rows)
                for mode, rows in schedule_rows.items()
            }
        shifted_schedules = {
            (block, offset): _shifted_windows(schedule, offset)
            for block, schedule in schedules.items()
            for offset in SHIFT_OFFSETS
        }
        total = sum((piece["end"] - piece["start"]).total_seconds() for piece in pieces)
        observed_possible = sum(max(0.0, (b - (a + HORIZON)).total_seconds()) for a, b in blocks)
        shift_possible = sum(max(0.0, (b - (a + HORIZON + timedelta(seconds=60))).total_seconds()) for a, b in blocks)
        observed_risk = sum(max(0.0, (_clip(p["start"], p["end"], a + HORIZON, b) or (a, a))[1].__sub__((_clip(p["start"], p["end"], a + HORIZON, b) or (a, a))[0]).total_seconds()) for p in pieces for a, b in [p["block"]])
        shift_risk = sum(max(0.0, (_clip(p["start"], p["end"], a + HORIZON + timedelta(seconds=60), b - timedelta(seconds=60)) or (a, a))[1].__sub__((_clip(p["start"], p["end"], a + HORIZON + timedelta(seconds=60), b - timedelta(seconds=60)) or (a, a))[0]).total_seconds()) for p in pieces for a, b in [p["block"]])
        coverage_out.append({**source, "coverage_block_count": len(blocks), "total_risk_seconds": total,
            "observed_domain_seconds": observed_risk, "shift_common_domain_seconds": shift_risk,
            "excluded_initial_history_seconds": total - observed_risk, "excluded_shift_margin_seconds": total - shift_risk,
            "observed_exclusion_reason": None if observed_possible else "insufficient_clock_coverage",
            "shift_exclusion_reason": None if shift_possible else "insufficient_shift_coverage"})

        analyses = [("observed_full", 0, HORIZON, timedelta(0))] + [("timing_shift_common_support", offset, HORIZON + timedelta(seconds=60), timedelta(seconds=60)) for offset in SHIFT_OFFSETS]
        events = withdrawal_by_domain.get(domain, [])
        for kind, offset, history, tail in analyses:
            values = {state: {"time": 0.0, "count": 0, "opening": 0, "cov": defaultdict(float), "covtime": defaultdict(float)} for state in STATES}
            for piece in pieces:
                a, b = piece["block"]
                lower, upper = a + history, b - tail
                windows = shifted_schedules[(piece["block"], offset)]
                market_index = market_by_epoch.get((piece["raw"]["instrument"], piece["raw"]["partition_id"], piece["raw"]["event_date"], piece["raw"]["coverage_epoch_id"]))
                for left, right, state in _segments(piece, lower, upper, windows):
                    seconds = (right - left).total_seconds()
                    values[state]["time"] += seconds
                    for field in ("member_count", "eligible_visible_qty"):
                        value = _finite(piece["raw"][field])
                        if value is not None:
                            values[state]["cov"][field] += seconds * value
                            values[state]["covtime"][field] += seconds
                    for field in ("mean_member_age_seconds", "oldest_member_age_seconds"):
                        opening_age = _finite(piece["raw"][field])
                        if opening_age is not None:
                            elapsed_at_left = (left - piece["raw"]["start_ts"]).total_seconds()
                            values[state]["cov"][field] += seconds * (opening_age + elapsed_at_left + seconds / 2)
                            values[state]["covtime"][field] += seconds
                    if market_index is not None:
                        for market_seconds, market_values in _market_values(market_index, left, right):
                            for field, value in market_values.items():
                                values[state]["cov"][field] += market_seconds * value
                                values[state]["covtime"][field] += market_seconds
            for event in events:
                point = event["event_ts"]
                if not isinstance(point, datetime):
                    continue
                candidate = [p for p in pieces if p["raw"]["coverage_epoch_id"] == event["coverage_epoch_id"] and p["block"][0] + history <= point < p["block"][1] - tail]
                if not candidate or not _contains_piece(candidate, point):
                    continue
                block = candidate[0]["block"]
                windows = shifted_schedules[(block, offset)]
                passive, aggressive, equality = _mode_at_time(windows, point, observed=(kind == "observed_full"), sort_index=event["sort_index"])
                state = _state(passive, aggressive)
                values[state]["count"] += 1
                if equality:
                    values[state]["opening"] += 1
            internal[(key, kind, offset)] = values
            for state in STATES:
                value = values[state]
                time, count = value["time"], value["count"]
                anomaly = time == 0.0 and count > 0
                reason = "zero_time_positive_points_support_anomaly" if anomaly else ("zero_time_at_risk" if time == 0 else None)
                state_rows.append({**source, "analysis_kind": kind, "offset_seconds": offset, "state": state,
                    "time_at_risk_seconds": time, "withdrawal_count": count,
                    "withdrawal_intensity": count / time if time > 0 else None, "null_reason": reason,
                    "support_anomaly": anomaly, "opening_equality_count": value["opening"]})
                balance_rows.append({**source, "analysis_kind": kind, "offset_seconds": offset, "state": state, "time_at_risk_seconds": time,
                    "member_count_time_seconds": value["covtime"]["member_count"], "member_count_missing_time_seconds": time - value["covtime"]["member_count"], "mean_member_count": value["cov"]["member_count"] / value["covtime"]["member_count"] if value["covtime"]["member_count"] else None,
                    "eligible_visible_qty_time_seconds": value["covtime"]["eligible_visible_qty"], "eligible_visible_qty_missing_time_seconds": time - value["covtime"]["eligible_visible_qty"], "mean_eligible_visible_qty": value["cov"]["eligible_visible_qty"] / value["covtime"]["eligible_visible_qty"] if value["covtime"]["eligible_visible_qty"] else None,
                    "mean_member_age_time_seconds": value["covtime"]["mean_member_age_seconds"], "mean_member_age_missing_time_seconds": time - value["covtime"]["mean_member_age_seconds"], "analytical_mean_member_age_seconds": value["cov"]["mean_member_age_seconds"] / value["covtime"]["mean_member_age_seconds"] if value["covtime"]["mean_member_age_seconds"] else None,
                    "oldest_member_age_time_seconds": value["covtime"]["oldest_member_age_seconds"], "oldest_member_age_missing_time_seconds": time - value["covtime"]["oldest_member_age_seconds"], "analytical_oldest_member_age_seconds": value["cov"]["oldest_member_age_seconds"] / value["covtime"]["oldest_member_age_seconds"] if value["covtime"]["oldest_member_age_seconds"] else None,
                    "spread_time_seconds": value["covtime"]["spread"], "spread_missing_time_seconds": time - value["covtime"]["spread"], "mean_spread": value["cov"]["spread"] / value["covtime"]["spread"] if value["covtime"]["spread"] else None,
                    "depth_time_seconds": value["covtime"]["depth"], "depth_missing_time_seconds": time - value["covtime"]["depth"], "mean_depth": value["cov"]["depth"] / value["covtime"]["depth"] if value["covtime"]["depth"] else None,
                    "prior_event_count_60s_time_seconds": value["covtime"]["prior_event_count_60s"], "prior_event_count_60s_missing_time_seconds": time - value["covtime"]["prior_event_count_60s"], "mean_prior_event_count_60s": value["cov"]["prior_event_count_60s"] / value["covtime"]["prior_event_count_60s"] if value["covtime"]["prior_event_count_60s"] else None,
                    "market_covariate_support": "market_intervals_absent" if not market_present else ("finite_market_covariates_observed" if any(value["covtime"][field] for field in MARKET_FIELDS) else "no_finite_market_covariates_on_risk")})

    contrast_rows: list[dict[str, Any]] = []
    shift_support_rows: list[dict[str, Any]] = []
    for key in sorted(groups, key=str):
        source = groups[key]["source"]
        for kind, offsets in (("observed_full", (0,)), ("timing_shift_common_support", SHIFT_OFFSETS)):
            for offset in offsets:
                values = internal[(key, kind, offset)]
                for mode, state in (("passive", "own_passive_only"), ("aggressive", "own_aggressive_only")):
                    left, ref = values[state], values["no_own_observed"]
                    comparable = left["time"] > 0 and ref["time"] > 0
                    contrast_rows.append({**source, "analysis_kind": kind, "offset_seconds": offset, "comparison_mode": mode,
                        "mode_time_seconds": left["time"], "reference_time_seconds": ref["time"], "mode_withdrawal_count": left["count"], "reference_withdrawal_count": ref["count"],
                        "mode_intensity": left["count"] / left["time"] if left["time"] else None, "reference_intensity": ref["count"] / ref["time"] if ref["time"] else None,
                        "intensity_difference": left["count"] / left["time"] - ref["count"] / ref["time"] if comparable else None,
                        "weight_seconds": min(left["time"], ref["time"]) if comparable else 0.0, "comparison_status": "comparable" if comparable else "not_estimable"})
        for mode, state in (("passive", "own_passive_only"), ("aggressive", "own_aggressive_only")):
            valid = [internal[(key, "timing_shift_common_support", offset)][state]["time"] > 0 and internal[(key, "timing_shift_common_support", offset)]["no_own_observed"]["time"] > 0 for offset in SHIFT_OFFSETS]
            zero = internal[(key, "timing_shift_common_support", 0)]
            weight = min(zero[state]["time"], zero["no_own_observed"]["time"]) if all(valid) else 0.0
            shift_support_rows.append({**source, "comparison_mode": mode, "supported_all_offsets": all(valid), "unsupported_offset_count": len(valid) - sum(valid), "zero_weight_seconds": weight,
                "support_reason": None if all(valid) else "nonpositive_mode_or_reference_time_at_one_or_more_offsets"})

    summaries: list[dict[str, Any]] = []
    contrast_by_scope: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    support_map = {(tuple(row[f] for f in GROUP_FIELDS), row["comparison_mode"]): row for row in shift_support_rows}
    for row in contrast_rows:
        contrast_by_scope[(row["instrument"], row["identity_level"], row["analysis_kind"], row["offset_seconds"], row["comparison_mode"])].append(row)
    for scope, rows in sorted(contrast_by_scope.items(), key=str):
        _, _, kind, _, mode = scope
        usable = [row for row in rows if row["comparison_status"] == "comparable" and (kind != "timing_shift_common_support" or support_map[(tuple(row[f] for f in GROUP_FIELDS), mode)]["supported_all_offsets"])]
        weights = [support_map[(tuple(row[f] for f in GROUP_FIELDS), mode)]["zero_weight_seconds"]
                   if kind == "timing_shift_common_support" else row["weight_seconds"] for row in usable]
        weight = sum(weights)
        summaries.append({"instrument": scope[0], "identity_level": scope[1], "analysis_kind": kind, "offset_seconds": scope[3], "comparison_mode": mode,
            "group_count": len(usable), "weight_seconds": weight,
            "weighted_intensity_difference": sum(w * row["intensity_difference"] for w, row in zip(weights, usable, strict=True)) / weight if weight else None,
            "summary_status": "estimable" if weight else "not_estimable"})

    return EmpiricalControlsV2Result(
        _frame(state_rows, STATE_SCHEMA), _frame(contrast_rows, CONTRAST_SCHEMA), _frame(summaries, SUMMARY_SCHEMA),
        _frame(coverage_out, COVERAGE_SCHEMA), _frame(balance_rows, BALANCE_SCHEMA), _frame(shift_support_rows, SHIFT_SUPPORT_SCHEMA),
    )
