"""Legacy placebo descriptors and audited empirical schedule placebos.

The empirical API below never invents executions.  It relocates only canonical
execution-cluster anchors to observed, actor-side risk-boundary clocks and then
rebuilds the fixed exposure panel with the existing intersection routine.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
import hashlib
import json
import math
from numbers import Integral, Real
from typing import Any

import numpy as np
import polars as pl

from .withdrawal_risk import (
    ACTOR_SCHEMA,
    WithdrawalRiskResult,
    canonical_execution_cluster_rows,
    intersect_observed_execution_exposure,
)


# Kept byte-for-byte semantically compatible with the descriptor-only API.
def add_time_shift_placebo(events: pl.DataFrame, *, shift_events: int) -> pl.DataFrame:
    if "sort_index" not in events.columns:
        raise ValueError("events must contain sort_index for time-shift placebo")
    return events.with_columns(
        [
            pl.lit("time_shift").alias("placebo_type"),
            (pl.col("sort_index") + shift_events).alias("placebo_sort_index"),
        ]
    )


def add_wrong_side_placebo(events: pl.DataFrame) -> pl.DataFrame:
    if "deceptive_side" not in events.columns:
        raise ValueError("events must contain deceptive_side for wrong-side placebo")
    return events.with_columns(
        [
            pl.lit("wrong_side").alias("placebo_type"),
            pl.when(pl.col("deceptive_side") == "bid")
            .then(pl.lit("ask"))
            .when(pl.col("deceptive_side") == "ask")
            .then(pl.lit("bid"))
            .otherwise(None)
            .alias("placebo_deceptive_side"),
        ]
    )


PLACEBO_SCHEDULE_SCHEMA: dict[str, pl.DataType] = {
    "draw_id": pl.Int64,
    "schedule_block_id": pl.String,
    "partition_id": pl.String,
    "event_date": pl.Date,
    **ACTOR_SCHEMA,
    "side": pl.String,
    "activity_band": pl.String,
    "source_execution_cluster_id": pl.String,
    "pseudo_execution_cluster_id": pl.String,
    "source_anchor_order": pl.Int64,
    "source_anchor_ts": pl.Datetime("us"),
    "source_anchor_sort_index": pl.Int64,
    "pseudo_anchor_ts": pl.Datetime("us"),
    "pseudo_anchor_sort_index": pl.Int64,
    "accepted": pl.Boolean,
    "rejection_reason": pl.String,
    "support_boundary_count": pl.Int64,
    "supported_start_count": pl.Int64,
}
PLACEBO_SUPPORT_SCHEMA: dict[str, pl.DataType] = {
    "draw_id": pl.Int64,
    "schedule_block_id": pl.String,
    "partition_id": pl.String,
    "event_date": pl.Date,
    **ACTOR_SCHEMA,
    "side": pl.String,
    "activity_band": pl.String,
    "schedule_cluster_count": pl.Int64,
    "support_boundary_count": pl.Int64,
    "supported_start_count": pl.Int64,
    "accepted": pl.Boolean,
    "rejection_reason": pl.String,
    "selected_start_ts": pl.Datetime("us"),
}
PLACEBO_STATISTICS_SCHEMA: dict[str, pl.DataType] = {
    "draw_id": pl.Int64,
    "row_kind": pl.String,
    "partition_id": pl.String,
    "event_date": pl.Date,
    **ACTOR_SCHEMA,
    "side": pl.String,
    "contrast_label": pl.String,
    "comparison_label": pl.String,
    "time_at_risk_seconds": pl.Float64,
    "withdrawal_count": pl.Int64,
    "withdrawal_intensity": pl.Float64,
    "comparison_intensity_difference": pl.Float64,
}

# These are deliberately a closed set: an absent state is emitted with a zero
# denominator and null intensity rather than silently dropped or treated as zero.
_CONTRAST_STATES = ("own_only", "other_only", "mixed", "no_qualifying_fill")


@dataclass(frozen=True)
class EmpiricalPlaceboResult:
    """Observed and schedule-placebo exposure rescoring artifacts.

    ``draw_id == 0`` in ``placebo_statistics`` is the observed schedule.
    Draws 1..``draw_count`` contain a full new risk-panel intersection.  A
    rejected block contributes no pseudo execution; its actor-side posture is
    consequently reclassified by the same no-fill/mixed logic as any other
    panel time.
    """

    observed_risk: WithdrawalRiskResult
    placebo_schedules: pl.DataFrame
    placebo_support: pl.DataFrame
    placebo_statistics: pl.DataFrame


def _frame(rows: list[dict[str, Any]], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, infer_schema_length=None).select(
        pl.col(name).cast(dtype, strict=False) for name, dtype in schema.items()
    )


def _validate_draw_count(draw_count: Any) -> int:
    if type(draw_count) is not int or draw_count <= 0:
        raise ValueError("draw_count must be a positive integer")
    return draw_count


def _validate_seed(seed: Any) -> int:
    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or not 0 <= int(seed) <= 2**64 - 1
    ):
        raise ValueError("seed must be an integer from 0 through 2**64-1")
    return int(seed)


def _as_cluster_rows(execution_clusters: Iterable[Mapping[str, Any]] | pl.DataFrame) -> list[dict[str, Any]]:
    raw_rows = execution_clusters.to_dicts() if isinstance(execution_clusters, pl.DataFrame) else list(execution_clusters)
    # Reuse the canonical cluster validator/deduplicator.  Child fills remain
    # invalid here just as they are for observed exposure construction.
    clusters = canonical_execution_cluster_rows(raw_rows)
    _validate_source_schedule_order(raw_rows, clusters)
    for index, cluster in enumerate(clusters):
        band = cluster.get("activity_band")
        if not isinstance(band, str) or not band.strip():
            raise ValueError(
                f"invalid execution cluster row {index}: activity_band must be nonempty text for empirical placebo blocks"
            )
    return clusters


def _validate_source_schedule_order(
    raw_rows: list[Mapping[str, Any]], clusters: list[dict[str, Any]],
) -> None:
    """Reject causal input-order errors before canonical schedule sorting.

    Canonical deduplication intentionally keeps exact repeated cluster IDs, so
    those repetitions cannot imply a second causal anchor.  Interleaved blocks
    are valid, but each block's surviving anchors must retain nondecreasing
    timestamps and strictly increasing canonical sort indices in source order.
    """
    by_id = {cluster["execution_cluster_id"]: cluster for cluster in clusters}
    seen_ids: set[str] = set()
    by_block: dict[tuple[str, date, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for raw in raw_rows:
        cluster_id = raw["execution_cluster_id"]
        if cluster_id in seen_ids:
            continue
        seen_ids.add(cluster_id)
        cluster = by_id.get(cluster_id)
        if cluster is not None:
            by_block[_block_key(cluster)].append(cluster)
    for source in by_block.values():
        for previous, current in zip(source, source[1:]):
            if (
                current["cluster_end_ts"] < previous["cluster_end_ts"]
                or current["cluster_last_sort_index"] <= previous["cluster_last_sort_index"]
            ):
                raise ValueError("nonordered_schedule")


def _band_windows(
    values: Mapping[str, tuple[datetime, datetime]] | None,
) -> dict[str, tuple[datetime, datetime]]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise ValueError("activity_band_windows must be a mapping of band names to datetime pairs")
    result: dict[str, tuple[datetime, datetime]] = {}
    for band, window in values.items():
        if not isinstance(band, str) or not band.strip() or not isinstance(window, tuple) or len(window) != 2:
            raise ValueError("activity_band_windows must be a mapping of band names to datetime pairs")
        start, end = window
        if not isinstance(start, datetime) or not isinstance(end, datetime):
            raise ValueError("activity_band_windows must contain positive datetime intervals")
        result[band] = (start, end)
    return result


def _is_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _naive_utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc).replace(tzinfo=None) if _is_aware(value) else value


def _normalized_clock_frame(
    frame: pl.DataFrame, *, clock_fields: tuple[str, ...], event_date_clock: str,
) -> pl.DataFrame:
    """Return a UTC-naive copy with event dates derived from the UTC clock."""
    if frame.is_empty():
        return frame
    rows = []
    for source in frame.iter_rows(named=True):
        row = dict(source)
        for field in clock_fields:
            if isinstance(row.get(field), datetime):
                row[field] = _naive_utc(row[field])
        clock = row.get(event_date_clock)
        if isinstance(clock, datetime):
            row["event_date"] = clock.date()
        rows.append(row)
    return pl.DataFrame(
        rows,
        schema_overrides={**{field: pl.Datetime("us") for field in clock_fields}, "event_date": pl.Date},
    ).select(frame.columns)


def _validate_and_normalize_rescore_clocks(
    risk_result: WithdrawalRiskResult,
    raw_clusters: Iterable[Mapping[str, Any]],
    band_windows: Mapping[str, tuple[datetime, datetime]],
) -> tuple[WithdrawalRiskResult, dict[str, tuple[datetime, datetime]]]:
    """Require one clock convention and copy aware inputs to naive UTC.

    This mirrors ``choose_event_timestamp``: aware clocks are accepted only as
    one aware convention and are normalized on ingress.  Risk interval dates
    and withdrawal dates are consequently derived from their UTC clocks rather
    than retaining a potentially local-day label.
    """
    clocks: list[datetime] = []
    for interval in risk_result.intervals.iter_rows(named=True):
        clocks.extend(value for field in ("start_ts", "end_ts") if isinstance((value := interval.get(field)), datetime))
    for withdrawal in risk_result.withdrawal_events.iter_rows(named=True):
        if isinstance((value := withdrawal.get("event_ts")), datetime):
            clocks.append(value)
    for cluster in raw_clusters:
        if isinstance(cluster, Mapping) and isinstance((value := cluster.get("cluster_end_ts")), datetime):
            clocks.append(value)
    for start, end in band_windows.values():
        clocks.extend((start, end))
    conventions = {_is_aware(clock) for clock in clocks}
    if len(conventions) > 1:
        raise ValueError("rescore clocks must be consistently all naive or all timezone-aware")
    normalized_windows = {
        band: (_naive_utc(start), _naive_utc(end))
        for band, (start, end) in band_windows.items()
    }
    if any(end <= start for start, end in normalized_windows.values()):
        raise ValueError("activity_band_windows must contain positive datetime intervals")
    if not conventions or conventions == {False}:
        return risk_result, normalized_windows
    return (
        WithdrawalRiskResult(
            risk_result.spells,
            _normalized_clock_frame(
                risk_result.intervals, clock_fields=("start_ts", "end_ts"), event_date_clock="start_ts",
            ),
            risk_result.membership,
            _normalized_clock_frame(
                risk_result.withdrawal_events, clock_fields=("event_ts",), event_date_clock="event_ts",
            ),
            risk_result.transitions,
            risk_result.diagnostics,
            risk_result.actor_summary,
            risk_result.exposure_intervals,
            risk_result.exposure_contrasts,
            risk_result.exposure_strata,
        ),
        normalized_windows,
    )


def _block_key(cluster: Mapping[str, Any]) -> tuple[str, date, str, str, str, str]:
    return (
        str(cluster["partition_id"]),
        cluster["cluster_end_ts"].date(),
        str(cluster["actor_key"]),
        str(cluster["identity_level"]),
        str(cluster["event_side"]),
        str(cluster["activity_band"]),
    )


def _block_id(key: tuple[str, date, str, str, str, str]) -> str:
    canonical_key = json.dumps(
        [key[0], key[1].isoformat(), *key[2:]], ensure_ascii=True, separators=(",", ":"),
    ).encode("ascii")
    return f"placebo-block-{hashlib.sha256(canonical_key).hexdigest()[:32]}"


def _identity_values(row: Mapping[str, Any]) -> dict[str, Any]:
    return {field: row[field] for field in ACTOR_SCHEMA}


def _block_record(
    *, draw_id: int, block_id: str, source: Mapping[str, Any], schedule_cluster_count: int,
    support_boundary_count: int, supported_start_count: int, accepted: bool,
    rejection_reason: str | None, selected_start_ts: datetime | None,
) -> dict[str, Any]:
    return {
        "draw_id": draw_id,
        "schedule_block_id": block_id,
        "partition_id": source["partition_id"],
        "event_date": source["cluster_end_ts"].date(),
        **_identity_values(source),
        "side": source["event_side"],
        "activity_band": source["activity_band"],
        "schedule_cluster_count": schedule_cluster_count,
        "support_boundary_count": support_boundary_count,
        "supported_start_count": supported_start_count,
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "selected_start_ts": selected_start_ts,
    }


def _schedule_record(
    *, draw_id: int, block_id: str, source: Mapping[str, Any], source_anchor_order: int,
    pseudo_anchor_ts: datetime | None, pseudo_anchor_sort_index: int | None,
    accepted: bool, rejection_reason: str | None,
    support_boundary_count: int, supported_start_count: int,
) -> dict[str, Any]:
    return {
        "draw_id": draw_id,
        "schedule_block_id": block_id,
        "partition_id": source["partition_id"],
        "event_date": source["cluster_end_ts"].date(),
        **_identity_values(source),
        "side": source["event_side"],
        "activity_band": source["activity_band"],
        "source_execution_cluster_id": source["execution_cluster_id"],
        # The source ID remains authoritative cluster membership.  A draw is a
        # separate schedule, so no synthetic economic cluster identifier exists.
        "pseudo_execution_cluster_id": source["execution_cluster_id"] if accepted else None,
        "source_anchor_order": source_anchor_order,
        "source_anchor_ts": source["cluster_end_ts"],
        "source_anchor_sort_index": source["cluster_last_sort_index"],
        "pseudo_anchor_ts": pseudo_anchor_ts,
        "pseudo_anchor_sort_index": pseudo_anchor_sort_index,
        "accepted": accepted,
        "rejection_reason": rejection_reason,
        "support_boundary_count": support_boundary_count,
        "supported_start_count": supported_start_count,
    }


def _risk_support_key(row: Mapping[str, Any]) -> tuple[str, date, str, str, str]:
    return (
        str(row["partition_id"]), row["event_date"], str(row["actor_key"]),
        str(row["identity_level"]), str(row["side"]),
    )


def _risk_support_interval_index(
    risk_result: WithdrawalRiskResult,
) -> dict[tuple[str, date, str, str, str], list[dict[str, Any]]]:
    """Index support once per call: O(intervals + blocks), before each draw."""
    indexed: dict[tuple[str, date, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in risk_result.intervals.iter_rows(named=True):
        indexed[_risk_support_key(row)].append(row)
    for intervals in indexed.values():
        intervals.sort(key=lambda row: (row["start_ts"], row["end_ts"], row["risk_interval_id"]))
    return dict(indexed)


def _support_boundary_signature(interval: Mapping[str, Any]) -> tuple[tuple[str, str], ...]:
    """Compare all semantic support fields while ignoring generated interval IDs."""
    return tuple(sorted(
        (str(name), repr(value))
        for name, value in interval.items()
        if name not in {"risk_interval_id", "duration_seconds", "exposure_quantity"}
    ))


def _unique_support_boundaries(intervals: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate exact support records; reject ambiguous equal-time anchors."""
    by_start: dict[datetime, list[dict[str, Any]]] = defaultdict(list)
    for interval in intervals:
        by_start[interval["start_ts"]].append(dict(interval))
    unique: list[dict[str, Any]] = []
    for start_ts in sorted(by_start):
        candidates = by_start[start_ts]
        signatures = {_support_boundary_signature(interval) for interval in candidates}
        if len(signatures) != 1:
            raise ValueError("conflicting risk support boundaries share start_ts")
        unique.append(candidates[0])
    return unique


def _deduplicated_risk_intervals(risk_result: WithdrawalRiskResult) -> WithdrawalRiskResult:
    """Remove only byte-for-byte equivalent interval records for rescoring."""
    rows: list[dict[str, Any]] = []
    seen: set[tuple[tuple[str, str], ...]] = set()
    for source in risk_result.intervals.iter_rows(named=True):
        signature = tuple(sorted((str(name), repr(value)) for name, value in source.items()))
        if signature not in seen:
            seen.add(signature)
            rows.append(source)
    if len(rows) == risk_result.intervals.height:
        return risk_result
    return replace(risk_result, intervals=pl.DataFrame(rows, schema=risk_result.intervals.schema))


def _support_boundary_count(
    intervals: Iterable[Mapping[str, Any]], *, band_window: tuple[datetime, datetime] | None,
) -> int:
    """Count only anchor clocks belonging to the prespecified activity band."""
    if band_window is None:
        return sum(1 for interval in intervals if interval.get("start_sort_index") is not None)
    start, end = band_window
    return sum(
        1 for interval in intervals
        if interval.get("start_sort_index") is not None and start <= interval["start_ts"] < end
    )


def _block_rng(seed: int, key: tuple[str, date, str, str, str, str]) -> np.random.Generator:
    """Create a reproducible, block-local stream without process-random hashes."""
    canonical_key = json.dumps(
        [seed, key[0], key[1].isoformat(), *key[2:]],
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("ascii")
    digest = hashlib.sha256(canonical_key).digest()
    entropy = [int.from_bytes(digest[offset:offset + 4], "big") for offset in range(0, len(digest), 4)]
    return np.random.default_rng(np.random.SeedSequence(entropy))


def _has_continuous_same_spell_support(
    translated: list[datetime], intervals_by_start: Mapping[datetime, dict[str, Any]],
    intervals: list[dict[str, Any]],
) -> bool:
    """Require observed same-spell coverage through the final translated anchor.

    The endpoint is the final anchor itself: the later fixed outcome horizon is
    deliberately allowed to be censored by ordinary risk-panel construction.
    """
    anchor_intervals = [intervals_by_start[anchor] for anchor in translated]
    spell_id = anchor_intervals[0]["risk_spell_id"]
    if any(interval["risk_spell_id"] != spell_id for interval in anchor_intervals[1:]):
        return False
    covered_until = anchor_intervals[0]["end_ts"]
    final_anchor = translated[-1]
    for interval in intervals:
        if interval["risk_spell_id"] != spell_id or interval["end_ts"] <= covered_until:
            continue
        if interval["start_ts"] > covered_until:
            break
        covered_until = interval["end_ts"]
        if covered_until >= final_anchor:
            return True
    return covered_until >= final_anchor


def _supported_schedule_starts(
    source: list[dict[str, Any]], intervals: list[dict[str, Any]], *, horizon: timedelta,
    band_window: tuple[datetime, datetime] | None,
) -> tuple[list[datetime], str | None]:
    """Return exact boundary-clock translations for one canonical schedule.

    The first source anchor is translated to a candidate observed *risk-interval
    start*.  Every later anchor is translated by its original timedelta and must
    itself be another observed interval start.  This exact-match rule is the
    documented clock mapping: it preserves relative gaps without interpolation,
    wrapping, or fabricated event clocks.  Since starts are positive-risk
    half-open boundaries, acceptance uses only pre-anchor support.
    """
    ordered = sorted(
        source,
        key=lambda row: (row["cluster_end_ts"], row["cluster_last_sort_index"], row["execution_cluster_id"]),
    )
    first = ordered[0]["cluster_end_ts"]
    offsets = [row["cluster_end_ts"] - first for row in ordered]
    if any(offset < timedelta(0) for offset in offsets):
        return [], "nonordered_schedule"
    all_intervals_by_start = {interval["start_ts"]: interval for interval in intervals}
    intervals_by_start = {
        start: interval for start, interval in all_intervals_by_start.items()
        if interval.get("start_sort_index") is not None
    }
    boundaries = sorted(intervals_by_start)
    if not boundaries:
        return [], (
            "no_observed_boundary_ordering_support"
            if all_intervals_by_start else "no_at_risk_boundary_support"
        )
    valid: list[datetime] = []
    failure_reasons: set[str] = set()
    for candidate in boundaries:
        translated = [candidate + offset for offset in offsets]
        if any(anchor not in intervals_by_start for anchor in translated):
            failure_reasons.add(
                "no_observed_boundary_ordering_support"
                if any(
                    anchor in all_intervals_by_start
                    and all_intervals_by_start[anchor].get("start_sort_index") is None
                    for anchor in translated
                )
                else "no_supported_schedule_start"
            )
            continue
        if not _has_continuous_same_spell_support(translated, intervals_by_start, intervals):
            failure_reasons.add("at_risk_gap_or_spell_closure")
            continue
        # Schedules cannot cross an event date.  An optional band window makes
        # a caller's prespecified intraday activity-band boundary enforceable.
        if any(anchor.date() != first.date() or (anchor + horizon).date() != first.date() for anchor in translated):
            failure_reasons.add("boundary_crossing")
            continue
        if band_window is not None and any(
            anchor < band_window[0] or anchor + horizon > band_window[1] for anchor in translated
        ):
            failure_reasons.add("activity_band_boundary_crossing")
            continue
        valid.append(candidate)
    if valid:
        return valid, None
    # Prefer a concrete boundary diagnostic once an otherwise valid exact-clock
    # translation was available; a lack of clock support remains distinguishable.
    if "activity_band_boundary_crossing" in failure_reasons:
        return [], "activity_band_boundary_crossing"
    if "boundary_crossing" in failure_reasons:
        return [], "boundary_crossing"
    if "at_risk_gap_or_spell_closure" in failure_reasons:
        return [], "at_risk_gap_or_spell_closure"
    if "no_observed_boundary_ordering_support" in failure_reasons:
        return [], "no_observed_boundary_ordering_support"
    return [], "no_supported_schedule_start"


def _stratum_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        row["partition_id"], row["event_date"], row["actor_key"], row["actor_id"],
        row["identity_level"], row["identity_source"], row["identity_fallback_flag"], row["side"],
    )


def _statistics_rows(result: WithdrawalRiskResult, *, draw_id: int) -> list[dict[str, Any]]:
    """Aggregate actual contrast rows; never reuse an observed score for a draw."""
    strata: dict[tuple[Any, ...], dict[str, Any]] = {}
    for interval in result.intervals.iter_rows(named=True):
        key = _stratum_key(interval)
        strata.setdefault(key, {
            "partition_id": interval["partition_id"], "event_date": interval["event_date"],
            **_identity_values(interval), "side": interval["side"],
        })
    values: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {
        key: {state: {"duration": 0.0, "withdrawal_ids": set()} for state in _CONTRAST_STATES}
        for key in strata
    }
    for contrast in result.exposure_contrasts.iter_rows(named=True):
        key = _stratum_key(contrast)
        state = contrast["contrast_label"]
        if key not in values or state not in _CONTRAST_STATES:
            raise AssertionError("exposure contrast does not belong to an observed risk stratum")
        value = values[key][state]
        value["duration"] += float(contrast["duration_seconds"])
        if contrast["withdrawal_event_id"]:
            value["withdrawal_ids"].update(str(contrast["withdrawal_event_id"]).split("|"))

    rows: list[dict[str, Any]] = []
    aggregate = {state: {"duration": 0.0, "withdrawal_ids": set()} for state in _CONTRAST_STATES}
    for key in sorted(strata, key=str):
        source = strata[key]
        for state in _CONTRAST_STATES:
            value = values[key][state]
            denominator = float(value["duration"])
            numerator = len(value["withdrawal_ids"])
            if denominator == 0.0 and numerator:
                raise AssertionError("positive withdrawal numerator with zero time-at-risk denominator")
            intensity = numerator / denominator if denominator > 0 else None
            rows.append({
                "draw_id": draw_id, "row_kind": "state", **source,
                "contrast_label": state, "comparison_label": None,
                "time_at_risk_seconds": denominator, "withdrawal_count": numerator,
                "withdrawal_intensity": intensity, "comparison_intensity_difference": None,
            })
            aggregate[state]["duration"] += denominator
            aggregate[state]["withdrawal_ids"].update(value["withdrawal_ids"])

    aggregate_intensity: dict[str, float | None] = {}
    for state in _CONTRAST_STATES:
        denominator = float(aggregate[state]["duration"])
        numerator = len(aggregate[state]["withdrawal_ids"])
        if denominator == 0.0 and numerator:
            raise AssertionError("positive withdrawal numerator with zero time-at-risk denominator")
        aggregate_intensity[state] = numerator / denominator if denominator > 0 else None
    for right in ("other_only", "no_qualifying_fill"):
        left_value = aggregate_intensity["own_only"]
        right_value = aggregate_intensity[right]
        rows.append({
            "draw_id": draw_id,
            "row_kind": "aggregate_contrast",
            "partition_id": None,
            "event_date": None,
            "actor_key": None,
            "actor_id": None,
            "identity_level": None,
            "identity_source": None,
            "identity_fallback_flag": None,
            "side": None,
            "contrast_label": "own_only",
            "comparison_label": f"own_minus_{right}",
            "time_at_risk_seconds": float(aggregate["own_only"]["duration"]),
            "withdrawal_count": len(aggregate["own_only"]["withdrawal_ids"]),
            "withdrawal_intensity": left_value,
            "comparison_intensity_difference": (
                left_value - right_value if left_value is not None and right_value is not None else None
            ),
        })
    return rows


def rescore_empirical_placebos(
    risk_result: WithdrawalRiskResult,
    execution_clusters: Iterable[Mapping[str, Any]] | pl.DataFrame,
    *,
    withdrawal_window_seconds: float,
    draw_count: int,
    seed: int,
    activity_band_windows: Mapping[str, tuple[datetime, datetime]] | None = None,
) -> EmpiricalPlaceboResult:
    """Generate audited placebo schedules and rescore their actual risk panels.

    ``activity_band`` is required on every canonical cluster and forms part of a
    block with partition, date, actor identity, and posture side.
    ``activity_band_windows`` must declare every band as
    ``{band: (inclusive_start, inclusive_end)}``, so candidate support is never
    sampled across an unobserved intraday band boundary.  Each canonical block
    gets a stable local substream derived from ``seed`` and its canonical key,
    so unrelated blocks cannot perturb its draws.  No global random state or
    outcome-dependent rejection is used. All clocks must be either naive or
    aware; aware clocks are copied to naive UTC before any rescore. Support
    preparation is O(intervals + blocks/candidates) once per call; rescoring is
    necessarily repeated once per requested draw.
    """
    if not isinstance(risk_result, WithdrawalRiskResult):
        raise ValueError("risk_result must be a WithdrawalRiskResult")
    draw_count = _validate_draw_count(draw_count)
    seed = _validate_seed(seed)
    if (
        isinstance(withdrawal_window_seconds, bool)
        or not isinstance(withdrawal_window_seconds, Real)
        or not math.isfinite(withdrawal_window_seconds)
        or withdrawal_window_seconds <= 0
    ):
        raise ValueError("withdrawal_window_seconds must be a finite positive real")
    horizon = timedelta(seconds=float(withdrawal_window_seconds))
    band_windows = _band_windows(activity_band_windows)
    raw_clusters = execution_clusters.to_dicts() if isinstance(execution_clusters, pl.DataFrame) else list(execution_clusters)
    risk_result, band_windows = _validate_and_normalize_rescore_clocks(
        risk_result, raw_clusters, band_windows,
    )
    clusters = _as_cluster_rows(raw_clusters)
    missing_bands = sorted({str(cluster["activity_band"]) for cluster in clusters} - set(band_windows))
    if missing_bands:
        raise ValueError("activity_band_windows must define every cluster activity_band")

    grouped: dict[tuple[str, date, str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for cluster in clusters:
        grouped[_block_key(cluster)].append(cluster)
    blocks = [(key, sorted(rows, key=lambda row: (
        row["cluster_end_ts"], row["cluster_last_sort_index"], row["execution_cluster_id"],
    ))) for key, rows in sorted(grouped.items(), key=lambda item: item[0])]

    support_index = _risk_support_interval_index(risk_result)
    prepared_blocks: list[tuple[
        tuple[str, date, str, str, str, str], list[dict[str, Any]], str,
        list[dict[str, Any]], int, list[datetime], str | None, np.random.Generator,
    ]] = []
    for key, source in blocks:
        band_window = band_windows[key[-1]]
        # Schedule band stays in the block key but not the support key: every
        # activity band shares the same actor-side risk-boundary clock index.
        support_intervals = _unique_support_boundaries(support_index.get(key[:-1], []))
        valid_starts, reason = _supported_schedule_starts(
            source, support_intervals, horizon=horizon, band_window=band_window,
        )
        prepared_blocks.append((
            key, source, _block_id(key), support_intervals,
            _support_boundary_count(support_intervals, band_window=band_window),
            valid_starts, reason, _block_rng(seed, key),
        ))

    # Support ambiguity is rejected above, before an observed or randomized
    # rescore. This remains the genuine observed intersection for draw zero.
    risk_result = _deduplicated_risk_intervals(risk_result)
    observed = intersect_observed_execution_exposure(
        risk_result, clusters, withdrawal_window_seconds=withdrawal_window_seconds,
    )

    schedule_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = []
    statistics_rows = _statistics_rows(observed, draw_id=0)

    for draw_id in range(1, draw_count + 1):
        pseudo_clusters: list[dict[str, Any]] = []
        for key, source, block_id, support_intervals, support_boundary_count, valid_starts, reason, rng in prepared_blocks:
            if not valid_starts:
                support_rows.append(_block_record(
                    draw_id=draw_id, block_id=block_id, source=source[0], schedule_cluster_count=len(source),
                    support_boundary_count=support_boundary_count, supported_start_count=0, accepted=False,
                    rejection_reason=reason, selected_start_ts=None,
                ))
                for order, cluster in enumerate(source):
                    schedule_rows.append(_schedule_record(
                        draw_id=draw_id, block_id=block_id, source=cluster, source_anchor_order=order,
                        pseudo_anchor_ts=None, pseudo_anchor_sort_index=None,
                        accepted=False, rejection_reason=reason,
                        support_boundary_count=support_boundary_count, supported_start_count=0,
                    ))
                continue
            selected_start = valid_starts[int(rng.integers(0, len(valid_starts)))]
            source_start = source[0]["cluster_end_ts"]
            support_rows.append(_block_record(
                draw_id=draw_id, block_id=block_id, source=source[0], schedule_cluster_count=len(source),
                support_boundary_count=support_boundary_count, supported_start_count=len(valid_starts), accepted=True,
                rejection_reason=None, selected_start_ts=selected_start,
            ))
            boundary_sort_indices = {
                interval["start_ts"]: interval.get("start_sort_index")
                for interval in support_intervals
            }
            for order, cluster in enumerate(source):
                pseudo_anchor = selected_start + (cluster["cluster_end_ts"] - source_start)
                pseudo_anchor_sort_index = boundary_sort_indices[pseudo_anchor]
                if pseudo_anchor_sort_index is None:
                    raise AssertionError("accepted pseudo anchor lacks canonical boundary ordering")
                schedule_rows.append(_schedule_record(
                    draw_id=draw_id, block_id=block_id, source=cluster, source_anchor_order=order,
                    pseudo_anchor_ts=pseudo_anchor,
                    pseudo_anchor_sort_index=int(pseudo_anchor_sort_index),
                    accepted=True, rejection_reason=None,
                    support_boundary_count=support_boundary_count, supported_start_count=len(valid_starts),
                ))
                pseudo = dict(cluster)
                pseudo["cluster_end_ts"] = pseudo_anchor
                pseudo["cluster_last_sort_index"] = int(pseudo_anchor_sort_index)
                pseudo_clusters.append(pseudo)
        # Crucially, even a fully rejected draw is recomputed: it is the fixed
        # risk panel with no pseudo own-execution schedule, not copied scores.
        rescored = intersect_observed_execution_exposure(
            risk_result, pseudo_clusters, withdrawal_window_seconds=withdrawal_window_seconds,
        )
        statistics_rows.extend(_statistics_rows(rescored, draw_id=draw_id))

    return EmpiricalPlaceboResult(
        observed_risk=observed,
        placebo_schedules=_frame(schedule_rows, PLACEBO_SCHEDULE_SCHEMA),
        placebo_support=_frame(support_rows, PLACEBO_SUPPORT_SCHEMA),
        placebo_statistics=_frame(statistics_rows, PLACEBO_STATISTICS_SCHEMA),
    )


# Descriptive aliases keep the public operation discoverable without duplicating
# a second implementation or weakening the explicit ``rescore`` semantics.
generate_empirical_placebos = rescore_empirical_placebos
build_empirical_negative_controls = rescore_empirical_placebos
