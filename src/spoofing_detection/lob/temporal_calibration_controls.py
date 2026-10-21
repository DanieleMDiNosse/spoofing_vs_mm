"""Raw-population shifted-control W association driver.

This module is deliberately a narrow, non-CLI milestone: it builds uncapped
risk from one complete raw partition, constructs canonical 100-ms execution
clusters without detector gates, and converts the existing v2 control accounting
into auditable count-excess association curves.  It proposes no detector
parameter and never labels a cancellation as manipulative.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import json
import math
from typing import Any, Mapping, Sequence

import polars as pl

from .actor_identity import actor_identity_from_event
from .empirical_clock import CLOCK_POLICY_ID, resolve_empirical_clocks
from .empirical_controls_v2 import EmpiricalControlsV2Result, SHIFT_OFFSETS, analyze_controls_v2
from .execution_clusters import LIFECYCLE_BREAK_CLASSES, classify_execution_anchor, cluster_execution_fills
from .panel import ReplayHooks, replay_events
from .spoofing_clock import SPOOFING_DETECTION_CLOCK_CONTRACT, resolve_spoofing_clocks
from .temporal_calibration import proposed_window
from .temporal_calibration_inputs import collect_uncapped_risk
from .withdrawal_risk_v2 import CompactRiskResult


DEFAULT_WINDOWS_SECONDS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0)
_SUPPORT_HORIZON_SECONDS = 20.0
_NONZERO_OFFSETS = tuple(offset for offset in SHIFT_OFFSETS if offset != 0)
_IDENTITY_FIELDS = ("actor_key", "actor_id", "identity_level", "identity_source", "identity_fallback_flag")
_GROUP_FIELDS = ("instrument", "partition_id", "event_date", *_IDENTITY_FIELDS, "side", "time_band_id")

_CLUSTER_SCHEMA = {
    "instrument": pl.String, "partition_id": pl.String, "event_date": pl.Date,
    "actor_key": pl.String, "actor_id": pl.String, "identity_level": pl.String,
    "identity_source": pl.String, "identity_fallback_flag": pl.Boolean,
    "event_side": pl.String, "execution_anchor_mode": pl.String,
    "cluster_end_ts": pl.Datetime, "cluster_last_sort_index": pl.Int64,
    "execution_cluster_id": pl.String, "execution_quantity": pl.Float64,
    "clock_policy_id": pl.String, "fill_anchor_ns": pl.Int64,
}
CONTRIBUTION_SCHEMA = {
    **{name: pl.String for name in ("instrument", "partition_id", "actor_key", "actor_id", "identity_level", "identity_source", "side", "time_band_id")},
    "event_date": pl.Date, "identity_fallback_flag": pl.Boolean,
    "execution_anchor_mode": pl.String, "role_additivity": pl.String, "proposal_eligible": pl.Boolean,
    "window_seconds": pl.Float64, "support_horizon_seconds": pl.Float64,
    "observed_count": pl.Int64, "observed_risk_seconds": pl.Float64,
    "shift_minus_60_count": pl.Int64, "shift_minus_60_risk_seconds": pl.Float64,
    "shift_minus_30_count": pl.Int64, "shift_minus_30_risk_seconds": pl.Float64,
    "shift_zero_count": pl.Int64, "shift_zero_risk_seconds": pl.Float64,
    "shift_plus_30_count": pl.Int64, "shift_plus_30_risk_seconds": pl.Float64,
    "shift_plus_60_count": pl.Int64, "shift_plus_60_risk_seconds": pl.Float64,
    "valid_shift_offset_count": pl.Int64, "support_anomaly": pl.Boolean,
    "expected_background_count": pl.Float64, "excess_count": pl.Float64,
    "contribution_status": pl.String, "exclusion_reason": pl.String,
    "clock_policy_id": pl.String,
}
W_CURVE_SCHEMA = {
    "instrument": pl.String, "identity_level": pl.String, "execution_anchor_mode": pl.String,
    "window_seconds": pl.Float64, "excess_count": pl.Float64,
    "included_group_count": pl.Int64, "excluded_group_count": pl.Int64,
    "supported_risk_seconds": pl.Float64, "aggregation": pl.String,
}
PROPOSAL_SCHEMA = {
    "instrument": pl.String, "identity_level": pl.String, "execution_anchor_mode": pl.String,
    "proposed_window_seconds": pl.Float64, "proposal_status": pl.String,
    "search_cap_binding": pl.Boolean, "window_count": pl.Int64,
    "included_group_count": pl.Int64, "excluded_group_count": pl.Int64,
}


@dataclass(frozen=True)
class ShiftedControlWResult:
    """In-memory, auditable output for one raw partition association run."""

    risk: CompactRiskResult
    risk_age_cap_seconds: float
    clusters: pl.DataFrame
    controls: dict[float, EmpiricalControlsV2Result]
    contributions: pl.DataFrame
    curve: pl.DataFrame
    proposals: pl.DataFrame
    provenance: dict[str, Any]


def _frame(rows: list[dict[str, Any]], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, infer_schema_length=None).select(
        pl.col(name).cast(dtype, strict=False) for name, dtype in schema.items()
    )


def _event_date(event: Mapping[str, Any]) -> date:
    value = event.get("TRADEDATE")
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def _ns_to_datetime(value: int) -> datetime:
    seconds, remainder = divmod(value, 1_000_000_000)
    return datetime(1970, 1, 1) + timedelta(seconds=seconds, microseconds=remainder // 1_000)


def _source_index(local: int, original: Mapping[int, int] | None) -> int:
    value = local if original is None else original.get(local)
    if type(value) is not int or value <= 0:
        raise ValueError("original_sort_indices must map every local index to a positive integer")
    return value


def certify_clock_equivalence(raw_events: pl.DataFrame) -> dict[str, Any]:
    """Certify raw-event equivalence of v3 and empirical clock semantics.

    The two policy IDs deliberately differ.  This adapter therefore compares
    each independently resolved semantic field after canonical normalization;
    it does not relabel a detector policy string as an empirical policy string.
    """
    expected_contract = {
        "replay_clock": "SEQUENCETIME", "duration_clock": "SEQUENCETIME",
        "cancellation_removal_clock": "SEQUENCETIME", "fill_anchor_clock": "TRADETIME",
        "timestamp_representation": "UTC epoch nanoseconds", "semantic_fallback": "forbidden",
        "interval_convention": "left_closed_right_open",
    }
    if SPOOFING_DETECTION_CLOCK_CONTRACT != expected_contract:
        raise ValueError("v3 clock contract is not the certified empirical semantic contract")
    checked = 0
    mismatches: list[dict[str, Any]] = []

    def inspect(event: dict[str, Any], active: dict[str, Any], partition_id: str | None) -> None:
        nonlocal checked
        v3, empirical = resolve_spoofing_clocks(event), resolve_empirical_clocks(event)
        checked += 1
        fields = ("replay_time_ns", "fill_anchor_ns", "cancellation_removal_ns")
        if any(getattr(v3, name) != getattr(empirical, name) for name in fields):
            mismatches.append({"sort_index": event.get("sort_index"), "fields": fields})

    replay_events(raw_events, config=None, max_rows=None, hooks=ReplayHooks(on_pre_event=inspect), retain_events=False)
    if mismatches:
        raise ValueError("v3/empirical clock values differ on the raw partition")
    return {
        "certified": True, "checked_event_count": checked,
        "v3_contract": dict(SPOOFING_DETECTION_CLOCK_CONTRACT),
        "empirical_clock_policy_id": CLOCK_POLICY_ID,
        "certification_method": "semantic_contract_and_per_event_clock_value_equality",
    }


def _collect_canonical_clusters(
    raw_events: pl.DataFrame, *, instrument: str, original_sort_indices: Mapping[int, int] | None,
) -> pl.DataFrame:
    """Replay all valid fills and cluster them under the canonical 100-ms rule."""
    replay_rows: list[dict[str, Any]] = []

    def pre(event: dict[str, Any], active: dict[str, Any], partition_id: str | None) -> None:
        local = int(event["sort_index"])
        row = dict(event)
        row["partition_id"] = partition_id
        row["sort_index"] = _source_index(local, original_sort_indices)
        clocks = resolve_empirical_clocks(event)
        row["fill_anchor_ns"] = clocks.fill_anchor_ns
        row["replay_time_ns"] = clocks.replay_time_ns
        row["event_ts"] = _ns_to_datetime(clocks.replay_time_ns) if type(clocks.replay_time_ns) is int else None
        if row.get("event_class") == "fill":
            actor, role = actor_identity_from_event(event), classify_execution_anchor(event)
            quantity = event.get("LASTSHARES")
            if actor is None or role is None or type(clocks.fill_anchor_ns) is not int or type(clocks.replay_time_ns) is not int:
                return
            if type(quantity) not in (int, float) or isinstance(quantity, bool) or not math.isfinite(quantity) or quantity <= 0:
                return
            if role == "passive":
                order = active.get(str(event.get("ORDERID")))
                price = None if order is None else order.price
                source = "active_order_price"
            else:
                price, source = event.get("LASTTRADEDPX"), "LASTTRADEDPX"
            if type(price) not in (int, float) or isinstance(price, bool) or not math.isfinite(price) or price <= 0:
                return
            row.update({
                "fill_qty": float(quantity), "event_price": float(price), "execution_price_source": source,
                "actor_key": actor.actor_key, "actor_id": actor.actor_id, "identity_level": actor.identity_level,
                "identity_source": actor.identity_source, "identity_fallback_flag": actor.identity_fallback_flag,
            })
            replay_rows.append(row)
        elif row.get("event_class") in LIFECYCLE_BREAK_CLASSES:
            replay_rows.append(row)

    replay_events(raw_events, config=None, max_rows=None, hooks=ReplayHooks(on_pre_event=pre), retain_events=False)
    clusters, _ = cluster_execution_fills(replay_rows, max_gap_ms=100)
    rows = []
    for cluster in clusters:
        fill_ns = cluster.get("cluster_end_fill_anchor_ns")
        if type(fill_ns) is not int:
            raise ValueError("canonical cluster is missing its exact fill anchor")
        rows.append({
            "instrument": instrument, "partition_id": cluster["partition_id"],
            "event_date": cluster["cluster_start_ts"].date(),
            "actor_key": cluster["actor_key"], "actor_id": cluster["actor_id"],
            "identity_level": cluster["identity_level"], "identity_source": cluster["identity_source"],
            "identity_fallback_flag": cluster["identity_fallback_flag"], "event_side": cluster["event_side"],
            "execution_anchor_mode": cluster["execution_anchor_mode"], "cluster_end_ts": cluster["cluster_end_ts"],
            "cluster_last_sort_index": cluster["cluster_last_sort_index"],
            "execution_cluster_id": cluster["execution_cluster_id"], "execution_quantity": cluster["execution_quantity"],
            "clock_policy_id": CLOCK_POLICY_ID, "fill_anchor_ns": fill_ns,
        })
    return _frame(rows, _CLUSTER_SCHEMA)


def _offset_name(offset: int) -> str:
    return "zero" if offset == 0 else (f"plus_{offset}" if offset > 0 else f"minus_{abs(offset)}")


def _row_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    return tuple(row[field] for field in _GROUP_FIELDS)


def _contributions_for_window(result: EmpiricalControlsV2Result, *, window_seconds: float) -> list[dict[str, Any]]:
    state_rows = result.state_statistics.to_dicts()
    observed: dict[tuple[tuple[Any, ...], str], dict[str, Any]] = {}
    shifted: dict[tuple[tuple[Any, ...], str, int], dict[str, Any]] = {}
    for row in state_rows:
        role = {"own_passive_only": "passive", "own_aggressive_only": "aggressive", "own_mixed": "mixed"}.get(row["state"])
        if role is None:
            continue
        key = _row_key(row)
        if row["analysis_kind"] == "timing_shift_common_support":
            shifted[(key, role, int(row["offset_seconds"]))] = row
            if row["offset_seconds"] == 0:
                observed[(key, role)] = row
    rows: list[dict[str, Any]] = []
    for key, role in sorted(observed, key=str):
        source = observed[(key, role)]
        values = {offset: shifted.get((key, role, offset)) for offset in SHIFT_OFFSETS}
        all_rows = [source, *values.values()]
        anomaly = any(item is not None and bool(item["support_anomaly"]) for item in all_rows)
        positive = source["time_at_risk_seconds"] > 0 and all(
            item is not None and item["time_at_risk_seconds"] > 0 for item in values.values()
        )
        valid_offsets = sum(item is not None and item["time_at_risk_seconds"] > 0 and not item["support_anomaly"] for item in values.values())
        eligible = role != "mixed"
        expected = excess = None
        exclusion = None
        if anomaly:
            exclusion = "support_anomaly"
        elif not positive:
            exclusion = "missing_common_positive_risk_time"
        elif role == "mixed":
            exclusion = "mixed_role_nonadditive"
        else:
            rates = [values[offset]["withdrawal_count"] / values[offset]["time_at_risk_seconds"] for offset in _NONZERO_OFFSETS]
            expected = sum(rates) / len(rates) * source["time_at_risk_seconds"]
            excess = source["withdrawal_count"] - expected
        status = "included" if expected is not None else ("descriptive_nonadditive" if role == "mixed" and positive and not anomaly else "excluded_common_support")
        row = {**{field: source[field] for field in _GROUP_FIELDS},
               "execution_anchor_mode": role, "role_additivity": "nonadditive" if role == "mixed" else "separate",
               "proposal_eligible": eligible, "window_seconds": float(window_seconds),
               "support_horizon_seconds": _SUPPORT_HORIZON_SECONDS,
               "observed_count": source["withdrawal_count"], "observed_risk_seconds": source["time_at_risk_seconds"],
               "valid_shift_offset_count": valid_offsets, "support_anomaly": anomaly,
               "expected_background_count": expected, "excess_count": excess,
               "contribution_status": status, "exclusion_reason": exclusion, "clock_policy_id": CLOCK_POLICY_ID}
        for offset, item in values.items():
            name = _offset_name(offset)
            row[f"shift_{name}_count"] = None if item is None else item["withdrawal_count"]
            row[f"shift_{name}_risk_seconds"] = None if item is None else item["time_at_risk_seconds"]
        rows.append(row)
    return rows


def _curve_from_contributions(contributions: pl.DataFrame) -> pl.DataFrame:
    rows = []
    base = ("instrument", "identity_level", "execution_anchor_mode", "window_seconds")
    for key, group in contributions.group_by(list(base), maintain_order=True):
        named = dict(zip(base, key, strict=True))
        included = group.filter(pl.col("contribution_status") == "included")
        excess = included.get_column("excess_count").sum() if included.height else None
        rows.append({**named, "excess_count": excess, "included_group_count": included.height,
                     "excluded_group_count": group.height - included.height,
                     "supported_risk_seconds": included.get_column("observed_risk_seconds").sum() if included.height else 0.0,
                     "aggregation": "sum_group_excess_counts"})
    return _frame(rows, W_CURVE_SCHEMA).sort("instrument", "identity_level", "execution_anchor_mode", "window_seconds")


def propose_w_association(curve: pl.DataFrame) -> pl.DataFrame:
    """Apply the frozen plateau rule separately to passive/aggressive curves."""
    if set(curve.columns) != set(W_CURVE_SCHEMA):
        raise ValueError("curve must use the exact shifted-control W schema")
    rows = []
    for (instrument, identity, role), group in curve.group_by(["instrument", "identity_level", "execution_anchor_mode"], maintain_order=True):
        ordered = group.sort("window_seconds")
        included = int(ordered.get_column("included_group_count").sum())
        excluded = int(ordered.get_column("excluded_group_count").sum())
        windows, excess = ordered.get_column("window_seconds").to_list(), ordered.get_column("excess_count").to_list()
        if role == "mixed":
            proposal = {"window_seconds": None, "proposal_status": "not_applicable_mixed_nonadditive", "search_cap_binding": False}
        elif len(windows) < 3 or any(value is None for value in excess) or included == 0:
            proposal = {"window_seconds": None, "proposal_status": "insufficient_support", "search_cap_binding": False}
        else:
            proposal = proposed_window(windows, excess, search_cap_seconds=max(windows))
        rows.append({"instrument": instrument, "identity_level": identity, "execution_anchor_mode": role,
                     "proposed_window_seconds": proposal["window_seconds"], "proposal_status": proposal["proposal_status"],
                     "search_cap_binding": proposal["search_cap_binding"], "window_count": len(windows),
                     "included_group_count": included, "excluded_group_count": excluded})
    return _frame(rows, PROPOSAL_SCHEMA).sort("instrument", "identity_level", "execution_anchor_mode")


def run_shifted_control_w(
    raw_events: pl.DataFrame, *, instrument: str, windows_seconds: Sequence[float] = DEFAULT_WINDOWS_SECONDS,
    original_sort_indices: Mapping[int, int] | None = None,
    _prepared_risk: tuple[CompactRiskResult, float] | None = None,
) -> ShiftedControlWResult:
    """Run the W association curve on a full raw partition without detector gates."""
    if not isinstance(raw_events, pl.DataFrame):
        raise ValueError("raw_events must be a Polars DataFrame")
    if not isinstance(instrument, str) or not instrument:
        raise ValueError("instrument must be nonempty")
    windows = tuple(float(value) for value in windows_seconds)
    if not windows or any(not math.isfinite(value) or value <= 0 for value in windows) or tuple(sorted(set(windows))) != windows:
        raise ValueError("windows_seconds must be finite, positive, and strictly increasing")
    certification = certify_clock_equivalence(raw_events)
    if _prepared_risk is None:
        risk, _, age_cap = collect_uncapped_risk(raw_events, instrument=instrument, original_sort_indices=original_sort_indices)
    else:
        # Internal verified-stage path; the stage validates the entire reference
        # graph and cap/span diagnostics before supplying this cached population.
        risk, age_cap = _prepared_risk
        if not isinstance(risk, CompactRiskResult) or not math.isfinite(age_cap) or age_cap <= 0:
            raise ValueError("verified uncapped risk and positive age cap required")
    if "transition_reason" in risk.transitions.columns and "age_expiry" in risk.transitions.get_column("transition_reason").to_list():
        raise ValueError("uncapped risk adapter emitted age_expiry")
    clusters = _collect_canonical_clusters(raw_events, instrument=instrument, original_sort_indices=original_sort_indices)
    controls: dict[float, EmpiricalControlsV2Result] = {}
    contributions = []
    for window in windows:
        control = analyze_controls_v2(
            risk.intervals, risk.withdrawal_events, clusters, risk.coverage_epochs,
            market_intervals=risk.market_intervals, clock_policy_id=CLOCK_POLICY_ID,
            window_seconds=window, support_horizon_seconds=_SUPPORT_HORIZON_SECONDS,
            point_window_half_open=True,
        )
        controls[window] = control
        contributions.extend(_contributions_for_window(control, window_seconds=window))
    contribution_frame = _frame(contributions, CONTRIBUTION_SCHEMA)
    curve = _curve_from_contributions(contribution_frame)
    proposals = propose_w_association(curve)
    provenance = {
        "input_population": "all_valid_canonical_execution_fills", "candidate_gate": "not_applied",
        "execution_cluster_gap_ms": 100, "execution_anchor_modes": ["passive", "aggressive"],
        "identity_mode": "client_then_firm", "support_horizon_seconds": _SUPPORT_HORIZON_SECONDS,
        "point_window_half_open": True, "windows_seconds": list(windows),
        "common_shift_offsets_seconds": list(SHIFT_OFFSETS), "clock_equivalence": certification,
        "risk_age_cap_seconds": age_cap,
    }
    return ShiftedControlWResult(risk, age_cap, clusters, controls, contribution_frame, curve, proposals, provenance)


def serialize_w_association(result: ShiftedControlWResult) -> str:
    """Serialize public milestone tables with JSON nulls and no non-finite tokens."""
    if not isinstance(result, ShiftedControlWResult):
        raise ValueError("ShiftedControlWResult required")
    def default(value: Any) -> Any:
        if isinstance(value, (date, datetime)):
            return value.isoformat()
        raise TypeError(f"not JSON serializable: {type(value).__name__}")
    payload = {"provenance": result.provenance, "contributions": result.contributions.to_dicts(),
               "curve": result.curve.to_dicts(), "proposals": result.proposals.to_dicts()}
    return json.dumps(payload, default=default, allow_nan=False, sort_keys=True, separators=(",", ":"))
