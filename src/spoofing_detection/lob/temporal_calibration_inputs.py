"""Frozen temporal-calibration design and canonical raw-input adapters.

This module deliberately prepares observed population inputs only.  It neither
selects outcomes nor writes a calibration bundle; a runner owns those actions.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import hashlib
import json
import math
from pathlib import Path
from typing import Any, cast

import polars as pl

from .actor_identity import actor_identity_from_event, actor_identity_from_order
from .config import LOBConfig
from .empirical_clock import CLOCK_POLICY_ID, resolve_empirical_clocks
from .execution_clusters import LIFECYCLE_BREAK_CLASSES, classify_execution_anchor, cluster_execution_fills
from .panel import ReplayHooks, book_summary, replay_events
from .spoofing_metrics import compute_exploratory_metrics
from .withdrawal_risk_v2 import CompactRiskResult, compute_compact_risk
from scripts.validate_spoofing_detection_bundle import (
    V3_CLOCK_CONTRACT,
    V3_CLOCK_POLICY_ID,
    V3_OUTPUT_SCHEMA_VERSION,
)

DESIGN_VERSION = "temporal_calibration_v1"
_FROZEN_DESIGN: dict[str, Any] = {
    "design_version": DESIGN_VERSION,
    "source_registry": "configs/spoofing_empirical_controls.json",
    "paper_baseline_metadata": "paper/generated/metadata.json",
    "identity_levels": ["client_original", "firm"],
    "identity_mode": "client_then_firm",
    "top_n": 10,
    "quantiles": [0.5, 0.9, 0.95, 0.99],
    "withdrawal_windows_seconds": [0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20],
    "endpoint_horizons_seconds": [0.01, 0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 20],
    "timing_shift_seconds": [-60, -30, 0, 30, 60],
    "common_support_horizon_seconds": 20,
    "risanamento_calibration_last_date": "2024-09-30",
    "risanamento_validation_first_date": "2024-10-01",
    "bootstrap_seed": 20261005,
    "bootstrap_replicates": 500,
    "bootstrap_block_days": 5,
    "bootstrap_min_supported_days": 10,
    "tail_relative_tolerance": 0.1,
    "tau_reference_weights": [0.1, 0.5],
    "outcome_selection": "forbidden",
    "parameter_application": "proposal_only",
    "resources": {"buffer_rows": 10000, "max_rss_mb": 8192, "timeout_seconds": 14400},
}

ORIGIN_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.String, "partition_id": pl.String, "event_date": pl.Date,
    "physical_order_key": pl.String, "order_id": pl.String, "first_seen_local_index": pl.Int64,
    "first_seen_source_index": pl.Int64, "origin_replay_ns": pl.Int64,
    "origin_class": pl.String, "actor_key": pl.String, "identity_level": pl.String,
    "side": pl.String,
}
AGE_EXPOSURE_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.String, "partition_id": pl.String, "event_date": pl.Date,
    "physical_order_key": pl.String, "order_id": pl.String, "first_seen_sort_index": pl.Int64,
    "actor_key": pl.String, "identity_level": pl.String, "side": pl.String,
    "coverage_epoch_id": pl.String, "eligible_start_ns": pl.Int64, "eligible_end_ns": pl.Int64,
    "duration_ns": pl.Int64, "origin_replay_ns": pl.Int64, "age_start_ns": pl.Int64,
    "age_end_ns": pl.Int64, "start_reason": pl.String, "end_reason": pl.String,
    "left_censored": pl.Boolean, "right_censored": pl.Boolean,
    "weighting": pl.String,
}
AGE_SPELL_SCHEMA: dict[str, pl.DataType] = {
    **AGE_EXPOSURE_SCHEMA, "spell_membership_count": pl.Int64,
}
GAP_PAIR_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.String, "identity_level": pl.String, "actor_key": pl.String,
    "execution_anchor_mode": pl.String, "partition_id": pl.String, "event_date": pl.Date,
    "previous_child_source_index": pl.Int64, "child_source_index": pl.Int64, "gap_ns": pl.Int64,
}
MARKET_STATE_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.String, "partition_id": pl.String, "event_date": pl.Date,
    "source_index": pl.Int64, "local_sort_index": pl.Int64, "replay_time_ns": pl.Int64,
    "actor_key": pl.String, "identity_level": pl.String, "market_best_bid": pl.Float64,
    "market_best_ask": pl.Float64, "market_mid": pl.Float64,
}


def frozen_design() -> dict[str, Any]:
    """Return a fresh JSON-safe copy of the immutable proposal design."""
    return json.loads(json.dumps(_FROZEN_DESIGN))


def _finite_number(value: Any, *, positive: bool = False) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else True)


def validate_design(design: Mapping[str, Any]) -> None:
    """Reject any absent, malformed, or non-frozen primary setting."""
    if not isinstance(design, Mapping) or set(design) != set(_FROZEN_DESIGN):
        raise ValueError("design must contain exactly the frozen temporal-calibration keys")
    try:
        json.dumps(design, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("design must be JSON-safe without NaN or infinity") from exc
    for field in ("quantiles", "withdrawal_windows_seconds", "endpoint_horizons_seconds", "tau_reference_weights"):
        values = design[field]
        if not isinstance(values, list) or not values or any(not _finite_number(v, positive=True) for v in values):
            raise ValueError(f"{field} must be finite positive numeric values, never booleans")
    if (design["quantiles"] != sorted(design["quantiles"]) or any(v <= 0 or v >= 1 for v in design["quantiles"]) or
            design["withdrawal_windows_seconds"] != sorted(design["withdrawal_windows_seconds"]) or
            design["endpoint_horizons_seconds"] != sorted(design["endpoint_horizons_seconds"])):
        raise ValueError("probability/window ordering or bounds incompatible")
    for field in ("timing_shift_seconds",):
        values = design[field]
        if not isinstance(values, list) or not values or any(type(v) not in (int, float) or isinstance(v, bool) or not math.isfinite(v) for v in values):
            raise ValueError(f"{field} must be finite numeric values")
    for field in ("common_support_horizon_seconds", "bootstrap_seed", "bootstrap_replicates", "bootstrap_block_days", "bootstrap_min_supported_days", "tail_relative_tolerance"):
        if not _finite_number(design[field], positive=True):
            raise ValueError(f"{field} must be positive finite and not boolean")
    if any(type(design[f]) is not int or design[f] <= 0 for f in ("bootstrap_seed", "bootstrap_replicates", "bootstrap_block_days", "bootstrap_min_supported_days")):
        raise ValueError("bootstrap fields must be positive integers")
    if design["identity_levels"] != ["client_original", "firm"] or design["identity_mode"] != "client_then_firm":
        raise ValueError("identity policy must preserve client_then_firm")
    if design["outcome_selection"] != "forbidden" or design["parameter_application"] != "proposal_only":
        raise ValueError("outcome selection and parameter application are frozen")
    if date.fromisoformat(design["risanamento_calibration_last_date"]) >= date.fromisoformat(design["risanamento_validation_first_date"]):
        raise ValueError("chronological calibration/validation split is invalid")
    resources = design["resources"]
    if not isinstance(resources, Mapping) or set(resources) != {"buffer_rows", "max_rss_mb", "timeout_seconds"}:
        raise ValueError("resources must have exactly buffer_rows/max_rss_mb/timeout_seconds")
    if any(type(resources[k]) is not int or resources[k] <= 0 for k in resources):
        raise ValueError("resources must be positive integers")
    if json.dumps(design, sort_keys=True, separators=(",", ":")) != json.dumps(_FROZEN_DESIGN, sort_keys=True, separators=(",", ":")):
        raise ValueError("primary design settings are frozen; estimator split cannot silently differ")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1_048_576), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_index(local: int, original: Mapping[int, int] | None) -> int:
    value = local if original is None else original.get(local)
    if type(value) is not int or value <= 0:
        raise ValueError(f"original_sort_indices missing or invalid local sort_index {local}")
    return value


def _event_date(event: Mapping[str, Any]) -> date:
    raw = event.get("TRADEDATE")
    if isinstance(raw, datetime):
        return raw.date()
    return date.fromisoformat(str(raw)[:10])


def _partition(event: Mapping[str, Any]) -> str:
    return "|".join("" if event.get(k) is None else str(event.get(k)) for k in ("TRADEDATE", "MIC", "MARKETCODE", "SYMBOLINDEX", "EMM (*)"))


def _physical_order_key(partition_id: str, event_date: date, order_id: str, first_seen_source_index: int) -> str:
    """Match the canonical compact-risk physical lifecycle key exactly."""
    return "|".join((partition_id, event_date.isoformat(), order_id, str(first_seen_source_index)))


def _empty(schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    return pl.DataFrame(schema=schema)


def _frame(rows: list[dict[str, Any]], schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    if not rows:
        return _empty(schema)
    return pl.DataFrame(rows, infer_schema_length=None).select(pl.col(name).cast(dtype, strict=False) for name, dtype in schema.items())


def _ns_to_datetime(value: int) -> datetime:
    seconds, remainder = divmod(value, 1_000_000_000)
    return datetime(1970, 1, 1) + timedelta(seconds=seconds, microseconds=remainder // 1_000)


@dataclass(frozen=True)
class _OriginCollector:
    instrument: str
    original: Mapping[int, int] | None
    rows: list[dict[str, Any]]

    def post(self, event: dict[str, Any], active: dict[str, Any], partition_id: str | None, flags: list[str]) -> None:
        """Record only the order created by this observed new-order event."""
        if event.get("event_class") != "new_order":
            return
        local = int(event["sort_index"])
        order = active.get(str(event.get("ORDERID")))
        if order is None or int(order.first_seen_sort_index) != local:
            return
        actor = actor_identity_from_order(order)
        if actor is None:
            return
        source = _source_index(local, self.original)
        part, day = _partition(event), _event_date(event)
        self.rows.append({
            "instrument": self.instrument, "partition_id": part, "event_date": day,
            "physical_order_key": _physical_order_key(part, day, str(order.order_id), source),
            "order_id": str(order.order_id), "first_seen_local_index": local,
            "first_seen_source_index": source,
            "origin_replay_ns": resolve_empirical_clocks(event).replay_time_ns,
            "origin_class": "new_order", "actor_key": actor.actor_key,
            "identity_level": actor.identity_level, "side": order.side,
        })


def _partition_span_seconds(raw_events: pl.DataFrame) -> float:
    spans: dict[str, list[int]] = defaultdict(list)
    def seen(event: dict[str, Any], active: dict[str, Any], partition_id: str | None) -> None:
        value = resolve_empirical_clocks(event).replay_time_ns
        if value is not None:
            spans[_partition(event)].append(value)
    replay_events(raw_events, config=LOBConfig(snapshot_mode="none"), max_rows=None, hooks=ReplayHooks(on_pre_event=seen), retain_events=False)
    return max(((max(v) - min(v)) / 1_000_000_000 for v in spans.values() if v), default=0.0)


def collect_origins(raw_events: pl.DataFrame, original_sort_indices: Mapping[int, int] | None = None) -> tuple[pl.DataFrame, float]:
    """Observe canonical new-order origins without assigning a synthetic age zero."""
    rows: list[dict[str, Any]] = []
    collector = _OriginCollector("", original_sort_indices, rows)
    # Instrument is intentionally populated by callers that need it; origin identity is partition-local.
    replay_events(raw_events, config=LOBConfig(snapshot_mode="none"), max_rows=None,
                  hooks=ReplayHooks(on_post_event=collector.post), retain_events=False)
    return _frame(rows, ORIGIN_SCHEMA), max(120.0, _partition_span_seconds(raw_events) + 1.0)


def collect_uncapped_risk(raw_events: pl.DataFrame, *, instrument: str, original_sort_indices: Mapping[int, int] | None = None,
                          sink: Callable[[str, list[dict[str, Any]]], None] | None = None) -> tuple[CompactRiskResult, pl.DataFrame, float]:
    """Run canonical compact risk with an age cap strictly above every partition span."""
    origins, cap = collect_origins(raw_events, original_sort_indices)
    origins = origins.with_columns(pl.lit(instrument).alias("instrument"))
    result = compute_compact_risk(raw_events, instrument=instrument, top_n=10, max_order_age_seconds=cap,
                                  config=LOBConfig(snapshot_mode="none"),
                                  original_sort_indices=original_sort_indices, sink=sink,
                                  clock_policy_id=CLOCK_POLICY_ID, buffer_rows=10_000)
    return result, origins, cap


_CENSORED_MEMBERSHIP_END_REASONS = frozenset({
    "end_of_partition", "partition_boundary", "coverage_end", "clock_ambiguous", "quarantine",
})


def build_age_inputs(membership: pl.DataFrame, origins: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, dict[str, int]]:
    """Build physical-order-time age exposure and survival spells from risk output.

    The primary age population is restricted to membership intervals whose order
    origin was observed as a canonical ``new_order``.  A missing origin remains
    unknown rather than receiving an event-time age of zero.  Each membership
    interval supplies its elapsed order-time duration exactly once; the spell
    table only coalesces contiguous ``membership_change`` fragments within the
    same coverage epoch for censoring diagnostics.
    """
    membership_required = {
        "instrument", "physical_order_key", "partition_id", "event_date", "order_id",
        "first_seen_sort_index", "actor_key", "identity_level", "side", "coverage_epoch_id",
        "eligible_start_ns", "eligible_end_ns", "start_reason", "end_reason",
    }
    origin_required = {
        "instrument", "partition_id", "event_date", "physical_order_key", "order_id",
        "first_seen_source_index", "origin_replay_ns", "origin_class", "actor_key",
        "identity_level", "side",
    }
    if not membership_required <= set(membership.columns) or not origin_required <= set(origins.columns):
        raise ValueError("risk membership and observed-origin columns required")

    origin_by_key: dict[tuple[str, str, int], dict[str, Any]] = {}
    for origin in origins.iter_rows(named=True):
        key = (str(origin["partition_id"]), str(origin["order_id"]), int(origin["first_seen_source_index"]))
        if key in origin_by_key:
            raise ValueError("ambiguous observed origin for partition/order/first-seen source index")
        origin_by_key[key] = origin

    diagnostics: defaultdict[str, int] = defaultdict(int)
    exposures: list[dict[str, Any]] = []
    for item in membership.iter_rows(named=True):
        start, end = item["eligible_start_ns"], item["eligible_end_ns"]
        if type(start) is not int or type(end) is not int or end <= start:
            diagnostics["excluded_nonpositive_duration_count"] += 1
            continue
        key = (str(item["partition_id"]), str(item["order_id"]), int(item["first_seen_sort_index"]))
        origin = origin_by_key.get(key)
        if origin is None or origin["origin_class"] != "new_order" or type(origin["origin_replay_ns"]) is not int:
            diagnostics["unknown_origin_exposure_count"] += 1
            continue
        try:
            membership_key = _physical_order_key(
                str(item["partition_id"]), item["event_date"], str(item["order_id"]), int(item["first_seen_sort_index"]),
            )
            origin_key = _physical_order_key(
                str(origin["partition_id"]), origin["event_date"], str(origin["order_id"]), int(origin["first_seen_source_index"]),
            )
        except (TypeError, ValueError, AttributeError) as exc:
            raise ValueError("membership/origin physical provenance malformed") from exc
        if (item["physical_order_key"] != membership_key or origin["physical_order_key"] != origin_key
                or any(item[field] != origin[field] for field in (
                    "instrument", "partition_id", "event_date", "physical_order_key", "order_id",
                ))):
            raise ValueError("membership/origin physical provenance mismatch")
        if any(item[field] != origin[field] for field in ("actor_key", "identity_level", "side")):
            raise ValueError("membership/origin identity mismatch")
        origin_ns = origin["origin_replay_ns"]
        if origin_ns > start:
            diagnostics["excluded_origin_after_exposure_count"] += 1
            continue
        row = {
            "instrument": item["instrument"], "partition_id": item["partition_id"], "event_date": item["event_date"],
            "physical_order_key": item["physical_order_key"], "order_id": item["order_id"],
            "first_seen_sort_index": item["first_seen_sort_index"], "actor_key": item["actor_key"],
            "identity_level": item["identity_level"], "side": item["side"],
            "coverage_epoch_id": item["coverage_epoch_id"], "eligible_start_ns": start, "eligible_end_ns": end,
            "duration_ns": end - start, "origin_replay_ns": origin_ns, "age_start_ns": start - origin_ns,
            "age_end_ns": end - origin_ns, "start_reason": item["start_reason"], "end_reason": item["end_reason"],
            "left_censored": False, "right_censored": item["end_reason"] in _CENSORED_MEMBERSHIP_END_REASONS,
            "weighting": "physical_order_time",
        }
        exposures.append(row)
        diagnostics["known_origin_exposure_count"] += 1

    exposures.sort(key=lambda row: (
        row["partition_id"], row["order_id"], row["first_seen_sort_index"], row["coverage_epoch_id"],
        row["eligible_start_ns"], row["eligible_end_ns"],
    ))
    spells: list[dict[str, Any]] = []
    for row in exposures:
        if spells:
            previous = spells[-1]
            same_spell = all(previous[field] == row[field] for field in (
                "instrument", "partition_id", "event_date", "physical_order_key", "order_id",
                "first_seen_sort_index", "actor_key", "identity_level", "side", "coverage_epoch_id",
                "origin_replay_ns",
            ))
            contiguous_change = (previous["eligible_end_ns"] == row["eligible_start_ns"]
                                 and previous["end_reason"] == "membership_change"
                                 and row["start_reason"] == "membership_change")
            if same_spell and contiguous_change:
                previous["eligible_end_ns"] = row["eligible_end_ns"]
                previous["duration_ns"] += row["duration_ns"]
                previous["age_end_ns"] = row["age_end_ns"]
                previous["end_reason"] = row["end_reason"]
                previous["right_censored"] = row["right_censored"]
                previous["spell_membership_count"] += 1
                diagnostics["coalesced_membership_rows"] += 1
                continue
        spells.append({**row, "spell_membership_count": 1})
    diagnostics["age_exposure_count"] = len(exposures)
    diagnostics["spell_count"] = len(spells)
    return _frame(exposures, AGE_EXPOSURE_SCHEMA), _frame(spells, AGE_SPELL_SCHEMA), dict(diagnostics)


def collect_gap_pairs(raw_events: pl.DataFrame, *, instrument: str, original_sort_indices: Mapping[int, int] | None = None,
                      sink: Callable[[str, list[dict[str, Any]]], None] | None = None) -> tuple[pl.DataFrame, dict[str, int]]:
    """Build adjacent child-fill gaps from canonical pre-event callbacks.

    Fills require positive finite ``LASTSHARES``.  Lifecycle events are retained
    as clustering breaks; cluster membership supplies the audited pair boundary.
    """
    events: list[dict[str, Any]] = []
    diagnostics = defaultdict(int, {
        "raw_fill_count": 0,
        "accepted_fill_count": 0,
        "excluded_invalid_execution_role": 0,
        "excluded_missing_actor": 0,
        "excluded_missing_fill_or_replay_clock": 0,
        "excluded_nonpositive_or_invalid_lastshares": 0,
        "excluded_missing_or_invalid_execution_price": 0,
        "negative_adjacent_compatible_gap_count": 0,
    })
    # This is intentionally only an adjacency diagnostic.  Cluster membership
    # remains entirely owned by ``cluster_execution_fills``.
    previous_compatible_anchor: dict[tuple[Any, ...], int] = {}

    def compatibility_key(row: Mapping[str, Any], role: str) -> tuple[Any, ...]:
        """The canonical cluster grouping plus its passive-price condition."""
        return (
            row["partition_id"], str(row.get("ORDERID")), row["actor_key"], role,
            row.get("side_label"), row.get("execution_sweep_id"),
            row["event_price"] if role == "passive" else None,
            row["event_ts"].date(),
        )

    def pre(event: dict[str, Any], active: dict[str, Any], partition_id: str | None) -> None:
        local = int(event["sort_index"]); source = _source_index(local, original_sort_indices)
        decision = resolve_empirical_clocks(event)
        row = dict(event); row["partition_id"] = _partition(event); row["sort_index"] = source
        row["event_ts"] = _ns_to_datetime(decision.replay_time_ns) if decision.replay_time_ns is not None else None
        row["fill_anchor_ns"], row["replay_time_ns"] = decision.fill_anchor_ns, decision.replay_time_ns
        if row.get("event_class") == "fill":
            diagnostics["raw_fill_count"] += 1
            role = classify_execution_anchor(event)
            actor = actor_identity_from_event(event)
            quantity = event.get("LASTSHARES")
            if role is None:
                diagnostics["excluded_invalid_execution_role"] += 1
                return
            if actor is None:
                diagnostics["excluded_missing_actor"] += 1
                return
            fill_anchor_ns, replay_time_ns = decision.fill_anchor_ns, decision.replay_time_ns
            if type(fill_anchor_ns) is not int or type(replay_time_ns) is not int:
                diagnostics["excluded_missing_fill_or_replay_clock"] += 1
                return
            if type(quantity) not in (int, float) or isinstance(quantity, bool) or not math.isfinite(quantity) or quantity <= 0:
                diagnostics["excluded_nonpositive_or_invalid_lastshares"] += 1
                return
            if role == "aggressive":
                price, source_name = event.get("LASTTRADEDPX"), "LASTTRADEDPX"
            else:
                order = active.get(str(event.get("ORDERID")))
                price, source_name = (None if order is None else order.price), "active_order_price"
            if type(price) not in (int, float) or isinstance(price, bool) or not math.isfinite(price) or price <= 0:
                diagnostics["excluded_missing_or_invalid_execution_price"] += 1
                return
            row.update({
                "fill_qty": float(quantity), "event_price": float(price), "execution_price_source": source_name,
                "actor_key": actor.actor_key, "actor_id": actor.actor_id, "identity_level": actor.identity_level,
                "identity_source": actor.identity_source, "identity_fallback_flag": actor.identity_fallback_flag,
            })
            key = compatibility_key(row, role)
            previous_anchor = previous_compatible_anchor.get(key)
            if previous_anchor is not None and fill_anchor_ns < previous_anchor:
                diagnostics["negative_adjacent_compatible_gap_count"] += 1
            previous_compatible_anchor[key] = fill_anchor_ns
            diagnostics["accepted_fill_count"] += 1
            events.append(row)
        elif row.get("event_class") in LIFECYCLE_BREAK_CLASSES:
            order_id = event.get("ORDERID")
            if order_id is not None:
                lifecycle_key = (row["partition_id"], str(order_id))
                for key in list(previous_compatible_anchor):
                    if key[:2] == lifecycle_key:
                        previous_compatible_anchor.pop(key)
            events.append(row)
    replay_events(raw_events, config=LOBConfig(snapshot_mode="none"), max_rows=None,
                  hooks=ReplayHooks(on_pre_event=pre), retain_events=False)
    span_cap_ms = max(100.0, _partition_span_seconds(raw_events) * 1000.0 + 1.0)
    clusters, members = cluster_execution_fills(events, max_gap_ms=math.ceil(span_cap_ms))
    diagnostics["cluster_member_count"] = len(members)
    diagnostics["excluded_cluster_member_count"] = diagnostics["accepted_fill_count"] - len(members)
    if diagnostics["excluded_cluster_member_count"] < 0:
        raise ValueError("cluster members cannot exceed admitted fills")
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for member in members:
        by_cluster[str(member["execution_cluster_id"])].append(member)
    rows: list[dict[str, Any]] = []
    for cluster_id, children in by_cluster.items():
        ordered = sorted(children, key=lambda item: int(item["child_sort_index"]))
        for previous, child in zip(ordered, ordered[1:]):
            first, second = previous.get("child_fill_anchor_ns"), child.get("child_fill_anchor_ns")
            if type(first) is not int or type(second) is not int:
                diagnostics["excluded_missing_fill_anchor"] += 1
                continue
            rows.append({"instrument": instrument, "identity_level": child["identity_level"], "actor_key": child["actor_key"],
                         "execution_anchor_mode": child["execution_anchor_mode"], "partition_id": child["partition_id"],
                         "event_date": child["child_event_ts"].date(), "previous_child_source_index": previous["child_sort_index"],
                         "child_source_index": child["child_sort_index"], "gap_ns": second - first})
    diagnostics["pair_count"] = len(rows)
    if sink is not None and rows:
        sink("gap_pairs", rows)
        return _empty(GAP_PAIR_SCHEMA), dict(diagnostics)
    return _frame(rows, GAP_PAIR_SCHEMA), dict(diagnostics)


def collect_market_states(raw_events: pl.DataFrame, *, instrument: str, original_sort_indices: Mapping[int, int] | None = None,
                          sink: Callable[[str, list[dict[str, Any]]], None] | None = None) -> pl.DataFrame:
    """Emit only canonical post-event market summaries; no fill or candidate linkage."""
    rows: list[dict[str, Any]] = []
    def post(event: dict[str, Any], active: dict[str, Any], partition_id: str | None, flags: list[str]) -> None:
        local = int(event["sort_index"]); actor = actor_identity_from_event(event); clock = resolve_empirical_clocks(event)
        summary = book_summary(active, prefix="post", top_n=10)
        rows.append({"instrument": instrument, "partition_id": _partition(event), "event_date": _event_date(event),
                     "source_index": _source_index(local, original_sort_indices), "local_sort_index": local,
                     "replay_time_ns": clock.replay_time_ns, "actor_key": actor.actor_key if actor else None,
                     "identity_level": actor.identity_level if actor else None, "market_best_bid": summary["post_best_bid"],
                     "market_best_ask": summary["post_best_ask"], "market_mid": summary["post_mid"]})
    replay_events(raw_events, config=LOBConfig(snapshot_mode="none"), max_rows=None,
                  hooks=ReplayHooks(on_post_event=post), retain_events=False)
    if sink is not None and rows:
        sink("market_states", rows)
        return _empty(MARKET_STATE_SCHEMA)
    return _frame(rows, MARKET_STATE_SCHEMA)


def annotate_coverage_epochs(frame: pl.DataFrame, coverage: pl.DataFrame) -> pl.DataFrame:
    """Attach unambiguous coverage epochs using canonical replay-clock bounds.

    Quarantine ends are exclusive: the boundary event belongs to neither epoch.
    Other final boundaries remain inclusive so source-order ties are preserved.
    """
    required = {"partition_id", "coverage_epoch_id", "start_ns", "end_ns", "end_reason"}
    if not required <= set(coverage.columns) or not {"partition_id", "replay_time_ns"} <= set(frame.columns):
        raise ValueError("coverage/replay clock boundary columns required")
    rows = []
    for item in frame.iter_rows(named=True):
        t = item["replay_time_ns"]
        matches = coverage.filter((pl.col("partition_id") == item["partition_id"]) & (pl.col("start_ns") <= t) &
                                  ((pl.col("end_ns") > t) | ((pl.col("end_ns") == t) & (pl.col("end_reason") != "quarantine"))))
        if matches.height > 1:
            raise ValueError("ambiguous coverage epoch replay-clock tie")
        item["coverage_epoch_id"] = matches.item(0, "coverage_epoch_id") if matches.height else None
        rows.append(item)
    return pl.DataFrame(rows, schema={**frame.schema, "coverage_epoch_id": pl.String}, strict=False) if rows else frame.with_columns(pl.lit(None, dtype=pl.String).alias("coverage_epoch_id"))


def wide_metrics(raw_events: pl.DataFrame, *, tick_size: float, kernel_weights: Mapping[str, Mapping[int, float]],
                 age_cap: float, state_actor_keys: set[str] | None = None) -> Any:
    """Strict exploratory-metric adapter; all baseline metric inputs are explicit."""
    if not _finite_number(tick_size, positive=True) or not _finite_number(age_cap, positive=True):
        raise ValueError("tick_size and age_cap must be finite positive")
    return compute_exploratory_metrics(raw_events, top_n=10, tick_size=float(tick_size), window_seconds=10,
        withdrawal_window_seconds=20, reversion_horizon_seconds=20, max_rows=None, include_level_columns=True,
        max_deceptive_order_age_seconds=float(age_cap), execution_cluster_max_gap_ms=100,
        state_actor_keys=state_actor_keys, execution_anchor_modes=("passive", "aggressive"),
        empirical_kernel_weights=kernel_weights, use_v3_clocks=True)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _resolve(root: Path, name: str) -> Path:
    candidate = Path(name)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def preflight(design: Mapping[str, Any], instrument: str, output_root: Path | str, *, dates: list[str] | None = None,
              allow_existing: bool = False, project_root: Path | str = ".") -> dict[str, Any]:
    """Read-only input certification for a subsequent runner.

    No directory is made, no parquet is rewritten, and an existing output root is
    refused unless a future resume validator explicitly owns that path.
    """
    validate_design(design)
    root = Path(project_root).resolve(); output = Path(output_root).resolve()
    if output.exists() and not allow_existing:
        raise FileExistsError(f"output root already exists: {output}")
    registry_path = _resolve(root, design["source_registry"])
    metadata_path = _resolve(root, design["paper_baseline_metadata"])
    if not registry_path.is_file() or not metadata_path.is_file():
        raise ValueError("registry and paper baseline metadata must exist")
    registry, paper = _read_json(registry_path), _read_json(metadata_path)
    run = registry.get("baseline_runs", {}).get(instrument)
    if not isinstance(run, dict):
        raise ValueError(f"instrument absent from source registry: {instrument}")
    # This preflight consumes only the raw event input; quote/kernel registry
    # entries describe older baseline workflows and are deliberately not read.
    source_paths, source_hashes = run.get("source_paths"), run.get("source_hashes")
    if not isinstance(source_paths, dict) or not isinstance(source_hashes, dict):
        raise ValueError("registry source paths/hashes required")
    raw_relative = source_paths.get("raw_events")
    raw_expected = source_hashes.get("raw_events_sha256")
    if not isinstance(raw_relative, str) or not isinstance(raw_expected, str):
        raise ValueError("raw_events source path and hash required")
    raw_path = _resolve(root, raw_relative)
    if output == raw_path or output in raw_path.parents:
        raise ValueError("raw/baseline input cannot be output root")
    if not raw_path.is_file():
        raise ValueError("missing source input: raw_events")
    raw_actual = sha256(raw_path)
    if raw_actual != raw_expected:
        raise ValueError("source hash mismatch: raw_events")
    observed = {"raw_events": {"path": str(raw_path), "sha256": raw_actual}}
    try:
        schema = pl.read_parquet_schema(raw_path)
        raw_values = pl.scan_parquet(raw_path).select("TRADEDATE").unique().collect().get_column("TRADEDATE").to_list()
        raw_dates = []
        for value in raw_values:
            if isinstance(value, datetime):
                if value.time() != datetime.min.time():
                    raise ValueError("TRADEDATE datetime must be midnight")
                raw_dates.append(value.date().isoformat())
            elif isinstance(value, date):
                raw_dates.append(value.isoformat())
            elif isinstance(value, str) and date.fromisoformat(value).isoformat() == value:
                raw_dates.append(value)
            else:
                raise ValueError("invalid calendar TRADEDATE")
    except Exception as exc:
        raise ValueError("corrupt or unreadable raw parquet") from exc
    selected = dates if dates is not None else run.get("selected_event_dates")
    if not isinstance(selected, list) or not selected or len(set(selected)) != len(selected):
        raise ValueError("explicit unique selected dates required")
    if any(not isinstance(v, str) or date.fromisoformat(v).isoformat() != v for v in selected) or not set(selected) <= set(raw_dates):
        raise ValueError("selected dates malformed or absent from raw partitions")
    if not {"ORDERID", "TRADEDATE", "SEQUENCETIME", "FIRMID", "NMSC_ORIGINALCLIENTIDSHORTCODE"} <= set(schema):
        raise ValueError("raw parquet schema incompatible")

    baseline = paper.get("runs", {}).get(instrument)
    paper_hashes = baseline.get("input_hashes") if isinstance(baseline, dict) else None
    if not isinstance(baseline, dict) or not isinstance(paper_hashes, Mapping):
        raise ValueError("paper v3 baseline provenance required")
    run_relative, expected_metadata_hash = baseline.get("run_dir"), paper_hashes.get("run_metadata")
    if not isinstance(run_relative, str) or not isinstance(expected_metadata_hash, str):
        raise ValueError("paper run metadata hash required")
    run_dir = _resolve(root, run_relative)
    run_metadata_path = run_dir / "metadata.json"
    if not run_metadata_path.is_file():
        raise ValueError("v3 run metadata missing")
    run_metadata_hash = sha256(run_metadata_path)
    if run_metadata_hash != expected_metadata_hash:
        raise ValueError("paper run metadata hash mismatch")
    run_metadata = _read_json(run_metadata_path)
    for payload in (baseline, run_metadata):
        if (payload.get("output_schema_version") != V3_OUTPUT_SCHEMA_VERSION
                or payload.get("clock_policy_id") != V3_CLOCK_POLICY_ID
                or payload.get("clock_contract") != V3_CLOCK_CONTRACT):
            raise ValueError("exact v3 baseline clock/schema contract required")
    metadata_hashes = run_metadata.get("input_hashes")
    if not isinstance(metadata_hashes, Mapping) or metadata_hashes.get("raw_events_sha256") != raw_actual:
        raise ValueError("v3 run raw input fingerprint mismatch")
    config_value, config_expected = run_metadata.get("config"), run_metadata.get("config_sha256")
    if not isinstance(config_value, str) or not isinstance(config_expected, str):
        raise ValueError("v3 run config identity required")
    config_path = _resolve(root, config_value)
    if not config_path.is_file():
        raise ValueError("v3 run config missing")
    config_actual = sha256(config_path)
    if config_actual != config_expected or metadata_hashes.get("config_sha256") != config_actual:
        raise ValueError("v3 run config hash mismatch")
    baseline_config = _read_json(config_path)
    if baseline_config.get("spoofing_detection_clock_policy_v3") != {
        "policy_id": V3_CLOCK_POLICY_ID, "contract": V3_CLOCK_CONTRACT,
    }:
        raise ValueError("effective v3 run config clock policy incompatible")
    metrics = baseline_config.get("metrics", {})
    expected_metrics = {"top_n": 10, "withdrawal_window_seconds": 2.0, "max_deceptive_order_age_seconds": 90.0,
                        "actor_identity_mode": "client_then_firm", "execution_anchor_modes": ["passive", "aggressive"]}
    if any(metrics.get(key) != value for key, value in expected_metrics.items()):
        raise ValueError("baseline run configuration incompatible with frozen primary settings")
    tick_size = run_metadata.get("tick_size")
    if type(tick_size) not in (int, float) or not _finite_number(tick_size, positive=True):
        raise ValueError("actual v3 tick_size must be finite and positive")
    tick_size_value = float(cast(float, tick_size))
    return {"design": frozen_design(), "instrument": instrument, "selected_dates": list(selected),
            "observed_inputs": observed, "raw_schema": {name: str(dtype) for name, dtype in schema.items()},
            "source_identity": {"registry_sha256": sha256(registry_path), "paper_metadata_sha256": sha256(metadata_path),
                                "run_metadata_sha256": run_metadata_hash, "module_sha256": sha256(Path(__file__))},
            "baseline": {"clock_policy_id": V3_CLOCK_POLICY_ID, "clock_contract": V3_CLOCK_CONTRACT,
                         "tick_size": tick_size_value, "run_config": {"path": str(config_path), "sha256": config_actual,
                                                                         "metrics": expected_metrics}},
            "output_root": str(output)}

