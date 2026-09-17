#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from datetime import date, datetime, time
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
SCRIPTS_DIR = REPO_ROOT / "scripts"
for import_path in (SRC_DIR, SCRIPTS_DIR):
    if str(import_path) not in sys.path:
        sys.path.insert(0, str(import_path))

from build_spoofing_negative_control_report import build_empirical_report  # noqa: E402
from compute_spoofing_metrics import _infer_state_actor_keys  # noqa: E402
from spoofing_detection.lob.depth_kernel_calibration import load_empirical_kernel_weights  # noqa: E402
from spoofing_detection.lob.episode_artifacts import episode_frames  # noqa: E402
from spoofing_detection.lob.negative_controls import rescore_empirical_placebos  # noqa: E402
from spoofing_detection.lob.panel import SORT_COLUMNS, _partition_id, sort_events  # noqa: E402
from spoofing_detection.lob.replay_observation import (  # noqa: E402
    EVENT_TIMESTAMP_PRECEDENCE,
    choose_event_timestamp,
)
from spoofing_detection.lob.spoofing_metrics import (  # noqa: E402
    compute_exploratory_metrics,
    infer_tick_size_from_best_quotes,
)
from spoofing_detection.lob.withdrawal_risk import (  # noqa: E402
    CLOCK_REGRESSION_POLICY,
    compute_withdrawal_risk,
)

DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "spoofing_empirical_controls.json"
CANONICAL_PARTITION_COLUMNS = tuple(SORT_COLUMNS[:5])
SORT_INDEX_POLICY = "sort_index_is_derived_after_canonical_stable_sort"
REQUIRED_RAW_COLUMNS = set(SORT_COLUMNS)
REQUIRED_QUOTE_PANEL_COLUMNS = {
    "sort_index",
    "TRADEDATE",
    "pre_best_bid",
    "pre_best_ask",
    "post_best_bid",
    "post_best_ask",
}
REQUIRED_EMPIRICAL_KERNEL_COLUMNS = {"instrument_id", "side", "rank", "kernel_weight"}
REQUIRED_EXECUTION_CLUSTER_COLUMNS = {
    "execution_cluster_id", "partition_id", "actor_key", "actor_id", "identity_level",
    "identity_source", "identity_fallback_flag", "execution_anchor_mode", "execution_quantity",
    "event_side", "cluster_end_ts", "cluster_last_sort_index",
}
CANONICAL_BASELINE_ARTIFACTS = {
    "execution_metrics": "execution_metrics",
    "candidate_deceptive_orders": "candidate_deceptive_orders",
    "execution_cluster_members": "execution_cluster_members",
    "execution_cancel_candidates": "execution_cancel_candidates",
    "rejected_executions": "rejected_executions",
    "spoofing_compatible_events": "spoofing_compatible_events",
    "candidate_episodes": "candidate_episodes",
    "episode_cluster_members": "episode_cluster_members",
    "episode_withdrawals": "episode_withdrawals",
    "episode_anchor_summary": "episode_anchor_summary",
    "actor_day_episode_summary": "actor_day_episode_summary",
}
STATE_SERIES_EXCLUSION_REASON = (
    "excluded_from_rowwise_equivalence: state_time_series is potentially multi-million-row; "
    "all downstream empirical inputs are reconciled detector outputs listed in canonical_equivalence"
)
REQUIRED_STRATUM_COLUMNS = {
    "instrument",
    "HDR_PARTITIONID",
    "TRADEDATE",
    "actor_key",
    "posture_side",
}
REQUIRED_OUTPUT_SCHEMA = {
    "risk_spells": {"risk_spell_id", "instrument", "partition_id", "event_date", "actor_key", "posture_side"},
    "risk_intervals": {"risk_interval_id", "risk_spell_id", "execution_anchor_mode", "duration_seconds"},
    "exposure_intervals": {"execution_anchor_mode", "exposure_mask", "duration_seconds"},
    "risk_membership": {"risk_membership_id", "physical_order_key", "duration_seconds"},
    "withdrawal_events": {"withdrawal_event_id", "physical_order_key", "visible_qty_removed"},
    "episode_risk_links": {
        "instrument", "episode_id", "execution_cluster_id", "risk_spell_id",
        "risk_interval_id", "physical_order_key",
    },
    "placebo_schedules": {"draw_id", "schedule_block_id", "accepted", "rejection_reason"},
    "placebo_support": {"draw_id", "schedule_block_id", "accepted", "rejection_reason"},
    "placebo_statistics": {"draw_id", "row_kind", "time_at_risk_seconds", "withdrawal_count"},
    "coverage_and_balance": {"instrument", "event_date", "time_at_risk_seconds", "withdrawal_count"},
    "stratum_statistics": {"instrument", "contrast_label", "time_at_risk_seconds", "withdrawal_count"},
    "clock_coverage_audit": {
        "instrument", "partition_id", "event_date", "total_event_count",
        "accepted_event_count", "missing_timestamp_event_count",
        "clock_regression_event_count", "quarantined_event_count",
        "quarantine_episode_count", "recovered_quarantine_episode_count",
        "recovered_quarantine_duration_seconds", "unquantified_quarantine_event_count",
        "open_quarantine_at_end_flag", "first_accepted_ts", "last_accepted_ts",
        "observed_accepted_span_seconds", "time_at_risk_seconds",
        "coverage_loss_fraction", "terminal_right_censored_spell_count",
        "partition_boundary_censored_spell_count", "clock_ambiguous_censored_spell_count",
    },
    "pre_event_covariates": {
        "risk_interval_id", "contrast_label", "anchor_ts", "eligible_visible_qty",
        "member_count", "mean_member_age_seconds", "oldest_member_age_seconds",
        "pre_spread_ticks", "pre_total_depth", "prior_event_count_60s",
        "pre_mid_log_return_sd_60s", "activity_band", "snapshot_sort_index",
        "anchor_sort_index", "missing_reasons",
    },
    "common_support_audit": {
        "instrument", "event_date", "identity_level", "side", "comparison_label",
        "covariate", "total_seconds", "supported_seconds", "outside_support_seconds",
        "support_established", "supported", "null_fraction",
    },
}
CLOCK_COVERAGE_AUDIT_SCHEMA = {
    "instrument": pl.String,
    "partition_id": pl.String,
    "event_date": pl.Date,
    "total_event_count": pl.Int64,
    "accepted_event_count": pl.Int64,
    "missing_timestamp_event_count": pl.Int64,
    "clock_regression_event_count": pl.Int64,
    "quarantined_event_count": pl.Int64,
    "quarantine_episode_count": pl.Int64,
    "recovered_quarantine_episode_count": pl.Int64,
    "recovered_quarantine_duration_seconds": pl.Float64,
    "unquantified_quarantine_event_count": pl.Int64,
    "open_quarantine_at_end_flag": pl.Boolean,
    "first_accepted_ts": pl.Datetime("us"),
    "last_accepted_ts": pl.Datetime("us"),
    "observed_accepted_span_seconds": pl.Float64,
    "time_at_risk_seconds": pl.Float64,
    "coverage_loss_fraction": pl.Float64,
    "terminal_right_censored_spell_count": pl.Int64,
    "partition_boundary_censored_spell_count": pl.Int64,
    "clock_ambiguous_censored_spell_count": pl.Int64,
}
V1_REVERSION_HORIZON_SECONDS = 2.0
V1_HORIZON_BOUNDARY_POLICY = "no_cross_instrument_partition_or_day_horizons"
V1_STATE_TIMING = "pre_event"
V1_OUTCOME_OR_GATE_SELECTION = "forbidden"
V1_EXECUTION_ANCHOR_MODES = ["passive", "aggressive"]
V1_IDENTITY_MODE = "client_then_firm"
COMPARABILITY_CONTRACT = {
    "version": "pre_anchor_covariates_v1",
    "anchor": "exposure_contrast_start",
    "quote_boundary_policy": "latest_same_partition_at_or_before_anchor_pre_state_canonical_sort_tiebreak",
    "history_window_seconds": 60.0,
    "history_interval": "[anchor-60s,anchor)",
    "outcomes_used": False,
    "matching": "forbidden",
    "adjusted_effect": "forbidden",
}
PRE_EVENT_COVARIATES = (
    "eligible_visible_qty", "member_count", "mean_member_age_seconds",
    "oldest_member_age_seconds", "pre_spread_ticks", "pre_total_depth",
    "prior_event_count_60s", "pre_mid_log_return_sd_60s",
)
# This is used only for an entirely empty selected replay and is output-inert:
# no raw event can consume the tick, so exact live/frozen frame equivalence remains required.
EMPTY_CANONICAL_PLACEHOLDER_TICK_SIZE = 0.1
EMPTY_CANONICAL_EXECUTION_METRICS_SCHEMA = {
    "execution_cluster_id": pl.String,
    "partition_id": pl.String,
    "actor_key": pl.String,
    "actor_id": pl.String,
    "identity_level": pl.String,
    "identity_source": pl.String,
    "identity_fallback_flag": pl.Boolean,
    "execution_anchor_mode": pl.String,
    "execution_quantity": pl.Float64,
    "event_side": pl.String,
    "cluster_end_ts": pl.Datetime("us"),
    "cluster_last_sort_index": pl.Int64,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _nonempty_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _positive_finite_duration(value: object, *, name: str) -> float | int:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive duration")
    return value


def _path(value: object, *, name: str) -> Path:
    path = Path(_nonempty_string(value, name=name)).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def _require_hash(path: Path, expected: object, *, name: str) -> None:
    expected_text = _nonempty_string(expected, name=name)
    if not path.is_file():
        raise FileNotFoundError(f"required source not found: {path}")
    actual = _sha256(path)
    if actual != expected_text:
        raise ValueError(f"source hash mismatch for {name}: {path}")


def _require_parquet_schema(
    path: Path, *, required_columns: set[str], source_label: str, run_name: str
) -> None:
    try:
        schema = set(pl.scan_parquet(path).collect_schema().names())
    except Exception as exc:  # Polars errors vary by file backend and version.
        raise ValueError(f"{source_label} schema is unreadable for baseline run {run_name}") from exc
    missing = sorted(required_columns - schema)
    if missing:
        raise ValueError(f"{source_label} schema missing required columns: {', '.join(missing)}")


def _validate_clock(clock: Mapping[str, Any]) -> set[str]:
    required = {
        "event_timestamp_precedence",
        "partition_columns",
        "stable_sort_columns",
        "sort_index_policy",
        "risk_clock_regression_policy",
    }
    configured = set(clock)
    missing = sorted(required - configured)
    if missing:
        raise ValueError(f"clock missing required fields: {', '.join(missing)}")
    unexpected = sorted(configured - required)
    if unexpected:
        raise ValueError(f"clock has unsupported fields: {', '.join(unexpected)}")
    if clock["event_timestamp_precedence"] != list(EVENT_TIMESTAMP_PRECEDENCE):
        raise ValueError("clock.event_timestamp_precedence must match replay EVENT_TIMESTAMP_PRECEDENCE")
    if clock["partition_columns"] != list(CANONICAL_PARTITION_COLUMNS):
        raise ValueError("clock.partition_columns must match canonical replay partition columns")
    if clock["stable_sort_columns"] != SORT_COLUMNS:
        raise ValueError("clock.stable_sort_columns must match panel.SORT_COLUMNS")
    if clock["sort_index_policy"] != SORT_INDEX_POLICY:
        raise ValueError("clock.sort_index_policy must match the canonical stable-sort derivation policy")
    if clock["risk_clock_regression_policy"] != CLOCK_REGRESSION_POLICY:
        raise ValueError("clock.risk_clock_regression_policy must match withdrawal-risk clock policy")
    return set(EVENT_TIMESTAMP_PRECEDENCE) | set(CANONICAL_PARTITION_COLUMNS) | set(SORT_COLUMNS)


def _validate_output_schema(value: object) -> None:
    output_schema = _mapping(value, name="output_schema")
    for table, required_columns in REQUIRED_OUTPUT_SCHEMA.items():
        columns = output_schema.get(table)
        if not isinstance(columns, list) or not all(isinstance(column, str) for column in columns):
            raise ValueError(f"output_schema.{table} must declare its columns for zero-row output")
        missing = sorted(required_columns - set(columns))
        if missing:
            raise ValueError(f"output_schema.{table} missing required columns: {', '.join(missing)}")


def _validate_comparability_contract(value: object) -> None:
    """Require the fixed no-outcome pre-anchor comparability design."""
    contract = _mapping(value, name="comparability")
    if dict(contract) != COMPARABILITY_CONTRACT:
        raise ValueError("comparability must exactly match the pre-anchor no-adjustment contract")


def _validate_activity_bands(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise ValueError("strata.activity_bands must be a non-empty prespecified list")
    parsed: list[tuple[time, time, dict[str, Any]]] = []
    names: set[str] = set()
    for index, raw in enumerate(value):
        band = dict(_mapping(raw, name=f"strata.activity_bands[{index}]"))
        name = _nonempty_string(band.get("name"), name=f"strata.activity_bands[{index}].name")
        if name in names:
            raise ValueError("strata.activity_bands names must be unique")
        names.add(name)
        try:
            start = time.fromisoformat(_nonempty_string(band.get("start_time"), name="start_time"))
            end = time.fromisoformat(_nonempty_string(band.get("end_time"), name="end_time"))
        except ValueError as exc:
            raise ValueError("activity band times must use ISO local-time format") from exc
        if start >= end:
            raise ValueError("activity bands must have positive, non-overnight windows")
        parsed.append((start, end, {"name": name, "start_time": start, "end_time": end}))
    parsed.sort(key=lambda item: item[0])
    if any(current[0] < previous[1] for previous, current in zip(parsed, parsed[1:])):
        raise ValueError("activity bands must not overlap")
    return [item[2] for item in parsed]


def _validate_design_contract(config: Mapping[str, Any]) -> tuple[Path, set[str], dict[str, Any]]:
    if _nonempty_string(config.get("design_version"), name="design_version") != "empirical_controls_v1":
        raise ValueError("design_version must be empirical_controls_v1")
    output_root = _path(config.get("output_root"), name="output_root")
    if output_root.exists():
        raise FileExistsError(f"output already exists and will not be overwritten: {output_root}")

    clock_columns = _validate_clock(_mapping(config.get("clock"), name="clock"))
    horizons = _mapping(config.get("horizons"), name="horizons")
    if horizons.get("anchor") != "execution_cluster_end":
        raise ValueError("horizons.anchor must be execution_cluster_end")
    _positive_finite_duration(
        horizons.get("withdrawal_window_seconds"), name="horizons.withdrawal_window_seconds"
    )
    reversion_horizon_seconds = _positive_finite_duration(
        horizons.get("reversion_horizon_seconds"), name="horizons.reversion_horizon_seconds"
    )
    if reversion_horizon_seconds != V1_REVERSION_HORIZON_SECONDS:
        raise ValueError(
            f"horizons.reversion_horizon_seconds must be {V1_REVERSION_HORIZON_SECONDS}"
        )
    if horizons.get("inclusive_upper_bound") is not True:
        raise ValueError("horizons.inclusive_upper_bound must be true")
    if horizons.get("boundary_policy") != V1_HORIZON_BOUNDARY_POLICY:
        raise ValueError(
            f"horizons.boundary_policy must be {V1_HORIZON_BOUNDARY_POLICY}"
        )

    eligibility = _mapping(config.get("eligibility_policy"), name="eligibility_policy")
    if eligibility.get("active_visible_only") is not True:
        raise ValueError("eligibility_policy.active_visible_only must be true")
    if type(eligibility.get("top_n")) is not int or eligibility["top_n"] <= 0:
        raise ValueError("eligibility_policy.top_n must be positive")
    _positive_finite_duration(
        eligibility.get("max_order_age_seconds"), name="eligibility_policy.max_order_age_seconds"
    )
    if eligibility.get("state_timing") != V1_STATE_TIMING:
        raise ValueError(f"eligibility_policy.state_timing must be {V1_STATE_TIMING}")
    if eligibility.get("outcome_or_gate_selection") != V1_OUTCOME_OR_GATE_SELECTION:
        raise ValueError(
            "eligibility_policy.outcome_or_gate_selection must be "
            f"{V1_OUTCOME_OR_GATE_SELECTION}"
        )

    strata = _mapping(config.get("strata"), name="strata")
    block_columns = strata.get("block_columns")
    if not isinstance(block_columns, list) or not all(isinstance(column, str) for column in block_columns):
        raise ValueError("strata.block_columns must be a list of columns")
    missing_boundaries = sorted(REQUIRED_STRATUM_COLUMNS - set(block_columns))
    if missing_boundaries or strata.get("allow_cross_boundary") is not False:
        details = ", ".join(missing_boundaries) if missing_boundaries else "allow_cross_boundary"
        raise ValueError(f"split boundaries must remain disjoint by instrument, partition, and day: {details}")
    if strata.get("execution_anchor_modes") != V1_EXECUTION_ANCHOR_MODES:
        raise ValueError("strata.execution_anchor_modes must be ['passive', 'aggressive']")
    if strata.get("identity_mode") != V1_IDENTITY_MODE:
        raise ValueError(f"strata.identity_mode must be {V1_IDENTITY_MODE}")
    _validate_activity_bands(strata.get("activity_bands"))

    if type(config.get("seed")) is not int:
        raise ValueError("seed must be an integer")
    if type(config.get("placebo_draws")) is not int or config["placebo_draws"] <= 0:
        raise ValueError("placebo_draws must be positive")
    _validate_output_schema(config.get("output_schema"))
    _validate_comparability_contract(config.get("comparability"))
    return output_root, clock_columns, {
        "top_n": eligibility["top_n"],
        "withdrawal_window_seconds": horizons["withdrawal_window_seconds"],
        "reversion_horizon_seconds": horizons["reversion_horizon_seconds"],
        "max_deceptive_order_age_seconds": eligibility["max_order_age_seconds"],
        "actor_identity_mode": strata["identity_mode"],
        "execution_anchor_modes": strata["execution_anchor_modes"],
    }


def _load_effective_metrics(
    baseline_config_path: Path, *, name: str, expected_parameters: Mapping[str, Any]
) -> dict[str, Any]:
    try:
        baseline_config = _mapping(
            json.loads(baseline_config_path.read_text()),
            name=f"baseline config {baseline_config_path}",
        )
    except json.JSONDecodeError as exc:
        raise ValueError(f"baseline config is not valid JSON: {baseline_config_path}") from exc
    metrics = _mapping(
        baseline_config.get("metrics"),
        name=f"baseline config metrics for {name}",
    )
    effective_parameters = {
        str(key): value for key, value in metrics.items() if not str(key).startswith("_comment_")
    }
    for key, expected in expected_parameters.items():
        if key not in effective_parameters:
            raise ValueError(f"baseline config metrics missing required parameter: {key}")
        if effective_parameters[key] != expected:
            raise ValueError(
                f"baseline config metrics {key} does not match empirical design for {name}"
            )
    return effective_parameters


def _validate_canonical_baseline_artifacts(
    metadata_path: Path, metadata: Mapping[str, Any], *, name: str
) -> None:
    """Verify all frozen outputs needed to gate a live canonical rescore."""
    hashes = _mapping(metadata.get("artifact_hashes"), name="baseline metadata artifact_hashes")
    for artifact_name in CANONICAL_BASELINE_ARTIFACTS:
        path = metadata_path.parent / f"{artifact_name}.parquet"
        _require_hash(
            path,
            hashes.get(artifact_name),
            name=f"baseline_runs.{name}.{artifact_name}",
        )
        _require_parquet_schema(
            path, required_columns=set(), source_label=artifact_name, run_name=name
        )


def _validate_baseline_run(
    *,
    name: str,
    run: Mapping[str, Any],
    clock_columns: set[str],
    output_root: Path,
    expected_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    metadata_path = _path(run.get("baseline_metadata"), name=f"baseline_runs.{name}.baseline_metadata")
    source_paths = _mapping(run.get("source_paths"), name=f"baseline_runs.{name}.source_paths")
    source_hashes = _mapping(run.get("source_hashes"), name=f"baseline_runs.{name}.source_hashes")
    expected_path_keys = {
        "raw_events": "input",
        "quote_panel": "quote_panel",
        "empirical_depth_kernel": "empirical_depth_kernel",
        "baseline_config": "config",
    }
    expected_hash_keys = {
        "raw_events": "raw_events_sha256",
        "quote_panel": "quote_panel_sha256",
        "empirical_depth_kernel": "empirical_depth_kernel_sha256",
        "baseline_config": "baseline_config_sha256",
    }

    _require_hash(
        metadata_path,
        source_hashes.get("baseline_metadata_sha256"),
        name=f"baseline_runs.{name}.baseline_metadata_sha256",
    )
    try:
        metadata = _mapping(json.loads(metadata_path.read_text()), name=f"baseline metadata {metadata_path}")
    except json.JSONDecodeError as exc:
        raise ValueError(f"baseline metadata is not valid JSON: {metadata_path}") from exc
    execution_metrics_path = metadata_path.parent / "execution_metrics.parquet"
    artifact_hashes = _mapping(metadata.get("artifact_hashes"), name="baseline metadata artifact_hashes")
    _require_hash(
        execution_metrics_path,
        artifact_hashes.get("execution_metrics"),
        name=f"baseline_runs.{name}.execution_metrics",
    )
    _require_parquet_schema(
        execution_metrics_path,
        required_columns=REQUIRED_EXECUTION_CLUSTER_COLUMNS,
        source_label="execution_metrics",
        run_name=name,
    )
    _validate_canonical_baseline_artifacts(metadata_path, metadata, name=name)

    for key, expected in {
        "parameter_source": "json_config_only",
        "config_section": "metrics",
        "output_schema_version": "actor_execution_anchor_v2",
        "actor_identity_mode": "client_then_firm",
    }.items():
        if key not in metadata:
            raise ValueError(f"baseline metadata missing required key: {key}")
        if metadata[key] != expected:
            raise ValueError(f"baseline metadata has incompatible {key}: {metadata[key]!r}")
    if metadata.get("execution_anchor_modes") != ["passive", "aggressive"]:
        raise ValueError("baseline metadata has incompatible execution_anchor_modes")

    metadata_hashes = _mapping(metadata.get("input_hashes"), name="baseline metadata input_hashes")
    resolved_sources: dict[str, Path] = {}
    for source_key, metadata_key in expected_path_keys.items():
        configured_path = _path(source_paths.get(source_key), name=f"baseline_runs.{name}.{source_key}")
        if metadata_key not in metadata:
            raise ValueError(f"baseline metadata missing required path: {metadata_key}")
        metadata_path_value = _path(metadata[metadata_key], name=f"baseline metadata.{metadata_key}")
        if configured_path != metadata_path_value:
            raise ValueError(f"baseline metadata path mismatch for {source_key}")
        if configured_path == output_root or output_root in configured_path.parents:
            raise ValueError("output_root must be isolated from baseline sources")
        _require_hash(
            configured_path,
            source_hashes.get(expected_hash_keys[source_key]),
            name=f"baseline_runs.{name}.{expected_hash_keys[source_key]}",
        )
        metadata_hash_key = "config_sha256" if source_key == "baseline_config" else expected_hash_keys[source_key]
        if metadata_hashes.get(metadata_hash_key) != source_hashes[expected_hash_keys[source_key]]:
            raise ValueError(f"baseline metadata hash mismatch for {source_key}")
        resolved_sources[source_key] = configured_path

    effective_parameters = _load_effective_metrics(
        resolved_sources["baseline_config"],
        name=name,
        expected_parameters=expected_parameters,
    )

    selected_dates = run.get("selected_event_dates")
    if not isinstance(selected_dates, list) or not selected_dates or not all(
        isinstance(value, str) and value for value in selected_dates
    ):
        raise ValueError(f"baseline_runs.{name}.selected_event_dates must be a non-empty explicit list")
    if len(selected_dates) != len(set(selected_dates)):
        raise ValueError(f"baseline_runs.{name}.selected_event_dates must not contain duplicates")
    for selected_date in selected_dates:
        try:
            parsed_date = date.fromisoformat(selected_date)
        except ValueError as exc:
            raise ValueError(
                f"baseline_runs.{name}.selected_event_dates must use strict ISO YYYY-MM-DD dates"
            ) from exc
        if parsed_date.isoformat() != selected_date:
            raise ValueError(
                f"baseline_runs.{name}.selected_event_dates must use strict ISO YYYY-MM-DD dates"
            )

    _require_parquet_schema(
        resolved_sources["raw_events"],
        required_columns=REQUIRED_RAW_COLUMNS | clock_columns,
        source_label="raw input",
        run_name=name,
    )
    _require_parquet_schema(
        resolved_sources["quote_panel"],
        required_columns=REQUIRED_QUOTE_PANEL_COLUMNS,
        source_label="quote panel",
        run_name=name,
    )
    _require_parquet_schema(
        resolved_sources["empirical_depth_kernel"],
        required_columns=REQUIRED_EMPIRICAL_KERNEL_COLUMNS,
        source_label="empirical depth kernel",
        run_name=name,
    )
    return effective_parameters


def validate_preflight(config_path: Path) -> dict[str, Any]:
    try:
        config = _mapping(json.loads(config_path.read_text()), name="empirical controls config")
    except json.JSONDecodeError as exc:
        raise ValueError(f"empirical controls config is not valid JSON: {config_path}") from exc
    if config.get("design_version") == "empirical_controls_v2_light":
        from spoofing_detection.lob.empirical_controls_v2_io import validate_preflight_v2
        return validate_preflight_v2(config_path)
    output_root, clock_columns, expected_parameters = _validate_design_contract(config)
    baseline_runs = _mapping(config.get("baseline_runs"), name="baseline_runs")
    if not baseline_runs:
        raise ValueError("baseline_runs must not be empty")
    effective_parameters: dict[str, dict[str, Any]] = {}
    for name, run_value in baseline_runs.items():
        run_name = _nonempty_string(name, name="baseline run name")
        effective_parameters[run_name] = _validate_baseline_run(
            name=run_name,
            run=_mapping(run_value, name=f"baseline_runs.{run_name}"),
            clock_columns=clock_columns,
            output_root=output_root,
            expected_parameters=expected_parameters,
        )
    return {
        "design_version": config["design_version"],
        "output_root": output_root,
        "run_count": len(baseline_runs),
        "effective_parameters": effective_parameters,
    }


def _concat(frames: list[pl.DataFrame]) -> pl.DataFrame:
    if not frames:
        return pl.DataFrame()
    return pl.concat(frames, how="diagonal_relaxed")


def _clock_event_date(row: Mapping[str, Any], event_ts: datetime | None) -> date:
    raw = row.get("TRADEDATE")
    if raw is not None:
        try:
            return date.fromisoformat(str(raw)[:10])
        except ValueError:
            pass
    if event_ts is not None:
        return event_ts.date()
    raise ValueError("clock coverage audit requires TRADEDATE or an observable event timestamp")


def _clock_audit_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(schema=CLOCK_COVERAGE_AUDIT_SCHEMA)
    return pl.DataFrame(rows, infer_schema_length=None).select(
        pl.col(name).cast(dtype, strict=False) for name, dtype in CLOCK_COVERAGE_AUDIT_SCHEMA.items()
    )


def build_clock_coverage_audit(
    instrument: str,
    raw_events: pl.DataFrame,
    *,
    intervals: pl.DataFrame,
    diagnostics: pl.DataFrame,
    spells: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Audit observable replay-clock coverage in exact canonical event order.

    A missing timestamp starts an unknown-duration quarantine.  It closes only at
    the next accepted boundary, and does not permit state or elapsed time to be
    carried across it.  A regressed timestamp remains quarantined until a later
    timestamp reaches the prior accepted high-water mark; only those uninterrupted
    regression episodes contribute quantified (therefore lower-bound) loss.
    """
    del diagnostics  # Reconciliation is performed after all instrument frames are collected.
    if raw_events.is_empty():
        return _clock_audit_frame([])

    risk_seconds: dict[tuple[str, date], float] = {}
    if not intervals.is_empty():
        for row in intervals.select("partition_id", "event_date", "duration_seconds").iter_rows(named=True):
            key = (str(row["partition_id"]), row["event_date"])
            risk_seconds[key] = risk_seconds.get(key, 0.0) + float(row["duration_seconds"] or 0.0)
    censored: dict[tuple[str, date, str], int] = {}
    if spells is not None and not spells.is_empty():
        for row in spells.select("partition_id", "event_date", "end_reason").iter_rows(named=True):
            key = (str(row["partition_id"]), row["event_date"], str(row["end_reason"]))
            censored[key] = censored.get(key, 0) + 1

    rows: list[dict[str, Any]] = []
    sorted_rows = sort_events(raw_events).iter_rows(named=True)
    current_key: tuple[str, date] | None = None
    state: dict[str, Any] | None = None

    def finish() -> None:
        if state is None or current_key is None:
            return
        episode_kind = state["episode_kind"]
        if episode_kind is not None:
            state["open_quarantine_at_end_flag"] = True
            state["unquantified_quarantine_event_count"] += state["episode_regression_event_count"]
        first = state["first_accepted_ts"]
        last = state["last_accepted_ts"]
        span = (last - first).total_seconds() if first is not None and last is not None else 0.0
        recovered = state["recovered_quarantine_duration_seconds"]
        denominator = span + recovered
        rows.append({
            "instrument": instrument,
            "partition_id": current_key[0],
            "event_date": current_key[1],
            "total_event_count": state["total_event_count"],
            "accepted_event_count": state["accepted_event_count"],
            "missing_timestamp_event_count": state["missing_timestamp_event_count"],
            "clock_regression_event_count": state["clock_regression_event_count"],
            "quarantined_event_count": state["quarantined_event_count"],
            "quarantine_episode_count": state["quarantine_episode_count"],
            "recovered_quarantine_episode_count": state["recovered_quarantine_episode_count"],
            "recovered_quarantine_duration_seconds": recovered,
            "unquantified_quarantine_event_count": state["unquantified_quarantine_event_count"],
            "open_quarantine_at_end_flag": state["open_quarantine_at_end_flag"],
            "first_accepted_ts": first,
            "last_accepted_ts": last,
            "observed_accepted_span_seconds": span,
            "time_at_risk_seconds": risk_seconds.get(current_key, 0.0),
            # This omits all missing/open loss and is explicitly a quantified lower bound.
            "coverage_loss_fraction": recovered / denominator if denominator > 0 else None,
            "terminal_right_censored_spell_count": censored.get((*current_key, "coverage_end"), 0),
            "partition_boundary_censored_spell_count": censored.get((*current_key, "partition_boundary"), 0),
            "clock_ambiguous_censored_spell_count": censored.get((*current_key, "clock_ambiguous"), 0),
        })

    for raw in sorted_rows:
        event_ts = choose_event_timestamp(raw)
        key = (str(_partition_id(raw)), _clock_event_date(raw, event_ts))
        if key != current_key:
            finish()
            current_key = key
            state = {
                "total_event_count": 0, "accepted_event_count": 0,
                "missing_timestamp_event_count": 0, "clock_regression_event_count": 0,
                "quarantined_event_count": 0, "quarantine_episode_count": 0,
                "recovered_quarantine_episode_count": 0,
                "recovered_quarantine_duration_seconds": 0.0,
                "unquantified_quarantine_event_count": 0,
                "open_quarantine_at_end_flag": False,
                "first_accepted_ts": None, "last_accepted_ts": None,
                "high_watermark": None, "episode_kind": None,
                "episode_regression_event_count": 0,
            }
        assert state is not None
        state["total_event_count"] += 1
        if event_ts is None:
            state["missing_timestamp_event_count"] += 1
            state["quarantined_event_count"] += 1
            state["unquantified_quarantine_event_count"] += 1
            if state["episode_kind"] == "regression":
                state["unquantified_quarantine_event_count"] += state["episode_regression_event_count"]
                state["quarantine_episode_count"] += 1
                state["episode_kind"] = "missing"
                state["episode_regression_event_count"] = 0
            elif state["episode_kind"] is None:
                state["quarantine_episode_count"] += 1
                state["episode_kind"] = "missing"
            continue
        if state["high_watermark"] is not None and event_ts < state["high_watermark"]:
            state["clock_regression_event_count"] += 1
            state["quarantined_event_count"] += 1
            if state["episode_kind"] is None:
                state["quarantine_episode_count"] += 1
                state["episode_kind"] = "regression"
            state["episode_regression_event_count"] += 1
            continue

        state["accepted_event_count"] += 1
        state["first_accepted_ts"] = state["first_accepted_ts"] or event_ts
        state["last_accepted_ts"] = event_ts
        if state["episode_kind"] == "regression":
            state["recovered_quarantine_episode_count"] += 1
            state["recovered_quarantine_duration_seconds"] += (
                event_ts - state["high_watermark"]
            ).total_seconds()
        elif state["episode_kind"] == "missing":
            state["unquantified_quarantine_event_count"] += state["episode_regression_event_count"]
        state["episode_kind"] = None
        state["episode_regression_event_count"] = 0
        state["high_watermark"] = event_ts
    finish()
    return _clock_audit_frame(rows)


def _validate_clock_coverage_reconciliation(
    audit: pl.DataFrame, diagnostics: pl.DataFrame, intervals: pl.DataFrame,
) -> dict[str, bool]:
    """Fail closed unless the published audit equals risk diagnostics and intervals."""
    expected: dict[tuple[str, str], dict[str, int]] = {}
    for row in audit.iter_rows(named=True):
        key = (str(row["instrument"]), str(row["partition_id"]))
        if key in expected:
            raise AssertionError("clock coverage audit must have one row per instrument partition and day")
        expected[key] = {
            "missing_event_timestamp": int(row["missing_timestamp_event_count"]),
            "clock_regression": int(row["clock_regression_event_count"]),
        }
        if int(row["accepted_event_count"]) + int(row["quarantined_event_count"]) != int(row["total_event_count"]):
            raise AssertionError("clock coverage audit accepted and quarantined counts do not total events")
        if int(row["missing_timestamp_event_count"]) + int(row["clock_regression_event_count"]) != int(row["quarantined_event_count"]):
            raise AssertionError("clock coverage audit quarantine reason counts do not reconcile")
    observed = {key: {"missing_event_timestamp": 0, "clock_regression": 0} for key in expected}
    for row in diagnostics.filter(pl.col("reason").is_in(["missing_event_timestamp", "clock_regression"])).iter_rows(named=True):
        key = (str(row["instrument"]), str(row["partition_id"]))
        if key not in observed:
            raise AssertionError("risk diagnostic has no clock coverage audit partition")
        observed[key][str(row["reason"])] += 1
    if observed != expected:
        raise AssertionError("clock coverage audit does not reconcile to risk diagnostics")
    risk_seconds: dict[tuple[str, str, date], float] = {}
    for row in intervals.select("instrument", "partition_id", "event_date", "duration_seconds").iter_rows(named=True):
        key = (str(row["instrument"]), str(row["partition_id"]), row["event_date"])
        risk_seconds[key] = risk_seconds.get(key, 0.0) + float(row["duration_seconds"] or 0.0)
    for row in audit.iter_rows(named=True):
        key = (str(row["instrument"]), str(row["partition_id"]), row["event_date"])
        if not math.isclose(float(row["time_at_risk_seconds"]), risk_seconds.get(key, 0.0), rel_tol=1e-9, abs_tol=1e-9):
            raise AssertionError("clock coverage audit does not reconcile to risk intervals")
    return {
        "clock_coverage_counts_reconcile_to_risk_diagnostics": True,
        "clock_coverage_time_at_risk_reconciles_to_risk_intervals": True,
    }


def _instrument_frame(frame: pl.DataFrame, instrument: str, **aliases: str) -> pl.DataFrame:
    expressions = [pl.lit(instrument).alias("instrument")]
    expressions.extend(pl.col(source).alias(target) for target, source in aliases.items())
    return frame.with_columns(expressions)


def _read_execution_clusters(
    clusters: pl.DataFrame, selected_dates: set[str], activity_bands: list[dict[str, Any]],
) -> tuple[pl.DataFrame, int]:
    """Keep only selected, prespecified-band live canonical execution clusters."""
    if clusters.is_empty():
        return clusters.with_columns(pl.lit(None, dtype=pl.String).alias("activity_band")), 0
    dated = clusters.filter(pl.col("cluster_end_ts").dt.date().cast(pl.String).is_in(selected_dates))
    rows: list[dict[str, Any]] = []
    excluded = 0
    for row in dated.iter_rows(named=True):
        anchor = row["cluster_end_ts"]
        matched = next(
            (band for band in activity_bands if band["start_time"] <= anchor.time() < band["end_time"]),
            None,
        )
        if matched is None:
            excluded += 1
            continue
        row["activity_band"] = f"{anchor.date().isoformat()}:{matched['name']}"
        rows.append(row)
    if not rows:
        return dated.head(0).with_columns(pl.lit(None, dtype=pl.String).alias("activity_band")), excluded
    return pl.DataFrame(rows, infer_schema_length=None), excluded


def _selected_date_frame(frame: pl.DataFrame, selected_dates: set[str]) -> pl.DataFrame:
    """Filter selected trading dates without sorting or otherwise changing event order."""
    return frame.filter(
        pl.col("TRADEDATE").cast(pl.Date).is_in(
            [date.fromisoformat(value) for value in sorted(selected_dates)]
        )
    )


def _canonical_live_frames(
    run: Mapping[str, Any], *, effective_parameters: Mapping[str, Any], selected_dates: set[str]
) -> dict[str, pl.DataFrame]:
    """Recompute the frozen detector outputs from selected raw replay inputs."""
    raw_events = _selected_date_frame(
        pl.read_parquet(_path(run["source_paths"]["raw_events"], name="raw_events")), selected_dates
    )
    quote_panel = _selected_date_frame(
        pl.read_parquet(_path(run["source_paths"]["quote_panel"], name="quote_panel")), selected_dates
    )
    execution_anchor_modes = tuple(effective_parameters["execution_anchor_modes"])
    state_actor_mode = effective_parameters.get("state_client_mode", "all")
    state_actor_keys = _infer_state_actor_keys(
        raw_events, mode=state_actor_mode, execution_anchor_modes=execution_anchor_modes
    )
    empty_selected_replay = raw_events.height == 0 and quote_panel.height == 0
    tick_size = (
        EMPTY_CANONICAL_PLACEHOLDER_TICK_SIZE
        if empty_selected_replay else infer_tick_size_from_best_quotes(quote_panel)
    )
    result = compute_exploratory_metrics(
        raw_events,
        top_n=effective_parameters["top_n"],
        tick_size=tick_size,
        window_seconds=effective_parameters.get("window_seconds", 1.0),
        withdrawal_window_seconds=effective_parameters["withdrawal_window_seconds"],
        reversion_horizon_seconds=effective_parameters["reversion_horizon_seconds"],
        execution_cluster_max_gap_ms=effective_parameters.get("execution_cluster_max_gap_ms", 100),
        max_deceptive_order_age_seconds=effective_parameters["max_deceptive_order_age_seconds"],
        include_level_columns=not effective_parameters.get("compact_state", False),
        state_actor_keys=state_actor_keys,
        execution_anchor_modes=execution_anchor_modes,
        empirical_kernel_weights=load_empirical_kernel_weights(
            _path(run["source_paths"]["empirical_depth_kernel"], name="empirical_depth_kernel")
        ),
    )
    return {
        "execution_metrics": (
            pl.DataFrame(schema=EMPTY_CANONICAL_EXECUTION_METRICS_SCHEMA)
            if empty_selected_replay else result.execution_metrics
        ),
        "candidate_deceptive_orders": result.candidate_deceptive_orders,
        "execution_cluster_members": result.execution_cluster_members,
        "execution_cancel_candidates": result.execution_cancel_candidates,
        "rejected_executions": result.rejected_executions,
        "spoofing_compatible_events": result.spoofing_compatible_events,
        **episode_frames(result.episode_result),
    }


def _canonical_sort(frame: pl.DataFrame) -> pl.DataFrame:
    """Normalize only unordered row representation using scalar stable output keys."""
    scalar_columns = [
        name
        for name, dtype in frame.schema.items()
        if not dtype.is_nested() and dtype != pl.Object
    ]
    return frame.sort(scalar_columns, nulls_last=False) if scalar_columns else frame


def _semantic_frame_hash(frame: pl.DataFrame) -> str:
    canonical = _canonical_sort(frame)
    payload = json.dumps(
        {"schema": {name: str(dtype) for name, dtype in canonical.schema.items()}, "rows": canonical.to_dicts()},
        default=str,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _compare_canonical_frame(
    artifact: str, frozen: pl.DataFrame, live: pl.DataFrame, *, frozen_sha256: str
) -> dict[str, Any]:
    schema_equal = frozen.schema == live.schema
    frozen_sorted = _canonical_sort(frozen)
    live_sorted = _canonical_sort(live) if schema_equal else live
    equal = schema_equal and frozen_sorted.equals(live_sorted, null_equal=True)
    if not schema_equal:
        diagnostic = "schema differs"
    elif frozen.height != live.height:
        diagnostic = f"row count differs: frozen={frozen.height}, live={live.height}"
    elif not equal:
        diagnostic = "first canonical row differs"
    else:
        diagnostic = "exact semantic equivalence"
    return {
        "artifact": artifact,
        "frozen_row_count": frozen.height,
        "live_row_count": live.height,
        "equal": equal,
        "frozen_sha256": frozen_sha256,
        "live_semantic_sha256": _semantic_frame_hash(live),
        "diagnostic": diagnostic,
        "state_series_exclusion": STATE_SERIES_EXCLUSION_REASON,
    }


def _recompute_and_verify_canonical_runs(
    config: Mapping[str, Any], effective_parameters: Mapping[str, Mapping[str, Any]]
) -> tuple[dict[str, dict[str, pl.DataFrame]], pl.DataFrame]:
    """Fail closed unless every selected live detector frame equals its frozen baseline."""
    live_runs: dict[str, dict[str, pl.DataFrame]] = {}
    audit_rows: list[dict[str, Any]] = []
    for instrument, raw_run in sorted(config["baseline_runs"].items()):
        run = _mapping(raw_run, name=f"baseline_runs.{instrument}")
        selected_dates = set(run["selected_event_dates"])
        live = _canonical_live_frames(
            run, effective_parameters=effective_parameters[instrument], selected_dates=selected_dates
        )
        metadata_path = _path(run["baseline_metadata"], name="baseline_metadata")
        metadata = _mapping(json.loads(metadata_path.read_text()), name="baseline metadata")
        hashes = _mapping(metadata["artifact_hashes"], name="baseline metadata artifact_hashes")
        rows = []
        for artifact in CANONICAL_BASELINE_ARTIFACTS:
            frozen_path = metadata_path.parent / f"{artifact}.parquet"
            row = _compare_canonical_frame(
                artifact,
                pl.read_parquet(frozen_path),
                live[artifact],
                frozen_sha256=_nonempty_string(hashes[artifact], name=f"artifact_hashes.{artifact}"),
            )
            row["instrument"] = instrument
            rows.append(row)
        mismatches = [row for row in rows if not row["equal"]]
        if mismatches:
            first = mismatches[0]
            raise ValueError(
                "canonical equivalence failed for "
                f"{instrument}.{first['artifact']}: {first['diagnostic']}"
            )
        live_runs[instrument] = live
        audit_rows.extend(rows)
    return live_runs, pl.DataFrame(audit_rows)


def _coverage_rows(instrument: str, risk: Any, selected_dates: set[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    censor_reasons = {"coverage_end", "partition_boundary", "clock_ambiguous"}
    for selected_date in sorted(selected_dates):
        event_date = date.fromisoformat(selected_date)
        spells = risk.spells.filter(pl.col("event_date") == event_date)
        intervals = risk.intervals.filter(pl.col("event_date") == event_date)
        withdrawals = risk.withdrawal_events.filter(pl.col("event_date") == event_date)
        actors = risk.actor_summary.filter(pl.col("event_date") == event_date)
        censored = spells.filter(pl.col("end_reason").is_in(censor_reasons))
        reasons = (
            censored.group_by("end_reason").len().sort("end_reason").to_dicts()
            if not censored.is_empty() else []
        )
        rows.append({
            "instrument": instrument,
            "event_date": selected_date,
            "actor_count": actors.get_column("actor_key").n_unique() if not actors.is_empty() else 0,
            "actor_day_count": actors.get_column("actor_key").n_unique() if not actors.is_empty() else 0,
            "risk_spell_count": spells.height,
            "time_at_risk_seconds": float(intervals.get_column("duration_seconds").sum() or 0.0),
            "withdrawal_count": withdrawals.height,
            "censor_count": censored.height,
            "censor_reason": json.dumps(reasons, sort_keys=True),
        })
    return rows


def _episode_risk_links(risk: Any, candidates: pl.DataFrame, members: pl.DataFrame) -> pl.DataFrame:
    """Link detector episodes after risk construction by physical lifecycle overlap.

    Multiple links do not multiply risk duration; unlinked/no-fill risk is retained.
    """
    schema = {name: pl.String for name in (
        "episode_id", "execution_cluster_id", "risk_spell_id", "risk_interval_id", "physical_order_key",
    )}
    if candidates.is_empty() or members.is_empty() or risk.intervals.is_empty():
        return pl.DataFrame(schema=schema)
    candidate_keys = candidates.select(
        "partition_id", "actor_key", "execution_cluster_id",
        pl.col("deceptive_side").alias("side"),
        pl.col("deceptive_order_id").alias("order_id"),
        pl.col("deceptive_order_first_seen_sort_index").alias("first_seen_sort_index"),
    ).unique()
    linked = candidate_keys.join(
        members.select("partition_id", "actor_key", "execution_cluster_id", "episode_id").unique(),
        on=["partition_id", "actor_key", "execution_cluster_id"], how="inner",
    ).join(
        risk.membership, on=["partition_id", "actor_key", "side", "order_id", "first_seen_sort_index"], how="inner",
    ).join(
        risk.intervals.select("partition_id", "event_date", "actor_key", "side", "risk_spell_id", "risk_interval_id", "start_ts", "end_ts"),
        on=["partition_id", "event_date", "actor_key", "side"], how="inner",
    ).filter(
        (pl.col("start_ts") < pl.col("eligible_end_ts")) &
        (pl.col("end_ts") > pl.col("eligible_start_ts"))
    )
    return linked.select(
        "episode_id", "execution_cluster_id",
        "risk_spell_id", "risk_interval_id", "physical_order_key",
    ).unique().sort(list(schema))


def _validate_episode_risk_links(links: pl.DataFrame, intervals: pl.DataFrame) -> None:
    """Fail closed on relational provenance before publishing the bundle."""
    link_keys = [
        "instrument", "episode_id", "execution_cluster_id", "risk_spell_id",
        "risk_interval_id", "physical_order_key",
    ]
    missing = sorted(set(link_keys) - set(links.columns))
    if missing:
        raise ValueError("episode-risk links missing required columns: " + ", ".join(missing))
    if links.select(link_keys).is_duplicated().any():
        raise ValueError("episode-risk link keys are not unique")
    interval_keys = ["instrument", "risk_interval_id"]
    missing_intervals = sorted(set(interval_keys) - set(intervals.columns))
    if missing_intervals:
        raise ValueError("risk intervals missing link target columns: " + ", ".join(missing_intervals))
    if intervals.select(interval_keys).is_duplicated().any():
        raise ValueError("risk interval identifiers are not unique within instrument")
    orphaned = links.join(intervals.select(interval_keys), on=interval_keys, how="anti")
    if not orphaned.is_empty():
        raise ValueError("episode-risk links reference unknown instrument risk intervals")


def _finite_number(value: object) -> float | None:
    """Return a finite numeric scalar, treating booleans and non-finite values as missing."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return float(value)


def _aligned_selected_quote_panel(
    raw_events: pl.DataFrame, quote_panel: pl.DataFrame,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return selected canonical raw and quote rows aligned by replay ordinal.

    The panel's persisted ``sort_index`` belongs to the unfiltered replay.  It
    is used only to restore its canonical row order; downstream selected replay
    indices are the dense ordinals returned here.
    """
    canonical_raw = sort_events(raw_events)
    canonical_quotes = quote_panel.sort("sort_index")
    if canonical_raw.height != canonical_quotes.height:
        raise AssertionError("selected raw and quote panel row counts must match")
    required_partition_columns = set(CANONICAL_PARTITION_COLUMNS)
    missing = sorted(required_partition_columns - set(canonical_quotes.columns))
    if missing and not canonical_quotes.is_empty():
        raise ValueError(
            "selected quote panel cannot verify canonical partition alignment; missing columns: "
            + ", ".join(missing)
        )
    for ordinal, (raw, quote) in enumerate(zip(
        canonical_raw.iter_rows(named=True), canonical_quotes.iter_rows(named=True), strict=True,
    )):
        if str(raw["TRADEDATE"])[:10] != str(quote["TRADEDATE"])[:10]:
            raise AssertionError(f"selected raw/quote date mismatch at canonical ordinal {ordinal}")
        raw_partition = tuple(str(raw[column]) for column in CANONICAL_PARTITION_COLUMNS[1:])
        quote_partition = tuple(str(quote[column]) for column in CANONICAL_PARTITION_COLUMNS[1:])
        if not canonical_quotes.is_empty() and raw_partition != quote_partition:
            raise AssertionError(f"selected raw/quote partition mismatch at canonical ordinal {ordinal}")
    return canonical_raw, canonical_quotes.with_columns(
        pl.Series("selected_sort_index", range(canonical_quotes.height), dtype=pl.Int64)
    )


def _pre_event_covariates(
    instrument: str, raw_events: pl.DataFrame, quote_panel: pl.DataFrame,
    contrasts: pl.DataFrame, intervals: pl.DataFrame, *, canonical_tick_size: float,
    activity_bands: list[dict[str, Any]],
) -> pl.DataFrame:
    """Build strictly pre-anchor covariates for each observed contrast segment.

    The segment's existing risk state is the replay pre-state.  Quote values use
    the latest same-partition *pre* quote boundary at or before its anchor; the
    event-count and return windows exclude the anchor itself.
    """
    if not math.isfinite(canonical_tick_size) or canonical_tick_size <= 0:
        raise ValueError("canonical_tick_size must be finite and positive")
    schema = {
        "instrument": pl.String, "risk_interval_id": pl.String, "partition_id": pl.String,
        "event_date": pl.Date, "actor_key": pl.String, "identity_level": pl.String,
        "side": pl.String, "contrast_label": pl.String, "anchor_ts": pl.Datetime("us"), "duration_seconds": pl.Float64,
        "activity_band": pl.String, "snapshot_sort_index": pl.Int64, "anchor_sort_index": pl.Int64,
        **{name: pl.Float64 for name in PRE_EVENT_COVARIATES}, "missing_reasons": pl.String,
    }
    if contrasts.is_empty():
        return pl.DataFrame(schema=schema)
    interval_fields = ["risk_interval_id", *PRE_EVENT_COVARIATES[:4]]
    joined = contrasts.join(intervals.select(interval_fields), on="risk_interval_id", how="left", validate="m:1")
    canonical_raw, canonical_quotes = _aligned_selected_quote_panel(raw_events, quote_panel)
    raw_boundaries: dict[int, tuple[str, datetime]] = {}
    event_times: dict[str, list[tuple[datetime, int]]] = {}
    for index, event in enumerate(canonical_raw.iter_rows(named=True)):
        timestamp = choose_event_timestamp(event)
        if timestamp is not None:
            partition = _partition_id(event)
            raw_boundaries[index] = (partition, timestamp)
            event_times.setdefault(partition, []).append((timestamp, index))
    quote_rows: dict[str, list[dict[str, object]]] = {}
    for quote in canonical_quotes.iter_rows(named=True):
        sort_index = quote["selected_sort_index"]
        if sort_index not in raw_boundaries:
            continue
        partition, timestamp = raw_boundaries[sort_index]
        bid, ask = _finite_number(quote.get("pre_best_bid")), _finite_number(quote.get("pre_best_ask"))
        mid = (bid + ask) / 2 if bid is not None and ask is not None and ask > bid else None
        quote_rows.setdefault(partition, []).append({
            "timestamp": timestamp, "bid": bid, "ask": ask, "mid": mid,
            "sort_index": sort_index,
            "depth": (None if _finite_number(quote.get("pre_bid_visible_qty_total")) is None
                      or _finite_number(quote.get("pre_ask_visible_qty_total")) is None
                      else _finite_number(quote.get("pre_bid_visible_qty_total")) + _finite_number(quote.get("pre_ask_visible_qty_total"))),
        })
    for values in quote_rows.values():
        values.sort(key=lambda value: (value["timestamp"], value["sort_index"]))
    rows: list[dict[str, object]] = []
    for segment in joined.iter_rows(named=True):
        anchor = segment["start_ts"]
        partition = str(segment["partition_id"])
        reasons: dict[str, str] = {}
        row = {name: segment.get(name) for name in schema if name not in PRE_EVENT_COVARIATES and name != "missing_reasons"}
        row["anchor_ts"] = segment["start_ts"]
        row["instrument"] = instrument
        matched_band = next(
            (band for band in activity_bands if band["start_time"] <= anchor.time() < band["end_time"]),
            None,
        )
        row["activity_band"] = (
            f"{anchor.date().isoformat()}:{matched_band['name']}" if matched_band is not None else None
        )
        if matched_band is None:
            reasons["activity_band"] = "anchor_outside_prespecified_activity_band"
        anchor_sort_index = segment.get("start_sort_index")
        if anchor_sort_index is None:
            prior_boundaries = [index for timestamp, index in event_times.get(partition, []) if timestamp <= anchor]
            anchor_sort_index = prior_boundaries[-1] if prior_boundaries else None
        elif anchor_sort_index not in raw_boundaries:
            raise AssertionError("risk interval anchor sort_index is not a selected canonical replay ordinal")
        row["anchor_sort_index"] = anchor_sort_index
        for name in PRE_EVENT_COVARIATES[:4]:
            value = _finite_number(segment.get(name))
            row[name] = value
            if value is None:
                reasons[name] = "missing_pre_interval_state"
        prior_quotes = [
            value for value in quote_rows.get(partition, [])
            if anchor_sort_index is not None
            and (value["timestamp"] < anchor or (value["timestamp"] == anchor and value["sort_index"] <= anchor_sort_index))
        ]
        boundary = prior_quotes[-1] if prior_quotes else None
        row["snapshot_sort_index"] = boundary["sort_index"] if boundary is not None else None
        if boundary is None:
            row["pre_spread_ticks"] = row["pre_total_depth"] = None
            reasons["pre_spread_ticks"] = reasons["pre_total_depth"] = "no_pre_quote_boundary"
        else:
            bid, ask = boundary["bid"], boundary["ask"]
            spread = ask - bid if bid is not None and ask is not None else None
            row["pre_spread_ticks"] = spread / canonical_tick_size if spread is not None and spread > 0 else None
            row["pre_total_depth"] = boundary["depth"]
            if row["pre_spread_ticks"] is None:
                reasons["pre_spread_ticks"] = "nonpositive_or_unavailable_spread_or_tick"
            if row["pre_total_depth"] is None:
                reasons["pre_total_depth"] = "unavailable_pre_total_depth"
        lower = anchor.timestamp() - 60.0
        prior_events = [
            value for value in event_times.get(partition, [])
            if anchor_sort_index is not None
            and lower <= value[0].timestamp()
            and (value[0] < anchor or (value[0] == anchor and value[1] < anchor_sort_index))
        ]
        row["prior_event_count_60s"] = float(len(prior_events))
        history = [
            value for value in quote_rows.get(partition, [])
            if anchor_sort_index is not None
            and lower <= value["timestamp"].timestamp()
            and (value["timestamp"] < anchor or (value["timestamp"] == anchor and value["sort_index"] < anchor_sort_index))
            and value["mid"] is not None and value["mid"] > 0
        ]
        returns = [math.log(history[index]["mid"] / history[index - 1]["mid"]) for index in range(1, len(history))]
        row["pre_mid_log_return_sd_60s"] = (
            math.sqrt(sum((value - sum(returns) / len(returns)) ** 2 for value in returns) / (len(returns) - 1))
            if len(returns) >= 2 else None
        )
        if row["pre_mid_log_return_sd_60s"] is None:
            reasons["pre_mid_log_return_sd_60s"] = "insufficient_finite_pre_mid_return_history"
        row["missing_reasons"] = json.dumps(reasons, sort_keys=True) if reasons else None
        rows.append(row)
    return pl.DataFrame(rows, schema=schema, strict=False)


def _common_support_audit(covariates: pl.DataFrame) -> pl.DataFrame:
    """Audit finite covariate-range overlap without matching or outcome use."""
    schema = {
        "instrument": pl.String, "event_date": pl.Date, "identity_level": pl.String,
        "side": pl.String, "comparison_label": pl.String, "contrast_label": pl.String, "covariate": pl.String,
        "total_seconds": pl.Float64, "supported_seconds": pl.Float64,
        "outside_support_seconds": pl.Float64, "support_established": pl.Boolean,
        "supported": pl.Int64, "null_fraction": pl.Float64,
    }
    if covariates.is_empty():
        return pl.DataFrame(schema=schema)
    comparisons = (("own_only", "other_only"), ("own_only", "no_qualifying_fill"))
    rows: list[dict[str, object]] = []
    keys = ("instrument", "event_date", "identity_level", "side")
    for group_key, group in covariates.group_by(list(keys), maintain_order=True):
        base = dict(zip(keys, group_key, strict=True))
        for left, right in comparisons:
            label = f"{left}_vs_{right}"
            left_frame, right_frame = group.filter(pl.col("contrast_label") == left), group.filter(pl.col("contrast_label") == right)
            intersections: dict[str, tuple[float, float]] = {}
            diagnostics: list[dict[str, object]] = []
            for covariate in PRE_EVENT_COVARIATES:
                left_values = [_finite_number(value) for value in left_frame.get_column(covariate).to_list()]
                right_values = [_finite_number(value) for value in right_frame.get_column(covariate).to_list()]
                finite_left, finite_right = [v for v in left_values if v is not None], [v for v in right_values if v is not None]
                established = bool(finite_left and finite_right and max(min(finite_left), min(finite_right)) <= min(max(finite_left), max(finite_right)))
                if established:
                    intersections[covariate] = (max(min(finite_left), min(finite_right)), min(max(finite_left), max(finite_right)))
                for state, frame in ((left, left_frame), (right, right_frame)):
                    total = float(frame.get_column("duration_seconds").sum() or 0.0)
                    lower_bound = max(min(finite_left), min(finite_right)) if established else None
                    upper_bound = min(max(finite_left), max(finite_right)) if established else None
                    supported_seconds = sum(float(item["duration_seconds"]) for item in frame.iter_rows(named=True) if established and (value := _finite_number(item[covariate])) is not None and lower_bound <= value <= upper_bound)
                    null_seconds = sum(float(item["duration_seconds"]) for item in frame.iter_rows(named=True) if _finite_number(item[covariate]) is None)
                    diagnostics.append({**base, "comparison_label": label, "covariate": covariate, "contrast_label": state,
                        "total_seconds": total, "supported_seconds": supported_seconds,
                        "outside_support_seconds": total - supported_seconds, "support_established": established,
                        "supported": int(established and supported_seconds > 0),
                        "null_fraction": null_seconds / total if total > 0 else None})
            jointly_established = bool(
                float(left_frame.get_column("duration_seconds").sum() or 0.0) > 0
                and float(right_frame.get_column("duration_seconds").sum() or 0.0) > 0
                and len(intersections) == len(PRE_EVENT_COVARIATES)
            )
            for diagnostic in diagnostics:
                diagnostic["support_established"] = jointly_established
                if not jointly_established:
                    diagnostic["supported_seconds"] = 0.0
                    diagnostic["outside_support_seconds"] = diagnostic["total_seconds"]
                    diagnostic["supported"] = 0
                rows.append(diagnostic)
            for state, frame in ((left, left_frame), (right, right_frame)):
                total = float(frame.get_column("duration_seconds").sum() or 0.0)
                supported_seconds = sum(
                    float(item["duration_seconds"])
                    for item in frame.iter_rows(named=True)
                    if jointly_established and all(
                        (value := _finite_number(item[covariate])) is not None
                        and intersections[covariate][0] <= value <= intersections[covariate][1]
                        for covariate in PRE_EVENT_COVARIATES
                    )
                )
                null_seconds = sum(
                    float(item["duration_seconds"])
                    for item in frame.iter_rows(named=True)
                    if any(_finite_number(item[covariate]) is None for covariate in PRE_EVENT_COVARIATES)
                )
                rows.append({**base, "comparison_label": label, "covariate": "__all_required__", "contrast_label": state,
                    "total_seconds": total, "supported_seconds": supported_seconds,
                    "outside_support_seconds": total - supported_seconds, "support_established": jointly_established,
                    "supported": int(jointly_established and supported_seconds > 0),
                    "null_fraction": null_seconds / total if total > 0 else None})
    return pl.DataFrame(rows, schema=schema, strict=False)


def _profile_balance(covariates: pl.DataFrame) -> pl.DataFrame:
    """Compute finite-observation duration-weighted profile means and missingness."""
    return covariates.group_by("event_date", "identity_level", "side", "contrast_label").agg(
        pl.col("duration_seconds").sum().alias("time_at_risk_seconds"),
        *[
            pl.when(pl.col(name).is_finite()).then(pl.col("duration_seconds")).otherwise(0.0)
            .sum().alias(f"{name}_observed_seconds")
            for name in PRE_EVENT_COVARIATES
        ],
        *[
            pl.when(pl.col(name).is_finite()).then(0.0).otherwise(pl.col("duration_seconds"))
            .sum().alias(f"{name}_missing_seconds")
            for name in PRE_EVENT_COVARIATES
        ],
        *[
            pl.when(pl.col(name).is_finite()).then(pl.col("duration_seconds") * pl.col(name)).otherwise(0.0)
            .sum().alias(f"{name}_finite_seconds")
            for name in PRE_EVENT_COVARIATES
        ],
    ).with_columns(*[
        pl.when(pl.col(f"{name}_observed_seconds") > 0)
        .then(pl.col(f"{name}_finite_seconds") / pl.col(f"{name}_observed_seconds"))
        .otherwise(None).alias(f"mean_{name}") for name in PRE_EVENT_COVARIATES
    ]).drop([f"{name}_finite_seconds" for name in PRE_EVENT_COVARIATES]).sort(
        "event_date", "identity_level", "side", "contrast_label"
    )


def _write_bundle_artifacts(
    root: Path,
    config: Mapping[str, Any],
    config_path: Path,
    *,
    live_canonical_runs: Mapping[str, Mapping[str, pl.DataFrame]],
    canonical_equivalence: pl.DataFrame,
) -> dict[str, Any]:
    collected: dict[str, list[pl.DataFrame]] = {
        key: [] for key in (
            "risk_spells", "risk_intervals", "risk_membership", "exposure_intervals",
            "withdrawal_events", "episode_risk_links", "placebo_schedules",
            "placebo_support", "placebo_statistics", "stratum_statistics",
            "risk_diagnostics", "risk_transitions", "profile_balance", "clock_coverage_audit",
            "pre_event_covariates", "common_support_audit",
        )
    }
    exposure_contrasts: list[pl.DataFrame] = []
    coverage_rows: list[dict[str, Any]] = []
    source_graph: dict[str, Any] = {}
    exclusion_reasons: dict[str, int] = {}
    observed_anchor_modes: set[str] = set()
    horizon = float(config["horizons"]["withdrawal_window_seconds"])
    top_n = int(config["eligibility_policy"]["top_n"])
    max_age = float(config["eligibility_policy"]["max_order_age_seconds"])
    activity_bands = _validate_activity_bands(config["strata"]["activity_bands"])

    for instrument, raw_run in sorted(config["baseline_runs"].items()):
        run = _mapping(raw_run, name=f"baseline_runs.{instrument}")
        selected_dates = set(run["selected_event_dates"])
        raw_path = _path(run["source_paths"]["raw_events"], name="raw_events")
        raw_events = pl.read_parquet(raw_path).filter(
            pl.col("TRADEDATE").cast(pl.Date).is_in(
                [date.fromisoformat(value) for value in sorted(selected_dates)]
            )
        )
        clusters, outside_band_count = _read_execution_clusters(
            live_canonical_runs[instrument]["execution_metrics"], selected_dates, activity_bands
        )
        exclusion_reasons["execution_cluster_outside_prespecified_activity_band"] = (
            exclusion_reasons.get("execution_cluster_outside_prespecified_activity_band", 0)
            + outside_band_count
        )
        modes = config["strata"]["execution_anchor_modes"]
        if not clusters.is_empty():
            clusters = clusters.filter(pl.col("execution_anchor_mode").is_in(modes))
            # Use the replay's Python scalar serialization, not Polars' datetime
            # string formatting (which adds fractional seconds to midnight).
            raw_partition_ids = {
                _partition_id(row)
                for row in raw_events.select(
                    "TRADEDATE", "MIC", "MARKETCODE", "SYMBOLINDEX", "EMM (*)"
                ).unique().iter_rows(named=True)
            }
            unknown_partitions = sorted(set(clusters.get_column("partition_id")) - raw_partition_ids)
            if unknown_partitions:
                raise ValueError(
                    "execution clusters reference partitions absent from selected raw replay: "
                    + ", ".join(unknown_partitions[:5])
                )
        base_risk = compute_withdrawal_risk(
            raw_events, top_n=top_n, max_order_age_seconds=max_age,
        )
        windows = {
            f"{selected_date}:{band['name']}": (
                datetime.combine(date.fromisoformat(selected_date), band["start_time"]),
                datetime.combine(date.fromisoformat(selected_date), band["end_time"]),
            )
            for selected_date in selected_dates
            for band in activity_bands
        }
        placebo = rescore_empirical_placebos(
            base_risk,
            clusters,
            withdrawal_window_seconds=horizon,
            draw_count=int(config["placebo_draws"]),
            seed=int(config["seed"]),
            activity_band_windows=windows,
        )
        observed = placebo.observed_risk
        observed_anchor_modes.update(clusters.get_column("execution_anchor_mode").drop_nulls().to_list())
        collected["risk_diagnostics"].append(_instrument_frame(observed.diagnostics, instrument))
        collected["risk_transitions"].append(_instrument_frame(observed.transitions, instrument))
        collected["clock_coverage_audit"].append(build_clock_coverage_audit(
            instrument,
            raw_events,
            intervals=observed.intervals,
            diagnostics=observed.diagnostics,
            spells=observed.spells,
        ))
        for reason in observed.diagnostics.group_by("reason").len().iter_rows(named=True):
            key = f"risk:{reason['reason']}"
            exclusion_reasons[key] = exclusion_reasons.get(key, 0) + int(reason["len"])
        quote_panel = _selected_date_frame(
            pl.read_parquet(_path(run["source_paths"]["quote_panel"], name="quote_panel")), selected_dates
        )
        empty_selected_replay = raw_events.height == 0 and quote_panel.height == 0
        canonical_tick_size = (
            EMPTY_CANONICAL_PLACEHOLDER_TICK_SIZE
            if empty_selected_replay else infer_tick_size_from_best_quotes(quote_panel)
        )
        covariates = _pre_event_covariates(
            instrument, raw_events, quote_panel, observed.exposure_contrasts, observed.intervals,
            canonical_tick_size=canonical_tick_size, activity_bands=activity_bands,
        )
        collected["pre_event_covariates"].append(covariates)
        balance = _profile_balance(covariates)
        balance = balance.with_columns(
            pl.col("mean_eligible_visible_qty").alias("mean_profile_visible_qty"),
            pl.col("mean_member_count").alias("mean_profile_order_count"),
        )
        collected["profile_balance"].append(_instrument_frame(balance, instrument))
        exposure_contrasts.append(_instrument_frame(observed.exposure_contrasts, instrument))
        coverage_rows.extend(_coverage_rows(instrument, observed, selected_dates))
        collected["risk_spells"].append(
            _instrument_frame(observed.spells, instrument, posture_side="side")
        )
        collected["risk_intervals"].append(
            _instrument_frame(observed.intervals, instrument).with_columns(
                pl.lit(None, dtype=pl.String).alias("execution_anchor_mode")
            )
        )
        collected["risk_membership"].append(_instrument_frame(observed.membership, instrument))
        collected["exposure_intervals"].append(_instrument_frame(observed.exposure_intervals, instrument))
        collected["withdrawal_events"].append(_instrument_frame(observed.withdrawal_events, instrument))
        links = _episode_risk_links(
            observed,
            live_canonical_runs[instrument]["candidate_deceptive_orders"],
            live_canonical_runs[instrument]["episode_cluster_members"],
        )
        collected["episode_risk_links"].append(_instrument_frame(links, instrument))
        collected["placebo_schedules"].append(_instrument_frame(placebo.placebo_schedules, instrument))
        collected["placebo_support"].append(_instrument_frame(placebo.placebo_support, instrument))
        collected["placebo_statistics"].append(_instrument_frame(placebo.placebo_statistics, instrument))
        strata = _instrument_frame(observed.exposure_strata, instrument).with_columns(
            pl.col("duration_seconds").alias("time_at_risk_seconds"),
            pl.when(pl.col("duration_seconds") > 0)
            .then(pl.col("withdrawal_count") / pl.col("duration_seconds"))
            .otherwise(None)
            .alias("withdrawal_intensity"),
        )
        collected["stratum_statistics"].append(strata)
        metadata_path = _path(run["baseline_metadata"], name="baseline_metadata")
        configured_sources = run["source_paths"]
        source_graph[instrument] = {
            "selected_event_dates": sorted(selected_dates),
            "raw_events": {"path": str(raw_path), "sha256": _sha256(raw_path)},
            "baseline_metadata": {"path": str(metadata_path), "sha256": _sha256(metadata_path)},
            "execution_metrics": {
                "path": str(metadata_path.parent / "execution_metrics.parquet"),
                "sha256": _sha256(metadata_path.parent / "execution_metrics.parquet"),
            },
            **{
                source_name: {
                    "path": str(_path(configured_sources[source_name], name=source_name)),
                    "sha256": _sha256(_path(configured_sources[source_name], name=source_name)),
                }
                for source_name in ("quote_panel", "empirical_depth_kernel", "baseline_config")
            },
        }

    outputs = {name: _concat(frames) for name, frames in collected.items()}
    outputs["common_support_audit"] = _common_support_audit(outputs["pre_event_covariates"])
    _validate_episode_risk_links(outputs["episode_risk_links"], outputs["risk_intervals"])
    clock_coverage_checks = _validate_clock_coverage_reconciliation(
        outputs["clock_coverage_audit"], outputs["risk_diagnostics"], outputs["risk_intervals"],
    )
    canonical_equivalence.write_parquet(root / "canonical_equivalence.parquet")
    for name, frame in outputs.items():
        suffix = ".csv" if name in {"stratum_statistics", "profile_balance", "common_support_audit"} else ".parquet"
        path = root / f"{name}{suffix}"
        frame.write_csv(path) if suffix == ".csv" else frame.write_parquet(path)
    coverage = pl.DataFrame(coverage_rows)
    coverage.write_csv(root / "coverage_and_balance.csv")
    intervals = outputs["risk_intervals"]
    contrasts = _concat(exposure_contrasts)
    interval_seconds = float(intervals.get_column("duration_seconds").sum() or 0.0)
    contrast_seconds = float(contrasts.get_column("duration_seconds").sum() or 0.0)
    tolerance = 1e-9 * max(1.0, abs(interval_seconds))
    if abs(interval_seconds - contrast_seconds) > tolerance:
        raise AssertionError("exposure intervals do not partition time at risk")
    support_duration_ok = True
    for row in outputs["common_support_audit"].group_by("comparison_label", "covariate", "contrast_label").agg(pl.col("total_seconds").sum()).iter_rows(named=True):
        expected = float(contrasts.filter(pl.col("contrast_label") == row["contrast_label"]).get_column("duration_seconds").sum() or 0.0)
        support_duration_ok &= math.isclose(float(row["total_seconds"]), expected, rel_tol=1e-9, abs_tol=1e-9)
    if not support_duration_ok:
        raise AssertionError("common-support audit duration does not reconcile to exposure contrasts")
    validation = {
        "status": "pass",
        "checks": {
            "risk_interval_seconds": interval_seconds,
            "exposure_interval_seconds": contrast_seconds,
            "physical_withdrawal_keys_unique": not outputs["withdrawal_events"].select(
                "instrument", "physical_order_key"
            ).is_duplicated().any(),
            "common_support_audit_durations_reconcile_to_exposure_contrasts": support_duration_ok,
            **clock_coverage_checks,
        },
    }
    if not validation["checks"]["physical_withdrawal_keys_unique"]:
        raise AssertionError("physical withdrawal keys are not unique")
    (root / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True))
    (root / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True))
    (root / "versions.txt").write_text(
        f"python={platform.python_version()}\npolars={pl.__version__}\n"
    )
    (root / "command.txt").write_text(
        f"python scripts/build_spoofing_empirical_negative_controls.py --config {config_path}\n"
    )
    (root / "run.log").write_text("exit_status=0\n")
    artifacts = {}
    for path in sorted(root.iterdir()):
        if path.name != "manifest.json":
            artifacts[path.name] = {"path": path.name, "sha256": _sha256(path)}
    support = outputs["placebo_support"]
    if not support.is_empty():
        for row in (
            support.filter(~pl.col("accepted"))
            .group_by("rejection_reason")
            .len()
            .iter_rows(named=True)
        ):
            reason = f"placebo:{row['rejection_reason']}"
            exclusion_reasons[reason] = exclusion_reasons.get(reason, 0) + int(row["len"])
    summary = {
        "risk_spell_count": outputs["risk_spells"].height,
        "time_at_risk_seconds": interval_seconds,
        "withdrawal_count": outputs["withdrawal_events"].height,
        "accepted_placebo_blocks": support.filter(pl.col("accepted")).height,
        "rejected_placebo_blocks": support.filter(~pl.col("accepted")).height,
        "placebo_statistics_rows": outputs["placebo_statistics"].height,
    }
    return {
        "design_version": config["design_version"],
        "seed": config["seed"],
        "bundle_audit_version": 2,
        "configured_execution_anchor_modes": list(config["strata"]["execution_anchor_modes"]),
        "observed_execution_anchor_modes": [mode for mode in ("passive", "aggressive") if mode in observed_anchor_modes],
        "effective_clock": {
            "clock_contract_version": 2,
            "event_timestamp_precedence": list(EVENT_TIMESTAMP_PRECEDENCE),
            "partition_columns": list(CANONICAL_PARTITION_COLUMNS),
            "stable_sort_columns": list(SORT_COLUMNS),
            "sort_index_policy": SORT_INDEX_POLICY,
            "risk_clock_regression_policy": CLOCK_REGRESSION_POLICY,
            # Retained for build_spoofing_negative_control_report compatibility.
            "timestamp_precedence": list(EVENT_TIMESTAMP_PRECEDENCE),
            "timezone": "aware normalized to UTC-naive; naive source timezone not established",
        },
        "comparability": {
            "status": "unadjusted; common-support audit is descriptive only",
            "contract": COMPARABILITY_CONTRACT,
            "observed": list(PRE_EVENT_COVARIATES),
            "unavailable": [],
        },
        "clock": config["clock"],
        "risk_policy": config["eligibility_policy"],
        "units": {
            "risk_spells.parquet": {"duration_seconds": "seconds"},
            "risk_diagnostics.parquet": {"sort_index": "canonical feed position", "reason": "quarantine reason"},
            "risk_transitions.parquet": {"visible_qty_pre": "shares", "transition_reason": "competing event or lifecycle boundary"},
            "profile_balance.csv": {
                "time_at_risk_seconds": "posture seconds", "mean_eligible_visible_qty": "shares", "mean_member_count": "orders", "mean_mean_member_age_seconds": "seconds", "mean_oldest_member_age_seconds": "seconds", "mean_pre_spread_ticks": "ticks", "mean_pre_total_depth": "shares", "mean_prior_event_count_60s": "events", "mean_pre_mid_log_return_sd_60s": "log-return",
                **{f"{name}_observed_seconds": "seconds" for name in PRE_EVENT_COVARIATES},
                **{f"{name}_missing_seconds": "seconds" for name in PRE_EVENT_COVARIATES},
            },
            "pre_event_covariates.parquet": {"anchor_ts": "UTC-naive event clock", "snapshot_sort_index": "selected canonical replay ordinal", "anchor_sort_index": "selected canonical replay ordinal", "eligible_visible_qty": "shares", "member_count": "orders", "mean_member_age_seconds": "seconds", "oldest_member_age_seconds": "seconds", "pre_spread_ticks": "ticks", "pre_total_depth": "shares", "prior_event_count_60s": "events", "pre_mid_log_return_sd_60s": "log-return standard deviation"},
            "common_support_audit.csv": {"total_seconds": "seconds", "supported_seconds": "seconds", "outside_support_seconds": "seconds", "null_fraction": "fraction"},
            "risk_intervals.parquet": {"duration_seconds": "seconds", "eligible_visible_qty": "shares"},
            "risk_membership.parquet": {"duration_seconds": "seconds"},
            "exposure_intervals.parquet": {"duration_seconds": "seconds", "execution_quantity": "shares"},
            "withdrawal_events.parquet": {"visible_qty_removed": "shares"},
            "episode_risk_links.parquet": {
                "episode_id": "dimensionless episode identifier",
                "execution_cluster_id": "dimensionless execution-cluster identifier",
                "risk_spell_id": "dimensionless risk-spell identifier",
                "risk_interval_id": "dimensionless risk-interval identifier",
                "physical_order_key": "dimensionless physical lifecycle identifier",
            },
            "placebo_schedules.parquet": {
                "draw_id": "count", "timestamps": "UTC-naive event clock", "sort_indices": "canonical event order",
            },
            "placebo_support.parquet": {
                "draw_id": "count", "support_boundary_count": "count", "supported_start_count": "count",
            },
            "placebo_statistics.parquet": {
                "time_at_risk_seconds": "seconds", "withdrawal_count": "physical cancellations",
                "withdrawal_intensity": "cancellations per second",
            },
            "coverage_and_balance.csv": {
                "time_at_risk_seconds": "seconds", "withdrawal_count": "physical cancellations",
            },
            "clock_coverage_audit.parquet": {
                "total_event_count": "events", "accepted_event_count": "events",
                "missing_timestamp_event_count": "events", "clock_regression_event_count": "events",
                "quarantined_event_count": "events", "quarantine_episode_count": "episodes",
                "recovered_quarantine_episode_count": "episodes",
                "recovered_quarantine_duration_seconds": "seconds",
                "unquantified_quarantine_event_count": "events",
                "observed_accepted_span_seconds": "seconds", "time_at_risk_seconds": "seconds",
                "coverage_loss_fraction": "quantified lower-bound fraction",
                "terminal_right_censored_spell_count": "spells",
                "partition_boundary_censored_spell_count": "spells",
                "clock_ambiguous_censored_spell_count": "spells",
            },
            "stratum_statistics.csv": {
                "time_at_risk_seconds": "seconds", "withdrawal_count": "physical cancellations",
                "withdrawal_intensity": "cancellations per second",
            },
            "canonical_equivalence.parquet": {
                "equal": "boolean exact semantic equivalence gate",
                "frozen_row_count": "rows",
                "live_row_count": "rows",
            },
        },
        "code_snapshot": {
            "git_head": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                capture_output=True, text=True,
            ).stdout.strip(),
            "files": {
                str(path.relative_to(REPO_ROOT)): {"path": str(path), "sha256": _sha256(path)}
                for path in sorted((REPO_ROOT / "src" / "spoofing_detection" / "lob").glob("*.py"))
            } | {
                str(path.relative_to(REPO_ROOT)): {"path": str(path), "sha256": _sha256(path)}
                for path in (
                    Path(__file__).resolve(),
                    (REPO_ROOT / "scripts" / "build_spoofing_negative_control_report.py").resolve(),
                )
            },
        },
        "sources": source_graph,
        "exclusion_reasons": exclusion_reasons,
        "summary": summary,
        "artifacts": artifacts,
    }


def run_empirical_controls(config_path: Path) -> Path:
    """Run the frozen empirical design and atomically publish a validated bundle."""
    validated = validate_preflight(config_path)
    config = _mapping(json.loads(config_path.read_text()), name="empirical controls config")
    live_canonical_runs, canonical_equivalence = _recompute_and_verify_canonical_runs(
        config, validated["effective_parameters"]
    )
    output_root = Path(validated["output_root"])
    if output_root.exists():
        raise FileExistsError(f"output root already exists: {output_root}")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.", dir=output_root.parent))
    try:
        manifest = _write_bundle_artifacts(
            temporary,
            config,
            config_path,
            live_canonical_runs=live_canonical_runs,
            canonical_equivalence=canonical_equivalence,
        )
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        report_path = build_empirical_report(bundle_root=temporary)
        manifest["artifacts"]["report.md"] = {
            "path": "report.md", "sha256": _sha256(report_path),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
        os.replace(temporary, output_root)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return output_root


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the frozen empirical spoofing negative-control design.",
        allow_abbrev=False,
    )
    parser.add_argument("--config", type=Path, required=True,
                        help="Explicit versioned configuration; no implicit legacy run.")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate the design and inputs without creating an output directory.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    validated = validate_preflight(args.config)
    if args.validate_only:
        print(
            f"validated empirical controls design {validated['design_version']} "
            f"for {validated['run_count']} baseline run(s); no output created"
        )
        return
    output_root = run_empirical_controls(args.config)
    print(output_root)


if __name__ == "__main__":
    main()
