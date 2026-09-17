#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.negative_controls import (  # noqa: E402
    add_time_shift_placebo,
    add_wrong_side_placebo,
)


REQUIRED_EMPIRICAL_ARTIFACTS = {
    "config.json",
    "versions.txt",
    "command.txt",
    "run.log",
    "validation.json",
    "risk_spells.parquet",
    "risk_intervals.parquet",
    "risk_membership.parquet",
    "exposure_intervals.parquet",
    "withdrawal_events.parquet",
    "episode_risk_links.parquet",
    "placebo_schedules.parquet",
    "coverage_and_balance.csv",
    "clock_coverage_audit.parquet",
    "stratum_statistics.csv",
    "placebo_support.parquet",
    "placebo_statistics.parquet",
}
AUDIT_TABLE_COLUMNS = {
    "risk_diagnostics.parquet": {"instrument", "partition_id", "sort_index", "reason", "detail"},
    "risk_transitions.parquet": {"instrument", "partition_id", "physical_order_key", "transition_reason", "visible_qty_pre"},
    "profile_balance.csv": {"instrument", "event_date", "identity_level", "side", "contrast_label", "time_at_risk_seconds", "mean_eligible_visible_qty", "mean_member_count", "mean_pre_spread_ticks", "mean_pre_total_depth", "mean_prior_event_count_60s", "mean_pre_mid_log_return_sd_60s", "eligible_visible_qty_observed_seconds", "eligible_visible_qty_missing_seconds", "member_count_observed_seconds", "member_count_missing_seconds", "mean_member_age_seconds_observed_seconds", "mean_member_age_seconds_missing_seconds", "oldest_member_age_seconds_observed_seconds", "oldest_member_age_seconds_missing_seconds", "pre_spread_ticks_observed_seconds", "pre_spread_ticks_missing_seconds", "pre_total_depth_observed_seconds", "pre_total_depth_missing_seconds", "prior_event_count_60s_observed_seconds", "prior_event_count_60s_missing_seconds", "pre_mid_log_return_sd_60s_observed_seconds", "pre_mid_log_return_sd_60s_missing_seconds"},
    "pre_event_covariates.parquet": {"risk_interval_id", "contrast_label", "anchor_ts", "activity_band", "snapshot_sort_index", "anchor_sort_index", "eligible_visible_qty", "member_count", "mean_member_age_seconds", "oldest_member_age_seconds", "pre_spread_ticks", "pre_total_depth", "prior_event_count_60s", "pre_mid_log_return_sd_60s", "missing_reasons"},
    "common_support_audit.csv": {"instrument", "event_date", "identity_level", "side", "comparison_label", "covariate", "total_seconds", "supported_seconds", "outside_support_seconds", "support_established", "supported", "null_fraction"},
}
REQUIRED_SOURCE_KINDS = {
    "raw_events", "baseline_metadata", "execution_metrics", "quote_panel",
    "empirical_depth_kernel", "baseline_config",
}
COMPARABILITY_CONTRACT = {
    "version": "pre_anchor_covariates_v1", "anchor": "exposure_contrast_start",
    "quote_boundary_policy": "latest_same_partition_at_or_before_anchor_pre_state_canonical_sort_tiebreak",
    "history_window_seconds": 60.0, "history_interval": "[anchor-60s,anchor)",
    "outcomes_used": False, "matching": "forbidden", "adjusted_effect": "forbidden",
}
SCIENTIFIC_TABLES = {
    name for name in REQUIRED_EMPIRICAL_ARTIFACTS if name.endswith((".parquet", ".csv"))
}
REQUIRED_TABLE_COLUMNS = {
    "risk_spells.parquet": {"risk_spell_id", "instrument", "partition_id", "event_date", "actor_key", "posture_side", "duration_seconds"},
    "risk_intervals.parquet": {"risk_interval_id", "risk_spell_id", "execution_anchor_mode", "duration_seconds"},
    "risk_membership.parquet": {"risk_membership_id", "physical_order_key", "duration_seconds"},
    "exposure_intervals.parquet": {"execution_anchor_mode", "exposure_mask", "duration_seconds"},
    "withdrawal_events.parquet": {"withdrawal_event_id", "physical_order_key", "visible_qty_removed"},
    "episode_risk_links.parquet": {
        "instrument", "episode_id", "execution_cluster_id", "risk_spell_id",
        "risk_interval_id", "physical_order_key",
    },
    "placebo_schedules.parquet": {"draw_id", "schedule_block_id", "accepted", "rejection_reason"},
    "placebo_support.parquet": {"draw_id", "accepted", "rejection_reason"},
    "placebo_statistics.parquet": {"draw_id", "row_kind", "time_at_risk_seconds", "withdrawal_count"},
    "coverage_and_balance.csv": {"instrument", "event_date", "actor_count", "actor_day_count", "risk_spell_count", "time_at_risk_seconds", "withdrawal_count", "censor_count", "censor_reason"},
    "clock_coverage_audit.parquet": {
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
    "stratum_statistics.csv": {"instrument", "contrast_label", "time_at_risk_seconds", "withdrawal_count", "withdrawal_intensity"},
}
REQUIRED_UNIT_FIELDS = {
    "risk_spells.parquet": {"duration_seconds"},
    "risk_intervals.parquet": {"duration_seconds"},
    "risk_membership.parquet": {"duration_seconds"},
    "exposure_intervals.parquet": {"duration_seconds"},
    "withdrawal_events.parquet": {"visible_qty_removed"},
    "episode_risk_links.parquet": {
        "episode_id", "execution_cluster_id", "risk_spell_id", "risk_interval_id",
        "physical_order_key",
    },
    "placebo_schedules.parquet": {"draw_id", "timestamps", "sort_indices"},
    "placebo_support.parquet": {"draw_id", "support_boundary_count", "supported_start_count"},
    "placebo_statistics.parquet": {"time_at_risk_seconds", "withdrawal_count", "withdrawal_intensity"},
    "coverage_and_balance.csv": {"time_at_risk_seconds", "withdrawal_count"},
    "clock_coverage_audit.parquet": {
        "total_event_count", "accepted_event_count", "missing_timestamp_event_count",
        "clock_regression_event_count", "quarantined_event_count",
        "quarantine_episode_count", "recovered_quarantine_episode_count",
        "recovered_quarantine_duration_seconds", "unquantified_quarantine_event_count",
        "observed_accepted_span_seconds", "time_at_risk_seconds", "coverage_loss_fraction",
        "terminal_right_censored_spell_count", "partition_boundary_censored_spell_count",
        "clock_ambiguous_censored_spell_count",
    },
    "stratum_statistics.csv": {"time_at_risk_seconds", "withdrawal_count", "withdrawal_intensity"},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_hashed_file(entry: object, *, label: str) -> None:
    if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
        raise ValueError(f"invalid provenance entry: {label}")
    path = Path(entry["path"])
    if not path.is_file():
        raise FileNotFoundError(path)
    if entry.get("sha256") != _sha256(path):
        raise ValueError(f"provenance hash mismatch: {label}")


def _validate_manifest_provenance(manifest: Mapping[str, object]) -> None:
    if isinstance(manifest.get("seed"), bool) or not isinstance(manifest.get("seed"), int):
        raise ValueError("manifest seed must be an integer")
    for field in ("clock", "risk_policy", "exclusion_reasons"):
        if not isinstance(manifest.get(field), Mapping):
            raise ValueError(f"manifest {field} must be an object")
    units = manifest.get("units")
    if not isinstance(units, Mapping) or not SCIENTIFIC_TABLES <= set(units):
        raise ValueError("manifest units must cover every scientific output table")
    for table, required_fields in REQUIRED_UNIT_FIELDS.items():
        declared = units.get(table)
        if not isinstance(declared, Mapping) or not required_fields <= set(declared):
            raise ValueError(f"manifest units are incomplete for {table}")
        if any(not isinstance(declared[field], str) or not declared[field].strip() for field in required_fields):
            raise ValueError(f"manifest units contain an empty declaration for {table}")
    snapshot = manifest.get("code_snapshot")
    if not isinstance(snapshot, Mapping) or not isinstance(snapshot.get("git_head"), str):
        raise ValueError("manifest code_snapshot is incomplete")
    files = snapshot.get("files")
    if not isinstance(files, Mapping) or not files:
        raise ValueError("manifest code_snapshot.files must be non-empty")
    for name, entry in files.items():
        _validate_hashed_file(entry, label=f"code_snapshot.files.{name}")
    sources = manifest.get("sources")
    if not isinstance(sources, Mapping) or not sources:
        raise ValueError("manifest sources must be non-empty")
    for instrument, raw_sources in sources.items():
        if not isinstance(raw_sources, Mapping):
            raise ValueError(f"manifest sources.{instrument} must be an object")
        missing = REQUIRED_SOURCE_KINDS - set(raw_sources)
        if missing:
            raise ValueError(f"manifest sources.{instrument} missing: {', '.join(sorted(missing))}")
        for name in REQUIRED_SOURCE_KINDS:
            _validate_hashed_file(raw_sources[name], label=f"sources.{instrument}.{name}")


def _sum(frame: pl.DataFrame, column: str) -> float:
    # Header-only CSVs infer String columns; the empty additive total is zero.
    # Do not coerce nonempty malformed columns or change undefined intensities.
    values = frame.get_column(column)
    return 0.0 if values.is_empty() else float(values.sum() or 0.0)


def _validate_nonnegative_finite(frame: pl.DataFrame, columns: set[str], *, table: str) -> None:
    for column in columns & set(frame.columns):
        for value in frame.get_column(column).drop_nulls().to_list():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError(f"{table}.{column} must contain finite non-negative numbers")


def _validate_clock_coverage_audit(audit: pl.DataFrame, *, intervals: pl.DataFrame) -> None:
    nonnegative = {
        "total_event_count", "accepted_event_count", "missing_timestamp_event_count",
        "clock_regression_event_count", "quarantined_event_count",
        "quarantine_episode_count", "recovered_quarantine_episode_count",
        "recovered_quarantine_duration_seconds", "unquantified_quarantine_event_count",
        "observed_accepted_span_seconds", "time_at_risk_seconds",
        "terminal_right_censored_spell_count", "partition_boundary_censored_spell_count",
        "clock_ambiguous_censored_spell_count",
    }
    _validate_nonnegative_finite(audit, nonnegative, table="clock_coverage_audit.parquet")
    interval_seconds: dict[tuple[object, object, object], float] = {}
    for row in intervals.select("instrument", "partition_id", "event_date", "duration_seconds").iter_rows(named=True):
        key = (row["instrument"], row["partition_id"], row["event_date"])
        interval_seconds[key] = interval_seconds.get(key, 0.0) + float(row["duration_seconds"] or 0.0)
    for row in audit.iter_rows(named=True):
        if row["accepted_event_count"] + row["quarantined_event_count"] != row["total_event_count"]:
            raise ValueError("clock coverage audit accepted plus quarantined counts must equal total events")
        if row["missing_timestamp_event_count"] + row["clock_regression_event_count"] != row["quarantined_event_count"]:
            raise ValueError("clock coverage audit missing plus regression counts must equal quarantined events")
        fraction = row["coverage_loss_fraction"]
        if fraction is not None and (not isinstance(fraction, (int, float)) or not math.isfinite(fraction) or not 0 <= fraction <= 1):
            raise ValueError("clock coverage audit coverage_loss_fraction must be null or in [0, 1]")
        key = (row["instrument"], row["partition_id"], row["event_date"])
        if not math.isclose(
            float(row["time_at_risk_seconds"]), interval_seconds.get(key, 0.0), rel_tol=1e-9, abs_tol=1e-9
        ):
            raise ValueError("clock coverage audit time_at_risk_seconds does not reconcile to risk intervals")


def _validate_scientific_accounting(tables: Mapping[str, pl.DataFrame]) -> None:
    for name, frame in tables.items():
        _validate_nonnegative_finite(
            frame,
            {
                "duration_seconds", "time_at_risk_seconds", "withdrawal_count",
                "visible_qty_removed", "withdrawal_intensity",
            },
            table=name,
        )
        if {"time_at_risk_seconds", "withdrawal_count", "withdrawal_intensity"} <= set(frame.columns):
            for row in frame.select(
                "time_at_risk_seconds", "withdrawal_count", "withdrawal_intensity"
            ).iter_rows(named=True):
                seconds = row["time_at_risk_seconds"]
                count = row["withdrawal_count"]
                intensity = row["withdrawal_intensity"]
                if seconds is None or count is None:
                    raise ValueError(f"{name}: intensity requires an observed numerator and denominator")
                if seconds == 0:
                    if count != 0 or intensity is not None:
                        raise ValueError(f"{name}: zero-time intensity must be null with zero withdrawals")
                elif intensity is None or not math.isclose(
                    intensity, count / seconds, rel_tol=1e-12, abs_tol=0.0
                ):
                    # Only serialization rounding is tolerated, not denominator regularization.
                    raise ValueError(f"{name}: intensity does not equal withdrawals per second")
    spells = tables["risk_spells.parquet"]
    intervals = tables["risk_intervals.parquet"]
    withdrawals = tables["withdrawal_events.parquet"]
    coverage = tables["coverage_and_balance.csv"]
    strata = tables["stratum_statistics.csv"]
    exposure = tables["exposure_intervals.parquet"]
    statistics = tables["placebo_statistics.parquet"]
    clock_coverage = tables["clock_coverage_audit.parquet"]
    _validate_clock_coverage_audit(clock_coverage, intervals=intervals)
    risk_seconds = _sum(intervals, "duration_seconds")
    tolerance = 1e-9 * max(1.0, abs(risk_seconds))
    duration_checks = {
        "spell/interval duration": (_sum(spells, "duration_seconds"), risk_seconds),
        "coverage/risk duration": (_sum(coverage, "time_at_risk_seconds"), risk_seconds),
        "stratum/risk duration": (_sum(strata, "time_at_risk_seconds"), risk_seconds),
    }
    for label, (left, right) in duration_checks.items():
        if abs(left - right) > tolerance:
            raise ValueError(f"cross-artifact accounting mismatch: {label}")
    count_checks = {
        "coverage spell count": (_sum(coverage, "risk_spell_count"), spells.height),
        "coverage withdrawal count": (_sum(coverage, "withdrawal_count"), withdrawals.height),
        "stratum withdrawal count": (_sum(strata, "withdrawal_count"), withdrawals.height),
    }
    for label, (left, right) in count_checks.items():
        if left != right:
            raise ValueError(f"cross-artifact accounting mismatch: {label}")
    if _sum(exposure, "duration_seconds") > risk_seconds + tolerance:
        raise ValueError("exposure duration exceeds time at risk")
    if withdrawals.get_column("physical_order_key").is_duplicated().any():
        raise ValueError("physical withdrawal keys must be unique")
    state_rows = statistics.filter(pl.col("row_kind") == "state")
    if state_rows.is_empty() and (not intervals.is_empty() or not withdrawals.is_empty()):
        raise ValueError("placebo statistics must contain state rows for observed risk")
    for draw_id, draw in state_rows.group_by("draw_id"):
        if abs(_sum(draw, "time_at_risk_seconds") - risk_seconds) > tolerance:
            raise ValueError(f"placebo draw {draw_id[0]} does not partition time at risk")
        if _sum(draw, "withdrawal_count") != withdrawals.height:
            raise ValueError(f"placebo draw {draw_id[0]} does not account for withdrawals")


def build_report(*, events_path: Path, output_path: Path, shift_events: int) -> None:
    events = pl.read_parquet(events_path)
    time_shift = add_time_shift_placebo(events, shift_events=shift_events)
    wrong_side = add_wrong_side_placebo(events)
    lines = [
        "# Spoofing Negative-Control Report",
        "",
        "Status: `descriptor_only`.",
        "",
        "This legacy report only enumerates unscored placebo descriptors; it is not an empirical validation.",
        "",
        f"Real candidate events: {events.height}",
        f"time_shift placebo events: {time_shift.height}",
        f"wrong_side placebo events: {wrong_side.height}",
        "",
        "## Interpretation",
        "",
        "These counts do not estimate association, intent, false-positive rate, precision, or recall.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def _load_validated_bundle(bundle_root: Path) -> tuple[dict, dict[str, pl.DataFrame]]:
    manifest_path = bundle_root / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    if not isinstance(manifest, Mapping) or manifest.get("design_version") != "empirical_controls_v1":
        raise ValueError("manifest must declare design_version empirical_controls_v1")
    _validate_manifest_provenance(manifest)
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("manifest artifacts must be an object")
    has_bundle_audit = (
        isinstance(manifest.get("bundle_audit_version"), int)
        and manifest["bundle_audit_version"] >= 1
    )
    required_artifacts = REQUIRED_EMPIRICAL_ARTIFACTS | (
        set(AUDIT_TABLE_COLUMNS) if has_bundle_audit else set()
    )
    missing = sorted(required_artifacts - set(artifacts))
    if missing:
        raise ValueError("manifest artifact graph is incomplete: " + ", ".join(missing))
    resolved: dict[str, Path] = {}
    root = bundle_root.resolve()
    for name in required_artifacts:
        entry = artifacts[name]
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise ValueError(f"invalid manifest artifact entry: {name}")
        path = (bundle_root / entry["path"]).resolve()
        if path != root and root not in path.parents:
            raise ValueError(f"artifact path escapes bundle root: {name}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if entry.get("sha256") != _sha256(path):
            raise ValueError(f"artifact hash mismatch: {name}")
        resolved[name] = path
    validation = json.loads(resolved["validation.json"].read_text())
    if not isinstance(validation, Mapping) or validation.get("status") != "pass":
        raise ValueError("bundle validation status is not pass")
    loaded = {
        name: (pl.read_csv(resolved[name]) if name.endswith(".csv") else pl.read_parquet(resolved[name]))
        for name in required_artifacts if name.endswith((".csv", ".parquet"))
    }
    table_columns = REQUIRED_TABLE_COLUMNS | (AUDIT_TABLE_COLUMNS if has_bundle_audit else {})
    for name, required_columns in table_columns.items():
        missing_columns = sorted(required_columns - set(loaded[name].columns))
        if missing_columns:
            raise ValueError(f"artifact schema missing columns for {name}: {', '.join(missing_columns)}")
    _validate_scientific_accounting(loaded)
    frames = {
        "coverage": loaded["coverage_and_balance.csv"],
        "strata": loaded["stratum_statistics.csv"],
        "support": loaded["placebo_support.parquet"],
        "statistics": loaded["placebo_statistics.parquet"],
        "spells": loaded["risk_spells.parquet"],
        "episode_risk_links": loaded["episode_risk_links.parquet"],
        "withdrawals": loaded["withdrawal_events.parquet"],
        "clock_coverage_audit.parquet": loaded["clock_coverage_audit.parquet"],
    }
    if has_bundle_audit:
        for name in AUDIT_TABLE_COLUMNS:
            if not manifest.get("units", {}).get(name):
                raise ValueError(f"missing audit units for {name}")
            frames[name] = loaded[name]
        for name, fields in {
            "profile_balance.csv": {"time_at_risk_seconds", "eligible_visible_qty_observed_seconds", "eligible_visible_qty_missing_seconds", "member_count_observed_seconds", "member_count_missing_seconds", "mean_member_age_seconds_observed_seconds", "mean_member_age_seconds_missing_seconds", "oldest_member_age_seconds_observed_seconds", "oldest_member_age_seconds_missing_seconds", "pre_spread_ticks_observed_seconds", "pre_spread_ticks_missing_seconds", "pre_total_depth_observed_seconds", "pre_total_depth_missing_seconds", "prior_event_count_60s_observed_seconds", "prior_event_count_60s_missing_seconds", "pre_mid_log_return_sd_60s_observed_seconds", "pre_mid_log_return_sd_60s_missing_seconds"},
            "pre_event_covariates.parquet": {"anchor_ts", "snapshot_sort_index", "anchor_sort_index", "eligible_visible_qty", "member_count", "mean_member_age_seconds", "oldest_member_age_seconds", "pre_spread_ticks", "pre_total_depth", "prior_event_count_60s", "pre_mid_log_return_sd_60s"},
            "common_support_audit.csv": {"total_seconds", "supported_seconds", "outside_support_seconds", "null_fraction"},
        }.items():
            if not fields <= set(manifest["units"][name]):
                raise ValueError(f"manifest units are incomplete for {name}")
        configured = manifest.get("configured_execution_anchor_modes")
        observed = manifest.get("observed_execution_anchor_modes")
        if configured != ["passive", "aggressive"] or not isinstance(observed, list) or not set(observed) <= set(configured):
            raise ValueError("invalid configured/observed execution anchor modes")
        if manifest.get("effective_clock", {}).get("timestamp_precedence") != ["TRADETIME", "BOOKOUTTIME", "BOOKIN", "SEQUENCETIME"]:
            raise ValueError("effective clock must match canonical replay")
        if manifest.get("comparability", {}).get("contract") != COMPARABILITY_CONTRACT:
            raise ValueError("manifest comparability contract is not the required pre-anchor no-adjustment design")
        risk_seconds = _sum(loaded["risk_intervals.parquet"], "duration_seconds")
        if not math.isclose(_sum(loaded["profile_balance.csv"], "time_at_risk_seconds"), risk_seconds, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("profile balance must partition risk duration")
        audit = loaded["common_support_audit.csv"]
        for row in audit.iter_rows(named=True):
            total, supported, outside, null_fraction = (row[name] for name in ("total_seconds", "supported_seconds", "outside_support_seconds", "null_fraction"))
            if any(value is None or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 for value in (total, supported, outside)):
                raise ValueError("common-support audit durations must be finite and non-negative")
            if not math.isclose(float(total), float(supported) + float(outside), rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError("common-support audit supported plus outside duration does not reconcile")
            if null_fraction is not None and (not isinstance(null_fraction, (int, float)) or not math.isfinite(null_fraction) or not 0 <= null_fraction <= 1):
                raise ValueError("common-support audit null_fraction must be null or in [0, 1]")
            if not row["support_established"] and row["supported"] != 0:
                raise ValueError("common-support audit unsupported row must have supported=0")
    summary = manifest.get("summary")
    if not isinstance(summary, Mapping):
        raise ValueError("manifest summary must be an object")
    support = frames["support"]
    observed_totals = {
        "risk_spell_count": frames["spells"].height,
        "time_at_risk_seconds": float(frames["coverage"].get_column("time_at_risk_seconds").sum()),
        "withdrawal_count": frames["withdrawals"].height,
        "accepted_placebo_blocks": support.filter(pl.col("accepted")).height,
        "rejected_placebo_blocks": support.filter(~pl.col("accepted")).height,
        "placebo_statistics_rows": frames["statistics"].height,
    }
    for key, observed in observed_totals.items():
        declared = summary.get(key)
        if declared != observed:
            raise ValueError(f"manifest summary total mismatch for {key}: declared={declared!r}, observed={observed!r}")
    return dict(manifest), frames


def _format_optional(value: object) -> str:
    return "unavailable (zero denominator)" if value is None else str(value)


def build_empirical_report(*, bundle_root: Path, output_path: Path | None = None) -> Path:
    """Validate an empirical bundle and write its descriptive report atomically."""
    manifest, frames = _load_validated_bundle(bundle_root)
    summary = manifest["summary"]
    coverage = frames["coverage"]
    clock_coverage = frames["clock_coverage_audit.parquet"]
    strata = frames["strata"]
    support = frames["support"]
    statistics = frames["statistics"]
    reasons = (
        support.filter(~pl.col("accepted"))
        .group_by("rejection_reason")
        .len()
        .sort("rejection_reason")
        .to_dicts()
        if not support.is_empty() else []
    )
    lines = [
        "# Empirical Spoofing Negative-Control Report",
        "",
        "This analysis is descriptive and exploratory. Control windows are not certified legitimate cases.",
        "It does not estimate false-positive rate, precision, recall, or intent and does not support a causal claim.",
        "Across-draw ranges are reference variability, not confidence intervals.",
        "",
        "## Coverage and support",
        "",
        f"- Instruments: {', '.join(sorted(map(str, coverage.get_column('instrument').unique().to_list())))}",
        f"- Distinct dates: {coverage.get_column('event_date').n_unique()}",
        f"- Actors (sum across instrument-days): {coverage.get_column('actor_count').sum()}",
        f"- Actor-days: {coverage.get_column('actor_day_count').sum()}",
        f"- Risk spells: {summary['risk_spell_count']}",
        f"- Seconds at risk: {summary['time_at_risk_seconds']}",
        f"- Physical withdrawals: {summary['withdrawal_count']}",
        f"- Censored spells: {coverage.get_column('censor_count').sum()}",
        f"- Censor reasons by instrument-day: {json.dumps(coverage.select('instrument', 'event_date', 'censor_reason').to_dicts(), sort_keys=True)}",
        f"- Accepted placebo blocks: {summary['accepted_placebo_blocks']}",
        f"- Rejected placebo blocks: {summary['rejected_placebo_blocks']}",
        f"- Rejection reasons: {json.dumps(reasons, sort_keys=True)}",
        f"- Recovered quantified clock-loss seconds: {_sum(clock_coverage, 'recovered_quarantine_duration_seconds')}",
        f"- Missing-timestamp quarantine events: {_sum(clock_coverage, 'missing_timestamp_event_count')}",
        f"- Open clock quarantines at partition end: {clock_coverage.filter(pl.col('open_quarantine_at_end_flag')).height}",
        f"- Terminally right-censored spells: {_sum(clock_coverage, 'terminal_right_censored_spell_count')}",
        "",
        "Recovered quantified clock-coverage loss is a lower bound. Missing or open clock quarantine and terminal right-censoring have unquantified coverage loss; an absent withdrawal in those periods is not a negative observation.",
        "",
        "## Observed strata",
        "",
        "| Stratum | Numerator (physical withdrawals) | Denominator (seconds at risk) | Intensity |",
        "|---|---:|---:|---:|",
    ]
    stratum_sort = [column for column in ("instrument", "contrast_label", "execution_anchor_mode", "exposure_mask") if column in strata.columns]
    for row in strata.sort(stratum_sort).iter_rows(named=True):
        lines.append(
            f"| {row.get('instrument')} / {row.get('contrast_label')} | {row.get('withdrawal_count')} | "
            f"{row.get('time_at_risk_seconds')} | {_format_optional(row.get('withdrawal_intensity'))} |"
        )
    lines.extend(["", "## Placebo statistics", ""])
    aggregate_statistics = statistics.filter(pl.col("row_kind") == "aggregate_contrast")
    statistic_sort = [
        column for column in ("draw_id", "comparison_label", "contrast_label", "partition_id", "actor_key", "side")
        if column in aggregate_statistics.columns
    ]
    for row in aggregate_statistics.sort(statistic_sort).iter_rows(named=True):
        lines.append(
            f"- Draw {row.get('draw_id')}, {row.get('comparison_label')}: "
            f"numerator={row.get('withdrawal_count')}, denominator={row.get('time_at_risk_seconds')}, "
            f"intensity={_format_optional(row.get('withdrawal_intensity'))}, "
            f"difference={_format_optional(row.get('comparison_intensity_difference'))}."
        )
    lines.extend([
        "",
        "Mixed/overlapping exposure and censored or rejected support remain separate and are not silently treated as negatives.",
        "The legacy negative-control report remains `descriptor_only`.",
        "",
    ])
    if isinstance(manifest.get("bundle_audit_version"), int) and manifest["bundle_audit_version"] >= 1:
        diagnostics = frames["risk_diagnostics.parquet"]
        transitions = frames["risk_transitions.parquet"]
        lines.extend([
            "## Clock quarantine and competing events", "",
            "clock_regression and missing timestamps are quarantined, not negative outcomes; no clock is sorted or clamped.",
            f"Effective clock: {json.dumps(manifest['effective_clock'], sort_keys=True)}",
            f"Diagnostic reasons: {json.dumps(diagnostics.group_by('reason').len().sort('reason').to_dicts(), sort_keys=True)}",
            f"Competing transitions: {json.dumps(transitions.group_by('transition_reason').len().sort('transition_reason').to_dicts(), sort_keys=True)}",
            "", "## Descriptive profile balance", "",
            "Comparisons are unadjusted. Common-support claims below use the __all_required__ rows: every required pre-anchor covariate must be finite and lie in its state-range intersection. Per-covariate rows are diagnostics only; this performs no matching and does not estimate an adjusted effect.",
            "Profile means are duration-weighted, not independent samples; missing covariates and disjoint ranges remain outside support rather than being imputed.",
            f"Unavailable adjustment covariates: {', '.join(manifest['comparability']['unavailable']) or 'none declared'}.",
            frames["profile_balance.csv"].sort('instrument', 'event_date', 'identity_level', 'side', 'contrast_label').write_csv(),
        ])
        support_audit = frames["common_support_audit.csv"]
        lines.extend([
            "", "## Pre-anchor common-support audit", "",
            "All covariates are fixed before each contrast anchor. Outcomes are not used in this audit. The joint required-covariate rows used for support claims are reported first.",
            support_audit.filter(pl.col('covariate') == '__all_required__').sort('instrument', 'event_date', 'identity_level', 'side', 'comparison_label', 'contrast_label').write_csv(),
            "", "Per-covariate diagnostic rows:",
            support_audit.filter(pl.col('covariate') != '__all_required__').sort('instrument', 'event_date', 'identity_level', 'side', 'comparison_label', 'covariate', 'contrast_label').write_csv(),
        ])
    else:
        lines.extend(["", "Legacy bundle: quarantine and profile-balance audit unavailable; comparisons are unadjusted."])
    destination = output_path or bundle_root / "report.md"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text("\n".join(lines))
    temporary.replace(destination)
    return destination


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spoofing negative-control report.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--events", type=Path)
    source.add_argument("--bundle-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shift-events", type=int, default=50)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.bundle_root is not None:
        output = build_empirical_report(bundle_root=args.bundle_root, output_path=args.output)
    else:
        if args.output is None:
            raise ValueError("--output is required with --events")
        build_report(events_path=args.events, output_path=args.output, shift_events=args.shift_events)
        output = args.output
    print(output)


if __name__ == "__main__":
    main()
