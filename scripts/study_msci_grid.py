#!/usr/bin/env python3
"""Fail-closed preflight/cohort and primary compute/verify MSCI-grid CLI."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping

import polars as pl

from spoofing_detection.lob.msci_grid import validate_grid_config


EXPECTED_CLOCK_POLICY = "spoofing_detection_sequence_replay_trade_fill_sequence_cancel_v3_utc_provisional"
EXPECTED_QUANTITY_SCHEMA = "h1_age30_withdrawal1_wmsci_price_pipeline_v2_complete_finite_rev"
STATE_ATTESTATION_SCHEMA = "msci_grid_state_coverage_attestation_v2"
HISTORICAL_CHECKPOINT_SCHEMA = "h1_checkpoint_identity_v1"
HISTORICAL_REPLAY_SCHEMA = "h1_stage_checkpoint_v1"
HISTORICAL_REQUIRED_SOURCES = {
    "src/spoofing_detection/lob/spoofing_metrics.py": "spoofing_metrics.py",
    "src/spoofing_detection/lob/replay_observation.py": "replay_observation.py",
    "scripts/compute_spoofing_metrics.py": None,
}
REQUIRED_STATE_COLUMNS = {
    "partition_id",
    "actor_key",
    "sort_index",
    "replay_time_ns",
    "DWI",
    "L_bid_topN",
    "L_ask_topN",
}
REQUIRED_EXECUTION_COLUMNS = {
    "partition_id",
    "actor_key",
    "execution_cluster_id",
    "cluster_first_sort_index",
    "cluster_last_sort_index",
    "cluster_end_replay_time_ns",
}


class PreflightError(ValueError):
    """An invalid declared input identity or manifest contract."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must contain a JSON object: {path}")
    return value


def _confined_path(raw: object, *, root: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PreflightError(f"{label} must be a non-empty path string")
    candidate = Path(raw).expanduser()
    # Resolve lexically before existence validation so a traversal attempt is
    # reported as a confinement violation rather than disguised as a missing file.
    resolved = candidate.resolve(strict=False) if candidate.is_absolute() else (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PreflightError(f"{label} must be confined to bundle root {root}: {resolved}") from exc
    if not resolved.is_file():
        raise PreflightError(f"{label} is not a file: {resolved}")
    return resolved


def _require_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
        raise PreflightError(f"{label} must be a lowercase-or-uppercase SHA-256 hex digest")
    return value.lower()


def _verify_declared_hash(path: Path, declared: object, *, label: str) -> str:
    expected = _require_sha256(declared, label=f"declared {label} sha256")
    actual = sha256(path)
    if actual != expected:
        raise PreflightError(f"{label} sha256 mismatch: expected {expected}, got {actual}")
    return actual


def _schema(path: Path, *, label: str) -> dict[str, pl.DataType]:
    try:
        return dict(pl.scan_parquet(path).collect_schema())
    except Exception as exc:  # Polars' parquet errors have several concrete types.
        raise PreflightError(f"cannot read {label} parquet schema: {path}: {exc}") from exc


def _require_non_boolean_integral(schema: Mapping[str, pl.DataType], columns: set[str], *, label: str) -> None:
    invalid = sorted(
        f"{column}={schema[column]}"
        for column in columns
        if column in schema and (schema[column] == pl.Boolean or not schema[column].is_integer())
    )
    if invalid:
        raise PreflightError(f"{label} requires non-Boolean integral clock/index columns: {invalid}")


def _require_numeric_profiles(schema: Mapping[str, pl.DataType], *, label: str) -> None:
    columns = {"DWI", "L_bid_topN", "L_ask_topN"}
    invalid = sorted(
        f"{column}={schema[column]}"
        for column in columns
        if column in schema and (schema[column] == pl.Boolean or not schema[column].is_numeric())
    )
    if invalid:
        raise PreflightError(f"{label} requires numeric profile columns: {invalid}")


def _coverage_attestation(
    metadata: Mapping[str, Any],
    *,
    bundle: Path,
    state_sha256: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate frozen evidence, never infer coverage from rows or live code."""
    attestation = metadata.get("state_coverage_attestation")
    if not isinstance(attestation, Mapping):
        return None, [
            "missing frozen producer state_coverage_attestation; rows and live code cannot establish feed completeness"
        ]
    failures: list[str] = []
    if attestation.get("schema") != STATE_ATTESTATION_SCHEMA:
        failures.append("state_coverage_attestation has an unknown schema")
    if attestation.get("state_time_series_sha256") != state_sha256:
        failures.append("state_coverage_attestation is not bound to the validated state_time_series SHA-256")
    expected_semantics = {
        "event_boundary_evaluated": True,
        "transition_zero_emitted": True,
        "sparse_zero_until_reentry": True,
        "state_client_mode": "execution-actors",
    }
    if {key: attestation.get(key) for key in expected_semantics} != expected_semantics:
        failures.append("state_coverage_attestation lacks the required sparse event-boundary producer semantics")
    snapshot = attestation.get("producer_snapshot")
    source_key = attestation.get("producer_source_key")
    source_hashes = metadata.get("scientific_source_hashes")
    if not isinstance(snapshot, Mapping) or not isinstance(source_key, str) or not isinstance(source_hashes, Mapping):
        failures.append("state_coverage_attestation lacks a producer snapshot bound to source identity")
    else:
        try:
            snapshot_path = _confined_path(snapshot.get("path"), root=bundle, label="producer snapshot")
            snapshot_sha = _verify_declared_hash(snapshot_path, snapshot.get("sha256"), label="producer snapshot")
            if source_hashes.get(source_key) != snapshot_sha:
                failures.append("producer snapshot SHA-256 does not match metadata scientific_source_hashes")
        except PreflightError as exc:
            failures.append(str(exc))
    return dict(attestation), failures


def _confined_directory(raw: object, *, root: Path, label: str) -> Path:
    if not isinstance(raw, str) or not raw:
        raise PreflightError(f"{label} must be a non-empty path string")
    candidate = Path(raw).expanduser()
    resolved = candidate.resolve(strict=False) if candidate.is_absolute() else (root / candidate).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PreflightError(f"{label} must be confined to historical root {root}: {resolved}") from exc
    if not resolved.is_dir():
        raise PreflightError(f"{label} is not a directory: {resolved}")
    return resolved


def _command_checkpoint_dir(metadata: Mapping[str, Any], *, historical_root: Path) -> Path:
    command = metadata.get("command")
    if not isinstance(command, list):
        raise PreflightError("historical metadata has no command list for checkpoint binding")
    positions = [index for index, value in enumerate(command) if value == "--checkpoint-dir"]
    if len(positions) != 1 or positions[0] + 1 >= len(command):
        raise PreflightError("historical metadata has no unambiguous --checkpoint-dir binding")
    return _confined_directory(command[positions[0] + 1], root=historical_root, label="historical checkpoint directory")


def _successful_run_log(bundle: Path, *, instrument: str, historical_root: Path) -> Path:
    # The checkpointed H1 layout records logs alongside its immutable run root.
    log = bundle.parent.parent / "logs" / f"{instrument}.log"
    try:
        log.relative_to(historical_root)
    except ValueError as exc:
        raise PreflightError(f"historical worker log must be confined to historical root {historical_root}: {log}") from exc
    if not log.is_file():
        raise PreflightError(f"historical worker log is not a file: {log}")
    try:
        exits = [json.loads(line) for line in log.read_text().splitlines() if line.startswith("{")]
    except (OSError, json.JSONDecodeError) as exc:
        raise PreflightError(f"cannot read historical worker log: {log}: {exc}") from exc
    if not exits or exits[-1].get("event") != "run_exit" or exits[-1].get("status") != "ok":
        raise PreflightError(f"historical worker log lacks final successful run_exit: {log}")
    return log


def _historical_checkpoint_evidence(
    metadata: Mapping[str, Any], *, bundle: Path, state_sha256: str, instrument: str, historical_root: Path
) -> dict[str, Any]:
    """Validate frozen H1 checkpoint evidence; current bytes only prove identity equality."""
    try:
        bundle.resolve(strict=True).relative_to(historical_root)
    except ValueError as exc:
        raise PreflightError("source bundle must be confined to explicitly pinned historical root") from exc
    checkpoint = _command_checkpoint_dir(metadata, historical_root=historical_root)
    identity_path = checkpoint / "identity.json"
    identity = _read_json(identity_path, label="historical checkpoint identity")
    if identity.get("format") != HISTORICAL_CHECKPOINT_SCHEMA:
        raise PreflightError("historical checkpoint identity has incompatible format")
    replay_manifest_path = checkpoint / "replay" / "manifest.json"
    replay_manifest = _read_json(replay_manifest_path, label="historical replay manifest")
    frames = replay_manifest.get("frames")
    if replay_manifest.get("format") != HISTORICAL_REPLAY_SCHEMA or replay_manifest.get("stage") != "replay" or not isinstance(frames, list) or not frames:
        raise PreflightError("historical replay manifest has incompatible replay frame contract")
    frame = frames[0]
    if not isinstance(frame, Mapping) or frame.get("filename") != "frame_000.parquet":
        raise PreflightError("historical replay manifest lacks frame_000.parquet")
    frame_path = _confined_path("replay/frame_000.parquet", root=checkpoint, label="historical replay frame")
    frame_sha = _verify_declared_hash(frame_path, frame.get("sha256"), label="historical replay frame")
    if frame_sha != state_sha256 or frame.get("row_count") != metadata.get("row_counts", {}).get("state_time_series"):
        raise PreflightError("historical replay frame-0 hash or row count does not bind bundle state")
    sources = identity.get("source_sha256")
    metadata_sources = metadata.get("scientific_source_hashes")
    if not isinstance(sources, Mapping) or not isinstance(metadata_sources, Mapping):
        raise PreflightError("historical checkpoint or bundle lacks source identities")
    source_bindings: dict[str, dict[str, str]] = {}
    repo_root = Path(__file__).resolve().parents[1]
    for relative, metadata_key in HISTORICAL_REQUIRED_SOURCES.items():
        source_hash = _require_sha256(sources.get(relative), label=f"historical source {relative}")
        if metadata_key is None:
            if metadata.get("generator_sha256") != source_hash:
                raise PreflightError("historical compute generator hash does not bind bundle metadata")
        elif metadata_sources.get(metadata_key) != source_hash:
            raise PreflightError(f"historical source {relative} does not bind bundle scientific_source_hashes")
        live = repo_root / relative
        if not live.is_file() or sha256(live) != source_hash:
            raise PreflightError(f"live source bytes do not equal recorded historical source identity: {relative}")
        source_bindings[relative] = {"sha256": source_hash, "current_path": str(live)}
    inputs = identity.get("inputs")
    arguments = identity.get("resolved_compute_args")
    input_hashes = metadata.get("input_hashes")
    if not isinstance(inputs, Mapping) or not isinstance(arguments, Mapping) or not isinstance(input_hashes, Mapping):
        raise PreflightError("historical checkpoint lacks input or resolved-argument identity")
    config = inputs.get("config")
    raw = inputs.get("raw")
    if not isinstance(config, Mapping) or not isinstance(raw, Mapping):
        raise PreflightError("historical checkpoint lacks config/raw identity")
    config_path = _confined_path(config.get("path"), root=historical_root, label="historical checkpoint config")
    config_sha = _verify_declared_hash(config_path, config.get("sha256"), label="historical checkpoint config")
    raw_sha = _require_sha256(raw.get("sha256"), label="historical checkpoint raw input")
    metadata_raw_sha = _require_sha256(input_hashes.get("raw_events_sha256"), label="bundle raw input")
    if metadata.get("config_sha256") != config_sha or input_hashes.get("config_sha256") != config_sha or metadata_raw_sha != raw_sha:
        raise PreflightError("historical config/raw identity does not bind bundle metadata")
    expected_args = {"clock_policy_v3": True, "top_n": 10, "window_seconds": 1.0, "compact_state": True, "state_client_mode": "execution-actors"}
    if {key: arguments.get(key) for key in expected_args} != expected_args or metadata.get("compact_state") is not True or metadata.get("state_client_mode") != "execution-actors":
        raise PreflightError("historical resolved arguments do not establish required v3 compact execution-actor semantics")
    log_path = _successful_run_log(bundle, instrument=instrument, historical_root=historical_root)
    return {"schema": "msci_grid_historical_h1_checkpoint_evidence_v1", "checkpoint": str(checkpoint), "replay_frame_0": {"path": str(frame_path), "sha256": frame_sha, "row_count": frame["row_count"]}, "identity": {"path": str(identity_path), "source_bindings": source_bindings}, "config": {"path": str(config_path), "sha256": config_sha}, "raw_input": {"sha256": raw_sha}, "run_log": str(log_path), "semantics": {"event_boundary_evaluated": True, "transition_zero_emitted": True, "sparse_zero_until_reentry": True, "state_client_mode": "execution-actors"}, "integrity_limitation": "hash bindings are integrity evidence for protected historical inputs, not external authentication"}


def _check_quantity_binding(
    quantity_metadata: Mapping[str, Any],
    *,
    instrument: str,
    bundle: Path,
    metadata_path: Path,
    state_sha256: str,
    execution_sha256: str,
) -> None:
    if quantity_metadata.get("schema") != EXPECTED_QUANTITY_SCHEMA:
        raise PreflightError("quantity metadata schema is incompatible with the frozen H1 age30 withdrawal1 pipeline")
    contract = quantity_metadata.get("contract")
    if not isinstance(contract, Mapping) or contract.get("source_clock_policy_id") != EXPECTED_CLOCK_POLICY:
        raise PreflightError("quantity metadata source_clock_policy_id is incompatible")
    provenance = quantity_metadata.get("source_provenance")
    entry = provenance.get(instrument) if isinstance(provenance, Mapping) else None
    if not isinstance(entry, Mapping):
        raise PreflightError(f"quantity metadata has no source_provenance entry for {instrument}")
    raw_bundle = entry.get("bundle")
    if not isinstance(raw_bundle, str) or Path(raw_bundle).expanduser().resolve(strict=False) != bundle:
        raise PreflightError(f"quantity provenance bundle does not resolve to source bundle for {instrument}")
    if entry.get("metadata_sha256") != sha256(metadata_path):
        raise PreflightError(f"quantity provenance metadata_sha256 mismatch for {instrument}")
    artifact_hashes = entry.get("source_artifact_sha256")
    if not isinstance(artifact_hashes, Mapping):
        raise PreflightError(f"quantity provenance lacks source_artifact_sha256 for {instrument}")
    for key, actual in (("state_time_series", state_sha256), ("execution_metrics", execution_sha256)):
        if artifact_hashes.get(key) != actual:
            raise PreflightError(f"quantity provenance {key} SHA-256 mismatch for {instrument}")


def _scalar_count(frame: pl.LazyFrame) -> int:
    """Collect one scalar aggregate without a multi-query materialization spike."""
    return int(frame.collect(engine="streaming").item())


def _scan_state_diagnostics(state_path: Path, execution_path: Path) -> dict[str, int | str]:
    """Observe numeric/domain eligibility conditions without inferring grid support.

    This runs one bounded-column aggregate at a time.  The results describe
    downstream exclusions; only ambiguous keys and malformed types invalidate
    the input itself.  Complete [0,H] support is intentionally sampler-owned.
    """
    try:
        states = pl.scan_parquet(state_path).select("partition_id", "actor_key", "sort_index", "replay_time_ns", "DWI", "L_bid_topN", "L_ask_topN")
        executions = pl.scan_parquet(execution_path).select(
            "partition_id", "actor_key", "execution_cluster_id", "cluster_first_sort_index", "cluster_last_sort_index", "cluster_end_replay_time_ns"
        )
        observed_state_rows = _scalar_count(states.select(pl.len().alias("n")))
        observed_execution_rows = _scalar_count(executions.select(pl.len().alias("n")))
        duplicate_groups = _scalar_count(
            states.group_by("partition_id", "actor_key", "sort_index").len().filter(pl.col("len") > 1).select(pl.len().alias("n"))
        )
        if duplicate_groups:
            raise PreflightError(f"state sort_index is duplicate/ambiguous in {duplicate_groups} identity groups")
        invalid_state_anchors = _scalar_count(
            states.filter(pl.col("sort_index").is_null() | pl.col("replay_time_ns").is_null()).select(pl.len().alias("n"))
        )
        invalid_profiles = _scalar_count(
            states.filter(
                pl.col("DWI").is_null()
                | ~pl.col("DWI").is_finite()
                | (pl.col("DWI") < -1)
                | (pl.col("DWI") > 1)
                | pl.col("L_bid_topN").is_null()
                | ~pl.col("L_bid_topN").is_finite()
                | (pl.col("L_bid_topN") < 0)
                | (pl.col("L_bid_topN") > 1)
                | pl.col("L_ask_topN").is_null()
                | ~pl.col("L_ask_topN").is_finite()
                | (pl.col("L_ask_topN") < 0)
                | (pl.col("L_ask_topN") > 1)
            ).select(pl.len().alias("n"))
        )
        # A state with a canonical index but a null clock remains the selected
        # baseline/exact-t0 candidate.  Do not discard it and silently fall
        # back to an older row; the downstream diagnostic must report it.
        canonical_states = states.filter(pl.col("sort_index").is_not_null())
        valid_state_clocks = canonical_states.filter(pl.col("replay_time_ns").is_not_null())
        regressions = _scalar_count(
            valid_state_clocks.sort("partition_id", "actor_key", "sort_index")
            .with_columns(pl.col("replay_time_ns").diff().over("partition_id", "actor_key").alias("_clock_delta"))
            .filter(pl.col("_clock_delta") < 0)
            .select(pl.len().alias("n"))
        )
        invalid_execution_anchors = _scalar_count(
            executions.filter(
                pl.col("cluster_first_sort_index").is_null()
                | pl.col("cluster_last_sort_index").is_null()
                | pl.col("cluster_end_replay_time_ns").is_null()
                | (pl.col("cluster_first_sort_index") > pl.col("cluster_last_sort_index"))
            ).select(pl.len().alias("n"))
        )
        valid_executions = executions.filter(
            pl.col("cluster_first_sort_index").is_not_null()
            & pl.col("cluster_last_sort_index").is_not_null()
            & pl.col("cluster_end_replay_time_ns").is_not_null()
            & (pl.col("cluster_first_sort_index") <= pl.col("cluster_last_sort_index"))
        )
        state_keys = canonical_states.select("partition_id", "actor_key", "sort_index", "replay_time_ns").with_columns(
            pl.lit(1, dtype=pl.Int8).alias("_state_present")
        )
        missing_exact_t0 = _scalar_count(
            valid_executions.join(
                state_keys,
                left_on=["partition_id", "actor_key", "cluster_last_sort_index"],
                right_on=["partition_id", "actor_key", "sort_index"],
                how="left",
            )
            .filter(
                pl.col("_state_present").is_null()
                | pl.col("replay_time_ns").is_null()
                | (pl.col("replay_time_ns") != pl.col("cluster_end_replay_time_ns"))
            )
            .select(pl.len().alias("n"))
        )
        # The baseline must be a canonical state strictly before the first fill
        # but it may share the execution-end clock: canonical index precedence
        # is the strict condition.  It is an eligibility condition, not proof
        # of a complete future horizon.
        baseline_states = state_keys.rename(
            {"sort_index": "_state_sort_index", "replay_time_ns": "_baseline_replay_time_ns", "_state_present": "_baseline_present"}
        ).sort("partition_id", "actor_key", "_state_sort_index")
        missing_baseline = _scalar_count(
            valid_executions.sort("partition_id", "actor_key", "cluster_first_sort_index")
            .join_asof(
                baseline_states,
                left_on="cluster_first_sort_index",
                right_on="_state_sort_index",
                by=["partition_id", "actor_key"],
                strategy="backward",
                allow_exact_matches=False,
            )
            .filter(
                pl.col("_baseline_present").is_null()
                | pl.col("_baseline_replay_time_ns").is_null()
                | (pl.col("_baseline_replay_time_ns") > pl.col("cluster_end_replay_time_ns"))
            )
            .select(pl.len().alias("n"))
        )
    except PreflightError:
        raise
    except Exception as exc:
        raise PreflightError(f"cannot scan state/execution diagnostics: {exc}") from exc
    return {
        "status": "observed",
        "observed_state_rows": observed_state_rows,
        "observed_execution_rows": observed_execution_rows,
        "duplicate_sort_index_groups": duplicate_groups,
        "clock_regression_rows": regressions,
        "invalid_state_anchor_rows": invalid_state_anchors,
        "invalid_execution_anchor_rows": invalid_execution_anchors,
        "invalid_profile_rows": invalid_profiles,
        "missing_exact_t0_rows": missing_exact_t0,
        "missing_exact_baseline_rows": missing_baseline,
        "empirical_grid": "not_checked_until_sampling",
    }


def _instrument_report(
    *,
    bundle: Path,
    quantity_metadata: Mapping[str, Any],
    historical_root: Path | None,
) -> tuple[dict[str, Any], list[str], bool]:
    metadata_path = bundle / "metadata.json"
    metadata = _read_json(metadata_path, label="source metadata")
    instrument = bundle.name
    if metadata.get("clock_policy_id") != EXPECTED_CLOCK_POLICY:
        raise PreflightError(f"{instrument} metadata clock_policy_id is incompatible")
    if metadata.get("top_n") != 10 or metadata.get("window_seconds") != 1.0:
        raise PreflightError(f"{instrument} metadata must freeze top_n=10 and window_seconds=1.0")
    paths = metadata.get("paths")
    hashes = metadata.get("artifact_hashes")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping):
        raise PreflightError(f"{instrument} metadata lacks paths/artifact_hashes mappings")
    state_path = _confined_path(paths.get("state_time_series"), root=bundle, label=f"{instrument} state_time_series")
    execution_path = _confined_path(paths.get("execution_metrics"), root=bundle, label=f"{instrument} execution_metrics")
    state_sha = _verify_declared_hash(state_path, hashes.get("state_time_series"), label=f"{instrument} state_time_series")
    execution_sha = _verify_declared_hash(execution_path, hashes.get("execution_metrics"), label=f"{instrument} execution_metrics")
    state_schema = _schema(state_path, label=f"{instrument} state_time_series")
    execution_schema = _schema(execution_path, label=f"{instrument} execution_metrics")
    missing_state = sorted(REQUIRED_STATE_COLUMNS - set(state_schema))
    missing_execution = sorted(REQUIRED_EXECUTION_COLUMNS - set(execution_schema))
    if missing_state or missing_execution:
        raise PreflightError(
            f"{instrument} required parquet columns missing: state={missing_state}, execution={missing_execution}"
        )
    _require_non_boolean_integral(
        state_schema,
        {"sort_index", "replay_time_ns"},
        label=f"{instrument} state_time_series",
    )
    _require_non_boolean_integral(
        execution_schema,
        {"cluster_first_sort_index", "cluster_last_sort_index", "cluster_end_replay_time_ns"},
        label=f"{instrument} execution_metrics",
    )
    _require_numeric_profiles(state_schema, label=f"{instrument} state_time_series")
    _check_quantity_binding(
        quantity_metadata,
        instrument=instrument,
        bundle=bundle,
        metadata_path=metadata_path,
        state_sha256=state_sha,
        execution_sha256=execution_sha,
    )
    attestation, coverage_failures = _coverage_attestation(metadata, bundle=bundle, state_sha256=state_sha)
    entry: dict[str, Any] = {
        "instrument": instrument,
        "metadata": {"path": str(metadata_path), "sha256": sha256(metadata_path)},
        "artifacts": {
            "state_time_series": {"path": str(state_path), "sha256": state_sha},
            "execution_metrics": {"path": str(execution_path), "sha256": execution_sha},
        },
        "schema": {"state_required_columns_missing": missing_state, "execution_required_columns_missing": missing_execution},
        "declared_row_counts": {
            "state_time_series": metadata.get("row_counts", {}).get("state_time_series") if isinstance(metadata.get("row_counts"), Mapping) else None,
            "execution_metrics": metadata.get("row_counts", {}).get("execution_metrics") if isinstance(metadata.get("row_counts"), Mapping) else None,
        },
    }
    if coverage_failures and historical_root is not None:
        # A supplied historical root is an explicit request for this verifier,
        # never authority inferred from paths embedded in self-certified metadata.
        historical_evidence = _historical_checkpoint_evidence(
            metadata,
            bundle=bundle,
            state_sha256=state_sha,
            instrument=instrument,
            historical_root=historical_root,
        )
        entry["coverage_historical_evidence"] = historical_evidence
        entry["coverage_diagnostics"] = _scan_state_diagnostics(state_path, execution_path)
        return entry, [], True
    if coverage_failures:
        entry["coverage_diagnostics"] = {"status": "not_run", "reason": "missing_attestation", "reasons": coverage_failures}
        return entry, coverage_failures, False
    entry["coverage_attestation"] = attestation
    entry["coverage_diagnostics"] = _scan_state_diagnostics(state_path, execution_path)
    return entry, [], True


def _empty_counts() -> dict[str, int | None]:
    return {
        "instruments_discovered": 0,
        "instruments_validated": 0,
        "market_update_coverage_attested": 0,
        "actor_disappearance_coverage_attested": 0,
        "market_update_coverage_missing": 0,
        "actor_disappearance_coverage_missing": 0,
        "execution_rows_declared": 0,
        "state_rows_declared": 0,
        "diagnostics_instruments_observed": 0,
        "state_rows_observed": None,
        "execution_rows_observed": None,
        "duplicate_sort_index_groups_observed": None,
        "clock_regression_rows_observed": None,
        "invalid_state_anchor_rows_observed": None,
        "invalid_execution_anchor_rows_observed": None,
        "invalid_profile_rows_observed": None,
        "missing_exact_t0_rows_observed": None,
        "missing_exact_baseline_rows_observed": None,
    }


def preflight(
    config_path: Path,
    source_root: Path,
    quantity_root: Path,
    output_root: Path,
    *,
    command: list[str],
    historical_root: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Build an auditable preflight report and return its process exit code."""
    report: dict[str, Any] = {
        "format": "msci_grid_preflight_v1",
        "stage": "preflight",
        "status": "invalid_input",
        "command": command,
        "input_digests": {},
        "counts": _empty_counts(),
        "coverage": {
            "market_update": {"status": "unverified", "reason": "not yet evaluated"},
            "actor_disappearance": {"status": "unverified", "reason": "not yet evaluated"},
            "empirical_grid": "not_checked_until_sampling",
        },
        "diagnostic_execution": {"status": "not_run", "reason": "producer coverage evidence not yet evaluated"},
        "instruments": [],
        "deferred_to_task6": [
            "No cohort membership is inferred here: exact top-positive-W and U/Q checks require task 6's validated quantity membership join.",
            "No row sample is used as evidence of market-update or actor-disappearance coverage.",
        ],
        "minimum_additional_evidence": [],
        "readiness_blockers": [],
    }
    try:
        config = _read_json(config_path, label="MSCI grid config")
        validate_grid_config(config)
        source_root = source_root.resolve(strict=True)
        quantity_root = quantity_root.resolve(strict=True)
        if historical_root is not None:
            historical_root = historical_root.resolve(strict=True)
            if not historical_root.is_dir():
                raise PreflightError("historical-root must be a directory")
        if not source_root.is_dir() or not quantity_root.is_dir():
            raise PreflightError("source-root and quantity-root must be directories")
        quantity_metadata_path = quantity_root / "metadata.json"
        quantity_metadata = _read_json(quantity_metadata_path, label="quantity metadata")
        report["input_digests"] = {
            "config": {"path": str(config_path.resolve(strict=True)), "sha256": sha256(config_path)},
            "quantity_metadata": {"path": str(quantity_metadata_path), "sha256": sha256(quantity_metadata_path)},
        }
        metadata_paths = sorted(source_root.glob("*/metadata.json"))
        if not metadata_paths:
            raise PreflightError(f"source-root contains no instrument metadata.json files: {source_root}")
        for metadata_path in metadata_paths:
            bundle = metadata_path.parent.resolve(strict=True)
            report["counts"]["instruments_discovered"] += 1
            entry, coverage_failures, attested = _instrument_report(
                bundle=bundle,
                quantity_metadata=quantity_metadata,
                historical_root=historical_root,
            )
            report["instruments"].append(entry)
            report["input_digests"][f"source_{bundle.name}"] = entry["metadata"]
            report["counts"]["instruments_validated"] += 1
            declared = entry["declared_row_counts"]
            for output_key, count_key in (("execution_metrics", "execution_rows_declared"), ("state_time_series", "state_rows_declared")):
                if isinstance(declared.get(output_key), int):
                    report["counts"][count_key] += declared[output_key]
            if attested:
                report["counts"]["market_update_coverage_attested"] += 1
                report["counts"]["actor_disappearance_coverage_attested"] += 1
                diagnostics = entry["coverage_diagnostics"]
                if diagnostics.get("status") == "observed":
                    report["counts"]["diagnostics_instruments_observed"] += 1
                    observed_counts = {
                        "state_rows_observed": "observed_state_rows",
                        "execution_rows_observed": "observed_execution_rows",
                        "duplicate_sort_index_groups_observed": "duplicate_sort_index_groups",
                        "clock_regression_rows_observed": "clock_regression_rows",
                        "invalid_state_anchor_rows_observed": "invalid_state_anchor_rows",
                        "invalid_execution_anchor_rows_observed": "invalid_execution_anchor_rows",
                        "invalid_profile_rows_observed": "invalid_profile_rows",
                        "missing_exact_t0_rows_observed": "missing_exact_t0_rows",
                        "missing_exact_baseline_rows_observed": "missing_exact_baseline_rows",
                    }
                    for report_key, diagnostic_key in observed_counts.items():
                        if report["counts"][report_key] is None:
                            report["counts"][report_key] = 0
                        report["counts"][report_key] += int(diagnostics[diagnostic_key])
            else:
                report["counts"]["market_update_coverage_missing"] += 1
                report["counts"]["actor_disappearance_coverage_missing"] += 1
                report["readiness_blockers"].extend(f"{bundle.name}: {reason}" for reason in coverage_failures)
        if report["readiness_blockers"]:
            report["status"] = "blocked_missing_state_evidence"
            report["coverage"] = {
                "market_update": {"status": "unverified", "reason": "frozen producer attestation absent or invalid"},
                "actor_disappearance": {"status": "unverified", "reason": "frozen producer attestation absent or invalid"},
                "empirical_grid": "not_checked_until_sampling",
            }
            report["minimum_additional_evidence"] = [
                "For each instrument, preserve a producer snapshot inside the frozen bundle and a state_coverage_attestation bound to the state parquet SHA-256 and scientific_source_hashes.",
                "The attestation must explicitly cover state emission on every surrounding market update and explicit zero-profile emission on actor disappearance; otherwise rerun the minimum required source replay that produces such evidence.",
            ]
            report["diagnostic_execution"] = {
                "status": "not_run",
                "reason": "producer coverage evidence was not verified for every instrument",
            }
            return report, 1
        report["coverage"] = {
            "market_update": {"status": "attested", "reason": "all instruments have bound producer evidence"},
            "actor_disappearance": {"status": "attested", "reason": "all instruments have bound sparse producer evidence"},
            "empirical_grid": "not_checked_until_sampling",
        }
        observed = report["counts"]["diagnostics_instruments_observed"]
        report["diagnostic_execution"] = {
            "status": "observed" if observed == report["counts"]["instruments_validated"] else "partially_observed",
            "instruments_observed": observed,
        }
        report["status"] = "ready_for_task6_cohort_gate"
        return report, 0
    except (OSError, PreflightError, ValueError) as exc:
        report["status"] = "invalid_input"
        report["diagnostic_execution"] = {"status": "invalid", "reason": str(exc)}
        report["readiness_blockers"].append(str(exc))
        return report, 2


COHORT_FORMAT = "msci_grid_cohort_v1"
COHORT_REQUIRED_DIAGNOSTIC_COLUMNS = {
    "instrument", "partition_id", "execution_cluster_id", "actor_key", "actor_id", "identity_level",
    "identity_source", "execution_anchor_mode", "execution_side", "deceptive_side", "cluster_first_sort_index",
    "cluster_last_sort_index", "cluster_end_fill_anchor_ns", "WMSCI_event", "withdrawal_to_execution_ratio",
    "favorable_mid_move_pre_fill", "post_cancel_mid_reversion", "post_cancel_mid_reversion_complete", "episode_id",
}
COHORT_REQUIRED_MEMBERSHIP_COLUMNS = {
    "instrument", "execution_cluster_id", "finite_W", "positive_W", "exact_top_positive_W",
    "U_over_Q_gt_1_finite", "after_W_then_U_over_Q_gt_1",
}
COHORT_REQUIRED_SOURCE_COLUMNS = {
    "partition_id", "execution_cluster_id", "actor_key", "actor_id", "identity_level", "identity_source",
    "execution_anchor_mode", "execution_side", "deceptive_side", "cluster_first_sort_index",
    "cluster_last_sort_index", "cluster_end_fill_anchor_ns",
}
COHORT_IDENTITY_COLUMNS = (
    "actor_key", "actor_id", "identity_level", "identity_source", "execution_anchor_mode", "execution_side",
    "deceptive_side", "cluster_first_sort_index", "cluster_last_sort_index", "cluster_end_fill_anchor_ns",
)
COHORT_MEMBERSHIP_FLAG_COLUMNS = (
    "finite_W", "positive_W", "exact_top_positive_W", "U_over_Q_gt_1_finite", "after_W_then_U_over_Q_gt_1",
)
COHORT_REQUIRED_IDENTITY_COLUMNS = {
    "actor_key", "identity_level", "execution_anchor_mode", "execution_side", "deceptive_side",
    "cluster_first_sort_index", "cluster_last_sort_index", "cluster_end_fill_anchor_ns",
}


def _require_frame_columns(frame: pl.DataFrame, columns: set[str], *, label: str) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise PreflightError(f"{label} missing required columns: {missing}")


def _require_unique_non_null(frame: pl.DataFrame, columns: list[str], *, label: str) -> None:
    nulls = frame.select([pl.col(column).is_null().sum().alias(column) for column in columns]).row(0, named=True)
    missing = {column: int(count) for column, count in nulls.items() if count}
    if missing:
        raise PreflightError(f"{label} has null required IDs: {missing}")
    duplicates = frame.group_by(columns).len().filter(pl.col("len") > 1).height
    if duplicates:
        raise PreflightError(f"{label} has {duplicates} duplicate identity groups for {columns}")


def _require_boolean_non_null(frame: pl.DataFrame, columns: tuple[str, ...], *, label: str) -> None:
    invalid = sorted(f"{column}={frame.schema[column]}" for column in columns if frame.schema[column] != pl.Boolean)
    if invalid:
        raise PreflightError(f"{label} requires Boolean flags: {invalid}")
    nulls = frame.select([pl.col(column).is_null().sum().alias(column) for column in columns]).row(0, named=True)
    missing = {column: int(count) for column, count in nulls.items() if count}
    if missing:
        raise PreflightError(f"{label} has null required Boolean flags: {missing}")


def _quantity_artifact(quantity_root: Path, metadata: Mapping[str, Any], name: str) -> tuple[Path, str]:
    outputs = metadata.get("outputs")
    if not isinstance(outputs, Mapping):
        raise PreflightError("quantity metadata lacks immutable output hashes")
    expected = outputs.get(name)
    if not isinstance(expected, str):
        raise PreflightError(f"quantity metadata lacks output hash for {name}")
    path = _confined_path(name, root=quantity_root, label=f"quantity {name}")
    return path, _verify_declared_hash(path, expected, label=f"quantity {name}")


def _verify_preflight_binding(
    path: Path | None,
    *,
    config_sha256: str,
    quantity_metadata_sha256: str,
    source_metadata_sha256: Mapping[str, str],
) -> dict[str, Any] | None:
    if path is None:
        return None
    report = _read_json(path.resolve(strict=True), label="preflight report")
    if report.get("format") != "msci_grid_preflight_v1" or report.get("status") != "ready_for_task6_cohort_gate":
        raise PreflightError("preflight report is not a ready Task-6 gate")
    digests = report.get("input_digests")
    if not isinstance(digests, Mapping):
        raise PreflightError("preflight report lacks input digests")
    expected: dict[str, str] = {"config": config_sha256, "quantity_metadata": quantity_metadata_sha256}
    expected.update({f"source_{instrument}": digest for instrument, digest in source_metadata_sha256.items()})
    actual_keys = set(digests)
    if actual_keys != set(expected):
        raise PreflightError("preflight report input identities do not cover exactly the requested cohort inputs")
    for key, digest in expected.items():
        entry = digests.get(key)
        if not isinstance(entry, Mapping) or entry.get("sha256") != digest:
            raise PreflightError(f"preflight report digest mismatch for {key}")
    return {"path": str(path.resolve(strict=True)), "sha256": sha256(path)}


def _validated_source_executions(
    source_root: Path, quantity_metadata: Mapping[str, Any]
) -> tuple[pl.DataFrame, dict[str, Any], dict[str, str]]:
    provenance = quantity_metadata.get("source_provenance")
    if not isinstance(provenance, Mapping) or not provenance:
        raise PreflightError("quantity metadata lacks source_provenance")
    frames: list[pl.DataFrame] = []
    artifact_provenance: dict[str, Any] = {}
    metadata_digests: dict[str, str] = {}
    for instrument in sorted(provenance):
        entry = provenance[instrument]
        if not isinstance(instrument, str) or not isinstance(entry, Mapping):
            raise PreflightError("quantity source_provenance has an invalid instrument entry")
        bundle = (source_root / instrument).resolve(strict=True)
        if not bundle.is_dir():
            raise PreflightError(f"source bundle is not a directory for {instrument}: {bundle}")
        declared_bundle = entry.get("bundle")
        if not isinstance(declared_bundle, str) or Path(declared_bundle).expanduser().resolve(strict=False) != bundle:
            raise PreflightError(f"quantity source provenance bundle mismatch for {instrument}")
        metadata_path = bundle / "metadata.json"
        metadata_sha = _verify_declared_hash(metadata_path, entry.get("metadata_sha256"), label=f"source {instrument} metadata")
        metadata_digests[instrument] = metadata_sha
        metadata = _read_json(metadata_path, label=f"source {instrument} metadata")
        paths = metadata.get("paths")
        hashes = metadata.get("artifact_hashes")
        provenance_hashes = entry.get("source_artifact_sha256")
        if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or not isinstance(provenance_hashes, Mapping):
            raise PreflightError(f"source {instrument} lacks execution artifact provenance")
        execution_path = _confined_path(paths.get("execution_metrics"), root=bundle, label=f"source {instrument} execution_metrics")
        execution_sha = _verify_declared_hash(execution_path, hashes.get("execution_metrics"), label=f"source {instrument} execution_metrics")
        if provenance_hashes.get("execution_metrics") != execution_sha:
            raise PreflightError(f"quantity source execution hash mismatch for {instrument}")
        try:
            source = pl.read_parquet(execution_path).select(sorted(COHORT_REQUIRED_SOURCE_COLUMNS))
        except Exception as exc:
            raise PreflightError(f"cannot read source {instrument} execution metrics: {exc}") from exc
        _require_frame_columns(source, COHORT_REQUIRED_SOURCE_COLUMNS, label=f"source {instrument} execution metrics")
        _require_unique_non_null(source, ["partition_id", "execution_cluster_id"], label=f"source {instrument} execution metrics")
        frames.append(source.with_columns(pl.lit(instrument).alias("instrument")).select("instrument", pl.all().exclude("instrument")))
        artifact_provenance[instrument] = {
            "metadata": {"path": str(metadata_path), "sha256": metadata_sha},
            "execution_metrics": {"path": str(execution_path), "sha256": execution_sha, "row_count": source.height},
        }
    return pl.concat(frames), artifact_provenance, metadata_digests


def _reconstruct_membership(diagnostics: pl.DataFrame, membership: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, dict[str, int]]]:
    _require_unique_non_null(diagnostics, ["instrument", "partition_id", "execution_cluster_id"], label="quantity diagnostics")
    _require_unique_non_null(membership, ["instrument", "execution_cluster_id"], label="saved quantity membership")
    _require_boolean_non_null(membership, COHORT_MEMBERSHIP_FLAG_COLUMNS, label="saved quantity membership")
    diag_ids = diagnostics.select("instrument", "execution_cluster_id")
    membership_ids = membership.select("instrument", "execution_cluster_id")
    if diag_ids.join(membership_ids, on=["instrument", "execution_cluster_id"], how="anti").height or membership_ids.join(diag_ids, on=["instrument", "execution_cluster_id"], how="anti").height:
        raise PreflightError("saved quantity membership IDs do not match quantity diagnostics IDs")
    finite_w = pl.col("WMSCI_event").is_not_null() & pl.col("WMSCI_event").is_finite()
    exact_parts: list[pl.DataFrame] = []
    counts: dict[str, dict[str, int]] = {}
    for instrument in diagnostics.get_column("instrument").unique().sort().to_list():
        frame = diagnostics.filter(pl.col("instrument") == instrument)
        positive = frame.filter(finite_w & (pl.col("WMSCI_event") > 0)).sort(
            ["WMSCI_event", "execution_cluster_id"], descending=[True, False]
        )
        k = math.ceil(0.05 * positive.height)
        exact_parts.append(positive.head(k).select("instrument", "execution_cluster_id"))
        counts[str(instrument)] = {
            "all_quantity_rows": frame.height,
            "positive_finite_W": positive.height,
            "exact_top_positive_W": k,
            "cohort_after_U_over_Q_gt_1": 0,
        }
    exact_ids = pl.concat(exact_parts) if exact_parts else pl.DataFrame(schema={"instrument": pl.String, "execution_cluster_id": pl.String})
    reconstructed = diagnostics.join(exact_ids.with_columns(pl.lit(True).alias("_exact")), on=["instrument", "execution_cluster_id"], how="left").with_columns(
        pl.col("_exact").fill_null(False).alias("exact_top_positive_W_reconstructed"),
        (pl.col("withdrawal_to_execution_ratio").is_not_null() & pl.col("withdrawal_to_execution_ratio").is_finite() & (pl.col("withdrawal_to_execution_ratio") > 1)).alias("U_over_Q_gt_1_finite_reconstructed"),
    ).drop("_exact")
    saved = membership.select(
        "instrument", "execution_cluster_id", "finite_W", "positive_W", "exact_top_positive_W", "U_over_Q_gt_1_finite", "after_W_then_U_over_Q_gt_1"
    )
    checked = reconstructed.join(saved, on=["instrument", "execution_cluster_id"], how="inner")
    expected_finite = finite_w
    expected_positive = finite_w & (pl.col("WMSCI_event") > 0)
    expected_after = pl.col("exact_top_positive_W_reconstructed") & pl.col("U_over_Q_gt_1_finite_reconstructed")
    disagreement = checked.filter(
        (pl.col("finite_W") != expected_finite)
        | (pl.col("positive_W") != expected_positive)
        | (pl.col("exact_top_positive_W") != pl.col("exact_top_positive_W_reconstructed"))
        | (pl.col("U_over_Q_gt_1_finite") != pl.col("U_over_Q_gt_1_finite_reconstructed"))
        | (pl.col("after_W_then_U_over_Q_gt_1") != expected_after)
    )
    if disagreement.height:
        raise PreflightError(f"saved quantity membership disagrees with deterministic W/U reconstruction in {disagreement.height} rows")
    for instrument in counts:
        counts[instrument]["cohort_after_U_over_Q_gt_1"] = checked.filter(
            (pl.col("instrument") == instrument) & expected_after
        ).height
    return checked, counts


def cohort(
    config_path: Path,
    source_root: Path,
    quantity_root: Path,
    output_root: Path,
    *,
    command: list[str],
    preflight_report_path: Path | None = None,
) -> tuple[dict[str, Any], int]:
    """Select and persist the Task-6 price-independent execution cohort."""
    report: dict[str, Any] = {
        "format": COHORT_FORMAT, "stage": "cohort", "status": "invalid_input", "command": command,
        "input_digests": {}, "artifacts": {}, "counts": {}, "selection": {
            "top_positive_W_fraction": 0.05,
            "ranking": "finite_positive_WMSCI_event_desc_then_execution_cluster_id_asc_ceil_per_instrument",
            "quantity_gate": "finite_withdrawal_to_execution_ratio_strictly_gt_1",
            "price_gates": "not_applied", "shared_order_episode": "retained_reporting_only_not_a_gate",
        },
    }
    try:
        config = _read_json(config_path, label="MSCI grid config")
        validate_grid_config(config)
        if config.get("cohort") != "exact_top_positive_W_5pct_then_U_over_Q_gt_1_before_prices":
            raise PreflightError("MSCI grid config cohort contract is incompatible")
        config_path = config_path.resolve(strict=True)
        source_root = source_root.resolve(strict=True)
        quantity_root = quantity_root.resolve(strict=True)
        if not source_root.is_dir() or not quantity_root.is_dir():
            raise PreflightError("source-root and quantity-root must be directories")
        quantity_metadata_path = quantity_root / "metadata.json"
        quantity_metadata = _read_json(quantity_metadata_path, label="quantity metadata")
        if quantity_metadata.get("schema") != EXPECTED_QUANTITY_SCHEMA:
            raise PreflightError("quantity metadata schema is incompatible with Task-6 cohort selection")
        contract = quantity_metadata.get("contract")
        if not isinstance(contract, Mapping) or contract.get("top_positive_w_fraction") != 0.05 or contract.get("selection_order", [])[:2] != ["positive_W_exact_top_5pct", "U_over_Q_gt_1"] or contract.get("episode_role") != "recomputed_shared_order_reporting_only_not_a_selection_gate":
            raise PreflightError("quantity metadata does not freeze the required W/U selection contract")
        diagnostics_path, diagnostics_sha = _quantity_artifact(quantity_root, quantity_metadata, "refreshed_execution_diagnostics.csv")
        membership_path, membership_sha = _quantity_artifact(quantity_root, quantity_metadata, "step_memberships.csv")
        quantity_metadata_sha = sha256(quantity_metadata_path)
        sources, source_artifacts, source_metadata_digests = _validated_source_executions(source_root, quantity_metadata)
        preflight_binding = _verify_preflight_binding(
            preflight_report_path, config_sha256=sha256(config_path), quantity_metadata_sha256=quantity_metadata_sha,
            source_metadata_sha256=source_metadata_digests,
        )
        diagnostics = pl.read_csv(diagnostics_path)
        membership = pl.read_csv(membership_path)
        _require_frame_columns(diagnostics, COHORT_REQUIRED_DIAGNOSTIC_COLUMNS, label="quantity diagnostics")
        _require_frame_columns(membership, COHORT_REQUIRED_MEMBERSHIP_COLUMNS, label="saved quantity membership")
        checked, counts = _reconstruct_membership(diagnostics, membership)
        _require_unique_non_null(sources, ["instrument", "partition_id", "execution_cluster_id"], label="source execution anchors")
        quantity_keys = checked.select("instrument", "partition_id", "execution_cluster_id")
        source_keys = sources.select("instrument", "partition_id", "execution_cluster_id")
        missing_source = quantity_keys.join(source_keys, on=["instrument", "partition_id", "execution_cluster_id"], how="anti")
        extra_source = source_keys.join(quantity_keys, on=["instrument", "partition_id", "execution_cluster_id"], how="anti")
        if missing_source.height or extra_source.height:
            raise PreflightError(f"source/quantity execution anchor IDs are not one-to-one: missing_source={missing_source.height}, extra_source={extra_source.height}")
        joined = checked.join(sources, on=["instrument", "partition_id", "execution_cluster_id"], how="inner", suffix="_source")
        mismatch = None
        for column in COHORT_IDENTITY_COLUMNS:
            source_column = pl.col(f"{column}_source")
            if column in COHORT_REQUIRED_IDENTITY_COLUMNS:
                different = joined.filter(pl.col(column).is_null() | source_column.is_null() | pl.col(column).ne_missing(source_column))
            else:
                different = joined.filter(pl.col(column).ne_missing(source_column))
            if different.height:
                mismatch = column
                break
        if mismatch is not None:
            raise PreflightError(f"quantity/source actor or execution identity mismatch in {mismatch}")
        cohort_frame = joined.filter(pl.col("after_W_then_U_over_Q_gt_1")).select(
            "instrument", "partition_id", "execution_cluster_id", "actor_key", "actor_id", "identity_level", "identity_source",
            "execution_anchor_mode", "execution_side", "deceptive_side", "cluster_first_sort_index", "cluster_last_sort_index",
            "cluster_end_fill_anchor_ns", "WMSCI_event", "withdrawal_to_execution_ratio", "exact_top_positive_W",
            "U_over_Q_gt_1_finite", "after_W_then_U_over_Q_gt_1", "favorable_mid_move_pre_fill", "post_cancel_mid_reversion",
            "post_cancel_mid_reversion_complete", "episode_id",
        ).sort(["instrument", "execution_cluster_id"])
        if output_root.exists():
            raise PreflightError(f"output root already exists and will not be rewritten: {output_root}")
        output_root.mkdir(parents=True)
        cohort_path = output_root / "cohort.csv"
        cohort_frame.write_csv(cohort_path)
        report.update({
            "status": "complete",
            "input_digests": {
                "config": {"path": str(config_path), "sha256": sha256(config_path)},
                "quantity_metadata": {"path": str(quantity_metadata_path), "sha256": quantity_metadata_sha},
                **{f"source_{instrument}": {"sha256": digest} for instrument, digest in source_metadata_digests.items()},
            },
            "artifacts": {
                "quantity": {
                    "refreshed_execution_diagnostics": {"path": str(diagnostics_path), "sha256": diagnostics_sha, "row_count": diagnostics.height},
                    "step_memberships": {"path": str(membership_path), "sha256": membership_sha, "row_count": membership.height},
                },
                "source": source_artifacts,
                "cohort": {"path": str(cohort_path), "sha256": sha256(cohort_path), "row_count": cohort_frame.height},
            },
            "counts": counts,
            "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
            "preflight_binding": preflight_binding,
        })
        (output_root / "cohort_provenance.json").write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
        return report, 0
    except (OSError, PreflightError, ValueError, pl.exceptions.PolarsError) as exc:
        report["status"] = "invalid_input"
        report["error"] = str(exc)
        return report, 2


def _write_new_report(output_root: Path, report: Mapping[str, Any]) -> Path:
    output_root.mkdir(parents=True, exist_ok=False)
    path = output_root / "preflight.json"
    path.write_text(json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return path


def report_campaign(args: argparse.Namespace, command: list[str]) -> dict[str, Any]:
    """Verify every full-campaign setting against current code before rendering."""
    from scripts import msci_grid_study_consumer as consumer
    from scripts import msci_grid_study_report as report

    if (args.study_mode != "full" or args.instrument is not None or args.max_executions is not None
            or args.resume or args.setting_kind != "primary" or args.grid_step_ns is not None
            or args.lag_ns is not None or args.quantity_root is not None or args.historical_root is not None):
        raise ValueError("report requires full campaign without instrument/cap/resume/setting overrides")
    if args.cohort_root is None or args.preflight_report is None or args.primary_run_root is None:
        raise ValueError("report requires --cohort-root, --preflight-report and --primary-run-root")
    if args.output_root.expanduser().exists():
        raise ValueError("report output root must be new")
    raw = _read_json(args.config.expanduser(), label="config")
    validate_grid_config(raw)
    if (raw["grid_step_ns"], raw["lag_ns"], raw["horizon_ns"]) != (10_000_000, 50_000_000, 1_000_000_000):
        raise ValueError("report primary must be 10ms/50ms/1s")
    declared = {f"resolution_{step//1_000_000}ms_lag50ms": (step, raw["lag_ns"])
                for step in raw["grid_sensitivity_ns"]}
    declared.update({f"lag_{lag//1_000_000}ms_step{raw['lag_sensitivity_grid_ns']//1_000_000}ms":
                     (raw["lag_sensitivity_grid_ns"], lag) for lag in raw["lag_sensitivity_ns"]})
    if declared != {label: (step*1_000_000, lag*1_000_000) for label, (step, lag) in report.CASES.items()}:
        raise ValueError("report requires exact declared campaign settings")
    labeled = []
    for item in args.sensitivity_run:
        label, sep, path = item.partition("=")
        if not sep or not path:
            raise ValueError("--sensitivity-run requires LABEL=PATH")
        labeled.append((label, Path(path).expanduser().resolve(strict=True)))
    labels = [label for label, _ in labeled]
    if len(set(labels)) != len(labels) or set(labels) != set(declared):
        raise ValueError("missing, duplicate or extra sensitivity labels")
    primary_root = args.primary_run_root.expanduser().resolve(strict=True)
    roots = dict(labeled)
    identities = {}
    receipts = {}
    baseline = None
    aliases = {}
    for label, root in [("primary", primary_root), *labeled]:
        kind = "primary" if label == "primary" else label.split("_")[0]
        pair = None if label == "primary" else declared[label]
        if label == "resolution_10ms_lag50ms" and root == primary_root:
            kind, pair = "primary", None
            aliases[label] = "primary"
        elif label == "lag_50ms_step5ms" and root == roots["resolution_5ms_lag50ms"]:
            kind = "resolution"
            aliases[label] = "resolution_5ms_lag50ms"
        context = consumer._load_context(
            args.config.expanduser(), args.source_root.expanduser(), args.cohort_root.expanduser(),
            args.preflight_report.expanduser(), None, None, "full",
            coverage_witness_path=args.coverage_witness.expanduser() if args.coverage_witness else None,
            setting_kind=kind, grid_step_ns=pair[0] if pair else None, lag_ns=pair[1] if pair else None)
        common = {key: context[key] for key in ["cohort_sha", "cohort_provenance_sha", "preflight_sha",
                  "coverage_sidecar_sha", "scope", "scope_sha", "selected", "code_hashes"]}
        if baseline is None:
            baseline = common
        elif common != baseline:
            raise ValueError("campaign cohort/input/coverage/state/source identities differ")
        receipts[label] = consumer.verify(root, context, command=command)
        identities[label] = {f"{row['instrument']}::{row['partition_id']}": consumer._identity(row, context)
                             for row in context["selected"]}
    return report.generate_report(primary_root, args.cohort_root.expanduser(), labeled,
        args.output_root.expanduser(), expected_identities=identities,
        verification_metadata={"scientific_verification": receipts, "aliases": aliases,
                               "command": command, "report_module_sha256": sha256(Path(report.__file__))})


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--quantity-root", type=Path)
    parser.add_argument("--historical-root", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--preflight-report", type=Path)
    parser.add_argument("--coverage-witness", type=Path, help="explicit derived observed-event sidecar (compute/verify only)")
    parser.add_argument("--cohort-root", type=Path)
    parser.add_argument("--instrument")
    parser.add_argument("--study-mode", choices=("pilot", "full"), default="full")
    parser.add_argument("--max-executions", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--setting-kind", choices=("primary", "resolution", "lag"), default="primary")
    parser.add_argument("--grid-step-ns", type=int)
    parser.add_argument("--lag-ns", type=int)
    parser.add_argument("--primary-run-root", type=Path)
    parser.add_argument("--sensitivity-run", action="append", default=[], metavar="LABEL=PATH")
    parser.add_argument("--stage", choices=("preflight", "cohort", "compute", "verify", "report"), required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.expanduser()
    if args.stage in {"preflight", "cohort"} and output_root.exists():
        print(f"error: output root already exists and will not be rewritten: {output_root}", file=sys.stderr)
        return 2
    command = [str(Path(__file__).resolve()), *(argv if argv is not None else sys.argv[1:])]
    if args.stage == "report":
        try:
            result = report_campaign(args, command)
            print(f"MSCI_GRID_REPORT_COMPLETE {output_root} executions={result['counts']['cohort']}")
            return 0
        except (OSError, PreflightError, ValueError, KeyError, pl.exceptions.PolarsError) as exc:
            print(f"MSCI_GRID_REPORT_INVALID {exc}", file=sys.stderr)
            return 2
    if args.primary_run_root is not None or args.sensitivity_run:
        print("error: run-root options are only valid with report", file=sys.stderr)
        return 2
    if args.stage not in {"compute", "verify"} and (
        args.setting_kind != "primary" or args.grid_step_ns is not None or args.lag_ns is not None
    ):
        print("error: sensitivity setting options are only valid with compute/verify", file=sys.stderr)
        return 2
    if args.coverage_witness is not None and args.stage not in {"compute", "verify"}:
        print("error: --coverage-witness is only valid with compute/verify", file=sys.stderr)
        return 2
    if args.stage in {"compute", "verify"}:
        if args.study_mode == "full" and args.max_executions is not None:
            print("error: full --study-mode rejects --max-executions", file=sys.stderr)
            return 2
        if args.study_mode == "pilot" and (
            args.instrument != "RISANAMENTO" or args.max_executions is None or not 1 <= args.max_executions <= 5
        ):
            print("error: pilot --study-mode requires --instrument RISANAMENTO and --max-executions in 1..5", file=sys.stderr)
            return 2
        if args.quantity_root is not None or args.historical_root is not None:
            print("error: --quantity-root and --historical-root are not valid with compute/verify", file=sys.stderr)
            return 2
        if args.cohort_root is None or args.preflight_report is None:
            print("error: --cohort-root and --preflight-report are required with compute/verify", file=sys.stderr)
            return 2
        from scripts.msci_grid_study_consumer import ConsumerError, _load_context, compute, verify

        try:
            context = _load_context(
                args.config.expanduser(), args.source_root.expanduser(), args.cohort_root.expanduser(),
                args.preflight_report.expanduser(), args.instrument, args.max_executions, args.study_mode,
                coverage_witness_path=args.coverage_witness.expanduser() if args.coverage_witness is not None else None,
                setting_kind=args.setting_kind, grid_step_ns=args.grid_step_ns, lag_ns=args.lag_ns,
            )
            if args.stage == "compute":
                result = compute(output_root, context, resume=args.resume, command=command)
                print(f"MSCI_GRID_STUDY_COMPUTED {output_root} computed={result['computed']} reused={result['reused']} planned={result['planned']}")
            else:
                if args.resume:
                    raise ConsumerError("--resume is valid only with --stage compute")
                result = verify(output_root, context, command=command)
                print(f"MSCI_GRID_STUDY_VERIFIED {output_root} partitions={result['planned']} executions={result['executions']} invalid_exclusions={result['invalid_exclusions']}")
            return 0
        except (OSError, ConsumerError, ValueError, pl.exceptions.PolarsError) as exc:
            print(f"MSCI_GRID_STUDY_INVALID {exc}", file=sys.stderr)
            return 2
    if args.quantity_root is None:
        print("error: --quantity-root is required with preflight/cohort", file=sys.stderr)
        return 2
    if args.stage == "preflight":
        report, code = preflight(
            args.config.expanduser(),
            args.source_root.expanduser(),
            args.quantity_root.expanduser(),
            output_root,
            command=command,
            historical_root=args.historical_root.expanduser() if args.historical_root is not None else None,
        )
        report_path = _write_new_report(output_root, report)
        if code == 0:
            print(f"MSCI_GRID_PREFLIGHT_OK {report_path}")
        elif code == 1:
            print(f"MSCI_GRID_PREFLIGHT_BLOCKED {report_path}", file=sys.stderr)
        else:
            print(f"MSCI_GRID_PREFLIGHT_INVALID {report_path}", file=sys.stderr)
        return code
    if args.historical_root is not None:
        print("error: --historical-root is only valid with --stage preflight", file=sys.stderr)
        return 2
    report, code = cohort(
        args.config.expanduser(),
        args.source_root.expanduser(),
        args.quantity_root.expanduser(),
        output_root,
        command=command,
        preflight_report_path=args.preflight_report.expanduser() if args.preflight_report is not None else None,
    )
    if code == 0:
        print(f"MSCI_GRID_COHORT_OK {output_root / 'cohort_provenance.json'}")
    else:
        print(f"MSCI_GRID_COHORT_INVALID {report.get('error', 'unknown error')}", file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
