"""Fail-closed Task-7c compute/verify consumer for the frozen MSCI cohort."""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
import sys
from typing import Any

import polars as pl

from spoofing_detection.lob.msci_grid import _coverage_for_partition, validate_grid_config
from spoofing_detection.lob.msci_grid_runner import compute_partition_grids, verify_partition_grids
from spoofing_detection.lob.msci_grid_store import (
    _load_verified,
    _validate_identity,
    _validate_requested,
    read_or_compute_partition,
)


class ConsumerError(ValueError):
    """The immutable Task-6/Task-7 provenance or state contract is not met."""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ConsumerError(f"cannot read {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ConsumerError(f"{label} must be a JSON object")
    return value


def _hash_entry(entry: Any, label: str) -> str:
    if not isinstance(entry, Mapping) or not isinstance(entry.get("sha256"), str):
        raise ConsumerError(f"{label} lacks SHA-256 identity")
    return str(entry["sha256"])


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    ).hexdigest()


def _file_from_metadata(bundle: Path, metadata: Mapping[str, Any], key: str) -> tuple[Path, str]:
    paths, hashes = metadata.get("paths"), metadata.get("artifact_hashes")
    if not isinstance(paths, Mapping) or not isinstance(hashes, Mapping) or not isinstance(paths.get(key), str):
        raise ConsumerError(f"source metadata lacks {key} immutable artifact")
    path = Path(str(paths[key])).expanduser().resolve(strict=True)
    try:
        path.relative_to(bundle)
    except ValueError as exc:
        raise ConsumerError(f"source {key} is not confined to source bundle") from exc
    expected = hashes.get(key)
    if not isinstance(expected, str) or sha256(path) != expected:
        raise ConsumerError(f"source {key} hash mismatch")
    return path, expected


def _coverage_attested(entry: Mapping[str, Any], state_sha: str) -> Mapping[str, Any]:
    attestation = entry.get("coverage_attestation")
    historical = entry.get("coverage_historical_evidence")
    if isinstance(attestation, Mapping):
        semantics = {
            "event_boundary_evaluated": True,
            "transition_zero_emitted": True,
            "sparse_zero_until_reentry": True,
            "state_client_mode": "execution-actors",
        }
        if attestation.get("state_time_series_sha256") != state_sha or {key: attestation.get(key) for key in semantics} != semantics:
            raise ConsumerError("preflight attestation does not bind event-complete producer semantics to current states")
        return attestation
    if isinstance(historical, Mapping):
        semantics = historical.get("semantics")
        replay = historical.get("replay_frame_0")
        if not isinstance(semantics, Mapping) or not isinstance(replay, Mapping) or replay.get("sha256") != state_sha:
            raise ConsumerError("historical preflight evidence does not bind current states")
        if {key: semantics.get(key) for key in ("event_boundary_evaluated", "transition_zero_emitted", "sparse_zero_until_reentry", "state_client_mode")} != {"event_boundary_evaluated": True, "transition_zero_emitted": True, "sparse_zero_until_reentry": True, "state_client_mode": "execution-actors"}:
            raise ConsumerError("historical producer evidence lacks event-complete semantics")
        return historical
    raise ConsumerError("coverage_unverified: no immutable producer or historical checkpoint evidence")


def _partition_coverage(evidence: Mapping[str, Any], state_sha: str, partition: str) -> tuple[dict[str, Any], str]:
    """Pass through an explicit hash-bound witness; never derive support from rows."""
    witness = evidence.get("partition_coverage_witness")
    if not isinstance(witness, Mapping):
        coverage = {"segments": [], "invalid_intervals": []}
        return coverage, _canonical_hash({"status": "coverage_unverified", "partition_id": partition})
    if witness.get("state_time_series_sha256") != state_sha:
        raise ConsumerError("partition coverage witness is not bound to current states")
    segments = witness.get("segments")
    invalid = witness.get("invalid_intervals", [])
    coverage = {"segments": segments, "invalid_intervals": invalid}
    try:
        _coverage_for_partition(coverage, partition)
    except ValueError as exc:
        raise ConsumerError(f"partition coverage witness is invalid: {exc}") from exc
    if not isinstance(segments, list) or not isinstance(invalid, list):
        raise ConsumerError("partition coverage witness segments and invalid_intervals must be lists")
    normalized_segments: list[dict[str, Any]] = []
    normalized_invalid: list[dict[str, Any]] = []
    for segment in segments:
        if not isinstance(segment, Mapping) or segment.get("partition_id") != partition:
            continue
        if segment.get("attested") is not True or not isinstance(segment.get("source"), str) or not segment["source"]:
            raise ConsumerError("partition coverage witness segment lacks attested source")
        normalized_segments.append({
            "partition_id": partition, "start_ns": segment["start_ns"], "end_ns": segment["end_ns"],
            "attested": True, "source": segment["source"],
        })
    for interval in invalid:
        if not isinstance(interval, Mapping) or interval.get("partition_id") != partition:
            continue
        normalized_invalid.append({
            "partition_id": partition, "start_ns": interval["start_ns"], "end_ns": interval["end_ns"],
        })
    normalized = {"segments": normalized_segments, "invalid_intervals": normalized_invalid}
    return normalized, _canonical_hash({"state_sha256": state_sha, "partition_id": partition, **normalized})


def _primary_config(raw: Mapping[str, Any]) -> tuple[dict[str, Any], str, str]:
    validate_grid_config(raw)
    # Task 7c intentionally executes the frozen canonical primary only.
    config = dict(raw)
    config_key = f"primary_{config['grid_step_ns']}ns_lag{config['lag_ns']}ns_h{config['horizon_ns']}ns"
    return config, config_key, hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _fixed_setting(raw: Mapping[str, Any], kind: str, step: int | None, lag: int | None):
    """Derive only prespecified settings from the unchanged provenance-bound config."""
    config, key, digest = _primary_config(raw)
    if kind == "primary":
        if step is not None or lag is not None:
            raise ConsumerError("primary setting rejects explicit grid/lag overrides")
        return config, key, digest
    if kind not in {"resolution", "lag"}:
        raise ConsumerError("unknown setting kind")
    if type(step) is not int or type(lag) is not int:
        raise ConsumerError("sensitivity setting requires explicit integer step and lag")
    allowed = (
        step in raw["grid_sensitivity_ns"] and lag == raw["lag_ns"]
        if kind == "resolution" else
        step == raw["lag_sensitivity_grid_ns"] and lag in raw["lag_sensitivity_ns"]
    )
    if not allowed:
        raise ConsumerError("setting is not prespecified for the requested sensitivity kind")
    # The core validator checks the entire declared campaign against its
    # primary lag.  This effective config describes just one validated case;
    # campaign membership remains bound to the unchanged raw config above.
    config.update(grid_step_ns=step, lag_ns=lag, grid_sensitivity_ns=[step],
                  lag_sensitivity_ns=[lag], lag_sensitivity_grid_ns=step)
    validate_grid_config(config)
    key = f"{kind}_{step}ns_lag{lag}ns_h{config['horizon_ns']}ns"
    # The label is part of identity even for coincident numerical settings.
    return config, key, _canonical_hash({"kind": kind, "config": config})


def _code_hashes() -> dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    paths = {
        "study_cli": root / "scripts" / "study_msci_grid.py",
        "consumer": Path(__file__),
        "coverage_adapter": root / "scripts" / "msci_grid_coverage_witness.py",
        "grid": root / "src" / "spoofing_detection" / "lob" / "msci_grid.py",
        "runner": root / "src" / "spoofing_detection" / "lob" / "msci_grid_runner.py",
        "store": root / "src" / "spoofing_detection" / "lob" / "msci_grid_store.py",
        "spoofing_metrics": root / "src" / "spoofing_detection" / "lob" / "spoofing_metrics.py",
    }
    return {name: sha256(path) for name, path in paths.items()}


def _requested_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={"instrument": pl.String, "partition_id": pl.String, "actor_key": pl.String, "execution_cluster_id": pl.String}, strict=False)


def _preflight_instrument(report: Mapping[str, Any], instrument: str) -> Mapping[str, Any]:
    if report.get("format") != "msci_grid_preflight_v1" or report.get("status") != "ready_for_task6_cohort_gate":
        raise ConsumerError("preflight report is not a ready immutable Task-6 gate")
    matches = [entry for entry in report.get("instruments", []) if isinstance(entry, Mapping) and entry.get("instrument") == instrument]
    if len(matches) != 1:
        raise ConsumerError(f"preflight report has no unique instrument evidence for {instrument}")
    return matches[0]


def _load_context(
    config_path: Path, source_root: Path, cohort_root: Path, preflight_path: Path,
    instrument: str | None, max_executions: int | None, study_mode: str = "full",
    *, coverage_witness_path: Path | None = None,
    setting_kind: str = "primary", grid_step_ns: int | None = None, lag_ns: int | None = None,
) -> dict[str, Any]:
    if study_mode not in {"pilot", "full"}:
        raise ConsumerError("study-mode must be pilot or full")
    if study_mode == "full" and max_executions is not None:
        raise ConsumerError("full study-mode rejects --max-executions")
    if study_mode == "pilot" and (instrument != "RISANAMENTO" or isinstance(max_executions, bool) or not isinstance(max_executions, int) or not 1 <= max_executions <= 5):
        raise ConsumerError("pilot study-mode requires --instrument RISANAMENTO and --max-executions in 1..5")
    config_raw = _read_json(config_path.resolve(strict=True), "config")
    config, config_key, config_hash = _fixed_setting(config_raw, setting_kind, grid_step_ns, lag_ns)
    if setting_kind != "primary" and (study_mode != "full" or instrument is not None or max_executions is not None):
        raise ConsumerError("sensitivity requires the full cohort without instrument/cap selection")
    cohort_provenance_path = cohort_root.resolve(strict=True) / "cohort_provenance.json"
    cohort_path = cohort_root / "cohort.csv"
    cohort_report = _read_json(cohort_provenance_path, "cohort provenance")
    if cohort_report.get("format") != "msci_grid_cohort_v1" or cohort_report.get("status") != "complete":
        raise ConsumerError("cohort root is not a complete immutable Task-6 cohort")
    if sha256(cohort_path) != _hash_entry(cohort_report.get("artifacts", {}).get("cohort"), "cohort artifact"):
        raise ConsumerError("cohort CSV hash mismatch")
    preflight = _read_json(preflight_path.resolve(strict=True), "preflight report")
    sidecar = None
    sidecar_sha = None
    if coverage_witness_path is not None:
        from scripts.msci_grid_coverage_witness import load_sidecar
        try:
            sidecar = load_sidecar(coverage_witness_path, preflight_path)
        except (ValueError, KeyError, TypeError) as exc:
            raise ConsumerError(f"coverage sidecar invalid: {exc}") from exc
        sidecar_sha = sha256(coverage_witness_path)
    cohort_config_sha = _hash_entry(cohort_report.get("input_digests", {}).get("config"), "cohort config")
    if cohort_config_sha != sha256(config_path):
        raise ConsumerError("cohort config identity does not match requested config")
    if _hash_entry(preflight.get("input_digests", {}).get("config"), "preflight config") != sha256(config_path):
        raise ConsumerError("preflight config identity does not match requested config")
    # Do not infer identities as numbers or nullable epoch clocks as floats.
    cohort = pl.read_csv(cohort_path, schema_overrides={
        **{column: pl.String for column in ("instrument", "partition_id", "execution_cluster_id", "actor_key", "deceptive_side")},
        **{column: pl.Int64 for column in ("cluster_first_sort_index", "cluster_last_sort_index", "cluster_end_fill_anchor_ns")},
    })
    required = {"instrument", "partition_id", "execution_cluster_id", "actor_key", "deceptive_side", "cluster_first_sort_index", "cluster_last_sort_index", "cluster_end_fill_anchor_ns"}
    if missing := sorted(required - set(cohort.columns)):
        raise ConsumerError(f"cohort lacks required execution columns: {missing}")
    if instrument is not None:
        cohort = cohort.filter(pl.col("instrument") == instrument)
    cohort = cohort.sort(["instrument", "execution_cluster_id"])
    if max_executions is not None:
        if isinstance(max_executions, bool) or max_executions <= 0:
            raise ConsumerError("max-executions must be positive")
        cohort = cohort.head(max_executions)
    if cohort.height == 0:
        raise ConsumerError("selected cohort is empty")
    selected: list[dict[str, Any]] = []
    source_root = source_root.resolve(strict=True)
    for name in cohort.get_column("instrument").unique().sort().to_list():
        local = cohort.filter(pl.col("instrument") == name)
        bundle = (source_root / name).resolve(strict=True)
        metadata_path = bundle / "metadata.json"
        metadata = _read_json(metadata_path, f"source metadata {name}")
        metadata_sha = sha256(metadata_path)
        cohort_source = cohort_report.get("input_digests", {}).get(f"source_{name}")
        if _hash_entry(cohort_source, f"cohort source {name}") != metadata_sha or _hash_entry(preflight.get("input_digests", {}).get(f"source_{name}"), f"preflight source {name}") != metadata_sha:
            raise ConsumerError(f"source metadata provenance mismatch for {name}")
        execution_path, execution_sha = _file_from_metadata(bundle, metadata, "execution_metrics")
        state_path, state_sha = _file_from_metadata(bundle, metadata, "state_time_series")
        cohort_artifacts = cohort_report.get("artifacts", {}).get("source", {})
        if not isinstance(cohort_artifacts, Mapping) or not isinstance(cohort_artifacts.get(name), Mapping):
            raise ConsumerError(f"cohort provenance lacks source artifact identity for {name}")
        if _hash_entry(cohort_artifacts[name].get("execution_metrics"), "cohort execution") != execution_sha:
            raise ConsumerError(f"cohort execution artifact provenance mismatch for {name}")
        pf = _preflight_instrument(preflight, name)
        artifacts = pf.get("artifacts")
        if not isinstance(artifacts, Mapping) or _hash_entry(artifacts.get("execution_metrics"), "preflight execution") != execution_sha or _hash_entry(artifacts.get("state_time_series"), "preflight state") != state_sha:
            raise ConsumerError(f"preflight artifact provenance mismatch for {name}")
        coverage_evidence = _coverage_attested(pf, state_sha)
        if sidecar is not None:
            witness_entry = sidecar["instruments"].get(name)
            if not isinstance(witness_entry, Mapping) or witness_entry.get("artifacts") != artifacts:
                raise ConsumerError("coverage sidecar/current preflight artifact identity mismatch")
            if coverage_evidence.get("partition_coverage_witness", {}).get("segments"):
                raise ConsumerError("sidecar cannot override source historical support metadata")
            coverage_evidence = {**coverage_evidence, "partition_coverage_witness": witness_entry["partition_coverage_witness"]}
        ids = local.get_column("execution_cluster_id").to_list()
        source = pl.scan_parquet(execution_path).filter(pl.col("execution_cluster_id").is_in(ids)).collect(engine="streaming")
        source_required = {"partition_id", "actor_key", "execution_cluster_id", "deceptive_side", "cluster_first_sort_index", "cluster_last_sort_index", "cluster_end_fill_anchor_ns", "cluster_end_replay_time_ns"}
        if missing := sorted(source_required - set(source.columns)):
            raise ConsumerError(f"source execution metrics lacks canonical columns: {missing}")
        if source.height != local.height or source.select("partition_id", "execution_cluster_id").is_duplicated().any():
            raise ConsumerError(f"source canonical execution join is not one-to-one for {name}")
        joined = local.join(source, on=["partition_id", "execution_cluster_id"], how="inner", suffix="_source")
        if joined.height != local.height:
            raise ConsumerError(f"source canonical execution join is incomplete for {name}")
        for column in ("actor_key", "deceptive_side", "cluster_first_sort_index", "cluster_last_sort_index", "cluster_end_fill_anchor_ns"):
            if joined.filter(pl.col(column).is_null() | pl.col(f"{column}_source").is_null() | pl.col(column).ne_missing(pl.col(f"{column}_source"))).height:
                raise ConsumerError(f"cohort/source canonical execution mismatch in {column}")
        # Fill anchors bind cohort identity; replay anchors define the grid t0.
        # Their equality is not a contract of the v3 trade/sequence clocks.
        if source.schema["cluster_end_replay_time_ns"] != pl.Int64 or joined.filter(
            pl.col("cluster_end_replay_time_ns").is_null()
        ).height:
            raise ConsumerError("source cluster_end_replay_time_ns must be non-null Int64")
        for row in joined.select(
            pl.lit(name).alias("instrument"), "partition_id", "actor_key_source", "execution_cluster_id", "deceptive_side_source", "cluster_first_sort_index_source", "cluster_last_sort_index_source", "cluster_end_replay_time_ns"
        ).rename({"actor_key_source": "actor_key", "deceptive_side_source": "deceptive_side", "cluster_first_sort_index_source": "cluster_first_sort_index", "cluster_last_sort_index_source": "cluster_last_sort_index"}).to_dicts():
            coverage, coverage_sha = _partition_coverage(coverage_evidence, state_sha, str(row["partition_id"]))
            if sidecar_sha is not None:
                coverage_sha = _canonical_hash({"coverage": coverage_sha, "sidecar_manifest": sidecar_sha})
            selected.append({**row, "_state_path": str(state_path), "_state_sha": state_sha, "_execution_sha": execution_sha, "_coverage": coverage, "_coverage_sha": coverage_sha})
    scope = {
        "mode": study_mode, "instrument": instrument, "max_executions": max_executions,
        "ordering_rule": "instrument_then_execution_cluster_id_ascending",
        "selected_execution_ids": [
            {key: row[key] for key in ("instrument", "partition_id", "actor_key", "execution_cluster_id")}
            for row in selected
        ],
    }
    return {"config": config, "config_key": config_key, "config_hash": config_hash, **({"setting_contract": {"schema": "msci_grid_fixed_sensitivity_v1", "kind": setting_kind, "primary_config_sha256": sha256(config_path), "effective_config_sha256": config_hash}} if setting_kind != "primary" else {}), "cohort_sha": sha256(cohort_path), "cohort_provenance_sha": sha256(cohort_provenance_path), "preflight_sha": sha256(preflight_path), "coverage_sidecar_sha": sidecar_sha, "selected": selected, "scope": scope, "scope_sha": _canonical_hash(scope), "code_hashes": _code_hashes()}


def _states_for_actor(state_path: Path, partition: str, actor: str) -> pl.DataFrame:
    """Materialize one actor once, independent of other actor state volume."""
    return pl.scan_parquet(state_path).filter(
        (pl.col("partition_id") == partition) & (pl.col("actor_key") == actor)
    ).select("partition_id", "actor_key", "sort_index", "replay_time_ns", "DWI", "L_bid_topN", "L_ask_topN").collect(engine="streaming")


def _identity(row: Mapping[str, Any], context: Mapping[str, Any]) -> dict[str, Any]:
    validation_contract = {
        "schema": "msci_grid_validation_contract_v1", "config": context["config_hash"],
        "states": row["_state_sha"], "coverage_witness": row["_coverage_sha"],
        "verifier": context["code_hashes"]["runner"], "request_scope": context["scope_sha"],
    }
    return {"instrument": row["instrument"], "partition_id": row["partition_id"], "configuration_key": context["config_key"], "input_hashes": {"cohort": context["cohort_sha"], "cohort_provenance": context["cohort_provenance_sha"], "preflight": context["preflight_sha"], "executions": row["_execution_sha"], "states": row["_state_sha"], "request_scope": context["scope_sha"], "validation_contract": _canonical_hash(validation_contract)}, "config_hashes": {context["config_key"]: context["config_hash"]}, "code_hashes": context["code_hashes"]}


def _run_manifest(context: Mapping[str, Any], command: list[str]) -> dict[str, Any]:
    selected = context["selected"]
    partitions: dict[str, int] = {}
    partition_validation: dict[str, dict[str, str]] = {}
    for row in selected:
        key = f"{row['instrument']}::{row['partition_id']}"
        partitions[key] = partitions.get(key, 0) + 1
        partition_validation.setdefault(key, {
            "coverage_witness_sha256": row["_coverage_sha"],
            "validation_contract_sha256": _identity(row, context)["input_hashes"]["validation_contract"],
        })
    return {"format": "msci_grid_study_run_v1", "mode": context["scope"]["mode"], **({"setting_contract": context["setting_contract"]} if "setting_contract" in context else {}), "stage_contract": "prespecified_sensitivity" if "setting_contract" in context else "primary_only", "command": command, "settings": {"configuration_key": context["config_key"], "grid_step_ns": context["config"]["grid_step_ns"], "lag_ns": context["config"]["lag_ns"], "horizon_ns": context["config"]["horizon_ns"]}, "input_hashes": {"cohort": context["cohort_sha"], "cohort_provenance": context["cohort_provenance_sha"], "preflight": context["preflight_sha"], **({"coverage_sidecar": context["coverage_sidecar_sha"]} if context.get("coverage_sidecar_sha") else {})}, "code_hashes": context["code_hashes"], "software_versions": {"python": sys.version, "polars": pl.__version__}, "scope": {**context["scope"], "scope_sha256": context["scope_sha"]}, "requested_execution_ids": context["scope"]["selected_execution_ids"], "expected_partitions": partitions, "partition_validation": partition_validation}


def _read_verified(root: Path, identity: Mapping[str, Any], requested: pl.DataFrame):
    """Read-only store access: never create an attempt while verifying."""
    valid_identity = _validate_identity(identity)
    expected = _validate_requested(valid_identity, requested)
    complete = root / "partitions" / valid_identity["instrument"] / valid_identity["partition_id"] / valid_identity["configuration_key"] / "complete"
    if not complete.exists():
        raise ConsumerError(f"missing completed partition: {complete}")
    return _load_verified(complete, valid_identity, expected)


def _same_run_identity(stored: Mapping[str, Any], expected: Mapping[str, Any]) -> bool:
    """Commands are audit metadata, not a changing resume/verify identity."""
    return {key: value for key, value in stored.items() if key != "command"} == {
        key: value for key, value in expected.items() if key != "command"
    }


def _executions_frame(rows: list[dict[str, Any]]) -> pl.DataFrame:
    return pl.DataFrame([{
        key: row[key] for key in (
            "instrument", "partition_id", "actor_key", "execution_cluster_id", "deceptive_side",
            "cluster_first_sort_index", "cluster_last_sort_index", "cluster_end_replay_time_ns",
        )
    } for row in rows])


def _actor_rows(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    return [
        (actor, [row for row in rows if row["actor_key"] == actor])
        for actor in sorted({str(row["actor_key"]) for row in rows})
    ]


def _compute_and_verify(rows: list[dict[str, Any]], context: Mapping[str, Any]) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Keep each actor's states and scientific verification inside one callback unit."""
    trajectories: list[pl.DataFrame] = []
    summaries: list[pl.DataFrame] = []
    for actor, actor_rows in _actor_rows(rows):
        executions = _executions_frame(actor_rows)
        states = _states_for_actor(Path(actor_rows[0]["_state_path"]), str(actor_rows[0]["partition_id"]), actor)
        trajectory, summary = compute_partition_grids(executions, states, actor_rows[0]["_coverage"], context["config"])
        verify_partition_grids(
            executions, trajectory, summary, context["config"], states=states,
            coverage=actor_rows[0]["_coverage"],
        )
        trajectories.append(trajectory)
        summaries.append(summary)
    return pl.concat(trajectories, how="vertical"), pl.concat(summaries, how="vertical")


def _reverify_completion(completion: Any, rows: list[dict[str, Any]], context: Mapping[str, Any]) -> None:
    """Do not count store reuse as scientific reuse until it is reconstructed again."""
    for actor, actor_rows in _actor_rows(rows):
        executions = _executions_frame(actor_rows)
        states = _states_for_actor(Path(actor_rows[0]["_state_path"]), str(actor_rows[0]["partition_id"]), actor)
        trajectory = completion.trajectory.filter(pl.col("actor_key") == actor)
        summary = completion.summary.filter(pl.col("actor_key") == actor)
        verify_partition_grids(
            executions, trajectory, summary, context["config"], states=states,
            coverage=actor_rows[0]["_coverage"],
        )


def compute(root: Path, context: Mapping[str, Any], *, resume: bool, command: list[str]) -> dict[str, int]:
    manifest = _run_manifest(context, command)
    manifest_path = root / "run_manifest.json"
    if root.exists():
        if not resume:
            raise ConsumerError("output root exists; use --resume only for its exact immutable manifest")
        if not _same_run_identity(_read_json(manifest_path, "run manifest"), manifest):
            raise ConsumerError("resume manifest identity/mode/request mismatch")
    else:
        root.mkdir(parents=True)
        manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    computed = reused = 0
    by_partition: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in context["selected"]:
        by_partition.setdefault((row["instrument"], row["partition_id"]), []).append(row)
    for (_, partition), rows in by_partition.items():
        identity = _identity(rows[0], context)
        requested = _requested_frame(rows)
        completion = read_or_compute_partition(
            root / "partitions", identity, requested,
            lambda rows=rows: _compute_and_verify(rows, context),
        )
        if not completion.did_compute:
            _reverify_completion(completion, rows, context)
        computed += int(completion.did_compute)
        reused += int(not completion.did_compute)
    return {"computed": computed, "reused": reused, "planned": len(by_partition)}


def verify(root: Path, context: Mapping[str, Any], *, command: list[str]) -> dict[str, int]:
    if not _same_run_identity(_read_json(root / "run_manifest.json", "run manifest"), _run_manifest(context, command)):
        raise ConsumerError("run manifest does not match selected immutable study inputs")
    invalid = total = 0
    by_partition: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in context["selected"]:
        by_partition.setdefault((row["instrument"], row["partition_id"]), []).append(row)
    for (_, partition), rows in by_partition.items():
        requested = _requested_frame(rows)
        completion = _read_verified(root, _identity(rows[0], context), requested)
        _reverify_completion(completion, rows, context)
        total += completion.summary.height
        invalid += completion.summary.filter(~pl.col("complete_grid")).height
    return {"planned": len(by_partition), "executions": total, "invalid_exclusions": invalid}
