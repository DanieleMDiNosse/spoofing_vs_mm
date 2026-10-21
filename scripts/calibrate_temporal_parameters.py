#!/usr/bin/env python3
"""Stage-gated, checkpointed temporal-calibration reference extraction.

Only ``preflight`` and the bounded ``extract`` reference stage are runnable.
The latter persists source-population evidence; it deliberately does not estimate,
validate, report, or promote detector parameters.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from typing import Any, Mapping

import polars as pl

from spoofing_detection.lob.empirical_clock import CLOCK_POLICY_ID
from spoofing_detection.lob.empirical_controls_v2_budget import supervise
from spoofing_detection.lob.empirical_controls_v2_io import PartitionWriter, file_hash, scan_checkpoint, value_hash
from spoofing_detection.lob.empirical_controls_v2_partition import _OriginalIndices
from spoofing_detection.lob.panel import SORT_COLUMNS, _partition_id, sort_events
from spoofing_detection.lob.temporal_calibration_inputs import (
    AGE_EXPOSURE_SCHEMA,
    AGE_SPELL_SCHEMA,
    GAP_PAIR_SCHEMA,
    MARKET_STATE_SCHEMA,
    ORIGIN_SCHEMA,
    annotate_coverage_epochs,
    build_age_inputs,
    collect_gap_pairs,
    collect_market_states,
    collect_uncapped_risk,
    frozen_design,
    _partition_span_seconds,
    preflight,
    validate_design,
)
from spoofing_detection.lob.withdrawal_risk_v2 import SCHEMAS as RISK_SCHEMAS

CHECKPOINT_SCHEMA_VERSION = "temporal_calibration_checkpoint_v1"
REFERENCE_PARTITION_SCHEMA_VERSION = "temporal_calibration_reference_partition_v2"
INSTRUMENTS = ("NEXI", "FERRARI", "RISANAMENTO")
STAGES = ("preflight", "extract", "estimate", "validate", "report", "endpoints", "association",
          "profiles", "uncertainty", "final", "all")
_ROOT = Path(__file__).resolve().parents[1]
_BEHAVIOR_CLOSURE_FILES = tuple(
    _ROOT / relative for relative in (
        "scripts/calibrate_temporal_parameters.py",
        "src/spoofing_detection/lob/temporal_calibration_inputs.py",
        "src/spoofing_detection/lob/temporal_calibration.py",
        "src/spoofing_detection/lob/actor_identity.py",
        "src/spoofing_detection/lob/config.py",
        "src/spoofing_detection/lob/empirical_clock.py",
        "src/spoofing_detection/lob/execution_clusters.py",
        "src/spoofing_detection/lob/panel.py",
        "src/spoofing_detection/lob/withdrawal_risk_v2.py",
        "src/spoofing_detection/lob/empirical_controls_v2_io.py",
        "src/spoofing_detection/lob/empirical_controls_v2_partition.py",
        "src/spoofing_detection/lob/empirical_controls_v2_budget.py",
    )
)
_REFERENCE_SCHEMAS: dict[str, Mapping[str, pl.DataType]] = {
    "origin_catalog.parquet": ORIGIN_SCHEMA,
    "gap_pairs.parquet": GAP_PAIR_SCHEMA,
    "age_exposure.parquet": AGE_EXPOSURE_SCHEMA,
    "age_spells.parquet": AGE_SPELL_SCHEMA,
    "market_states.parquet": {**MARKET_STATE_SCHEMA, "coverage_epoch_id": pl.String},
}


def _reject_constant(token: str) -> None:
    raise ValueError(f"non-finite JSON constant forbidden: {token}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key forbidden: {key}")
        result[key] = value
    return result


def load_json_object(path: Path) -> dict[str, Any]:
    """Load a JSON object without duplicate keys or non-finite constants."""
    try:
        value = json.loads(path.read_text(), object_pairs_hook=_object_without_duplicates, parse_constant=_reject_constant)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def load_frozen_design(path: Path | str) -> dict[str, Any]:
    design = load_json_object(Path(path))
    validate_design(design)
    if value_hash(design) != value_hash(frozen_design()):
        raise ValueError("design differs from frozen temporal calibration design")
    return design


def behavior_closure_identity() -> dict[str, str]:
    """Hash every source file defining replay/risk/cluster/clock behavior."""
    closure: dict[str, str] = {}
    for path in _BEHAVIOR_CLOSURE_FILES:
        if not path.is_file():
            raise ValueError(f"behavior closure source missing: {path}")
        closure[str(path.relative_to(_ROOT))] = file_hash(path)
    return closure


def _hashes(values: Mapping[str, Any], *, context: str) -> dict[str, str]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{context} hashes required")
    result: dict[str, str] = {}
    for name, digest in values.items():
        if not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(f"invalid {context} hash")
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(f"invalid {context} hash") from exc
        result[name] = digest
    return result


def checkpoint_identity(design: Mapping[str, Any], observed: Mapping[str, Any]) -> dict[str, Any]:
    """Freeze design, actual inputs, and the full executable behavior closure."""
    validate_design(design)
    inputs = observed.get("observed_inputs")
    if not isinstance(inputs, Mapping) or not inputs:
        raise ValueError("preflight observed inputs required")
    input_hashes = _hashes({str(name): item.get("sha256") if isinstance(item, Mapping) else None
                            for name, item in inputs.items()}, context="input")
    source = observed.get("source_identity")
    if not isinstance(source, Mapping):
        raise ValueError("preflight source identity required")
    return {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "design_sha256": value_hash(dict(design)),
        "input_hashes": input_hashes,
        "preflight_source_identity": _hashes(source, context="preflight source"),
        "behavior_closure": behavior_closure_identity(),
    }


def extraction_identity(base: Mapping[str, Any], observed: Mapping[str, Any], *, is_pilot: bool) -> dict[str, Any]:
    selected = observed.get("selected_dates")
    if not isinstance(selected, list) or not selected or any(not isinstance(day, str) for day in selected):
        raise ValueError("selected dates required for extraction identity")
    return {**dict(base), "instrument": observed["instrument"], "selected_dates": list(selected), "is_pilot": is_pilot,
            "reference_partition_schema_version": REFERENCE_PARTITION_SCHEMA_VERSION}


def _safe_artifact(root: Path, name: str) -> Path:
    relative = Path(name)
    if not name or relative.is_absolute() or ".." in relative.parts:
        raise ValueError("invalid checkpoint artifact path")
    candidate = (root / relative).resolve()
    if root.resolve() not in candidate.parents or candidate == root.resolve():
        raise ValueError("checkpoint artifact escapes output root")
    return candidate


def _checkpoint_raw_catalog(root: Path, identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Recover the canonical raw catalog pinned by the immutable identity."""
    provenance_path = _safe_artifact(root, "private/provenance.json")
    provenance = load_json_object(provenance_path)
    observed = provenance.get("observed")
    inputs = observed.get("observed_inputs") if isinstance(observed, Mapping) else None
    raw = inputs.get("raw_events") if isinstance(inputs, Mapping) else None
    raw_name = raw.get("path") if isinstance(raw, Mapping) else None
    input_hashes = identity.get("input_hashes")
    expected_hash = input_hashes.get("raw_events") if isinstance(input_hashes, Mapping) else None
    if not isinstance(raw_name, str) or not isinstance(expected_hash, str):
        raise ValueError("checkpoint raw catalog provenance required")
    raw_path = Path(raw_name)
    if not raw_path.is_file() or file_hash(raw_path) != expected_hash:
        raise ValueError("checkpoint raw catalog source mismatch")
    return _catalog(raw_path)


def _expected_partition_graph(root: Path, identity: Mapping[str, Any]) -> list[dict[str, Any]]:
    selected = identity.get("selected_dates")
    instrument = identity.get("instrument")
    if (not isinstance(selected, list) or not selected or len(set(selected)) != len(selected)
            or any(not isinstance(day, str) or date.fromisoformat(day).isoformat() != day for day in selected)
            or not isinstance(instrument, str)):
        raise ValueError("checkpoint selected partition identity invalid")
    selected_set = set(selected)
    catalog = _checkpoint_raw_catalog(root, identity)
    chosen = [partition for partition in catalog if partition["event_date"] in selected_set]
    if not chosen or {partition["event_date"] for partition in chosen} != selected_set:
        raise ValueError("checkpoint selected dates absent from canonical raw catalog")
    return [{"partition_id": partition["partition_id"], "event_date": partition["event_date"],
             "path": f"private/partitions/{_safe_partition_name(instrument, str(partition['partition_id']))}",
             "identity": {"extraction_identity": dict(identity), "partition": dict(partition)}}
            for partition in chosen]


def verify_complete_checkpoint(output_root: Path | str, expected_identity: Mapping[str, Any], *,
                               expected_design: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Fail closed unless a complete root manifest and every artifact match."""
    root = Path(output_root).resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("complete checkpoint manifest required")
    try:
        manifest = load_json_object(manifest_path)
    except ValueError as exc:
        raise ValueError("invalid checkpoint manifest") from exc
    required = {"schema_version", "complete", "identity", "artifacts"}
    allowed = required | {"stage", "partitions", "not_run"}
    if not required <= set(manifest) or not set(manifest) <= allowed:
        raise ValueError("incompatible checkpoint manifest")
    if manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION or manifest["complete"] is not True:
        raise ValueError("complete checkpoint required")
    if manifest["identity"] != dict(expected_identity):
        raise ValueError("checkpoint identity mismatch")
    if "reference_partition_schema_version" in expected_identity and manifest.get("stage") != "extract":
        raise ValueError("reference extraction checkpoint stage required")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict):
        raise ValueError("checkpoint artifacts required")
    for name, digest in _hashes(artifacts, context="artifact").items():
        artifact = _safe_artifact(root, name)
        if not artifact.is_file() or file_hash(artifact) != digest:
            raise ValueError("checkpoint artifact hash mismatch")
    if manifest.get("stage") == "extract":
        frozen = frozen_design() if expected_design is None else dict(expected_design)
        validate_design(frozen)
        config_path = _safe_artifact(root, "config.json")
        config = load_json_object(config_path)
        if set(config) != {"design", "identity"} or config["design"] != frozen or config["identity"] != dict(expected_identity):
            raise ValueError("checkpoint config design/identity mismatch")
        expected_partitions = _expected_partition_graph(root, expected_identity)
        partitions = manifest.get("partitions")
        if not isinstance(partitions, list) or partitions != expected_partitions:
            raise ValueError("checkpoint partition graph is not canonical")
        for item in partitions:
            path = item["path"]
            if not isinstance(path, str) or _safe_artifact(root, path).parent.parent != (root / "private").resolve():
                raise ValueError("checkpoint partition graph path escapes canonical root")
            _verify_reference_partition(_safe_artifact(root, path), item["identity"])
    else:
        for item in manifest.get("partitions", []):
            if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                raise ValueError("checkpoint partition reference invalid")
            _verify_reference_partition(_safe_artifact(root, item["path"]), item["identity"])
    return manifest


def require_resume_checkpoint(output_root: Path | str, expected_identity: Mapping[str, Any], *,
                              expected_design: Mapping[str, Any] | None = None) -> dict[str, Any]:
    return verify_complete_checkpoint(output_root, expected_identity, expected_design=expected_design)


def _verify_incomplete_root(output_root: Path, design: Mapping[str, Any], identity: Mapping[str, Any]) -> None:
    """Verify the immutable config lock before continuing an unpublished root."""
    if not output_root.is_dir() or (output_root / "manifest.json").exists():
        raise ValueError("incomplete extraction root without complete manifest required")
    config_path = output_root / "config.json"
    if not config_path.is_file():
        raise ValueError("incomplete extraction config lock required")
    try:
        config = load_json_object(config_path)
    except ValueError as exc:
        raise ValueError("invalid incomplete extraction config lock") from exc
    if set(config) != {"design", "identity"} or config["design"] != dict(design) or config["identity"] != dict(identity):
        raise ValueError("incomplete extraction config identity mismatch")


def _verify_live_sources(identity: Mapping[str, Any], raw_path: Path, *, design: Mapping[str, Any], instrument: str,
                         output: Path, project_root: Path, dates: list[str]) -> None:
    """Refuse publication if the raw source or executable closure changes in flight."""
    input_hashes = identity.get("input_hashes")
    expected_raw = input_hashes.get("raw_events") if isinstance(input_hashes, Mapping) else None
    if not isinstance(expected_raw, str) or file_hash(raw_path) != expected_raw:
        raise ValueError("raw source changed during extraction")
    if identity.get("behavior_closure") != behavior_closure_identity():
        raise ValueError("behavior closure changed during extraction")
    refreshed = preflight(design, instrument, output, dates=dates, allow_existing=True, project_root=project_root)
    refreshed_identity = extraction_identity(checkpoint_identity(design, refreshed), refreshed,
                                             is_pilot=bool(identity.get("is_pilot")))
    if refreshed_identity != dict(identity):
        raise ValueError("source identity changed during extraction")


def _iso_date(value: str) -> str:
    try:
        parsed = date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("dates must be strict ISO YYYY-MM-DD") from exc
    if parsed.isoformat() != value:
        raise argparse.ArgumentTypeError("dates must be strict ISO YYYY-MM-DD")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a frozen temporal calibration stage.", allow_abbrev=False)
    parser.add_argument("--design", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGES, required=True)
    parser.add_argument("--instrument", choices=INSTRUMENTS, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dates", nargs="+", type=_iso_date)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _project_root(design_path: Path) -> Path:
    if design_path.parent.name != "configs":
        raise ValueError("design must reside in a project configs directory")
    return design_path.parent.parent.resolve()


def _json_safe_preflight(observed: Mapping[str, Any], identity: Mapping[str, Any], *, resumed: bool,
                         is_pilot: bool) -> dict[str, Any]:
    return {"status": "preflight_ok", "instrument": observed["instrument"], "selected_dates": observed["selected_dates"],
            "design": observed["design"], "observed_inputs": observed["observed_inputs"],
            "source_identity": observed["source_identity"], "checkpoint_identity": dict(identity),
            "is_pilot": is_pilot, "resume_verified": resumed}


def _catalog(raw_path: Path) -> list[dict[str, Any]]:
    scan = pl.scan_parquet(raw_path)
    schema = scan.collect_schema()
    if not set(SORT_COLUMNS[:5]) <= set(schema.names()):
        raise ValueError("raw partition columns missing")
    grouped = scan.group_by(SORT_COLUMNS[:5]).agg(pl.len().alias("rows")).sort(SORT_COLUMNS[:5], nulls_last=True).collect()
    result: list[dict[str, Any]] = []
    offset = 0
    for row in grouped.iter_rows(named=True):
        raw_day = row["TRADEDATE"]
        if isinstance(raw_day, datetime):
            if raw_day.time() != datetime.min.time():
                raise ValueError("Datetime TRADEDATE must be normalized to midnight")
            event_day = raw_day.date().isoformat()
        elif isinstance(raw_day, date):
            event_day = raw_day.isoformat()
        else:
            event_day = str(raw_day)
        if date.fromisoformat(event_day).isoformat() != event_day:
            raise ValueError("non-ISO raw partition date")
        count = row["rows"]
        if type(count) is not int or count <= 0:
            raise ValueError("nonempty partition row count required")
        # Keep native partition IDs, but serialize dates explicitly for worker jobs.
        serialized_row = {**row, "TRADEDATE": raw_day.isoformat() if isinstance(raw_day, (datetime, date)) else raw_day}
        result.append({**serialized_row, "partition_id": _partition_id(row), "event_date": event_day, "index_offset": offset})
        offset += count
    return result


def _load_partition(raw_path: Path, partition: Mapping[str, Any]) -> pl.DataFrame:
    scan = pl.scan_parquet(raw_path)
    schema = scan.collect_schema()
    predicate = None
    for column in SORT_COLUMNS[:5]:
        value, dtype = partition[column], schema[column]
        if isinstance(dtype, pl.Datetime) and isinstance(value, str):
            expression = pl.col(column) == pl.lit(datetime.fromisoformat(value)).cast(dtype)
        elif dtype == pl.Date and isinstance(value, str):
            expression = pl.col(column) == pl.lit(date.fromisoformat(value)).cast(dtype)
        else:
            expression = pl.col(column).cast(pl.String) == str(value)
        predicate = expression if predicate is None else predicate & expression
    frame = scan.filter(predicate).collect(engine="streaming")
    if frame.height != partition["rows"]:
        raise ValueError("partition loader did not recover the complete canonical partition")
    return sort_events(frame)


def _safe_partition_name(instrument: str, partition_id: str) -> str:
    return f"{instrument}-{value_hash({'instrument': instrument, 'partition_id': partition_id})[:20]}"


def _write_parquet_atomic(frame: pl.DataFrame, path: Path, schema: Mapping[str, pl.DataType]) -> None:
    selected = frame.select([pl.col(name).cast(dtype, strict=True) for name, dtype in schema.items()])
    temporary = path.with_name(f".{path.name}.tmp")
    selected.write_parquet(temporary)
    os.replace(temporary, path)


def _reference_partition_identity(job: Mapping[str, Any]) -> dict[str, Any]:
    return {"extraction_identity": job["extraction_identity"], "partition": job["partition"]}


def _verify_reference_partition(path: Path, identity: Mapping[str, Any]) -> dict[str, Any]:
    manifest_path = path / "reference_manifest.json"
    if not manifest_path.is_file():
        raise ValueError("complete reference partition manifest required")
    manifest = load_json_object(manifest_path)
    if set(manifest) != {"schema_version", "complete", "identity", "artifacts", "risk_cache_key", "risk_provenance"}:
        raise ValueError("incompatible reference partition manifest")
    if manifest["schema_version"] != REFERENCE_PARTITION_SCHEMA_VERSION or manifest["complete"] is not True:
        raise ValueError("complete reference partition required")
    if manifest["identity"] != dict(identity):
        raise ValueError("reference partition identity mismatch")
    artifacts = _hashes(manifest["artifacts"], context="reference artifact")
    if set(artifacts) != set(_REFERENCE_SCHEMAS) | {"risk/manifest.json", "reference_diagnostics.json"}:
        raise ValueError("reference partition artifact graph mismatch")
    for name, schema in _REFERENCE_SCHEMAS.items():
        target = _safe_artifact(path, name)
        if not target.is_file() or file_hash(target) != artifacts[name]:
            raise ValueError("reference partition artifact hash mismatch")
        if list(pl.read_parquet_schema(target).items()) != list(schema.items()):
            raise ValueError("reference partition schema mismatch")
    diagnostics_path = _safe_artifact(path, "reference_diagnostics.json")
    if not diagnostics_path.is_file() or file_hash(diagnostics_path) != artifacts["reference_diagnostics.json"]:
        raise ValueError("reference partition diagnostics hash mismatch")
    diagnostics = load_json_object(diagnostics_path)
    required_diagnostics = {"age_cap_seconds", "partition_span_seconds", "age", "gaps"}
    if set(diagnostics) != required_diagnostics or not isinstance(diagnostics["age"], dict) or not isinstance(diagnostics["gaps"], dict):
        raise ValueError("reference partition diagnostics incompatible")
    if (type(diagnostics["age_cap_seconds"]) not in (int, float)
            or type(diagnostics["partition_span_seconds"]) not in (int, float)
            or diagnostics["age_cap_seconds"] < diagnostics["partition_span_seconds"]):
        raise ValueError("reference partition cap/span diagnostics incompatible")
    risk = path / "risk"
    if file_hash(risk / "manifest.json") != artifacts["risk/manifest.json"]:
        raise ValueError("reference partition risk manifest hash mismatch")
    scan_checkpoint(risk, cache_key=manifest["risk_cache_key"], schemas=RISK_SCHEMAS,
                    provenance=manifest["risk_provenance"])
    return manifest


def _run_partition_worker(job_path: Path) -> None:
    job = load_json_object(job_path)
    path = Path(job["partition_path"])
    identity = _reference_partition_identity(job)
    if path.exists():
        _verify_reference_partition(path, identity)
        return
    parent = path.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{path.name}.incomplete-", dir=parent))
    try:
        partition = job["partition"]
        raw = _load_partition(Path(job["raw_path"]), partition)
        original = _OriginalIndices(int(partition["rows"]), int(partition["index_offset"]))
        risk_key = value_hash({"reference_partition": identity, "risk": "uncapped_full_partition_span",
                               "clock_policy_id": CLOCK_POLICY_ID})
        provenance = {"clock_policy_id": CLOCK_POLICY_ID}
        with PartitionWriter(temporary / "risk", cache_key=risk_key, schemas=RISK_SCHEMAS,
                             buffer_rows=int(job["buffer_rows"]), provenance=provenance) as writer:
            _, origins, cap = collect_uncapped_risk(raw, instrument=str(job["instrument"]),
                                                    original_sort_indices=original, sink=writer.append)
            writer.complete()
        risk_tables = scan_checkpoint(temporary / "risk", cache_key=risk_key, schemas=RISK_SCHEMAS, provenance=provenance)
        membership = risk_tables["membership"].collect()
        coverage = risk_tables["coverage_epochs"].collect()
        ages, spells, age_diagnostics = build_age_inputs(membership, origins)
        gaps, gap_diagnostics = collect_gap_pairs(raw, instrument=str(job["instrument"]), original_sort_indices=original)
        states = collect_market_states(raw, instrument=str(job["instrument"]), original_sort_indices=original)
        states = annotate_coverage_epochs(states, coverage)
        _write_parquet_atomic(origins, temporary / "origin_catalog.parquet", ORIGIN_SCHEMA)
        _write_parquet_atomic(gaps, temporary / "gap_pairs.parquet", GAP_PAIR_SCHEMA)
        _write_parquet_atomic(ages, temporary / "age_exposure.parquet", AGE_EXPOSURE_SCHEMA)
        _write_parquet_atomic(spells, temporary / "age_spells.parquet", AGE_SPELL_SCHEMA)
        _write_parquet_atomic(states, temporary / "market_states.parquet", _REFERENCE_SCHEMAS["market_states.parquet"])
        _write_json(temporary / "reference_diagnostics.json", {
            "age_cap_seconds": cap,
            "partition_span_seconds": _partition_span_seconds(raw),
            "age": age_diagnostics,
            "gaps": gap_diagnostics,
        })
        artifacts = {name: file_hash(temporary / name) for name in _REFERENCE_SCHEMAS}
        artifacts["risk/manifest.json"] = file_hash(temporary / "risk" / "manifest.json")
        artifacts["reference_diagnostics.json"] = file_hash(temporary / "reference_diagnostics.json")
        manifest = {"schema_version": REFERENCE_PARTITION_SCHEMA_VERSION, "complete": True, "identity": identity,
                    "artifacts": artifacts, "risk_cache_key": risk_key, "risk_provenance": provenance}
        _write_json(temporary / "reference_manifest.json", manifest)
        _verify_reference_partition(temporary, identity)
        os.rename(temporary, path)
    except Exception:
        # Preserve this never-published diagnostic directory; do not append to it on resume.
        raise


def _worker_environment(project_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    # The frozen design may live in a synthetic/project fixture, but the worker
    # must import the implementation that launched it rather than assuming that
    # project root contains a source checkout.
    additions = [str(_ROOT / "src"), str(_ROOT), str(project_root)]
    environment["PYTHONPATH"] = os.pathsep.join(additions + ([environment["PYTHONPATH"]] if environment.get("PYTHONPATH") else []))
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _all_files(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.name != "manifest.json" and ".incomplete-" not in path.as_posix():
            result[path.relative_to(root).as_posix()] = file_hash(path)
    return result


def _write_root_tables(root: Path, catalog: list[dict[str, Any]], selected: list[dict[str, Any]], instrument: str) -> None:
    catalog_frame = pl.DataFrame(catalog).select([*SORT_COLUMNS[:5], "partition_id", "event_date", "index_offset", "rows"])
    catalog_frame.write_csv(root / "partition_catalog.csv")
    coverage_frames: list[pl.DataFrame] = []
    support_rows: list[dict[str, Any]] = []
    for item in selected:
        base = root / item["path"]
        reference = _verify_reference_partition(base, item["identity"])
        risk = scan_checkpoint(base / "risk", cache_key=reference["risk_cache_key"], schemas=RISK_SCHEMAS,
                               provenance=reference["risk_provenance"])
        coverage_frames.append(risk["coverage_epochs"].collect())
        tables = {name: pl.read_parquet(base / name) for name in _REFERENCE_SCHEMAS}
        for level in ("client_original", "firm"):
            support_rows.append({"instrument": instrument, "partition_id": item["partition_id"], "event_date": item["event_date"],
                                 "identity_level": level,
                                 "origin_rows": tables["origin_catalog.parquet"].filter(pl.col("identity_level") == level).height,
                                 "gap_pair_rows": tables["gap_pairs.parquet"].filter(pl.col("identity_level") == level).height,
                                 "age_exposure_rows": tables["age_exposure.parquet"].filter(pl.col("identity_level") == level).height,
                                 "market_state_rows": tables["market_states.parquet"].filter(pl.col("identity_level") == level).height})
    coverage = pl.concat(coverage_frames, how="vertical") if coverage_frames else pl.DataFrame(schema=RISK_SCHEMAS["coverage_epochs"])
    coverage.write_csv(root / "reference_coverage.csv")
    pl.DataFrame(support_rows).write_csv(root / "identity_support.csv")


def _extract(design: Mapping[str, Any], observed: Mapping[str, Any], identity: Mapping[str, Any], *, output: Path,
             project_root: Path, resume: bool, command: list[str]) -> dict[str, Any]:
    if resume and (output / "manifest.json").is_file():
        manifest = require_resume_checkpoint(output, identity, expected_design=design)
        return {"status": "extract_complete", "instrument": observed["instrument"], "selected_dates": observed["selected_dates"],
                "is_pilot": identity["is_pilot"], "resume_verified": True, "partition_count": len(manifest.get("partitions", []))}
    if output.exists():
        if not resume:
            raise FileExistsError(f"output root already exists: {output}")
        _verify_incomplete_root(output, design, identity)
    elif resume:
        raise ValueError("incomplete extraction root without complete manifest required")
    raw_path = Path(observed["observed_inputs"]["raw_events"]["path"])
    source_guard = {"design": design, "instrument": str(observed["instrument"]), "output": output,
                    "project_root": project_root, "dates": list(observed["selected_dates"])}
    _verify_live_sources(identity, raw_path, **source_guard)
    catalog = _catalog(raw_path)
    selected_days = set(observed["selected_dates"])
    chosen = [row for row in catalog if row["event_date"] in selected_days]
    if not chosen:
        raise ValueError("no complete raw partitions selected")
    if not output.exists():
        output.mkdir(parents=True)
        (output / "private").mkdir()
        _write_json(output / "config.json", {"design": dict(design), "identity": dict(identity)})
        _write_json(output / "versions.json", {"python": platform.python_version(), "polars": pl.__version__})
        _write_json(output / "command.json", {"stage": "extract", "instrument": observed["instrument"], "dates": observed["selected_dates"],
                                                "resume": False, "argv": command})
        _write_json(output / "private" / "provenance.json", {"observed": dict(observed)})
    try:
        partitions: list[dict[str, Any]] = []
        resources = design["resources"]
        for partition in chosen:
            name = _safe_partition_name(str(observed["instrument"]), str(partition["partition_id"]))
            relative = f"private/partitions/{name}"
            item_identity = {"extraction_identity": dict(identity), "partition": dict(partition)}
            target = output / relative
            if target.exists():
                _verify_reference_partition(target, item_identity)
            else:
                job = {"partition_path": str(target), "raw_path": str(raw_path), "partition": partition,
                       "instrument": observed["instrument"], "extraction_identity": dict(identity),
                       "buffer_rows": resources["buffer_rows"]}
                job_path = output / "private" / f"{name}.job.json"
                _write_json(job_path, job)
                try:
                    _verify_live_sources(identity, raw_path, **source_guard)
                    supervise([sys.executable, str(Path(__file__).resolve()), "--partition-job", str(job_path)],
                              max_rss_mb=resources["max_rss_mb"], timeout_seconds=resources["timeout_seconds"],
                              cwd=project_root, env=_worker_environment(project_root),
                              diagnostics_path=output / "private" / "supervision" / f"{name}.json")
                    _verify_live_sources(identity, raw_path, **source_guard)
                finally:
                    job_path.unlink(missing_ok=True)
                _verify_reference_partition(target, item_identity)
            partitions.append({"partition_id": partition["partition_id"], "event_date": partition["event_date"],
                               "path": relative, "identity": item_identity})
        _verify_live_sources(identity, raw_path, **source_guard)
        _write_root_tables(output, catalog, partitions, str(observed["instrument"]))
        _verify_live_sources(identity, raw_path, **source_guard)
        not_run = {"broad_metrics": "next milestone; no synthetic placeholder persisted",
                   "controls": "next milestone; no estimates or parameter proposals persisted",
                   "estimate": "not run", "validate": "not run", "report": "not run"}
        artifacts = _all_files(output)
        manifest = {"schema_version": CHECKPOINT_SCHEMA_VERSION, "complete": True, "identity": dict(identity),
                    "artifacts": artifacts, "stage": "extract", "partitions": partitions, "not_run": not_run}
        _verify_live_sources(identity, raw_path, **source_guard)
        _write_json(output / "manifest.json", manifest)
        verify_complete_checkpoint(output, identity, expected_design=design)
        return {"status": "extract_complete", "instrument": observed["instrument"], "selected_dates": observed["selected_dates"],
                "is_pilot": identity["is_pilot"], "resume_verified": resume, "partition_count": len(partitions)}
    except Exception:
        # Root has no complete manifest and is deliberately retained for diagnosis only.
        raise


def main(argv: list[str] | None = None) -> None:
    actual = list(sys.argv[1:] if argv is None else argv)
    if len(actual) == 2 and actual[0] == "--partition-job":
        _run_partition_worker(Path(actual[1]))
        return
    args = parse_args(actual)
    design_path = args.design.resolve()
    design = load_frozen_design(design_path)
    if args.dates is not None and len(set(args.dates)) != len(args.dates):
        raise ValueError("dates must be unique")
    observed = preflight(design, args.instrument, args.output_root, dates=args.dates,
                         allow_existing=args.resume or args.stage not in {"preflight", "extract"},
                         project_root=_project_root(design_path))
    base_identity = checkpoint_identity(design, observed)
    is_pilot = args.dates is not None
    identity = extraction_identity(base_identity, observed, is_pilot=is_pilot)
    if args.stage == "preflight":
        resumed = False
        if args.resume:
            require_resume_checkpoint(args.output_root, identity, expected_design=design)
            resumed = True
        print(json.dumps(_json_safe_preflight(observed, identity, resumed=resumed, is_pilot=is_pilot), sort_keys=True, allow_nan=False))
        return
    if args.stage == "extract":
        print(json.dumps(_extract(design, observed, identity, output=args.output_root.resolve(), project_root=_project_root(design_path),
                                  resume=args.resume, command=actual), sort_keys=True, allow_nan=False))
        return
    if args.stage == "all":
        _extract(design, observed, identity, output=args.output_root.resolve(),
                 project_root=_project_root(design_path), resume=args.resume, command=actual)
    from spoofing_detection.lob.temporal_calibration_report import run_estimate, run_validate, run_report
    stored = load_json_object(args.output_root.resolve() / "manifest.json") if (args.output_root.resolve() / "manifest.json").is_file() else None
    if stored is None:
        raise ValueError("complete extraction checkpoint required")
    stored_identity = stored.get("identity")
    if args.dates is None and isinstance(stored_identity, Mapping) and stored_identity.get("is_pilot") is True:
        raise ValueError("explicit --dates required for a pilot extraction checkpoint")
    verify_complete_checkpoint(args.output_root, identity, expected_design=design)
    from spoofing_detection.lob.temporal_calibration_endpoints import run_endpoint_diagnostics
    from spoofing_detection.lob.temporal_calibration_w_stage import run_w_association
    from spoofing_detection.lob.temporal_calibration_profiles import run_profile_diagnostics
    from spoofing_detection.lob.temporal_calibration_uncertainty import run_uncertainty_diagnostics
    from spoofing_detection.lob.temporal_calibration_final_report import run_final_report
    functions = {"estimate": run_estimate, "validate": run_validate, "report": run_report,
                 "endpoints": run_endpoint_diagnostics, "association": run_w_association,
                 "profiles": run_profile_diagnostics, "uncertainty": run_uncertainty_diagnostics,
                 "final": run_final_report}
    if args.stage == "all":
        results = {stage: functions[stage](args.output_root.resolve(), design, args.instrument, dates=args.dates)
                   for stage in ("estimate", "validate", "association", "endpoints", "profiles", "uncertainty", "report", "final")}
        result = {"status": "implementation_pipeline_complete", "instrument": args.instrument,
                  "is_pilot": is_pilot, "stages": results}
    else:
        result = functions[args.stage](args.output_root.resolve(), design, args.instrument, dates=args.dates)
    print(json.dumps(result, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
