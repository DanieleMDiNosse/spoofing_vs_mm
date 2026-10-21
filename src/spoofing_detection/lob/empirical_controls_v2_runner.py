"""Sequential, resource-supervised construction of empirical-controls v2 bundles.

This module consumes a separately certified baseline. It never certifies or
rescales detector outputs and never promotes results into the paper/dashboard.
"""
from __future__ import annotations

import argparse
from datetime import date, datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile
from typing import Any, Mapping

import polars as pl

from . import empirical_controls_v2 as accounting
from .empirical_controls_v2_analysis import SCHEMAS as ANALYSIS_SCHEMAS, analyze_partition, bound_cache_key
from .empirical_controls_v2_budget import supervise
from .empirical_controls_v2_io import (
    VERSION, file_hash, risk_cache_key, risk_producer_identity, scan_checkpoint,
    validate_config, validate_preflight_v2, value_hash,
)
from .empirical_controls_v2_partition import cache_partition
from .empirical_clock import CLOCK_POLICY_ID, resolve_empirical_clocks
from .empirical_controls_v2_report import (
    BUNDLE_SCHEMA, build_report, summarize_contributions, validate_bundle,
)
from .panel import SORT_COLUMNS, sort_events
from .withdrawal_risk_v2 import SCHEMAS as RISK_SCHEMAS

CLUSTER_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.String, "partition_id": pl.String, "event_date": pl.Date,
    "actor_key": pl.String, "actor_id": pl.String, "identity_level": pl.String,
    "identity_source": pl.String, "identity_fallback_flag": pl.Boolean,
    "event_side": pl.String, "execution_anchor_mode": pl.String,
    "cluster_end_ts": pl.Datetime("us"), "cluster_last_sort_index": pl.Int64,
    "execution_cluster_id": pl.String, "execution_quantity": pl.Float64,
    "clock_policy_id": pl.String, "fill_anchor_ns": pl.Int64,
}
ROOT_ANALYSIS = {
    "state_statistics": accounting.STATE_SCHEMA,
    "contrast_statistics": accounting.CONTRAST_SCHEMA,
    "coverage": accounting.COVERAGE_SCHEMA,
    "balance": accounting.BALANCE_SCHEMA,
    "shift_support": accounting.SHIFT_SUPPORT_SCHEMA,
    "summary_contributions": accounting.SUMMARY_SCHEMA,
}


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    source_root = Path(__file__).resolve().parents[2]
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(source_root), environment["PYTHONPATH"]]
        if environment.get("PYTHONPATH") else [str(source_root)]
    )
    return environment


def _preflight_worker(config_path: Path, result_path: Path) -> None:
    validated = validate_preflight_v2(config_path)
    result_path.write_text(json.dumps(validated, default=str, sort_keys=True))


def validate_preflight_supervised(config_path: Path) -> dict[str, Any]:
    """Validate all certified inputs under the declared whole-process budget."""
    config = _json(config_path)
    validate_config(config)
    output = config.get("output_root")
    if isinstance(output, str) and Path(output).resolve().exists():
        raise FileExistsError(f"output root already exists: {Path(output).resolve()}")
    resources = config["resources"]
    with tempfile.TemporaryDirectory(prefix="empirical-controls-v2-preflight-") as directory:
        result_path = Path(directory) / "validated.json"
        supervise(
            [sys.executable, "-m", "spoofing_detection.lob.empirical_controls_v2_runner",
             "--preflight-job", str(config_path), str(result_path)],
            max_rss_mb=resources["max_rss_mb"], timeout_seconds=resources["timeout_seconds"],
            cwd=Path(__file__).resolve().parents[3], env=_worker_environment(),
        )
        validated = _json(result_path)
    validated["output_root"] = Path(validated["output_root"])
    for run in validated["runs"].values():
        run["sources"] = {name: Path(path) for name, path in run["sources"].items()}
        run["certificate_path"] = Path(run["certificate_path"])
    return validated


def load_partition(raw_path: Path, partition: Mapping[str, Any]) -> pl.DataFrame:
    """Read one complete canonical partition with predicate pushdown."""
    scan = pl.scan_parquet(raw_path)
    missing = set(SORT_COLUMNS[:5]) - set(scan.collect_schema().names())
    if missing:
        raise ValueError(f"raw partition columns missing: {sorted(missing)}")
    predicate = None
    for column in SORT_COLUMNS[:5]:
        value = partition.get(column)
        dtype = scan.collect_schema()[column]
        if isinstance(dtype, pl.Datetime) and isinstance(value, str):
            parsed = datetime.fromisoformat(value)
            expression = pl.col(column) == pl.lit(parsed).cast(dtype)
        elif dtype == pl.Date and isinstance(value, str):
            expression = pl.col(column) == pl.lit(date.fromisoformat(value[:10])).cast(dtype)
        else:
            expression = pl.col(column).cast(pl.String) == str(value)
        predicate = expression if predicate is None else predicate & expression
    frame = scan.filter(predicate).collect(engine="streaming")
    if frame.height != partition.get("rows"):
        raise ValueError("partition loader did not recover the certified complete row count")
    return frame


def attach_fill_clock_anchors(
    clusters: pl.DataFrame,
    raw_events: pl.DataFrame,
    *,
    index_offset: int,
    clock_policy_id: str | None,
) -> pl.DataFrame:
    """Bind each cluster's final child fill to its semantic TRADETIME anchor."""
    if clock_policy_id is None:
        return clusters.with_columns(
            pl.lit(None, dtype=pl.String).alias("clock_policy_id"),
            pl.lit(None, dtype=pl.Int64).alias("fill_anchor_ns"),
        )
    if clock_policy_id != CLOCK_POLICY_ID:
        raise ValueError(f"unknown clock policy: {clock_policy_id}")
    target_indices = set(clusters["cluster_last_sort_index"].to_list())
    anchors = []
    native_clock_columns = [
        name for name in ("SEQUENCETIME", "BOOKIN", "BOOKOUTTIME", "TRADETIME")
        if name in raw_events.columns and raw_events.schema[name] == pl.Datetime("ns")
    ]
    canonical = sort_events(raw_events.with_columns(
        pl.col(name).cast(pl.Int64).alias(name) for name in native_clock_columns
    )) if native_clock_columns else sort_events(raw_events)
    for local_index, event in enumerate(canonical.iter_rows(named=True), start=1):
        global_index = index_offset + local_index
        if global_index not in target_indices:
            continue
        decision = resolve_empirical_clocks(event)
        if decision.fill_anchor_ns is not None:
            anchors.append({
                "cluster_last_sort_index": global_index,
                "fill_anchor_ns": decision.fill_anchor_ns,
            })
    anchor_frame = pl.DataFrame(
        anchors,
        schema={"cluster_last_sort_index": pl.Int64, "fill_anchor_ns": pl.Int64},
    )
    result = clusters.join(anchor_frame, on="cluster_last_sort_index", how="left", validate="m:1")
    if result["fill_anchor_ns"].null_count():
        raise ValueError("execution cluster lacks exact fill TRADETIME anchor")
    if "cluster_end_fill_anchor_ns" in result.columns:
        mismatched = result.filter(
            pl.col("cluster_end_fill_anchor_ns").is_null()
            | (pl.col("cluster_end_fill_anchor_ns") != pl.col("fill_anchor_ns"))
        )
        if mismatched.height:
            raise ValueError("detector fill anchor mismatch with canonical raw Fill")
        result = result.drop("cluster_end_fill_anchor_ns")
    return result.with_columns(pl.lit(clock_policy_id).alias("clock_policy_id"))


def _write_partition_clusters(
    source: Path, partition: Mapping[str, Any], instrument: str, destination: Path,
    *, raw_events: pl.DataFrame, clock_policy_id: str | None,
) -> None:
    scan = pl.scan_parquet(source).filter(pl.col("partition_id") == partition["partition_id"])
    schema = set(scan.collect_schema().names())
    required_without_added = set(CLUSTER_SCHEMA) - {
        "instrument", "event_date", "clock_policy_id", "fill_anchor_ns",
    }
    if not required_without_added <= schema:
        raise ValueError("certified execution cluster table lacks required v2 columns")
    expressions = []
    for name, dtype in CLUSTER_SCHEMA.items():
        if name == "instrument":
            expressions.append(pl.lit(instrument).cast(dtype).alias(name))
        elif name == "event_date":
            expressions.append(pl.lit(date.fromisoformat(partition["event_date"])).cast(dtype).alias(name))
        else:
            expressions.append(pl.col(name).cast(dtype, strict=True))
    if clock_policy_id == CLOCK_POLICY_ID and "cluster_end_fill_anchor_ns" in schema:
        selected = scan.select([*expressions[:-2], pl.col("cluster_end_fill_anchor_ns")]).collect(engine="streaming")
    elif clock_policy_id == CLOCK_POLICY_ID:
        raise ValueError("certified v3 detector clusters lack final Fill anchor")
    else:
        selected = scan.select(expressions[:-2]).collect(engine="streaming")
    anchored = attach_fill_clock_anchors(
        selected,
        raw_events,
        index_offset=int(partition["index_offset"]),
        clock_policy_id=clock_policy_id,
    )
    destination.unlink(missing_ok=True)
    anchored.select(expressions).write_parquet(destination)


def _safe_partition_name(instrument: str, partition_id: str) -> str:
    digest = hashlib.sha256(f"{instrument}\0{partition_id}".encode()).hexdigest()[:20]
    return f"{instrument.replace('/', '_')}-{digest}"


def _worker(job_path: Path) -> None:
    job = _json(job_path)
    partition = job["partition"]
    raw = load_partition(Path(job["raw_path"]), partition)
    _write_partition_clusters(
        Path(job["cluster_source"]), partition, job["instrument"], Path(job["cluster_path"]),
        raw_events=raw, clock_policy_id=job.get("clock_policy_id"),
    )
    clusters = pl.read_parquet(job["cluster_path"])
    cluster_identity = file_hash(Path(job["cluster_path"]))
    cache_partition(
        raw, path=Path(job["risk_path"]), partition=partition, instrument=job["instrument"],
        cache_key=job["risk_cache_key"], buffer_rows=job["buffer_rows"], resume=True,
        clock_policy_id=job.get("clock_policy_id"),
    )
    analyze_partition(
        Path(job["risk_path"]), risk_cache_key=job["risk_cache_key"], clusters=clusters,
        output_path=Path(job["analysis_path"]), analysis_cache_key=job["analysis_cache_label"],
        buffer_rows=job["buffer_rows"], resume=True, cluster_identity=cluster_identity,
        clock_policy_id=job.get("clock_policy_id"),
    )


def _write_lazy(frame: pl.LazyFrame | None, path: Path, schema: Mapping[str, pl.DataType]) -> None:
    path.unlink(missing_ok=True)
    if frame is None:
        pl.DataFrame(schema=schema).write_parquet(path)
    else:
        frame.select([pl.col(name).cast(dtype, strict=True) for name, dtype in schema.items()]).sink_parquet(path)


def _units(relative: str) -> dict[str, str]:
    if "state_statistics" in relative:
        return {"time_at_risk_seconds": "posture-seconds", "withdrawal_count": "physical cancellations", "withdrawal_intensity": "cancellations per posture-second"}
    if "contrast_statistics" in relative or "summary" in relative or "shift_support" in relative:
        return {"weight_seconds": "posture-seconds", "intensity_difference": "cancellations per posture-second"}
    if "coverage" in relative or "balance" in relative:
        return {"duration_fields": "seconds", "covariates": "source units documented by field"}
    return {"content": "schema-defined values or provenance text"}


def _artifact_graph(
    root: Path, partitions: list[dict[str, Any]], sources: Mapping[str, Mapping[str, Any]],
) -> dict[str, dict[str, Any]]:
    result = {}
    root_names = {
        *(f"{name}.parquet" for name in ROOT_ANALYSIS), "summary.parquet", "summary.csv",
        "coverage.csv", "balance.csv", "shift_support.csv", "timing_sensitivity.csv",
        "config.json", "versions.txt", "command.txt", "validation.json", "resume_identity.json",
    }
    paths = [root / name for name in root_names]
    for item in sources.values():
        paths.extend([root / item["certificate_path"], root / item["evidence_path"]])
    for item in partitions:
        paths.append(root / item["cluster_path"])
        for directory_name in ("risk_path", "analysis_path"):
            directory = root / item[directory_name]
            paths.extend(path for path in directory.rglob("*") if path.is_file())
    for path in sorted(set(paths)):
        if not path.is_file():
            raise ValueError(f"declared bundle artifact missing: {path}")
        relative = path.relative_to(root).as_posix()
        result[relative] = {"sha256": file_hash(path), "units": _units(relative)}
    return result


def execute_v2(config_path: Path) -> Path:
    """Build and atomically publish one complete v2 bundle from certified inputs."""
    validated = validate_preflight_supervised(config_path)
    config = validated["config"]
    output = Path(validated["output_root"])
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.parent / f".{output.name}.incomplete-v2"
    resume_identity = value_hash({
        "config": file_hash(config_path),
        "sources": {instrument: run["artifact_hashes"] for instrument, run in sorted(validated["runs"].items())},
        "risk_producer": risk_producer_identity(),
    })
    if temporary.exists():
        if not temporary.is_dir() or not (temporary / "resume_identity.json").is_file():
            raise ValueError(f"unrecognized incomplete v2 workspace: {temporary}")
        if _json(temporary / "resume_identity.json").get("identity") != resume_identity:
            raise ValueError("incomplete v2 workspace belongs to different certified inputs/config")
    else:
        temporary.mkdir()
        (temporary / "resume_identity.json").write_text(json.dumps({"identity": resume_identity}))
    partitions: list[dict[str, Any]] = []
    checkpoint_provenance = (
        {"clock_policy_id": config.get("clock_policy_id")}
        if config.get("clock_policy_id") == CLOCK_POLICY_ID else None
    )
    try:
        resources = config["resources"]
        for instrument, run in sorted(validated["runs"].items()):
            selected = set(run["selected_event_dates"])
            for partition in (row for row in run["partitions"] if row["event_date"] in selected):
                name = _safe_partition_name(instrument, partition["partition_id"])
                base = temporary / "partitions" / name
                base.mkdir(parents=True, exist_ok=True)
                cluster_path = base / "execution_clusters.parquet"
                risk_path = base / "risk"
                analysis_path = base / "analysis"
                risk_key = risk_cache_key(config, run["artifact_hashes"], run["producer_source_identity"])
                analysis_label = value_hash({
                    "risk": risk_key, "offsets": config["offset_seconds"], "window": config["window_seconds"],
                    "cluster_source": run["artifact_hashes"]["execution_metrics"],
                    "partition_id": partition["partition_id"], "clock_policy_id": config.get("clock_policy_id"),
                })
                job = {
                    "raw_path": str(run["sources"]["raw_events"]), "partition": {key: (str(value) if key in SORT_COLUMNS[:5] else value) for key, value in partition.items()},
                    "instrument": instrument, "cluster_source": str(run["sources"]["execution_metrics"]),
                    "cluster_path": str(cluster_path),
                    "risk_path": str(risk_path), "analysis_path": str(analysis_path),
                    "risk_cache_key": risk_key, "analysis_cache_label": analysis_label,
                    "buffer_rows": resources["buffer_rows"],
                    "clock_policy_id": config.get("clock_policy_id"),
                }
                job_path = base / "job.json"
                checkpoint_complete = False
                if risk_path.exists() and analysis_path.exists() and cluster_path.is_file():
                    scan_checkpoint(risk_path, cache_key=risk_key, schemas=RISK_SCHEMAS,
                                    provenance=checkpoint_provenance)
                    expected_analysis_key = bound_cache_key(
                        risk_path, cluster_identity=file_hash(cluster_path), requested=analysis_label,
                    )
                    scan_checkpoint(analysis_path, cache_key=expected_analysis_key, schemas=ANALYSIS_SCHEMAS,
                                    provenance=checkpoint_provenance)
                    checkpoint_complete = True
                if not checkpoint_complete:
                    job_path.write_text(json.dumps(job, sort_keys=True))
                    supervise(
                        [sys.executable, "-m", "spoofing_detection.lob.empirical_controls_v2_runner", "--partition-job", str(job_path)],
                        max_rss_mb=resources["max_rss_mb"], timeout_seconds=resources["timeout_seconds"],
                        cwd=Path(__file__).resolve().parents[3], env=_worker_environment(),
                        diagnostics_path=(
                            output.parent / f".{output.name}.supervision-v2" / f"{name}.json"
                        ),
                    )
                job_path.unlink(missing_ok=True)
                risk_manifest = _json(base / "risk" / "manifest.json")
                analysis_manifest = _json(base / "analysis" / "manifest.json")
                for abandoned in base.glob(".*.incomplete-*"):
                    if abandoned.is_dir():
                        shutil.rmtree(abandoned)
                partitions.append({
                    "instrument": instrument, "partition_id": partition["partition_id"], "event_date": partition["event_date"],
                    "risk_path": (base / "risk").relative_to(temporary).as_posix(), "risk_cache_key": risk_manifest["cache_key"],
                    "analysis_path": (base / "analysis").relative_to(temporary).as_posix(), "analysis_cache_key": analysis_manifest["cache_key"],
                    "analysis_cache_label": analysis_label,
                    "cluster_path": cluster_path.relative_to(temporary).as_posix(),
                    "cluster_sha256": file_hash(cluster_path),
                })
        analysis_frames: dict[str, list[pl.LazyFrame]] = {name: [] for name in ROOT_ANALYSIS}
        for item in partitions:
            tables = scan_checkpoint(
                temporary / item["analysis_path"], cache_key=item["analysis_cache_key"],
                schemas=ANALYSIS_SCHEMAS, provenance=checkpoint_provenance,
            )
            for name in ROOT_ANALYSIS:
                analysis_frames[name].append(tables[name])
        for name, schema in ROOT_ANALYSIS.items():
            combined = pl.concat(analysis_frames[name], how="vertical") if analysis_frames[name] else None
            _write_lazy(combined, temporary / f"{name}.parquet", schema)
        contributions = pl.scan_parquet(temporary / "summary_contributions.parquet")
        summary = summarize_contributions(contributions)
        summary.write_parquet(temporary / "summary.parquet")
        summary.write_csv(temporary / "summary.csv")
        pl.scan_parquet(temporary / "coverage.parquet").sink_csv(temporary / "coverage.csv")
        pl.scan_parquet(temporary / "balance.parquet").sink_csv(temporary / "balance.csv")
        pl.scan_parquet(temporary / "shift_support.parquet").sink_csv(temporary / "shift_support.csv")
        summary.filter(pl.col("analysis_kind") == "timing_shift_common_support").write_csv(temporary / "timing_sensitivity.csv")
        shutil.copyfile(config_path, temporary / "config.json")
        (temporary / "versions.txt").write_text(f"python={platform.python_version()}\npolars={pl.__version__}\n")
        (temporary / "command.txt").write_text(f"{sys.executable} -m spoofing_detection.lob.empirical_controls_v2_runner --config {config_path.resolve()}\n")
        (temporary / "validation.json").write_text(json.dumps({"status": "passed", "checks": ["partition_checkpoints", "root_table_schemas", "weighted_summary"]}, indent=2))
        provenance = temporary / "provenance"
        provenance.mkdir(exist_ok=True)
        source_provenance = {}
        for instrument, run in sorted(validated["runs"].items()):
            provenance_name = _safe_partition_name(instrument, "certificate")
            certificate = provenance / f"{provenance_name}.json"
            shutil.copyfile(run["certificate_path"], certificate)
            if file_hash(certificate) != run["certificate_sha256"]:
                raise ValueError("copied baseline certificate hash mismatch")
            certificate_data = _json(certificate)
            evidence_entry = certificate_data.get("evidence")
            if not isinstance(evidence_entry, dict) or not isinstance(evidence_entry.get("path"), str):
                raise ValueError("baseline certificate evidence path missing")
            evidence_source = Path(evidence_entry["path"]).resolve()
            evidence = provenance / f"{provenance_name}-evidence.json"
            shutil.copyfile(evidence_source, evidence)
            if file_hash(evidence) != evidence_entry.get("sha256"):
                raise ValueError("copied baseline certification evidence hash mismatch")
            source_provenance[instrument] = {
                "certificate_path": certificate.relative_to(temporary).as_posix(),
                "certificate_sha256": run["certificate_sha256"],
                "evidence_path": evidence.relative_to(temporary).as_posix(),
                "evidence_sha256": file_hash(evidence),
                "producer_source_identity": run["producer_source_identity"],
                "artifact_hashes": dict(sorted(run["artifact_hashes"].items())),
            }
        source_identity = value_hash(source_provenance)
        manifest = {"design_version": VERSION, "bundle_schema": BUNDLE_SCHEMA, "status": "complete",
                    "clock_policy_id": config.get("clock_policy_id"),
                    "source_identity": source_identity, "sources": source_provenance,
                    "risk_producer_identity": risk_producer_identity(), "partitions": partitions,
                    "artifacts": _artifact_graph(temporary, partitions, source_provenance)}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        report = build_report(temporary)
        manifest["artifacts"]["report.md"] = {"sha256": file_hash(report), "units": _units("report.md")}
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
        validate_bundle(temporary)
        os.rename(temporary, output)
        return output
    except Exception:
        # Preserve incomplete work for diagnosis; it has no final bundle path and
        # cannot pass checkpoint/manifest validation.
        raise


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--partition-job", type=Path)
    parser.add_argument("--preflight-job", nargs=2, type=Path, metavar=("CONFIG", "RESULT"))
    args = parser.parse_args(argv)
    if sum(option is not None for option in (args.config, args.partition_job, args.preflight_job)) != 1:
        parser.error("exactly one of --config, --partition-job or --preflight-job is required")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.preflight_job is not None:
        _preflight_worker(*args.preflight_job)
    elif args.partition_job is not None:
        _worker(args.partition_job)
    else:
        print(execute_v2(args.config))


if __name__ == "__main__":
    main()
