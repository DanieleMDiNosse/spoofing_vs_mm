"""Fail-closed validation and descriptive reporting for v2 bundles."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

import polars as pl
from polars.testing import assert_frame_equal

from . import empirical_controls_v2 as accounting
from .empirical_controls_v2_analysis import SCHEMAS as ANALYSIS_SCHEMAS, bound_cache_key
from .empirical_controls_v2_io import (
    CLOCK_POLICY_ID, EMPIRICAL_POLICY_CONTRACT, file_hash, risk_cache_key,
    risk_producer_identity, scan_checkpoint, validate_config, value_hash,
)
from .withdrawal_risk_v2 import SCHEMAS as RISK_SCHEMAS

DESIGN_VERSION = "empirical_controls_v2_light"
BUNDLE_SCHEMA = "empirical_controls_v2_bundle_2"
SUMMARY_KEYS = ("instrument", "identity_level", "analysis_kind", "offset_seconds", "comparison_mode")
ROOT_SCHEMAS: dict[str, Mapping[str, pl.DataType]] = {
    "state_statistics.parquet": accounting.STATE_SCHEMA,
    "contrast_statistics.parquet": accounting.CONTRAST_SCHEMA,
    "coverage.parquet": accounting.COVERAGE_SCHEMA,
    "balance.parquet": accounting.BALANCE_SCHEMA,
    "shift_support.parquet": accounting.SHIFT_SUPPORT_SCHEMA,
    "summary_contributions.parquet": accounting.SUMMARY_SCHEMA,
    "summary.parquet": accounting.SUMMARY_SCHEMA,
}
REQUIRED_FILES = set(ROOT_SCHEMAS) | {
    "summary.csv", "coverage.csv", "balance.csv", "shift_support.csv", "timing_sensitivity.csv",
    "config.json", "versions.txt", "command.txt", "validation.json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _safe_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("bundle artifact path must be nonempty and relative")
    target = (root / relative).resolve()
    resolved_root = root.resolve()
    if target == resolved_root or resolved_root not in target.parents:
        raise ValueError("bundle artifact path escapes bundle root")
    return target


def summarize_contributions(contributions: pl.LazyFrame) -> pl.DataFrame:
    """Combine partition/actor summaries using their original time weights."""
    missing = set(accounting.SUMMARY_SCHEMA) - set(contributions.collect_schema().names())
    if missing:
        raise ValueError(f"summary contributions missing columns: {sorted(missing)}")
    selected = contributions.select(list(accounting.SUMMARY_SCHEMA))
    count = pl.col("group_count")
    weight = pl.col("weight_seconds")
    difference = pl.col("weighted_intensity_difference")
    status = pl.col("summary_status")
    invalid = (
        count.is_null() | (count < 0) | weight.is_null() | (~weight.is_finite()) | (weight < 0)
        | ((weight > 0) & (difference.is_null() | ~difference.is_finite() | (status != "estimable")))
        | ((weight == 0) & (difference.is_not_null() | (status != "not_estimable")))
    ).fill_null(True)
    if selected.filter(invalid).select(pl.len()).collect().item():
        raise ValueError("summary contribution values must be finite with nonnegative denominators")
    grouped = (
        selected.with_columns(
            pl.when(weight > 0).then(weight * difference).otherwise(0.0).alias("_numerator")
        )
        .group_by(list(SUMMARY_KEYS))
        .agg(
            count.sum().alias("group_count"),
            weight.sum().alias("weight_seconds"),
            pl.col("_numerator").sum().alias("_numerator"),
        )
        .with_columns(
            pl.when(weight > 0).then(pl.col("_numerator") / weight).otherwise(None)
            .alias("weighted_intensity_difference"),
            pl.when(weight > 0).then(pl.lit("estimable")).otherwise(pl.lit("not_estimable"))
            .alias("summary_status"),
        )
        .select([pl.col(name).cast(dtype, strict=True) for name, dtype in accounting.SUMMARY_SCHEMA.items()])
        .sort(list(SUMMARY_KEYS))
    )
    return grouped.collect()


def _validate_frame(path: Path, schema: Mapping[str, pl.DataType]) -> pl.LazyFrame:
    frame = pl.scan_parquet(path)
    if dict(frame.collect_schema()) != dict(schema):
        raise ValueError(f"schema mismatch: {path.name}")
    return frame


def _validate_state_accounting(states: pl.LazyFrame) -> None:
    time = pl.col("time_at_risk_seconds")
    count = pl.col("withdrawal_count")
    intensity = pl.col("withdrawal_intensity")
    support_anomaly = pl.col("support_anomaly")
    invalid = (
        time.is_null() | (~time.is_finite()) | (time < 0) | count.is_null() | (count < 0)
        # A zero-duration risk boundary can carry a canonical withdrawal point.
        # It is reportable only as the explicitly flagged, null-intensity support
        # anomaly emitted by the accounting stage.
        | ((time == 0) & (intensity.is_not_null() | (support_anomaly != (count > 0))))
        | ((time > 0) & support_anomaly)
        | ((time > 0) & (
            intensity.is_null() | (~intensity.is_finite())
            | ((intensity - count / time).abs() > 1e-12 * pl.max_horizontal(pl.lit(1.0), (count / time).abs()))
        ))
    ).fill_null(True)
    if states.filter(invalid).select(pl.len()).collect().item():
        raise ValueError("persisted withdrawal intensity disagrees with numerator/denominator")


def _semantic_sort(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(frame.columns, nulls_last=True) if frame.height else frame


def _validate_csv_export(actual: Path, expected: pl.LazyFrame | pl.DataFrame) -> None:
    with tempfile.TemporaryDirectory(prefix="empirical-controls-v2-csv-") as directory:
        target = Path(directory) / actual.name
        if isinstance(expected, pl.LazyFrame):
            expected.sink_csv(target)
        else:
            expected.write_csv(target)
        if _sha256(actual) != _sha256(target):
            raise ValueError(f"CSV contents do not match validated parquet source: {actual.name}")


def _validate_selected_cluster_policy(clusters: pl.LazyFrame) -> None:
    """Require every persisted v3 cluster to retain its exact fill-time anchor."""
    schema = dict(clusters.collect_schema())
    if schema.get("clock_policy_id") != pl.String or schema.get("fill_anchor_ns") != pl.Int64:
        raise ValueError("selected bundle cluster schema lacks policy or native fill anchor")
    invalid = clusters.filter(
        pl.col("clock_policy_id").is_null()
        | (pl.col("clock_policy_id") != CLOCK_POLICY_ID)
        | pl.col("fill_anchor_ns").is_null()
    ).select(pl.len()).collect().item()
    if invalid:
        raise ValueError("selected bundle cluster has incompatible policy or native fill anchor")


def validate_bundle(root: Path) -> dict[str, Any]:
    """Authenticate and reconcile a complete persisted v2 bundle."""
    root = root.resolve()
    manifest = _json(root / "manifest.json")
    if (manifest.get("design_version") != DESIGN_VERSION or manifest.get("bundle_schema") != BUNDLE_SCHEMA
            or manifest.get("status") != "complete" or not manifest.get("source_identity")):
        raise ValueError("incomplete or incompatible v2 bundle manifest")
    validation = _json(root / "validation.json")
    if validation.get("status") != "passed":
        raise ValueError("bundle validation status is not passed")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not REQUIRED_FILES <= set(artifacts):
        raise ValueError("bundle required artifact graph is incomplete")
    for relative, entry in artifacts.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("sha256"), str):
            raise ValueError("artifact hash declaration required")
        units = entry.get("units")
        if not isinstance(units, dict) or not units or any(not str(v).strip() for v in units.values()):
            raise ValueError("nonempty artifact units declaration required")
        target = _safe_path(root, relative)
        if not target.is_file() or _sha256(target) != entry["sha256"]:
            raise ValueError(f"artifact hash mismatch: {relative}")
    actual_files = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    if actual_files != set(artifacts) | {"manifest.json"}:
        raise ValueError("bundle contains undeclared files or declares absent artifacts")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or not sources or manifest.get("source_identity") != value_hash(sources):
        raise ValueError("structured certified source provenance is missing or inconsistent")
    if manifest.get("risk_producer_identity") != risk_producer_identity():
        raise ValueError("bundle risk producer source identity differs from current implementation")
    for instrument, source in sources.items():
        if not isinstance(instrument, str) or not isinstance(source, dict):
            raise ValueError("invalid certified source provenance entry")
        certificate_path = source.get("certificate_path")
        certificate = _safe_path(root, certificate_path)
        evidence_path = source.get("evidence_path")
        evidence_file = _safe_path(root, evidence_path)
        hashes = source.get("artifact_hashes")
        certificate_data = _json(certificate)
        certificate_artifacts = certificate_data.get("artifacts")
        certificate_hashes = (
            {name: entry.get("sha256") for name, entry in certificate_artifacts.items()}
            if isinstance(certificate_artifacts, dict)
            and all(isinstance(entry, dict) for entry in certificate_artifacts.values())
            else None
        )
        certificate_evidence = certificate_data.get("evidence")
        evidence = _json(evidence_file)
        if (not isinstance(hashes, dict) or not hashes
                or any(not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
                       for name, digest in hashes.items())
                or not isinstance(source.get("producer_source_identity"), (str, dict))
                or not source["producer_source_identity"]
                or (isinstance(source["producer_source_identity"], dict)
                    and any(not isinstance(name, str) or not isinstance(digest, str) or len(digest) != 64
                            for name, digest in source["producer_source_identity"].items()))
                or source.get("certificate_sha256") != _sha256(certificate)
                or source.get("evidence_sha256") != _sha256(evidence_file)
                or certificate_path not in artifacts
                or evidence_path not in artifacts
                or artifacts[certificate_path]["sha256"] != source["certificate_sha256"]
                or artifacts[evidence_path]["sha256"] != source["evidence_sha256"]
                or certificate_data.get("certificate_version") != "canonical_execution_inputs_v2"
                or certificate_hashes != hashes
                or not isinstance(certificate_evidence, dict)
                or certificate_evidence.get("sha256") != source["evidence_sha256"]
                or evidence.get("status") != "passed"
                or evidence.get("artifact_hashes") != hashes
                or evidence.get("producer_source_identity") != source["producer_source_identity"]):
            raise ValueError("invalid persisted baseline certification provenance")
    config = _json(root / "config.json")
    if config.get("clock_policy_id") != manifest.get("clock_policy_id"):
        raise ValueError("bundle manifest clock policy is not bound to config")
    selected_policy = config.get("clock_policy_id") == CLOCK_POLICY_ID
    if selected_policy:
        validate_config(config)
        if (config["clock"].get("timezone") != "UTC"
                or config["clock"].get("empirical_policy_contract") != EMPIRICAL_POLICY_CONTRACT):
            raise ValueError("selected bundle config has no UTC clock contract")
        for source in sources.values():
            certificate_data = _json(_safe_path(root, source["certificate_path"]))
            if certificate_data.get("clock") != config["clock"]:
                raise ValueError("selected bundle certificate clock does not match config")
    frames = {name: _validate_frame(root / name, schema) for name, schema in ROOT_SCHEMAS.items()}
    _validate_state_accounting(frames["state_statistics.parquet"])
    expected_summary = summarize_contributions(frames["summary_contributions.parquet"])
    summary = frames["summary.parquet"].collect()
    assert_frame_equal(_semantic_sort(summary), _semantic_sort(expected_summary))
    _validate_csv_export(root / "summary.csv", summary)
    _validate_csv_export(root / "coverage.csv", frames["coverage.parquet"])
    _validate_csv_export(root / "balance.csv", frames["balance.parquet"])
    _validate_csv_export(root / "shift_support.csv", frames["shift_support.parquet"])
    _validate_csv_export(
        root / "timing_sensitivity.csv",
        summary.filter(pl.col("analysis_kind") == "timing_shift_common_support"),
    )
    partitions = manifest.get("partitions")
    if not isinstance(partitions, list):
        raise ValueError("partition declarations required")
    partition_row_counts = {name: 0 for name in ANALYSIS_SCHEMAS}
    for item in partitions:
        if not isinstance(item, dict):
            raise ValueError("invalid partition declaration")
        risk = _safe_path(root, item.get("risk_path"))
        analysis = _safe_path(root, item.get("analysis_path"))
        cluster = _safe_path(root, item.get("cluster_path"))
        for required in (str(Path(item["risk_path"]) / "manifest.json"), str(Path(item["analysis_path"]) / "manifest.json"), item["cluster_path"]):
            if required not in artifacts:
                raise ValueError("partition dependency is absent from artifact graph")
        source = sources.get(item.get("instrument"))
        if not isinstance(source, dict):
            raise ValueError("partition has no matching certified source provenance")
        expected_risk_key = risk_cache_key(
            config, source["artifact_hashes"], source["producer_source_identity"],
        )
        if expected_risk_key != item.get("risk_cache_key"):
            raise ValueError("risk checkpoint is not bound to declared source/config/producer code")
        provenance = {"clock_policy_id": config.get("clock_policy_id")} if selected_policy else None
        scan_checkpoint(risk, cache_key=item.get("risk_cache_key"), schemas=RISK_SCHEMAS, provenance=provenance)
        scan_checkpoint(analysis, cache_key=item.get("analysis_cache_key"), schemas=ANALYSIS_SCHEMAS, provenance=provenance)
        if not cluster.is_file():
            raise ValueError("partition cluster table missing")
        if item.get("cluster_sha256") != file_hash(cluster):
            raise ValueError("partition cluster hash mismatch")
        if not accounting.CLUSTER_REQUIRED <= set(pl.scan_parquet(cluster).collect_schema().names()):
            raise ValueError("partition cluster schema incomplete")
        if selected_policy:
            _validate_selected_cluster_policy(pl.scan_parquet(cluster))
        expected_analysis_key = bound_cache_key(
            risk, cluster_identity=item["cluster_sha256"], requested=item.get("analysis_cache_label"),
        )
        if expected_analysis_key != item.get("analysis_cache_key"):
            raise ValueError("analysis checkpoint is not bound to declared risk/clusters/source")
        analysis_manifest = _json(analysis / "manifest.json")
        for name, entries in analysis_manifest["tables"].items():
            partition_row_counts[name] += sum(entry["rows"] for entry in entries)
    for name in ANALYSIS_SCHEMAS:
        root_rows = frames[f"{name}.parquet"].select(pl.len()).collect().item()
        if root_rows != partition_row_counts[name]:
            raise ValueError(f"root table row count does not reconcile with partition checkpoints: {name}")
    return {"partition_count": len(partitions), "summary_group_count": expected_summary.height,
            "state_row_count": frames["state_statistics.parquet"].select(pl.len()).collect().item()}


def build_report(root: Path) -> Path:
    """Validate first, then render explicitly descriptive interpretation guardrails."""
    stats = validate_bundle(root)
    report = root / "report.md"
    report.write_text(
        "# Empirical controls v2\n\n"
        "This report describes an exploratory unadjusted association between observed own executions and "
        "eligible opposite-side physical cancellation intensity per posture-second. It is not a probability.\n\n"
        "`no_own_observed` means no attributed observed own-execution window, not absence of economic trading. "
        "The fixed timing shifts use one common clock domain and the offset-zero weights at every offset. "
        "There is no causality, intent, or false-positive-rate claim, and no p-value or confidence interval.\n\n"
        f"Validated partitions: {stats['partition_count']}; summary groups: {stats['summary_group_count']}; "
        f"state rows: {stats['state_row_count']}.\n"
    )
    return report
