#!/usr/bin/env python3
"""Versioned multiscale withdrawal analysis; explicit inputs, no legacy promotion.

The original paper release and raw inputs are never overwritten. Scientific
parameters are JSON-only; --validate-only is strictly read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "withdrawal_multiscale_v1"
CLOCK_POLICY = "spoofing_detection_sequence_replay_trade_fill_sequence_cancel_v3_utc_provisional"
FIXED = {
    "schema": SCHEMA,
    "top_n": 10,
    "source_clock_policy": CLOCK_POLICY,
    "depth_denominator": "opposite_market_top10_exact_pre_first_fill_all_participants",
    "age_clock": "first_fill_sequence_minus_lifecycle_first_seen_sequence_ns",
    "age_weight_policy": "frozen_baseline_shared_P_and_U_half_life_v1",
    "grid_step_ns": 10_000_000,
    "lag_ns": 50_000_000,
    "priority_policy": "separate_W_deciles_with_conditional_J_flags_not_candidate_gate",
    "candidate_policy": "positive_raw_candidate_and_attributed_withdrawal_no_percentile_no_price_gate",
    "kernel_policy": "retain_verified_H1_depth_weights_no_recalibration",
    "assignment_policy": "native_latest_prior_cluster_sort_index_per_cell",
    "withdrawal_semantics": "gross_capped_explicit_cancel_same_observed_lifecycle",
    "price_selection": False,
    "uq_selection": False,
    "publication_status": "exploratory_not_promoted",
}
CONFIG_KEYS = set(FIXED) | {"sources", "baseline_manifest", "horizons_ns", "age_policies", "review_fraction"}
POLICIES = {
    "legacy_age30": (30_000_000_000, None),
    "all_ages_unweighted": (None, None),
    "exp_h30": (None, 30_000_000_000),
    "exp_h120": (None, 120_000_000_000),
    "exp_h300": (None, 300_000_000_000),
}
EXEC_ANCHORS = [
    "partition_id", "actor_key", "execution_cluster_id", "cluster_first_sort_index",
    "cluster_last_sort_index", "cluster_start_replay_time_ns", "cluster_end_replay_time_ns",
    "cluster_end_fill_anchor_ns", "execution_quantity", "execution_anchor_mode", "deceptive_side",
]
STATE_COLUMNS = ["partition_id", "actor_key", "sort_index", "replay_time_ns", "DWI", "L_bid_topN", "L_ask_topN"]
RAW_COLUMNS = ["TRADEDATE", "ORDERID", "ORDEREVENTTYPE (*)", "ORDERSIDE (*)", "ORDERPX",
               "LEAVESQTY", "DISPLAYEDQTY", "SEQUENCETIME", "TRADETIME", "FIRMID",
               "NMSC_ORIGINALCLIENTIDSHORTCODE", "PASSIVEORDER", "AGGRESSIVEORDER"]


def validate_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError("config must contain exactly the versioned contract keys")
    for key, expected in FIXED.items():
        if type(config[key]) is not type(expected) or config[key] != expected:
            raise ValueError(f"incompatible scientific contract: {key}")
    for key in ("sources", "baseline_manifest"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f"{key} must be a nonempty path")
    horizons = config["horizons_ns"]
    if not isinstance(horizons, list) or not horizons:
        raise ValueError("horizons_ns must be nonempty")
    if any(type(h) is not int or h < config["lag_ns"] or h % config["grid_step_ns"] for h in horizons):
        raise ValueError("horizons must be integer grid multiples and at least the lag")
    if len(set(horizons)) != len(horizons):
        raise ValueError("duplicate horizons")
    fraction = config["review_fraction"]
    if isinstance(fraction, bool) or not isinstance(fraction, (float, int)) or not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError("review_fraction must be finite in (0,1]")
    policies = config["age_policies"]
    if not isinstance(policies, list) or not policies:
        raise ValueError("age_policies must be nonempty")
    identifiers = []
    for policy in policies:
        if not isinstance(policy, dict) or set(policy) != {"id", "max_age_ns", "half_life_ns"}:
            raise ValueError("age policy requires exactly id/max_age_ns/half_life_ns")
        identifier = policy["id"]
        if not isinstance(identifier, str) or identifier not in POLICIES:
            raise ValueError("unknown age policy")
        actual = (policy["max_age_ns"], policy["half_life_ns"])
        expected = POLICIES[identifier]
        for value, reference in zip(actual, expected, strict=True):
            if type(value) is not type(reference) or value != reference:
                raise ValueError(f"age policy relabeling: {identifier}")
        identifiers.append(identifier)
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate age policies")


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def check_binding(path: str | Path, expected: str) -> dict[str, str]:
    if not isinstance(expected, str) or re.fullmatch(r"[0-9a-f]{64}", expected) is None:
        raise ValueError("invalid sha256 digest syntax")
    path = Path(path)
    if sha256(path) != expected:
        raise ValueError(f"changed input: {path}")
    return {"path": str(path.resolve()), "sha256": expected}


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def read_json(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def require_parquet(path: str | Path, columns: list[str]) -> dict:
    path = Path(path)
    schema = pl.read_parquet_schema(path)
    missing = set(columns) - set(schema)
    if missing:
        raise ValueError(f"missing columns in {path}: {sorted(missing)}")
    # Read the actual required pages, not only a hash or footer.
    frame = pl.read_parquet(path, columns=columns)
    return {"rows": frame.height, "schema": {name: str(schema[name]) for name in columns}}


def preflight(config: dict, instrument: str) -> dict:
    validate_config(config)
    if instrument not in {"FERRARI", "NEXI", "RISANAMENTO"}:
        raise ValueError("unsupported instrument")
    registry_path = resolve(config["sources"])
    registry = read_json(registry_path)
    bundle = resolve(registry["sources"][instrument]["bundle"])
    metadata_path = bundle / "metadata.json"
    bindings = {"source_metadata": check_binding(metadata_path, registry["metadata_hashes"][instrument])}
    metadata = read_json(metadata_path)
    if metadata["clock_policy_id"] != CLOCK_POLICY or metadata["output_schema_version"] != "actor_execution_anchor_v3_clocked":
        raise ValueError("incompatible source clock/schema")
    if type(metadata["top_n"]) is not int or metadata["top_n"] != 10 or metadata["max_rows"] is not None:
        raise ValueError("source must be a complete top10 replay")
    if sorted(metadata["execution_anchor_modes"]) != ["aggressive", "passive"]:
        raise ValueError("source must preserve both execution roles")
    tick = metadata["tick_size"]
    if isinstance(tick, bool) or not isinstance(tick, (int, float)) or not math.isfinite(tick) or tick <= 0:
        raise ValueError("invalid source tick size")
    raw = resolve(metadata["input"])
    kernel = resolve(metadata["empirical_depth_kernel"])
    bindings["raw"] = check_binding(raw, metadata["input_hashes"]["raw_events_sha256"])
    bindings["kernel"] = check_binding(kernel, metadata["input_hashes"]["empirical_depth_kernel_sha256"])
    schemas = {"raw": require_parquet(raw, RAW_COLUMNS)}
    for name, columns in [("execution_metrics", EXEC_ANCHORS), ("state_time_series", STATE_COLUMNS),
                           ("execution_cluster_members", ["partition_id", "execution_cluster_id", "child_sort_index", "child_event_id"])]:
        path = resolve(metadata["paths"][name])
        bindings[name] = check_binding(path, metadata["artifact_hashes"][name])
        schemas[name] = require_parquet(path, columns)
    from spoofing_detection.lob.depth_kernel_calibration import load_empirical_kernel_weights
    load_empirical_kernel_weights(kernel)
    baseline_path = resolve(config["baseline_manifest"])
    baseline = read_json(baseline_path)
    witness_path = resolve(baseline["actual_selection_config"]["coverage_witness"])
    matches = [b for b in baseline["inputs_and_code"].values() if resolve(b["path"]) == witness_path]
    if not matches or any(b["sha256"] != matches[0]["sha256"] for b in matches):
        raise ValueError("coverage witness has no unambiguous baseline binding")
    bindings["coverage_witness"] = check_binding(witness_path, matches[0]["sha256"])
    witness = read_json(witness_path)
    if witness["clock_policy_id"] != CLOCK_POLICY:
        raise ValueError("incompatible coverage clock")
    coverage = witness["instruments"][instrument]["partition_coverage_witness"]
    if coverage["state_time_series_sha256"] != bindings["state_time_series"]["sha256"]:
        raise ValueError("coverage/state binding mismatch")
    bindings["sources_registry"] = {"path": str(registry_path), "sha256": sha256(registry_path)}
    bindings["baseline_manifest"] = {"path": str(baseline_path), "sha256": sha256(baseline_path)}
    return {"status": "preflight_verified", "instrument": instrument, "metadata": metadata,
            "bindings": bindings, "schemas": schemas, "coverage": coverage}


PREPARE_VERSION = "withdrawal_multiscale_prepare_v1"
FRAME_NAMES = ("executions", "candidates", "cancellations", "rejected_executions", "execution_cluster_members")
EXEC_KEYS = ["partition_id", "actor_key", "execution_cluster_id"]
MEMBER_KEYS = EXEC_KEYS + ["child_sort_index", "child_event_id"]


def source_code_bindings() -> dict:
    # Freeze actual working-tree bytes, not HEAD: this repo can contain dirty
    # scientific fixes. Include LOB dependencies and future scoring consumers.
    paths = [Path(__file__).resolve(), *sorted((ROOT / "src/spoofing_detection/lob").glob("*.py"))]
    return {str(path.relative_to(ROOT)): {"path": str(path), "sha256": sha256(path)} for path in paths}


def prepare_identity(config: dict, config_path: Path, result: dict) -> dict:
    return {"version": PREPARE_VERSION, "schema": SCHEMA,
            "instrument": result["instrument"], "config": config,
            "config_file": {"path": str(config_path.resolve()), "sha256": sha256(config_path)},
            "input_bindings": result["bindings"], "code_bindings": source_code_bindings(),
            "versions": {"python": sys.version, "polars": pl.__version__}}


def require_unique_keys(frame: pl.DataFrame, keys: list[str], label: str) -> None:
    if set(keys) - set(frame.columns):
        raise ValueError(f"{label}: missing keys")
    for key in keys:
        if frame[key].null_count():
            raise ValueError(f"{label}: null key {key}")
        if frame.schema[key] == pl.String and frame.filter(pl.col(key) == "").height:
            raise ValueError(f"{label}: empty key {key}")
    if frame.select(keys).n_unique() != frame.height:
        raise ValueError(f"{label}: duplicate keys")


def compare_exact(actual: pl.DataFrame, original: pl.DataFrame,
                  columns: list[str], keys: list[str], label: str) -> None:
    for frame in (actual, original):
        require_unique_keys(frame, keys, label)
        if set(columns) - set(frame.columns):
            raise ValueError(f"{label}: missing comparison columns")
    left, right = (frame.select(columns).sort(keys) for frame in (actual, original))
    # DataFrame.equals deliberately treats two nulls as equal, but rejects a
    # one-sided null (unlike filter(col_a != col_b), which silently drops it).
    if left.schema != right.schema or not left.equals(right, null_equal=True):
        raise ValueError(f"{label}: exact anchor/member mismatch")


def prepare_frames(result: dict) -> tuple[dict[str, pl.DataFrame], dict[str, str]]:
    from spoofing_detection.lob.io import read_events
    from spoofing_detection.lob.depth_kernel_calibration import load_empirical_kernel_weights
    from spoofing_detection.lob.spoofing_metrics import _stream_metric_inputs
    metadata = result["metadata"]
    _, executions, candidates, cancellations, rejected, members = _stream_metric_inputs(
        read_events(resolve(metadata["input"])), top_n=10, tick_size=metadata["tick_size"],
        execution_cluster_max_gap_ms=metadata["execution_cluster_max_gap_ms"],
        empirical_kernel_weights=load_empirical_kernel_weights(resolve(metadata["empirical_depth_kernel"])),
        max_rows=None, include_level_columns=False, max_deceptive_order_age_seconds=None,
        state_actor_keys=set(), execution_anchor_modes=metadata["execution_anchor_modes"],
        use_v3_clocks=True, retain_withdrawal_baseline=True)
    original_executions = pl.read_parquet(resolve(metadata["paths"]["execution_metrics"]))
    compare_exact(executions, original_executions, EXEC_ANCHORS, EXEC_KEYS, "execution anchors")
    original_members = pl.read_parquet(resolve(metadata["paths"]["execution_cluster_members"]))
    compare_exact(members, original_members, original_members.columns, MEMBER_KEYS, "execution members")
    instrument = pl.lit(result["instrument"], dtype=pl.String).alias("instrument")
    frames = dict(zip(FRAME_NAMES, (executions.with_columns(instrument), candidates.with_columns(instrument),
                                 cancellations, rejected, members), strict=True))
    return frames, {"execution_metrics": "exact", "execution_cluster_members": "exact"}


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n")


def verify_prepared(output: Path, identity: dict) -> dict:
    """Verify a marked final in place, without replay or any filesystem write."""
    try:
        marker = read_json(output / "COMPLETE.json")
        manifest_path = output / "artifact_manifest.json"
        expected_marker = {"status": "prepared", "version": PREPARE_VERSION,
                           "artifact_manifest_sha256": sha256(manifest_path)}
        if marker != expected_marker:
            raise ValueError("invalid COMPLETE marker")
        manifest = read_json(manifest_path)
        names = {f"{name}.parquet" for name in FRAME_NAMES} | {"prepare_receipt.json"}
        if set(manifest) != {"version", "artifacts"} or manifest["version"] != PREPARE_VERSION or set(manifest["artifacts"]) != names:
            raise ValueError("invalid prepare artifact manifest")
        for name in sorted(names):
            check_binding(output / name, manifest["artifacts"][name])
        receipt = read_json(output / "prepare_receipt.json")
        if receipt.get("identity") != identity:
            raise ValueError("incompatible prepare identity")
        if receipt.get("status") != "prepared" or receipt.get("anchor_checks") != {
                "execution_metrics": "exact", "execution_cluster_members": "exact"}:
            raise ValueError("invalid prepare receipt")
        rows = {name: pl.read_parquet(output / f"{name}.parquet").height for name in FRAME_NAMES}
        if receipt.get("rows") != rows:
            raise ValueError("prepare row counts mismatch")
        return receipt
    except (OSError, KeyError, TypeError, pl.exceptions.PolarsError) as exc:
        raise ValueError(f"corrupt marked prepare: {output}") from exc


def publish_prepare(output: Path, result: dict, identity: dict) -> dict:
    # Every attempt is private and on the same filesystem as the final. Never
    # reuse abandoned attempts, even if they already contain a COMPLETE marker.
    output.parent.mkdir(parents=True, exist_ok=True)
    private = output.with_name(f".{output.name}.attempt-{uuid.uuid4().hex}")
    private.mkdir()
    frames, checks = prepare_frames(result)
    receipt = {"status": "prepared", "identity": identity,
               "rows": {name: frame.height for name, frame in frames.items()}, "anchor_checks": checks}
    for name, frame in frames.items():
        frame.write_parquet(private / f"{name}.parquet")
    write_json(private / "prepare_receipt.json", receipt)
    names = [f"{name}.parquet" for name in FRAME_NAMES] + ["prepare_receipt.json"]
    write_json(private / "artifact_manifest.json", {
        "version": PREPARE_VERSION, "artifacts": {name: sha256(private / name) for name in names}})
    # Completion precedes publication; a failure cannot expose a marked final.
    write_json(private / "COMPLETE.json", {"status": "prepared", "version": PREPARE_VERSION,
        "artifact_manifest_sha256": sha256(private / "artifact_manifest.json")})
    verify_prepared(private, identity)
    if output.exists():
        raise FileExistsError(output)
    os.rename(private, output)
    return verify_prepared(output, identity)


ANALYSIS_VERSION = "withdrawal_multiscale_derive_v1"
PROFILE_BATCH_SIZE = 32


def digest_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def analysis_identity(prepared: Path, identity: dict, result: dict) -> dict:
    verify_prepared(prepared, identity)
    return {"version": ANALYSIS_VERSION, "prepare_identity": identity,
            "prepare_manifest": check_binding(prepared / "artifact_manifest.json",
                                                sha256(prepared / "artifact_manifest.json")),
            "coverage": result["coverage"], "profile_batch_size": PROFILE_BATCH_SIZE,
            "profile_support_bound_ns": max(identity["config"]["horizons_ns"])}


def analysis_stage(output: Path) -> Path:
    return output.with_name(f".{output.name}.derived")


def compute_profiles(executions: pl.DataFrame, result: dict, horizons: list[int]) -> pl.DataFrame:
    from spoofing_detection.lob.withdrawal_multiscale import profile_summaries
    # Only the requested partition/actor states are materialized per batch.
    local = executions.select(["partition_id", "actor_key"]).unique()
    states = (pl.scan_parquet(resolve(result["metadata"]["paths"]["state_time_series"]))
              .join(local.lazy(), on=["partition_id", "actor_key"], how="semi").collect())
    return profile_summaries(executions, states, result["coverage"], horizons)


def validate_profiles(summary: pl.DataFrame, executions: pl.DataFrame, horizons: list[int]) -> None:
    from spoofing_detection.lob.withdrawal_multiscale import KEYS, PROFILE_SCHEMA
    keys = KEYS + ["horizon_ns"]
    require_unique_keys(summary, keys, "profile membership")
    expected = executions.select(KEYS).join(pl.DataFrame({"horizon_ns": horizons}), how="cross")
    if summary.schema != PROFILE_SCHEMA or not summary.select(keys).sort(keys).equals(expected.sort(keys)):
        raise ValueError("profile exact membership/schema mismatch")
    if summary["complete_grid"].null_count():
        raise ValueError("profile complete_grid must be non-null Boolean")


def profile_batches(executions: pl.DataFrame):
    from spoofing_detection.lob.withdrawal_multiscale import KEYS
    for group in executions.sort(KEYS).partition_by(KEYS[:3], maintain_order=True):
        for offset in range(0, group.height, PROFILE_BATCH_SIZE):
            batch = group.slice(offset, PROFILE_BATCH_SIZE)
            yield digest_json(batch.select(KEYS).rows()), batch


def batch_identity(batch: pl.DataFrame, identity: dict, horizons: list[int]) -> dict:
    from spoofing_detection.lob.withdrawal_multiscale import KEYS
    return {"analysis_identity": identity, "members": batch.select(KEYS).to_dicts(),
            "membership_sha256": digest_json(batch.select(KEYS).rows()), "horizons_ns": horizons}


def verify_profile_batch(root: Path, binding: dict, batch: pl.DataFrame, horizons: list[int]) -> pl.DataFrame:
    marker = read_json(root / "COMPLETE.json")
    manifest_path = root / "manifest.json"
    check_binding(manifest_path, marker.get("manifest_sha256"))
    manifest = read_json(manifest_path)
    if marker.get("status") != "profile_verified" or manifest.get("identity") != binding:
        raise ValueError("incompatible profile batch identity")
    if set(manifest) != {"identity", "summary_sha256"}:
        raise ValueError("invalid profile batch manifest")
    check_binding(root / "summary.parquet", manifest["summary_sha256"])
    summary = pl.read_parquet(root / "summary.parquet")
    validate_profiles(summary, batch, horizons)
    return summary


def collect_profiles(root: Path, executions: pl.DataFrame, identity: dict,
                     result: dict, horizons: list[int], *, compute: bool) -> pl.DataFrame:
    from spoofing_detection.lob.withdrawal_multiscale import PROFILE_SCHEMA
    summaries = []
    expected = set()
    for name, batch in profile_batches(executions):
        expected.add(name)
        target = root / name
        binding = batch_identity(batch, identity, horizons)
        if not target.exists():
            if not compute:
                raise ValueError("missing completed profile batch")
            root.mkdir(parents=True, exist_ok=True)
            private = root / f".attempt-{uuid.uuid4().hex}"
            private.mkdir()
            summary = compute_profiles(batch, result, horizons)
            validate_profiles(summary, batch, horizons)
            summary.write_parquet(private / "summary.parquet")
            write_json(private / "manifest.json", {"identity": binding,
                       "summary_sha256": sha256(private / "summary.parquet")})
            write_json(private / "COMPLETE.json", {"status": "profile_verified",
                       "manifest_sha256": sha256(private / "manifest.json")})
            verify_profile_batch(private, binding, batch, horizons)
            if target.exists():
                raise FileExistsError(target)
            os.rename(private, target)
        # Never reuse an unmarked partial or tolerate a changed scope.
        summaries.append(verify_profile_batch(target, binding, batch, horizons))
    if root.exists() and {p.name for p in root.iterdir() if not p.name.startswith(".attempt-")} != expected:
        raise ValueError("profile batch scope mismatch")
    return pl.concat(summaries) if summaries else pl.DataFrame(schema=PROFILE_SCHEMA)


def prepared_inputs(prepared: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    return tuple(pl.read_parquet(prepared / f"{name}.parquet")
                 for name in ("executions", "candidates", "cancellations"))


def profile_scope(executions: pl.DataFrame, population: pl.DataFrame) -> pl.DataFrame:
    from spoofing_detection.lob.withdrawal_multiscale import KEYS
    return executions.join(population.filter(pl.col("is_candidate")).select(KEYS).unique(), on=KEYS, how="semi")


def attach_profiles(population: pl.DataFrame, profiles: pl.DataFrame) -> pl.DataFrame:
    from spoofing_detection.lob.withdrawal_multiscale import KEYS
    joined = population.join(profiles.with_columns(pl.lit(True).alias("_profile_computed")),
                             on=KEYS + ["horizon_ns"], how="left", validate="m:1")
    return joined.with_columns(
        pl.when(pl.col("_profile_computed").is_null()).then(pl.lit("not_computed_no_withdrawal"))
        .when(pl.col("complete_grid")).then(pl.lit("computed_complete"))
        .otherwise(pl.lit("computed_invalid")).alias("profile_status"),
        pl.col("complete_grid").fill_null(False)).drop("_profile_computed")


def seal_derived(root: Path, identity: dict) -> None:
    names = ["population.parquet", "cancellation_audit.parquet", "profiles.parquet", "derive_receipt.json"]
    write_json(root / "derived_manifest.json", {"version": ANALYSIS_VERSION,
               "artifacts": {name: sha256(root / name) for name in names}})
    write_json(root / "DERIVED.json", {"status": "derived", "version": ANALYSIS_VERSION,
               "artifact_manifest_sha256": sha256(root / "derived_manifest.json")})


def verify_derived(root: Path, identity: dict, prepared: Path, result: dict) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    marker = read_json(root / "DERIVED.json")
    manifest_path = root / "derived_manifest.json"
    check_binding(manifest_path, marker.get("artifact_manifest_sha256"))
    if marker != {"status": "derived", "version": ANALYSIS_VERSION,
                  "artifact_manifest_sha256": sha256(manifest_path)}:
        raise ValueError("invalid DERIVED marker")
    manifest = read_json(manifest_path)
    names = {"population.parquet", "cancellation_audit.parquet", "profiles.parquet", "derive_receipt.json"}
    if set(manifest) != {"version", "artifacts"} or manifest["version"] != ANALYSIS_VERSION or set(manifest["artifacts"]) != names:
        raise ValueError("invalid derived artifact manifest")
    for name in names:
        check_binding(root / name, manifest["artifacts"][name])
    receipt = read_json(root / "derive_receipt.json")
    if receipt.get("identity") != identity or receipt.get("status") != "derived":
        raise ValueError("incompatible derive identity")
    ex, _, _ = prepared_inputs(prepared)
    population = pl.read_parquet(root / "population.parquet")
    audit = pl.read_parquet(root / "cancellation_audit.parquet")
    profiles = pl.read_parquet(root / "profiles.parquet")
    config = identity["prepare_identity"]["config"]
    if population.height != ex.height * len(config["horizons_ns"]) * len(config["age_policies"]):
        raise ValueError("derive matrix row count mismatch")
    summaries = collect_profiles(root / "profiles", profile_scope(ex, population), identity,
                                 result, config["horizons_ns"], compute=False)
    compare_exact(profiles, summaries, profiles.columns,
                  ["instrument", *EXEC_KEYS, "horizon_ns"], "completed profiles")
    if receipt.get("rows") != {"population": population.height, "audit": audit.height, "profiles": profiles.height}:
        raise ValueError("derive receipt row counts mismatch")
    return population, audit, profiles


def derive_analysis(output: Path, prepared: Path, identity: dict, result: dict, *, resume: bool) -> str:
    from spoofing_detection.lob.withdrawal_multiscale import evaluate_cells, add_priority_flags
    from spoofing_detection.lob.spoofing_metrics import _build_execution_cancel_candidates
    if output.exists():
        if not resume:
            raise FileExistsError(output)
        verify_analysis(output, prepared, identity, result, publish=False)
        return "reused"
    root = analysis_stage(output)
    if root.exists():
        if not resume:
            raise FileExistsError(root)
        if read_json(root / "identity.json") != identity:
            raise ValueError("incompatible derive staging identity")
        if (root / "DERIVED.json").exists():
            verify_derived(root, identity, prepared, result)
            return "reused_derived"
    else:
        root.mkdir(parents=True)
        write_json(root / "identity.json", identity)
    ex, candidates, cancellations = prepared_inputs(prepared)
    config = identity["prepare_identity"]["config"]
    horizons = config["horizons_ns"]
    links = _build_execution_cancel_candidates(ex, cancellations, candidates,
        window_seconds=max(horizons) / 1_000_000_000, use_v3_clocks=True)
    population, audit = evaluate_cells(ex, candidates, links, cancellations, horizons, config["age_policies"])
    profiles = collect_profiles(root / "profiles", profile_scope(ex, population), identity, result, horizons, compute=True)
    population = add_priority_flags(attach_profiles(population, profiles), config["review_fraction"])
    population.write_parquet(root / "population.parquet")
    audit.write_parquet(root / "cancellation_audit.parquet")
    profiles.write_parquet(root / "profiles.parquet")
    write_json(root / "derive_receipt.json", {"status": "derived", "identity": identity,
               "rows": {"population": population.height, "audit": audit.height, "profiles": profiles.height}})
    seal_derived(root, identity)
    verify_derived(root, identity, prepared, result)
    return "derived"


def verify_baseline(population: pl.DataFrame, result: dict, config: dict) -> dict:
    if 1_000_000_000 not in config["horizons_ns"] or "legacy_age30" not in [p["id"] for p in config["age_policies"]]:
        return {"status": "not_in_design"}
    legacy = population.filter((pl.col("horizon_ns") == 1_000_000_000) & (pl.col("age_policy") == "legacy_age30"))
    source = pl.read_parquet(resolve(result["metadata"]["paths"]["execution_metrics"]))
    mapping = {"candidate_qty_raw": "candidate_deceptive_visible_qty_pre",
               "withdrawal_qty_raw": "matched_deceptive_cancel_visible_qty_window",
               "WMSCI_self": "withdrawal_profile_scale_event"}
    left = legacy.select(EXEC_KEYS + list(mapping)).sort(EXEC_KEYS)
    right = source.select(EXEC_KEYS + [pl.col(old).alias(new) for new, old in mapping.items()]).sort(EXEC_KEYS)
    if left.height != right.height or not left.select(EXEC_KEYS).equals(right.select(EXEC_KEYS)):
        raise ValueError("legacy H1 baseline identity mismatch")
    for a, b in zip(left.iter_rows(named=True), right.iter_rows(named=True), strict=True):
        for name in mapping:
            # The historical zero-candidate W is null; the new population uses 0.
            old = 0.0 if name == "WMSCI_self" and b[name] is None and b["candidate_qty_raw"] == 0 else b[name]
            if old is None or not math.isclose(a[name], old, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"legacy H1 baseline mismatch: {name}; lifecycle semantic changes require explicit review")
    return {"status": "verified", "rows_checked": left.height,
            "zero_candidate_W_null_normalized_to_zero": True}


def verify_analysis(output: Path, prepared: Path, identity: dict, result: dict, *, publish: bool) -> dict:
    from spoofing_detection.lob.withdrawal_multiscale import KEYS
    from spoofing_detection.lob.withdrawal_multiscale_verification import verify_accounting, verify_priorities
    root = output if output.exists() else analysis_stage(output)
    complete = (root / "COMPLETE.json").exists()
    if output.exists() and not complete:
        raise ValueError("unmarked final analysis must not be reused")
    if complete:
        marker = read_json(root / "COMPLETE.json")
        path = root / "artifact_manifest.json"
        check_binding(path, marker.get("artifact_manifest_sha256"))
        if marker.get("status") != "verified" or marker.get("version") != ANALYSIS_VERSION:
            raise ValueError("invalid analysis COMPLETE marker")
        manifest = read_json(path)
        expected = {"derived_manifest.json", "DERIVED.json", "verification.json", "identity.json"}
        if set(manifest) != {"version", "artifacts"} or manifest["version"] != ANALYSIS_VERSION or set(manifest["artifacts"]) != expected:
            raise ValueError("invalid final analysis manifest")
        for name in expected:
            check_binding(root / name, manifest["artifacts"][name])
    if read_json(root / "identity.json") != identity:
        raise ValueError("incompatible analysis identity")
    population, audit, profiles = verify_derived(root, identity, prepared, result)
    if complete:
        receipt = read_json(root / "verification.json")
        if receipt.get("status") != "verified" or receipt.get("identity") != identity:
            raise ValueError("invalid verification receipt")
    else:
        ex, candidates, cancellations = prepared_inputs(prepared)
        config = identity["prepare_identity"]["config"]
        accounting = verify_accounting(ex, candidates, cancellations, population, audit,
                                      config["horizons_ns"], config["age_policies"])
        priorities = verify_priorities(population, config["review_fraction"])
        baseline = verify_baseline(population, result, config)
        # Prove every saved profile field is joined without policy-dependent pruning.
        base = population.drop([c for c in profiles.columns if c not in KEYS + ["horizon_ns"]] + ["profile_status"])
        joined = attach_profiles(base, profiles)
        compare_exact(population, joined, profiles.columns + ["age_policy", "profile_status"],
                      KEYS + ["horizon_ns", "age_policy"], "profile join")
        sample = profile_scope(ex, population).sort(KEYS).head(PROFILE_BATCH_SIZE)
        if sample.height:
            recomputed = compute_profiles(sample, result, config["horizons_ns"])
            saved = profiles.join(sample.select(KEYS), on=KEYS, how="semi")
            compare_exact(saved, recomputed, profiles.columns, KEYS + ["horizon_ns"], "profile sample")
        receipt = {"status": "verified", "identity": identity, "accounting": accounting,
                   "priorities": priorities, "baseline": baseline,
                   "profiles": {"native_verified_at_derive": profiles.height,
                                "recomputed_execution_sample": sample.height,
                                "sample_rule": "first_32_sorted_full_keys", "independent_all_profiles": False}}
        write_json(root / "verification.json", receipt)
        names = ["derived_manifest.json", "DERIVED.json", "verification.json", "identity.json"]
        write_json(root / "artifact_manifest.json", {"version": ANALYSIS_VERSION,
                   "artifacts": {name: sha256(root / name) for name in names}})
        write_json(root / "COMPLETE.json", {"status": "verified", "version": ANALYSIS_VERSION,
                   "artifact_manifest_sha256": sha256(root / "artifact_manifest.json")})
    if publish and root != output:
        if output.exists():
            raise FileExistsError(output)
        os.rename(root, output)
        return verify_analysis(output, prepared, identity, result, publish=False)
    return receipt


def publish_report(analysis: Path, output: Path, config: dict, config_path: Path, result: dict) -> dict:
    from spoofing_detection.lob.withdrawal_multiscale_reporting import build_tables
    if not analysis.exists() or not (analysis / "COMPLETE.json").exists():
        raise ValueError("report requires a verified final analysis")
    identity = read_json(analysis / "identity.json")
    prepared = Path(identity["prepare_manifest"]["path"]).parent
    expected = analysis_identity(prepared, prepare_identity(config, config_path, result), result)
    verify_analysis(analysis, prepared, expected, result, publish=False)
    for source in (analysis, prepared):
        if output == source or source in output.parents or output in source.parents:
            raise ValueError("report output must not overlap analysis or prepared inputs")
    if output.exists():
        raise FileExistsError(output)
    population = pl.read_parquet(analysis / "population.parquet")
    tables = build_tables(population)
    output.parent.mkdir(parents=True, exist_ok=True)
    private = output.with_name(f".{output.name}.attempt-{uuid.uuid4().hex}")
    private.mkdir()
    for name, frame in tables.items():
        frame.write_csv(private / f"{name}.csv")
    provenance = {"status": "report_verified", "instrument": result["instrument"],
                  "analysis_manifest": check_binding(analysis / "artifact_manifest.json", sha256(analysis / "artifact_manifest.json")),
                  "config": check_binding(config_path, sha256(config_path)),
                  "report_code": source_code_bindings(),
                  "table_rows": {name: frame.height for name, frame in tables.items()}}
    write_json(private / "provenance.json", provenance)
    lines = ["# Private multiscale withdrawal report", "",
        f"Instrument: {result['instrument']}. Status: exploratory; not promoted to manuscript.", "",
        "W_self = log1p(P_g/Q) log1p(U_g/Q) U_g/P_g; W_market uses U_g/D instead.",
        "D is all-participant opposite-side top-10 visible depth before the first fill, frozen.",
        "P_g/U_g share frozen baseline lifecycle age weights; raw quantities remain available.",
        "Withdrawals are gross explicit cancellations, not net pressure or manipulative intent.",
        "The two scores are dependent: W_market = W_self P_g/D.", "",
        "## Counts by horizon and age policy", "",
        "Each row is a separate specification; do not sum cells as independent observations.", ""]
    for row in tables["population_counts"].iter_rows(named=True):
        lines.append(json.dumps(row, sort_keys=True, allow_nan=False))
    lines.extend(["", "## Review and support", "",
        "review.csv retains every raw candidate, including outside both deciles and with missing J.",
        "Deciles only set review-priority flags. Price signs and U/Q do not select candidates.",
        "common_support_counts.csv uses the union of candidates at any horizon, without requiring withdrawal at every horizon.",
        "The represented-event witness does not certify external feed completeness. Clock common-origin remains an operational assumption.",
        "External CONSOB alerts are not independent manipulation labels. No accuracy, prevalence or causality is inferred.", ""])
    (private / "report.md").write_text("\n".join(lines))
    names = [f"{name}.csv" for name in tables] + ["report.md", "provenance.json"]
    write_json(private / "artifact_manifest.json", {"version": "withdrawal_multiscale_report_v1",
               "artifacts": {name: sha256(private / name) for name in names}})
    write_json(private / "COMPLETE.json", {"status": "report_verified",
               "artifact_manifest_sha256": sha256(private / "artifact_manifest.json")})
    verify_report(private)
    if output.exists():
        raise FileExistsError(output)
    os.rename(private, output)
    return verify_report(output)


def verify_report(root: Path) -> dict:
    marker = read_json(root / "COMPLETE.json")
    check_binding(root / "artifact_manifest.json", marker.get("artifact_manifest_sha256"))
    if marker.get("status") != "report_verified":
        raise ValueError("invalid report completion marker")
    manifest = read_json(root / "artifact_manifest.json")
    if manifest.get("version") != "withdrawal_multiscale_report_v1":
        raise ValueError("invalid report version")
    for name, digest in manifest["artifacts"].items():
        check_binding(root / name, digest)
    receipt = read_json(root / "provenance.json")
    for name, rows in receipt["table_rows"].items():
        # Row-count verification must not coerce original numeric-looking IDs.
        if pl.read_csv(root / f"{name}.csv", infer_schema=False).height != rows:
            raise ValueError("report row count mismatch")
    return receipt


def assert_output_disjoint(output: Path, config_path: Path, result: dict) -> None:
    """Never quarantine an input file or a directory containing bound inputs."""
    output = output.resolve()
    protected = [config_path.resolve(), Path(__file__).resolve()]
    protected.extend(Path(binding["path"]).resolve() for binding in result["bindings"].values())
    protected.extend(path.resolve() for path in (ROOT / "src/spoofing_detection/lob").glob("*.py"))
    for source in protected:
        if output == source or output in source.parents:
            raise ValueError(f"output/input overlap: {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare")
    prepare.add_argument("--config", type=Path, required=True)
    prepare.add_argument("--instrument", choices=["FERRARI", "NEXI", "RISANAMENTO"], required=True)
    prepare.add_argument("--output-root", type=Path, required=True)
    prepare.add_argument("--validate-only", action="store_true")
    prepare.add_argument("--resume", action="store_true")
    report = sub.add_parser("report")
    report.add_argument("--config", type=Path, required=True)
    report.add_argument("--instrument", choices=["FERRARI", "NEXI", "RISANAMENTO"], required=True)
    report.add_argument("--analysis-root", type=Path, required=True)
    report.add_argument("--output-root", type=Path, required=True)
    for command in ("derive", "verify"):
        analysis = sub.add_parser(command)
        analysis.add_argument("--config", type=Path, required=True)
        analysis.add_argument("--instrument", choices=["FERRARI", "NEXI", "RISANAMENTO"], required=True)
        analysis.add_argument("--prepare-root", type=Path, required=True)
        analysis.add_argument("--output-root" if command == "derive" else "--analysis-root", type=Path, required=True)
        if command == "derive":
            analysis.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    config = read_json(args.config)
    result = preflight(config, args.instrument)
    target = args.analysis_root if args.command == "verify" else args.output_root
    assert_output_disjoint(target, args.config, result)
    if args.command == "report":
        receipt = publish_report(args.analysis_root.resolve(), args.output_root.resolve(), config, args.config, result)
        print(json.dumps({"status": receipt["status"], "instrument": args.instrument,
                          "output_root": str(args.output_root.resolve()), "table_rows": receipt["table_rows"]}))
        return 0
    if args.command != "prepare":
        prepared = args.prepare_root.resolve()
        identity = analysis_identity(prepared, prepare_identity(config, args.config, result), result)
        output = (args.output_root if args.command == "derive" else args.analysis_root).resolve()
        if output == prepared or prepared in output.parents or output in prepared.parents:
            raise ValueError("analysis output must not overlap prepared inputs")
        if args.command == "derive":
            status = derive_analysis(output, prepared, identity, result, resume=args.resume)
        else:
            status = verify_analysis(output, prepared, identity, result, publish=True)["status"]
        print(json.dumps({"status": status, "instrument": args.instrument, "output_root": str(output)}))
        return 0
    output = args.output_root.resolve()
    identity = prepare_identity(config, args.config, result)
    exists = output.exists()
    if exists and not args.resume:
        raise FileExistsError(output)
    complete = exists and (output / "COMPLETE.json").exists()
    if complete:
        verify_prepared(output, identity)
    if args.validate_only:
        print(json.dumps({"status": result["status"], "instrument": args.instrument,
                          "verified_input_count": len(result["bindings"])}))
        return 0
    if complete:
        print(json.dumps({"status": "reused", "instrument": args.instrument, "output_root": str(output)}))
        return 0
    if exists:
        interrupted = output.with_name(f"{output.name}.interrupted-{uuid.uuid4().hex}")
        os.rename(output, interrupted)
    receipt = publish_prepare(output, result, identity)
    print(json.dumps({"status": receipt["status"], "instrument": args.instrument,
                      "output_root": str(output), "rows": receipt["rows"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
