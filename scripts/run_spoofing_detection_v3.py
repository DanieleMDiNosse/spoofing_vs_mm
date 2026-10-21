#!/usr/bin/env python
"""Create and execute one isolated, policy-bound v3 detector bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from spoofing_detection.lob.empirical_controls_v2_budget import supervise

RUN_CONFIG_PATH = REPO_ROOT / "configs" / "spoofing_detection_v3_rerun.json"
CLOCK_AUDIT_ROOT = REPO_ROOT / "outputs" / "event_clock_fields_audit_v3_20260927"
V3_CLOCK_POLICY_ID = "spoofing_detection_sequence_replay_trade_fill_sequence_cancel_v3_utc_provisional"
V3_CLOCK_CONTRACT = {
    "replay_clock": "SEQUENCETIME",
    "duration_clock": "SEQUENCETIME",
    "cancellation_removal_clock": "SEQUENCETIME",
    "fill_anchor_clock": "TRADETIME",
    "timestamp_representation": "UTC epoch nanoseconds",
    "semantic_fallback": "forbidden",
    "interval_convention": "left_closed_right_open",
}


@dataclass(frozen=True)
class PreparedRun:
    instrument: str
    run_root: Path
    bundle_dir: Path
    manifest_path: Path
    log_path: Path
    command: list[str]
    resource_budget: dict[str, int]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def _load_spec(instrument: str) -> tuple[dict[str, Any], dict[str, Any], Path]:
    run_config = _load_json(RUN_CONFIG_PATH)
    if run_config.get("policy_id") != V3_CLOCK_POLICY_ID or run_config.get("clock_contract") != V3_CLOCK_CONTRACT:
        raise ValueError("v3 rerun config does not declare the exact detector clock policy")
    allowed = run_config.get("instruments")
    if instrument not in allowed:
        raise ValueError(f"unknown instrument: {instrument}")
    registry_path = _resolve_repo_path(str(run_config["source_registry"]))
    registry = _load_json(registry_path)
    source = registry.get("baseline_runs", {}).get(instrument)
    if not isinstance(source, dict):
        raise ValueError(f"missing source registry entry for {instrument}")
    detector_config = _resolve_repo_path(str(run_config["detector_config"]))
    detector_payload = _load_json(detector_config)
    expected_config = {
        "policy_id": V3_CLOCK_POLICY_ID,
        "contract": V3_CLOCK_CONTRACT,
    }
    if detector_payload.get("spoofing_detection_clock_policy_v3") != expected_config:
        raise ValueError("detector config does not declare the exact v3 clock policy")
    return source, run_config, detector_config


def _source_hashes(source: dict[str, Any]) -> dict[str, str]:
    paths = source.get("source_paths")
    expected = source.get("source_hashes")
    if not isinstance(paths, dict) or not isinstance(expected, dict):
        raise ValueError("source registry must declare paths and hashes")
    checked: dict[str, str] = {}
    for name, value in paths.items():
        path = _resolve_repo_path(str(value))
        if not path.is_file():
            raise ValueError(f"required source file is missing: {path}")
        actual = _sha256(path)
        expected_hash = expected.get(f"{name}_sha256")
        if actual != expected_hash:
            raise ValueError(f"source hash mismatch: {name}")
        checked[f"{name}_sha256"] = actual
    baseline_metadata = _resolve_repo_path(str(source.get("baseline_metadata", "")))
    expected_metadata_hash = expected.get("baseline_metadata_sha256")
    if not baseline_metadata.is_file():
        raise ValueError(f"required baseline metadata is missing: {baseline_metadata}")
    metadata_hash = _sha256(baseline_metadata)
    if metadata_hash != expected_metadata_hash:
        raise ValueError("source hash mismatch: baseline_metadata")
    checked["baseline_metadata_sha256"] = metadata_hash
    return checked


def _resource_budget(run_config: dict[str, Any], instrument: str) -> dict[str, int]:
    budget = run_config.get("resource_budgets", {}).get(instrument)
    if not isinstance(budget, dict) or set(budget) != {"max_rss_mb", "timeout_seconds"}:
        raise ValueError(f"explicit RSS and timeout budget required for {instrument}")
    if any(type(value) is not int or value <= 0 for value in budget.values()):
        raise ValueError(f"positive integer resource budget required for {instrument}")
    return budget


def _clock_preflight(instrument: str, source: dict[str, Any], hashes: dict[str, str]) -> dict[str, Any]:
    """Bind the current source and UTC semantics to a completed all-event audit."""
    provenance = _load_json(CLOCK_AUDIT_ROOT / "provenance.json")
    audit_script = REPO_ROOT / "scripts" / "audit_event_clock_fields.py"
    if provenance.get("script_sha256") != _sha256(audit_script):
        raise ValueError("clock audit producer changed; rerun the all-event audit")
    if provenance.get("config_sha256") != _sha256(REPO_ROOT / "configs" / "spoofing_empirical_controls.json"):
        raise ValueError("clock audit source registry changed")
    if "SEQUENCETIME/BOOKIN/BOOKOUTTIME: documented UTC epoch nanoseconds" not in provenance.get("timestamp_representation", ""):
        raise ValueError("clock audit does not establish documented UTC epoch-ns source representation")
    evidence = provenance["sources"][instrument]
    if (evidence["source_sha256"] != hashes["raw_events_sha256"] or
            provenance["selected_event_dates"][instrument] != source["selected_event_dates"]):
        raise ValueError("clock audit source or selected-date mismatch")
    rows = evidence["selected_rows"]
    sequence = evidence["parse_counts"]["SEQUENCETIME"]
    if rows < 1 or sequence != {"present": rows, "missing": 0, "malformed": 0}:
        raise ValueError("mandatory replay clock is missing or malformed")
    availability = pl.read_csv(CLOCK_AUDIT_ROOT / "availability.csv").filter(pl.col("instrument") == instrument)
    dates = set(source["selected_event_dates"])
    if set(availability["event_date"].to_list()) != dates:
        raise ValueError("clock audit date coverage differs from selected dates")
    required = availability.filter((pl.col("field") == "SEQUENCETIME") | ((pl.col("event_label") == "Fill") & (pl.col("field") == "TRADETIME")))
    if (required.filter(pl.col("field") == "SEQUENCETIME").select(pl.sum("total_rows")).item() != rows or
            required.filter((pl.col("event_label") == "Fill") & (pl.col("field") == "TRADETIME")).is_empty()):
        raise ValueError("mandatory event-clock coverage is incomplete")
    if required.select(pl.sum("missing", "malformed")).row(0) != (0, 0):
        raise ValueError("mandatory event clocks missing or malformed")
    fills = required.filter((pl.col("event_label") == "Fill") & (pl.col("field") == "TRADETIME")).select(pl.sum("total_rows")).item()
    if fills != evidence["parse_counts"]["TRADETIME"]["present"]:
        raise ValueError("fill anchor count differs from audit provenance")
    ordering = pl.read_csv(CLOCK_AUDIT_ROOT / "ordering.csv").filter((pl.col("instrument") == instrument) & (pl.col("field") == "SEQUENCETIME"))
    if (ordering.is_empty() or set(ordering["TRADEDATE"].to_list()) != dates or
            set(ordering["comparison"].to_list()) != {"adjacent_regressions", "high_watermark_regressions"} or
            ordering.select(pl.sum("regressions")).item() != 0):
        raise ValueError("canonical replay sequence clock regresses")
    pairwise = pl.read_csv(CLOCK_AUDIT_ROOT / "pairwise.csv").filter(
        (pl.col("instrument") == instrument) & (pl.col("event_label") == "Fill") &
        (pl.col("left_field") == "TRADETIME") & (pl.col("right_field") == "SEQUENCETIME")
    )
    fill_dates = set(required.filter((pl.col("event_label") == "Fill") & (pl.col("field") == "TRADETIME"))["event_date"].to_list())
    if (pairwise.is_empty() or set(pairwise["event_date"].to_list()) != fill_dates or
            pairwise.select(pl.sum("comparable_rows")).item() != fills or
            pairwise.select(pl.sum("later", "equal")).row(0) != (0, 0)):
        raise ValueError("fill anchor does not precede replay time")
    return {"status": "input_clock_preflight_passed", "instrument": instrument,
            "selected_event_dates": source["selected_event_dates"], "selected_rows": rows,
            "fill_rows": fills, "audit_provenance_sha256": _sha256(CLOCK_AUDIT_ROOT / "provenance.json"),
            "audit_table_sha256": {name: _sha256(CLOCK_AUDIT_ROOT / f"{name}.csv")
                                   for name in ("availability", "ordering", "pairwise")},
            "audit_script_sha256": provenance["script_sha256"],
            "raw_events_sha256": hashes["raw_events_sha256"],
            "timestamp_representation": provenance["timestamp_representation"],
            "clock_contract": V3_CLOCK_CONTRACT,
            "limitation": "Source audit checks field parsing/order, not independent proof of TRADETIME timezone or detector output validity."}


def prepare_run(run_root: Path, instrument: str) -> PreparedRun:
    run_root = Path(run_root)
    if run_root.exists():
        raise ValueError(f"run root must not already exist: {run_root}")
    source, run_config, detector_config = _load_spec(instrument)
    resource_budget = _resource_budget(run_config, instrument)
    source_hashes = _source_hashes(source)
    preflight = _clock_preflight(instrument, source, source_hashes)
    detector_payload = _load_json(detector_config)
    metrics = detector_payload.get("metrics")
    if not isinstance(metrics, dict):
        raise ValueError("detector config metrics section is required")
    metrics["empirical_depth_kernel"] = str(_resolve_repo_path(source["source_paths"]["empirical_depth_kernel"]))
    # Atomic claim: a concurrent invocation must not enter this bundle.
    try:
        run_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError(f"run root must not already exist: {run_root}") from exc
    bundle_dir = run_root / "bundles" / instrument
    manifest_path = run_root / "manifests" / f"{instrument}.json"
    log_path = run_root / "logs" / f"{instrument}.json"
    config_copy = run_root / "configs" / f"{instrument}_spoofing_detection_v3.json"
    preflight_path = run_root / "preflight" / f"{instrument}.json"
    for directory in (bundle_dir, manifest_path.parent, log_path.parent, config_copy.parent, preflight_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    preflight_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n")
    config_copy.write_text(json.dumps(detector_payload, indent=2, sort_keys=True) + "\n")
    paths = source["source_paths"]
    command = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "compute_spoofing_metrics.py"),
        "--config", str(config_copy),
        "--input", str(_resolve_repo_path(paths["raw_events"])),
        "--quote-panel", str(_resolve_repo_path(paths["quote_panel"])),
        "--output-dir", str(bundle_dir),
        "--clock-policy-v3",
    ]
    manifest = {
        "status": "running",
        "instrument": instrument,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "clock_policy_id": V3_CLOCK_POLICY_ID,
        "clock_contract": V3_CLOCK_CONTRACT,
        "selected_event_dates": source["selected_event_dates"],
        "source_paths": paths,
        "source_hashes": source_hashes,
        "resource_budget": resource_budget,
        "preflight_sha256": _sha256(preflight_path),
        "detector_config_sha256": _sha256(config_copy),
        "program_sha256": {
            name: _sha256(REPO_ROOT / "scripts" / name)
            for name in ("compute_spoofing_metrics.py", "validate_spoofing_detection_bundle.py")
        },
        "command": command,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return PreparedRun(instrument, run_root, bundle_dir, manifest_path, log_path, command, resource_budget)


def assert_bundle_provenance(metadata: dict[str, Any], manifest: dict[str, Any]) -> None:
    """Reject a bundle whose recorded inputs differ from the prepared manifest."""
    input_hashes = metadata.get("input_hashes")
    if not isinstance(input_hashes, dict):
        raise ValueError("bundle metadata input_hashes mapping is required")
    source_hashes = manifest.get("source_hashes")
    if not isinstance(source_hashes, dict):
        raise ValueError("prepared manifest source_hashes mapping is required")
    expected = {
        "raw_events_sha256": source_hashes.get("raw_events_sha256"),
        "quote_panel_sha256": source_hashes.get("quote_panel_sha256"),
        "empirical_depth_kernel_sha256": source_hashes.get("empirical_depth_kernel_sha256"),
        "config_sha256": manifest.get("detector_config_sha256"),
    }
    for key, value in expected.items():
        if not isinstance(value, str) or input_hashes.get(key) != value:
            raise ValueError(f"bundle input provenance mismatch: {key}")


def run(prepared: PreparedRun) -> None:
    manifest = _load_json(prepared.manifest_path)
    try:
        for name, expected in manifest["program_sha256"].items():
            if _sha256(REPO_ROOT / "scripts" / name) != expected:
                raise ValueError(f"program changed since preparation: {name}")
        preflight_path = prepared.run_root / "preflight" / f"{prepared.instrument}.json"
        if _sha256(preflight_path) != manifest["preflight_sha256"]:
            raise ValueError("clock preflight changed since preparation")
        recorded = _load_json(preflight_path)
        for name, expected_hash in recorded["audit_table_sha256"].items():
            if _sha256(CLOCK_AUDIT_ROOT / f"{name}.csv") != expected_hash:
                raise ValueError(f"clock audit table changed since preparation: {name}")
        if _sha256(CLOCK_AUDIT_ROOT / "provenance.json") != recorded["audit_provenance_sha256"]:
            raise ValueError("clock audit provenance changed since preparation")
        completed = supervise(
            prepared.command, cwd=REPO_ROOT,
            max_rss_mb=prepared.resource_budget["max_rss_mb"],
            timeout_seconds=prepared.resource_budget["timeout_seconds"],
            diagnostics_path=prepared.run_root / "logs" / f"{prepared.instrument}_supervision.json",
        )
        log = {"returncode": completed["returncode"], "stdout": completed["output"],
               "peak_rss_mb": completed["peak_rss_mb"], "wall_seconds": completed["wall_seconds"]}
        prepared.log_path.write_text(json.dumps(log, indent=2, sort_keys=True))
        validator = [sys.executable, str(REPO_ROOT / "scripts" / "validate_spoofing_detection_bundle.py"), str(prepared.bundle_dir)]
        checked = supervise(
            validator, cwd=REPO_ROOT,
            max_rss_mb=prepared.resource_budget["max_rss_mb"],
            timeout_seconds=min(prepared.resource_budget["timeout_seconds"], 1800),
            diagnostics_path=prepared.run_root / "logs" / f"{prepared.instrument}_validation_supervision.json",
        )
        log["validator_returncode"] = checked["returncode"]
        log["validator_stdout"] = checked["output"]
        log["validator_peak_rss_mb"] = checked["peak_rss_mb"]
        log["validator_wall_seconds"] = checked["wall_seconds"]
        prepared.log_path.write_text(json.dumps(log, indent=2, sort_keys=True))
        for name, expected in manifest["program_sha256"].items():
            if _sha256(REPO_ROOT / "scripts" / name) != expected:
                raise ValueError(f"program changed during run: {name}")
        metadata = _load_json(prepared.bundle_dir / "metadata.json")
        assert_bundle_provenance(metadata, manifest)
        manifest.update(
            status="complete", row_counts=metadata.get("row_counts"),
            artifact_hashes=metadata.get("artifact_hashes"),
        )
    except BaseException as exc:
        manifest["status"] = "failed"
        prepared.log_path.write_text(json.dumps({"failure_type": type(exc).__name__, "failure": str(exc)[-10000:]}, indent=2))
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        raise RuntimeError(f"v3 detector run failed for {prepared.instrument}; inspect {prepared.log_path}") from exc
    finally:
        prepared.manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one isolated v3 spoofing-detection instrument bundle.")
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--instrument", required=True, choices=("NEXI", "FERRARI", "RISANAMENTO"))
    parser.add_argument("--prepare-only", action="store_true", help="Create the immutable run manifest without running the detector.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    prepared = prepare_run(args.run_root, args.instrument)
    if not args.prepare_only:
        run(prepared)
    print(prepared.bundle_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
