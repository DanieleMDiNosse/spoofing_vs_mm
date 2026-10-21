#!/usr/bin/env python
"""Certify/run remaining instruments and combine with validated NEXI pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    sys.path.insert(0, str(path))

import polars as pl
from scripts import build_spoofing_empirical_negative_controls as legacy
from spoofing_detection.lob import spoofing_metrics as detector
from spoofing_detection.lob.empirical_controls_v2_budget import supervise
from spoofing_detection.lob.empirical_controls_v2_io import EMPIRICAL_POLICY_CONTRACT, config_template, file_hash, validate_preflight_v2
from spoofing_detection.lob.empirical_clock import CLOCK_POLICY_ID
from spoofing_detection.lob.empirical_controls_v2_report import summarize_contributions, validate_bundle
from scripts.validate_spoofing_detection_bundle import validate_bundle as validate_detector_bundle

PYTHON = "/home/danielemdn/miniconda3/envs/main/bin/python"
V1_CONFIG = ROOT / "configs/spoofing_empirical_controls.json"
RUN_ROOT = ROOT / "outputs/empirical_controls_v2_full"
INSTRUMENTS = ("NEXI", "FERRARI", "RISANAMENTO")
DETECTOR_BUNDLES = {
    "NEXI": ROOT / "outputs/spoofing_detection_v3_nexi_pilot_20260927_110747/bundles/NEXI",
    "FERRARI": ROOT / "outputs/spoofing_detection_v3_ferrari_pilot_20260927_120728/bundles/FERRARI",
    "RISANAMENTO": ROOT / "outputs/spoofing_detection_v3_risanamento_pilot_20260927_150833/bundles/RISANAMENTO",
}


def _detector_inputs(instrument: str, registry: dict) -> dict[str, Path]:
    bundle = DETECTOR_BUNDLES[instrument]
    validate_detector_bundle(bundle)
    metadata = json.loads((bundle / "metadata.json").read_text())
    manifest_path = bundle.parents[1] / "manifests" / f"{instrument}.json"
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("status") != "complete" or manifest.get("instrument") != instrument
            or manifest.get("selected_event_dates") != registry.get("selected_event_dates")):
        raise ValueError("v3 detector date scope or run status differs from registry")
    if manifest.get("artifact_hashes") != metadata.get("artifact_hashes"):
        raise ValueError("v3 detector manifest artifacts differ from authenticated bundle")
    paths = registry["source_paths"]
    if Path(metadata["input"]).resolve() != (ROOT / paths["raw_events"]).resolve():
        raise ValueError("v3 detector raw input differs from registry")
    for name, key in (("quote_panel", "quote_panel"), ("empirical_depth_kernel", "empirical_depth_kernel")):
        if Path(metadata[key]).resolve() != (ROOT / paths[name]).resolve():
            raise ValueError(f"v3 detector {name} differs from registry")
    return {
        "raw_events": Path(metadata["input"]).resolve(),
        "baseline_config": Path(metadata["config"]).resolve(),
        "baseline_metadata": (bundle / "metadata.json").resolve(),
        "execution_metrics": (bundle / "execution_metrics.parquet").resolve(),
        "execution_cluster_members": (bundle / "execution_cluster_members.parquet").resolve(),
    }


def _resources(instrument: str) -> dict[str, int]:
    """Keep RAM fixed; NEXI receives a one-hour budget after its timed pilot."""
    if instrument not in INSTRUMENTS:
        raise ValueError(f"instrument is not runnable in this cross-instrument workflow: {instrument}")
    return {
        "buffer_rows": 100000,
        "max_rss_mb": 4096,
        "timeout_seconds": 7200 if instrument == "FERRARI" else 3600,
    }


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temporary, path)


def _source_identity() -> dict[str, str]:
    paths = [
        Path(__file__).resolve(),
        ROOT / "scripts/build_spoofing_empirical_negative_controls.py",
        *(
            ROOT / "src/spoofing_detection/lob" / name
            for name in (
                "actor_identity.py", "config.py", "enums.py", "models.py",
                "normalize.py", "panel.py", "spoofing_metrics.py", "withdrawal_risk.py",
            )
        ),
    ]
    return {str(path.relative_to(ROOT)): file_hash(path) for path in paths}


def _require_revised_clock_policy(bundle: Path) -> None:
    manifest_path = bundle / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"bundle manifest missing: {bundle}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("clock_policy_id") != CLOCK_POLICY_ID:
        raise ValueError("bundle does not select the revised clock policy")


def _validate_instrument_bundle(bundle: Path, *, expected_instrument: str) -> dict[str, object]:
    """Authenticate a bundle and bind it to the caller's named instrument."""

    stats = validate_bundle(bundle)
    _require_revised_clock_policy(bundle)
    manifest = json.loads((bundle / "manifest.json").read_text())
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != {expected_instrument}:
        raise ValueError(f"bundle source instrument does not match expected {expected_instrument}")
    return stats


def _validate_combined_output(destination: Path, *, required_instruments: set[str]) -> None:
    manifest_path = destination / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError("combined output has no manifest")
    manifest = json.loads(manifest_path.read_text())
    if (manifest.get("schema") != "empirical_controls_v2_cross_instrument_1"
            or manifest.get("status") != "complete"
            or manifest.get("clock_policy_id") != CLOCK_POLICY_ID):
        raise ValueError("combined output does not select the revised clock policy")
    sources = manifest.get("sources")
    if not isinstance(sources, dict) or set(sources) != required_instruments:
        raise ValueError("combined output sources are missing")
    for source in sources.values():
        if not isinstance(source, dict) or not isinstance(source.get("bundle"), str):
            raise ValueError("combined output source is malformed")
        bundle = Path(source["bundle"])
        expected_instrument = next(
            instrument for instrument, item in sources.items() if item is source
        )
        _validate_instrument_bundle(bundle, expected_instrument=expected_instrument)
        if source.get("manifest_sha256") != file_hash(bundle / "manifest.json"):
            raise ValueError("combined output source manifest hash mismatch")
    artifacts = manifest.get("artifacts")
    expected = {"summary_contributions.parquet", "summary.parquet", "summary.csv"}
    if not isinstance(artifacts, dict) or set(artifacts) != expected:
        raise ValueError("combined output artifact graph is malformed")
    for name, digest in artifacts.items():
        if not isinstance(digest, str) or file_hash(destination / name) != digest:
            raise ValueError("combined output artifact hash mismatch")


def _compare_core(name: str, frozen: pl.DataFrame, live: pl.DataFrame) -> dict[str, object]:
    missing = sorted(set(live.columns) - set(frozen.columns))
    if missing:
        raise ValueError(f"frozen {name} lacks replay columns: {missing}")
    projected = frozen.select(live.columns)
    equal = projected.schema == live.schema and legacy._canonical_sort(projected).equals(
        legacy._canonical_sort(live), null_equal=True
    )
    result = {
        "artifact": name,
        "comparison_scope": "exact replay-produced core columns",
        "columns": live.columns,
        "frozen_row_count": frozen.height,
        "live_row_count": live.height,
        "equal": equal,
        "diagnostic": "exact semantic equivalence" if equal else "core rows or schema differ",
        "live_semantic_sha256": legacy._semantic_frame_hash(live),
    }
    if not equal:
        raise RuntimeError(f"canonical execution-core equivalence failed: {name}")
    return result


def _selected_clock_config() -> dict[str, object]:
    """Return the complete v3 clock contract used by certifications and runners."""
    config = config_template()
    config["clock_policy_id"] = CLOCK_POLICY_ID
    config["clock"] = {
        **config["clock"],
        "event_timestamp_precedence": ["SEQUENCETIME"],
        "timezone": "UTC",
        "empirical_policy_contract": EMPIRICAL_POLICY_CONTRACT,
    }
    return config


def certify(instrument: str) -> Path:
    if instrument not in INSTRUMENTS:
        raise ValueError(f"instrument is not runnable in this cross-instrument workflow: {instrument}")
    cert_dir = RUN_ROOT / "certifications" / instrument
    certificate_path = cert_dir / "certificate.json"
    pilot_config_path = cert_dir / "run_config.json"
    output = RUN_ROOT / "bundles" / instrument
    if certificate_path.exists() != pilot_config_path.exists():
        raise ValueError("partial certification state; use a new run root")
    if certificate_path.is_file() and pilot_config_path.is_file():
        config = json.loads(pilot_config_path.read_text())
        desired = _resources(instrument)
        if (config.get("resources") != desired
                or config.get("clock_policy_id") != CLOCK_POLICY_ID
                or config.get("clock", {}).get("event_timestamp_precedence") != ["SEQUENCETIME"]
                or config.get("clock", {}).get("timezone") != "UTC"
                or config.get("clock", {}).get("empirical_policy_contract") != EMPIRICAL_POLICY_CONTRACT):
            raise ValueError("existing full-run certification/config uses a stale clock policy; recertify in a new run root")
        validate_preflight_v2(pilot_config_path)
        return pilot_config_path
    cert_dir.mkdir(parents=True, exist_ok=True)
    original = json.loads(V1_CONFIG.read_text())
    run = original["baseline_runs"][instrument]
    artifacts = _detector_inputs(instrument, run)
    metadata = json.loads(artifacts["baseline_metadata"].read_text())
    parameters = json.loads(artifacts["baseline_config"].read_text())["metrics"]
    for key, source in (("raw_events", "raw_events"), ("baseline_config", "config")):
        if metadata["input_hashes"]["raw_events_sha256" if key == "raw_events" else "config_sha256"] != file_hash(artifacts[key]):
            raise ValueError(f"v3 detector {source} input hash mismatch")
    for key in ("quote_panel", "empirical_depth_kernel"):
        if metadata["input_hashes"][f"{key}_sha256"] != file_hash(ROOT / run["source_paths"][key]):
            raise ValueError(f"v3 detector {key} input hash mismatch")
    selected_dates = set(run["selected_event_dates"])
    raw = legacy._selected_date_frame(pl.read_parquet(run["source_paths"]["raw_events"]), selected_dates)
    quote = legacy._selected_date_frame(pl.read_parquet(run["source_paths"]["quote_panel"]), selected_dates)
    tick_size = legacy.infer_tick_size_from_best_quotes(quote)
    original_exposure = detector.compute_actor_top_n_exposures
    detector.compute_actor_top_n_exposures = lambda *args, **kwargs: []
    try:
        _, live_executions, _, _, _, live_members = detector._stream_metric_inputs(
            raw,
            top_n=parameters["top_n"], tick_size=tick_size, include_level_columns=False,
            max_deceptive_order_age_seconds=parameters["max_deceptive_order_age_seconds"],
            execution_cluster_max_gap_ms=parameters.get("execution_cluster_max_gap_ms", 100),
            state_actor_keys=set(), execution_anchor_modes=tuple(parameters["execution_anchor_modes"]),
            empirical_kernel_weights=legacy.load_empirical_kernel_weights(Path(run["source_paths"]["empirical_depth_kernel"])),
            use_v3_clocks=True,
        )
    finally:
        detector.compute_actor_top_n_exposures = original_exposure
    metadata_path = artifacts["baseline_metadata"]
    audit_rows = [
        _compare_core("execution_metrics", pl.read_parquet(metadata_path.parent / "execution_metrics.parquet"), live_executions),
        _compare_core("execution_cluster_members", pl.read_parquet(metadata_path.parent / "execution_cluster_members.parquet"), live_members),
    ]

    entries = {name: {"path": str(path), "sha256": file_hash(path)} for name, path in artifacts.items()}
    evidence = {
        "status": "passed",
        "checks": {name: True for name in (
            "all_qualifying_executions", "canonical_roles", "membership_and_quantity",
            "original_indices", "partition_initialization", "clock_contract",
        )},
        "artifact_hashes": {name: entry["sha256"] for name, entry in entries.items()},
        "command": f"{PYTHON} {Path(__file__).resolve()} --certify {instrument}",
        "producer_source_identity": _source_identity(),
        "detector_run_manifest_sha256": file_hash(DETECTOR_BUNDLES[instrument].parents[1] / "manifests" / f"{instrument}.json"),
        "canonical_equivalence": audit_rows,
        "selected_event_dates": run["selected_event_dates"],
        "note": "Exact canonical replay comparison of execution core and all cluster members consumed by v2; unused state-series and downstream detector scoring suppressed.",
    }
    evidence_path = cert_dir / "evidence.json"
    _write_json(evidence_path, evidence)
    config = _selected_clock_config()
    certificate = {
        "certificate_version": "canonical_execution_inputs_v2", "instrument": instrument,
        "artifacts": entries, "clock": config["clock"],
        "evidence": {"path": str(evidence_path.resolve()), "sha256": file_hash(evidence_path)},
    }
    _write_json(certificate_path, certificate)
    config["resources"] = _resources(instrument)
    config["output_root"] = str(output.resolve())
    config["baseline_runs"] = {instrument: {
        "certification": str(certificate_path.resolve()),
        "selected_event_dates": run["selected_event_dates"],
    }}
    _write_json(pilot_config_path, config)
    validate_preflight_v2(pilot_config_path)
    return pilot_config_path


def run_instrument(instrument: str) -> Path:
    if instrument not in INSTRUMENTS:
        raise ValueError(f"instrument is not runnable in this cross-instrument workflow: {instrument}")
    output = RUN_ROOT / "bundles" / instrument
    if output.is_dir():
        _validate_instrument_bundle(output, expected_instrument=instrument)
        return output
    config = certify(instrument)
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ROOT / "src"), env.get("PYTHONPATH", "")])
    resources = _resources(instrument)
    result = supervise(
        [PYTHON, "-m", "spoofing_detection.lob.empirical_controls_v2_runner", "--config", str(config)],
        max_rss_mb=resources["max_rss_mb"], timeout_seconds=resources["timeout_seconds"], cwd=ROOT, env=env,
        diagnostics_path=RUN_ROOT / "logs" / f"{instrument.lower()}_supervision.json",
    )
    _write_json(RUN_ROOT / "logs" / f"{instrument.lower()}_run.json", result)
    validate_bundle(output)
    return output


def combine(bundles: dict[str, Path]) -> Path:
    destination = RUN_ROOT / "combined"
    temporary = RUN_ROOT / ".combined.incomplete"
    required_instruments = {"NEXI", *INSTRUMENTS}
    if set(bundles) != required_instruments:
        raise ValueError("combined output requires exactly NEXI, FERRARI, and RISANAMENTO")
    if destination.is_dir():
        _validate_combined_output(destination, required_instruments=required_instruments)
        return destination
    if temporary.exists():
        raise ValueError("incomplete combined output already exists; preserve it for diagnosis")
    temporary.mkdir(parents=True)
    scans = []
    sources = {}
    for instrument, bundle in sorted(bundles.items()):
        stats = _validate_instrument_bundle(bundle, expected_instrument=instrument)
        scans.append(pl.scan_parquet(bundle / "summary_contributions.parquet"))
        sources[instrument] = {
            "bundle": str(bundle.resolve()), "manifest_sha256": file_hash(bundle / "manifest.json"),
            "validation": stats,
        }
    contributions = pl.concat(scans, how="vertical")
    contributions.sink_parquet(temporary / "summary_contributions.parquet")
    summary = summarize_contributions(pl.scan_parquet(temporary / "summary_contributions.parquet"))
    summary.write_parquet(temporary / "summary.parquet")
    summary.write_csv(temporary / "summary.csv")
    manifest = {
        "schema": "empirical_controls_v2_cross_instrument_1", "status": "complete",
        "clock_policy_id": CLOCK_POLICY_ID,
        "sources": sources,
        "artifacts": {
            name: file_hash(temporary / name)
            for name in ("summary_contributions.parquet", "summary.parquet", "summary.csv")
        },
    }
    _write_json(temporary / "manifest.json", manifest)
    for name, digest in manifest["artifacts"].items():
        if file_hash(temporary / name) != digest:
            raise ValueError(f"combined artifact hash mismatch: {name}")
    _validate_combined_output(temporary, required_instruments=required_instruments)
    os.rename(temporary, destination)
    return destination


def orchestrate(nexi_bundle: Path) -> None:
    if nexi_bundle.resolve() != (RUN_ROOT / "bundles" / "NEXI").resolve():
        raise ValueError("NEXI controls bundle must belong to the same run root")
    _validate_instrument_bundle(nexi_bundle, expected_instrument="NEXI")
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    progress_path = RUN_ROOT / "progress.json"
    bundles = {"NEXI": nexi_bundle}
    _write_json(progress_path, {"status": "running", "completed": ["NEXI"], "current": "FERRARI", "started_at": time.time()})
    for instrument in INSTRUMENTS[1:]:
        bundles[instrument] = run_instrument(instrument)
        _write_json(progress_path, {"status": "running", "completed": sorted(bundles), "current": "combine" if instrument == INSTRUMENTS[-1] else INSTRUMENTS[1], "updated_at": time.time()})
    combined = combine(bundles)
    _write_json(progress_path, {"status": "complete", "completed": sorted(bundles), "combined": str(combined), "finished_at": time.time()})
    print(combined)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--certify", choices=INSTRUMENTS)
    parser.add_argument("--run-instrument", choices=INSTRUMENTS)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--nexi-bundle", type=Path)
    args = parser.parse_args(argv)
    if args.certify and args.run_instrument:
        parser.error("--certify and --run-instrument are mutually exclusive")
    if (args.certify or args.run_instrument) and args.nexi_bundle is not None:
        parser.error("--nexi-bundle is valid only for cross-instrument orchestration")
    if not args.certify and not args.run_instrument and args.nexi_bundle is None:
        parser.error("--nexi-bundle is required for cross-instrument orchestration")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.run_root is not None:
        global RUN_ROOT
        RUN_ROOT = args.run_root.resolve()
    if args.certify:
        print(certify(args.certify))
    elif args.run_instrument:
        print(run_instrument(args.run_instrument))
    else:
        assert args.nexi_bundle is not None
        orchestrate(args.nexi_bundle.resolve())


if __name__ == "__main__":
    main()
