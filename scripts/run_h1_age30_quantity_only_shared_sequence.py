#!/usr/bin/env python3
"""Run the bounded H1 age-30 quantity-only shared-sequence reruns sequentially.

The driver is intentionally self-contained: it discovers the historically
validated inputs from the phase-1 reconciliation note, verifies their hashes,
creates fresh authoritative JSON configs, records every exact command, and
writes durable status/checkpoint evidence outside the per-instrument bundles.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
INSTRUMENTS = ("FERRARI", "NEXI", "RISANAMENTO")
PHASE1_RECONCILIATION = REPO_ROOT / "astra_metric_audit/notes/phase1_reconciliation.json"
COMPUTE_SCRIPT = REPO_ROOT / "scripts/compute_spoofing_metrics.py"
WORKER_SCRIPT = REPO_ROOT / "scripts/h1_instrumented_worker.py"
PYTHON = Path("/home/danielemdn/miniconda3/envs/main/bin/python")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_phase1() -> dict[str, Any]:
    payload = json.loads(PHASE1_RECONCILIATION.read_text())
    bundles = payload.get("bundles")
    if not isinstance(bundles, dict) or set(INSTRUMENTS) - set(bundles):
        raise ValueError("phase1 reconciliation does not contain all three required instruments")
    return bundles


def source_config_for(instrument: str, historical_bundle: Path, expected_sha256: str) -> Path:
    """Select the config colocated with the exact historical validated bundle.

    Some identical NEXI pilot configs were copied into prior pilot roots, so a
    repository-wide hash match is not a unique provenance selector.
    """
    path = historical_bundle.parents[1] / "configs" / f"{instrument}_spoofing_detection_v3.json"
    if not path.is_file():
        raise FileNotFoundError(f"historical config is absent for {instrument}: {path}")
    if sha256(path) != expected_sha256:
        raise ValueError(f"historical config hash does not match phase-1 reconciliation for {instrument}")
    return path


def rss_bytes(pid: int) -> int | None:
    status = Path(f"/proc/{pid}/status")
    try:
        for line in status.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def system_memory_snapshot() -> dict[str, int | None]:
    """Read host availability alongside the in-process detector worker RSS."""
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value, *_ = line.replace(":", "").split()
            if key in {"MemAvailable", "SwapFree"}:
                values[key] = int(value) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return {
        "mem_available_bytes": values.get("MemAvailable"),
        "swap_free_bytes": values.get("SwapFree"),
    }


def worker_command(compute_argv: list[str]) -> list[str]:
    """Wrap canonical compute arguments without changing their semantics."""
    return [
        str(PYTHON), str(WORKER_SCRIPT), "--progress-seconds", "30", "--", *compute_argv,
    ]


def terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=60)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def tail(path: Path, max_bytes: int = 12000) -> str:
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        handle.seek(max(0, size - max_bytes))
        return handle.read().decode("utf-8", errors="replace")


def validate_completed_output(output_dir: Path, expected_config_hash: str) -> dict[str, Any]:
    """Validate published metadata/counts and new lifecycle score invariants."""
    import polars as pl

    metadata_path = output_dir / "metadata.json"
    if not metadata_path.exists():
        raise ValueError("completed worker did not publish metadata.json")
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("config_sha256") != expected_config_hash:
        raise ValueError("metadata config hash does not match the generated H1 config")
    if metadata.get("window_seconds") != 1.0:
        raise ValueError("metadata window_seconds is not H1=1.0")
    if metadata.get("max_deceptive_order_age_seconds") != 30.0:
        raise ValueError("metadata max_deceptive_order_age_seconds is not 30.0")
    if metadata.get("clock_policy_id") is None:
        raise ValueError("metadata does not declare the required v3 clock policy")

    artifact_keys = {
        "candidate_episodes": "candidate_episodes.parquet",
        "episode_cluster_members": "episode_cluster_members.parquet",
        "episode_withdrawals": "episode_withdrawals.parquet",
        "candidate_lifecycles": "candidate_lifecycles.parquet",
        "episode_anchor_summary": "episode_anchor_summary.parquet",
        "actor_day_episode_summary": "actor_day_episode_summary.parquet",
    }
    row_counts: dict[str, int] = {}
    for key, filename in artifact_keys.items():
        artifact = output_dir / filename
        if not artifact.exists():
            raise ValueError(f"missing required episode artifact: {filename}")
        count = pl.scan_parquet(artifact).select(pl.len()).collect().item()
        expected = metadata.get("row_counts", {}).get(key)
        if count != expected:
            raise ValueError(f"row count mismatch for {key}: artifact={count}, metadata={expected}")
        row_counts[key] = count

    episodes = pl.read_parquet(output_dir / artifact_keys["candidate_episodes"])
    lifecycles = pl.read_parquet(output_dir / artifact_keys["candidate_lifecycles"])
    required_episode_columns = {
        "episode_id", "total_execution_quantity", "primary_lifecycle_union_exposure_quantity",
        "primary_supported_unique_cancellation_quantity",
        "primary_withdrawal_to_lifecycle_exposure_ratio", "primary_quantity_only_wmsci",
    }
    required_lifecycle_columns = {
        "episode_id", "lifecycle_union_exposure_quantity", "raw_unique_cancellation_quantity",
        "supported_unique_cancellation_quantity",
    }
    if required_episode_columns - set(episodes.columns):
        raise ValueError("candidate_episodes is missing primary quantity-only score columns")
    if required_lifecycle_columns - set(lifecycles.columns):
        raise ValueError("candidate_lifecycles is missing lifecycle support columns")

    invalid_lifecycles = lifecycles.filter(
        (pl.col("supported_unique_cancellation_quantity") < 0)
        | (pl.col("supported_unique_cancellation_quantity") > pl.col("lifecycle_union_exposure_quantity"))
        | (pl.col("supported_unique_cancellation_quantity") > pl.col("raw_unique_cancellation_quantity"))
    )
    if invalid_lifecycles.height:
        raise ValueError(f"{invalid_lifecycles.height} lifecycle support-cap violations")

    totals = lifecycles.group_by("episode_id").agg(
        pl.col("lifecycle_union_exposure_quantity").sum().alias("_P"),
        pl.col("supported_unique_cancellation_quantity").sum().alias("_U"),
    )
    reconciled = episodes.join(totals, on="episode_id", how="left").with_columns(
        pl.col("_P").fill_null(0.0), pl.col("_U").fill_null(0.0)
    )
    mismatch = reconciled.filter(
        ((pl.col("primary_lifecycle_union_exposure_quantity") - pl.col("_P")).abs() > 1e-9)
        | ((pl.col("primary_supported_unique_cancellation_quantity") - pl.col("_U")).abs() > 1e-9)
        | (
            pl.col("primary_withdrawal_to_lifecycle_exposure_ratio").is_not_null()
            & (
                (pl.col("primary_withdrawal_to_lifecycle_exposure_ratio") < 0)
                | (pl.col("primary_withdrawal_to_lifecycle_exposure_ratio") > 1)
            )
        )
    )
    if mismatch.height:
        raise ValueError(f"{mismatch.height} episode-to-lifecycle P/U reconciliation violations")

    invalid_scores = 0
    max_score_abs_error = 0.0
    for row in reconciled.select(sorted(required_episode_columns)).iter_rows(named=True):
        p = row["primary_lifecycle_union_exposure_quantity"]
        q = row["total_execution_quantity"]
        u = row["primary_supported_unique_cancellation_quantity"]
        observed = row["primary_quantity_only_wmsci"]
        if p is None or q is None or u is None or q <= 0:
            if observed is not None:
                invalid_scores += 1
            continue
        expected = 0.0 if p <= 0 or u <= 0 else math.log1p(p / q) * math.log1p(u / q) * (u / p)
        if observed is None:
            invalid_scores += 1
            continue
        max_score_abs_error = max(max_score_abs_error, abs(observed - expected))
        if abs(observed - expected) > 1e-9:
            invalid_scores += 1
    if invalid_scores:
        raise ValueError(f"{invalid_scores} primary quantity-only score invariant violations")

    return {
        "metadata_path": str(metadata_path),
        "row_counts": row_counts,
        "candidate_lifecycle_count": lifecycles.height,
        "episode_count": episodes.height,
        "max_primary_quantity_only_score_abs_error": max_score_abs_error,
        "invariants": {
            "P": "lifecycle_union_exposure_quantity; maximum admissible pre-cancel quantity per lifecycle",
            "U_lte_P": True,
            "Q": "total_execution_quantity across all episode member executions",
            "REV_weight": "supported cancellation quantity contract declared by output metadata",
        },
    }


def build_plan(output_root: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundles = load_phase1()
    configs_dir = output_root / "configs"
    runs_dir = output_root / "runs"
    logs_dir = output_root / "logs"
    configs_dir.mkdir(parents=True)
    runs_dir.mkdir()
    logs_dir.mkdir()
    plan: list[dict[str, Any]] = []

    for instrument in INSTRUMENTS:
        historical_bundle = Path(bundles[instrument]["validator"]["bundle"])
        historical_metadata = json.loads((historical_bundle / "metadata.json").read_text())
        expected = bundles[instrument]["declared_input_hashes"]
        source_config = source_config_for(instrument, historical_bundle, expected["config_sha256"])
        source_payload = json.loads(source_config.read_text())
        generated_payload = json.loads(json.dumps(source_payload))
        generated_payload["metrics"]["window_seconds"] = 1.0
        generated_payload["metrics"]["max_deceptive_order_age_seconds"] = 30.0
        generated_config = configs_dir / f"{instrument}_h1_age30_quantity_only_shared_sequence.json"
        generated_config.write_text(json.dumps(generated_payload, indent=2, sort_keys=True) + "\n")
        if generated_payload["grid"] != source_payload["grid"]:
            raise ValueError(f"{instrument} generated config changed grid parameters")
        changed = {
            key: (source_payload["metrics"][key], generated_payload["metrics"][key])
            for key in generated_payload["metrics"]
            if not key.startswith("_comment_") and source_payload["metrics"].get(key) != generated_payload["metrics"].get(key)
        }
        if changed != {"window_seconds": (source_payload["metrics"]["window_seconds"], 1.0), "max_deceptive_order_age_seconds": (source_payload["metrics"]["max_deceptive_order_age_seconds"], 30.0)}:
            raise ValueError(f"{instrument} generated config changed keys other than the authorized H1/age30 values")

        paths = {
            "input": Path(historical_metadata["input"]),
            "quote_panel": Path(historical_metadata["quote_panel"]),
            "empirical_depth_kernel": Path(historical_metadata["empirical_depth_kernel"]),
        }
        expected_hashes = {
            "input": expected["raw_events_sha256"],
            "quote_panel": expected["quote_panel_sha256"],
            "empirical_depth_kernel": expected["empirical_depth_kernel_sha256"],
        }
        actual_hashes: dict[str, str] = {}
        for key, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{instrument} {key} is absent: {path}")
            actual_hashes[key] = sha256(path)
            if actual_hashes[key] != expected_hashes[key]:
                raise ValueError(f"{instrument} {key} hash does not match phase-1 reconciliation")

        output_dir = runs_dir / instrument
        compute_argv = [
            "--config", str(generated_config),
            "--input", str(paths["input"]), "--quote-panel", str(paths["quote_panel"]),
            "--output-dir", str(output_dir), "--clock-policy-v3",
        ]
        command = worker_command(compute_argv)
        plan.append({
            "instrument": instrument,
            "historical_bundle": str(historical_bundle),
            "source_config": str(source_config),
            "source_config_sha256": sha256(source_config),
            "config": str(generated_config),
            "config_sha256": sha256(generated_config),
            "input_paths": {key: str(value) for key, value in paths.items()},
            "input_hashes": actual_hashes,
            "phase1_expected_input_hashes": expected_hashes,
            "output_dir": str(output_dir),
            "log": str(logs_dir / f"{instrument}.log"),
            "command": command,
            "status": "planned",
        })

    manifest = {
        "schema": "h1_age30_quantity_only_shared_sequence_rerun_v1",
        "created_at_utc": utc_now(),
        "driver": str(Path(__file__).resolve()),
        "driver_sha256": sha256(Path(__file__).resolve()),
        "compute_script": str(COMPUTE_SCRIPT),
        "compute_script_sha256": sha256(COMPUTE_SCRIPT),
        "worker_wrapper": str(WORKER_SCRIPT),
        "worker_wrapper_sha256": sha256(WORKER_SCRIPT),
        "python": str(PYTHON),
        "experiment": {
            "label": "H1 age30 quantity-only shared-sequence",
            "quantity_only_score": "log1p(P/Q)*log1p(U/Q)*(U/P)",
            "P": "sum lifecycle maximum admissible pre-cancel snapshot quantity",
            "U": "supported unique physical cancellation quantity capped once per lifecycle by P",
            "Q": "sum all member execution quantity",
            "REV": "supported cancellation quantity contract",
            "window_seconds": 1.0,
            "max_deceptive_order_age_seconds": 30.0,
            "clock_policy": "spoofing_detection_clock_policy_v3",
            "execution": "sequential",
        },
        "runs": plan,
    }
    return manifest, plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--per-instrument-wall-seconds", type=int, default=7200)
    parser.add_argument("--rss-sample-seconds", type=float, default=5.0)
    parser.add_argument("--available-memory-floor-bytes", type=float, default=2 * 1024**3)
    args = parser.parse_args()
    if not PYTHON.is_file():
        raise FileNotFoundError(f"required interpreter is missing: {PYTHON}")
    if args.per_instrument_wall_seconds <= 0 or not math.isfinite(args.rss_sample_seconds) or args.rss_sample_seconds <= 0:
        raise ValueError("wall and RSS sample durations must be positive")
    if not math.isfinite(args.available_memory_floor_bytes) or args.available_memory_floor_bytes <= 0:
        raise ValueError("available memory floor must be a finite positive number of bytes")
    output_root = args.output_root.resolve()
    if output_root.exists() and any(output_root.iterdir()):
        raise ValueError(f"refusing to use non-empty output root: {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "driver.pid").write_text(f"{os.getpid()}\n")

    manifest, plan = build_plan(output_root)
    manifest_path = output_root / "manifest.json"
    checkpoint_path = output_root / "checkpoint.json"
    manifest["driver_pid"] = os.getpid()
    manifest["per_instrument_wall_seconds"] = args.per_instrument_wall_seconds
    manifest["available_memory_floor_bytes"] = args.available_memory_floor_bytes
    manifest["worker_wrapper"] = str(WORKER_SCRIPT)
    manifest["worker_wrapper_sha256"] = sha256(WORKER_SCRIPT)
    write_json(manifest_path, manifest)
    checkpoint: dict[str, Any] = {
        "schema": manifest["schema"], "driver_pid": os.getpid(), "updated_at_utc": utc_now(),
        "status": "running", "runs": {entry["instrument"]: {"status": "planned"} for entry in plan},
    }
    write_json(checkpoint_path, checkpoint)

    for entry in plan:
        instrument = entry["instrument"]
        log_path = Path(entry["log"])
        started = time.monotonic()
        checkpoint["runs"][instrument] = {"status": "running", "started_at_utc": utc_now(), "log": str(log_path)}
        checkpoint["updated_at_utc"] = utc_now()
        write_json(checkpoint_path, checkpoint)
        peak_rss = 0
        timed_out = False
        memory_failure = None
        memory_failure_sample = None
        telemetry_path = log_path.with_suffix(".resources.jsonl")
        with log_path.open("w") as log, telemetry_path.open("w") as telemetry:
            log.write("COMMAND: " + json.dumps(entry["command"]) + "\n")
            log.write("STARTED_AT_UTC: " + checkpoint["runs"][instrument]["started_at_utc"] + "\n")
            log.flush()
            process = subprocess.Popen(
                entry["command"], cwd=REPO_ROOT, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                while process.poll() is None:
                    rss = rss_bytes(process.pid)
                    if rss is not None:
                        peak_rss = max(peak_rss, rss)
                    sample = {
                        "child_pid": process.pid,
                        "elapsed_seconds": time.monotonic() - started,
                        "child_rss_bytes": rss,
                        **system_memory_snapshot(),
                    }
                    print(json.dumps(sample, sort_keys=True), file=telemetry, flush=True)
                    available = sample["mem_available_bytes"]
                    if available is None or available < args.available_memory_floor_bytes:
                        memory_failure = "memory_unavailable" if available is None else "memory_floor_breach"
                        memory_failure_sample = sample
                        terminate_process_group(process)
                        break
                    if time.monotonic() - started > args.per_instrument_wall_seconds:
                        timed_out = True
                        terminate_process_group(process)
                        break
                    time.sleep(args.rss_sample_seconds)
            finally:
                terminate_process_group(process)
            return_code = process.wait()
            rss = rss_bytes(process.pid)
            if rss is not None:
                peak_rss = max(peak_rss, rss)
        elapsed = time.monotonic() - started
        result = {
            "status": "failed", "started_at_utc": checkpoint["runs"][instrument]["started_at_utc"],
            "finished_at_utc": utc_now(), "exit_code": return_code,
            "elapsed_seconds": elapsed, "peak_sampled_rss_bytes": peak_rss,
            "failure_type": memory_failure or ("wall_timeout" if timed_out else (None if return_code == 0 else "worker_nonzero_exit")),
            "supervisor_telemetry": str(telemetry_path),
            "memory_floor_breach": memory_failure_sample,
            "log": str(log_path), "log_tail": tail(log_path),
        }
        if return_code == 0 and not timed_out and memory_failure is None:
            try:
                result["validation"] = validate_completed_output(Path(entry["output_dir"]), entry["config_sha256"])
                result["status"] = "completed"
            except Exception as exc:  # Preserve computational output but do not certify it.
                result["failure_type"] = "post_run_validation_failed"
                result["validation_error"] = f"{type(exc).__name__}: {exc}"
        checkpoint["runs"][instrument] = result
        checkpoint["updated_at_utc"] = utc_now()
        write_json(checkpoint_path, checkpoint)
        manifest["runs"] = plan
        for manifest_entry in manifest["runs"]:
            if manifest_entry["instrument"] == instrument:
                manifest_entry.update(result)
        write_json(manifest_path, manifest)
        if result["status"] != "completed":
            checkpoint["status"] = "failed"
            checkpoint["updated_at_utc"] = utc_now()
            write_json(checkpoint_path, checkpoint)
            return 1

    checkpoint["status"] = "completed"
    checkpoint["completed_at_utc"] = utc_now()
    checkpoint["updated_at_utc"] = utc_now()
    write_json(checkpoint_path, checkpoint)
    manifest["completed_at_utc"] = utc_now()
    write_json(manifest_path, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
