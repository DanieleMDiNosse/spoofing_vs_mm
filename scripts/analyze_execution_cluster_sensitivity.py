#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import polars as pl


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_output(instrument: str, gap_ms: int, output_dir: Path) -> dict[str, object]:
    execution_path = output_dir / "execution_metrics.parquet"
    members_path = output_dir / "execution_cluster_members.parquet"
    candidates_path = output_dir / "execution_cancel_candidates.parquet"
    for required in (execution_path, members_path, candidates_path):
        if not required.exists():
            raise FileNotFoundError(required)
    executions = pl.read_parquet(execution_path)
    members = pl.read_parquet(members_path)
    candidates = pl.read_parquet(candidates_path)
    if executions.get_column("execution_cluster_id").is_duplicated().any():
        raise ValueError(f"duplicate clusters at gap={gap_ms}")
    member_key = [column for column in ("partition_id", "child_sort_index") if column in members.columns]
    if members.select(member_key).is_duplicated().any():
        raise ValueError(f"raw child fill assigned more than once at gap={gap_ms}")
    causal_columns = {"cancel_sort_index", "cluster_last_sort_index"}
    missing_causal_columns = causal_columns - set(candidates.columns)
    if missing_causal_columns:
        raise ValueError(
            f"missing cancellation-causality columns at gap={gap_ms}: "
            f"{sorted(missing_causal_columns)}"
        )
    noncausal = candidates.filter(
        pl.col("cancel_sort_index") <= pl.col("cluster_last_sort_index")
    )
    if not noncausal.is_empty():
        raise ValueError(f"non-causal cancellation candidate at gap={gap_ms}")
    assigned = candidates.filter(pl.col("assigned_flag").fill_null(False))
    cancel_key = [column for column in ("partition_id", "cancel_sort_index", "candidate_order_id") if column in assigned.columns]
    if cancel_key and assigned.select(cancel_key).is_duplicated().any():
        raise ValueError(f"cancellation assigned more than once at gap={gap_ms}")
    matched_column = "has_matched_deceptive_cancel_window"
    matched = executions.filter(pl.col(matched_column).fill_null(False)) if matched_column in executions.columns else executions.head(0)
    matched_ids = set(matched.get_column("execution_cluster_id").to_list())
    assigned_ids = set(assigned.get_column("execution_cluster_id").to_list())
    if matched_ids != assigned_ids:
        raise ValueError(f"matched-cluster flags disagree with assigned cancellations at gap={gap_ms}")
    return {
        "instrument": instrument,
        "execution_cluster_max_gap_ms": gap_ms,
        "execution_cluster_count": executions.height,
        "raw_fill_message_count": members.height,
        "matched_cluster_count": matched.height,
        "assigned_candidate_count": assigned.height,
        "candidate_link_count": candidates.height,
        "max_WMSCI": matched.select(pl.col("WMSCI_event").max()).item()
        if "WMSCI_event" in matched.columns and not matched.is_empty()
        else None,
        "median_WMSCI": matched.select(pl.col("WMSCI_event").median()).item()
        if "WMSCI_event" in matched.columns and not matched.is_empty()
        else None,
        "max_withdrawal_to_fill_ratio": matched.select(pl.col("withdrawal_to_fill_ratio").max()).item()
        if "withdrawal_to_fill_ratio" in matched.columns and not matched.is_empty()
        else None,
        "execution_metrics_sha256": _sha256(execution_path),
        "members_sha256": _sha256(members_path),
        "candidates_sha256": _sha256(candidates_path),
    }


def build_command(args: argparse.Namespace, *, gap_ms: int, output_dir: Path) -> list[str]:
    command = [
        args.python,
        str(args.compute_script),
        "--input",
        str(args.input),
        "--output-dir",
        str(output_dir),
        "--config",
        str(args.config),
        "--execution-cluster-max-gap-ms",
        str(gap_ms),
    ]
    if args.empirical_kernel is not None:
        command.extend(["--empirical-depth-kernel", str(args.empirical_kernel)])
    if args.tick_size is not None:
        command.extend(["--tick-size", str(args.tick_size)])
    if args.compact_state:
        command.append("--compact-state")
    return command


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rerun cluster-aware spoofing metrics across inter-fill-gap thresholds.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--instrument", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/spoofing_detection_parameters.json"))
    parser.add_argument("--empirical-kernel", type=Path)
    parser.add_argument("--tick-size", type=float)
    parser.add_argument("--gap-ms", type=int, nargs="+", default=[25, 50, 100, 250])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--compute-script", type=Path, default=Path("scripts/compute_spoofing_metrics.py"))
    parser.add_argument("--compact-state", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reuse-existing", action="store_true")
    args = parser.parse_args(argv)
    if any(gap < 0 for gap in args.gap_ms):
        parser.error("--gap-ms values must be non-negative")
    if len(set(args.gap_ms)) != len(args.gap_ms):
        parser.error("--gap-ms values must be unique")
    return args


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    commands: list[list[str]] = []
    for gap_ms in sorted(args.gap_ms):
        output_dir = args.output_root / f"{args.instrument}_cluster_gap_{gap_ms}ms"
        command = build_command(args, gap_ms=gap_ms, output_dir=output_dir)
        commands.append(command)
        required = output_dir / "execution_cancel_candidates.parquet"
        if not (args.reuse_existing and required.exists()):
            if output_dir.exists() and any(output_dir.iterdir()):
                raise FileExistsError(f"refusing to overwrite non-empty output directory: {output_dir}")
            subprocess.run(command, check=True)
        rows.append(summarize_output(args.instrument, gap_ms, output_dir))

    summary = pl.DataFrame(rows).sort("execution_cluster_max_gap_ms")
    if summary.get_column("raw_fill_message_count").n_unique() != 1:
        raise ValueError("raw child-fill population changed across sensitivity runs")
    summary.write_csv(args.output_root / "cluster_sensitivity.csv")
    summary.write_parquet(args.output_root / "cluster_sensitivity.parquet")
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "analytical_unit": "execution_cluster",
        "input": str(args.input),
        "input_sha256": _sha256(args.input),
        "config": str(args.config),
        "config_sha256": _sha256(args.config),
        "empirical_kernel": str(args.empirical_kernel) if args.empirical_kernel else None,
        "empirical_kernel_sha256": _sha256(args.empirical_kernel) if args.empirical_kernel else None,
        "commands": commands,
    }
    (args.output_root / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
