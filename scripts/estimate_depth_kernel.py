#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.depth_kernel_calibration import calibrate_empirical_depth_kernel, summarise_instrument_kernel
from spoofing_detection.lob.spoofing_metrics import infer_tick_size_from_best_quotes


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate an empirical LOB depth kernel from a full instrument sample.")
    parser.add_argument("--input", type=Path, required=True, help="Full raw instrument parquet event file")
    parser.add_argument(
        "--quote-panel",
        type=Path,
        default=None,
        help="Reconstructed quote panel for tick-size inference; not read when --tick-size is supplied",
    )
    parser.add_argument("--instrument-id", required=True, help="Instrument identifier stored in artifacts")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--top-n", type=int, default=5, help="Top book levels to calibrate")
    parser.add_argument("--horizon-seconds", type=float, default=10.0, help="Forward horizon H for hit and mid-change profiles")
    parser.add_argument("--tick-size", type=float, default=None, help="Explicit tick size; otherwise inferred from --quote-panel")
    parser.add_argument("--protection-floor", type=float, default=0.0, help="Floor for 1 - hit_probability")
    parser.add_argument("--visibility-floor", type=float, default=0.0, help="Floor for absolute covariance visibility")
    parser.add_argument("--max-rows", type=int, default=None, help="Smoke-test row cap; marks output as non-production")
    return parser.parse_args(argv)


def _write_summary_report(*, output_path: Path, metadata: dict[str, Any], kernel: pl.DataFrame) -> None:
    lines = [
        "# Empirical LOB depth-kernel calibration",
        "",
        "This artifact estimates an instrument-specific empirical depth kernel from the full available raw sample unless `max_rows` is set. The kernel is a surveillance-model input, not a label and not proof of intent.",
        "",
        "## Parameters",
        "",
        f"- input: `{metadata['input']}`",
        f"- quote_panel: `{metadata.get('quote_panel')}`",
        f"- instrument_id: {metadata['instrument_id']}",
        f"- top_n: {metadata['top_n']}",
        f"- horizon_seconds: {metadata['horizon_seconds']}",
        f"- tick_size: {metadata['tick_size']}",
        f"- protection_floor: {metadata['protection_floor']}",
        f"- visibility_floor: {metadata['visibility_floor']}",
        f"- max_rows: {metadata['max_rows']}",
        f"- is_full_sample_calibration: {metadata['is_full_sample_calibration']}",
        "",
        "## Instrument summary",
        "",
    ]
    for key, value in metadata["instrument_summary"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Kernel weights", ""])
    if kernel.is_empty():
        lines.append("No kernel rows were estimated.")
    else:
        cols = [
            col
            for col in (
                "instrument_id",
                "side",
                "rank",
                "depth_distance_ticks",
                "exposure_count",
                "hit_probability",
                "visibility_covariance",
                "kernel_weight",
            )
            if col in kernel.columns
        ]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in kernel.sort(["side", "rank"]).select(cols).to_dicts():
            lines.append("| " + " | ".join(str(row.get(col)) for col in cols) + " |")
    output_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_events = pl.read_parquet(args.input)
    raw_events_for_compute = raw_events.head(args.max_rows) if args.max_rows is not None else raw_events
    if args.tick_size is not None:
        tick_size = float(args.tick_size)
    else:
        if args.quote_panel is None:
            raise ValueError("--quote-panel is required unless --tick-size is provided")
        tick_size = infer_tick_size_from_best_quotes(pl.read_parquet(args.quote_panel))

    kernel = calibrate_empirical_depth_kernel(
        raw_events_for_compute,
        instrument_id=args.instrument_id,
        top_n=args.top_n,
        tick_size=tick_size,
        horizon_seconds=args.horizon_seconds,
        protection_floor=args.protection_floor,
        visibility_floor=args.visibility_floor,
    )
    summary_df = summarise_instrument_kernel(kernel)
    instrument_summary = summary_df.to_dicts()[0] if not summary_df.is_empty() else {
        "instrument_id": args.instrument_id,
        "kappa_hat_instrument": None,
        "lambda_hat_instrument": None,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "kernel_parquet": args.output_dir / "empirical_depth_kernel.parquet",
        "kernel_csv": args.output_dir / "empirical_depth_kernel.csv",
        "metadata": args.output_dir / "metadata.json",
        "summary_report": args.output_dir / "summary_report.md",
    }
    kernel.write_parquet(paths["kernel_parquet"])
    kernel.write_csv(paths["kernel_csv"])
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "quote_panel": str(args.quote_panel) if args.quote_panel is not None else None,
        "output_dir": str(args.output_dir),
        "instrument_id": args.instrument_id,
        "top_n": args.top_n,
        "horizon_seconds": args.horizon_seconds,
        "tick_size": tick_size,
        "protection_floor": args.protection_floor,
        "visibility_floor": args.visibility_floor,
        "raw_event_rows": raw_events.height,
        "input_rows_for_calibration": raw_events_for_compute.height,
        "max_rows": args.max_rows,
        "is_full_sample_calibration": args.max_rows is None,
        "instrument_summary": instrument_summary,
        "weight_sums": kernel.group_by(["instrument_id", "side"]).agg(pl.col("kernel_weight").sum().alias("weight_sum")).to_dicts()
        if not kernel.is_empty()
        else [],
        "paths": {key: str(path) for key, path in paths.items()},
        "runtime": {"python": sys.version, "platform": platform.platform(), "polars": pl.__version__},
        "command": sys.argv if argv is None else ["estimate_depth_kernel.py", *argv],
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    _write_summary_report(output_path=paths["summary_report"], metadata=metadata, kernel=kernel)

    print(f"output_dir: {args.output_dir}")
    print(f"tick_size: {tick_size}")
    print(f"kernel_rows: {kernel.height}")
    print(f"metadata: {paths['metadata']}")
    print(f"summary_report: {paths['summary_report']}")


if __name__ == "__main__":
    main()
