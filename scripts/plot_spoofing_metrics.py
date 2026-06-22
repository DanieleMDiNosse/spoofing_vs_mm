#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.spoofing_metric_plots import write_spoofing_metric_dashboard


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot multilevel top-n spoofing surveillance diagnostics.")
    parser.add_argument("--execution-metrics", type=Path, required=True, help="execution_metrics.parquet")
    parser.add_argument("--state-time-series", type=Path, default=None, help="client_metric_time_series.parquet")
    parser.add_argument("--mcps-scores", type=Path, default=None, help="client_mcps_scores.parquet")
    parser.add_argument("--output-html", type=Path, required=True, help="Output dashboard HTML")
    parser.add_argument("--title", default="Multilevel top-n spoofing surveillance metrics", help="Dashboard title")
    parser.add_argument("--client-id", default=None, help="Optional client id to show in DWI time series")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    execution_metrics = pl.read_parquet(args.execution_metrics)
    state_time_series = pl.read_parquet(args.state_time_series) if args.state_time_series is not None else None
    mcps_scores = pl.read_parquet(args.mcps_scores) if args.mcps_scores is not None else None
    write_spoofing_metric_dashboard(
        execution_metrics=execution_metrics,
        state_time_series=state_time_series,
        mcps_scores=mcps_scores,
        output_html=args.output_html,
        title=args.title,
        client_id=args.client_id,
    )
    print(f"html: {args.output_html}")
    print(f"executions: {execution_metrics.height}")
    if state_time_series is not None:
        print(f"state_rows: {state_time_series.height}")
    if mcps_scores is not None:
        print(f"mcps_rows: {mcps_scores.height}")


if __name__ == "__main__":
    main()
