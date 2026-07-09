#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.client_identity_audit import audit_missing_client_trading_capacity
from spoofing_detection.lob.depth_kernel_calibration import load_empirical_kernel_weights
from spoofing_detection.lob.spoofing_metric_plots import write_spoofing_metric_dashboard
from spoofing_detection.lob.spoofing_metrics import (
    compute_exploratory_metrics,
    compute_mcps_scores,
    infer_tick_size_from_best_quotes,
)
from spoofing_detection.lob.spoofing_config import DEFAULT_SPOOFING_CONFIG_PATH, load_spoofing_config_defaults


_CONFIGURABLE_DEFAULT_KEYS = {
    "depth_grid",
    "kappa",
    "lambda_",
    "epsilon",
    "window_seconds",
    "max_deceptive_order_age_seconds",
    "gamma_grid",
    "tick_size",
    "max_rows",
    "make_dashboard",
    "empirical_depth_kernel",
}


def _parse_int_grid(text: str) -> list[int]:
    values = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("grid must contain at least one value")
    if any(value <= 0 for value in values):
        raise ValueError("depth values must be positive")
    return values


def _parse_float_grid(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("grid must contain at least one value")
    return values


def _depth_output_paths(root: Path, top_n: int) -> dict[str, Path]:
    depth_dir = root / f"topn_{top_n}"
    return {
        "state_time_series": depth_dir / "client_metric_time_series.parquet",
        "execution_metrics": depth_dir / "execution_metrics.parquet",
        "candidate_deceptive_orders": depth_dir / "candidate_deceptive_orders.parquet",

        "rejected_executions": depth_dir / "rejected_executions.parquet",
        "client_mcps_scores": depth_dir / "client_mcps_scores.parquet",
        "dashboard": depth_dir / "spoofing_metric_dashboard.html",
    }


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _depth_outputs_complete(paths: dict[str, Path]) -> bool:
    required = (
        "state_time_series",
        "execution_metrics",
        "candidate_deceptive_orders",
        "rejected_executions",
        "client_mcps_scores",
    )
    return all(paths[name].exists() and paths[name].stat().st_size > 0 for name in required)


def _depth_counts_from_files(paths: dict[str, Path]) -> dict[str, int]:
    state_columns = pl.read_parquet(paths["state_time_series"]).columns
    counts = {
        name: pl.read_parquet(paths[name]).height
        for name in (
            "state_time_series",
            "execution_metrics",
            "candidate_deceptive_orders",

            "client_mcps_scores",
        )
    }
    counts["state_level_columns_included"] = any(column.startswith("bid_level_1_") for column in state_columns)
    return counts


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_SPOOFING_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_spoofing_config_defaults(
        config_path=config_args.config,
        section="grid",
        allowed_keys=_CONFIGURABLE_DEFAULT_KEYS,
    )

    parser = argparse.ArgumentParser(description="Run multidepth top-n MSCI/MCPS spoofing metrics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=config_args.config,
        help="JSON config file containing spoofing parameter defaults",
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw input parquet event file")
    parser.add_argument("--quote-panel", type=Path, default=None, help="Quote panel for tick-size inference")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--depth-grid", default="1,2,3,5,10", help="Comma-separated top-n depths")
    parser.add_argument("--kappa", type=float, default=1.0, help="Execution-risk protection parameter")
    parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0, help="Visibility-decay parameter")
    parser.add_argument("--epsilon", type=float, default=1e-12, help="Small denominator stabilizer")
    parser.add_argument("--window-seconds", type=float, default=1.0, help="Clock-time post-execution window")
    parser.add_argument(
        "--max-deceptive-order-age-seconds",
        type=float,
        default=600.0,
        help="Maximum age of candidate deceptive orders before the execution, in seconds",
    )
    parser.add_argument("--gamma-grid", default="0.25,0.5,0.75,1.0", help="Comma-separated MSCI thresholds")
    parser.add_argument("--tick-size", type=float, default=None, help="Optional explicit tick size")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional raw-row cap for smoke runs")
    parser.add_argument(
        "--empirical-depth-kernel",
        type=Path,
        default=None,
        help="Optional empirical_depth_kernel parquet/csv artifact. When set, rank weights override scalar kappa/lambda in DWI/MSCI weighting.",
    )
    parser.add_argument(
        "--make-dashboard",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Write one dashboard per depth",
    )
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)
    if args.empirical_depth_kernel is not None and not isinstance(args.empirical_depth_kernel, Path):
        args.empirical_depth_kernel = Path(args.empirical_depth_kernel)
    return args


def _write_grid_summary(path: Path, *, metadata: dict[str, Any], combined_scores: pl.DataFrame) -> None:
    lines = [
        "# Multidepth top-n MSCI/MCPS spoofing grid",
        "",
        "This grid follows the active manuscript model and computes MCPS across several book depths.",
        "The scores are surveillance cues, not labels and not proof of intent.",
        "",
        "## Multidepth interpretation",
        "",
        "- High MCPS at n=1 means the conditional profile collapse is visible close to the best quote.",
        "- Low MCPS at n=1 but higher MCPS at larger n means the suspicious profile is deeper in the book.",
        "- Stable high MCPS across several depths is stronger evidence for repeated conditional behavior than a single-depth spike.",
        "- Candidate deceptive orders are restricted to the configured pre-execution age window before each small execution.",
        "",
        "## Parameters",
        "",
        f"- input: `{metadata['input']}`",
        f"- depth_grid: {metadata['depth_grid']}",
        f"- kappa: {metadata['kappa']}",
        f"- lambda: {metadata['lambda_']}",
        f"- epsilon: {metadata['epsilon']}",
        f"- window_seconds: {metadata['window_seconds']}",
        f"- max_deceptive_order_age_seconds: {metadata['max_deceptive_order_age_seconds']}",
        f"- gamma_grid: {metadata['gamma_grid']}",
        f"- tick_size: {metadata['tick_size']}",
        "",
        "## Top clients across depths",
        "",
    ]
    if combined_scores.is_empty():
        lines.append("No MCPS rows.")
    else:
        cols = [
            col
            for col in ("client_id", "top_n", "gamma", "executions", "finite_msci_executions", "MCPS", "max_MSCI")
            if col in combined_scores.columns
        ]
        rows = combined_scores.sort(["MCPS", "max_MSCI", "executions"], descending=[True, True, True]).head(25).select(cols).to_dicts()
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("| " + " | ".join(["---"] * len(cols)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(col)) for col in cols) + " |")
    path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    depth_grid = _parse_int_grid(args.depth_grid)
    gamma_grid = _parse_float_grid(args.gamma_grid)
    raw_events = pl.read_parquet(args.input)
    raw_events_for_compute = raw_events.head(args.max_rows) if args.max_rows is not None else raw_events
    if args.tick_size is not None:
        tick_size = args.tick_size
    else:
        if args.quote_panel is None:
            raise ValueError("--quote-panel is required unless --tick-size is provided")
        tick_size = infer_tick_size_from_best_quotes(pl.read_parquet(args.quote_panel))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client_audit = audit_missing_client_trading_capacity(raw_events_for_compute)
    empirical_kernel_weights = (
        load_empirical_kernel_weights(args.empirical_depth_kernel) if args.empirical_depth_kernel is not None else None
    )
    combined_score_frames: list[pl.DataFrame] = []
    per_depth_counts: dict[str, dict[str, int]] = {}

    for top_n in depth_grid:
        paths = _depth_output_paths(args.output_dir, top_n)
        if _depth_outputs_complete(paths):
            scores = pl.read_parquet(paths["client_mcps_scores"])
            if args.make_dashboard and not paths["dashboard"].exists():
                write_spoofing_metric_dashboard(
                    execution_metrics=pl.read_parquet(paths["execution_metrics"]),
                    state_time_series=pl.read_parquet(paths["state_time_series"]),
                    mcps_scores=scores,
                    output_html=paths["dashboard"],
                    title=f"Multilevel spoofing metrics — top n = {top_n}",
                )
            combined_score_frames.append(scores)
            per_depth_counts[str(top_n)] = _depth_counts_from_files(paths)
            continue

        result = compute_exploratory_metrics(
            raw_events_for_compute,
            top_n=top_n,
            tick_size=tick_size,
            kappa=args.kappa,
            lambda_=args.lambda_,
            epsilon=args.epsilon,
            window_seconds=args.window_seconds,
            include_level_columns=top_n <= 5,
            max_deceptive_order_age_seconds=args.max_deceptive_order_age_seconds,
            empirical_kernel_weights=empirical_kernel_weights,
        )
        scores = compute_mcps_scores(result.execution_metrics, gamma_grid=gamma_grid)
        _write_parquet(result.state_time_series, paths["state_time_series"])
        _write_parquet(result.execution_metrics, paths["execution_metrics"])
        _write_parquet(result.candidate_deceptive_orders, paths["candidate_deceptive_orders"])

        _write_parquet(result.rejected_executions, paths["rejected_executions"])
        _write_parquet(scores, paths["client_mcps_scores"])
        if args.make_dashboard:
            write_spoofing_metric_dashboard(
                execution_metrics=result.execution_metrics,
                state_time_series=result.state_time_series,
                mcps_scores=scores,
                output_html=paths["dashboard"],
                title=f"Multilevel spoofing metrics — top n = {top_n}",
            )
        combined_score_frames.append(scores)
        per_depth_counts[str(top_n)] = {
            "state_time_series": result.state_time_series.height,
            "execution_metrics": result.execution_metrics.height,
            "candidate_deceptive_orders": result.candidate_deceptive_orders.height,

            "client_mcps_scores": scores.height,
            "state_level_columns_included": top_n <= 5,
        }

    combined_scores = pl.concat(combined_score_frames, how="diagonal_relaxed") if combined_score_frames else pl.DataFrame()
    combined_path = args.output_dir / "combined_client_mcps_scores.parquet"
    _write_parquet(combined_scores, combined_path)
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "quote_panel": str(args.quote_panel) if args.quote_panel is not None else None,
        "output_dir": str(args.output_dir),
        "config": str(args.config) if args.config is not None and args.config.exists() else None,
        "depth_grid": depth_grid,
        "kappa": args.kappa,
        "lambda_": args.lambda_,
        "epsilon": args.epsilon,
        "window_seconds": args.window_seconds,
        "max_deceptive_order_age_seconds": args.max_deceptive_order_age_seconds,
        "gamma_grid": gamma_grid,
        "tick_size": tick_size,
        "max_rows": args.max_rows,
        "empirical_depth_kernel": str(args.empirical_depth_kernel) if args.empirical_depth_kernel is not None else None,
        "kernel_mode": "empirical" if args.empirical_depth_kernel is not None else "parametric",
        "client_identity_audit": client_audit,
        "per_depth_counts": per_depth_counts,
        "combined_client_mcps_scores": str(combined_path),
        "command": sys.argv,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    summary_path = args.output_dir / "summary_report.md"
    _write_grid_summary(summary_path, metadata=metadata, combined_scores=combined_scores)

    print(f"output_dir: {args.output_dir}")
    print(f"tick_size: {tick_size}")
    print(f"combined_client_mcps_scores: {combined_scores.height}")
    print(f"metadata: {metadata_path}")
    print(f"summary_report: {summary_path}")


if __name__ == "__main__":
    main()
