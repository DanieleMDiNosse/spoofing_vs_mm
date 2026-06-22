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
from spoofing_detection.lob.spoofing_metrics import (
    compute_exploratory_metrics,
    compute_mcps_scores,
    infer_tick_size_from_best_quotes,
)


def _parse_float_grid(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("grid must contain at least one value")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute multilevel top-n spoofing surveillance metrics.")
    parser.add_argument("--input", type=Path, required=True, help="Raw input parquet event file")
    parser.add_argument(
        "--quote-panel",
        type=Path,
        default=None,
        help="Reconstructed lob_event_state_panel.parquet used only for tick-size inference",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--top-n", type=int, default=3, help="Market top-N levels used for client depth profiles")
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
    return parser.parse_args(argv)


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _finite_count(df: pl.DataFrame, column: str) -> int:
    if df.is_empty() or column not in df.columns:
        return 0
    return int(df.select(pl.col(column).is_not_null().sum()).item())


def _markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    if not rows:
        return ["No rows to show."]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    out = [header, sep]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col)) for col in cols) + " |")
    return out


def _top_execution_lines(execution_metrics: pl.DataFrame, limit: int = 20) -> list[str]:
    if execution_metrics.is_empty() or "MSCI" not in execution_metrics.columns:
        return ["No eligible executions."]
    cols = [
        col
        for col in (
            "sort_index",
            "event_ts",
            "client_id",
            "execution_side",
            "deceptive_side",
            "fill_qty",
            "DWI_pre_window",
            "DWI_post_window",
            "SCI",
            "collapse_opposite_side",
            "collapse_same_side",
            "MSCI",
            "candidate_deceptive_visible_qty_pre",
            "has_matched_deceptive_cancel_window",
        )
        if col in execution_metrics.columns
    ]
    rows = (
        execution_metrics.filter(pl.col("MSCI").is_not_null())
        .sort("MSCI", descending=True)
        .head(limit)
        .select(cols)
        .to_dicts()
    )
    return _markdown_table(rows, cols) if rows else ["No executions with finite MSCI."]


def _top_deceptive_cancel_lines(execution_metrics: pl.DataFrame, limit: int = 20) -> list[str]:
    if execution_metrics.is_empty() or "has_matched_deceptive_cancel_window" not in execution_metrics.columns:
        return ["Matched deceptive-order cancellations were not computed."]
    matched = execution_metrics.filter(pl.col("has_matched_deceptive_cancel_window"))
    if matched.is_empty():
        return ["No execution directly cancelled a pre-existing candidate deceptive order inside the window."]
    cols = [
        col
        for col in (
            "sort_index",
            "event_ts",
            "client_id",
            "execution_side",
            "deceptive_side",
            "fill_qty",
            "candidate_deceptive_visible_qty_pre",
            "matched_deceptive_cancel_visible_qty_window",
            "matched_deceptive_cancel_order_ids_window",
            "matched_deceptive_cancel_fraction_window",
            "MSCI",
        )
        if col in matched.columns
    ]
    rows = (
        matched.sort(["matched_deceptive_cancel_visible_qty_window", "MSCI"], descending=[True, True])
        .head(limit)
        .select(cols)
        .to_dicts()
    )
    return _markdown_table(rows, cols)


def _top_mcps_lines(mcps_scores: pl.DataFrame, limit: int = 20) -> list[str]:
    if mcps_scores.is_empty():
        return ["No MCPS rows."]
    cols = [
        col
        for col in (
            "client_id",
            "top_n",
            "gamma",
            "executions",
            "finite_msci_executions",
            "msci_above_gamma_count",
            "MCPS",
            "max_MSCI",
            "mean_MSCI",
            "candidate_profile_share",
            "matched_deceptive_cancel_share",
        )
        if col in mcps_scores.columns
    ]
    rows = mcps_scores.sort(["MCPS", "max_MSCI", "executions"], descending=[True, True, True]).head(limit).select(cols).to_dicts()
    return _markdown_table(rows, cols)


def _client_audit_lines(client_audit: dict[str, Any]) -> list[str]:
    lines = []
    for key, value in client_audit.items():
        label = key.replace("claim_holds", "client identity claim supported").replace("_", " ")
        lines.append(f"- {label}: {value}")
    return lines


def _write_summary_report(
    *,
    output_path: Path,
    metadata: dict[str, Any],
    client_audit: dict[str, Any],
    execution_metrics: pl.DataFrame,
    state_time_series: pl.DataFrame,
    candidate_deceptive_orders: pl.DataFrame,
    mcps_scores: pl.DataFrame,
) -> None:
    matched_count = 0
    if not execution_metrics.is_empty() and "has_matched_deceptive_cancel_window" in execution_metrics.columns:
        matched_count = int(execution_metrics.select(pl.col("has_matched_deceptive_cancel_window").sum()).item())
    candidate_count = 0
    if not execution_metrics.is_empty() and "candidate_deceptive_order_count_pre" in execution_metrics.columns:
        candidate_count = int(execution_metrics.select((pl.col("candidate_deceptive_order_count_pre") > 0).sum()).item())

    lines = [
        "# Multilevel top-n spoofing surveillance metrics",
        "",
        "This report follows the active manuscript model. The scores are surveillance cues, not labels and not proof of intent.",
        "",
        "## How to read this report",
        "",
        "- DWI tells whether a client is ask-heavy or bid-heavy in the weighted top-n book profile.",
        "- SCI is the absolute DWI change from immediately before a small passive execution to the post-execution window.",
        "- Collapse measures how much weighted liquidity disappears after the execution on each side of the book.",
        "- MSCI is high only when DWI changes sharply and the opposite side collapses more than the execution side.",
        "- MCPS is a client-level repetition score: the fraction of small executions whose MSCI is above gamma.",
        "- A candidate deceptive profile is the same client's pre-existing visible depth on the side opposite to the small execution, posted within the configured pre-execution age window.",
        "",
        "## Parameters",
        "",
        f"- input: `{metadata['input']}`",
        f"- quote_panel: `{metadata.get('quote_panel')}`",
        f"- top_n: {metadata['top_n']}",
        f"- kappa: {metadata['kappa']}",
        f"- lambda: {metadata['lambda_']}",
        f"- epsilon: {metadata['epsilon']}",
        f"- window_seconds: {metadata['window_seconds']}",
        f"- max_deceptive_order_age_seconds: {metadata['max_deceptive_order_age_seconds']}",
        f"- gamma_grid: {metadata['gamma_grid']}",
        f"- tick_size: {metadata['tick_size']}",
        "- identity: NMSC_ORIGINALCLIENTIDSHORTCODE only",
        "- market orders included: false",
        "- event selection: matched deceptive-order cancellations only",
        "",
        "## Client identity audit",
        "",
    ]
    lines.extend(_client_audit_lines(client_audit))
    lines.extend(["", "## Row counts", ""])
    for key, value in metadata["row_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            f"- clients_with_topN_profile: {state_time_series.get_column('client_id').n_unique() if not state_time_series.is_empty() and 'client_id' in state_time_series.columns else 0}",
            f"- finite_SCI_executions: {_finite_count(execution_metrics, 'SCI')}",
            f"- finite_MSCI_executions: {_finite_count(execution_metrics, 'MSCI')}",
            f"- executions_with_candidate_deceptive_profile_pre: {candidate_count}",
            f"- executions_with_matched_deceptive_cancel_window: {matched_count}",
            f"- candidate_deceptive_order_rows: {candidate_deceptive_orders.height}",
            "",
            "## Top clients by MCPS",
            "",
        ]
    )
    lines.extend(_top_mcps_lines(mcps_scores))
    lines.extend(["", "## Top executions by MSCI", ""])
    lines.extend(_top_execution_lines(execution_metrics))
    lines.extend(["", "## Top matched deceptive-order cancellations", ""])
    lines.extend(_top_deceptive_cancel_lines(execution_metrics))
    output_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    gamma_grid = _parse_float_grid(args.gamma_grid)
    raw_events = pl.read_parquet(args.input)
    raw_events_for_compute = raw_events.head(args.max_rows) if args.max_rows is not None else raw_events

    if args.tick_size is not None:
        tick_size = args.tick_size
    else:
        if args.quote_panel is None:
            raise ValueError("--quote-panel is required unless --tick-size is provided")
        tick_size = infer_tick_size_from_best_quotes(pl.read_parquet(args.quote_panel))

    client_audit = audit_missing_client_trading_capacity(raw_events_for_compute)
    result = compute_exploratory_metrics(
        raw_events_for_compute,
        top_n=args.top_n,
        tick_size=tick_size,
        kappa=args.kappa,
        lambda_=args.lambda_,
        epsilon=args.epsilon,
        window_seconds=args.window_seconds,
        max_deceptive_order_age_seconds=args.max_deceptive_order_age_seconds,
    )
    mcps_scores = compute_mcps_scores(result.execution_metrics, gamma_grid=gamma_grid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "state_time_series": args.output_dir / "client_metric_time_series.parquet",
        "execution_metrics": args.output_dir / "execution_metrics.parquet",
        "candidate_deceptive_orders": args.output_dir / "candidate_deceptive_orders.parquet",
        "rejected_executions": args.output_dir / "rejected_executions.parquet",
        "client_mcps_scores": args.output_dir / "client_mcps_scores.parquet",
        "metadata": args.output_dir / "metadata.json",
        "summary_report": args.output_dir / "summary_report.md",
    }
    _write_parquet(result.state_time_series, paths["state_time_series"])
    _write_parquet(result.execution_metrics, paths["execution_metrics"])
    _write_parquet(result.candidate_deceptive_orders, paths["candidate_deceptive_orders"])

    _write_parquet(result.rejected_executions, paths["rejected_executions"])
    _write_parquet(mcps_scores, paths["client_mcps_scores"])

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "quote_panel": str(args.quote_panel) if args.quote_panel is not None else None,
        "output_dir": str(args.output_dir),
        "top_n": args.top_n,
        "kappa": args.kappa,
        "lambda_": args.lambda_,
        "epsilon": args.epsilon,
        "window_seconds": args.window_seconds,
        "max_deceptive_order_age_seconds": args.max_deceptive_order_age_seconds,
        "gamma_grid": gamma_grid,
        "tick_size": tick_size,
        "identity": "NMSC_ORIGINALCLIENTIDSHORTCODE",
        "client_only": True,
        "market_orders_included": False,
        "event_selection": "matched_deceptive_order_cancellations_only",
        "max_rows": args.max_rows,
        "client_identity_audit": client_audit,
        "row_counts": {
            "input_rows_for_compute": raw_events_for_compute.height,
            "state_time_series": result.state_time_series.height,
            "execution_metrics": result.execution_metrics.height,
            "candidate_deceptive_orders": result.candidate_deceptive_orders.height,

            "rejected_executions": result.rejected_executions.height,
            "client_mcps_scores": mcps_scores.height,
        },
        "paths": {key: str(path) for key, path in paths.items()},
        "command": sys.argv,
    }
    paths["metadata"].write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    _write_summary_report(
        output_path=paths["summary_report"],
        metadata=metadata,
        client_audit=client_audit,
        execution_metrics=result.execution_metrics,
        state_time_series=result.state_time_series,
        candidate_deceptive_orders=result.candidate_deceptive_orders,
        mcps_scores=mcps_scores,
    )

    print(f"output_dir: {args.output_dir}")
    print(f"tick_size: {tick_size}")
    for key in (
        "state_time_series",
        "execution_metrics",
        "candidate_deceptive_orders",

        "rejected_executions",
        "client_mcps_scores",
    ):
        print(f"{key}: {metadata['row_counts'][key]}")
    print(f"metadata: {paths['metadata']}")
    print(f"summary_report: {paths['summary_report']}")


if __name__ == "__main__":
    main()