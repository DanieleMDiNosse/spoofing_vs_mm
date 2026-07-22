#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
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
from spoofing_detection.lob.normalize import to_str_or_none
from spoofing_detection.lob.spoofing_metrics import (
    compute_exploratory_metrics,
    compute_mcps_scores,
    infer_tick_size_from_best_quotes,
)
from spoofing_detection.lob.spoofing_config import DEFAULT_SPOOFING_CONFIG_PATH, load_spoofing_config_defaults


_CONFIGURABLE_DEFAULT_KEYS = {
    "top_n",
    "kappa",
    "lambda_",
    "epsilon",
    "window_seconds",
    "withdrawal_window_seconds",
    "reversion_horizon_seconds",
    "execution_cluster_max_gap_ms",
    "max_deceptive_order_age_seconds",
    "gamma_grid",
    "tick_size",
    "max_rows",
    "state_client_mode",
    "compact_state",
    "empirical_depth_kernel",
}


def _population_metadata() -> dict[str, str]:
    return {
        "analytical_event_population": "all_passive_execution_clusters",
        "mcps_population": "all_attributable_client_execution_clusters",
        "review_event_selection": "canonically_assigned_matched_withdrawal_clusters_only",
    }


def _parse_float_grid(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("grid must contain at least one value")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_SPOOFING_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    config_defaults = load_spoofing_config_defaults(
        config_path=config_args.config,
        section="metrics",
        allowed_keys=_CONFIGURABLE_DEFAULT_KEYS,
    )

    parser = argparse.ArgumentParser(description="Compute multilevel top-n spoofing surveillance metrics.")
    parser.add_argument(
        "--config",
        type=Path,
        default=config_args.config,
        help="JSON config file containing spoofing parameter defaults",
    )
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
        "--withdrawal-window-seconds",
        type=float,
        default=2.0,
        help="Maximum delay from execution-cluster end to attributed cancellation",
    )
    parser.add_argument(
        "--reversion-horizon-seconds",
        type=float,
        default=2.0,
        help="Price-reversion horizon measured from each actual cancellation",
    )
    parser.add_argument(
        "--execution-cluster-max-gap-ms",
        type=int,
        default=100,
        help="Maximum gap between passive child fills merged into one execution cluster",
    )
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
        "--state-client-mode",
        choices=("all", "passive-fill-clients"),
        default="all",
        help=(
            "Reduce client_metric_time_series memory by emitting DWI state rows only for clients that can enter "
            "passive fill surveillance events. Use 'all' to preserve the full legacy state table."
        ),
    )
    parser.add_argument(
        "--compact-state",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Omit per-level diagnostic columns from client_metric_time_series while keeping DWI/L_bid/L_ask metrics.",
    )
    parser.set_defaults(**config_defaults)
    args = parser.parse_args(argv)
    if args.empirical_depth_kernel is not None and not isinstance(args.empirical_depth_kernel, Path):
        args.empirical_depth_kernel = Path(args.empirical_depth_kernel)
    if args.execution_cluster_max_gap_ms < 0:
        parser.error("--execution-cluster-max-gap-ms must be non-negative")
    if args.withdrawal_window_seconds <= 0:
        parser.error("--withdrawal-window-seconds must be positive")
    if args.reversion_horizon_seconds <= 0:
        parser.error("--reversion-horizon-seconds must be positive")
    return args


def _infer_state_client_ids(raw_events: pl.DataFrame, *, mode: str) -> set[str] | None:
    if mode == "all":
        return None
    if mode != "passive-fill-clients":
        raise ValueError(f"unknown state client mode: {mode}")

    required = {
        "ORDEREVENTTYPE (*)",
        "PASSIVEORDER",
        "AGGRESSIVEORDER",
        "ORDERTYPE (*)",
        "NMSC_ORIGINALCLIENTIDSHORTCODE",
    }
    missing = sorted(required - set(raw_events.columns))
    if missing:
        raise ValueError(f"cannot infer passive-fill clients; missing columns: {', '.join(missing)}")

    client_rows = raw_events.filter(
        (pl.col("ORDEREVENTTYPE (*)") == 3)
        & (pl.col("PASSIVEORDER").cast(pl.Utf8).str.to_uppercase() == "Y")
        & (pl.col("AGGRESSIVEORDER").cast(pl.Utf8).fill_null("N").str.to_uppercase() != "Y")
        & pl.col("ORDERTYPE (*)").is_in([2, 5])
        & pl.col("NMSC_ORIGINALCLIENTIDSHORTCODE").is_not_null()
    )
    return {
        client_id
        for value in client_rows.get_column("NMSC_ORIGINALCLIENTIDSHORTCODE").unique().to_list()
        if (client_id := to_str_or_none(value)) is not None
    }


def _write_parquet(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path)


def _write_csv(df: pl.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if df.width == 0:
        path.write_text("")
    else:
        df.write_csv(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_count(df: pl.DataFrame, column: str) -> int:
    if df.is_empty() or column not in df.columns:
        return 0
    return int(df.select(pl.col(column).is_not_null().sum()).item())


def _true_count(df: pl.DataFrame, column: str) -> int:
    if df.is_empty() or column not in df.columns:
        return 0
    return int(df.select(pl.col(column).fill_null(False).sum()).item())


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
            "favorable_mid_move_pre_fill",
            "post_cancel_mid_reversion",
            "spoofing_compatible_sequence",
            "execution_price_advantage_vs_posture_mid",
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
            "mean_favorable_mid_move_pre_fill",
            "mean_post_cancel_mid_reversion",
            "mean_execution_price_advantage_vs_posture_mid",
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
        "- SCI is the absolute DWI change from immediately before an execution cluster to the post-cluster window.",
        "- Collapse measures how much weighted liquidity disappears after the cluster on each side of the book.",
        "- MSCI is high only when DWI changes sharply and the opposite side collapses more than the execution side.",
        "- Price-response diagnostics are signed so positive values indicate a movement or execution price advantage favorable to the passive fill side; they are economic consistency checks, not causal proof.",
        "- MCPS is a client-level repetition score: the fraction of execution clusters whose MSCI is above gamma.",
        "- A candidate deceptive profile is the same client's pre-existing visible depth on the side opposite to the execution cluster, posted within the configured pre-execution age window; the name denotes a screening candidate, not proven intent.",
        "- Each raw passive child fill belongs to exactly one cluster; each eligible cancellation is assigned to at most one cluster for metric totals.",
        "- Rapid withdrawal uses its own execution-to-cancel window; it is not the SCI collapse horizon.",
        "- Post-cancel reversion is measured from the state immediately before each assigned physical cancellation to that cancellation's own reversion horizon, then quantity-and-delay weighted within the cluster.",
        "- The spoofing-compatible sequence is a transparent descriptive gate, not a statistical test, score, or intent label: rapid attributed cancellation, fill smaller than withdrawn quantity, favorable pre-fill mid move, and positive cancel-anchored mid reversion must all be present.",
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
        f"- withdrawal_window_seconds: {metadata.get('withdrawal_window_seconds', 2.0)}",
        f"- reversion_horizon_seconds: {metadata.get('reversion_horizon_seconds', 2.0)}",
        f"- execution_cluster_max_gap_ms: {metadata.get('execution_cluster_max_gap_ms', 100)}",
        f"- max_deceptive_order_age_seconds: {metadata['max_deceptive_order_age_seconds']}",
        f"- gamma_grid: {metadata['gamma_grid']}",
        f"- tick_size: {metadata['tick_size']}",
        "- identity: NMSC_ORIGINALCLIENTIDSHORTCODE only",
        "- market orders included: false",
        "- MCPS population: all attributable-client execution clusters",
        "- review-event selection: canonically assigned matched-withdrawal clusters only",
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
            f"- clusters_with_observed_post_window_state: {_true_count(execution_metrics, 'has_post_window_state')}",
            f"- clusters_with_candidate_profile_pre: {candidate_count}",
            f"- clusters_with_assigned_matched_cancel_window: {matched_count}",
            f"- candidate_deceptive_order_rows: {candidate_deceptive_orders.height}",
            "",
            "## Top clients by MCPS",
            "",
        ]
    )
    lines.extend(_top_mcps_lines(mcps_scores))
    lines.extend(["", "## Top execution clusters by MSCI (finite MSCI only)", ""])
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
    state_client_ids = _infer_state_client_ids(raw_events_for_compute, mode=args.state_client_mode)
    empirical_kernel_weights = (
        load_empirical_kernel_weights(args.empirical_depth_kernel) if args.empirical_depth_kernel is not None else None
    )
    result = compute_exploratory_metrics(
        raw_events_for_compute,
        top_n=args.top_n,
        tick_size=tick_size,
        kappa=args.kappa,
        lambda_=args.lambda_,
        epsilon=args.epsilon,
        window_seconds=args.window_seconds,
        withdrawal_window_seconds=args.withdrawal_window_seconds,
        reversion_horizon_seconds=args.reversion_horizon_seconds,
        execution_cluster_max_gap_ms=args.execution_cluster_max_gap_ms,
        max_deceptive_order_age_seconds=args.max_deceptive_order_age_seconds,
        include_level_columns=not args.compact_state,
        state_client_ids=state_client_ids,
        empirical_kernel_weights=empirical_kernel_weights,
    )
    mcps_scores = compute_mcps_scores(result.execution_metrics, gamma_grid=gamma_grid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "state_time_series": args.output_dir / "client_metric_time_series.parquet",
        "execution_metrics": args.output_dir / "execution_metrics.parquet",
        "candidate_deceptive_orders": args.output_dir / "candidate_deceptive_orders.parquet",
        "execution_cluster_members": args.output_dir / "execution_cluster_members.parquet",
        "execution_cancel_candidates": args.output_dir / "execution_cancel_candidates.parquet",
        "spoofing_compatible_events": args.output_dir / "spoofing_compatible_events.parquet",
        "rejected_executions": args.output_dir / "rejected_executions.parquet",
        "client_mcps_scores": args.output_dir / "client_mcps_scores.parquet",
        "metadata": args.output_dir / "metadata.json",
        "summary_report": args.output_dir / "summary_report.md",
    }
    _write_parquet(result.state_time_series, paths["state_time_series"])
    _write_parquet(result.execution_metrics, paths["execution_metrics"])
    _write_parquet(result.candidate_deceptive_orders, paths["candidate_deceptive_orders"])
    _write_parquet(result.execution_cluster_members, paths["execution_cluster_members"])
    _write_parquet(result.execution_cancel_candidates, paths["execution_cancel_candidates"])
    _write_csv(result.execution_cluster_members, args.output_dir / "execution_cluster_members.csv")
    _write_csv(result.execution_cancel_candidates, args.output_dir / "execution_cancel_candidates.csv")
    _write_parquet(result.spoofing_compatible_events, paths["spoofing_compatible_events"])
    _write_csv(
        result.spoofing_compatible_events,
        args.output_dir / "spoofing_compatible_events.csv",
    )
    _write_parquet(result.rejected_executions, paths["rejected_executions"])
    _write_parquet(mcps_scores, paths["client_mcps_scores"])

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "quote_panel": str(args.quote_panel) if args.quote_panel is not None else None,
        "output_dir": str(args.output_dir),
        "config": str(args.config) if args.config is not None and args.config.exists() else None,
        "top_n": args.top_n,
        "kappa": args.kappa,
        "lambda_": args.lambda_,
        "epsilon": args.epsilon,
        "window_seconds": args.window_seconds,
        "withdrawal_window_seconds": args.withdrawal_window_seconds,
        "reversion_horizon_seconds": args.reversion_horizon_seconds,
        "execution_cluster_max_gap_ms": args.execution_cluster_max_gap_ms,
        "analytical_unit": "execution_cluster",
        "raw_audit_unit": "child_fill_message",
        "max_deceptive_order_age_seconds": args.max_deceptive_order_age_seconds,
        "gamma_grid": gamma_grid,
        "tick_size": tick_size,
        "identity": "NMSC_ORIGINALCLIENTIDSHORTCODE",
        "client_only": False,
        "event_metrics_include_unattributable_rows": True,
        "client_scores_exclude_unattributable_rows": True,
        "market_orders_included": False,
        "event_selection": "all_passive_execution_clusters",
        "behavioral_gate": (
            "rapid_attributed_cancel AND fill_qty_lt_withdrawn_qty AND favorable_pre_fill_mid_move AND "
            "positive_cancel_anchored_mid_reversion"
        ),
        **_population_metadata(),
        "max_rows": args.max_rows,
        "state_client_mode": args.state_client_mode,
        "state_client_count": len(state_client_ids) if state_client_ids is not None else None,
        "compact_state": args.compact_state,
        "empirical_depth_kernel": str(args.empirical_depth_kernel) if args.empirical_depth_kernel is not None else None,
        "kernel_mode": "empirical" if args.empirical_depth_kernel is not None else "parametric",
        "client_identity_audit": client_audit,
        "row_counts": {
            "input_rows_for_compute": raw_events_for_compute.height,
            "state_time_series": result.state_time_series.height,
            "execution_metrics": result.execution_metrics.height,
            "candidate_deceptive_orders": result.candidate_deceptive_orders.height,
            "execution_cluster_members": result.execution_cluster_members.height,
            "execution_cancel_candidates": result.execution_cancel_candidates.height,
            "spoofing_compatible_events": result.spoofing_compatible_events.height,
            "rejected_executions": result.rejected_executions.height,
            "client_mcps_scores": mcps_scores.height,
        },
        "paths": {key: str(path) for key, path in paths.items()},
        "input_hashes": {
            "raw_events_sha256": _sha256(args.input),
            "quote_panel_sha256": _sha256(args.quote_panel) if args.quote_panel is not None else None,
            "config_sha256": _sha256(args.config) if args.config is not None and args.config.exists() else None,
            "empirical_depth_kernel_sha256": _sha256(args.empirical_depth_kernel)
            if args.empirical_depth_kernel is not None
            else None,
        },
        "artifact_hashes": {
            key: _sha256(path)
            for key, path in paths.items()
            if key not in {"metadata", "summary_report"} and path.exists()
        },
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