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
from spoofing_detection.lob.spoofing_metric_plots import write_spoofing_metric_dashboard
from spoofing_detection.lob.spoofing_metrics import (
    MSCI_DEFINITION,
    MSCI_RESTING_PROFILE_DEFINITION,
    MSCI_RANGE,
    RATIO_ZERO_DENOMINATOR_POLICY,
    WITHDRAWAL_PROFILE_SCALE_DEFINITION,
    compute_exploratory_metrics,
    compute_mcps_scores,
    infer_tick_size_from_best_quotes,
)
from spoofing_detection.lob.spoofing_config import (
    DEFAULT_SPOOFING_CONFIG_PATH,
    load_spoofing_config_defaults,
    parse_execution_anchor_modes,
    validate_actor_identity_mode,
)


_CONFIGURABLE_DEFAULT_KEYS = {
    "depth_grid",
    "kappa",
    "lambda_",
    "window_seconds",
    "withdrawal_window_seconds",
    "reversion_horizon_seconds",
    "execution_cluster_max_gap_ms",
    "max_deceptive_order_age_seconds",
    "gamma_grid",
    "tick_size",
    "max_rows",
    "make_dashboard",
    "empirical_depth_kernel",
    "actor_identity_mode",
    "execution_anchor_modes",
}

OUTPUT_SCHEMA_VERSION = "actor_execution_anchor_v2"
SCORE_GROUPING = ["actor_key", "execution_anchor_mode"]
FIRM_FALLBACK_SEMANTICS = "aggregate only when client_original_id is missing"


def _analysis_metadata() -> dict[str, str]:
    return {
        "analytical_unit": "execution_cluster",
        "raw_audit_unit": "child_fill_message",
        "event_selection": "selected_execution_anchor_clusters",
        "behavioral_gate": (
            "rapid_attributed_cancel AND fill_qty_lt_withdrawn_qty AND favorable_pre_fill_mid_move AND "
            "positive_cancel_anchored_mid_reversion"
        ),
        "analytical_event_population": "all_selected_execution_anchor_clusters",
        "mcps_population": "all_attributable_actor_execution_clusters_stratified_by_anchor",
        "review_event_selection": "canonically_assigned_matched_withdrawal_clusters_only",
    }


def _msci_metadata() -> dict[str, Any]:
    return {
        "msci_definition": MSCI_DEFINITION,
        "msci_range": list(MSCI_RANGE),
        "metric_definitions": {
            "MSCI_resting_profile": MSCI_RESTING_PROFILE_DEFINITION,
            "withdrawal_profile_scale_event": WITHDRAWAL_PROFILE_SCALE_DEFINITION,
        },
        "legacy_metric_aliases": {
            "MSCI": "MSCI_resting_profile",
            "WMSCI_event": "withdrawal_profile_scale_event",
            "fill_qty": "execution_quantity",
            "withdrawal_to_fill_ratio": "withdrawal_to_execution_ratio",
        },
        "execution_specific_fields": {
            "common": ["execution_quantity", "execution_vwap", "MSCI_resting_profile"],
            "passive_only": [
                "passive_execution_quantity",
                "passive_execution_vwap",
                "passive_same_level_market_visible_qty_pre",
                "passive_same_level_actor_visible_qty_pre",
                "passive_smallness_fraction_market_level",
                "passive_smallness_fraction_actor_level",
                "WMSCI_passive",
            ],
            "aggressive_only": [
                "aggressive_execution_quantity",
                "aggressive_execution_vwap",
                "aggressive_child_fill_count",
                "aggressive_execution_price_level_count",
                "aggressive_execution_price_min",
                "aggressive_execution_price_max",
                "aggressive_execution_sweep_id",
                "WMSCI_aggressive",
            ],
            "not_applicable_policy": "null_not_zero",
        },
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


def _parse_execution_anchor_modes(text: str) -> tuple[str, ...]:
    return parse_execution_anchor_modes(text)


def _observed_execution_anchor_modes(executions: pl.DataFrame) -> set[str]:
    if executions.is_empty() or "execution_anchor_mode" not in executions.columns:
        return set()
    return set(executions.get_column("execution_anchor_mode").drop_nulls().unique().to_list())


def _depth_output_paths(root: Path, top_n: int) -> dict[str, Path]:
    depth_dir = root / f"topn_{top_n}"
    return {
        "state_time_series": depth_dir / "actor_metric_time_series.parquet",
        "execution_metrics": depth_dir / "execution_metrics.parquet",
        "candidate_deceptive_orders": depth_dir / "candidate_deceptive_orders.parquet",
        "execution_cluster_members": depth_dir / "execution_cluster_members.parquet",
        "execution_cancel_candidates": depth_dir / "execution_cancel_candidates.parquet",
        "spoofing_compatible_events": depth_dir / "spoofing_compatible_events.parquet",
        "rejected_executions": depth_dir / "rejected_executions.parquet",
        "actor_mcps_scores": depth_dir / "actor_mcps_scores.parquet",
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
        "execution_cluster_members",
        "execution_cancel_candidates",
        "spoofing_compatible_events",
        "rejected_executions",
        "actor_mcps_scores",
    )
    return all(paths[name].exists() and paths[name].stat().st_size > 0 for name in required)


def _sha256(path: Path | None) -> str | None:
    if path is None:
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _can_reuse_depth_outputs(
    paths: dict[str, Path],
    *,
    metadata_path: Path,
    expected_metadata: dict[str, Any],
    top_n: int,
) -> bool:
    if not _depth_outputs_complete(paths) or not metadata_path.exists():
        return False
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return top_n in metadata.get("depth_grid", []) and all(
        metadata.get(key) == value for key, value in expected_metadata.items()
    )


def _depth_counts_from_files(paths: dict[str, Path]) -> dict[str, int]:
    state_columns = pl.read_parquet(paths["state_time_series"]).columns
    counts = {
        name: pl.read_parquet(paths[name]).height
        for name in (
            "state_time_series",
            "execution_metrics",
            "candidate_deceptive_orders",
            "execution_cluster_members",
            "execution_cancel_candidates",
            "spoofing_compatible_events",
            "actor_mcps_scores",
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
    parser.add_argument("--window-seconds", type=float, default=1.0, help="Clock-time post-execution window")
    parser.add_argument("--withdrawal-window-seconds", type=float, default=2.0, help="Post-cluster withdrawal outcome window")
    parser.add_argument("--reversion-horizon-seconds", type=float, default=2.0, help="Post-cancellation price-reversion horizon")
    parser.add_argument("--execution-cluster-max-gap-ms", type=int, default=100, help="Maximum inclusive gap between child fills in one execution cluster")
    parser.add_argument(
        "--max-deceptive-order-age-seconds",
        type=float,
        default=600.0,
        help="Maximum age of candidate deceptive orders before the execution, in seconds",
    )
    parser.add_argument("--gamma-grid", default="0,0.1,0.25,0.5,1.0,1.5", help="Comma-separated signed MSCI thresholds")
    parser.add_argument("--tick-size", type=float, default=None, help="Optional explicit tick size")
    parser.add_argument("--max-rows", type=int, default=None, help="Optional raw-row cap for smoke runs")
    parser.add_argument(
        "--actor-identity-mode",
        choices=("client_then_firm",),
        default="client_then_firm",
        help="Resolve original client first, then use namespaced firm fallback only when client is missing.",
    )
    parser.add_argument(
        "--execution-anchor-modes",
        default="passive",
        help="Comma-separated execution branches; canonical order is passive,aggressive.",
    )
    parser.add_argument(
        "--reuse-depth-outputs",
        action="store_true",
        help="Reuse complete per-depth outputs only when their metadata matches all current inputs and parameters",
    )
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
    try:
        args.actor_identity_mode = validate_actor_identity_mode(args.actor_identity_mode)
        args.execution_anchor_modes = _parse_execution_anchor_modes(args.execution_anchor_modes)
    except ValueError as exc:
        parser.error(str(exc))
    if args.empirical_depth_kernel is not None and not isinstance(args.empirical_depth_kernel, Path):
        args.empirical_depth_kernel = Path(args.empirical_depth_kernel)
    if args.execution_cluster_max_gap_ms < 0:
        parser.error("--execution-cluster-max-gap-ms must be non-negative")
    if args.withdrawal_window_seconds <= 0:
        parser.error("--withdrawal-window-seconds must be positive")
    if args.reversion_horizon_seconds <= 0:
        parser.error("--reversion-horizon-seconds must be positive")
    return args


def _grouped_counts(
    frame: pl.DataFrame,
    group_columns: list[str],
    *,
    flag_column: str | None = None,
) -> list[dict[str, Any]]:
    if frame.is_empty() or any(column not in frame.columns for column in group_columns):
        return []
    selected = frame
    if flag_column is not None:
        if flag_column not in selected.columns:
            return []
        selected = selected.filter(pl.col(flag_column).fill_null(False))
    if selected.is_empty():
        return []
    return selected.group_by(group_columns).len(name="rows").sort(group_columns).to_dicts()


def _missing_actor_identity_rows(raw_events: pl.DataFrame) -> int:
    def missing(column: str) -> pl.Expr:
        if column not in raw_events.columns:
            return pl.lit(True)
        text = pl.col(column).cast(pl.Utf8, strict=False).str.strip_chars().str.to_lowercase()
        return pl.col(column).is_null() | text.is_in(["", "nan", "none", "null"])

    return int(
        raw_events.select(
            (missing("NMSC_ORIGINALCLIENTIDSHORTCODE") & missing("FIRMID")).sum()
        ).item()
    )


def _stratified_top_rows(frame: pl.DataFrame, *, limit: int) -> list[dict[str, Any]]:
    ranked = frame.sort(
        ["MCPS_resting_profile", "max_MSCI_resting_profile", "executions"],
        descending=[True, True, True],
    )
    strata = [column for column in ("identity_level", "execution_anchor_mode") if column in ranked.columns]
    actor_column = next(
        (column for column in ("actor_key", "actor_id", "client_id") if column in ranked.columns),
        None,
    )
    if actor_column is not None:
        ranked = ranked.unique(subset=[*strata, actor_column], keep="first", maintain_order=True)
    if not strata:
        return ranked.head(limit).to_dicts()
    rows: list[dict[str, Any]] = []
    strata_rows = sorted(
        ranked.select(strata).unique().iter_rows(named=True),
        key=lambda row: tuple(str(row[column]) for column in strata),
    )
    for stratum in strata_rows:
        predicate = pl.lit(True)
        for column in strata:
            value = stratum[column]
            predicate &= pl.col(column).is_null() if value is None else pl.col(column) == value
        rows.extend(ranked.filter(predicate).head(limit).to_dicts())
    return rows


def _market_observation(execution_anchor_modes: tuple[str, ...]) -> str:
    if set(execution_anchor_modes) == {"passive", "aggressive"}:
        return "both_passive_and_aggressive_execution_branches_when_selected"
    return f"{execution_anchor_modes[0]}_execution_branch_only"


def _actor_execution_audit(
    execution_metrics: pl.DataFrame,
    rejected_executions: pl.DataFrame,
    *,
    rows_missing_client_and_firm_identity: int,
) -> dict[str, Any]:
    return {
        "execution_clusters_by_anchor_mode": _grouped_counts(
            execution_metrics,
            ["execution_anchor_mode"],
        ),
        "execution_clusters_by_identity_level": _grouped_counts(
            execution_metrics,
            ["identity_level"],
        ),
        "rejected_executions_by_reason": _grouped_counts(
            rejected_executions,
            ["reject_reason"],
        ),
        "matched_withdrawal_by_anchor_and_identity": _grouped_counts(
            execution_metrics,
            ["execution_anchor_mode", "identity_level"],
            flag_column="has_matched_deceptive_cancel_window",
        ),
        "strict_sequence_by_anchor_and_identity": _grouped_counts(
            execution_metrics,
            ["execution_anchor_mode", "identity_level"],
            flag_column="spoofing_compatible_sequence",
        ),
        "rows_missing_client_and_firm_identity": rows_missing_client_and_firm_identity,
    }


def _write_grid_summary(path: Path, *, metadata: dict[str, Any], combined_scores: pl.DataFrame) -> None:
    lines = [
        "# Multidepth top-n resting-profile MSCI/MCPS grid",
        "",
        "This grid computes resting-profile MCPS across several book depths.",
        "The scores are surveillance cues, not labels and not proof of intent.",
        "The spoofing-compatible sequence is a descriptive same-episode conjunction, not a statistical test or manipulation label.",
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
        f"- ratio_zero_denominator_policy: {metadata['ratio_zero_denominator_policy']}",
        f"- window_seconds: {metadata['window_seconds']}",
        f"- withdrawal_window_seconds: {metadata['withdrawal_window_seconds']}",
        f"- reversion_horizon_seconds: {metadata['reversion_horizon_seconds']}",
        f"- execution_cluster_max_gap_ms: {metadata['execution_cluster_max_gap_ms']}",
        f"- max_deceptive_order_age_seconds: {metadata['max_deceptive_order_age_seconds']}",
        f"- gamma_grid: {metadata['gamma_grid']}",
        f"- tick_size: {metadata['tick_size']}",
        f"- actor_identity_mode: {metadata['actor_identity_mode']}",
        f"- execution_anchor_modes: {metadata['execution_anchor_modes']}",
        "",
        "## Top actors within identity and execution-anchor strata (best MCPS row per actor)",
        "",
    ]
    if combined_scores.is_empty():
        lines.append("No MCPS rows.")
    else:
        cols = [
            col
            for col in (
                "actor_key",
                "actor_id",
                "identity_level",
                "execution_anchor_mode",
                "top_n",
                "gamma",
                "executions",
                "finite_msci_executions",
                "MCPS_resting_profile",
                "max_MSCI_resting_profile",
            )
            if col in combined_scores.columns
        ]
        rows = [
            {column: row.get(column) for column in cols}
            for row in _stratified_top_rows(combined_scores, limit=25)
        ]
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
    expected_metadata: dict[str, Any] = {
        "input": str(args.input.resolve()),
        "input_sha256": _sha256(args.input),
        "quote_panel": str(args.quote_panel.resolve()) if args.quote_panel is not None else None,
        "quote_panel_sha256": _sha256(args.quote_panel),
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "actor_identity_mode": args.actor_identity_mode,
        "execution_anchor_modes": list(args.execution_anchor_modes),
        "identity": "actor_key",
        "client_only": False,
        "firm_fallback_semantics": FIRM_FALLBACK_SEMANTICS,
        "score_grouping": SCORE_GROUPING,
        "market_orders_included": "aggressive" in args.execution_anchor_modes,
        "market_observation": _market_observation(args.execution_anchor_modes),
        "kappa": args.kappa,
        "lambda_": args.lambda_,
        "ratio_zero_denominator_policy": RATIO_ZERO_DENOMINATOR_POLICY,
        "window_seconds": args.window_seconds,
        "withdrawal_window_seconds": args.withdrawal_window_seconds,
        "reversion_horizon_seconds": args.reversion_horizon_seconds,
        "execution_cluster_max_gap_ms": args.execution_cluster_max_gap_ms,
        "max_deceptive_order_age_seconds": args.max_deceptive_order_age_seconds,
        "gamma_grid": gamma_grid,
        "tick_size": tick_size,
        "max_rows": args.max_rows,
        "empirical_depth_kernel": (
            str(args.empirical_depth_kernel.resolve())
            if args.empirical_depth_kernel is not None
            else None
        ),
        "empirical_depth_kernel_sha256": _sha256(args.empirical_depth_kernel),
        "kernel_mode": "empirical" if args.empirical_depth_kernel is not None else "parametric",
        **_analysis_metadata(),
        **_msci_metadata(),
    }
    combined_score_frames: list[pl.DataFrame] = []
    per_depth_counts: dict[str, dict[str, int]] = {}

    for top_n in depth_grid:
        paths = _depth_output_paths(args.output_dir, top_n)
        if args.reuse_depth_outputs and _can_reuse_depth_outputs(
            paths,
            metadata_path=args.output_dir / "metadata.json",
            expected_metadata=expected_metadata,
            top_n=top_n,
        ):
            scores = pl.read_parquet(paths["actor_mcps_scores"])
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
            window_seconds=args.window_seconds,
            withdrawal_window_seconds=args.withdrawal_window_seconds,
            reversion_horizon_seconds=args.reversion_horizon_seconds,
            execution_cluster_max_gap_ms=args.execution_cluster_max_gap_ms,
            include_level_columns=top_n <= 5,
            max_deceptive_order_age_seconds=args.max_deceptive_order_age_seconds,
            execution_anchor_modes=args.execution_anchor_modes,
            empirical_kernel_weights=empirical_kernel_weights,
        )
        scores = compute_mcps_scores(result.execution_metrics, gamma_grid=gamma_grid)
        _write_parquet(result.state_time_series, paths["state_time_series"])
        _write_parquet(result.execution_metrics, paths["execution_metrics"])
        _write_parquet(result.candidate_deceptive_orders, paths["candidate_deceptive_orders"])
        _write_parquet(result.execution_cluster_members, paths["execution_cluster_members"])
        _write_parquet(result.execution_cancel_candidates, paths["execution_cancel_candidates"])
        _write_parquet(result.spoofing_compatible_events, paths["spoofing_compatible_events"])
        _write_parquet(result.rejected_executions, paths["rejected_executions"])
        _write_parquet(scores, paths["actor_mcps_scores"])
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
            "execution_cluster_members": result.execution_cluster_members.height,
            "execution_cancel_candidates": result.execution_cancel_candidates.height,
            "spoofing_compatible_events": result.spoofing_compatible_events.height,
            "actor_mcps_scores": scores.height,
            "state_level_columns_included": top_n <= 5,
        }

    combined_scores = pl.concat(combined_score_frames, how="diagonal_relaxed") if combined_score_frames else pl.DataFrame()
    combined_path = args.output_dir / "combined_actor_mcps_scores.parquet"
    _write_parquet(combined_scores, combined_path)
    representative_paths = _depth_output_paths(args.output_dir, depth_grid[0])
    representative_execution_metrics = pl.read_parquet(representative_paths["execution_metrics"])
    observed_execution_anchor_modes = _observed_execution_anchor_modes(representative_execution_metrics)
    for top_n in depth_grid[1:]:
        observed_execution_anchor_modes.update(
            _observed_execution_anchor_modes(
                pl.read_parquet(_depth_output_paths(args.output_dir, top_n)["execution_metrics"])
            )
        )
    actor_execution_audit = _actor_execution_audit(
        representative_execution_metrics,
        pl.read_parquet(representative_paths["rejected_executions"]),
        rows_missing_client_and_firm_identity=_missing_actor_identity_rows(raw_events_for_compute),
    )
    actor_execution_audit["representative_top_n"] = depth_grid[0]
    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        **expected_metadata,
        "output_dir": str(args.output_dir),
        "config": str(args.config) if args.config is not None and args.config.exists() else None,
        "depth_grid": depth_grid,
        **_analysis_metadata(),
        "reuse_depth_outputs": args.reuse_depth_outputs,
        "client_identity_audit": client_audit,
        "actor_execution_audit": actor_execution_audit,
        "observed_execution_anchor_modes": list(parse_execution_anchor_modes(observed_execution_anchor_modes))
        if observed_execution_anchor_modes
        else [],
        "per_depth_counts": per_depth_counts,
        "combined_actor_mcps_scores": str(combined_path),
        "command": sys.argv,
    }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    summary_path = args.output_dir / "summary_report.md"
    _write_grid_summary(summary_path, metadata=metadata, combined_scores=combined_scores)

    print(f"output_dir: {args.output_dir}")
    print(f"tick_size: {tick_size}")
    print(f"combined_actor_mcps_scores: {combined_scores.height}")
    print(f"metadata: {metadata_path}")
    print(f"summary_report: {summary_path}")


if __name__ == "__main__":
    main()
