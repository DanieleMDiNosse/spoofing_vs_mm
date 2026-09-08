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

from spoofing_detection.lob.actor_identity import (
    normalize_client_original_identity_value,
    resolve_actor_identity,
)
from spoofing_detection.lob.client_identity_audit import audit_missing_client_trading_capacity
from spoofing_detection.lob.depth_kernel_calibration import load_empirical_kernel_weights
from spoofing_detection.lob.enums import normalize_enum_code
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
    reject_parameter_overrides,
    spoofing_config_provenance,
    validate_actor_identity_mode,
)


_CONFIGURABLE_DEFAULT_KEYS = {
    "top_n",
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
    "actor_identity_mode",
    "execution_anchor_modes",
}

_CONFIG_PARAMETER_OPTIONS = {
    "--top-n",
    "--window-seconds",
    "--withdrawal-window-seconds",
    "--reversion-horizon-seconds",
    "--execution-cluster-max-gap-ms",
    "--max-deceptive-order-age-seconds",
    "--gamma-grid",
    "--tick-size",
    "--max-rows",
    "--state-client-mode",
    "--compact-state",
    "--no-compact-state",
    "--empirical-depth-kernel",
    "--actor-identity-mode",
    "--execution-anchor-modes",
}


def _population_metadata() -> dict[str, str]:
    return {
        "analytical_event_population": "all_selected_execution_anchor_clusters",
        "mcps_population": "all_attributable_actor_execution_clusters_stratified_by_anchor",
        "review_event_selection": "canonically_assigned_matched_withdrawal_clusters_only",
    }


OUTPUT_SCHEMA_VERSION = "actor_execution_anchor_v2"
SCORE_GROUPING = ["actor_key", "execution_anchor_mode"]
FIRM_FALLBACK_SEMANTICS = "aggregate only when client_original_id is missing"


def _parse_float_grid(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("grid must contain at least one value")
    return values


def _parse_execution_anchor_modes(text: str) -> tuple[str, ...]:
    return parse_execution_anchor_modes(text)


def _normalized_enum_code(column: str) -> pl.Expr:
    return pl.col(column).map_elements(normalize_enum_code, return_dtype=pl.Int64)


def _observed_execution_anchor_modes(executions: pl.DataFrame) -> list[str]:
    if executions.is_empty() or "execution_anchor_mode" not in executions.columns:
        return []
    observed = executions.get_column("execution_anchor_mode").drop_nulls().unique().to_list()
    return list(parse_execution_anchor_modes(observed)) if observed else []


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser.add_argument("--config", type=Path, default=DEFAULT_SPOOFING_CONFIG_PATH)
    config_args, _ = config_parser.parse_known_args(argv)
    try:
        config_defaults = load_spoofing_config_defaults(
            config_path=config_args.config,
            section="metrics",
            allowed_keys=_CONFIGURABLE_DEFAULT_KEYS,
        )
    except (OSError, ValueError) as exc:
        config_parser.error(str(exc))

    parser = argparse.ArgumentParser(
        description="Compute multilevel top-n spoofing surveillance metrics.",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=config_args.config,
        help="Authoritative JSON file containing all spoofing metric parameters",
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw input parquet event file")
    parser.add_argument(
        "--quote-panel",
        type=Path,
        default=None,
        help="Reconstructed lob_event_state_panel.parquet used only for tick-size inference",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--top-n", type=int, help="Market top-N levels used for actor depth profiles")

    parser.add_argument("--window-seconds", type=float, help="Clock-time post-execution window")
    parser.add_argument(
        "--withdrawal-window-seconds",
        type=float,
        help="Maximum delay from execution-cluster end to attributed cancellation",
    )
    parser.add_argument(
        "--reversion-horizon-seconds",
        type=float,
        help="Price-reversion horizon measured from each actual cancellation",
    )
    parser.add_argument(
        "--execution-cluster-max-gap-ms",
        type=int,
        help="Maximum gap between same-anchor child fills merged into one execution cluster",
    )
    parser.add_argument(
        "--max-deceptive-order-age-seconds",
        type=float,
        help="Maximum age of candidate deceptive orders before the execution, in seconds",
    )
    parser.add_argument("--gamma-grid", help="Comma-separated signed MSCI thresholds")
    parser.add_argument("--tick-size", type=float, help="Optional explicit tick size")
    parser.add_argument("--max-rows", type=int, help="Optional raw-row cap for smoke runs")
    parser.add_argument(
        "--actor-identity-mode",
        choices=("client_then_firm",),
        help="Resolve the original client shortcode first, then use firm fallback only when client is missing.",
    )
    parser.add_argument(
        "--execution-anchor-modes",
        help="Comma-separated execution branches; canonical order is passive,aggressive.",
    )
    parser.add_argument(
        "--empirical-depth-kernel",
        type=Path,
        help="Required empirical_depth_kernel parquet/csv artifact containing rank weights for both sides.",
    )
    parser.add_argument(
        "--state-client-mode",
        choices=("all", "execution-actors", "passive-fill-clients"),
        help=(
            "Choose actors represented in actor_metric_time_series. 'execution-actors' retains every client/firm "
            "actor observed on a selected passive/aggressive fill while avoiding state rows for non-executing actors; "
            "'all' emits all active actor profiles; 'passive-fill-clients' is the legacy client-only filter."
        ),
    )
    parser.add_argument(
        "--compact-state",
        action=argparse.BooleanOptionalAction,
        help="Omit per-level diagnostic columns from actor_metric_time_series while keeping DWI/L_bid/L_ask metrics.",
    )
    parser.set_defaults(**config_defaults)
    reject_parameter_overrides(
        parser,
        argv,
        parameter_options=_CONFIG_PARAMETER_OPTIONS,
    )
    args = parser.parse_args(argv)
    try:
        args.actor_identity_mode = validate_actor_identity_mode(args.actor_identity_mode)
        args.execution_anchor_modes = _parse_execution_anchor_modes(args.execution_anchor_modes)
    except ValueError as exc:
        parser.error(str(exc))
    valid_state_client_modes = ("all", "execution-actors", "passive-fill-clients")
    if args.state_client_mode not in valid_state_client_modes:
        parser.error(
            "--state-client-mode must be one of: " + ", ".join(valid_state_client_modes)
        )
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
        (_normalized_enum_code("ORDEREVENTTYPE (*)") == 3)
        & (pl.col("PASSIVEORDER").cast(pl.Utf8).str.to_uppercase() == "Y")
        & (pl.col("AGGRESSIVEORDER").cast(pl.Utf8).fill_null("N").str.to_uppercase() != "Y")
        & _normalized_enum_code("ORDERTYPE (*)").is_in([2, 5])
        & pl.col("NMSC_ORIGINALCLIENTIDSHORTCODE").is_not_null()
    )
    return {
        client_id
        for value in client_rows.get_column("NMSC_ORIGINALCLIENTIDSHORTCODE").unique().to_list()
        if (client_id := normalize_client_original_identity_value(value)) is not None
    }


def _infer_state_actor_keys(
    raw_events: pl.DataFrame,
    *,
    mode: str,
    execution_anchor_modes: tuple[str, ...],
) -> set[str] | None:
    if mode == "all":
        return None
    if mode == "passive-fill-clients":
        client_ids = _infer_state_client_ids(raw_events, mode=mode)
        return {f"client_original:{client_id}" for client_id in client_ids or ()}
    if mode != "execution-actors":
        raise ValueError(f"unknown state actor mode: {mode}")

    required = {
        "ORDEREVENTTYPE (*)",
        "PASSIVEORDER",
        "AGGRESSIVEORDER",
        "NMSC_ORIGINALCLIENTIDSHORTCODE",
        "FIRMID",
    }
    missing = sorted(required - set(raw_events.columns))
    if missing:
        raise ValueError(f"cannot infer execution actors; missing columns: {', '.join(missing)}")

    passive_flag = pl.col("PASSIVEORDER").cast(pl.String).fill_null("").str.to_uppercase() == "Y"
    aggressive_flag = pl.col("AGGRESSIVEORDER").cast(pl.String).fill_null("").str.to_uppercase() == "Y"
    selected_role = pl.lit(False)
    if "passive" in execution_anchor_modes:
        selected_role = selected_role | (passive_flag & ~aggressive_flag)
    if "aggressive" in execution_anchor_modes:
        selected_role = selected_role | (aggressive_flag & ~passive_flag)

    identity_rows = (
        raw_events.filter((_normalized_enum_code("ORDEREVENTTYPE (*)") == 3) & selected_role)
        .select("NMSC_ORIGINALCLIENTIDSHORTCODE", "FIRMID")
        .unique()
    )
    actor_keys: set[str] = set()
    for row in identity_rows.iter_rows(named=True):
        identity = resolve_actor_identity(
            client_original_id=row["NMSC_ORIGINALCLIENTIDSHORTCODE"],
            firm_id=row["FIRMID"],
        )
        if identity is not None:
            actor_keys.add(identity.actor_key)
    return actor_keys


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


def _markdown_table(rows: list[dict[str, Any]], cols: list[str]) -> list[str]:
    if not rows:
        return ["No rows to show."]
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    out = [header, sep]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col)) for col in cols) + " |")
    return out


def _stratified_top_rows(
    frame: pl.DataFrame,
    *,
    columns: list[str],
    sort_by: list[str],
    descending: list[bool],
    limit: int,
) -> list[dict[str, Any]]:
    ranked = frame.sort(sort_by, descending=descending)
    strata = [column for column in ("identity_level", "execution_anchor_mode") if column in ranked.columns]
    if not strata:
        return ranked.head(limit).select(columns).to_dicts()

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
        rows.extend(ranked.filter(predicate).head(limit).select(columns).to_dicts())
    return rows


def _top_execution_lines(execution_metrics: pl.DataFrame, limit: int = 20) -> list[str]:
    if execution_metrics.is_empty() or "MSCI_resting_profile" not in execution_metrics.columns:
        return ["No eligible executions."]
    cols = [
        col
        for col in (
            "sort_index",
            "event_ts",
            "actor_id",
            "identity_level",
            "execution_anchor_mode",
            "execution_side",
            "deceptive_side",
            "execution_quantity",
            "DWI_pre_window",
            "DWI_post_window",
            "SCI",
            "collapse_opposite_side",
            "collapse_same_side",
            "MSCI_resting_profile",
            "favorable_mid_move_pre_fill",
            "post_cancel_mid_reversion",
            "spoofing_compatible_sequence",
            "execution_price_advantage_vs_posture_mid",
            "candidate_deceptive_visible_qty_pre",
            "has_matched_deceptive_cancel_window",
        )
        if col in execution_metrics.columns
    ]
    rows = _stratified_top_rows(
        execution_metrics.filter(pl.col("MSCI_resting_profile").is_not_null()),
        columns=cols,
        sort_by=["MSCI_resting_profile"],
        descending=[True],
        limit=limit,
    )
    return (
        _markdown_table(rows, cols)
        if rows
        else ["No executions with finite MSCI_resting_profile."]
    )


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
            "actor_id",
            "identity_level",
            "execution_anchor_mode",
            "execution_side",
            "deceptive_side",
            "execution_quantity",
            "candidate_deceptive_visible_qty_pre",
            "matched_deceptive_cancel_visible_qty_window",
            "matched_deceptive_cancel_order_ids_window",
            "matched_deceptive_cancel_fraction_window",
            "MSCI_resting_profile",
            "withdrawal_profile_scale_event",
        )
        if col in matched.columns
    ]
    rows = _stratified_top_rows(
        matched,
        columns=cols,
        sort_by=[
            "matched_deceptive_cancel_visible_qty_window",
            "MSCI_resting_profile",
        ],
        descending=[True, True],
        limit=limit,
    )
    return _markdown_table(rows, cols)


def _top_mcps_lines(mcps_scores: pl.DataFrame, limit: int = 20) -> list[str]:
    if mcps_scores.is_empty():
        return ["No MCPS rows."]
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
            "msci_above_gamma_count",
            "MCPS_resting_profile",
            "max_MSCI_resting_profile",
            "mean_MSCI_resting_profile",
            "mean_favorable_mid_move_pre_fill",
            "mean_post_cancel_mid_reversion",
            "mean_execution_price_advantage_vs_posture_mid",
            "candidate_profile_share",
            "matched_deceptive_cancel_share",
        )
        if col in mcps_scores.columns
    ]
    rows = _stratified_top_rows(
        mcps_scores,
        columns=cols,
        sort_by=["MCPS_resting_profile", "max_MSCI_resting_profile", "executions"],
        descending=[True, True, True],
        limit=limit,
    )
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
        "- DWI tells whether an actor is ask-heavy or bid-heavy in the weighted top-n book profile.",
        "- SCI is the absolute DWI change from immediately before an execution cluster to the post-cluster window.",
        "- Collapse measures how much weighted liquidity disappears after the cluster on each side of the book.",
        "- MSCI_resting_profile is the common signed resting-profile contrast SCI / 2 + opposite-side collapse - same-side collapse; it ranges from -1 to 2, and negative values mean same-side collapse dominates. MSCI is retained only as a legacy alias.",
        "- withdrawal_profile_scale_event measures assigned resting withdrawal relative to branch-specific executed quantity. WMSCI_event is retained only as a legacy alias and is not a transformed MSCI.",
        "- Passive-only same-level and resting-fill diagnostics are null for aggressive anchors; aggressive sweep diagnostics are null for passive anchors.",
        "- Price-response diagnostics are signed so positive values indicate a movement or execution price advantage favorable to the passive fill side; they are economic consistency checks, not causal proof.",
        "- MCPS_resting_profile is an actor-and-anchor repetition score: the fraction of execution clusters whose MSCI_resting_profile is above gamma. MCPS is retained only as a legacy alias.",
        "- A candidate deceptive profile is the same actor's pre-existing visible depth on the side opposite to the execution cluster, posted within the configured pre-execution age window; the name denotes a screening candidate, not proven intent.",
        "- Each eligible child fill belongs to exactly one branch-specific cluster; each eligible cancellation is assigned to at most one cluster for metric totals.",
        "- Rapid withdrawal uses its own execution-to-cancel window; it is not the SCI collapse horizon.",
        "- Post-cancel reversion is measured from the state immediately before each assigned physical cancellation to that cancellation's own reversion horizon, then quantity-and-delay weighted within the cluster.",
        "- The spoofing-compatible sequence is a transparent descriptive gate, not a statistical test, score, or intent label: rapid attributed cancellation, fill smaller than withdrawn quantity, favorable pre-fill mid move, and positive cancel-anchored mid reversion must all be present.",
        "",
        "## Parameters",
        "",
        f"- input: `{metadata['input']}`",
        f"- quote_panel: `{metadata.get('quote_panel')}`",
        f"- top_n: {metadata['top_n']}",

        f"- ratio_zero_denominator_policy: {metadata['ratio_zero_denominator_policy']}",
        f"- window_seconds: {metadata['window_seconds']}",
        f"- withdrawal_window_seconds: {metadata.get('withdrawal_window_seconds', 2.0)}",
        f"- reversion_horizon_seconds: {metadata.get('reversion_horizon_seconds', 2.0)}",
        f"- execution_cluster_max_gap_ms: {metadata.get('execution_cluster_max_gap_ms', 100)}",
        f"- max_deceptive_order_age_seconds: {metadata['max_deceptive_order_age_seconds']}",
        f"- gamma_grid: {metadata['gamma_grid']}",
        f"- tick_size: {metadata['tick_size']}",
        f"- output_schema_version: {metadata['output_schema_version']}",
        f"- actor_identity_mode: {metadata['actor_identity_mode']}",
        f"- execution_anchor_modes: {metadata['execution_anchor_modes']}",
        f"- firm_fallback_semantics: {metadata['firm_fallback_semantics']}",
        "- MCPS population: all attributable actor execution clusters, stratified by anchor",
        "- review-event selection: canonically assigned matched-withdrawal clusters only",
        "",
        "## Client identity audit",
        "",
    ]
    lines.extend(_client_audit_lines(client_audit))
    lines.extend(["", "## Actor identity and execution-anchor audit", ""])
    for key, value in metadata.get("actor_execution_audit", {}).items():
        lines.append(f"- {key}: {json.dumps(value, sort_keys=True, default=str)}")
    lines.extend(["", "## Row counts", ""])
    for key, value in metadata["row_counts"].items():
        lines.append(f"- {key}: {value}")
    lines.extend(
        [
            f"- actors_with_topN_profile: {state_time_series.get_column('actor_key').n_unique() if not state_time_series.is_empty() and 'actor_key' in state_time_series.columns else 0}",
            f"- finite_SCI_executions: {_finite_count(execution_metrics, 'SCI')}",
            f"- finite_MSCI_resting_profile_executions: {_finite_count(execution_metrics, 'MSCI_resting_profile')}",
            f"- clusters_with_observed_post_window_state: {_true_count(execution_metrics, 'has_post_window_state')}",
            f"- clusters_with_candidate_profile_pre: {candidate_count}",
            f"- clusters_with_assigned_matched_cancel_window: {matched_count}",
            f"- candidate_deceptive_order_rows: {candidate_deceptive_orders.height}",
            "",
            "## Top actors by resting-profile MCPS, stratified by identity level and execution anchor",
            "",
        ]
    )
    lines.extend(_top_mcps_lines(mcps_scores))
    lines.extend(
        [
            "",
            "## Top execution clusters by resting-profile MSCI, stratified by identity level and execution anchor",
            "",
        ]
    )
    lines.extend(_top_execution_lines(execution_metrics))
    lines.extend(["", "## Top matched deceptive-order cancellations, stratified by identity level and execution anchor", ""])
    lines.extend(_top_deceptive_cancel_lines(execution_metrics))
    output_path.write_text("\n".join(lines) + "\n")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.empirical_depth_kernel is None:
        raise ValueError("empirical_depth_kernel must be configured; the parametric kernel fallback has been removed")
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
    state_actor_keys = _infer_state_actor_keys(
        raw_events_for_compute,
        mode=args.state_client_mode,
        execution_anchor_modes=args.execution_anchor_modes,
    )
    empirical_kernel_weights = load_empirical_kernel_weights(args.empirical_depth_kernel)
    result = compute_exploratory_metrics(
        raw_events_for_compute,
        top_n=args.top_n,
        tick_size=tick_size,

        window_seconds=args.window_seconds,
        withdrawal_window_seconds=args.withdrawal_window_seconds,
        reversion_horizon_seconds=args.reversion_horizon_seconds,
        execution_cluster_max_gap_ms=args.execution_cluster_max_gap_ms,
        max_deceptive_order_age_seconds=args.max_deceptive_order_age_seconds,
        include_level_columns=not args.compact_state,
        state_actor_keys=state_actor_keys,
        execution_anchor_modes=args.execution_anchor_modes,
        empirical_kernel_weights=empirical_kernel_weights,
    )
    mcps_scores = compute_mcps_scores(result.execution_metrics, gamma_grid=gamma_grid)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "state_time_series": args.output_dir / "actor_metric_time_series.parquet",
        "execution_metrics": args.output_dir / "execution_metrics.parquet",
        "candidate_deceptive_orders": args.output_dir / "candidate_deceptive_orders.parquet",
        "execution_cluster_members": args.output_dir / "execution_cluster_members.parquet",
        "execution_cancel_candidates": args.output_dir / "execution_cancel_candidates.parquet",
        "spoofing_compatible_events": args.output_dir / "spoofing_compatible_events.parquet",
        "rejected_executions": args.output_dir / "rejected_executions.parquet",
        "actor_mcps_scores": args.output_dir / "actor_mcps_scores.parquet",
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
    _write_parquet(mcps_scores, paths["actor_mcps_scores"])

    metadata: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "quote_panel": str(args.quote_panel) if args.quote_panel is not None else None,
        "output_dir": str(args.output_dir),
        **spoofing_config_provenance(args.config, section="metrics"),
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "actor_identity_mode": args.actor_identity_mode,
        "execution_anchor_modes": list(args.execution_anchor_modes),
        "observed_execution_anchor_modes": _observed_execution_anchor_modes(result.execution_metrics),
        "score_grouping": SCORE_GROUPING,
        "firm_fallback_semantics": FIRM_FALLBACK_SEMANTICS,
        "top_n": args.top_n,

        "ratio_zero_denominator_policy": RATIO_ZERO_DENOMINATOR_POLICY,
        "window_seconds": args.window_seconds,
        "withdrawal_window_seconds": args.withdrawal_window_seconds,
        "reversion_horizon_seconds": args.reversion_horizon_seconds,
        "execution_cluster_max_gap_ms": args.execution_cluster_max_gap_ms,
        "analytical_unit": "execution_cluster",
        "raw_audit_unit": "child_fill_message",
        "max_deceptive_order_age_seconds": args.max_deceptive_order_age_seconds,
        "msci_definition": MSCI_DEFINITION,
        "metric_definitions": {
            "MSCI_resting_profile": MSCI_RESTING_PROFILE_DEFINITION,
            "withdrawal_profile_scale_event": WITHDRAWAL_PROFILE_SCALE_DEFINITION,
        },
        "legacy_metric_aliases": {
            "MSCI": "MSCI_resting_profile",
            "WMSCI_event": "withdrawal_profile_scale_event",
            "fill_qty": "execution_quantity",
            "event_price": "execution_vwap",
        },
        "execution_branch_applicability": {
            "common_resting_profile": [
                "MSCI_resting_profile",
                "SCI",
                "collapse_opposite_side",
                "collapse_same_side",
                "withdrawal_profile_scale_event",
            ],
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
            "non_applicable_policy": "null_not_zero",
        },
        "msci_range": list(MSCI_RANGE),
        "gamma_grid": gamma_grid,
        "tick_size": tick_size,
        "identity": "actor_key",
        "client_only": False,
        "event_metrics_include_unattributable_rows": True,
        "actor_scores_exclude_unattributable_rows": True,
        "market_orders_included": "aggressive" in args.execution_anchor_modes,
        "event_selection": "selected_execution_anchor_clusters",
        "behavioral_gate": (
            "rapid_attributed_cancel AND fill_qty_lt_withdrawn_qty AND favorable_pre_fill_mid_move AND "
            "positive_cancel_anchored_mid_reversion"
        ),
        **_population_metadata(),
        "max_rows": args.max_rows,
        "state_client_mode": args.state_client_mode,
        "state_client_count": (
            len(state_actor_keys)
            if args.state_client_mode == "passive-fill-clients" and state_actor_keys is not None
            else None
        ),
        "state_actor_selection_mode": args.state_client_mode,
        "state_actor_count": len(state_actor_keys) if state_actor_keys is not None else None,
        "compact_state": args.compact_state,
        "empirical_depth_kernel": str(args.empirical_depth_kernel),
        "kernel_mode": "empirical",
        "client_identity_audit": client_audit,
        "actor_execution_audit": {
            "execution_clusters_by_anchor_mode": _grouped_counts(
                result.execution_metrics,
                ["execution_anchor_mode"],
            ),
            "execution_clusters_by_identity_level": _grouped_counts(
                result.execution_metrics,
                ["identity_level"],
            ),
            "rejected_executions_by_reason": _grouped_counts(
                result.rejected_executions,
                ["reject_reason"],
            ),
            "matched_withdrawal_by_anchor_and_identity": _grouped_counts(
                result.execution_metrics,
                ["execution_anchor_mode", "identity_level"],
                flag_column="has_matched_deceptive_cancel_window",
            ),
            "strict_sequence_by_anchor_and_identity": _grouped_counts(
                result.execution_metrics,
                ["execution_anchor_mode", "identity_level"],
                flag_column="spoofing_compatible_sequence",
            ),
            "rows_missing_client_and_firm_identity": _missing_actor_identity_rows(raw_events_for_compute),
        },
        "row_counts": {
            "input_rows_for_compute": raw_events_for_compute.height,
            "state_time_series": result.state_time_series.height,
            "execution_metrics": result.execution_metrics.height,
            "candidate_deceptive_orders": result.candidate_deceptive_orders.height,
            "execution_cluster_members": result.execution_cluster_members.height,
            "execution_cancel_candidates": result.execution_cancel_candidates.height,
            "spoofing_compatible_events": result.spoofing_compatible_events.height,
            "rejected_executions": result.rejected_executions.height,
            "actor_mcps_scores": mcps_scores.height,
        },
        "paths": {key: str(path) for key, path in paths.items()},
        "input_hashes": {
            "raw_events_sha256": _sha256(args.input),
            "quote_panel_sha256": _sha256(args.quote_panel) if args.quote_panel is not None else None,
            "config_sha256": _sha256(args.config) if args.config is not None and args.config.exists() else None,
            "empirical_depth_kernel_sha256": _sha256(args.empirical_depth_kernel),
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
        "actor_mcps_scores",
    ):
        print(f"{key}: {metadata['row_counts'][key]}")
    print(f"metadata: {paths['metadata']}")
    print(f"summary_report: {paths['summary_report']}")


if __name__ == "__main__":
    main()