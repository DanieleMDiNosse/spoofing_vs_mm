#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.io import reconstruct_file
from spoofing_detection.lob.panel import reconstruct_dataframe


@dataclass(frozen=True)
class Problem:
    severity: str
    problem: str
    evidence: str
    count: int
    example_sort_index: int | None
    possible_fix: str
    status: str = "open"


@dataclass(frozen=True)
class WarningDetail:
    flag: str
    source: str
    status: str
    explanation: str
    recommended_action: str
    observed_evidence: dict[str, Any]
    examples: list[dict[str, Any]]


def _bool_sum(df: pl.DataFrame, column: str) -> int:
    if column not in df.columns or df.is_empty():
        return 0
    return int(df.select(pl.col(column).fill_null(False).sum()).item())


def _first_sort_index(df: pl.DataFrame) -> int | None:
    if df.is_empty() or "sort_index" not in df.columns:
        return None
    value = df.select(pl.col("sort_index").first()).item()
    return int(value) if value is not None else None


def _escape_markdown_cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ")


def check_row_counts(
    *,
    input_rows: int,
    panel_rows: int,
    normalized_rows: int,
    agent_rows: int,
) -> list[Problem]:
    problems: list[Problem] = []
    if panel_rows != input_rows:
        problems.append(
            Problem(
                severity="error",
                problem="panel_rows does not equal input_rows",
                evidence=f"input_rows={input_rows}, panel_rows={panel_rows}",
                count=abs(panel_rows - input_rows),
                example_sort_index=None,
                possible_fix=(
                    "Inspect sort/filter logic in reconstruct_dataframe(); rerun reconstruction "
                    "without max_rows and ensure include_all_events remains true."
                ),
            )
        )
    if normalized_rows != input_rows:
        problems.append(
            Problem(
                severity="error",
                problem="normalized_rows does not equal input_rows",
                evidence=f"input_rows={input_rows}, normalized_rows={normalized_rows}",
                count=abs(normalized_rows - input_rows),
                example_sort_index=None,
                possible_fix="Inspect normalize_event failures and strict enum handling before panel construction.",
            )
        )
    expected_agent_rows = input_rows * 2
    if agent_rows != expected_agent_rows:
        problems.append(
            Problem(
                severity="error",
                problem="agent_event_state_rows does not equal input_rows * 2",
                evidence=f"expected={expected_agent_rows}, observed={agent_rows}",
                count=abs(agent_rows - expected_agent_rows),
                example_sort_index=None,
                possible_fix="Inspect agent_long_rows() and configured agent dimensions in LOBConfig.",
            )
        )
    return problems


def check_spread_flags(panel: pl.DataFrame) -> list[Problem]:
    checks = [
        (
            "book_crossed_pre_flag",
            "Crossed pre-book state",
            "Inspect previous event mutation; active book is already crossed before this event.",
        ),
        (
            "book_crossed_post_flag",
            "Crossed post-book state",
            "Inspect marketable order lifecycle handling, deferred residual flushing, and stop-limit visibility.",
        ),
        (
            "book_locked_pre_flag",
            "Locked pre-book state",
            "Inspect previous event mutation; active book is already locked before this event.",
        ),
        (
            "book_locked_post_flag",
            "Locked post-book state",
            "Inspect marketable New/Fill residual handling and same-price bid/ask active depth.",
        ),
    ]
    problems: list[Problem] = []
    for column, label, fix in checks:
        count = _bool_sum(panel, column)
        if count:
            examples = panel.filter(pl.col(column).fill_null(False))
            problems.append(
                Problem(
                    severity="error",
                    problem=label,
                    evidence=f"{column} true on {count} rows",
                    count=count,
                    example_sort_index=_first_sort_index(examples),
                    possible_fix=fix,
                )
            )

    for prefix in ("pre", "post"):
        bid_col = f"{prefix}_best_bid"
        ask_col = f"{prefix}_best_ask"
        if bid_col not in panel.columns or ask_col not in panel.columns:
            continue
        bad = panel.filter(
            pl.col(bid_col).is_not_null()
            & pl.col(ask_col).is_not_null()
            & (pl.col(bid_col) >= pl.col(ask_col))
        )
        if bad.height:
            problems.append(
                Problem(
                    severity="error",
                    problem=f"{prefix} best bid is not strictly below best ask",
                    evidence=f"{bid_col} >= {ask_col} on {bad.height} rows",
                    count=bad.height,
                    example_sort_index=_first_sort_index(bad),
                    possible_fix=(
                        "Trace the example event and active depth around it; if the row is marketable or conditional, "
                        "tighten non-resting classification or deferred residual flushing."
                    ),
                )
            )
    return problems


def check_top_n_depth(panel: pl.DataFrame, *, top_n: int) -> list[Problem]:
    problems: list[Problem] = []
    for prefix in ("pre", "post"):
        for side in ("bid", "ask"):
            best_col = f"{prefix}_best_{side}"
            level_1_col = f"{prefix}_{side}_level_1_price"
            if best_col in panel.columns and level_1_col in panel.columns:
                bad_best = panel.filter(
                    pl.col(best_col).is_not_null()
                    & pl.col(level_1_col).is_not_null()
                    & (pl.col(best_col) != pl.col(level_1_col))
                )
                if bad_best.height:
                    problems.append(
                        Problem(
                            severity="error",
                            problem=f"{prefix} {side} level 1 price does not equal best {side}",
                            evidence=f"{best_col} != {level_1_col} on {bad_best.height} rows",
                            count=bad_best.height,
                            example_sort_index=_first_sort_index(bad_best),
                            possible_fix="Inspect book_summary() best-price and top-level emission consistency.",
                        )
                    )

            for level in range(1, top_n + 1):
                qty_col = f"{prefix}_{side}_level_{level}_visible_qty"
                if qty_col in panel.columns:
                    bad_qty = panel.filter(pl.col(qty_col).is_not_null() & (pl.col(qty_col) < 0))
                    if bad_qty.height:
                        problems.append(
                            Problem(
                                severity="error",
                                problem=f"Negative {prefix} {side} visible quantity at level {level}",
                                evidence=f"{qty_col} < 0 on {bad_qty.height} rows",
                                count=bad_qty.height,
                                example_sort_index=_first_sort_index(bad_qty),
                                possible_fix="Inspect ActiveOrder quantity normalization and fill/cancel mutation logic.",
                            )
                        )
            for level in range(1, top_n):
                left = f"{prefix}_{side}_level_{level}_price"
                right = f"{prefix}_{side}_level_{level + 1}_price"
                if left not in panel.columns or right not in panel.columns:
                    continue
                if side == "bid":
                    bad_order = panel.filter(
                        pl.col(left).is_not_null()
                        & pl.col(right).is_not_null()
                        & (pl.col(left) <= pl.col(right))
                    )
                    expected = "strictly descending"
                else:
                    bad_order = panel.filter(
                        pl.col(left).is_not_null()
                        & pl.col(right).is_not_null()
                        & (pl.col(left) >= pl.col(right))
                    )
                    expected = "strictly ascending"
                if bad_order.height:
                    problems.append(
                        Problem(
                            severity="error",
                            problem=f"{prefix} {side} price levels are not {expected}",
                            evidence=f"{left}/{right} violate order on {bad_order.height} rows",
                            count=bad_order.height,
                            example_sort_index=_first_sort_index(bad_order),
                            possible_fix=(
                                "Inspect book_summary() grouping/sorting and same-price aggregation before top-N emission."
                            ),
                        )
                    )
    return problems


def _explode_flag_counts(df: pl.DataFrame, column: str) -> pl.DataFrame:
    if column not in df.columns or df.is_empty():
        return pl.DataFrame({"flag": [], "len": [], "example_sort_index": []})
    return (
        df.filter(pl.col(column).is_not_null())
        .select(["sort_index", pl.col(column).cast(pl.Utf8).str.split(";").alias("flag")])
        .explode("flag")
        .filter(pl.col("flag").is_not_null() & (pl.col("flag") != ""))
        .group_by("flag")
        .agg(
            pl.len().alias("len"),
            pl.col("sort_index").first().alias("example_sort_index"),
        )
        .sort("len", descending=True)
    )


def check_issue_flags(panel: pl.DataFrame, normalized: pl.DataFrame) -> list[Problem]:
    problems: list[Problem] = []
    lob_counts = _explode_flag_counts(panel, "lob_issue_flags")
    for row in lob_counts.iter_rows(named=True):
        flag = row["flag"]
        count = int(row["len"])
        severity = "warning" if flag in {"marketable_order_not_resting", "modify_for_unseen_order"} else "error"
        if flag == "marketable_order_not_resting":
            fix = (
                "Audit sampled rows; if they are legitimate marketable lifecycles, keep as expected caveat; "
                "otherwise improve marketability/deferred-residual classification."
            )
        elif flag == "modify_for_unseen_order":
            fix = (
                "Inspect prior events for the order ID and decide whether missing reload context, "
                "partition reset, or hidden-state handling is needed."
            )
        else:
            fix = "Trace the flagged event class and add a targeted state-machine test before changing reconstruction logic."
        problems.append(
            Problem(
                severity=severity,
                problem=f"LOB issue flag: {flag}",
                evidence=f"lob_issue_flags contains {flag!r} on {count} rows",
                count=count,
                example_sort_index=int(row["example_sort_index"]),
                possible_fix=fix,
            )
        )

    norm_counts = _explode_flag_counts(normalized, "normalization_issue_flags")
    for row in norm_counts.iter_rows(named=True):
        flag = row["flag"]
        count = int(row["len"])
        if flag == "missing_client_original_id":
            fix = (
                "Carry missingness into client-level features or restrict client-level analysis to rows with "
                "reliable client IDs; firm-level analysis remains usable."
            )
        elif flag == "missing_price_for_potential_resting_event":
            fix = "Audit event/order type mapping; if non-resting by design, update is_visible_resting_event() or enum classification."
        elif flag == "non_resting_unpriced_event":
            fix = "Expected for market/unpriced non-resting events; keep out of visible depth and document counts."
        else:
            fix = "Inspect normalize_event() and add a regression test for this normalization issue."
        problems.append(
            Problem(
                severity="warning",
                problem=f"Normalization issue flag: {flag}",
                evidence=f"normalization_issue_flags contains {flag!r} on {count} rows",
                count=count,
                example_sort_index=int(row["example_sort_index"]),
                possible_fix=fix,
            )
        )
    return problems


def check_marketable_lifecycle_flags(panel: pl.DataFrame) -> list[Problem]:
    required = {"lob_issue_flags", "book_crossed_post_flag", "book_locked_post_flag", "sort_index"}
    if not required.issubset(set(panel.columns)):
        return []
    flagged = panel.filter(
        pl.col("lob_issue_flags").is_not_null()
        & pl.col("lob_issue_flags").cast(pl.Utf8).str.contains("marketable_order_not_resting")
    )
    if flagged.is_empty():
        return []
    bad = flagged.filter(
        pl.col("book_crossed_post_flag").fill_null(False) | pl.col("book_locked_post_flag").fill_null(False)
    )
    if bad.is_empty():
        return []
    return [
        Problem(
            severity="error",
            problem="Marketable non-resting row still creates locked/crossed post-book state",
            evidence=f"{bad.height} marketable_order_not_resting rows have locked/crossed post flags",
            count=bad.height,
            example_sort_index=_first_sort_index(bad),
            possible_fix=(
                "Inspect deferred residual handling in _apply_event() and _flush_pending_aggressive_residuals(); "
                "add a synthetic lifecycle test for the example sort_index pattern."
            ),
        )
    ]


def check_stop_limit_visibility(panel: pl.DataFrame) -> list[Problem]:
    required = {
        "event_order_type_label",
        "event_side",
        "event_price",
        "pre_best_bid",
        "pre_best_ask",
        "post_best_bid",
        "post_best_ask",
        "sort_index",
    }
    if not required.issubset(set(panel.columns)):
        return []
    stop_limit = panel.filter(pl.col("event_order_type_label") == "stop_limit_or_stop_limit_on_quote")
    if stop_limit.is_empty():
        return []
    ask_bad = stop_limit.filter(
        (pl.col("event_side") == "ask")
        & pl.col("event_price").is_not_null()
        & (pl.col("post_best_ask") == pl.col("event_price"))
        & ((pl.col("pre_best_ask").is_null()) | (pl.col("pre_best_ask") != pl.col("event_price")))
    )
    bid_bad = stop_limit.filter(
        (pl.col("event_side") == "bid")
        & pl.col("event_price").is_not_null()
        & (pl.col("post_best_bid") == pl.col("event_price"))
        & ((pl.col("pre_best_bid").is_null()) | (pl.col("pre_best_bid") != pl.col("event_price")))
    )
    bad_frames = [frame for frame in (ask_bad, bid_bad) if not frame.is_empty()]
    if not bad_frames:
        return []
    bad = pl.concat(bad_frames, how="vertical")
    return [
        Problem(
            severity="error",
            problem="Stop-limit row appears to seed visible best price before trigger",
            evidence=f"{bad.height} stop-limit rows changed the visible best price to their own price",
            count=bad.height,
            example_sort_index=_first_sort_index(bad),
            possible_fix=(
                "Keep stop-limit orders out of is_visible_resting_event() until a documented stop-limit trigger "
                "or dark-to-lit transformation makes them visible."
            ),
        )
    ]


def _flagged_rows(df: pl.DataFrame, column: str, flag: str) -> pl.DataFrame:
    if column not in df.columns or df.is_empty():
        return df.head(0)
    return df.filter(pl.col(column).is_not_null() & pl.col(column).cast(pl.Utf8).str.contains(flag))


def _format_group_counts(df: pl.DataFrame, column: str) -> str:
    if df.is_empty() or column not in df.columns:
        return "none"
    counts = df.group_by(column).len().sort("len", descending=True)
    parts = []
    for row in counts.iter_rows(named=True):
        key = row[column]
        label = "null" if key is None else str(key)
        parts.append(f"{label}={int(row['len'])}")
    return ", ".join(parts) if parts else "none"


def _example_rows(df: pl.DataFrame, columns: list[str], *, limit: int) -> list[dict[str, Any]]:
    if df.is_empty():
        return []
    available = [column for column in columns if column in df.columns]
    if not available:
        return []
    examples: list[dict[str, Any]] = []
    for row in df.select(available).head(limit).to_dicts():
        examples.append({key: value for key, value in row.items() if value is not None})
    return examples


def build_warning_details(
    panel: pl.DataFrame,
    normalized: pl.DataFrame,
    *,
    examples_per_warning: int = 3,
) -> list[WarningDetail]:
    panel_example_columns = [
        "sort_index",
        "TRADEDATE",
        "event_class",
        "event_order_type_label",
        "ORDERID",
        "event_side",
        "event_price",
        "event_leaves_qty",
        "event_displayed_qty",
        "pre_best_bid",
        "pre_best_ask",
        "post_best_bid",
        "post_best_ask",
        "lob_issue_flags",
    ]
    normalized_example_columns = [
        "sort_index",
        "TRADEDATE",
        "event_class",
        "event_order_type_label",
        "ORDERID",
        "side_label",
        "ORDERPX",
        "LEAVESQTY",
        "DISPLAYEDQTY",
        "normalization_issue_flags",
    ]

    details: list[WarningDetail] = []

    marketable = _flagged_rows(panel, "lob_issue_flags", "marketable_order_not_resting")
    if not marketable.is_empty():
        details.append(
            WarningDetail(
                flag="marketable_order_not_resting",
                source="lob_issue_flags",
                status="handled_visible_book_caveat",
                explanation=(
                    "These rows are priced limit/iceberg rows that would lock or cross the current visible book. "
                    "The reconstruction intentionally leaves these rows not inserted into visible resting depth; this prevents "
                    "the row-order artifact that previously created artificial locked/crossed books."
                ),
                recommended_action=(
                    "Not a blocker for exploratory visible-book features if examples look like marketable lifecycles. "
                    "Before final claims, sample-check executions around these rows and document that marketable "
                    "incoming orders are excluded from passive resting liquidity."
                ),
                observed_evidence={
                    "rows": marketable.height,
                    "by_event_class": _format_group_counts(marketable, "event_class"),
                    "by_order_type": _format_group_counts(marketable, "event_order_type_label"),
                    "by_side": _format_group_counts(marketable, "event_side"),
                    "post_locked_or_crossed_rows": marketable.filter(
                        pl.col("book_crossed_post_flag").fill_null(False)
                        | pl.col("book_locked_post_flag").fill_null(False)
                    ).height,
                },
                examples=_example_rows(marketable, panel_example_columns, limit=examples_per_warning),
            )
        )

    modify_unseen = _flagged_rows(panel, "lob_issue_flags", "modify_for_unseen_order")
    if not modify_unseen.is_empty():
        details.append(
            WarningDetail(
                flag="modify_for_unseen_order",
                source="lob_issue_flags",
                status="open_low_count_audit",
                explanation=(
                    "A modify row referenced an order that was not active in the reconstructed visible state. "
                    "The current implementation can still apply the updated visible state, but this indicates missing "
                    "prior visible context, hidden/non-visible prior state, or a partition/session boundary effect."
                ),
                recommended_action=(
                    "Inspect these low-count examples before final scientific use. If they share a pattern, add a "
                    "targeted lifecycle test and either seed the missing state from reload context or classify the prior "
                    "state as intentionally non-visible."
                ),
                observed_evidence={
                    "rows": modify_unseen.height,
                    "by_order_type": _format_group_counts(modify_unseen, "event_order_type_label"),
                    "by_side": _format_group_counts(modify_unseen, "event_side"),
                    "post_locked_or_crossed_rows": modify_unseen.filter(
                        pl.col("book_crossed_post_flag").fill_null(False)
                        | pl.col("book_locked_post_flag").fill_null(False)
                    ).height,
                },
                examples=_example_rows(modify_unseen, panel_example_columns, limit=examples_per_warning),
            )
        )

    missing_client = _flagged_rows(normalized, "normalization_issue_flags", "missing_client_original_id")
    if not missing_client.is_empty():
        details.append(
            WarningDetail(
                flag="missing_client_original_id",
                source="normalization_issue_flags",
                status="data_limitation_client_level",
                explanation=(
                    "The event lacks `NMSC_ORIGINALCLIENTIDSHORTCODE`. Firm-level state remains usable, but client-level "
                    "active-liquidity or spoofing features would mix known and unknown clients unless missingness is "
                    "handled explicitly."
                ),
                recommended_action=(
                    "For firm-level analysis, carry the missing-client flag as provenance. For client-level analysis, "
                    "either restrict to rows with reliable client IDs or create an explicit unknown-client bucket and "
                    "report coverage."
                ),
                observed_evidence={
                    "rows": missing_client.height,
                    "by_event_class": _format_group_counts(missing_client, "event_class"),
                    "by_order_type": _format_group_counts(missing_client, "event_order_type_label"),
                    "by_side": _format_group_counts(missing_client, "side_label"),
                },
                examples=_example_rows(missing_client, normalized_example_columns, limit=examples_per_warning),
            )
        )

    non_resting_unpriced = _flagged_rows(normalized, "normalization_issue_flags", "non_resting_unpriced_event")
    if not non_resting_unpriced.is_empty():
        details.append(
            WarningDetail(
                flag="non_resting_unpriced_event",
                source="normalization_issue_flags",
                status="expected_non_visible_flow",
                explanation=(
                    "These are unpriced event rows whose order type is not visible resting liquidity, such as market "
                    "or stop-market orders. They are retained in the event panel but excluded from visible depth."
                ),
                recommended_action=(
                    "Not a blocker for visible-book reconstruction. Keep them available for event-flow features, but "
                    "do not count them as passive displayed depth."
                ),
                observed_evidence={
                    "rows": non_resting_unpriced.height,
                    "by_event_class": _format_group_counts(non_resting_unpriced, "event_class"),
                    "by_order_type": _format_group_counts(non_resting_unpriced, "event_order_type_label"),
                    "by_side": _format_group_counts(non_resting_unpriced, "side_label"),
                },
                examples=_example_rows(non_resting_unpriced, normalized_example_columns, limit=examples_per_warning),
            )
        )

    missing_price = _flagged_rows(normalized, "normalization_issue_flags", "missing_price_for_potential_resting_event")
    if not missing_price.is_empty():
        details.append(
            WarningDetail(
                flag="missing_price_for_potential_resting_event",
                source="normalization_issue_flags",
                status="open_mapping_or_data_issue",
                explanation=(
                    "A row class that may normally update visible resting liquidity had no order price. This can be a "
                    "data problem or an order-type mapping that should be classified as non-visible instead."
                ),
                recommended_action=(
                    "Audit examples by event class and order type. If they are conditional/non-visible orders, update "
                    "normalization classification; otherwise add a targeted test and decide whether to drop or impute."
                ),
                observed_evidence={
                    "rows": missing_price.height,
                    "by_event_class": _format_group_counts(missing_price, "event_class"),
                    "by_order_type": _format_group_counts(missing_price, "event_order_type_label"),
                    "by_side": _format_group_counts(missing_price, "side_label"),
                },
                examples=_example_rows(missing_price, normalized_example_columns, limit=examples_per_warning),
            )
        )

    return details


def _format_evidence_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, default=str, sort_keys=True)
    return str(value)


def _render_example_table(examples: list[dict[str, Any]]) -> list[str]:
    if not examples:
        return ["", "No examples captured."]
    columns: list[str] = []
    for example in examples:
        for key in example:
            if key not in columns:
                columns.append(key)
    lines = ["", "Example rows:", "", "| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for example in examples:
        lines.append(
            "| "
            + " | ".join(_escape_markdown_cell(example.get(column, "")) for column in columns)
            + " |"
        )
    return lines


def dataframe_fingerprint(df: pl.DataFrame) -> str:
    rows = df.to_dicts()
    payload = json.dumps(rows, default=str, sort_keys=True, separators=(",", ":"), allow_nan=True).encode(
        "utf-8"
    )
    return hashlib.sha256(payload).hexdigest()


def check_determinism(input_path: Path, *, top_n: int) -> list[Problem]:
    df = pl.read_parquet(input_path)
    config = LOBConfig(top_n=top_n, snapshot_mode="none")
    first = reconstruct_dataframe(df, config=config)
    second = reconstruct_dataframe(df, config=config)
    first_panel = dataframe_fingerprint(first.panel)
    second_panel = dataframe_fingerprint(second.panel)
    first_norm = dataframe_fingerprint(first.normalized_events)
    second_norm = dataframe_fingerprint(second.normalized_events)
    if first_panel == second_panel and first_norm == second_norm:
        return []
    return [
        Problem(
            severity="error",
            problem="Non-deterministic reconstruction output",
            evidence=(
                f"panel fingerprints {first_panel} vs {second_panel}; "
                f"normalized fingerprints {first_norm} vs {second_norm}"
            ),
            count=1,
            example_sort_index=None,
            possible_fix=(
                "Stabilize sort_events() tie-breakers and avoid iteration over unordered containers in "
                "book_summary()/agent aggregation."
            ),
        )
    ]


def render_markdown_report(
    *,
    source_file: str,
    output_dir: str,
    summary: dict[str, Any],
    problems: list[Problem],
    warning_details: list[WarningDetail] | None = None,
) -> str:
    created_at = datetime.now(timezone.utc).isoformat()
    hard_error_categories = sum(1 for problem in problems if problem.severity == "error")
    warnings = sum(1 for problem in problems if problem.severity == "warning")
    lines = [
        "# LOB Reconstruction Verification Report",
        "",
        f"- created_at_utc: `{created_at}`",
        f"- source_file: `{source_file}`",
        f"- output_dir: `{output_dir}`",
        "- scope: one-file verification on the smallest available parquet only",
        f"- hard_error_categories: `{hard_error_categories}`",
        f"- warning_categories: `{warnings}`",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend(
        [
            "",
            "## Problem inventory",
            "",
            "| Severity | Problem | Evidence | Count | Example sort_index | Possible fix | Status |",
            "|---|---|---:|---:|---:|---|---|",
        ]
    )
    if problems:
        for problem in problems:
            example = "" if problem.example_sort_index is None else str(problem.example_sort_index)
            lines.append(
                "| "
                + " | ".join(
                    [
                        _escape_markdown_cell(problem.severity),
                        _escape_markdown_cell(problem.problem),
                        _escape_markdown_cell(problem.evidence),
                        str(problem.count),
                        example,
                        _escape_markdown_cell(problem.possible_fix),
                        _escape_markdown_cell(problem.status),
                    ]
                )
                + " |"
            )
    else:
        lines.append(
            "| info | No hard verification problems found | All implemented checks passed | 0 |  | "
            "Keep current tests and rerun on additional files before feature extraction. | closed |"
        )
    if warning_details:
        lines.extend(["", "## Warning details and examples", ""])
        for detail in warning_details:
            lines.extend(
                [
                    f"### `{detail.flag}`",
                    "",
                    f"- source: `{detail.source}`",
                    f"- status: `{detail.status}`",
                    f"- explanation: {detail.explanation}",
                    f"- recommended_action: {detail.recommended_action}",
                    "- observed_evidence:",
                ]
            )
            for key, value in detail.observed_evidence.items():
                lines.append(f"  - {key}: `{_format_evidence_value(value)}`")
            lines.extend(_render_example_table(detail.examples))
    lines.extend(
        [
            "",
            "## Scientific caveats",
            "",
            "- This report verifies visible-book reconstruction behavior, not exact FIFO queue position.",
            "- This report uses one parquet file only; it does not prove correctness on all instruments/days.",
            "- Synthetic unit tests remain necessary for specific event-sequence semantics.",
            "- Any warning marked as a data limitation should be propagated into downstream spoofing-feature interpretation.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify current LOB reconstruction on one parquet file.")
    parser.add_argument("--input", type=Path, required=True, help="Input parquet file to verify")
    parser.add_argument(
        "--output-dir", type=Path, required=True, help="Directory for reconstruction outputs and verification report"
    )
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument(
        "--snapshot-mode",
        default="end_of_partition",
        choices=["none", "every_event_for_sample", "issue_rows_only", "end_of_partition"],
    )
    parser.add_argument("--skip-determinism", action="store_true", help="Skip duplicate in-memory reconstruction check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = LOBConfig(top_n=args.top_n, snapshot_mode=args.snapshot_mode)
    paths = reconstruct_file(args.input, args.output_dir, config=config)

    input_rows = int(pl.scan_parquet(args.input).select(pl.len()).collect().item())
    panel = pl.read_parquet(paths.panel_path)
    normalized = pl.read_parquet(paths.normalized_path)
    agent = pl.read_parquet(paths.agent_panel_path)

    summary: dict[str, Any] = {
        "input_rows": input_rows,
        "panel_rows": panel.height,
        "normalized_rows": normalized.height,
        "agent_event_state_rows": agent.height,
        "book_crossed_pre_rows": _bool_sum(panel, "book_crossed_pre_flag"),
        "book_crossed_post_rows": _bool_sum(panel, "book_crossed_post_flag"),
        "book_locked_pre_rows": _bool_sum(panel, "book_locked_pre_flag"),
        "book_locked_post_rows": _bool_sum(panel, "book_locked_post_flag"),
        "top_n": args.top_n,
        "snapshot_mode": args.snapshot_mode,
    }

    problems: list[Problem] = []
    problems.extend(
        check_row_counts(
            input_rows=input_rows,
            panel_rows=panel.height,
            normalized_rows=normalized.height,
            agent_rows=agent.height,
        )
    )
    problems.extend(check_spread_flags(panel))
    problems.extend(check_marketable_lifecycle_flags(panel))
    problems.extend(check_stop_limit_visibility(panel))
    problems.extend(check_top_n_depth(panel, top_n=args.top_n))
    problems.extend(check_issue_flags(panel, normalized))
    if not args.skip_determinism:
        problems.extend(check_determinism(args.input, top_n=args.top_n))
    warning_details = build_warning_details(panel, normalized)

    report = render_markdown_report(
        source_file=str(args.input),
        output_dir=str(args.output_dir),
        summary=summary,
        problems=problems,
        warning_details=warning_details,
    )
    report_path = args.output_dir / "reconstruction_verification_report.md"
    report_path.write_text(report)
    print(f"verification_report: {report_path}")


if __name__ == "__main__":
    main()
