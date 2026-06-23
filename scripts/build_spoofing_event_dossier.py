#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl


class EventBundle:
    def __init__(self, event: dict[str, Any], event_log: pl.DataFrame, queue: pl.DataFrame) -> None:
        self.event = event
        self.event_log = event_log
        self.queue = queue


def select_event_bundle(
    event_id: str,
    review_events: pl.DataFrame,
    event_log: pl.DataFrame,
    queue: pl.DataFrame,
) -> EventBundle:
    event_rows = review_events.filter(pl.col("review_event_id") == event_id)
    if event_rows.height != 1:
        raise ValueError(f"expected exactly one event for {event_id}, found {event_rows.height}")
    sort_cols = [col for col in ("snapshot_sort_index", "side", "level", "queue_position") if col in queue.columns]
    event_log_sort = "sort_index" if "sort_index" in event_log.columns else None
    filtered_log = event_log.filter(pl.col("review_event_id") == event_id)
    filtered_queue = queue.filter(pl.col("review_event_id") == event_id)
    return EventBundle(
        event=event_rows.row(0, named=True),
        event_log=filtered_log.sort(event_log_sort) if event_log_sort else filtered_log,
        queue=filtered_queue.sort(sort_cols) if sort_cols else filtered_queue,
    )


def build_stage_depth_summary(queue: pl.DataFrame) -> pl.DataFrame:
    if queue.is_empty():
        return pl.DataFrame()
    candidate_expr = (
        pl.when(pl.col("is_candidate_deceptive_order") | pl.col("is_matched_deceptive_cancel_order"))
        .then(pl.col("visible_qty"))
        .otherwise(0)
        .sum()
        .alias("candidate_visible_qty")
    )
    return (
        queue.group_by(["snapshot_phase", "side", "level", "price"])
        .agg(
            pl.max("level_visible_qty").alias("total_visible_qty"),
            candidate_expr,
            pl.first("client_queue_dict").alias("client_queue_dict"),
        )
        .with_columns(
            (pl.col("candidate_visible_qty") / pl.col("total_visible_qty"))
            .fill_nan(0)
            .fill_null(0)
            .alias("candidate_level_share"),
            pl.col("snapshot_phase").alias("phase"),
        )
        .select(
            [
                "phase",
                "side",
                "level",
                "price",
                "total_visible_qty",
                "candidate_visible_qty",
                "candidate_level_share",
                "client_queue_dict",
            ]
        )
        .sort(["phase", "side", "level"])
    )


def build_parameter_robustness(event_sort_index: int, parameter_grid_root: Path | None) -> pl.DataFrame:
    if parameter_grid_root is None:
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(parameter_grid_root.glob("kappa_*_lambda_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        execution_path = metadata_path.parent / "execution_metrics.parquet"
        if not execution_path.exists():
            continue
        metrics = pl.read_parquet(execution_path)
        if "sort_index" not in metrics.columns:
            continue
        hit = metrics.filter(pl.col("sort_index") == event_sort_index)
        if hit.is_empty():
            rows.append(
                {
                    "kappa": metadata.get("kappa"),
                    "lambda": metadata.get("lambda_"),
                    "matched": False,
                    "MSCI": None,
                    "rank_by_MSCI": None,
                }
            )
            continue
        ranked = metrics.sort("MSCI", descending=True).with_row_index("rank_by_MSCI", offset=1)
        ranked_hit = ranked.filter(pl.col("sort_index") == event_sort_index)
        row = hit.row(0, named=True)
        rows.append(
            {
                "kappa": metadata.get("kappa"),
                "lambda": metadata.get("lambda_"),
                "matched": bool(row.get("has_matched_deceptive_cancel_window")),
                "MSCI": row.get("MSCI"),
                "SCI": row.get("SCI"),
                "collapse_opposite_side": row.get("collapse_opposite_side"),
                "collapse_same_side": row.get("collapse_same_side"),
                "rank_by_MSCI": ranked_hit.row(0, named=True).get("rank_by_MSCI") if not ranked_hit.is_empty() else None,
            }
        )
    return pl.DataFrame(rows).sort(["kappa", "lambda"]) if rows else pl.DataFrame()


def _markdown_table(df: pl.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if df.is_empty() or not columns:
        return "No rows.\n"
    selected = [col for col in columns if col in df.columns]
    if not selected:
        return "No requested columns available.\n"
    shown = df.select(selected)
    if limit is not None:
        shown = shown.head(limit)
    cols = shown.columns
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in shown.to_dicts():
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines) + "\n"


def render_dossier_markdown(
    *,
    event: dict[str, Any],
    event_log: pl.DataFrame,
    stage_depth: pl.DataFrame,
    robustness: pl.DataFrame,
) -> str:
    event_id = event.get("review_event_id")
    lines = [
        f"# Event dossier: {event_id}",
        "",
        "## Event identity",
        "",
        f"- event_id: {event_id}",
        f"- client_id: {event.get('client_id')}",
        f"- event_ts: {event.get('event_ts')}",
        f"- execution_side: {event.get('execution_side')}",
        f"- deceptive_side: {event.get('deceptive_side')}",
        f"- fill_qty: {event.get('fill_qty')}",
        "",
        "## Model scores",
        "",
    ]
    score_keys = [
        "DWI_pre_window",
        "DWI_post_window",
        "SCI",
        "collapse_opposite_side",
        "collapse_same_side",
        "MSCI",
        "candidate_deceptive_visible_qty_pre",
        "matched_deceptive_cancel_visible_qty_window",
        "matched_deceptive_cancel_fraction_window",
        "candidate_deceptive_order_ids_pre",
        "matched_deceptive_cancel_order_ids_window",
    ]
    for key in score_keys:
        if key in event:
            lines.append(f"- {key}: {event.get(key)}")
    lines += ["", "## Stage depth summary", ""]
    lines.append(
        _markdown_table(
            stage_depth,
            [
                "phase",
                "side",
                "level",
                "price",
                "total_visible_qty",
                "candidate_visible_qty",
                "candidate_level_share",
                "client_queue_dict",
            ],
        )
    )
    lines += ["", "## Actual event log", ""]
    lines.append(
        _markdown_table(
            event_log,
            [
                "sort_index",
                "event_ts",
                "event_class",
                "side",
                "price",
                "ORDERID",
                "client_id",
                "leaves_qty",
                "displayed_qty",
                "last_shares",
                "is_execution_order",
                "is_candidate_deceptive_order",
                "is_matched_deceptive_cancel_order",
            ],
            limit=60,
        )
    )
    lines += ["", "## Kappa/lambda robustness", ""]
    lines.append(_markdown_table(robustness, robustness.columns if not robustness.is_empty() else []))
    return "\n".join(lines).strip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an LLM-ready dossier for one spoofing-like event.")
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parameter-grid-root", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    review_dir = args.review_dir
    output_dir = args.output_dir or review_dir / "llm_reviews" / args.event_id
    output_dir.mkdir(parents=True, exist_ok=True)

    events = pl.read_parquet(review_dir / "matched_spoofing_events.parquet")
    event_log = pl.read_parquet(review_dir / "matched_spoofing_event_log.parquet")
    queue = pl.read_parquet(review_dir / "matched_spoofing_lob_queue.parquet")

    bundle = select_event_bundle(args.event_id, events, event_log, queue)
    stage_depth = build_stage_depth_summary(bundle.queue)
    robustness = build_parameter_robustness(int(bundle.event["sort_index"]), args.parameter_grid_root)
    markdown = render_dossier_markdown(
        event=bundle.event,
        event_log=bundle.event_log,
        stage_depth=stage_depth,
        robustness=robustness,
    )

    (output_dir / "dossier.md").write_text(markdown)
    (output_dir / "dossier.json").write_text(
        json.dumps(
            {
                "event": bundle.event,
                "event_log": bundle.event_log.to_dicts(),
                "stage_depth": stage_depth.to_dicts(),
                "robustness": robustness.to_dicts(),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    print(output_dir / "dossier.md")


if __name__ == "__main__":
    main()
