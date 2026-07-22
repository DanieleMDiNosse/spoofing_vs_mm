#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import polars as pl

DOSSIER_SCHEMA_VERSION = "2.0-cluster-first"


class EventBundle:
    def __init__(
        self,
        event: dict[str, Any],
        event_log: pl.DataFrame,
        queue: pl.DataFrame,
        child_members: pl.DataFrame | None = None,
        cancel_candidates: pl.DataFrame | None = None,
    ) -> None:
        self.event = event
        self.event_log = event_log
        self.queue = queue
        self.child_members = child_members if child_members is not None else pl.DataFrame()
        self.cancel_candidates = cancel_candidates if cancel_candidates is not None else pl.DataFrame()


def _empty_like(df: pl.DataFrame) -> pl.DataFrame:
    return df.head(0)


def _id_column(df: pl.DataFrame) -> str | None:
    if "execution_cluster_id" in df.columns:
        return "execution_cluster_id"
    if "review_event_id" in df.columns:
        return "review_event_id"
    return None


def _filter_id(df: pl.DataFrame, event_id: str) -> pl.DataFrame:
    column = _id_column(df)
    if column is None:
        return _empty_like(df)
    return df.filter(pl.col(column).cast(pl.String) == event_id)


def select_event_bundle(
    event_id: str,
    review_events: pl.DataFrame,
    event_log: pl.DataFrame,
    queue: pl.DataFrame,
    *,
    cluster_members: pl.DataFrame | None = None,
    cancel_candidates: pl.DataFrame | None = None,
) -> EventBundle:
    """Select exactly one cluster-first review bundle.

    Legacy S... review IDs remain readable only when the event artifact has no
    cluster identifier.  A cluster-capable artifact never falls back to a
    message-level identifier, which prevents stale reviews from being reused.
    """
    primary = "execution_cluster_id" if "execution_cluster_id" in review_events.columns else "review_event_id"
    event_rows = review_events.filter(pl.col(primary).cast(pl.String) == event_id)
    if event_rows.height != 1:
        raise ValueError(f"expected exactly one event for {event_id}, found {event_rows.height}")
    event = event_rows.row(0, named=True)
    canonical_id = str(event.get("execution_cluster_id") or event.get("review_event_id"))
    sort_cols = [col for col in ("snapshot_sort_index", "side", "level", "queue_position") if col in queue.columns]
    event_log_sort = "sort_index" if "sort_index" in event_log.columns else None
    filtered_log = _filter_id(event_log, canonical_id)
    filtered_queue = _filter_id(queue, canonical_id)
    members = _filter_id(cluster_members, canonical_id) if cluster_members is not None else pl.DataFrame()
    candidates = _filter_id(cancel_candidates, canonical_id) if cancel_candidates is not None else pl.DataFrame()
    return EventBundle(
        event=event,
        event_log=filtered_log.sort(event_log_sort) if event_log_sort else filtered_log,
        queue=filtered_queue.sort(sort_cols) if sort_cols else filtered_queue,
        child_members=members.sort("child_sort_index") if "child_sort_index" in members.columns else members,
        cancel_candidates=candidates.sort("sort_index") if "sort_index" in candidates.columns else candidates,
    )


def build_stage_depth_summary(queue: pl.DataFrame) -> pl.DataFrame:
    required = {"snapshot_phase", "side", "level", "price", "visible_qty", "level_visible_qty"}
    if queue.is_empty() or not required.issubset(queue.columns):
        return pl.DataFrame()
    candidate = pl.col("is_candidate_deceptive_order") if "is_candidate_deceptive_order" in queue.columns else pl.lit(False)
    matched = pl.col("is_matched_deceptive_cancel_order") if "is_matched_deceptive_cancel_order" in queue.columns else pl.lit(False)
    client_queue = pl.first("client_queue_dict").alias("client_queue_dict") if "client_queue_dict" in queue.columns else pl.lit(None).alias("client_queue_dict")
    return (
        queue.group_by(["snapshot_phase", "side", "level", "price"])
        .agg(pl.max("level_visible_qty").alias("total_visible_qty"), pl.when(candidate | matched).then(pl.col("visible_qty")).otherwise(0).sum().alias("candidate_visible_qty"), client_queue)
        .with_columns((pl.col("candidate_visible_qty") / pl.col("total_visible_qty")).fill_nan(0).fill_null(0).alias("candidate_level_share"), pl.col("snapshot_phase").alias("phase"))
        .select(["phase", "side", "level", "price", "total_visible_qty", "candidate_visible_qty", "candidate_level_share", "client_queue_dict"])
        .sort(["phase", "side", "level"])
    )


def build_focal_timeline(
    event: dict[str, Any],
    event_log: pl.DataFrame,
    *,
    child_members: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Select only rows defining the selected cluster's withdrawal sequence."""
    if event_log.is_empty() or "sort_index" not in event_log.columns:
        return pl.DataFrame()
    focal_sort_index = int(event.get("cluster_first_sort_index", event.get("sort_index", -1)))
    candidate = pl.col("is_candidate_deceptive_order") if "is_candidate_deceptive_order" in event_log.columns else pl.lit(False)
    matched = pl.col("is_matched_deceptive_cancel_order") if "is_matched_deceptive_cancel_order" in event_log.columns else pl.lit(False)
    event_class = pl.col("event_class") if "event_class" in event_log.columns else pl.lit("")
    child_indexes = {int(v) for v in str(event.get("child_fill_sort_indices") or "").split(";") if v.isdigit()}
    cluster_id = event.get("execution_cluster_id")
    if cluster_id is not None and child_members is not None and not child_members.is_empty():
        child_indexes = set(
            child_members.filter(
                pl.col("execution_cluster_id").cast(pl.String) == str(cluster_id)
            )["child_sort_index"].cast(pl.Int64).to_list()
        )
    if cluster_id is not None and not child_indexes:
        raise ValueError(f"execution cluster {cluster_id} has no child-member provenance")
    if cluster_id is None and not child_indexes:
        child_indexes = {focal_sort_index}
    selected_execution = pl.col("sort_index").is_in(sorted(child_indexes))
    role = (
        pl.when(selected_execution).then(pl.lit("selected_passive_execution"))
        .when((pl.col("sort_index") < focal_sort_index) & candidate & event_class.is_in(["new_order", "reload_order", "modify_order"])).then(pl.lit("candidate_order_before_execution"))
        .when((pl.col("sort_index") > max(child_indexes, default=focal_sort_index)) & matched & (event_class == "cancel")).then(pl.lit("matched_cancel_after_execution"))
        .otherwise(None).alias("timeline_role")
    )
    return event_log.with_columns(role).filter(pl.col("timeline_role").is_not_null()).sort("sort_index")


def build_parameter_robustness(event_sort_index: int, parameter_grid_root: Path | None, execution_cluster_id: str | None = None) -> pl.DataFrame:
    if parameter_grid_root is None:
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(parameter_grid_root.glob("kappa_*_lambda_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        execution_path = metadata_path.parent / "execution_metrics.parquet"
        if not execution_path.exists():
            continue
        metrics = pl.read_parquet(execution_path)
        if execution_cluster_id and "execution_cluster_id" in metrics.columns:
            hit = metrics.filter(pl.col("execution_cluster_id").cast(pl.String) == execution_cluster_id)
        elif "sort_index" in metrics.columns:
            hit = metrics.filter(pl.col("sort_index") == event_sort_index)
        else:
            continue
        if hit.is_empty():
            rows.append({"kappa": metadata.get("kappa"), "lambda": metadata.get("lambda_"), "matched": False, "MSCI": None, "rank_by_MSCI": None})
            continue
        ranked = metrics.sort("MSCI", descending=True).with_row_index("rank_by_MSCI", offset=1) if "MSCI" in metrics.columns else metrics
        if execution_cluster_id and "execution_cluster_id" in ranked.columns:
            ranked_hit = ranked.filter(pl.col("execution_cluster_id").cast(pl.String) == execution_cluster_id)
        else:
            ranked_hit = ranked.filter(pl.col("sort_index") == event_sort_index)
        row = hit.row(0, named=True)
        rows.append({"kappa": metadata.get("kappa"), "lambda": metadata.get("lambda_"), "matched": bool(row.get("has_matched_deceptive_cancel_window")), "MSCI": row.get("MSCI"), "SCI": row.get("SCI"), "collapse_opposite_side": row.get("collapse_opposite_side"), "collapse_same_side": row.get("collapse_same_side"), "rank_by_MSCI": ranked_hit.row(0, named=True).get("rank_by_MSCI") if not ranked_hit.is_empty() and "rank_by_MSCI" in ranked_hit.columns else None})
    return pl.DataFrame(rows).sort(["kappa", "lambda"]) if rows else pl.DataFrame()


def _markdown_table(df: pl.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if df.is_empty() or not columns:
        return "No rows.\n"
    selected = [col for col in columns if col in df.columns]
    if not selected:
        return "No requested columns available.\n"
    shown = df.select(selected).head(limit) if limit is not None else df.select(selected)
    lines = ["| " + " | ".join(shown.columns) + " |", "| " + " | ".join(["---"] * len(shown.columns)) + " |"]
    lines.extend("| " + " | ".join(str(row.get(col, "")) for col in shown.columns) + " |" for row in shown.to_dicts())
    return "\n".join(lines) + "\n"


def _number(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def enrich_cancel_candidates(cancel_candidates: pl.DataFrame) -> pl.DataFrame:
    """Derive execution fractions only where both observable source quantities exist."""
    if cancel_candidates.is_empty():
        return cancel_candidates
    rows = []
    for row in cancel_candidates.to_dicts():
        original = next((_number(row.get(name)) for name in ("original_qty", "order_qty", "ORDERQTY") if _number(row.get(name)) is not None), None)
        leaves = next((_number(row.get(name)) for name in ("leaves_qty", "LEAVESQTY") if _number(row.get(name)) is not None), None)
        if original is not None and original > 0 and leaves is not None:
            executed = max(0.0, original - leaves)
            row["partially_executed_qty"] = executed
            row["partially_executed_fraction"] = executed / original
            row["execution_fraction_reconciles"] = abs((executed + leaves) - original) <= max(1e-9, original * 1e-9)
        rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)


def _data_quality(event: dict[str, Any], members: pl.DataFrame, candidates: pl.DataFrame) -> tuple[list[str], list[str]]:
    missing = [key for key in ("execution_cluster_id", "cluster_first_sort_index", "cluster_last_sort_index", "cluster_start_ts", "cluster_end_ts", "child_fill_count", "fill_qty") if event.get(key) is None]
    caveats = []
    client = event.get("client_id", event.get("event_client_original_id"))
    if client is None or str(client).strip() in {"", "0", "None", "null"}:
        caveats.append("Client attribution caveat: client_id is missing or zero; bilateral/inventory attribution cannot be inferred from this dossier.")
    if members.is_empty():
        caveats.append("Raw child-fill provenance is absent.")
    if candidates.is_empty():
        caveats.append("Cancellation candidate provenance is absent.")
    return missing, caveats


def render_dossier_markdown(*, event: dict[str, Any], event_log: pl.DataFrame, stage_depth: pl.DataFrame, robustness: pl.DataFrame, focal_timeline: pl.DataFrame | None = None, child_members: pl.DataFrame | None = None, cancel_candidates: pl.DataFrame | None = None) -> str:
    cluster_id = str(event.get("execution_cluster_id") or event.get("review_event_id"))
    members = child_members if child_members is not None else pl.DataFrame()
    candidates = enrich_cancel_candidates(cancel_candidates if cancel_candidates is not None else pl.DataFrame())
    missing, caveats = _data_quality(event, members, candidates)
    lines = [f"# Event dossier: {cluster_id}", "", "## Schema and provenance", "", f"- dossier_schema_version: {DOSSIER_SCHEMA_VERSION}", f"- execution_cluster_id: {cluster_id}", f"- child_fill_count: {event.get('child_fill_count')}", f"- missing_fields: {', '.join(missing) if missing else 'none'}", "", "## Execution cluster", ""]
    for key in ("execution_cluster_id", "review_event_id", "cluster_first_sort_index", "cluster_last_sort_index", "cluster_start_ts", "cluster_end_ts", "child_fill_count", "fill_qty", "event_order_id", "event_client_original_id", "execution_side", "deceptive_side", "event_price", "client_id"):
        if key in event:
            lines.append(f"- {key}: {event.get(key)}")
    lines += ["", "## Raw child fills", "", _markdown_table(members, ["execution_cluster_id", "child_sort_index", "child_event_id", "child_execution_id", "child_trade_uid", "child_event_ts", "child_fill_qty", "child_fill_price"]), "", "## Cancellation assignment", "", "Assigned rows are canonical links; unassigned rows are competing candidate links and are retained for audit.", "", _markdown_table(candidates, ["execution_cluster_id", "candidate_order_id", "ORDERID", "assigned_flag", "assignment_rule", "competing_cluster_count", "sort_index", "event_ts", "original_qty", "order_qty", "leaves_qty", "partially_executed_qty", "partially_executed_fraction", "execution_fraction_reconciles"]), "", "## Model scores", ""]
    score_keys = ["has_matched_deceptive_cancel_window", "WMSCI_event", "withdrawal_to_fill_ratio", "weighted_withdrawal_to_fill_ratio", "weighted_net_withdrawal_qty_window", "matched_deceptive_cancel_min_delay_seconds", "matched_deceptive_cancel_max_delay_seconds", "favorable_mid_move_pre_fill", "favorable_microprice_move_pre_fill", "post_cancel_mid_reversion", "post_cancel_microprice_reversion", "execution_price_advantage_vs_posture_mid", "execution_price_advantage_vs_posture_microprice", "DWI_pre_window", "DWI_post_window", "SCI", "collapse_opposite_side", "collapse_same_side", "MSCI", "candidate_deceptive_visible_qty_pre", "matched_deceptive_cancel_visible_qty_window", "matched_deceptive_cancel_fraction_window", "candidate_deceptive_order_ids_pre", "matched_deceptive_cancel_order_ids_window"]
    lines.extend(f"- {key}: {event.get(key)}" for key in score_keys if key in event)
    lines += ["", "## Candidate execution-risk fields", "", _markdown_table(candidates, [name for name in candidates.columns if any(token in name.lower() for token in ("risk", "fill", "execut", "quantity", "qty", "leaves"))], limit=30), "", "## Client bilateral and inventory context", ""]
    context_keys = [key for key in event if any(token in key.lower() for token in ("bilateral", "inventory", "position", "client_"))]
    lines.extend(f"- {key}: {event.get(key)}" for key in context_keys) if context_keys else lines.append("No client bilateral or inventory fields supplied.")
    lines += ["", "## Data quality", ""] + [f"- {caveat}" for caveat in caveats] + ["", "## Focal matched-withdrawal timeline", "", "Use this concise table for the selected execution cluster; the broader actual event log is context only.", "", _markdown_table(focal_timeline if focal_timeline is not None else pl.DataFrame(), ["timeline_role", "sort_index", "event_ts", "event_class", "side", "price", "ORDERID", "client_id", "leaves_qty", "displayed_qty", "last_shares"]), "", "## Stage depth summary", "", _markdown_table(stage_depth, ["phase", "side", "level", "price", "total_visible_qty", "candidate_visible_qty", "candidate_level_share", "client_queue_dict"]), "", "## Actual event log", "", _markdown_table(event_log, ["sort_index", "event_ts", "event_class", "side", "price", "ORDERID", "client_id", "leaves_qty", "displayed_qty", "last_shares", "is_execution_order", "is_candidate_deceptive_order", "is_matched_deceptive_cancel_order"], limit=60), "", "## Kappa/lambda robustness", "", _markdown_table(robustness, robustness.columns if not robustness.is_empty() else [])]
    return "\n".join(lines).strip() + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an LLM-ready dossier for one cluster-first spoofing-like event.")
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parameter-grid-root", type=Path, default=None)
    parser.add_argument("--execution-cluster-members", type=Path, default=None)
    parser.add_argument("--execution-cancel-candidates", type=Path, default=None)
    return parser.parse_args(argv)


def _read_optional(path: Path | None) -> pl.DataFrame:
    return pl.read_parquet(path) if path is not None and path.exists() else pl.DataFrame()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    review_dir = args.review_dir
    output_dir = args.output_dir or review_dir / "llm_reviews" / args.event_id
    output_dir.mkdir(parents=True, exist_ok=True)
    events = pl.read_parquet(review_dir / "matched_spoofing_events.parquet")
    event_log = pl.read_parquet(review_dir / "matched_spoofing_event_log.parquet")
    queue = pl.read_parquet(review_dir / "matched_spoofing_lob_queue.parquet")
    members_path = args.execution_cluster_members or review_dir / "execution_cluster_members.parquet"
    candidates_path = args.execution_cancel_candidates or review_dir / "execution_cancel_candidates.parquet"
    bundle = select_event_bundle(args.event_id, events, event_log, queue, cluster_members=_read_optional(members_path), cancel_candidates=_read_optional(candidates_path))
    stage_depth = build_stage_depth_summary(bundle.queue)
    focal_timeline = build_focal_timeline(
        bundle.event,
        bundle.event_log,
        child_members=bundle.child_members,
    )
    event_sort = int(bundle.event.get("cluster_first_sort_index", bundle.event.get("sort_index", 0)))
    cluster_id = str(bundle.event.get("execution_cluster_id") or bundle.event.get("review_event_id"))
    robustness = build_parameter_robustness(event_sort, args.parameter_grid_root, execution_cluster_id=cluster_id)
    markdown = render_dossier_markdown(event=bundle.event, event_log=bundle.event_log, stage_depth=stage_depth, robustness=robustness, focal_timeline=focal_timeline, child_members=bundle.child_members, cancel_candidates=bundle.cancel_candidates)
    (output_dir / "dossier.md").write_text(markdown)
    (output_dir / "dossier.json").write_text(json.dumps({"schema_version": DOSSIER_SCHEMA_VERSION, "execution_cluster_id": cluster_id, "event": bundle.event, "execution_cluster": bundle.event, "child_members": bundle.child_members.to_dicts(), "cancellation_assignment": enrich_cancel_candidates(bundle.cancel_candidates).to_dicts(), "focal_timeline": focal_timeline.to_dicts(), "event_log": bundle.event_log.to_dicts(), "stage_depth": stage_depth.to_dicts(), "robustness": robustness.to_dicts()}, indent=2, sort_keys=True, default=str))
    print(output_dir / "dossier.md")


if __name__ == "__main__":
    main()
