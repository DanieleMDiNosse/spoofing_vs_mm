#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.models import ActiveOrder
from spoofing_detection.lob.normalize import normalize_event
from spoofing_detection.lob.panel import (
    _apply_event,
    _fill_group_key,
    _flush_pending_aggressive_residuals,
    _partition_id,
    sort_events,
)
from spoofing_detection.lob.spoofing_metrics import _parse_ts, choose_event_timestamp


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an interactive review dashboard and exact queue parquet for matched spoofing-like events."
    )
    parser.add_argument("--input", type=Path, required=True, help="Raw input parquet event file")
    parser.add_argument("--execution-metrics", type=Path, required=True, help="execution_metrics.parquet from spoofing run")
    parser.add_argument(
        "--candidate-deceptive-orders",
        type=Path,
        required=True,
        help="candidate_deceptive_orders.parquet from spoofing run",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory")
    parser.add_argument("--top-n", type=int, default=10, help="Top book levels to include in queue snapshots")
    parser.add_argument("--pre-window-seconds", type=float, default=30.0, help="Seconds before execution to show")
    parser.add_argument("--post-window-seconds", type=float, default=5.0, help="Seconds after execution to show")
    parser.add_argument("--max-events", type=int, default=None, help="Optional cap on matched events for smoke runs")
    parser.add_argument(
        "--queue-snapshot-mode",
        choices=("all", "key-events"),
        default="all",
        help="Use 'key-events' to write queue snapshots only for nearest pre/execution/post events, reducing dashboard memory.",
    )
    parser.add_argument("--parameter-grid-root", type=Path, default=None, help="Optional root containing kappa/lambda sensitivity runs")
    parser.add_argument("--annotations", type=Path, default=None, help="Optional analyst annotation CSV")
    parser.add_argument("--client-session-alerts", type=Path, default=None, help="Optional client-session alert parquet")
    return parser.parse_args(argv)


def _split_ids(value: Any) -> set[str]:
    if value is None:
        return set()
    return {part for part in str(value).split(";") if part}


def _priority_key(value: Any, fallback: int) -> tuple[int, float | str, int]:
    if value is None:
        return (1, 0.0, fallback)
    text = str(value)
    try:
        return (0, float(text), fallback)
    except ValueError:
        return (0, text, fallback)


def _visible_qty(order: ActiveOrder) -> float:
    if order.leaves_qty <= 0 or order.displayed_qty <= 0:
        return 0.0
    return float(order.displayed_qty)


def _market_levels(active_orders: dict[str, ActiveOrder], *, side: str, top_n: int) -> list[float]:
    by_price: dict[float, float] = defaultdict(float)
    for order in active_orders.values():
        qty = _visible_qty(order)
        if qty > 0 and order.side == side:
            by_price[float(order.price)] += qty
    return sorted(by_price, reverse=(side == "bid"))[:top_n]


def _client_queue_dict(level_orders: list[ActiveOrder]) -> str:
    total = sum(_visible_qty(order) for order in level_orders)
    by_client: dict[str, dict[str, Any]] = {}
    for position, order in enumerate(level_orders, start=1):
        client = order.client_original_id or "<missing_client>"
        qty = _visible_qty(order)
        item = by_client.setdefault(
            client,
            {
                "perc_vol": 0.0,
                "priority": position,
                "visible_qty": 0.0,
                "order_count": 0,
            },
        )
        item["visible_qty"] += qty
        item["order_count"] += 1
        item["priority"] = min(int(item["priority"]), position)
    if total > 0:
        for item in by_client.values():
            item["perc_vol"] = item["visible_qty"] / total
    return json.dumps(by_client, sort_keys=True)


def _queue_rows_for_snapshot(
    *,
    review_event_id: str,
    review_client_id: str,
    execution_sort_index: int,
    execution_ts: datetime | None,
    snapshot_event: dict[str, Any],
    snapshot_ts: datetime | None,
    snapshot_phase: str,
    active_orders: dict[str, ActiveOrder],
    candidate_order_ids: set[str],
    matched_order_ids: set[str],
    top_n: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for side in ("bid", "ask"):
        prices = _market_levels(active_orders, side=side, top_n=top_n)
        for level, price in enumerate(prices, start=1):
            level_orders = [
                order
                for order in active_orders.values()
                if order.side == side and float(order.price) == float(price) and _visible_qty(order) > 0
            ]
            level_orders.sort(key=lambda order: _priority_key(order.order_priority, order.first_seen_sort_index))
            level_qty = sum(_visible_qty(order) for order in level_orders)
            client_dict = _client_queue_dict(level_orders)
            for queue_position, order in enumerate(level_orders, start=1):
                qty = _visible_qty(order)
                rows.append(
                    {
                        "review_event_id": review_event_id,
                        "execution_sort_index": execution_sort_index,
                        "execution_ts": execution_ts,
                        "review_client_id": review_client_id,
                        "snapshot_sort_index": snapshot_event.get("sort_index"),
                        "snapshot_ts": snapshot_ts,
                        "snapshot_phase": snapshot_phase,
                        "snapshot_event_class": snapshot_event.get("event_class"),
                        "snapshot_ORDERID": snapshot_event.get("ORDERID"),
                        "side": side,
                        "price": price,
                        "level": level,
                        "level_visible_qty": level_qty,
                        "queue_position": queue_position,
                        "ORDERID": order.order_id,
                        "ORDERPRIORITY": order.order_priority,
                        "client_id": order.client_original_id,
                        "firm_id": order.firm_id,
                        "displayed_qty": order.displayed_qty,
                        "leaves_qty": order.leaves_qty,
                        "visible_qty": qty,
                        "perc_level_volume": qty / level_qty if level_qty > 0 else None,
                        "client_queue_dict": client_dict,
                        "is_review_client": order.client_original_id == review_client_id,
                        "is_candidate_deceptive_order": order.order_id in candidate_order_ids,
                        "is_matched_deceptive_cancel_order": order.order_id in matched_order_ids,
                    }
                )
    return rows


def _event_log_row(review_event: dict[str, Any], event: dict[str, Any], ts: datetime | None) -> dict[str, Any]:
    return {
        "review_event_id": review_event["review_event_id"],
        "execution_sort_index": review_event["sort_index"],
        "sort_index": event.get("sort_index"),
        "event_ts": ts,
        "event_class": event.get("event_class"),
        "ORDERID": event.get("ORDERID"),
        "ORDERPRIORITY": event.get("ORDERPRIORITY"),
        "side": event.get("side_label"),
        "price": event.get("ORDERPX"),
        "displayed_qty": event.get("DISPLAYEDQTY"),
        "leaves_qty": event.get("LEAVESQTY"),
        "last_shares": event.get("LASTSHARES"),
        "client_id": event.get("client_original_id"),
        "firm_id": event.get("firm_id"),
        "is_review_client": event.get("client_original_id") == review_event.get("client_id"),
        "is_execution_order": str(event.get("ORDERID")) == str(review_event.get("ORDERID")),
        "is_candidate_deceptive_order": str(event.get("ORDERID")) in review_event["candidate_order_ids"],
        "is_matched_deceptive_cancel_order": str(event.get("ORDERID")) in review_event["matched_order_ids"],
    }


def _prepare_review_events(execution_metrics: pl.DataFrame, max_events: int | None) -> list[dict[str, Any]]:
    matched = execution_metrics.filter(pl.col("has_matched_deceptive_cancel_window"))
    if "MSCI" in matched.columns:
        matched = matched.sort(["MSCI", "sort_index"], descending=[True, False])
    if max_events is not None:
        matched = matched.head(max_events)
    out = []
    for idx, row in enumerate(matched.iter_rows(named=True), start=1):
        candidate_ids = _split_ids(row.get("candidate_deceptive_order_ids_pre"))
        matched_ids = _split_ids(row.get("matched_deceptive_cancel_order_ids_window"))
        event_ts = _parse_ts(row.get("event_ts"))
        out.append(
            {
                **row,
                "review_event_id": f"S{int(row['sort_index'])}",
                "event_ts_parsed": event_ts,
                "candidate_order_ids": candidate_ids,
                "matched_order_ids": matched_ids,
            }
        )
    return out


def reconstruct_review_windows(
    raw_events: pl.DataFrame,
    review_events: list[dict[str, Any]],
    *,
    top_n: int,
    pre_window_seconds: float,
    post_window_seconds: float,
    queue_snapshot_mode: str = "all",
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    config = LOBConfig(top_n=max(top_n, 1), snapshot_mode="none")
    sorted_events = sort_events(raw_events)
    events = [normalize_event(raw_row, sort_index=idx, config=config) for idx, raw_row in enumerate(sorted_events.iter_rows(named=True), start=1)]

    by_sort_index = {int(event["sort_index"]): event for event in review_events}
    sorted_review = sorted(review_events, key=lambda row: int(row["sort_index"]))
    windows = []
    for row in sorted_review:
        if row["event_ts_parsed"] is None:
            continue
        windows.append(
            {
                "review": row,
                "start": row["event_ts_parsed"] - timedelta(seconds=pre_window_seconds),
                "end": row["event_ts_parsed"] + timedelta(seconds=post_window_seconds),
                "queue_sort_indexes": set(),
            }
        )

    if queue_snapshot_mode not in {"all", "key-events"}:
        raise ValueError(f"unknown queue snapshot mode: {queue_snapshot_mode}")
    if queue_snapshot_mode == "key-events":
        event_times = [(int(event["sort_index"]), choose_event_timestamp(event)) for event in events]
        for window in windows:
            review_sort_index = int(window["review"]["sort_index"])
            pre = [idx for idx, ts in event_times if ts is not None and idx < review_sort_index and window["start"] <= ts]
            post = [idx for idx, ts in event_times if ts is not None and idx > review_sort_index and ts <= window["end"]]
            selected = {review_sort_index}
            if pre:
                selected.add(max(pre))
            if post:
                selected.add(min(post))
            window["queue_sort_indexes"] = selected

    active_orders: dict[str, ActiveOrder] = {}
    pending_aggressive_residuals: dict[str, tuple[dict[str, Any], tuple[Any, ...] | None]] = {}
    non_resting_order_ids: set[str] = set()
    current_partition_id: str | None = None
    queue_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    review_summary_rows: list[dict[str, Any]] = []

    for event_index, event in enumerate(events):
        partition_id = _partition_id(event)
        if current_partition_id is None:
            current_partition_id = partition_id
        elif partition_id != current_partition_id:
            _flush_pending_aggressive_residuals(active_orders, pending_aggressive_residuals, keep_group=None)
            active_orders = {}
            pending_aggressive_residuals = {}
            non_resting_order_ids = set()
            current_partition_id = partition_id

        event_ts = choose_event_timestamp(event)
        _apply_event(
            active_orders,
            event,
            pending_aggressive_residuals=pending_aggressive_residuals,
            non_resting_order_ids=non_resting_order_ids,
        )
        next_event = events[event_index + 1] if event_index + 1 < len(events) else None
        next_group = _fill_group_key(next_event) if next_event is not None else None
        _flush_pending_aggressive_residuals(active_orders, pending_aggressive_residuals, keep_group=next_group)

        if event_ts is not None:
            for window in windows:
                review = window["review"]
                if window["start"] <= event_ts <= window["end"]:
                    event_rows.append(_event_log_row(review, event, event_ts))
                    if int(event["sort_index"]) == int(review["sort_index"]):
                        phase = "execution"
                    elif int(event["sort_index"]) < int(review["sort_index"]):
                        phase = "pre"
                    else:
                        phase = "post"
                    if queue_snapshot_mode == "all" or int(event["sort_index"]) in window["queue_sort_indexes"]:
                        queue_rows.extend(
                            _queue_rows_for_snapshot(
                                review_event_id=review["review_event_id"],
                                review_client_id=review["client_id"],
                                execution_sort_index=int(review["sort_index"]),
                                execution_ts=review["event_ts_parsed"],
                                snapshot_event=event,
                                snapshot_ts=event_ts,
                                snapshot_phase=phase,
                                active_orders=active_orders,
                                candidate_order_ids=review["candidate_order_ids"],
                                matched_order_ids=review["matched_order_ids"],
                                top_n=top_n,
                            )
                        )

        if int(event["sort_index"]) in by_sort_index:
            review = by_sort_index[int(event["sort_index"])]
            review_summary_rows.append(
                {
                    "review_event_id": review["review_event_id"],
                    "sort_index": review["sort_index"],
                    "event_ts": review.get("event_ts"),
                    "client_id": review.get("client_id"),
                    "execution_side": review.get("execution_side"),
                    "deceptive_side": review.get("deceptive_side"),
                    "fill_qty": review.get("fill_qty"),
                    "MSCI": review.get("MSCI"),
                    "SCI": review.get("SCI"),
                    "collapse_opposite_side": review.get("collapse_opposite_side"),
                    "collapse_same_side": review.get("collapse_same_side"),
                    "candidate_deceptive_visible_qty_pre": review.get("candidate_deceptive_visible_qty_pre"),
                    "matched_deceptive_cancel_visible_qty_window": review.get("matched_deceptive_cancel_visible_qty_window"),
                    "matched_deceptive_cancel_fraction_window": review.get("matched_deceptive_cancel_fraction_window"),
                    "matched_deceptive_cancel_min_delay_seconds": review.get("matched_deceptive_cancel_min_delay_seconds"),
                    "matched_deceptive_cancel_max_delay_seconds": review.get("matched_deceptive_cancel_max_delay_seconds"),
                    "weighted_net_withdrawal_qty_window": review.get("weighted_net_withdrawal_qty_window"),
                    "withdrawal_to_fill_ratio": review.get("withdrawal_to_fill_ratio"),
                    "weighted_withdrawal_to_fill_ratio": review.get("weighted_withdrawal_to_fill_ratio"),
                    "WMSCI_event": review.get("WMSCI_event"),
                    "favorable_mid_move_pre_fill": review.get("favorable_mid_move_pre_fill"),
                    "favorable_microprice_move_pre_fill": review.get("favorable_microprice_move_pre_fill"),
                    "post_cancel_mid_reversion": review.get("post_cancel_mid_reversion"),
                    "post_cancel_microprice_reversion": review.get("post_cancel_microprice_reversion"),
                    "execution_price_advantage_vs_posture_mid": review.get("execution_price_advantage_vs_posture_mid"),
                    "execution_price_advantage_vs_posture_microprice": review.get("execution_price_advantage_vs_posture_microprice"),
                    "candidate_deceptive_order_ids_pre": review.get("candidate_deceptive_order_ids_pre"),
                    "matched_deceptive_cancel_order_ids_window": review.get("matched_deceptive_cancel_order_ids_window"),
                }
            )

    return (
        pl.DataFrame(review_summary_rows, infer_schema_length=None) if review_summary_rows else pl.DataFrame(),
        pl.DataFrame(event_rows, infer_schema_length=None) if event_rows else pl.DataFrame(),
        pl.DataFrame(queue_rows, infer_schema_length=None) if queue_rows else pl.DataFrame(),
    )


def _json_records(df: pl.DataFrame) -> str:
    return json.dumps(df.to_dicts(), default=str)


def _load_parameter_review_events(root: Path, max_events: int | None) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for metadata_path in sorted(root.glob("kappa_*_lambda_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        execution_path = metadata_path.parent / "execution_metrics.parquet"
        if not execution_path.exists():
            continue
        events = _prepare_review_events(pl.read_parquet(execution_path), max_events)
        for event in events:
            event.pop("event_ts_parsed", None)
            event.pop("candidate_order_ids", None)
            event.pop("matched_order_ids", None)
        runs.append(
            {
                "kappa": metadata.get("kappa"),
                "lambda": metadata.get("lambda_"),
                "label": f"kappa={metadata.get('kappa')}, lambda={metadata.get('lambda_')}",
                "events": events,
            }
        )
    return runs


def _load_llm_reviews(output_dir: Path) -> dict[str, str]:
    root = output_dir / "llm_reviews"
    reviews: dict[str, str] = {}
    if not root.exists():
        return reviews
    for response_path in root.glob("*/response.md"):
        reviews[response_path.parent.name] = response_path.read_text()
    return reviews


def _load_annotations(path: Path | None) -> pl.DataFrame:
    if path is None or not path.exists():
        return pl.DataFrame({"review_event_id": [], "analyst_label": [], "confidence": [], "benign_explanation": [], "notes": [], "reviewer": [], "reviewed_at_utc": []})
    from spoofing_detection.lob.annotations import validate_annotations

    return validate_annotations(pl.read_csv(path))


def _load_optional_parquet(path: Path | None) -> pl.DataFrame:
    if path is None or not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)


def write_review_artifacts(*, output_dir: Path, event_log: pl.DataFrame, queue: pl.DataFrame) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = output_dir / "matched_spoofing_event_log.parquet"
    queue_path = output_dir / "matched_spoofing_lob_queue.parquet"
    event_log.write_parquet(event_log_path)
    queue.write_parquet(queue_path)
    return {"event_log": event_log_path, "queue": queue_path}


def write_dashboard(
    path: Path,
    *,
    review_events: pl.DataFrame,
    event_log: pl.DataFrame,
    queue: pl.DataFrame,
    parameter_review_events: list[dict[str, Any]] | None = None,
    llm_reviews: dict[str, str] | None = None,
    annotations: pl.DataFrame | None = None,
    client_session_alerts: pl.DataFrame | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    html = f"""<!doctype html>
<html>
<head>
<meta charset=\"utf-8\" />
<title>Spoofing event review dashboard</title>
<script src=\"https://cdn.plot.ly/plotly-2.35.2.min.js\"></script>
<style>
body {{ font-family: Inter, Arial, sans-serif; margin: 0; background: #f7f8fb; color: #18202f; }}
.page {{ max-width: 1420px; margin: 24px auto; padding: 0 24px; }}
h1 {{ margin-bottom: 0.2rem; text-align: center; }}
.note {{ color: #4d5a6d; max-width: 1100px; margin-left: auto; margin-right: auto; }}
.intro {{ color: #364152; max-width: 1180px; line-height: 1.45; }}
.intro ul {{ margin: 0.5rem 0 0.2rem 1.2rem; padding: 0; }}
.intro li {{ margin: 0.25rem 0; }}
.parameter-table {{ margin-top: 0.7rem; max-width: 900px; }}
.parameter-table th {{ position: static; }}
.parameter-table td:first-child {{ font-weight: 600; color: #273449; white-space: nowrap; }}
.controls {{ display: flex; gap: 16px; align-items: center; justify-content: center; margin: 18px auto; flex-wrap: wrap; }}
select {{ padding: 8px; min-width: 520px; }}
.card {{ background: white; border: 1px solid #dfe5ef; border-radius: 10px; padding: 14px; margin: 14px auto; box-shadow: 0 1px 2px rgba(20,30,50,0.05); }}
#summary {{ font-size: 0.95rem; line-height: 1.45; }}
table {{ border-collapse: collapse; width: 100%; font-size: 0.86rem; }}
th, td {{ border-bottom: 1px solid #e8edf5; padding: 6px; text-align: left; }}
th {{ background: #f1f4f9; position: sticky; top: 0; }}
.badge {{ display: inline-block; padding: 2px 6px; border-radius: 6px; background: #fee2e2; color: #991b1b; font-weight: 600; }}
#llmReview pre {{ white-space: pre-wrap; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; }}
</style>
</head>
<body>
<div class=\"page\">
<h1>Spoofing event review dashboard</h1>
<p class=\"note\">This dashboard is intended as the manual review tool for the strongest candidate spoofing events. It shows only matched deceptive-order cancellations: the client first has candidate opposite-side liquidity, executes a small order, and then cancels the same candidate order inside the post-execution window. These are surveillance cues, not proof of intent.</p>
<div class=\"card intro\">
  <h2>How to read this dashboard</h2>
  <p>The model starts from a small passive execution by a client and checks whether the same client had a recent opposite-side candidate deceptive order and then cancelled that same order after the execution.</p>
  <ul>
    <li><b>DWI</b> is the multilevel distance-weighted imbalance of the client's top-N footprint: positive values are ask-heavy, negative values are bid-heavy.</li>
    <li><b>SCI</b> measures how abruptly DWI changes between the pre-execution and post-cancel snapshots.</li>
    <li><b>MSCI</b> combines SCI with side-specific collapse: it is high only when the opposite-side candidate liquidity collapses more than the same-side liquidity.</li>
    <li><b>Price-response diagnostics</b> are signed so positive values mean the mid-price, microprice, or execution price moved in the direction favorable to the small execution; they are economic consistency checks, not causal proof.</li>
    <li><b>Candidate deceptive orders</b> are same-client, opposite-side, top-N orders visible before the execution and posted within the episode age window.</li>
    <li><b>Matched spoofing-like events</b> are the strict subset where one of those candidate order IDs is cancelled after the execution.</li>
  </ul>
  <p>The chart overlays total visible depth, candidate-spoofer depth, and the executed small-order volume. These outputs are exploratory surveillance evidence, not proof of intent.</p>
  <table class=\"parameter-table\">
    <thead><tr><th>Parameter</th><th>Value</th><th>Meaning</th></tr></thead>
    <tbody>
      <tr><td>LOB review top-N depth</td><td>10</td><td>Number of bid and ask price levels reconstructed in the manual event-review chart.</td></tr>
      <tr><td>Metric-run top-N depth</td><td>3</td><td>Depth used by the MSCI/MCPS run that produced the selected candidate events.</td></tr>
      <tr><td>Candidate-order age window</td><td>600 seconds</td><td>Maximum time between first seeing the opposite-side candidate order and the small execution.</td></tr>
      <tr><td>Local review window</td><td>30 seconds before, 5 seconds after</td><td>Event-log and queue-reconstruction interval shown around each selected execution.</td></tr>
      <tr><td>Post-execution cancellation window</td><td>1 second</td><td>Window after the small execution in which a candidate order cancellation is matched.</td></tr>
      <tr><td>kappa</td><td>1</td><td>Depth-kernel parameter controlling the protection-from-execution component.</td></tr>
      <tr><td>lambda</td><td>1</td><td>Depth-kernel parameter controlling decay of informational visibility with distance.</td></tr>
      <tr><td>epsilon</td><td>1e-12</td><td>Small stabilizer used in denominators.</td></tr>
      <tr><td>MCPS gamma grid</td><td>configured run grid</td><td>Thresholds used to aggregate repeated high-MSCI executions into client-level MCPS scores.</td></tr>
    </tbody>
  </table>
</div>
<div class=\"card\"><h2>Client-session alerts</h2><div id=\"clientSessionAlerts\"></div></div>
<div class=\"card\"><div id=\"overview\"></div></div>
<div class=\"controls\"><label for=\"parameterSelect\"><b>Choose kappa/lambda</b></label><select id=\"parameterSelect\"></select><label for=\"eventSelect\"><b>Choose candidate event</b></label><select id=\"eventSelect\"></select></div>
<div class=\"card\" id=\"summary\"></div>
<div class=\"card\"><div id=\"lob\"></div></div>
<div class=\"card\"><h2>Analyst annotation</h2><div id=\"annotation\"></div></div>
<div class=\"card\"><h2>LLM surveillance review</h2><div id=\"llmReview\"></div></div>
<div class=\"card\"><h2>Actual events in zoom window</h2><div id=\"eventTable\"></div></div>
<script>
const baseReviewEvents = {_json_records(review_events)};
const parameterRuns = {json.dumps(parameter_review_events or [], default=str)};
let reviewEvents = parameterRuns.length ? parameterRuns[0].events : baseReviewEvents;
const eventLog = {_json_records(event_log)};
const queueRows = {_json_records(queue)};
const llmReviews = {json.dumps(llm_reviews or {}, default=str)};
const annotations = {_json_records(annotations if annotations is not None else pl.DataFrame())};
const clientSessionAlerts = {_json_records(client_session_alerts if client_session_alerts is not None else pl.DataFrame())};
function byEvent(id, rows) {{ return rows.filter(r => r.review_event_id === id); }}
function finiteNumber(value) {{ const number = Number(value); return Number.isFinite(number) ? number : null; }}
function metricText(value, digits=6) {{ const number = finiteNumber(value); return number === null ? 'NA' : number.toFixed(digits); }}
function scoreValue(ev) {{ const w = finiteNumber(ev.WMSCI_event); return w !== null ? w : finiteNumber(ev.MSCI); }}
function label(ev) {{ return `${{ev.review_event_id}} | ${{ev.event_ts}} | client=${{ev.client_id}} | WMSCI=${{metricText(ev.WMSCI_event, 4)}} | MSCI=${{metricText(ev.MSCI, 4)}} | FPM(mid)=${{metricText(ev.favorable_mid_move_pre_fill, 4)}} | REV(mid)=${{metricText(ev.post_cancel_mid_reversion, 4)}} | orders=${{ev.matched_deceptive_cancel_order_ids_window}}`; }}
function renderOverview() {{
  const x = reviewEvents.map(r => r.event_ts);
  const y = reviewEvents.map(scoreValue);
  const text = reviewEvents.map(label);
  Plotly.newPlot('overview', [{{x, y, text, mode:'markers', type:'scattergl', marker:{{color:'#d62728', size:9}}, hovertemplate:'%{{text}}<extra></extra>'}}],
    {{title:'All matched spoofing-like candidate events', xaxis:{{title:'execution time'}}, yaxis:{{title:'WMSCI event score'}}, margin:{{t:45}}}}, {{responsive:true}});
}}
function renderClientSessionAlerts() {{
  const el = document.getElementById('clientSessionAlerts');
  if (!clientSessionAlerts.length) {{ el.innerHTML = '<p>No client-session alerts loaded.</p>'; return; }}
  let html = ['<table><thead><tr><th>Client</th><th>Alert score</th><th>Events</th><th>MCPS</th><th>Action</th></tr></thead><tbody>'];
  for (const row of clientSessionAlerts) {{
    html.push(`<tr><td>${{escapeHtml(row.client_id)}}</td><td>${{Number(row.alert_score || 0).toFixed(3)}}</td><td>${{row.event_count || ''}}</td><td>${{Number(row.mcps_at_threshold || 0).toFixed(3)}}</td><td>${{escapeHtml(row.recommended_action || '')}}</td></tr>`);
  }}
  html.push('</tbody></table>');
  el.innerHTML = html.join('');
}}
function renderSummary(ev) {{
  document.getElementById('summary').innerHTML = `<h2>${{ev.review_event_id}} <span class="badge">matched deceptive-order cancellation</span></h2>
  <b>time:</b> ${{ev.event_ts}} &nbsp; <b>client:</b> ${{ev.client_id}} &nbsp; <b>execution side:</b> ${{ev.execution_side}} &nbsp; <b>deceptive side:</b> ${{ev.deceptive_side}}<br>
  <b>fill qty:</b> ${{ev.fill_qty}} &nbsp; <b>WMSCI:</b> ${{Number(ev.WMSCI_event || 0).toFixed(6)}} &nbsp; <b>MSCI:</b> ${{Number(ev.MSCI || 0).toFixed(6)}} &nbsp; <b>SCI:</b> ${{Number(ev.SCI || 0).toFixed(6)}}<br>
  <b>favorable pre-fill move:</b> mid ${{metricText(ev.favorable_mid_move_pre_fill)}} / microprice ${{metricText(ev.favorable_microprice_move_pre_fill)}} &nbsp; <b>post-cancel reversion:</b> mid ${{metricText(ev.post_cancel_mid_reversion)}} / microprice ${{metricText(ev.post_cancel_microprice_reversion)}}<br>
  <b>execution advantage vs posture:</b> mid ${{metricText(ev.execution_price_advantage_vs_posture_mid)}} / microprice ${{metricText(ev.execution_price_advantage_vs_posture_microprice)}}<br>
  <b>candidate visible qty pre:</b> ${{ev.candidate_deceptive_visible_qty_pre}} &nbsp; <b>matched cancel qty:</b> ${{ev.matched_deceptive_cancel_visible_qty_window}} &nbsp; <b>matched fraction:</b> ${{ev.matched_deceptive_cancel_fraction_window}}<br>
  <b>weighted withdrawal:</b> ${{ev.weighted_net_withdrawal_qty_window}} &nbsp; <b>withdrawal/fill:</b> ${{ev.withdrawal_to_fill_ratio}} &nbsp; <b>cancel delay:</b> ${{ev.matched_deceptive_cancel_min_delay_seconds}}–${{ev.matched_deceptive_cancel_max_delay_seconds}}s<br>
  <b>candidate order ids:</b> ${{ev.candidate_deceptive_order_ids_pre}}<br>
  <b>matched cancelled ids:</b> ${{ev.matched_deceptive_cancel_order_ids_window}}`;
}}
function renderLOB(ev) {{
  const rows = byEvent(ev.review_event_id, queueRows);
  const maxLevel = Math.max(...rows.map(r => Number(r.level) || 0), 1);
  const yLabels = [];
  for (let level = maxLevel; level >= 1; level--) yLabels.push(`bid ${{level}}`);
  for (let level = 1; level <= maxLevel; level++) yLabels.push(`ask ${{level}}`);

  function chooseSnapshot(phase) {{
    const phaseRows = rows.filter(r => r.snapshot_phase === phase);
    if (!phaseRows.length) return null;
    let sortIndex;
    if (phase === 'pre') sortIndex = Math.max(...phaseRows.map(r => Number(r.snapshot_sort_index)));
    else if (phase === 'execution') sortIndex = Number(ev.sort_index);
    else sortIndex = Math.max(...phaseRows.map(r => Number(r.snapshot_sort_index)));
    if (!phaseRows.some(r => Number(r.snapshot_sort_index) === sortIndex)) {{
      let best = phaseRows[0];
      for (const r of phaseRows) {{
        if (Math.abs(Number(r.snapshot_sort_index) - Number(ev.sort_index)) < Math.abs(Number(best.snapshot_sort_index) - Number(ev.sort_index))) best = r;
      }}
      sortIndex = Number(best.snapshot_sort_index);
    }}
    return phaseRows.filter(r => Number(r.snapshot_sort_index) === sortIndex);
  }}

  function bestQuotesForStage(phase) {{
    const chosen = chooseSnapshot(phase) || [];
    let bestBid = null;
    let bestAsk = null;
    for (const r of chosen) {{
      const price = Number(r.price);
      if (!Number.isFinite(price)) continue;
      if (r.side === 'bid' && (bestBid === null || price > bestBid)) bestBid = price;
      if (r.side === 'ask' && (bestAsk === null || price < bestAsk)) bestAsk = price;
    }}
    return {{bestBid, bestAsk}};
  }}

  function formatPrice(value) {{
    if (value === null || value === undefined || !Number.isFinite(Number(value))) return 'NA';
    return Number(value).toFixed(4).replace(/0+$/, '').replace(/\\.$/, '');
  }}

  function stageLabel(phase) {{
    const chosen = chooseSnapshot(phase) || [];
    if (!chosen.length) return 'no snapshot';
    const row = chosen[0];
    const ts = row.snapshot_ts || '';
    const sortIndex = row.snapshot_sort_index;
    const quotes = bestQuotesForStage(phase);
    const quoteText = `best bid ${{formatPrice(quotes.bestBid)}}<br>best ask ${{formatPrice(quotes.bestAsk)}}`;
    if (phase === 'execution') return `${{ts}}<br>sort ${{sortIndex}}<br>${{quoteText}}<br>fill ${{ev.fill_qty}}`;
    return `${{ts}}<br>sort ${{sortIndex}}<br>${{quoteText}}`;
  }}

  function stageArrays(phase) {{
    const chosen = chooseSnapshot(phase) || [];
    const byLevel = new Map();
    for (const r of chosen) {{
      const key = `${{r.side}}|${{r.level}}`;
      const item = byLevel.get(key) || {{total: Number(r.level_visible_qty) || 0, candidate: 0, executed: 0, price: r.price, queue: r.client_queue_dict}};
      if (r.is_candidate_deceptive_order || r.is_matched_deceptive_cancel_order) item.candidate += Number(r.visible_qty) || 0;
      byLevel.set(key, item);
    }}

    if (phase === 'execution') {{
      const eventRows = byEvent(ev.review_event_id, eventLog);
      const execEvent = eventRows.find(r => r.is_execution_order && Number(r.sort_index) === Number(ev.sort_index)) ||
        eventRows.find(r => r.is_execution_order && r.event_class === 'fill' && Number(r.last_shares || 0) > 0) ||
        eventRows.find(r => r.is_execution_order);
      if (execEvent) {{
        const execSide = execEvent.side;
        const execPrice = Number(execEvent.price);
        const execQty = Number(execEvent.last_shares || ev.fill_qty || 0);
        const sidePrices = Array.from(new Set(chosen.filter(r => r.side === execSide).map(r => Number(r.price))));
        sidePrices.sort((a, b) => execSide === 'bid' ? b - a : a - b);
        let execLevel = sidePrices.findIndex(p => Math.abs(p - execPrice) < 1e-12) + 1;
        if (execLevel <= 0) {{
          execLevel = sidePrices.filter(p => execSide === 'bid' ? p > execPrice : p < execPrice).length + 1;
        }}
        if (execLevel >= 1 && execLevel <= maxLevel && execQty > 0) {{
          const key = `${{execSide}}|${{execLevel}}`;
          const item = byLevel.get(key) || {{total: 0, candidate: 0, executed: 0, price: execPrice, queue: ''}};
          item.executed += execQty;
          item.price = item.price || execPrice;
          byLevel.set(key, item);
        }}
      }}
    }}

    const totalBid = [], totalAsk = [], candidateBid = [], candidateAsk = [], executedBid = [], executedAsk = [], hover = [];
    for (const label of yLabels) {{
      const [side, levelText] = label.split(' ');
      const item = byLevel.get(`${{side}}|${{levelText}}`) || {{total: 0, candidate: 0, executed: 0, price: '', queue: ''}};
      totalBid.push(side === 'bid' ? item.total : 0);
      totalAsk.push(side === 'ask' ? item.total : 0);
      candidateBid.push(side === 'bid' ? item.candidate : 0);
      candidateAsk.push(side === 'ask' ? item.candidate : 0);
      executedBid.push(side === 'bid' ? item.executed : 0);
      executedAsk.push(side === 'ask' ? item.executed : 0);
      hover.push(`${{label}} @ ${{item.price}}<br>total=${{item.total}}<br>candidate spoofer=${{item.candidate}}<br>executed volume=${{item.executed}}<br>client queue=${{item.queue}}`);
    }}
    return {{totalBid, totalAsk, candidateBid, candidateAsk, executedBid, executedAsk, hover}};
  }}

  const stages = [
    ['pre', '1. pre-execution/posturing'],
    ['execution', '2. small execution'],
    ['post', '3. post-cancel']
  ];
  const traces = [];
  const xaxes = ['x', 'x2', 'x3'];
  for (let i = 0; i < stages.length; i++) {{
    const [phase, title] = stages[i];
    const vals = stageArrays(phase);
    const common = {{y: yLabels, type:'bar', orientation:'h', hovertext: vals.hover, hovertemplate:'%{{hovertext}}<extra></extra>', xaxis: xaxes[i], yaxis:'y'}};
    traces.push({{...common, x: vals.totalBid, name:'bid total', legendgroup:'bid total', showlegend:i===0, marker:{{color:'rgba(44,160,44,0.24)'}}}});
    traces.push({{...common, x: vals.totalAsk, name:'ask total', legendgroup:'ask total', showlegend:i===0, marker:{{color:'rgba(214,39,40,0.22)'}}}});
    traces.push({{...common, x: vals.candidateBid, name:'candidate spoofer bid', legendgroup:'candidate bid', showlegend:i===0, marker:{{color:'rgba(0,100,0,0.95)'}}, width:0.48}});
    traces.push({{...common, x: vals.candidateAsk, name:'candidate spoofer ask', legendgroup:'candidate ask', showlegend:i===0, marker:{{color:'rgba(150,0,0,0.95)'}}, width:0.48}});
    traces.push({{...common, x: vals.executedBid, name:'executed volume bid', legendgroup:'executed bid', showlegend:i===0, marker:{{color:'rgba(0,55,0,1.0)', line:{{color:'#111', width:1}}}}, width:0.30}});
    traces.push({{...common, x: vals.executedAsk, name:'executed volume ask', legendgroup:'executed ask', showlegend:i===0, marker:{{color:'rgba(110,0,0,1.0)', line:{{color:'#111', width:1}}}}, width:0.30}});
  }}
  Plotly.newPlot('lob', traces,
    {{
      title:'LOB depth across spoofing stages: total volume and candidate-spoofer volume',
      barmode:'overlay',
      bargap:0.18,
      yaxis:{{categoryorder:'array', categoryarray:yLabels, autorange:'reversed'}},
      xaxis:{{domain:[0.00,0.30], title:{{text:'1. pre-execution/posturing', standoff:10}}, rangemode:'tozero'}},
      xaxis2:{{domain:[0.35,0.65], title:{{text:'2. small execution', standoff:10}}, rangemode:'tozero'}},
      xaxis3:{{domain:[0.70,1.00], title:{{text:'3. post-cancel', standoff:10}}, rangemode:'tozero'}},
      annotations:[
        {{text:'bid side', xref:'paper', yref:'paper', x:-0.055, y:0.77, textangle:-90, showarrow:false, font:{{color:'#1b7f1b', size:12}}}},
        {{text:'ask side', xref:'paper', yref:'paper', x:-0.055, y:0.27, textangle:-90, showarrow:false, font:{{color:'#b22222', size:12}}}},
        {{text:stageLabel('pre'), xref:'paper', yref:'paper', x:0.15, y:1.08, showarrow:false, align:'center', bgcolor:'rgba(255,255,255,0.86)', bordercolor:'#d7dde8', borderpad:4, font:{{size:11, color:'#364152'}}}},
        {{text:stageLabel('execution'), xref:'paper', yref:'paper', x:0.50, y:1.08, showarrow:false, align:'center', bgcolor:'rgba(255,255,255,0.86)', bordercolor:'#d7dde8', borderpad:4, font:{{size:11, color:'#364152'}}}},
        {{text:stageLabel('post'), xref:'paper', yref:'paper', x:0.85, y:1.08, showarrow:false, align:'center', bgcolor:'rgba(255,255,255,0.86)', bordercolor:'#d7dde8', borderpad:4, font:{{size:11, color:'#364152'}}}}
      ],
      legend:{{orientation:'h', y:-0.16}},
      margin:{{l:100,t:125,b:95}}
    }}, {{responsive:true}});
}}
function escapeHtml(text) {{ return String(text).replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c])); }}
function renderAnnotation(ev) {{
  const row = annotations.find(r => r.review_event_id === ev.review_event_id);
  const el = document.getElementById('annotation');
  if (!row) {{ el.innerHTML = '<p>No analyst annotation found for this event.</p>'; return; }}
  el.innerHTML = `<table>
    <tr><th>Label</th><td>${{escapeHtml(row.analyst_label)}}</td></tr>
    <tr><th>Confidence</th><td>${{Number(row.confidence || 0).toFixed(2)}}</td></tr>
    <tr><th>Benign explanation</th><td>${{escapeHtml(row.benign_explanation || '')}}</td></tr>
    <tr><th>Reviewer</th><td>${{escapeHtml(row.reviewer || '')}}</td></tr>
    <tr><th>Reviewed at</th><td>${{escapeHtml(row.reviewed_at_utc || '')}}</td></tr>
    <tr><th>Notes</th><td>${{escapeHtml(row.notes || '')}}</td></tr>
  </table>`;
}}
function renderLLMReview(ev) {{
  const review = llmReviews[ev.review_event_id];
  if (review) {{
    document.getElementById('llmReview').innerHTML = `<pre>${{escapeHtml(review)}}</pre>`;
  }} else {{
    document.getElementById('llmReview').innerHTML = `<p>No precomputed LLM review found for ${{ev.review_event_id}}.</p><p>Generate it with <code>scripts/build_spoofing_event_dossier.py</code> and <code>scripts/analyze_spoofing_event_with_llm.py</code>.</p>`;
  }}
}}
function renderEventTable(ev) {{
  const rows = byEvent(ev.review_event_id, eventLog).sort((a,b) => a.sort_index-b.sort_index);
  const html = ['<table><thead><tr><th>sort</th><th>time</th><th>class</th><th>side</th><th>price</th><th>ORDERID</th><th>client</th><th>leaves</th><th>displayed</th><th>last shares</th><th>flags</th></tr></thead><tbody>'];
  for (const r of rows) {{
    const flags = [r.is_execution_order?'execution':'', r.is_candidate_deceptive_order?'candidate':'', r.is_matched_deceptive_cancel_order?'matched-cancel':'', r.is_review_client?'client':''].filter(Boolean).join(', ');
    html.push(`<tr><td>${{r.sort_index}}</td><td>${{r.event_ts}}</td><td>${{r.event_class}}</td><td>${{r.side ?? ''}}</td><td>${{r.price ?? ''}}</td><td>${{r.ORDERID ?? ''}}</td><td>${{r.client_id ?? ''}}</td><td>${{r.leaves_qty ?? ''}}</td><td>${{r.displayed_qty ?? ''}}</td><td>${{r.last_shares ?? ''}}</td><td>${{flags}}</td></tr>`);
  }}
  html.push('</tbody></table>');
  document.getElementById('eventTable').innerHTML = html.join('');
}}
function update(id) {{ const ev = reviewEvents.find(r => r.review_event_id === id); renderSummary(ev); renderLOB(ev); renderAnnotation(ev); renderLLMReview(ev); renderEventTable(ev); }}
const select = document.getElementById('eventSelect');
const parameterSelect = document.getElementById('parameterSelect');
function populateEvents() {{
  select.innerHTML = '';
  for (const ev of reviewEvents) {{ const opt = document.createElement('option'); opt.value = ev.review_event_id; opt.textContent = label(ev); select.appendChild(opt); }}
}}
if (parameterRuns.length) {{
  for (let i = 0; i < parameterRuns.length; i++) {{ const opt = document.createElement('option'); opt.value = String(i); opt.textContent = parameterRuns[i].label; parameterSelect.appendChild(opt); }}
}} else {{
  const opt = document.createElement('option'); opt.value = 'base'; opt.textContent = 'current metric run'; parameterSelect.appendChild(opt);
  parameterSelect.disabled = true;
}}
parameterSelect.addEventListener('change', e => {{
  if (parameterRuns.length) reviewEvents = parameterRuns[Number(e.target.value)].events;
  populateEvents();
  renderOverview();
  if (reviewEvents.length) update(reviewEvents[0].review_event_id);
}});
select.addEventListener('change', e => update(e.target.value));
populateEvents();
renderClientSessionAlerts();
renderOverview();
if (reviewEvents.length) update(reviewEvents[0].review_event_id);
</script>
</div>
</body>
</html>
"""
    path.write_text(html)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    raw_events = pl.read_parquet(args.input)
    execution_metrics = pl.read_parquet(args.execution_metrics)
    # Candidate file is loaded to verify provenance and fail early if absent/corrupt.
    pl.read_parquet(args.candidate_deceptive_orders)
    review_events = _prepare_review_events(execution_metrics, args.max_events)
    parameter_review_events = _load_parameter_review_events(args.parameter_grid_root, args.max_events) if args.parameter_grid_root else []
    if parameter_review_events:
        by_id = {event["review_event_id"]: event for event in review_events}
        for run in parameter_review_events:
            for event in run["events"]:
                if event["review_event_id"] not in by_id:
                    clone = dict(event)
                    clone["event_ts_parsed"] = _parse_ts(clone.get("event_ts"))
                    clone["candidate_order_ids"] = _split_ids(clone.get("candidate_deceptive_order_ids_pre"))
                    clone["matched_order_ids"] = _split_ids(clone.get("matched_deceptive_cancel_order_ids_window"))
                    by_id[clone["review_event_id"]] = clone
        review_events = sorted(by_id.values(), key=lambda row: int(row["sort_index"]))
    if not review_events:
        raise ValueError("no matched deceptive-order cancellation events found")
    review_df, event_log_df, queue_df = reconstruct_review_windows(
        raw_events,
        review_events,
        top_n=args.top_n,
        pre_window_seconds=args.pre_window_seconds,
        post_window_seconds=args.post_window_seconds,
        queue_snapshot_mode=args.queue_snapshot_mode,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    review_path = args.output_dir / "matched_spoofing_events.parquet"
    event_log_path = args.output_dir / "matched_spoofing_event_log.parquet"
    queue_path = args.output_dir / "matched_spoofing_lob_queue.parquet"
    dashboard_path = args.output_dir / "matched_spoofing_event_review_dashboard.html"
    metadata_path = args.output_dir / "metadata.json"
    review_df.write_parquet(review_path)
    artifact_paths = write_review_artifacts(output_dir=args.output_dir, event_log=event_log_df, queue=queue_df)
    event_log_path = artifact_paths["event_log"]
    queue_path = artifact_paths["queue"]
    write_dashboard(
        dashboard_path,
        review_events=review_df,
        event_log=event_log_df,
        queue=queue_df,
        parameter_review_events=parameter_review_events,
        llm_reviews=_load_llm_reviews(args.output_dir),
        annotations=_load_annotations(args.annotations),
        client_session_alerts=_load_optional_parquet(args.client_session_alerts),
    )
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": str(args.input),
        "execution_metrics": str(args.execution_metrics),
        "candidate_deceptive_orders": str(args.candidate_deceptive_orders),
        "output_dir": str(args.output_dir),
        "top_n": args.top_n,
        "pre_window_seconds": args.pre_window_seconds,
        "post_window_seconds": args.post_window_seconds,
        "queue_snapshot_mode": args.queue_snapshot_mode,
        "parameter_grid_root": str(args.parameter_grid_root) if args.parameter_grid_root else None,
        "parameter_run_count": len(parameter_review_events),
        "review_event_count": review_df.height,
        "event_log_rows": event_log_df.height,
        "queue_rows": queue_df.height,
        "paths": {
            "matched_spoofing_events": str(review_path),
            "matched_spoofing_event_log": str(event_log_path),
            "matched_spoofing_lob_queue": str(queue_path),
            "dashboard": str(dashboard_path),
        },
        "command": sys.argv,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))
    print(f"matched_events: {review_df.height}")
    print(f"event_log_rows: {event_log_df.height}")
    print(f"queue_rows: {queue_df.height}")
    print(f"queue_parquet: {queue_path}")
    print(f"dashboard: {dashboard_path}")
    print(f"metadata: {metadata_path}")


if __name__ == "__main__":
    main()
