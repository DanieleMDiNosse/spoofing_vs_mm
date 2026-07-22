from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from pathlib import Path

import polars as pl

from spoofing_detection.lob.models import ActiveOrder


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_spoofing_event_review_dashboard.py"
_spec = importlib.util.spec_from_file_location("build_spoofing_event_review_dashboard", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_review)


def _order(
    order_id: str,
    *,
    client: str | None,
    qty: float,
    priority: str,
    first_seen: int,
    side: str = "bid",
    price: float = 10.0,
) -> ActiveOrder:
    return ActiveOrder(
        order_id=order_id,
        side=side,
        price=price,
        leaves_qty=qty,
        displayed_qty=qty,
        order_qty=qty,
        order_priority=priority,
        order_type_code=2,
        order_type_label="limit",
        time_in_force_code=None,
        firm_id="firm",
        client_original_id=client,
        first_seen_sort_index=first_seen,
        last_update_sort_index=first_seen,
        last_event_class="new_order",
    )


def _minimal_dashboard_frames() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    review_events = pl.DataFrame(
        [
            {
                "review_event_id": "S10",
                "event_ts": "2024",
                "client_id": "C1",
                "MSCI": 0.1,
                "sort_index": 10,
                "matched_deceptive_cancel_order_ids_window": "A",
                "favorable_mid_move_pre_fill": 0.02,
                "post_cancel_mid_reversion": 0.01,
                "execution_price_advantage_vs_posture_mid": 0.03,
            }
        ]
    )
    event_log = pl.DataFrame(
        [
            {
                "review_event_id": "S10",
                "sort_index": 10,
                "event_ts": "2024",
                "event_class": "fill",
                "is_review_client": True,
                "is_execution_order": True,
                "is_candidate_deceptive_order": False,
                "is_matched_deceptive_cancel_order": False,
            },
            {
                "review_event_id": "S10",
                "sort_index": 11,
                "event_ts": "2024",
                "event_class": "cancel",
                "is_review_client": True,
                "is_execution_order": False,
                "is_candidate_deceptive_order": True,
                "is_matched_deceptive_cancel_order": True,
            },
        ]
    )
    queue = pl.DataFrame([{"review_event_id": "S10", "snapshot_phase": "execution", "snapshot_sort_index": 10, "side": "bid", "level": 1, "price": 10.0, "level_visible_qty": 100.0, "visible_qty": 100.0, "is_candidate_deceptive_order": True, "is_matched_deceptive_cancel_order": False, "client_queue_dict": "{}"}])
    return review_events, event_log, queue


def _raw_event(
    seq: int,
    *,
    event_type: int,
    order_id: str,
    side: int,
    price: float,
    leaves_qty: float,
    displayed_qty: float,
    last_shares: float | None = None,
    trading_capacity: int = 1,
) -> dict:
    return {
        "TRADEDATE": "2024-01-02",
        "MIC": "XMIL",
        "MARKETCODE": "MTA",
        "SYMBOLINDEX": 123,
        "EMM (*)": 1,
        "ISIN": "TEST0000001",
        "SEQUENCETIME": f"2024-01-02T09:00:0{seq}",
        "HDR_APPLKEYSEQUENCENUMBER": seq,
        "HDR_HWMSEQUENCENUMBER": seq,
        "HDR_OFFSETID": seq,
        "ROW_NUMBER": seq,
        "EVENTID": f"E{seq}",
        "ORDEREVENTTYPE (*)": event_type,
        "ORDERID": order_id,
        "ORDERPRIORITY": str(seq),
        "ORDERSIDE (*)": side,
        "ORDERPX": price,
        "ORDERQTY": 25.0,
        "DISPLAYEDQTY": displayed_qty,
        "LEAVESQTY": leaves_qty,
        "LASTSHARES": last_shares,
        "LASTTRADEDPX": price if last_shares else None,
        "ORDERTYPE (*)": 2,
        "TIMEINFORCE (*)": 0,
        "PASSIVEORDER": "Y",
        "AGGRESSIVEORDER": "N",
        "FIRMID": "F1",
        "NMSC_ORIGINALCLIENTIDSHORTCODE": "C1",
        "ORDER_TRADINGCAPACITY (*)": trading_capacity,
        "ORDER_TRADINGCAPACITY (*) (Tooltip)": "1 : Dealing_on_own_account",
    }


def test_client_queue_dict_reports_client_percent_volume_and_priority():
    level_orders = [_order("A", client="client_1", qty=30.0, priority="1", first_seen=1), _order("B", client="client_2", qty=20.0, priority="2", first_seen=2), _order("C", client="client_1", qty=50.0, priority="3", first_seen=3)]
    payload = json.loads(_review._client_queue_dict(level_orders))
    assert payload["client_1"]["perc_vol"] == 0.8
    assert payload["client_1"]["priority"] == 1
    assert payload["client_1"]["visible_qty"] == 80.0
    assert payload["client_1"]["order_count"] == 2
    assert payload["client_2"]["perc_vol"] == 0.2
    assert payload["client_2"]["priority"] == 2


def test_parse_args_loads_event_review_parameters_from_config_with_cli_overrides(tmp_path: Path):
    config_path = tmp_path / "spoofing_parameters.json"
    config_path.write_text(
        json.dumps(
            {
                "event_review": {
                    "top_n": 10,
                    "pre_window_seconds": 30.0,
                    "post_window_seconds": 35.0,
                    "queue_snapshot_mode": "key-events",
                }
            }
        )
    )

    args = _review.parse_args(
        [
            "--config",
            str(config_path),
            "--input",
            str(tmp_path / "input.parquet"),
            "--execution-metrics",
            str(tmp_path / "execution_metrics.parquet"),
            "--candidate-deceptive-orders",
            str(tmp_path / "candidate_deceptive_orders.parquet"),
            "--output-dir",
            str(tmp_path / "event_review"),
            "--top-n",
            "12",
        ]
    )

    assert args.config == config_path
    assert args.top_n == 12
    assert args.pre_window_seconds == 30.0
    assert args.post_window_seconds == 35.0
    assert args.queue_snapshot_mode == "key-events"


def test_queue_rows_include_candidate_and_matched_flags_with_positions():
    active_orders = {"A": _order("A", client="client_1", qty=30.0, priority="1", first_seen=1), "B": _order("B", client="client_2", qty=20.0, priority="2", first_seen=2)}
    rows = _review._queue_rows_for_snapshot(
        review_event_id="E0001",
        review_client_id="client_1",
        execution_sort_index=10,
        execution_ts=None,
        snapshot_event={"sort_index": 10, "event_class": "fill", "ORDERID": "X"},
        snapshot_ts=None,
        snapshot_phase="execution",
        active_orders=active_orders,
        candidate_order_ids={"A"},
        matched_order_ids={"A"},
        top_n=1,
    )
    assert [row["ORDERID"] for row in rows] == ["A", "B"]
    assert [row["queue_position"] for row in rows] == [1, 2]
    assert rows[0]["is_review_client"] is True
    assert rows[0]["is_candidate_deceptive_order"] is True
    assert rows[0]["is_matched_deceptive_cancel_order"] is True
    assert rows[1]["is_review_client"] is False
    assert rows[1]["is_candidate_deceptive_order"] is False


def test_execution_snapshot_uses_pre_fill_visible_depth():
    raw_events = pl.DataFrame(
        [
            _raw_event(
                1,
                event_type=11,
                order_id="B1",
                side=1,
                price=100.0,
                leaves_qty=10.0,
                displayed_qty=10.0,
            ),
            _raw_event(
                2,
                event_type=1,
                order_id="A1",
                side=2,
                price=101.0,
                leaves_qty=25.0,
                displayed_qty=25.0,
            ),
            _raw_event(
                3,
                event_type=3,
                order_id="A1",
                side=2,
                price=101.0,
                leaves_qty=6.0,
                displayed_qty=6.0,
                last_shares=19.0,
            ),
        ]
    )
    review_events = [
        {
            "review_event_id": "S3",
            "sort_index": 3,
            "event_ts": "2024-01-02T09:00:03",
            "event_ts_parsed": datetime.fromisoformat("2024-01-02T09:00:03"),
            "client_id": "C1",
            "candidate_order_ids": set(),
            "matched_order_ids": set(),
        }
    ]

    _, _, queue = _review.reconstruct_review_windows(
        raw_events,
        review_events,
        top_n=10,
        pre_window_seconds=10.0,
        post_window_seconds=10.0,
        queue_snapshot_mode="key-events",
    )

    execution_level = queue.filter(
        (pl.col("review_event_id") == "S3")
        & (pl.col("snapshot_phase") == "execution")
        & (pl.col("side") == "ask")
        & (pl.col("price") == 101.0)
    )
    assert execution_level.select(pl.first("level_visible_qty")).item() == 25.0
    assert execution_level.select(pl.col("visible_qty").sum()).item() == 25.0


def test_reconstructed_review_exposes_execution_trading_capacity():
    raw_events = pl.DataFrame(
        [
            _raw_event(
                1,
                event_type=1,
                order_id="A1",
                side=2,
                price=101.0,
                leaves_qty=25.0,
                displayed_qty=25.0,
            ),
            _raw_event(
                2,
                event_type=3,
                order_id="A1",
                side=2,
                price=101.0,
                leaves_qty=6.0,
                displayed_qty=6.0,
                last_shares=19.0,
            ),
        ]
    )
    review_events = [
        {
            "review_event_id": "S2",
            "sort_index": 2,
            "event_ts": "2024-01-02T09:00:02",
            "event_ts_parsed": datetime.fromisoformat("2024-01-02T09:00:02"),
            "client_id": "C1",
            "candidate_order_ids": set(),
            "matched_order_ids": set(),
        }
    ]

    review, event_log, _ = _review.reconstruct_review_windows(
        raw_events,
        review_events,
        top_n=10,
        pre_window_seconds=10.0,
        post_window_seconds=10.0,
        queue_snapshot_mode="key-events",
    )

    assert review.select("trading_capacity_code").item() == 1
    assert review.select("trading_capacity_label").item() == "Dealing on own account"
    execution = event_log.filter(pl.col("is_execution_order"))
    assert execution.select("trading_capacity_code").item() == 1
    assert execution.select("trading_capacity_label").item() == "Dealing on own account"


def test_trading_capacity_uses_canonical_readable_label():
    assert _review._trading_capacity(
        {
            "order_trading_capacity_code": 2,
            "order_trading_capacity_label": "2 : matched_principal",
        }
    ) == (2, "Matched principal")


def test_dashboard_displays_trading_capacity_in_selector_summary_and_event_table(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()
    review_events = review_events.with_columns(
        pl.lit(1).alias("trading_capacity_code"),
        pl.lit("Dealing on own account").alias("trading_capacity_label"),
    )
    event_log = event_log.with_columns(
        pl.lit(1).alias("trading_capacity_code"),
        pl.lit("Dealing on own account").alias("trading_capacity_label"),
    )

    _review.write_dashboard(
        path,
        review_events=review_events,
        event_log=event_log,
        queue=queue,
    )

    html = path.read_text()
    assert "function capacityText" in html
    assert "function withBaseReviewContext" in html
    assert "withBaseReviewContext(parameterRuns[0].events)" in html
    assert "| trading capacity=${capacityText(ev)} |" in html
    assert "<b>trading capacity:</b> ${capacityText(ev)}" in html
    assert "<th>trading capacity</th>" in html
    assert "${capacityText(r)}" in html
    assert "<b>1</b> = Dealing on own account" in html
    assert "<b>2</b> = Matched principal" in html
    assert "<b>3</b> = Any other capacity" in html


def test_same_side_book_level_reports_bid_and_ask_rank_even_if_price_empty():
    active_orders = {
        "B1": _order("B1", client="c", qty=10.0, priority="1", first_seen=1, side="bid", price=10.0),
        "B3": _order("B3", client="c", qty=10.0, priority="2", first_seen=2, side="bid", price=9.0),
        "A1": _order("A1", client="c", qty=10.0, priority="3", first_seen=3, side="ask", price=10.5),
        "A3": _order("A3", client="c", qty=10.0, priority="4", first_seen=4, side="ask", price=11.5),
    }

    assert _review._same_side_book_level(active_orders, side="bid", price=9.5) == 2
    assert _review._same_side_book_level(active_orders, side="ask", price=11.0) == 2
    assert _review._same_side_book_level(active_orders, side="ask", price=None) is None


def test_same_side_book_position_distinguishes_resting_from_insertion_rank():
    active_orders = {
        "B1": _order("B1", client="c", qty=10.0, priority="1", first_seen=1, side="bid", price=10.0),
        "B2": _order("B2", client="c", qty=10.0, priority="2", first_seen=2, side="bid", price=9.0),
    }

    assert _review._same_side_book_position(active_orders, side="bid", price=9.0) == (2, True)
    assert _review._same_side_book_position(active_orders, side="bid", price=9.5) == (2, False)
    assert _review._same_side_book_position(active_orders, side="bid", price=None) == (None, False)


def test_dashboard_event_table_maps_full_book_rank_to_zoom_level(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()
    event_log = event_log.with_columns(
        pl.Series("book_level", [1, 43]),
        pl.Series("book_level_is_resting", [True, False]),
    )

    _review.write_dashboard(
        path,
        review_events=review_events,
        event_log=event_log,
        queue=queue,
        review_top_n=10,
    )

    html = path.read_text()
    assert "execution stage uses immediately pre-fill total depth" in html
    assert "const reviewTopN = 10" in html
    assert "function formatZoomLevel" in html
    assert "L${rank}" in html
    assert "outside top ${reviewTopN} (rank ${rank})" in html
    assert "not resting; would rank ${zoomLabel}" in html
    assert "formatZoomLevel(r.book_level, r.book_level_is_resting)" in html
    assert ">zoom level</th>" in html


def test_dashboard_includes_llm_review_panel(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()
    _review.write_dashboard(path, review_events=review_events, event_log=event_log, queue=queue, llm_reviews={})
    html = path.read_text()
    assert "LLM surveillance review" in html
    assert "llmReviews" in html
    assert "No precomputed LLM review found" in html


def test_dashboard_embeds_annotations_and_client_session_alerts(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()
    annotations = pl.DataFrame([{"review_event_id": "S10", "analyst_label": "weak_spoofing_like", "confidence": 0.7, "benign_explanation": "quote_refresh", "notes": "needs more context", "reviewer": "alice", "reviewed_at_utc": "2026-06-23T10:00:00Z"}])
    alerts = pl.DataFrame([{"client_id": "C1", "alert_score": 0.8, "event_count": 3, "mcps_at_threshold": 0.5, "recommended_action": "human_review"}])
    _review.write_dashboard(path, review_events=review_events, event_log=event_log, queue=queue, annotations=annotations, client_session_alerts=alerts)
    html = path.read_text()
    assert "Analyst annotation" in html
    assert "weak_spoofing_like" in html
    assert "Client-session alerts" in html
    assert "human_review" in html
    assert "WMSCI" in html
    assert "Price-response diagnostics" in html
    assert "favorable pre-fill mid move" in html
    assert "execution advantage vs posture mid" in html
    assert "event-row-client" in html
    assert "event-row-execution" in html
    assert "event-row-candidate" in html
    assert "event-row-matched-cancel" in html


def test_empirical_kernel_parameter_table_does_not_present_parametric_coefficients_as_active():
    html = _review._parameter_table_html(
        review_top_n=10,
        pre_window_seconds=30.0,
        post_window_seconds=35.0,
        metric_metadata={
            "top_n": 5,
            "window_seconds": 10.0,
            "withdrawal_window_seconds": 2.0,
            "max_deceptive_order_age_seconds": 90.0,
            "kernel_mode": "empirical",
            "empirical_depth_kernel": "outputs/kernel/empirical_depth_kernel.parquet",
            "kappa": 1.0,
            "lambda_": 1.0,
        },
    )

    assert "Depth-kernel mode" in html
    assert "empirical" in html
    assert "Empirical-kernel artifact" in html
    assert "outputs/kernel/empirical_depth_kernel.parquet" in html
    assert "Post-execution matched-withdrawal window" in html
    assert "<td>2 seconds</td>" in html
    assert "<td>10 seconds</td>" not in html
    assert "<td>kappa</td>" not in html
    assert "<td>lambda</td>" not in html


def test_dashboard_uses_kernel_agnostic_metric_run_selector_label(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()

    _review.write_dashboard(
        path,
        review_events=review_events,
        event_log=event_log,
        queue=queue,
        metric_metadata={"kernel_mode": "empirical"},
    )

    html = path.read_text()
    assert "Choose metric-run variant" in html
    assert "Choose kappa/lambda" not in html


def test_write_review_artifacts_saves_event_log(tmp_path):
    event_log = pl.DataFrame({"review_event_id": ["S1"], "sort_index": [1]})
    queue = pl.DataFrame({"review_event_id": ["S1"], "phase": ["pre"]})
    outputs = _review.write_review_artifacts(output_dir=tmp_path, event_log=event_log, queue=queue)
    assert outputs["event_log"].exists()
    assert outputs["queue"].exists()


def test_parse_args_supports_key_event_queue_snapshots(tmp_path):
    args = _review.parse_args(
        [
            "--input",
            str(tmp_path / "input.parquet"),
            "--execution-metrics",
            str(tmp_path / "metrics.parquet"),
            "--candidate-deceptive-orders",
            str(tmp_path / "candidates.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--queue-snapshot-mode",
            "key-events",
        ]
    )

    assert args.queue_snapshot_mode == "key-events"


def test_cluster_review_ids_and_dashboard_show_raw_child_fills_and_refresh_timestamp(tmp_path):
    cluster_id = "EC000000010-000000011"
    metrics = pl.DataFrame([{"execution_cluster_id": cluster_id, "cluster_first_sort_index": 10, "cluster_last_sort_index": 11, "child_fill_count": 2, "fill_qty": 30.0, "has_matched_deceptive_cancel_window": True}])
    review_events = _review._prepare_review_events(
        metrics,
        None,
        cluster_members=pl.DataFrame(
            [
                {"execution_cluster_id": cluster_id, "child_sort_index": 10},
                {"execution_cluster_id": cluster_id, "child_sort_index": 11},
            ]
        ),
    )
    assert review_events[0]["review_event_id"] == cluster_id
    assert review_events[0]["sort_index"] == 10
    path = tmp_path / "dashboard.html"
    events, event_log, queue = _minimal_dashboard_frames()
    _review.write_dashboard(path, review_events=events.with_columns(pl.lit(cluster_id).alias("review_event_id")), event_log=event_log.with_columns(pl.lit(cluster_id).alias("review_event_id")), queue=queue.with_columns(pl.lit(cluster_id).alias("review_event_id")), child_members=pl.DataFrame([{"execution_cluster_id": cluster_id, "child_sort_index": 10}]), dashboard_refreshed_at_utc="2026-07-16T12:00:00+00:00")
    html = path.read_text()
    assert "Raw child fills" in html
    assert "assigned vs competing cancellations" in html
    assert "dashboardRefreshedAtUtc = \"2026-07-16T12:00:00+00:00\"" in html
