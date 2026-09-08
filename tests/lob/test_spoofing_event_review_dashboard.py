from __future__ import annotations

import importlib.util
import inspect
import json
from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from spoofing_detection.lob.models import ActiveOrder


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_spoofing_event_review_dashboard.py"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "spoofing_detection_parameters.json"
_spec = importlib.util.spec_from_file_location("build_spoofing_event_review_dashboard", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_review)


def write_event_review_config(tmp_path: Path, **overrides: object) -> Path:
    payload = json.loads(CONFIG_PATH.read_text())
    payload["event_review"].update(overrides)
    config_path = tmp_path / "spoofing_parameters.json"
    config_path.write_text(json.dumps(payload))
    return config_path


def test_metric_metadata_accepts_signed_msci_provenance(tmp_path: Path):
    execution_metrics_path = tmp_path / "run" / "execution_metrics.parquet"
    execution_metrics_path.parent.mkdir()
    expected = {
        "msci_definition": _review.MSCI_DEFINITION,
        "msci_range": list(_review.MSCI_RANGE),
        "ratio_zero_denominator_policy": _review.RATIO_ZERO_DENOMINATOR_POLICY,
    }
    execution_metrics_path.parent.joinpath("metadata.json").write_text(json.dumps(expected))

    assert _review._load_metric_metadata(execution_metrics_path) == expected


def test_metric_run_parameters_preserve_semantic_provenance():
    metric_metadata = {
        "msci_definition": _review.MSCI_DEFINITION,
        "msci_range": list(_review.MSCI_RANGE),
        "ratio_zero_denominator_policy": _review.RATIO_ZERO_DENOMINATOR_POLICY,
        "execution_anchor_modes": ["passive", "aggressive"],
        "observed_execution_anchor_modes": ["aggressive"],
    }

    parameters = _review._metric_run_parameters(metric_metadata)

    assert parameters["msci_definition"] == _review.MSCI_DEFINITION
    assert parameters["msci_range"] == list(_review.MSCI_RANGE)
    assert parameters["ratio_zero_denominator_policy"] == _review.RATIO_ZERO_DENOMINATOR_POLICY
    assert parameters["execution_anchor_modes"] == ["passive", "aggressive"]
    assert parameters["observed_execution_anchor_modes"] == ["aggressive"]


def test_metric_metadata_rejects_missing_msci_provenance(tmp_path: Path):
    execution_metrics_path = tmp_path / "run" / "execution_metrics.parquet"

    with pytest.raises(ValueError, match="metric metadata missing"):
        _review._load_metric_metadata(execution_metrics_path)


def test_metric_metadata_rejects_legacy_msci_definition(tmp_path: Path):
    execution_metrics_path = tmp_path / "run" / "execution_metrics.parquet"
    execution_metrics_path.parent.mkdir()
    execution_metrics_path.parent.joinpath("metadata.json").write_text(
        json.dumps(
            {
                "msci_definition": (
                    "mean(clip(SCI / 2, 0, 1), clip(collapse_opposite_side, 0, 1), "
                    "clip(max(collapse_opposite_side - collapse_same_side, 0), 0, 1))"
                ),
                "msci_range": [0.0, 1.0],
            }
        )
    )

    with pytest.raises(ValueError, match="incompatible MSCI definition"):
        _review._load_metric_metadata(execution_metrics_path)


def test_metric_metadata_rejects_missing_ratio_zero_denominator_policy(tmp_path: Path):
    execution_metrics_path = tmp_path / "run" / "execution_metrics.parquet"
    execution_metrics_path.parent.mkdir()
    execution_metrics_path.parent.joinpath("metadata.json").write_text(
        json.dumps(
            {
                "msci_definition": _review.MSCI_DEFINITION,
                "msci_range": list(_review.MSCI_RANGE),
            }
        )
    )

    with pytest.raises(ValueError, match="incompatible ratio zero-denominator policy"):
        _review._load_metric_metadata(execution_metrics_path)


def test_parameter_review_events_reject_legacy_msci_definition(tmp_path: Path):
    run_dir = tmp_path / "kappa_1_lambda_0.5"
    run_dir.mkdir()
    run_dir.joinpath("metadata.json").write_text(
        json.dumps({"msci_definition": "obsolete_definition", "msci_range": [0.0, 1.0]})
    )
    pl.DataFrame().write_parquet(run_dir / "execution_metrics.parquet")

    with pytest.raises(ValueError, match="incompatible MSCI definition"):
        _review._load_parameter_review_events(tmp_path, max_events=None)


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


def test_actor_queue_dict_reports_actor_percent_volume_and_priority():
    level_orders = [_order("A", client="client_1", qty=30.0, priority="1", first_seen=1), _order("B", client="client_2", qty=20.0, priority="2", first_seen=2), _order("C", client="client_1", qty=50.0, priority="3", first_seen=3)]
    payload = json.loads(_review._actor_queue_dict(level_orders))
    assert payload["client_original:client_1"]["perc_vol"] == 0.8
    assert payload["client_original:client_1"]["priority"] == 1
    assert payload["client_original:client_1"]["visible_qty"] == 80.0
    assert payload["client_original:client_1"]["order_count"] == 2
    assert payload["client_original:client_2"]["perc_vol"] == 0.2
    assert payload["client_original:client_2"]["priority"] == 2


def test_parse_args_loads_event_review_parameters_only_from_config(tmp_path: Path):
    config_path = write_event_review_config(
        tmp_path,
        top_n=10,
        pre_window_seconds=30.0,
        post_window_seconds=35.0,
        queue_snapshot_mode="key-events",
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
        ]
    )

    assert args.config == config_path
    assert args.top_n == 10
    assert args.pre_window_seconds == 30.0
    assert args.post_window_seconds == 35.0
    assert args.queue_snapshot_mode == "key-events"


def test_parse_args_rejects_retired_dashboard_annotations_option(tmp_path: Path):
    with pytest.raises(SystemExit):
        _review.parse_args(
            [
                "--input",
                str(tmp_path / "input.parquet"),
                "--execution-metrics",
                str(tmp_path / "execution_metrics.parquet"),
                "--candidate-deceptive-orders",
                str(tmp_path / "candidate_deceptive_orders.parquet"),
                "--output-dir",
                str(tmp_path / "event_review"),
                "--annotations",
                str(tmp_path / "annotations.csv"),
            ]
        )


def test_dashboard_writer_has_no_annotation_or_llm_review_inputs():
    parameters = inspect.signature(_review.write_dashboard).parameters

    assert "annotations" not in parameters
    assert "llm_reviews" not in parameters


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


def test_key_event_post_snapshot_is_after_last_matched_cancel_not_first_post_event():
    raw_events = pl.DataFrame(
        [
            _raw_event(1, event_type=1, order_id="B1", side=1, price=100.0, leaves_qty=1.0, displayed_qty=1.0),
            _raw_event(2, event_type=1, order_id="C1", side=2, price=102.0, leaves_qty=14.0, displayed_qty=14.0),
            _raw_event(3, event_type=1, order_id="C2", side=2, price=103.0, leaves_qty=20.0, displayed_qty=20.0),
            _raw_event(4, event_type=3, order_id="B1", side=1, price=100.0, leaves_qty=0.0, displayed_qty=0.0, last_shares=1.0),
            _raw_event(5, event_type=1, order_id="U1", side=2, price=101.0, leaves_qty=5.0, displayed_qty=5.0),
            _raw_event(6, event_type=4, order_id="C1", side=2, price=102.0, leaves_qty=0.0, displayed_qty=0.0),
            _raw_event(7, event_type=1, order_id="U2", side=1, price=99.0, leaves_qty=5.0, displayed_qty=5.0),
            _raw_event(8, event_type=4, order_id="C2", side=2, price=103.0, leaves_qty=0.0, displayed_qty=0.0),
        ]
    )
    review_events = [
        {
            "review_event_id": "S4",
            "sort_index": 4,
            "cluster_last_sort_index": 4,
            "event_ts": "2024-01-02T09:00:04",
            "event_ts_parsed": datetime.fromisoformat("2024-01-02T09:00:04"),
            "client_id": "C1",
            "candidate_order_ids": {"C1", "C2"},
            "matched_order_ids": {"C1", "C2"},
        }
    ]

    review, _, queue = _review.reconstruct_review_windows(
        raw_events,
        review_events,
        top_n=10,
        pre_window_seconds=10.0,
        post_window_seconds=10.0,
        queue_snapshot_mode="key-events",
    )

    snapshots = queue.select("snapshot_phase", "snapshot_sort_index").unique().sort(
        "snapshot_sort_index"
    )
    assert snapshots.rows() == [("pre", 3), ("execution", 4), ("post", 8)]
    assert review.select("post_cancel_sort_index").item() == 8
    post_order_ids = queue.filter(pl.col("snapshot_phase") == "post").get_column("ORDERID")
    assert not post_order_ids.is_in(["C1", "C2"]).any()


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
    assert execution.select("actor_key").item() == "client_original:C1"
    assert execution.select("actor_id").item() == "C1"
    assert execution.select("identity_level").item() == "client_original"
    assert execution.select("identity_source").item() == "NMSC_ORIGINALCLIENTIDSHORTCODE"
    assert execution.select("identity_fallback_flag").item() is False
    assert execution.select("execution_anchor_mode").item() == "passive"


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
    assert "| capacità di negoziazione=${capacityPlainText(ev)} |" in html
    assert "<b>trading capacity:</b> ${escapeHtml(capacityText(ev))}" in html
    assert "<th>Capacità di negoziazione</th>" in html
    assert "${escapeHtml(capacityPlainText(r))}" in html
    assert "Dealing on own account" in html
    assert "Matched principal" in html
    assert "Any other capacity" in html
    assert "contrasto con segno" in html
    assert "riduzione sullo stesso lato prevale" in html
    assert "0–1 arithmetic mean" not in html


def test_dashboard_marks_small_executions_without_inflating_their_volume_scale(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()
    event_log = event_log.with_columns(
        pl.Series("side", ["bid", "ask"]),
        pl.Series("price", [10.0, 11.0]),
        pl.Series("last_shares", [1.0, None]),
    )
    queue = queue.with_columns(pl.lit(780.0).alias("level_visible_qty"))

    _review.write_dashboard(
        path,
        review_events=review_events,
        event_log=event_log,
        queue=queue,
    )

    html = path.read_text()
    assert "type:'scatter', mode:'markers+text'" in html
    assert "symbol:'diamond'" in html
    assert "cliponaxis:false" in html
    assert "eseguito ${formatQuantity(value)}" in html
    assert "sortIndex = Number(ev.post_cancel_sort_index)" in html
    assert "sortIndex = Number(ev.cluster_first_sort_index)" in html
    assert "Number(r.sort_index) === Number(ev.cluster_first_sort_index)" in html
    assert "Number(ev.sort_index)" not in html
    assert "Matching-engine sort order defines the three stages" in html
    assert "quantità totale, quantità del soggetto candidato e quantità eseguita" in html


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
    assert "oltre i primi ${reviewTopN} livelli (posizione ${rank})" in html
    assert "non presente nel book; si collocherebbe a ${zoomLabel}" in html
    assert "formatZoomLevel(r.book_level, r.book_level_is_resting)" in html
    assert ">Livello nel book</th>" in html


def test_dashboard_omits_analyst_annotation_and_llm_review_cards(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()
    _review.write_dashboard(
        path,
        review_events=review_events,
        event_log=event_log,
        queue=queue,
    )
    html = path.read_text()
    assert "Analyst annotation" not in html
    assert "weak_spoofing_like" not in html
    assert "LLM surveillance review" not in html
    assert "LLM review text" not in html
    assert 'id="annotation"' not in html
    assert 'id="llmReview"' not in html


def test_dashboard_embeds_session_alerts_without_annotations(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()
    alerts = pl.DataFrame(
        [
            {
                "client_id": "C1",
                "max_withdrawal_profile_scale_event": 8.0,
                "mean_withdrawal_profile_scale_event": 4.0,
                "max_MSCI_resting_profile": 0.9,
                "mean_MSCI_resting_profile": 0.4,
                "max_WMSCI_event": -99.0,
                "mean_WMSCI_event": -99.0,
                "max_MSCI": -99.0,
                "mean_MSCI": -99.0,
                "matched_event_count": 3,
                "event_count": 4,
                "recommended_action": "human_review",
            }
        ]
    )
    _review.write_dashboard(
        path,
        review_events=review_events,
        event_log=event_log,
        queue=queue,
        client_session_alerts=alerts,
    )
    html = path.read_text()
    assert "Analyst annotation" not in html
    assert "weak_spoofing_like" not in html
    assert "Actor-session alerts" in html
    assert "human_review" in html
    assert "Max withdrawal profile scale" in html
    assert "Mean withdrawal profile scale" in html
    assert "Max MSCI resting profile" in html
    assert "Mean MSCI resting profile" in html
    assert "ev.withdrawal_profile_scale_event ?? ev.WMSCI_event" in html
    assert "ev.MSCI_resting_profile ?? ev.MSCI" in html
    assert "row.max_MSCI_resting_profile ?? row.max_MSCI" in html
    assert "Alert score" not in html
    assert "alert_score" not in html
    assert "Price-response diagnostics" in html
    assert "favorable pre-fill mid move" in html
    assert "execution advantage vs posture mid" in html
    assert "event-row-actor" in html
    assert "event-row-execution" in html
    assert "event-row-candidate" in html
    assert "event-row-matched-cancel" in html


def test_prepare_review_events_ranks_by_canonical_withdrawal_scale():
    metrics = pl.DataFrame(
        {
            "sort_index": [1, 2, 3],
            "has_matched_deceptive_cancel_window": [True, True, True],
            "WMSCI_event": [1.0, 10.0, 5.0],
            "MSCI_resting_profile": [1.8, 0.2, 0.9],
            "MSCI": [-99.0, -99.0, -99.0],
            "execution_anchor_mode": ["passive", "passive", "passive"],
        }
    )

    review_events = _review._prepare_review_events(metrics, max_events=None)

    assert [row["review_event_id"] for row in review_events] == ["S2", "S3", "S1"]


def test_prepare_review_events_rejects_missing_wmsci_ranking_metric():
    metrics = pl.DataFrame(
        {
            "sort_index": [1],
            "has_matched_deceptive_cancel_window": [True],
            "MSCI_resting_profile": [1.8],
            "execution_anchor_mode": ["passive"],
        }
    )

    with pytest.raises(ValueError, match="withdrawal_profile_scale_event or WMSCI_event"):
        _review._prepare_review_events(metrics, max_events=None)


def test_dashboard_presents_wmsci_as_primary_event_metric(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()
    review_events = review_events.with_columns(
        pl.lit(3.25).alias("withdrawal_profile_scale_event"),
        pl.lit(0.4).alias("MSCI_resting_profile"),
    )
    _review.write_dashboard(
        path,
        review_events=review_events,
        event_log=event_log,
        queue=queue,
    )

    html = path.read_text()

    assert "WMSCI — intensità del ritiro attribuito" in html
    assert "ordinati per WMSCI decrescente" in html
    assert "MSCI del profilo a riposo" in html


def test_dashboard_opens_with_one_plain_language_wmsci_msci_guide_and_example(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()

    _review.write_dashboard(
        path,
        review_events=review_events,
        event_log=event_log,
        queue=queue,
    )

    html = path.read_text()

    assert html.index("Come leggere WMSCI e MSCI") < html.index("Cosa contiene questa dashboard")
    assert "WMSCI misura l’intensità del ritiro attribuito" in html
    assert "MSCI descrive come cambia la forma relativa del book" in html
    assert "WMSCI = 3,2" in html
    assert "MSCI = 0,8" in html
    assert "non significa “3,2 volte più sospetto”" in html
    assert "metrica principale di ordinamento" not in html
    assert "Il nome canonico nello schema v2 è" not in html


def test_review_population_summary_distinguishes_all_clusters_candidates_and_complete_sequences():
    execution_metrics = pl.DataFrame(
        {
            "has_matched_deceptive_cancel_window": [True, True, False, False],
            "spoofing_compatible_sequence": [True, False, False, False],
        }
    )
    visible_review_events = pl.DataFrame(
        {
            "has_matched_deceptive_cancel_window": [True],
            "spoofing_compatible_sequence": [True],
        }
    )

    assert _review._review_population_summary(
        execution_metrics,
        visible_review_events,
    ) == {
        "reconstructed_clusters": 4,
        "review_candidates": 2,
        "compatible_sequences": 1,
        "displayed_candidates": 1,
        "displayed_compatible_sequences": 1,
    }


def test_reconstructed_review_summary_preserves_each_behavioral_control():
    raw_event = _raw_event(
        1,
        event_type=3,
        order_id="EXEC-1",
        side=1,
        price=100.0,
        leaves_qty=0.0,
        displayed_qty=0.0,
        last_shares=5.0,
    )
    review_events = [
        {
            "review_event_id": "S1",
            "sort_index": 1,
            "event_ts": "2024-01-02T09:00:01",
            "event_ts_parsed": datetime.fromisoformat("2024-01-02T09:00:01"),
            "client_id": "C1",
            "execution_anchor_mode": "passive",
            "candidate_order_ids": {"CANCEL-1"},
            "matched_order_ids": {"CANCEL-1"},
            "child_fill_sort_indexes": {1},
            "has_matched_deceptive_cancel_window": True,
            "gate_rapid_matched_withdrawal": True,
            "gate_small_fill_relative_to_withdrawal": False,
            "gate_favorable_pre_fill_move": True,
            "gate_cancel_anchored_reversion": False,
            "spoofing_compatible_sequence": False,
        }
    ]

    summary, _, _ = _review.reconstruct_review_windows(
        pl.DataFrame([raw_event]),
        review_events,
        top_n=10,
        pre_window_seconds=10.0,
        post_window_seconds=10.0,
        queue_snapshot_mode="key-events",
    )

    assert summary.select(
        "has_matched_deceptive_cancel_window",
        "gate_rapid_matched_withdrawal",
        "gate_small_fill_relative_to_withdrawal",
        "gate_favorable_pre_fill_move",
        "gate_cancel_anchored_reversion",
        "spoofing_compatible_sequence",
    ).row(0) == (True, True, False, True, False, False)


def test_prepare_review_events_accepts_official_empty_execution_schema():
    from spoofing_detection.lob.spoofing_metrics import EXECUTION_EMPTY_SCHEMA

    assert _review._prepare_review_events(
        pl.DataFrame(schema=EXECUTION_EMPTY_SCHEMA),
        max_events=None,
    ) == []


def test_prepare_review_events_rejects_missing_observed_anchor():
    metrics = pl.DataFrame(
        [{"sort_index": 1, "has_matched_deceptive_cancel_window": True}]
    )

    with pytest.raises(ValueError, match="execution_anchor_mode"):
        _review._prepare_review_events(metrics, max_events=None)


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
            "ratio_zero_denominator_policy": _review.RATIO_ZERO_DENOMINATOR_POLICY,
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
    assert "Zero-denominator policy" in html
    assert _review.RATIO_ZERO_DENOMINATOR_POLICY in html
    assert "<td>epsilon</td>" not in html


def test_dashboard_parameter_table_wraps_long_empirical_kernel_paths(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()

    _review.write_dashboard(
        path,
        review_events=review_events,
        event_log=event_log,
        queue=queue,
        metric_metadata={
            "kernel_mode": "empirical",
            "empirical_depth_kernel": "/very/long/unbroken/provenance/path/to/empirical_depth_kernel.parquet",
        },
    )

    html = path.read_text()
    assert "table-layout: fixed" in html
    assert ".parameter-table td { overflow-wrap: anywhere; vertical-align: top; }" in html
    assert ".parameter-table td:nth-child(2) { width: 36%; }" in html
    assert "white-space: nowrap" not in html


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


def test_dashboard_explains_population_evidence_levels_and_four_controls_in_plain_italian(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()
    review_events = review_events.with_columns(
        pl.lit(True).alias("has_matched_deceptive_cancel_window"),
        pl.lit(True).alias("gate_rapid_matched_withdrawal"),
        pl.lit(False).alias("gate_small_fill_relative_to_withdrawal"),
        pl.lit(True).alias("gate_favorable_pre_fill_move"),
        pl.lit(False).alias("gate_cancel_anchored_reversion"),
        pl.lit(False).alias("spoofing_compatible_sequence"),
    )

    _review.write_dashboard(
        path,
        review_events=review_events,
        event_log=event_log,
        queue=queue,
        population_summary={
            "reconstructed_clusters": 250,
            "review_candidates": 80,
            "compatible_sequences": 12,
            "displayed_candidates": 1,
            "displayed_compatible_sequences": 0,
        },
    )

    html = path.read_text()
    assert "Cosa contiene questa dashboard" in html
    assert "Cluster ricostruiti" in html
    assert "Candidati da revisionare" in html
    assert "Sequenze complete" in html
    assert "Candidati mostrati in questo file" in html
    assert "Ritiro rapido attribuito allo stesso soggetto" in html
    assert "Quantità eseguita inferiore alla quantità ritirata" in html
    assert "Movimento del prezzo favorevole prima dell’esecuzione" in html
    assert "Inversione del prezzo dopo il ritiro" in html
    assert "Perché questo evento è presente" in html
    assert "Sì" in html
    assert "No" in html
    assert "Non disponibile" in html
    assert 'id="evidenceFilter"' in html
    assert "Solo sequenze complete" in html
    assert "Solo candidati da approfondire" in html
    assert "non probabilità né prove di intento" in html
    assert "non coincide con la valutazione dei quattro controlli del singolo evento" in html


def test_dashboard_preserves_missing_metrics_and_escapes_raw_event_text(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()
    _review.write_dashboard(
        path,
        review_events=review_events.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("WMSCI_event"),
            pl.lit(None, dtype=pl.Float64).alias("MSCI"),
            pl.lit(None, dtype=pl.Float64).alias("SCI"),
        ),
        event_log=event_log,
        queue=queue,
    )

    html = path.read_text()
    assert "Number(ev.WMSCI_event || 0)" not in html
    assert "metricText(withdrawalProfileScaleValue(ev), 6)" in html
    assert "finiteNumber(ev.withdrawal_profile_scale_event ?? ev.WMSCI_event)" in html
    assert "escapeHtml(actorText(ev))" in html
    assert "escapeHtml(r.ORDERID ?? '')" in html


def test_dashboard_escapes_script_json_and_parameter_table_values(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events, event_log, queue = _minimal_dashboard_frames()
    payload = "</script><script>window.__review_xss=1</script>"
    event_log = event_log.with_columns(pl.lit(payload).alias("ORDERID"))

    _review.write_dashboard(
        path,
        review_events=review_events,
        event_log=event_log,
        queue=queue,
        metric_metadata={
            "kernel_mode": "empirical",
            "empirical_depth_kernel": payload,
        },
    )

    html = path.read_text()
    assert payload not in html
    assert "\\u003c/script\\u003e\\u003cscript\\u003ewindow.__review_xss=1" in html
    assert "&lt;/script&gt;&lt;script&gt;window.__review_xss=1&lt;/script&gt;" in html


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
        ]
    )

    assert args.queue_snapshot_mode == "key-events"


def test_cluster_review_ids_and_dashboard_show_raw_child_fills_and_refresh_timestamp(tmp_path):
    cluster_id = "EC000000010-000000011"
    metrics = pl.DataFrame([{"execution_cluster_id": cluster_id, "cluster_first_sort_index": 10, "cluster_last_sort_index": 11, "child_fill_count": 2, "fill_qty": 30.0, "withdrawal_profile_scale_event": 1.0, "has_matched_deceptive_cancel_window": True, "execution_anchor_mode": "passive"}])
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
    _review.write_dashboard(
        path,
        review_events=events.with_columns(pl.lit(cluster_id).alias("review_event_id")),
        event_log=event_log.with_columns(pl.lit(cluster_id).alias("review_event_id")),
        queue=queue.with_columns(pl.lit(cluster_id).alias("review_event_id")),
        child_members=pl.DataFrame(
            [
                {
                    "execution_cluster_id": cluster_id,
                    "child_sort_index": 10,
                    "child_order_id": "CHILD-1",
                    "child_event_id": 75_590_479_516_795_608,
                }
            ]
        ),
        cancel_candidates=pl.DataFrame(
            [
                {
                    "execution_cluster_id": cluster_id,
                    "cancel_sort_index": 12,
                    "candidate_order_id": "CANCEL-1",
                    "assigned_flag": True,
                }
            ]
        ),
        dashboard_refreshed_at_utc="2026-07-16T12:00:00+00:00",
    )
    html = path.read_text()
    assert "Raw child fills" in html
    assert "assigned vs competing cancellations" in html
    assert "function byCluster(id, rows)" in html
    assert "r.execution_cluster_id === id" in html
    assert "function renderChildFills(ev)" in html
    assert "function renderCancelCandidates(ev)" in html
    assert "renderChildFills(ev); renderCancelCandidates(ev);" in html
    assert "CHILD-1" in html
    assert "CANCEL-1" in html
    assert '\"child_event_id\": \"75590479516795608\"' in html
    assert "dashboardRefreshedAtUtc = \"2026-07-16T12:00:00+00:00\"" in html


def test_reconstructed_firm_fallback_review_uses_actor_key_not_client_label():
    raw_rows = [
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
    for row in raw_rows:
        row["NMSC_ORIGINALCLIENTIDSHORTCODE"] = None
    review_events = [
        {
            "review_event_id": "S2",
            "sort_index": 2,
            "event_ts": "2024-01-02T09:00:02",
            "event_ts_parsed": datetime.fromisoformat("2024-01-02T09:00:02"),
            "actor_key": "firm:F1",
            "actor_id": "F1",
            "identity_level": "firm",
            "identity_source": "FIRMID",
            "identity_fallback_flag": True,
            "execution_anchor_mode": "aggressive",
            "candidate_order_ids": set(),
            "matched_order_ids": set(),
        }
    ]

    review, event_log, queue = _review.reconstruct_review_windows(
        pl.DataFrame(raw_rows),
        review_events,
        top_n=10,
        pre_window_seconds=10.0,
        post_window_seconds=10.0,
        queue_snapshot_mode="key-events",
    )

    assert review.select("actor_key").item() == "firm:F1"
    assert review.select("execution_anchor_mode").item() == "aggressive"
    execution = event_log.filter(pl.col("is_execution_order"))
    assert execution.select("is_review_actor").item() is True
    assert execution.select("is_review_client").item() is False
    assert execution.select("event_firm_id").item() == "F1"
    assert "client_id" not in event_log.columns
    assert "firm_id" not in event_log.columns
    assert "actor_queue_dict" in queue.columns
    assert "client_queue_dict" not in queue.columns
    assert queue.filter(pl.col("snapshot_phase") == "execution").select(
        pl.col("is_review_actor").any()
    ).item() is True


def test_dashboard_labels_actor_identity_anchor_and_raw_provenance(tmp_path):
    path = tmp_path / "dashboard.html"
    review_events = pl.DataFrame(
        [
            {
                "review_event_id": "EC1",
                "event_ts": "2024-01-02T09:00:02",
                "sort_index": 2,
                "actor_key": "firm:F1",
                "actor_id": "F1",
                "identity_level": "firm",
                "identity_source": "FIRMID",
                "identity_fallback_flag": True,
                "execution_anchor_mode": "aggressive",
                "WMSCI_event": 2.0,
                "MSCI": 0.4,
                "matched_deceptive_cancel_order_ids_window": "A1",
            }
        ]
    )
    event_log = pl.DataFrame(
        [
            {
                "review_event_id": "EC1",
                "sort_index": 2,
                "event_ts": "2024-01-02T09:00:02",
                "event_class": "fill",
                "client_id": None,
                "firm_id": "F1",
                "is_review_actor": True,
                "is_review_client": False,
                "is_execution_order": True,
                "is_candidate_deceptive_order": False,
                "is_matched_deceptive_cancel_order": False,
            }
        ]
    )
    queue = pl.DataFrame(
        [
            {
                "review_event_id": "EC1",
                "snapshot_phase": "execution",
                "snapshot_sort_index": 2,
                "side": "ask",
                "level": 1,
                "price": 101.0,
                "level_visible_qty": 25.0,
                "visible_qty": 25.0,
                "is_candidate_deceptive_order": False,
                "is_matched_deceptive_cancel_order": False,
                "client_queue_dict": "{}",
            }
        ]
    )
    alerts = pl.DataFrame(
        [
            {
                "actor_key": "firm:F1",
                "actor_id": "F1",
                "identity_level": "firm",
                "identity_fallback_flag": True,
                "identity_scope_warning": "firm_fallback_may_aggregate_multiple_clients",
                "execution_anchor_mode": "aggressive",
                "max_WMSCI_event": 2.0,
                "mean_WMSCI_event": 1.0,
                "max_MSCI": 0.4,
                "mean_MSCI": 0.2,
                "matched_event_count": 1,
                "event_count": 1,
                "recommended_action": "human_review",
            }
        ]
    )

    _review.write_dashboard(
        path,
        review_events=review_events,
        event_log=event_log,
        queue=queue,
        actor_session_alerts=alerts,
    )

    html = path.read_text()
    assert "Actor-session alerts" in html
    assert "<th>Soggetto<br><small>Actor</small></th>" in html
    assert "<th>Identificazione usata<br><small>Identity level</small></th>" in html
    assert "<th>Tipo di esecuzione<br><small>Execution anchor</small></th>" in html
    assert "firm_fallback_may_aggregate_multiple_clients" in html
    assert "<b>actor:</b> ${escapeHtml(actorText(ev))}" in html
    assert "<b>identity level:</b> ${escapeHtml(identityLevelText(ev))}" in html
    assert "<b>execution anchor:</b> ${escapeHtml(executionAnchorText(ev))}" in html
    assert "<th>Cliente originale</th><th>Firm originale</th>" in html
    assert "soggetto in revisione" in html
    assert "<b>client:</b>" not in html
