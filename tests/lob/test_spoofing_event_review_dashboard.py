from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl

from spoofing_detection.lob.models import ActiveOrder


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_spoofing_event_review_dashboard.py"
_spec = importlib.util.spec_from_file_location("build_spoofing_event_review_dashboard", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
_review = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_review)


def _order(order_id: str, *, client: str | None, qty: float, priority: str, first_seen: int) -> ActiveOrder:
    return ActiveOrder(
        order_id=order_id,
        side="bid",
        price=10.0,
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
    review_events = pl.DataFrame([{"review_event_id": "S10", "event_ts": "2024", "client_id": "C1", "MSCI": 0.1, "sort_index": 10, "matched_deceptive_cancel_order_ids_window": "A"}])
    event_log = pl.DataFrame([{"review_event_id": "S10", "sort_index": 10, "event_ts": "2024", "event_class": "fill"}])
    queue = pl.DataFrame([{"review_event_id": "S10", "snapshot_phase": "execution", "snapshot_sort_index": 10, "side": "bid", "level": 1, "price": 10.0, "level_visible_qty": 100.0, "visible_qty": 100.0, "is_candidate_deceptive_order": True, "is_matched_deceptive_cancel_order": False, "client_queue_dict": "{}"}])
    return review_events, event_log, queue


def test_client_queue_dict_reports_client_percent_volume_and_priority():
    level_orders = [_order("A", client="client_1", qty=30.0, priority="1", first_seen=1), _order("B", client="client_2", qty=20.0, priority="2", first_seen=2), _order("C", client="client_1", qty=50.0, priority="3", first_seen=3)]
    payload = json.loads(_review._client_queue_dict(level_orders))
    assert payload["client_1"]["perc_vol"] == 0.8
    assert payload["client_1"]["priority"] == 1
    assert payload["client_1"]["visible_qty"] == 80.0
    assert payload["client_1"]["order_count"] == 2
    assert payload["client_2"]["perc_vol"] == 0.2
    assert payload["client_2"]["priority"] == 2


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


def test_write_review_artifacts_saves_event_log(tmp_path):
    event_log = pl.DataFrame({"review_event_id": ["S1"], "sort_index": [1]})
    queue = pl.DataFrame({"review_event_id": ["S1"], "phase": ["pre"]})
    outputs = _review.write_review_artifacts(output_dir=tmp_path, event_log=event_log, queue=queue)
    assert outputs["event_log"].exists()
    assert outputs["queue"].exists()
