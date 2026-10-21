from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

from spoofing_detection.lob.empirical_clock import CLOCK_POLICY_ID
from spoofing_detection.lob.withdrawal_risk_v2 import compute_compact_risk

from tests.lob.test_withdrawal_risk_v2 import event


def test_sequence_replay_clock_avoids_false_bookout_to_fill_regression():
    rows = [
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:00:28.579000"),
        event(2, 1, "A", 2, 101, 10, 10, "OTHER", ts="2024-01-02 09:00:28.579500"),
        event(3, 3, "A", 2, 101, 0, 0, "OTHER", ts="2024-01-02 09:00:28.580000"),
    ]
    rows[1]["BOOKOUTTIME"] = "2024-01-02 09:00:28.579789424"
    rows[2]["TRADETIME"] = "2024-01-02 09:00:28.579000000"
    rows[2]["BOOKIN"] = None
    rows[2]["BOOKOUTTIME"] = "2024-01-02 09:00:28.580100000"

    legacy = compute_compact_risk(rows_to_frame(rows), instrument="TEST")
    revised = compute_compact_risk(
        rows_to_frame(rows),
        instrument="TEST",
        clock_policy_id=CLOCK_POLICY_ID,
    )

    assert legacy.diagnostics.filter(legacy.diagnostics["reason"] == "clock_regression").height == 1
    assert revised.diagnostics.filter(revised.diagnostics["reason"] == "clock_regression").is_empty()
    assert revised.coverage_epochs.height == 1
    assert revised.coverage_epochs.item(0, "end_ts") == datetime(2024, 1, 2, 9, 0, 28, 580000)


def test_cancel_event_uses_sequence_endpoint_without_bookin_requirement():
    rows = [
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 4, "B", 1, 100, 0, 0, ts="2024-01-02 09:30:01"),
    ]
    rows[1]["BOOKIN"] = "2024-01-02 09:30:00.999100"
    rows[1]["BOOKOUTTIME"] = "2024-01-02 09:30:01.000900"
    revised = compute_compact_risk(rows_to_frame(rows), instrument="TEST", clock_policy_id=CLOCK_POLICY_ID)
    assert revised.withdrawal_events.item(0, "event_ts") == datetime(2024, 1, 2, 9, 30, 1)

    rows[1]["BOOKIN"] = None
    without_bookin = compute_compact_risk(rows_to_frame(rows), instrument="TEST", clock_policy_id=CLOCK_POLICY_ID)
    assert without_bookin.withdrawal_events.item(0, "event_ts") == datetime(2024, 1, 2, 9, 30, 1)


def test_unknown_clock_policy_fails_before_replay():
    rows = rows_to_frame([event(1, 1, "B", 1, 100, 10, 10)])
    try:
        compute_compact_risk(rows, instrument="TEST", clock_policy_id="unknown")
    except ValueError as exc:
        assert "unknown clock policy" in str(exc)
    else:
        raise AssertionError("unknown clock policy accepted")


def test_fill_anchor_retains_tradetime_separate_from_sequence_replay_clock():
    from spoofing_detection.lob.empirical_clock import resolve_empirical_clocks

    fill = event(1, 3, "A", 2, 101, 0, 0, "OTHER", ts="2024-01-02 09:00:00.580000")
    fill["BOOKIN"] = None
    fill["TRADETIME"] = "2024-01-02 09:00:00.579000"

    decision = resolve_empirical_clocks(fill)

    assert decision.fill_anchor_ns == 1_704_186_000_579_000_000
    assert decision.replay_time_ns == 1_704_186_000_580_000_000


def test_sub_microsecond_sequence_difference_contributes_positive_duration():
    rows = [
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00.000000100"),
        event(2, 4, "B", 1, 100, 0, 0, ts="2024-01-02 09:30:00.000000900"),
    ]

    revised = compute_compact_risk(
        rows_to_frame(rows), instrument="TEST", clock_policy_id=CLOCK_POLICY_ID
    )

    assert revised.intervals.item(0, "duration_seconds") == 8e-7
    assert revised.membership.item(0, "duration_seconds") == 8e-7


def test_native_nanosecond_columns_preserve_sub_microsecond_duration():
    import polars as pl

    rows = rows_to_frame([
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00.000000100"),
        event(2, 4, "B", 1, 100, 0, 0, ts="2024-01-02 09:30:00.000000900"),
    ]).with_columns(
        pl.col(name).str.to_datetime(time_unit="ns")
        for name in ("SEQUENCETIME", "BOOKIN", "BOOKOUTTIME")
    )

    revised = compute_compact_risk(rows, instrument="TEST", clock_policy_id=CLOCK_POLICY_ID)

    assert revised.intervals.item(0, "duration_seconds") == 8e-7
    assert revised.intervals.item(0, "start_ns") == 1_704_187_800_000_000_100
    assert revised.intervals.item(0, "end_ns") == 1_704_187_800_000_000_900
    assert revised.coverage_epochs.item(0, "start_ns") == 1_704_187_800_000_000_100
    assert revised.coverage_epochs.item(0, "end_ns") == 1_704_187_800_000_000_900
    assert revised.market_intervals.item(0, "start_ns") == 1_704_187_800_000_000_100
    assert revised.market_intervals.item(0, "end_ns") == 1_704_187_800_000_000_900


def test_cancel_output_uses_sequence_endpoint_when_bookin_differs():
    rows = [
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 4, "B", 1, 100, 0, 0, ts="2024-01-02 09:30:01.000000900"),
    ]
    rows[1]["BOOKIN"] = "2024-01-02 09:30:01.000000100"

    revised = compute_compact_risk(
        rows_to_frame(rows), instrument="TEST", clock_policy_id=CLOCK_POLICY_ID
    )

    assert revised.withdrawal_events.item(0, "event_ns") == 1_704_187_801_000_000_900
    assert revised.intervals.item(0, "end_ns") == 1_704_187_801_000_000_900


def test_cluster_artifact_receives_exact_fill_anchor_and_policy_identity():
    from spoofing_detection.lob.empirical_controls_v2_runner import attach_fill_clock_anchors

    fill = event(1, 3, "A", 2, 101, 0, 0, "OTHER", ts="2024-01-02 09:00:00.580000")
    fill["TRADETIME"] = "2024-01-02 09:00:00.579000123"
    clusters = rows_to_frame([{
        "execution_cluster_id": "EC-A-1",
        "cluster_last_sort_index": 11,
        "cluster_end_ts": datetime(2024, 1, 2, 9, 0, 0, 580000),
    }])

    raw = rows_to_frame([fill]).with_columns(
        pl.col("TRADETIME").str.to_datetime(time_unit="ns")
    )
    anchored = attach_fill_clock_anchors(
        clusters, raw, index_offset=10, clock_policy_id=CLOCK_POLICY_ID
    )

    assert anchored.item(0, "fill_anchor_ns") == 1_704_186_000_579_000_123
    assert anchored.item(0, "clock_policy_id") == CLOCK_POLICY_ID


def test_cluster_anchor_binding_uses_canonical_not_physical_order():
    from spoofing_detection.lob.empirical_controls_v2_runner import attach_fill_clock_anchors

    later = event(2, 3, "A", 2, 101, 0, 0, "OTHER", ts="2024-01-02 09:00:02")
    later["TRADETIME"] = "2024-01-02 09:00:02.000000222"
    earlier = event(1, 3, "A", 2, 101, 0, 0, "OTHER", ts="2024-01-02 09:00:01")
    earlier["TRADETIME"] = "2024-01-02 09:00:01.000000111"
    clusters = rows_to_frame([{
        "execution_cluster_id": "EC-A-1",
        "cluster_last_sort_index": 1,
        "cluster_end_ts": datetime(2024, 1, 2, 9, 0, 1),
    }])

    anchored = attach_fill_clock_anchors(
        clusters, rows_to_frame([later, earlier]), index_offset=0,
        clock_policy_id=CLOCK_POLICY_ID,
    )

    assert anchored.item(0, "fill_anchor_ns") == 1_704_186_001_000_000_111


def test_cluster_anchor_binding_rejects_disagreement_with_v3_detector():
    from spoofing_detection.lob.empirical_controls_v2_runner import attach_fill_clock_anchors

    fill = event(1, 3, "A", 2, 101, 0, 0, "OTHER", ts="2024-01-02 09:00:00.580000")
    fill["TRADETIME"] = "2024-01-02 09:00:00.579000123"
    raw = rows_to_frame([fill]).with_columns(pl.col("TRADETIME").str.to_datetime(time_unit="ns"))
    clusters = rows_to_frame([{
        "execution_cluster_id": "EC-A-1", "cluster_last_sort_index": 1,
        "cluster_end_fill_anchor_ns": 1_704_186_000_579_000_124,
    }])
    with pytest.raises(ValueError, match="detector fill anchor mismatch"):
        attach_fill_clock_anchors(clusters, raw, index_offset=0, clock_policy_id=CLOCK_POLICY_ID)


def rows_to_frame(rows):
    import polars as pl

    return pl.DataFrame(rows)
