from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest
from polars.testing import assert_frame_equal


def event(seq, kind, order_id, side, price, leaves, displayed, client="C1", *, ts=None, symbol=1, day="2024-01-02", order_type=2, aggressive="N"):
    timestamp = ts if ts is not None else f"{day} 09:30:{seq:02d}"
    return {
        "TRADEDATE": day, "MIC": "XMIL", "MARKETCODE": "MTA", "SYMBOLINDEX": symbol, "EMM (*)": 1,
        "SEQUENCETIME": timestamp, "BOOKIN": timestamp, "BOOKOUTTIME": timestamp,
        "TRADETIME": timestamp if kind == 3 else None,
        "HDR_APPLKEYSEQUENCENUMBER": seq, "HDR_HWMSEQUENCENUMBER": seq, "HDR_OFFSETID": seq,
        "ROW_NUMBER": seq, "EVENTID": f"E{seq}-{symbol}", "ORDEREVENTTYPE (*)": kind,
        "ORDERID": order_id, "ORDERPRIORITY": str(seq), "ORDERSIDE (*)": side,
        "ORDERPX": price, "ORDERQTY": leaves, "DISPLAYEDQTY": displayed, "LEAVESQTY": leaves,
        "LASTSHARES": None, "LASTTRADEDPX": None, "ORDERTYPE (*)": order_type, "TIMEINFORCE (*)": 0,
        "PASSIVEORDER": "Y" if kind == 3 else "N", "AGGRESSIVEORDER": aggressive,
        "FIRMID": "F1", "NMSC_ORIGINALCLIENTIDSHORTCODE": client,
        "ORDER_TRADINGCAPACITY (*)": 3 if client is not None else 1,
    }


def compute(rows, **kwargs):
    from spoofing_detection.lob.withdrawal_risk_v2 import compute_compact_risk

    return compute_compact_risk(pl.DataFrame(rows), instrument="TEST", top_n=kwargs.pop("top_n", 1), **kwargs)


def test_no_fill_actor_has_time_at_risk_and_compact_schema():
    out = compute([
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 101, 10, 10, "OTHER", ts="2024-01-02 09:30:03"),
    ])
    row = out.intervals.filter(pl.col("actor_key") == "client_original:C1").row(0, named=True)
    assert row["duration_seconds"] == pytest.approx(3.0)
    assert set(("instrument", "coverage_epoch_id", "eligible_visible_qty", "oldest_member_age_seconds")) <= set(out.intervals.columns)
    assert out.withdrawal_events.is_empty()


def test_top_n_exit_reentry_preserves_old_lifecycle_origin():
    out = compute([
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 1, "H", 1, 101, 10, 10, "OTHER", ts="2024-01-02 09:30:01"),
        event(3, 4, "H", 1, 101, 0, 0, "OTHER", ts="2024-01-02 09:30:02"),
        event(4, 4, "B", 1, 100, 0, 0, ts="2024-01-02 09:30:04"),
    ])
    rows = out.membership.filter(pl.col("order_id") == "B").sort("eligible_start_ts").to_dicts()
    assert [row["first_seen_sort_index"] for row in rows] == [1, 1]
    assert [row["duration_seconds"] for row in rows] == [1.0, 2.0]
    assert out.withdrawal_events.filter(pl.col("order_id") == "B").item(0, "physical_order_key") == rows[0]["physical_order_key"]


def test_cancel_at_exact_age_expiry_is_eligible_point_without_extra_time():
    out = compute([
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 4, "B", 1, 100, 0, 0, ts="2024-01-02 09:30:05"),
    ], max_order_age_seconds=5)
    assert out.withdrawal_events.height == 1
    assert out.withdrawal_events.item(0, "visible_qty_removed") == 10
    assert out.intervals.item(0, "duration_seconds") == pytest.approx(5.0)
    assert out.transitions.item(0, "transition_reason") == "age_expiry"


def test_equal_clock_add_cancel_keeps_point_without_manufacturing_duration():
    out = compute([
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 4, "B", 1, 100, 0, 0, ts="2024-01-02 09:30:00"),
    ])
    assert out.intervals.is_empty()
    assert out.membership.is_empty()
    assert out.withdrawal_events.height == 1


@pytest.mark.parametrize('cancel_ts,expected', [
    ('2024-01-02 09:31:30', 1), ('2024-01-02 09:31:30.000001', 0),
])
def test_point_age_boundary_survives_an_earlier_same_timestamp_expiry(cancel_ts, expected):
    out = compute([
        event(1, 1, 'B', 1, 100, 10, 10, ts='2024-01-02 09:30:00'),
        event(2, 1, 'A', 2, 101, 10, 10, 'OTHER', ts='2024-01-02 09:31:30'),
        event(3, 4, 'B', 1, 100, 0, 0, ts=cancel_ts),
    ], max_order_age_seconds=90)
    assert out.withdrawal_events.height == expected
    own = out.intervals.filter(pl.col('actor_key') == 'client_original:C1')
    assert own['duration_seconds'].sum() == pytest.approx(90)


def test_order_id_reuse_creates_distinct_physical_lifecycles():
    out = compute([
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 4, "B", 1, 100, 0, 0, ts="2024-01-02 09:30:02"),
        event(3, 1, "B", 1, 100, 7, 7, ts="2024-01-02 09:30:03"),
        event(4, 4, "B", 1, 100, 0, 0, ts="2024-01-02 09:30:05"),
    ])
    assert out.withdrawal_events.get_column("physical_order_key").n_unique() == 2
    assert out.withdrawal_events.get_column("visible_qty_removed").to_list() == [10.0, 7.0]


def test_inactive_stop_and_pending_canonical_residual_do_not_enter_eligibility():
    out = compute([
        event(1, 1, "STOP", 1, 100, 10, 10, "STOP", ts="2024-01-02 09:30:00", order_type=4),
        event(2, 1, "A", 2, 101, 10, 10, "OTHER", ts="2024-01-02 09:30:01"),
    ])
    assert out.intervals.is_empty()
    assert out.membership.is_empty()


def test_pending_marketable_residual_never_enters_compact_eligibility():
    out = compute([
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 101, 10, 10, "OTHER", ts="2024-01-02 09:30:01"),
        event(3, 3, "RESIDUAL", 1, 101, 4, 4, "RESIDUAL", ts="2024-01-02 09:30:02", aggressive="Y"),
        event(4, 1, "LOW", 1, 99, 1, 1, "OTHER", ts="2024-01-02 09:30:03"),
    ])
    assert out.intervals.filter(pl.col("actor_key") == "client_original:RESIDUAL").is_empty()
    assert out.membership.filter(pl.col("actor_key") == "client_original:RESIDUAL").is_empty()


def test_missing_and_regressive_clock_censor_and_partition_never_bridges():
    rows = [
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 101, 10, 10, "OTHER", ts="2024-01-02 09:30:02"),
        event(3, 2, "B", 1, 100, 8, 8, ts="bad-clock"),
        event(4, 1, "X", 1, 200, 10, 10, "C2", ts="2024-01-02 09:30:01"),
        event(5, 1, "Y", 2, 201, 10, 10, "OTHER", ts="2024-01-02 09:30:04", symbol=2),
    ]
    # Canonical sorting uses sequence time; timestamp selection prefers book-out.
    # This event is canonically later but has a regressive observer clock.
    rows[3]["SEQUENCETIME"] = rows[3]["BOOKIN"] = "2024-01-02 09:30:03"
    rows[3]["BOOKOUTTIME"] = "2024-01-02 09:30:01"
    out = compute(rows)
    assert set(out.diagnostics.get_column("reason")) >= {"missing_event_timestamp", "clock_regression"}
    assert out.intervals.filter(pl.col("partition_id").str.ends_with("|1|1")).get_column("end_ts").max() == datetime(2024, 1, 2, 9, 30, 2)
    assert out.intervals.filter(pl.col("partition_id").str.ends_with("|2|1")).is_empty()


def test_expiry_heap_schedules_each_lifecycle_once_not_each_market_event(monkeypatch):
    import spoofing_detection.lob.withdrawal_risk_v2 as module
    original = module.heapq.heappush
    pushed = []
    def record(heap, value):
        pushed.append(value)
        return original(heap, value)
    monkeypatch.setattr(module.heapq, 'heappush', record)
    compute([event(seq, 1, f'A{seq}', 2, 101, 10, 10,
                   ts='2024-01-02 09:30:00') for seq in range(1, 31)])
    assert len(pushed) == len(set(pushed)) == 30


def test_sink_buffer_one_matches_in_memory_and_returns_empty_frames():
    rows = [
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 101, 10, 10, "OTHER", ts="2024-01-02 09:30:01"),
        event(3, 4, "B", 1, 100, 0, 0, ts="2024-01-02 09:30:03"),
    ]
    baseline = compute(rows)
    collected: dict[str, list[dict]] = {}
    streamed = compute(rows, sink=lambda name, chunk: collected.setdefault(name, []).extend(chunk), buffer_rows=1)
    for name in ("intervals", "membership", "withdrawal_events", "transitions", "coverage_epochs", "diagnostics", "market_intervals"):
        assert getattr(streamed, name).is_empty()
        expected = getattr(baseline, name).to_dicts()
        assert collected.get(name, []) == expected
    assert streamed.counters["intervals"] == baseline.intervals.height


def test_original_index_map_is_used_after_canonical_sorting():
    out = compute([
        event(20, 4, "B", 1, 100, 0, 0, ts="2024-01-02 09:30:02"),
        event(10, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
    ], original_sort_indices={1: 11, 2: 22})
    withdrawal = out.withdrawal_events.row(0, named=True)
    assert withdrawal["sort_index"] == 22
    assert withdrawal["first_seen_sort_index"] == 11
    assert withdrawal["physical_order_key"].endswith("|11")


def test_supplied_original_index_map_must_cover_every_canonical_event():
    with pytest.raises(ValueError, match="missing local sort_index 2"):
        compute([
            event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
            event(2, 4, "B", 1, 100, 0, 0, ts="2024-01-02 09:30:02"),
        ], original_sort_indices={1: 11})


def test_original_index_map_rejects_unmapped_nonemitting_event():
    with pytest.raises(ValueError, match="missing local sort_index 2"):
        compute([
            event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
            event(2, 1, "STOP", 1, 99, 10, 10, ts="2024-01-02 09:30:01", order_type=4),
        ], original_sort_indices={1: 11})


def test_v2_uses_callback_only_replay_and_never_mutates_live_orders(monkeypatch):
    import spoofing_detection.lob.withdrawal_risk_v2 as module
    from spoofing_detection.lob.panel import replay_events as canonical

    seen = {}
    def wrapped(raw_events, **kwargs):
        seen.update(kwargs)
        return canonical(raw_events, **kwargs)
    monkeypatch.setattr(module, "replay_events", wrapped)
    out = module.compute_compact_risk(pl.DataFrame([event(1, 1, "B", 1, 100, 1, 1)]), instrument="TEST")
    assert seen["retain_events"] is False
    assert seen["hooks"].on_pre_event is not None
    assert out.intervals.is_empty()


def test_market_intervals_use_standard_schema_and_expiry_keeps_prior_book_snapshot():
    from spoofing_detection.lob.withdrawal_risk_v2 import SCHEMAS

    out = compute([
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 1, "A", 2, 101, 10, 10, "OTHER", ts="2024-01-02 09:31:01"),
    ])
    assert list(SCHEMAS["market_intervals"]) == [
        "instrument", "partition_id", "event_date", "coverage_epoch_id", "start_ts", "end_ts",
        "spread", "depth", "prior_event_count_60s",
    ]
    rows = out.market_intervals.sort("start_ts").to_dicts()
    first = rows[0]
    assert first["end_ts"] == datetime(2024, 1, 2, 9, 31)
    assert first["depth"] == pytest.approx(10.0)
    assert first["prior_event_count_60s"] is None
    assert rows[1]["start_ts"] == datetime(2024, 1, 2, 9, 31)
    assert rows[1]["depth"] == pytest.approx(10.0)


def test_withdrawal_carries_its_closed_coverage_epoch_at_terminal_boundary():
    from spoofing_detection.lob.withdrawal_risk_v2 import SCHEMAS

    out = compute([
        event(1, 1, "B", 1, 100, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 4, "B", 1, 100, 0, 0, ts="2024-01-02 09:30:05"),
    ])
    withdrawal = out.withdrawal_events.row(0, named=True)
    coverage = out.coverage_epochs.row(0, named=True)
    assert "coverage_epoch_id" in SCHEMAS["withdrawal_events"]
    assert withdrawal["coverage_epoch_id"] == coverage["coverage_epoch_id"]
    assert withdrawal["event_ts"] == coverage["end_ts"] == datetime(2024, 1, 2, 9, 30, 5)
