from __future__ import annotations

import math

import polars as pl

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.panel import reconstruct_dataframe


def ev(seq: int, event_type: int, order_id: str, side: int | None, price, leaves, displayed, *,
       firm="F1", client="C1", order_type=2, order_qty=None, last_shares=None, last_px=None,
       priority=None):
    return {
        "TRADEDATE": "2024-01-02",
        "MIC": "XMIL",
        "MARKETCODE": "MTA",
        "SYMBOLINDEX": 123,
        "EMM (*)": 1,
        "ISIN": "TEST0000001",
        "SEQUENCETIME": seq,
        "HDR_APPLKEYSEQUENCENUMBER": seq,
        "HDR_HWMSEQUENCENUMBER": seq,
        "HDR_OFFSETID": seq,
        "ROW_NUMBER": seq,
        "EVENTID": f"E{seq}",
        "ORDEREVENTTYPE (*)": event_type,
        "ORDERID": order_id,
        "ORDERPRIORITY": str(priority or seq),
        "ORDERSIDE (*)": side,
        "ORDERPX": price,
        "ORDERQTY": order_qty if order_qty is not None else (leaves or 0),
        "DISPLAYEDQTY": displayed,
        "LEAVESQTY": leaves,
        "LASTSHARES": last_shares,
        "LASTTRADEDPX": last_px,
        "ORDERTYPE (*)": order_type,
        "TIMEINFORCE (*)": 0,
        "FIRMID": firm,
        "NMSC_ORIGINALCLIENTIDSHORTCODE": client,
    }


def reconstruct(rows):
    return reconstruct_dataframe(pl.DataFrame(rows), config=LOBConfig(top_n=10))


def row(panel: pl.DataFrame, seq: int) -> dict:
    return panel.filter(pl.col("sort_index") == seq).to_dicts()[0]


def test_reconstruction_uses_all_events_and_mutates_visible_order_state():
    rows = [
        ev(1, 11, "B1", 1, 100.0, 10, 10, firm="F1", client="C1"),
        ev(2, 11, "A1", 2, 101.0, 5, 5, firm="F2", client="C2"),
        ev(3, 1, "B2", 1, 99.0, 20, 20, firm="F1", client="C3"),
        ev(4, 2, "B2", 1, 100.5, 20, 20, firm="F1", client="C3"),
        ev(5, 3, "B2", 1, 100.5, 7, 7, firm="F1", client="C3", last_shares=13, last_px=100.5),
        ev(6, 3, "B2", 1, 100.5, 0, 0, firm="F1", client="C3", last_shares=7, last_px=100.5),
        ev(7, 4, "B1", 1, 100.0, 0, 0, firm="F1", client="C1"),
        ev(8, 1, "M1", 1, None, 0, 0, firm="F1", client="C4", order_type=1),
        ev(9, 1, "I1", 2, 102.0, 100, 10, firm="F3", client="C5", order_type=10),
        ev(10, 7, "I1", 2, 102.0, 90, 15, firm="F3", client="C5", order_type=10),
    ]

    result = reconstruct(rows)
    panel = result.panel
    normalized = result.normalized_events

    assert panel.height == len(rows)
    assert normalized.height == len(rows)
    assert "agent_key_mode" not in panel.columns
    assert "agent_firm_client_id" not in panel.columns
    assert "pre_firm_active_bid_visible_qty" in panel.columns
    assert "pre_client_original_active_bid_visible_qty" in panel.columns

    r1 = row(panel, 1)
    assert r1["event_class"] == "session_reload"
    assert r1["pre_best_bid"] is None
    assert r1["post_best_bid"] == 100.0
    assert r1["post_firm_active_bid_visible_qty"] == 10

    r3 = row(panel, 3)
    assert r3["event_class"] == "new_order"
    assert r3["pre_firm_active_bid_visible_qty"] == 10
    assert r3["post_firm_active_bid_visible_qty"] == 30
    assert r3["pre_client_original_active_bid_visible_qty"] == 0
    assert r3["post_client_original_active_bid_visible_qty"] == 20
    assert r3["pre_event_order_same_side_distance_price"] == 1.0

    r4 = row(panel, 4)
    assert r4["event_class"] == "modify_order"
    assert r4["post_best_bid"] == 100.5
    assert r4["post_bid_level_1_price"] == 100.5
    assert r4["post_bid_level_1_visible_qty"] == 20
    assert r4["post_bid_level_2_price"] == 100.0
    assert r4["post_bid_level_2_visible_qty"] == 10
    assert r4["post_event_order_same_side_distance_price"] == 0.0

    r5 = row(panel, 5)
    assert r5["event_class"] == "fill"
    assert r5["post_bid_level_1_visible_qty"] == 7
    assert r5["post_firm_active_bid_visible_qty"] == 17

    r6 = row(panel, 6)
    assert r6["post_best_bid"] == 100.0
    assert r6["post_firm_active_bid_visible_qty"] == 10

    r7 = row(panel, 7)
    assert r7["event_class"] == "cancel"
    assert r7["post_best_bid"] is None

    r8 = row(panel, 8)
    assert r8["event_class"] == "new_order"
    assert r8["event_order_type_label"] == "market"
    assert r8["post_best_bid"] is None
    assert r8["normalization_issue_flags"] == "non_resting_unpriced_event"

    r9 = row(panel, 9)
    assert r9["event_order_type_label"] == "iceberg"
    assert r9["post_ask_level_1_price"] == 101.0
    assert r9["post_ask_level_2_price"] == 102.0
    assert r9["post_ask_level_2_visible_qty"] == 10

    r10 = row(panel, 10)
    assert r10["event_class"] == "iceberg_refill"
    assert r10["post_ask_level_2_visible_qty"] == 15


def test_unknown_event_type_raises_before_state_mutation():
    rows = [ev(1, 999, "X", 1, 100.0, 1, 1)]
    try:
        reconstruct(rows)
    except Exception as exc:
        assert exc.__class__.__name__ == "UnknownEnumError"
        assert "ORDEREVENTTYPE" in str(exc)
    else:
        raise AssertionError("unknown event type should fail loudly")


def test_missing_client_identity_is_flagged_but_firm_state_is_kept():
    rows = [ev(1, 1, "B1", 1, 100.0, 10, 10, firm="F1", client=None)]
    result = reconstruct(rows)
    r1 = row(result.panel, 1)
    assert r1["client_original_id_missing_flag"] is True
    assert r1["post_firm_active_bid_visible_qty"] == 10
    assert r1["post_client_original_active_bid_visible_qty"] == 0
