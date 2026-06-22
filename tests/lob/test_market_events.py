from __future__ import annotations

import polars as pl

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.panel import reconstruct_dataframe


def market_event(seq: int, event_type: int):
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
        "ORDEREVENTTYPE (*)": event_type,
        "ORDERID": f"M{seq}",
        "ORDERPRIORITY": str(seq),
        "ORDERSIDE (*)": 1,
        "ORDERPX": None,
        "ORDERQTY": 10,
        "DISPLAYEDQTY": 0,
        "LEAVESQTY": 0,
        "LASTSHARES": 10 if event_type == 3 else None,
        "LASTTRADEDPX": 100.0 if event_type == 3 else None,
        "ORDERTYPE (*)": 1,
        "TIMEINFORCE (*)": 3,
        "FIRMID": "F1",
        "NMSC_ORIGINALCLIENTIDSHORTCODE": "C1",
    }


def unpriced_conditional_event(seq: int, *, order_type: int):
    row = market_event(seq, 1)
    row.update(
        {
            "ORDERID": f"C{seq}",
            "ORDERQTY": 10,
            "DISPLAYEDQTY": 10,
            "LEAVESQTY": 10,
            "LASTSHARES": None,
            "LASTTRADEDPX": None,
            "ORDERTYPE (*)": order_type,
            "TIMEINFORCE (*)": 0,
        }
    )
    return row


def test_unseen_market_fill_or_cancel_is_non_resting_not_missing_active_order():
    result = reconstruct_dataframe(
        pl.DataFrame([market_event(1, 3), market_event(2, 4)]),
        config=LOBConfig(),
    )

    assert result.panel.height == 2
    assert result.panel["lob_issue_flags"].to_list() == [None, None]
    assert "full_fill_for_unseen_order" not in result.validation["issue_counts"]
    assert "cancel_for_unseen_order" not in result.validation["issue_counts"]
    assert result.validation["issue_counts"]["non_resting_unpriced_event"] == 2


def test_unpriced_stop_market_order_is_non_resting_not_missing_resting_price():
    result = reconstruct_dataframe(
        pl.DataFrame([unpriced_conditional_event(1, order_type=3)]),
        config=LOBConfig(),
    )

    assert result.panel["post_best_bid"].to_list() == [None]
    assert result.panel["post_best_ask"].to_list() == [None]
    assert result.normalized_events["normalization_issue_flags"].to_list() == [
        "non_resting_unpriced_event"
    ]
    assert result.validation["issue_counts"]["non_resting_unpriced_event"] == 1
    assert "missing_price_for_potential_resting_event" not in result.validation["issue_counts"]
