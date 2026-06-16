from __future__ import annotations

import polars as pl

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.panel import reconstruct_dataframe


def ev(seq: int, event_type: int, order_id: str, side: int | None, price, leaves, displayed):
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
        "ORDERID": order_id,
        "ORDERPRIORITY": str(seq),
        "ORDERSIDE (*)": side,
        "ORDERPX": price,
        "ORDERQTY": leaves or 0,
        "DISPLAYEDQTY": displayed,
        "LEAVESQTY": leaves,
        "ORDERTYPE (*)": 2,
        "TIMEINFORCE (*)": 0,
        "FIRMID": "F1",
        "NMSC_ORIGINALCLIENTIDSHORTCODE": "C1",
    }


def test_late_issue_flags_do_not_break_polars_dataframe_construction():
    rows = [ev(seq, 11, f"B{seq}", 1, 100.0 + seq, 10, 10) for seq in range(1, 102)]
    rows.append(ev(102, 4, "UNKNOWN", 1, 99.0, 0, 0))

    result = reconstruct_dataframe(pl.DataFrame(rows), config=LOBConfig())

    assert result.panel.height == 102
    assert result.panel.filter(pl.col("sort_index") == 102).item(0, "lob_issue_flags") == "cancel_for_unseen_order"
    assert result.validation["issue_counts"]["cancel_for_unseen_order"] == 1
