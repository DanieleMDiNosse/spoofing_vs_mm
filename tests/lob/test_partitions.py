from __future__ import annotations

import polars as pl

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.panel import reconstruct_dataframe


def ev(seq: int, symbol: int, event_type: int, order_id: str, side: int, price: float, leaves: int, displayed: int):
    return {
        "TRADEDATE": "2024-01-02",
        "MIC": "XMIL",
        "MARKETCODE": "MTA",
        "SYMBOLINDEX": symbol,
        "EMM (*)": 1,
        "ISIN": f"TEST{symbol}",
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
        "ORDERQTY": leaves,
        "DISPLAYEDQTY": displayed,
        "LEAVESQTY": leaves,
        "ORDERTYPE (*)": 2,
        "TIMEINFORCE (*)": 0,
        "FIRMID": "F1",
        "NMSC_ORIGINALCLIENTIDSHORTCODE": "C1",
    }


def test_active_state_is_isolated_by_partition():
    rows = [
        ev(1, 111, 11, "SAME_ORDER_ID", 1, 100.0, 10, 10),
        ev(2, 111, 11, "A1", 2, 101.0, 5, 5),
        ev(1, 222, 11, "B2", 1, 200.0, 20, 20),
        ev(2, 222, 4, "SAME_ORDER_ID", 1, 100.0, 0, 0),
    ]

    result = reconstruct_dataframe(pl.DataFrame(rows), config=LOBConfig(snapshot_mode="end_of_partition"))
    panel = result.panel.sort("sort_index")

    symbol222_cancel = panel.filter((pl.col("SYMBOLINDEX") == 222) & (pl.col("event_class") == "cancel")).to_dicts()[0]
    assert symbol222_cancel["pre_best_bid"] == 200.0
    assert symbol222_cancel["post_best_bid"] == 200.0
    assert symbol222_cancel["lob_issue_flags"] == "cancel_for_unseen_order"

    snapshots = result.active_order_snapshots
    assert snapshots.filter(pl.col("partition_id").str.contains("111")).height == 2
    assert snapshots.filter(pl.col("partition_id").str.contains("222")).height == 1
    assert result.validation["partitions_processed"] == 2
    assert result.validation["active_orders_end"] == 3
