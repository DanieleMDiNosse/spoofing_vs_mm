from __future__ import annotations

import polars as pl

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.panel import reconstruct_dataframe


def ev(seq: int, event_type: int, order_id: str, side: int, price: float, leaves: int, displayed: int, *, firm: str = "F1", client: str | None = "C1"):
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
        "ORDERQTY": leaves,
        "DISPLAYEDQTY": displayed,
        "LEAVESQTY": leaves,
        "ORDERTYPE (*)": 2,
        "TIMEINFORCE (*)": 0,
        "FIRMID": firm,
        "NMSC_ORIGINALCLIENTIDSHORTCODE": client,
    }


def test_reconstruction_emits_agent_long_panel_and_debug_snapshots():
    rows = [
        ev(1, 11, "B1", 1, 100.0, 10, 10, firm="F1", client="C1"),
        ev(2, 11, "A1", 2, 101.0, 5, 5, firm="F2", client="C2"),
        ev(3, 1, "B2", 1, 99.0, 20, 20, firm="F1", client=None),
        ev(4, 4, "B1", 1, 100.0, 0, 0, firm="F1", client="C1"),
    ]

    result = reconstruct_dataframe(
        pl.DataFrame(rows),
        config=LOBConfig(snapshot_mode="every_event_for_sample"),
    )

    assert result.agent_event_state_panel.height == len(rows) * 2
    assert set(result.agent_event_state_panel["agent_dimension"].unique().to_list()) == {"firm", "client_original"}

    firm_row = result.agent_event_state_panel.filter(
        (pl.col("sort_index") == 3) & (pl.col("agent_dimension") == "firm")
    ).to_dicts()[0]
    assert firm_row["agent_id"] == "F1"
    assert firm_row["agent_id_source"] == "FIRMID"
    assert firm_row["agent_id_missing_flag"] is False
    assert firm_row["pre_agent_bid_visible_qty"] == 10
    assert firm_row["post_agent_bid_visible_qty"] == 30
    assert firm_row["post_agent_same_side_visible_qty"] == 30

    client_row = result.agent_event_state_panel.filter(
        (pl.col("sort_index") == 3) & (pl.col("agent_dimension") == "client_original")
    ).to_dicts()[0]
    assert client_row["agent_id"] is None
    assert client_row["agent_id_source"] == "NMSC_ORIGINALCLIENTIDSHORTCODE"
    assert client_row["agent_id_missing_flag"] is True
    assert client_row["pre_agent_bid_visible_qty"] == 0
    assert client_row["post_agent_bid_visible_qty"] == 0

    active = result.active_order_snapshots
    assert active.height == 8  # 1 + 2 + 3 + 2 active orders after the four events.
    latest = active.filter(pl.col("snapshot_sort_index") == 4)
    assert set(latest["ORDERID"].to_list()) == {"A1", "B2"}
    assert latest.item(0, "snapshot_reason") == "post_event"

    depth = result.price_level_depth_snapshots
    assert depth.filter((pl.col("snapshot_sort_index") == 3) & (pl.col("side") == "bid")).height == 2
    best_bid = depth.filter(
        (pl.col("snapshot_sort_index") == 3) & (pl.col("side") == "bid") & (pl.col("rank") == 1)
    ).to_dicts()[0]
    assert best_bid["price"] == 100.0
    assert best_bid["visible_qty"] == 10


def test_end_of_partition_snapshot_mode_emits_only_final_state():
    rows = [
        ev(1, 11, "B1", 1, 100.0, 10, 10),
        ev(2, 11, "A1", 2, 101.0, 5, 5),
        ev(3, 4, "B1", 1, 100.0, 0, 0),
    ]

    result = reconstruct_dataframe(pl.DataFrame(rows), config=LOBConfig(snapshot_mode="end_of_partition"))

    assert result.active_order_snapshots.height == 1
    assert result.active_order_snapshots.item(0, "ORDERID") == "A1"
    assert result.active_order_snapshots.item(0, "snapshot_sort_index") == 3
    assert result.active_order_snapshots.item(0, "snapshot_reason") == "end_of_partition"
    assert result.price_level_depth_snapshots.height == 1
    assert result.price_level_depth_snapshots.item(0, "side") == "ask"
