from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.io import reconstruct_file


def test_reconstruct_file_writes_panel_normalized_and_metadata(tmp_path: Path):
    input_path = tmp_path / "events.parquet"
    output_dir = tmp_path / "out"
    df = pl.DataFrame([
        {
            "TRADEDATE": "2024-01-02",
            "MIC": "XMIL",
            "MARKETCODE": "MTA",
            "SYMBOLINDEX": 123,
            "EMM (*)": 1,
            "ISIN": "TEST0000001",
            "SEQUENCETIME": 1,
            "HDR_APPLKEYSEQUENCENUMBER": 1,
            "HDR_HWMSEQUENCENUMBER": 1,
            "HDR_OFFSETID": 1,
            "ROW_NUMBER": 1,
            "EVENTID": "E1",
            "ORDEREVENTTYPE (*)": 11,
            "ORDERID": "B1",
            "ORDERPRIORITY": "1",
            "ORDERSIDE (*)": 1,
            "ORDERPX": 100.0,
            "ORDERQTY": 10,
            "DISPLAYEDQTY": 10,
            "LEAVESQTY": 10,
            "LASTSHARES": None,
            "LASTTRADEDPX": None,
            "ORDERTYPE (*)": 2,
            "TIMEINFORCE (*)": 1,
            "FIRMID": "F1",
            "NMSC_ORIGINALCLIENTIDSHORTCODE": "C1",
        }
    ])
    df.write_parquet(input_path)

    paths = reconstruct_file(input_path, output_dir, config=LOBConfig(top_n=10))

    assert paths.panel_path.exists()
    assert paths.normalized_path.exists()
    assert paths.agent_panel_path.exists()
    assert paths.active_orders_path.exists()
    assert paths.price_level_depth_path.exists()
    assert paths.metadata_path.exists()
    assert paths.validation_path.exists()

    panel = pl.read_parquet(paths.panel_path)
    normalized = pl.read_parquet(paths.normalized_path)
    agent_panel = pl.read_parquet(paths.agent_panel_path)
    active_orders = pl.read_parquet(paths.active_orders_path)
    price_depth = pl.read_parquet(paths.price_level_depth_path)
    metadata = json.loads(paths.metadata_path.read_text())

    assert panel.height == 1
    assert normalized.height == 1
    assert agent_panel.height == 2
    assert active_orders.height == 1
    assert price_depth.height == 1
    assert panel.item(0, "post_best_bid") == 100.0
    assert metadata["config"]["top_n"] == 10
    assert metadata["config"]["agent_dimensions"] == ["firm", "client_original"]
    assert metadata["config"]["snapshot_mode"] == "end_of_partition"
    assert metadata["config"]["use_parquet_values_as_is"] is True
    assert metadata["row_counts"]["panel"] == 1
    assert metadata["row_counts"]["agent_event_state_panel"] == 2
    assert metadata["row_counts"]["active_order_snapshots"] == 1
    assert metadata["row_counts"]["price_level_depth_snapshots"] == 1
