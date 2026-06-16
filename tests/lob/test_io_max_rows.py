from __future__ import annotations

from pathlib import Path

import polars as pl

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.io import reconstruct_file


def event(seq: int, event_type: int, order_id: str):
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
        "ORDERSIDE (*)": 1,
        "ORDERPX": 100.0,
        "ORDERQTY": 10,
        "DISPLAYEDQTY": 10 if event_type != 4 else 0,
        "LEAVESQTY": 10 if event_type != 4 else 0,
        "ORDERTYPE (*)": 2,
        "TIMEINFORCE (*)": 0,
        "FIRMID": "F1",
        "NMSC_ORIGINALCLIENTIDSHORTCODE": "C1",
    }


def test_max_rows_is_applied_after_deterministic_sort(tmp_path: Path):
    input_path = tmp_path / "unsorted.parquet"
    output_dir = tmp_path / "out"
    # Physical order is intentionally wrong: if max_rows is applied before
    # sorting, this keeps the cancel row and misses the reload seed.
    pl.DataFrame([event(2, 4, "B1"), event(1, 11, "B1")]).write_parquet(input_path)

    paths = reconstruct_file(input_path, output_dir, config=LOBConfig(), max_rows=1)
    panel = pl.read_parquet(paths.panel_path)

    assert panel.height == 1
    assert panel.item(0, "event_class") == "session_reload"
    assert panel.item(0, "post_best_bid") == 100.0
