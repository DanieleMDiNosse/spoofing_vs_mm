from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "estimate_depth_kernel.py"


def load_module():
    spec = importlib.util.spec_from_file_location("estimate_depth_kernel", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def raw_event(
    seq,
    event_type,
    order_id,
    side,
    price,
    qty,
    displayed,
    client,
    *,
    last_shares=None,
):
    timestamp = f"2024-01-02 09:30:{seq:02d}"
    return {
        "TRADEDATE": "2024-01-02",
        "MIC": "XMIL",
        "MARKETCODE": "MTA",
        "SYMBOLINDEX": 123,
        "EMM (*)": 1,
        "SEQUENCETIME": timestamp,
        "BOOKIN": timestamp,
        "BOOKOUTTIME": timestamp,
        "TRADETIME": None,
        "HDR_APPLKEYSEQUENCENUMBER": seq,
        "HDR_HWMSEQUENCENUMBER": seq,
        "HDR_OFFSETID": seq,
        "ROW_NUMBER": seq,
        "ORDEREVENTTYPE (*)": event_type,
        "ORDERID": order_id,
        "ORDERPRIORITY": str(seq),
        "ORDERSIDE (*)": side,
        "ORDERPX": price,
        "ORDERQTY": qty,
        "DISPLAYEDQTY": displayed,
        "LEAVESQTY": qty,
        "LASTSHARES": last_shares,
        "LASTTRADEDPX": price,
        "ORDERTYPE (*)": 2,
        "TIMEINFORCE (*)": 0,
        "FIRMID": "F1",
        "NMSC_ORIGINALCLIENTIDSHORTCODE": client,
        "ORDER_TRADINGCAPACITY (*)": 3,
        "PASSIVEORDER": "Y" if event_type == 3 else None,
        "AGGRESSIVEORDER": "N" if event_type == 3 else None,
    }


def test_parse_args_requires_full_sample_inputs(tmp_path: Path):
    module = load_module()

    args = module.parse_args(
        [
            "--input",
            str(tmp_path / "raw.parquet"),
            "--quote-panel",
            str(tmp_path / "lob_event_state_panel.parquet"),
            "--instrument-id",
            "RISANAMENTO",
            "--output-dir",
            str(tmp_path / "out"),
            "--top-n",
            "5",
            "--horizon-seconds",
            "10",
        ]
    )

    assert args.instrument_id == "RISANAMENTO"
    assert args.top_n == 5
    assert args.horizon_seconds == 10.0
    assert args.max_rows is None


def test_estimate_depth_kernel_cli_writes_reproducible_artifacts(tmp_path: Path):
    module = load_module()
    raw_path = tmp_path / "raw.parquet"
    quote_path = tmp_path / "quotes.parquet"
    output_dir = tmp_path / "out"
    pl.DataFrame(
        [
            raw_event(1, 1, "B1", 1, 100.0, 10, 10, "C1"),
            raw_event(2, 1, "B2", 1, 99.9, 20, 20, "C2"),
            raw_event(3, 1, "A1", 2, 100.1, 10, 10, "C3"),
            raw_event(4, 1, "A2", 2, 100.2, 20, 20, "C4"),
            raw_event(5, 3, "A2", 2, 100.2, 0, 0, "C4", last_shares=5),
        ]
    ).write_parquet(raw_path)
    pl.DataFrame({"post_best_bid": [100.0, 99.9], "post_best_ask": [100.1, 100.2]}).write_parquet(quote_path)

    module.main(
        [
            "--input",
            str(raw_path),
            "--quote-panel",
            str(quote_path),
            "--instrument-id",
            "ABC",
            "--output-dir",
            str(output_dir),
            "--top-n",
            "2",
            "--horizon-seconds",
            "10",
            "--visibility-floor",
            "1e-12",
        ]
    )

    kernel_path = output_dir / "empirical_depth_kernel.parquet"
    assert kernel_path.exists()
    assert (output_dir / "empirical_depth_kernel.csv").exists()
    assert (output_dir / "metadata.json").exists()
    assert (output_dir / "summary_report.md").exists()
    kernel = pl.read_parquet(kernel_path)
    sums = kernel.group_by("side").agg(pl.col("kernel_weight").sum().alias("weight_sum")).sort("side")
    assert sums["weight_sum"].to_list() == pytest.approx([1.0, 1.0])
    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert metadata["is_full_sample_calibration"] is True
    assert metadata["instrument_summary"]["instrument_id"] == "ABC"
