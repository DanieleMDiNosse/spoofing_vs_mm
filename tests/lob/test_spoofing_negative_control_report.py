from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_spoofing_negative_control_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_spoofing_negative_control_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_report_writes_markdown(tmp_path):
    module = _load_module()
    events = pl.DataFrame({"review_event_id": ["S1"], "sort_index": [10], "deceptive_side": ["bid"], "MSCI": [0.7]})
    events_path = tmp_path / "events.parquet"
    output = tmp_path / "report.md"
    events.write_parquet(events_path)
    module.build_report(events_path=events_path, output_path=output, shift_events=50)
    text = output.read_text()
    assert "# Spoofing Negative-Control Report" in text
    assert "time_shift" in text
    assert "wrong_side" in text
