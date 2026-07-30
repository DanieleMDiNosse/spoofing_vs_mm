from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_spoofing_calibration_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_spoofing_calibration_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_calibration_report_writes_outputs(tmp_path):
    module = _load_module()
    scores = pl.DataFrame(
        {
            "review_event_id": ["S1", "S2"],
            "execution_anchor_mode": ["passive", "aggressive"],
            "MSCI_resting_profile": [0.7, 0.2],
            "MSCI": [-99.0, -99.0],
        }
    )
    labels = pl.DataFrame(
        {
            "review_event_id": ["S1", "S2"],
            "analyst_label": ["weak_spoofing_like", "legitimate_market_making"],
            "confidence": [0.5, 0.5],
            "benign_explanation": ["", ""],
            "notes": ["", ""],
            "reviewer": ["a", "a"],
            "reviewed_at_utc": ["2026-06-23T10:00:00Z", "2026-06-23T10:00:00Z"],
        }
    )
    scores_path = tmp_path / "scores.parquet"
    labels_path = tmp_path / "labels.csv"
    output_dir = tmp_path / "calibration"
    scores.write_parquet(scores_path)
    labels.write_csv(labels_path)
    outputs = module.build_report(scores_path=scores_path, annotations_path=labels_path, output_dir=output_dir)
    assert outputs["csv"].exists()
    assert outputs["markdown"].exists()
    markdown = outputs["markdown"].read_text()
    table = pl.read_csv(outputs["csv"])
    assert "Threshold Calibration" in markdown
    assert "no aggressive threshold is inferred here" in markdown
    assert table["execution_anchor_mode"].unique().sort().to_list() == ["aggressive", "passive"]
    assert table["threshold"].unique().sort().to_list() == [0.0, 0.1, 0.25, 0.5, 1.0, 1.5]
    assert table["score_column"].unique().to_list() == ["MSCI_resting_profile"]
