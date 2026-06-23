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
    scores = pl.DataFrame({"review_event_id": ["S1"], "MSCI": [0.7]})
    labels = pl.DataFrame({"review_event_id": ["S1"], "analyst_label": ["weak_spoofing_like"], "confidence": [0.5], "benign_explanation": [""], "notes": [""], "reviewer": ["a"], "reviewed_at_utc": ["2026-06-23T10:00:00Z"]})
    scores_path = tmp_path / "scores.parquet"
    labels_path = tmp_path / "labels.csv"
    output_dir = tmp_path / "calibration"
    scores.write_parquet(scores_path)
    labels.write_csv(labels_path)
    outputs = module.build_report(scores_path=scores_path, annotations_path=labels_path, output_dir=output_dir)
    assert outputs["csv"].exists()
    assert outputs["markdown"].exists()
    assert "Threshold Calibration" in outputs["markdown"].read_text()
