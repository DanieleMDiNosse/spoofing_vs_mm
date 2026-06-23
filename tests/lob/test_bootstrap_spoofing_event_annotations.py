from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_spoofing_event_annotations.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bootstrap_spoofing_event_annotations", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_empty_annotations_creates_one_row_per_event():
    module = _load_module()
    events = pl.DataFrame({"review_event_id": ["S2", "S1"]})
    annotations = module.build_empty_annotations(events, reviewer="analyst")
    assert annotations.get_column("review_event_id").to_list() == ["S1", "S2"]
    assert annotations.get_column("analyst_label").to_list() == ["unclear_needs_more_context", "unclear_needs_more_context"]
    assert annotations.get_column("reviewer").to_list() == ["analyst", "analyst"]


def test_merge_existing_annotations_preserves_labels():
    module = _load_module()
    events = pl.DataFrame({"review_event_id": ["S1", "S2"]})
    existing = pl.DataFrame({"review_event_id": ["S1"], "analyst_label": ["legitimate_market_making"], "confidence": [0.8], "benign_explanation": ["quote_refresh"], "notes": ["looks benign"], "reviewer": ["alice"], "reviewed_at_utc": ["2026-06-23T10:00:00Z"]})
    merged = module.merge_existing_annotations(events, existing, reviewer="bob")
    row_s1 = merged.filter(pl.col("review_event_id") == "S1").row(0, named=True)
    row_s2 = merged.filter(pl.col("review_event_id") == "S2").row(0, named=True)
    assert row_s1["analyst_label"] == "legitimate_market_making"
    assert row_s2["analyst_label"] == "unclear_needs_more_context"
