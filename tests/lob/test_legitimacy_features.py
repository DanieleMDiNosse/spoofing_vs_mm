from __future__ import annotations

import polars as pl

from spoofing_detection.lob.legitimacy_features import compute_legitimacy_features


def test_compute_legitimacy_features_measures_symmetry_and_cancel_after_fill():
    events = pl.DataFrame(
        {
            "client_id": ["A", "A", "A", "A"],
            "side": ["bid", "ask", "bid", "ask"],
            "is_execution_order": [False, True, False, False],
            "is_matched_deceptive_cancel_order": [False, False, True, False],
            "displayed_qty": [100.0, 90.0, 100.0, 80.0],
            "last_shares": [0.0, 50.0, 0.0, 0.0],
        }
    )
    features = compute_legitimacy_features(events)
    row = features.row(0, named=True)
    assert row["client_id"] == "A"
    assert row["bid_event_share"] == 0.5
    assert row["ask_event_share"] == 0.5
    assert row["side_symmetry_score"] == 1.0
    assert row["matched_cancel_event_share"] == 0.25


def test_compute_legitimacy_features_handles_empty_input():
    features = compute_legitimacy_features(pl.DataFrame())
    assert features.is_empty()
    assert "client_id" in features.columns
