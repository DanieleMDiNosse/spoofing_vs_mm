from __future__ import annotations

import polars as pl

from spoofing_detection.lob.legitimacy_features import (
    compute_actor_legitimacy_features,
    compute_legitimacy_features,
)


def test_compute_actor_legitimacy_features_separates_anchor_branches():
    events = pl.DataFrame(
        {
            "actor_key": ["firm:F1"] * 4,
            "actor_id": ["F1"] * 4,
            "identity_level": ["firm"] * 4,
            "identity_source": ["FIRMID"] * 4,
            "identity_fallback_flag": [True] * 4,
            "execution_anchor_mode": ["passive", "passive", "aggressive", "aggressive"],
            "side": ["bid", "ask", "bid", "ask"],
            "is_execution_order": [False, True, True, False],
            "is_matched_deceptive_cancel_order": [True, False, False, True],
            "displayed_qty": [100.0, 90.0, 80.0, 70.0],
            "last_shares": [0.0, 50.0, 20.0, 0.0],
        }
    )

    features = compute_actor_legitimacy_features(events)

    assert set(features.get_column("execution_anchor_mode")) == {"passive", "aggressive"}
    assert features.get_column("event_count").to_list() == [2, 2]
    assert set(features.get_column("identity_level")) == {"firm"}


def test_compute_actor_legitimacy_features_excludes_unattributable_events():
    events = pl.DataFrame(
        {
            "actor_key": ["firm:F1", ""],
            "actor_id": ["F1", None],
            "identity_level": ["firm", None],
            "identity_source": ["FIRMID", None],
            "identity_fallback_flag": [True, None],
            "execution_anchor_mode": ["aggressive", "aggressive"],
            "side": ["bid", "ask"],
            "is_execution_order": [True, False],
            "is_matched_deceptive_cancel_order": [False, False],
            "displayed_qty": [80.0, 70.0],
            "last_shares": [20.0, 0.0],
        }
    )

    features = compute_actor_legitimacy_features(events)

    assert features.get_column("actor_key").to_list() == ["firm:F1"]
    assert features.get_column("event_count").to_list() == [1]


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
