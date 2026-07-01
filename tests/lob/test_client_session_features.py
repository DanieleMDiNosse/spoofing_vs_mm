from __future__ import annotations

import polars as pl

from spoofing_detection.lob.client_session_features import compute_client_session_features


def test_compute_client_session_features_aggregates_repeated_events():
    executions = pl.DataFrame(
        {
            "client_id": ["A", "A", "B"],
            "event_ts": ["2024-06-10T10:00:00", "2024-06-10T10:01:00", "2024-06-10T10:02:00"],
            "top_n": [3, 3, 3],
            "MSCI": [0.8, 0.2, 0.0],
            "SCI": [0.9, 0.3, 0.1],
            "collapse_opposite_side": [0.7, 0.1, 0.0],
            "collapse_same_side": [0.1, 0.2, 0.0],
            "matched_deceptive_cancel_fraction_window": [0.9, 0.4, 0.0],
            "fill_qty": [100.0, 50.0, 20.0],
            "has_matched_deceptive_cancel_window": [True, False, False],
            "WMSCI_event": [4.0, 0.0, 0.0],
            "withdrawal_to_fill_ratio": [10.0, 0.0, 0.0],
            "favorable_mid_move_pre_fill": [0.01, -0.01, None],
            "post_cancel_mid_reversion": [0.0, 0.02, None],
            "execution_price_advantage_vs_posture_mid": [0.03, -0.01, None],
        }
    )
    features = compute_client_session_features(executions, msci_threshold=0.5)
    row_a = features.filter(pl.col("client_id") == "A").row(0, named=True)
    assert row_a["event_count"] == 2
    assert row_a["msci_exceedance_count"] == 1
    assert row_a["mcps_at_threshold"] == 0.5
    assert row_a["max_MSCI"] == 0.8
    assert row_a["matched_event_count"] == 1
    assert row_a["matched_event_share"] == 0.5
    assert row_a["max_WMSCI_event"] == 4.0
    assert row_a["positive_fpm_mid_share"] == 0.5
    assert row_a["positive_reversion_mid_share"] == 0.5


def test_compute_client_session_features_handles_empty_input():
    features = compute_client_session_features(pl.DataFrame(), msci_threshold=0.5)
    assert features.is_empty()
    assert "client_id" in features.columns
