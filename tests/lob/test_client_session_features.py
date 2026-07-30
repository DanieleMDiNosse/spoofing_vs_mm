from __future__ import annotations

import polars as pl

from spoofing_detection.lob.client_session_features import (
    compute_actor_session_features,
    compute_client_session_features,
)


def test_compute_actor_session_features_separates_anchor_denominators_and_preserves_firm_identity():
    executions = pl.DataFrame(
        {
            "actor_key": ["firm:F1"] * 5,
            "actor_id": ["F1"] * 5,
            "identity_level": ["firm"] * 5,
            "identity_source": ["FIRMID"] * 5,
            "identity_fallback_flag": [True] * 5,
            "execution_anchor_mode": ["passive", "passive", "aggressive", "aggressive", "aggressive"],
            "execution_cluster_id": ["P1", "P2", "A1", "A2", "A3"],
            "MSCI": [0.8, 0.2, 0.9, 0.1, 0.7],
            "SCI": [0.9, 0.3, 1.0, 0.2, 0.8],
            "collapse_opposite_side": [0.7, 0.1, 0.8, 0.1, 0.6],
            "collapse_same_side": [0.1, 0.2, 0.1, 0.2, 0.1],
            "matched_deceptive_cancel_fraction_window": [0.9, 0.4, 0.8, 0.0, 0.5],
            "fill_qty": [100.0, 50.0, 20.0, 10.0, 30.0],
        }
    )

    features = compute_actor_session_features(executions, msci_threshold=0.5)
    by_anchor = {row["execution_anchor_mode"]: row for row in features.to_dicts()}

    assert by_anchor["passive"]["event_count"] == 2
    assert by_anchor["aggressive"]["event_count"] == 3
    assert all(row["actor_key"] == "firm:F1" for row in by_anchor.values())
    assert all(row["identity_level"] == "firm" for row in by_anchor.values())
    assert all(row["identity_fallback_flag"] is True for row in by_anchor.values())


def test_actor_session_features_leave_uncalibrated_anchor_threshold_metrics_null():
    executions = pl.DataFrame(
        {
            "actor_key": ["firm:F1", "firm:F1"],
            "actor_id": ["F1", "F1"],
            "identity_level": ["firm", "firm"],
            "identity_source": ["FIRMID", "FIRMID"],
            "identity_fallback_flag": [True, True],
            "execution_anchor_mode": ["passive", "aggressive"],
            "MSCI_resting_profile": [0.8, 0.9],
            "SCI": [0.7, 0.8],
            "collapse_opposite_side": [0.6, 0.7],
            "collapse_same_side": [0.1, 0.1],
            "matched_deceptive_cancel_fraction_window": [0.9, 0.8],
            "execution_quantity": [100.0, 50.0],
        }
    )

    features = compute_actor_session_features(
        executions,
        msci_threshold={"passive": 0.5, "aggressive": None},
    )
    by_anchor = {row["execution_anchor_mode"]: row for row in features.to_dicts()}

    assert by_anchor["passive"]["msci_threshold_applicable"] is True
    assert by_anchor["passive"]["msci_threshold"] == 0.5
    assert by_anchor["passive"]["msci_exceedance_count"] == 1
    assert by_anchor["passive"]["mcps_at_threshold"] == 1.0
    assert by_anchor["aggressive"]["msci_threshold_applicable"] is False
    assert by_anchor["aggressive"]["msci_threshold"] is None
    assert by_anchor["aggressive"]["msci_exceedance_count"] is None
    assert by_anchor["aggressive"]["mcps_at_threshold"] is None


def test_compute_client_session_features_aggregates_repeated_events():
    executions = pl.DataFrame(
        {
            "client_id": ["A", "A", "B"],
            "execution_cluster_id": ["EC1", "EC2", "EC3"],
            "child_fill_count": [5, 1, 1],
            "event_ts": ["2024-06-10T10:00:00", "2024-06-10T10:01:00", "2024-06-10T10:02:00"],
            "top_n": [3, 3, 3],
            "MSCI_resting_profile": [0.8, 0.2, 0.0],
            "MSCI": [0.0, 0.0, 0.0],
            "SCI": [0.9, 0.3, 0.1],
            "collapse_opposite_side": [0.7, 0.1, 0.0],
            "collapse_same_side": [0.1, 0.2, 0.0],
            "matched_deceptive_cancel_fraction_window": [0.9, 0.4, 0.0],
            "fill_qty": [100.0, 50.0, 20.0],
            "has_matched_deceptive_cancel_window": [True, False, False],
            "withdrawal_profile_scale_event": [4.0, 0.0, 0.0],
            "WMSCI_event": [40.0, 0.0, 0.0],
            "withdrawal_to_execution_ratio": [10.0, 0.0, 0.0],
            "withdrawal_to_fill_ratio": [100.0, 0.0, 0.0],
            "execution_quantity": [100.0, 50.0, 20.0],
            "favorable_mid_move_pre_fill": [0.01, -0.01, None],
            "post_cancel_mid_reversion": [0.0, 0.02, None],
            "execution_price_advantage_vs_posture_mid": [0.03, -0.01, None],
        }
    )
    features = compute_client_session_features(executions, msci_threshold=0.5)
    row_a = features.filter(pl.col("client_id") == "A").row(0, named=True)
    assert row_a["event_count"] == 2
    assert row_a["raw_fill_message_count"] == 6
    assert row_a["msci_exceedance_count"] == 1
    assert row_a["mcps_at_threshold"] == 0.5
    assert row_a["max_MSCI"] == 0.8
    assert row_a["max_MSCI_resting_profile"] == 0.8
    assert row_a["matched_event_count"] == 1
    assert row_a["matched_event_share"] == 0.5
    assert row_a["max_WMSCI_event"] == 4.0
    assert row_a["mean_WMSCI_event"] == 4.0
    assert row_a["max_withdrawal_profile_scale_event"] == 4.0
    assert row_a["mean_withdrawal_profile_scale_event"] == 4.0
    assert row_a["mean_withdrawal_to_fill_ratio"] == 10.0
    assert row_a["mean_withdrawal_to_execution_ratio"] == 10.0
    assert row_a["total_execution_quantity"] == 150.0
    assert row_a["positive_fpm_mid_share"] == 1.0
    assert row_a["positive_reversion_mid_share"] == 0.0
    assert row_a["mean_execution_price_advantage_vs_posture_mid"] == 0.03


def test_compute_client_session_features_handles_empty_input():
    features = compute_client_session_features(pl.DataFrame(), msci_threshold=0.5)
    assert features.is_empty()
    assert "client_id" in features.columns


def test_compute_client_session_features_ranks_by_wmsci_before_matched_share():
    executions = pl.DataFrame(
        {
            "client_id": ["high_share", "high_wmsci", "high_wmsci"],
            "MSCI": [1.8, 0.2, 0.1],
            "SCI": [1.0, 0.2, 0.1],
            "collapse_opposite_side": [1.0, 0.2, 0.1],
            "collapse_same_side": [0.0, 0.0, 0.0],
            "matched_deceptive_cancel_fraction_window": [1.0, 1.0, 0.0],
            "fill_qty": [10.0, 10.0, 10.0],
            "has_matched_deceptive_cancel_window": [True, True, False],
            "WMSCI_event": [2.0, 10.0, 0.0],
        }
    )

    features = compute_client_session_features(executions, msci_threshold=0.1)

    assert features["client_id"].to_list() == ["high_wmsci", "high_share"]


def test_compute_client_session_features_rejects_duplicate_cluster_rows():
    executions = pl.DataFrame(
        {
            "client_id": ["A", "A"],
            "execution_cluster_id": ["EC1", "EC1"],
            "MSCI": [0.8, 0.8],
            "SCI": [0.9, 0.9],
            "collapse_opposite_side": [0.7, 0.7],
            "collapse_same_side": [0.1, 0.1],
            "matched_deceptive_cancel_fraction_window": [0.9, 0.9],
            "fill_qty": [100.0, 100.0],
        }
    )

    try:
        compute_client_session_features(executions, msci_threshold=0.5)
    except ValueError as exc:
        assert "one row per execution_cluster_id" in str(exc)
    else:
        raise AssertionError("duplicate execution clusters must be rejected")
