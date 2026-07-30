from __future__ import annotations

import polars as pl

from spoofing_detection.lob.alert_objects import build_actor_session_alerts, build_client_session_alerts


def test_build_actor_session_alerts_preserves_firm_fallback_warning_and_anchor():
    risk = pl.DataFrame(
        {
            "actor_key": ["firm:F1", "firm:F1"],
            "actor_id": ["F1", "F1"],
            "identity_level": ["firm", "firm"],
            "identity_source": ["FIRMID", "FIRMID"],
            "identity_fallback_flag": [True, True],
            "execution_anchor_mode": ["passive", "aggressive"],
            "msci_threshold_applicable": [True, False],
            "msci_threshold": [0.5, None],
            "event_count": [5, 4],
            "matched_event_count": [3, 3],
            "matched_event_share": [0.6, 0.75],
            "mcps_at_threshold": [0.6, 0.5],
            "max_WMSCI_event": [10.0, 8.0],
            "mean_WMSCI_event": [4.0, 3.0],
            "max_MSCI": [0.9, 0.8],
            "mean_MSCI": [0.4, 0.3],
        }
    )
    legitimacy = pl.DataFrame(
        {
            "actor_key": ["firm:F1", "firm:F1"],
            "execution_anchor_mode": ["passive", "aggressive"],
            "side_symmetry_score": [0.2, 0.3],
            "matched_cancel_event_share": [0.5, 0.4],
        }
    )

    alerts = build_actor_session_alerts(risk, legitimacy, min_events=3, min_mcps=0.5)

    assert alerts.get_column("execution_anchor_mode").to_list() == ["passive"]
    assert set(alerts.get_column("identity_level")) == {"firm"}
    assert set(alerts.get_column("identity_fallback_flag")) == {True}
    assert set(alerts.get_column("identity_scope_warning")) == {
        "firm_fallback_may_aggregate_multiple_clients"
    }
    assert set(alerts.get_column("recommended_action")) == {"human_review"}


def test_build_client_session_alerts_reports_direct_msci_and_wmsci_values():
    risk = pl.DataFrame(
        {
            "client_id": ["A"],
            "event_count": [5],
            "mcps_at_threshold": [0.6],
            "max_MSCI": [0.9],
            "mean_MSCI": [0.4],
            "matched_event_count": [3],
            "matched_event_share": [0.6],
            "max_WMSCI_event": [10.0],
            "mean_WMSCI_event": [4.0],
            "positive_fpm_mid_share": [0.5],
            "positive_reversion_mid_share": [0.25],
            "mean_execution_price_advantage_vs_posture_mid": [0.01],
        }
    )
    legitimacy = pl.DataFrame(
        {
            "client_id": ["A"],
            "side_symmetry_score": [0.2],
            "execution_event_share": [0.1],
            "matched_cancel_event_share": [0.5],
        }
    )
    alerts = build_client_session_alerts(risk, legitimacy, min_events=3, min_mcps=0.5)
    row = alerts.row(0, named=True)
    assert row["client_id"] == "A"
    assert row["recommended_action"] == "human_review"
    assert row["max_WMSCI_event"] == 10.0
    assert row["mean_WMSCI_event"] == 4.0
    assert row["max_MSCI"] == 0.9
    assert row["mean_MSCI"] == 0.4
    assert "alert_score" not in alerts.columns


def test_build_client_session_alerts_ranks_by_wmsci_before_msci_and_event_count():
    risk = pl.DataFrame(
        {
            "client_id": ["high_wmsci", "high_msci"],
            "event_count": [3, 100],
            "mcps_at_threshold": [0.1, 0.9],
            "max_MSCI": [0.2, 1.9],
            "mean_MSCI": [0.1, 1.4],
            "matched_event_count": [3, 90],
            "matched_event_share": [0.1, 0.9],
            "max_WMSCI_event": [8.0, 5.0],
            "mean_WMSCI_event": [4.0, 3.0],
            "positive_fpm_mid_share": [0.0, 1.0],
            "positive_reversion_mid_share": [0.0, 1.0],
        }
    )
    legitimacy = pl.DataFrame(
        {
            "client_id": ["high_wmsci", "high_msci"],
            "side_symmetry_score": [0.0, 1.0],
            "execution_event_share": [0.1, 0.1],
            "matched_cancel_event_share": [0.1, 0.9],
        }
    )

    alerts = build_client_session_alerts(risk, legitimacy, min_events=3, min_mcps=0.0)

    assert alerts["client_id"].to_list() == ["high_wmsci", "high_msci"]


def test_build_client_session_alerts_filters_low_event_counts():
    risk = pl.DataFrame({"client_id": ["A"], "event_count": [1], "mcps_at_threshold": [1.0], "max_MSCI": [1.0], "mean_MSCI": [1.0]})
    legitimacy = pl.DataFrame({"client_id": ["A"], "side_symmetry_score": [0.5], "execution_event_share": [0.5], "matched_cancel_event_share": [0.5]})
    alerts = build_client_session_alerts(risk, legitimacy, min_events=3, min_mcps=0.5)
    assert alerts.is_empty()


def test_build_client_session_alerts_prefers_canonical_metrics_over_legacy_aliases():
    risk = pl.DataFrame(
        {
            "client_id": ["canonical"],
            "event_count": [4],
            "matched_event_count": [4],
            "matched_event_share": [1.0],
            "max_withdrawal_profile_scale_event": [8.0],
            "mean_withdrawal_profile_scale_event": [4.0],
            "max_MSCI_resting_profile": [0.8],
            "mean_MSCI_resting_profile": [0.5],
            "max_WMSCI_event": [-99.0],
            "mean_WMSCI_event": [-99.0],
            "max_MSCI": [-99.0],
            "mean_MSCI": [-99.0],
        }
    )
    legitimacy = pl.DataFrame(
        {
            "client_id": ["canonical"],
            "side_symmetry_score": [1.0],
            "matched_cancel_event_share": [1.0],
        }
    )

    alerts = build_client_session_alerts(risk, legitimacy, min_events=3, min_mcps=0.0)

    assert alerts["max_withdrawal_profile_scale_event"].to_list() == [8.0]
    assert alerts["max_MSCI_resting_profile"].to_list() == [0.8]
    assert alerts["max_WMSCI_event"].to_list() == [8.0]
    assert alerts["max_MSCI"].to_list() == [0.8]


def test_build_client_session_alerts_filters_msci_only_sessions():
    risk = pl.DataFrame(
        {
            "client_id": ["matched", "msci_only"],
            "event_count": [10, 100],
            "mcps_at_threshold": [0.0, 0.9],
            "max_MSCI": [0.0, 1.0],
            "mean_MSCI": [0.0, 0.5],
            "matched_event_count": [4, 0],
            "matched_event_share": [0.4, 0.0],
            "max_WMSCI_event": [10.0, 0.0],
            "mean_WMSCI_event": [2.0, 0.0],
            "positive_fpm_mid_share": [0.5, 1.0],
            "positive_reversion_mid_share": [0.25, 1.0],
            "mean_execution_price_advantage_vs_posture_mid": [0.01, 0.02],
        }
    )
    legitimacy = pl.DataFrame(
        {
            "client_id": ["matched", "msci_only"],
            "side_symmetry_score": [1.0, 0.0],
            "execution_event_share": [0.1, 0.1],
            "matched_cancel_event_share": [0.4, 0.0],
        }
    )

    alerts = build_client_session_alerts(risk, legitimacy, min_events=3, min_mcps=0.05)

    assert alerts["client_id"].to_list() == ["matched"]


def test_build_client_session_alerts_requires_repeated_matched_withdrawals():
    risk = pl.DataFrame(
        {
            "client_id": ["one_off"],
            "event_count": [10],
            "mcps_at_threshold": [0.0],
            "max_MSCI": [0.0],
            "mean_MSCI": [0.0],
            "matched_event_count": [1],
            "matched_event_share": [0.1],
            "max_WMSCI_event": [20.0],
            "mean_WMSCI_event": [20.0],
        }
    )
    legitimacy = pl.DataFrame({"client_id": ["one_off"], "side_symmetry_score": [0.0], "execution_event_share": [0.1], "matched_cancel_event_share": [0.1]})

    alerts = build_client_session_alerts(risk, legitimacy, min_events=3, min_mcps=0.05)

    assert alerts.is_empty()


def test_build_client_session_alerts_keeps_repeated_low_share_clusters_when_share_floor_disabled():
    risk = pl.DataFrame(
        {
            "client_id": ["active_client"],
            "event_count": [1_000],
            "mcps_at_threshold": [0.0],
            "max_MSCI": [0.0],
            "mean_MSCI": [0.0],
            "matched_event_count": [30],
            "matched_event_share": [0.03],
            "max_WMSCI_event": [12.0],
            "mean_WMSCI_event": [2.0],
        }
    )
    legitimacy = pl.DataFrame({"client_id": ["active_client"], "side_symmetry_score": [1.0], "execution_event_share": [0.1], "matched_cancel_event_share": [0.03]})

    alerts = build_client_session_alerts(risk, legitimacy, min_events=3, min_mcps=0.0)

    assert alerts["client_id"].to_list() == ["active_client"]
