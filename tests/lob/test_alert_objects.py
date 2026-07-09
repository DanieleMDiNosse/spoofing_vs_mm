from __future__ import annotations

import math

import polars as pl

from spoofing_detection.lob.alert_objects import build_client_session_alerts


def test_build_client_session_alerts_combines_risk_and_legitimacy():
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
    assert row["alert_score"] > 0
    expected_score = 0.5 * 0.6 + 0.3 * (math.log1p(10.0) / 5.0) + 0.2 * ((0.5 + 0.25) / 2.0)
    assert row["alert_score"] == expected_score


def test_build_client_session_alerts_filters_low_event_counts():
    risk = pl.DataFrame({"client_id": ["A"], "event_count": [1], "mcps_at_threshold": [1.0], "max_MSCI": [1.0], "mean_MSCI": [1.0]})
    legitimacy = pl.DataFrame({"client_id": ["A"], "side_symmetry_score": [0.5], "execution_event_share": [0.5], "matched_cancel_event_share": [0.5]})
    alerts = build_client_session_alerts(risk, legitimacy, min_events=3, min_mcps=0.5)
    assert alerts.is_empty()


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
