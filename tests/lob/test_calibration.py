from __future__ import annotations

import polars as pl
import pytest

from spoofing_detection.lob.calibration import build_threshold_table


def test_build_threshold_table_counts_alerts_and_positive_labels():
    scores = pl.DataFrame({"review_event_id": ["S1", "S2", "S3"], "MSCI": [0.9, 0.4, 0.1]})
    labels = pl.DataFrame(
        {
            "review_event_id": ["S1", "S2", "S3"],
            "analyst_label": ["strong_spoofing_like", "legitimate_market_making", "weak_spoofing_like"],
        }
    )
    table = build_threshold_table(scores, labels, score_column="MSCI", thresholds=[0.0, 0.5])
    row = table.filter(pl.col("threshold") == 0.5).row(0, named=True)
    assert row["alert_count"] == 1
    assert row["positive_label_count"] == 1
    assert row["precision_proxy"] == 1.0


def test_build_threshold_table_uses_strict_above_threshold_semantics():
    scores = pl.DataFrame({"review_event_id": ["neutral", "positive"], "MSCI": [0.0, 0.1]})
    labels = pl.DataFrame(
        {
            "review_event_id": ["neutral", "positive"],
            "analyst_label": ["legitimate_market_making", "weak_spoofing_like"],
        }
    )

    table = build_threshold_table(scores, labels, score_column="MSCI", thresholds=[0.0])

    assert table.item(0, "alert_count") == 1
    assert table.item(0, "positive_label_count") == 1


def test_build_threshold_table_requires_and_preserves_execution_anchor_strata():
    scores = pl.DataFrame(
        {
            "review_event_id": ["passive", "aggressive"],
            "execution_anchor_mode": ["passive", "aggressive"],
            "MSCI_resting_profile": [0.9, 0.1],
        }
    )
    labels = pl.DataFrame(
        {
            "review_event_id": ["passive", "aggressive"],
            "analyst_label": ["strong_spoofing_like", "legitimate_market_making"],
        }
    )

    with pytest.raises(ValueError, match="cannot be pooled"):
        build_threshold_table(
            scores,
            labels,
            score_column="MSCI_resting_profile",
            thresholds=[0.5],
        )

    table = build_threshold_table(
        scores,
        labels,
        score_column="MSCI_resting_profile",
        thresholds=[0.5],
        strata_columns=("execution_anchor_mode",),
    )
    assert table.select("execution_anchor_mode", "alert_count").sort(
        "execution_anchor_mode"
    ).rows() == [("aggressive", 0), ("passive", 1)]
