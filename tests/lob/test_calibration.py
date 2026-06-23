from __future__ import annotations

import polars as pl

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
