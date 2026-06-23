from __future__ import annotations

import polars as pl

POSITIVE_LABELS = {"strong_spoofing_like", "moderate_spoofing_like", "weak_spoofing_like"}


def build_threshold_table(
    scores: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    score_column: str,
    thresholds: list[float],
) -> pl.DataFrame:
    if score_column not in scores.columns:
        raise ValueError(f"score column missing: {score_column}")
    joined = scores.join(labels.select(["review_event_id", "analyst_label"]), on="review_event_id", how="left")
    rows = []
    for threshold in thresholds:
        alerted = joined.filter(pl.col(score_column) >= threshold)
        positive_count = alerted.filter(pl.col("analyst_label").is_in(POSITIVE_LABELS)).height
        alert_count = alerted.height
        rows.append(
            {
                "score_column": score_column,
                "threshold": threshold,
                "alert_count": alert_count,
                "positive_label_count": positive_count,
                "precision_proxy": positive_count / alert_count if alert_count else None,
            }
        )
    return pl.DataFrame(rows)
