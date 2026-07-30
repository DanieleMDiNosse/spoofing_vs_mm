from __future__ import annotations

from collections.abc import Sequence

import polars as pl

POSITIVE_LABELS = {"strong_spoofing_like", "moderate_spoofing_like", "weak_spoofing_like"}


def build_threshold_table(
    scores: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    score_column: str,
    thresholds: list[float],
    strata_columns: Sequence[str] = (),
) -> pl.DataFrame:
    if score_column not in scores.columns:
        raise ValueError(f"score column missing: {score_column}")
    missing_strata = [column for column in strata_columns if column not in scores.columns]
    if missing_strata:
        raise ValueError(f"strata columns missing: {missing_strata}")
    if "execution_anchor_mode" in scores.columns and "execution_anchor_mode" not in strata_columns:
        raise ValueError(
            "execution_anchor_mode must be included in strata_columns; "
            "passive and aggressive scores cannot be pooled for calibration"
        )
    joined = scores.join(labels.select(["review_event_id", "analyst_label"]), on="review_event_id", how="left")
    rows = []
    groups = joined.partition_by(list(strata_columns), maintain_order=True) if strata_columns else [joined]
    for group in groups:
        stratum = {
            column: group.get_column(column).drop_nulls().unique().item()
            for column in strata_columns
        }
        for threshold in thresholds:
            alerted = group.filter(pl.col(score_column) > threshold)
            positive_count = alerted.filter(pl.col("analyst_label").is_in(POSITIVE_LABELS)).height
            alert_count = alerted.height
            rows.append(
                {
                    **stratum,
                    "score_column": score_column,
                    "threshold": threshold,
                    "alert_count": alert_count,
                    "positive_label_count": positive_count,
                    "precision_proxy": positive_count / alert_count if alert_count else None,
                }
            )
    return pl.DataFrame(rows)
