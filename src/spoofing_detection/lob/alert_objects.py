from __future__ import annotations

import polars as pl

ALERT_SCHEMA = {
    "client_id": pl.Utf8,
    "event_count": pl.UInt32,
    "mcps_at_threshold": pl.Float64,
    "max_MSCI": pl.Float64,
    "mean_MSCI": pl.Float64,
    "side_symmetry_score": pl.Float64,
    "matched_cancel_event_share": pl.Float64,
    "alert_score": pl.Float64,
    "recommended_action": pl.Utf8,
}


def empty_alerts() -> pl.DataFrame:
    return pl.DataFrame(schema=ALERT_SCHEMA)


def build_client_session_alerts(
    risk_features: pl.DataFrame,
    legitimacy_features: pl.DataFrame,
    *,
    min_events: int,
    min_mcps: float,
) -> pl.DataFrame:
    if risk_features.is_empty():
        return empty_alerts()
    joined = risk_features.join(legitimacy_features, on="client_id", how="left")
    alerts = (
        joined.with_columns(
            (
                pl.col("mcps_at_threshold").fill_null(0.0) * 0.5
                + pl.col("max_MSCI").fill_null(0.0).clip(0.0, 1.0) * 0.3
                + (1.0 - pl.col("side_symmetry_score").fill_null(0.5)).clip(0.0, 1.0) * 0.2
            ).alias("alert_score")
        )
        .filter((pl.col("event_count") >= min_events) & (pl.col("mcps_at_threshold") >= min_mcps))
        .with_columns(pl.lit("human_review").alias("recommended_action"))
        .sort(["alert_score", "event_count"], descending=[True, True])
    )
    if alerts.is_empty():
        return empty_alerts()
    for column, dtype in ALERT_SCHEMA.items():
        if column not in alerts.columns:
            alerts = alerts.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return alerts.select(list(ALERT_SCHEMA))
