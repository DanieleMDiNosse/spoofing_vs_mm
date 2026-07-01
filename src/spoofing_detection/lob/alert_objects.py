from __future__ import annotations

import polars as pl

ALERT_SCHEMA = {
    "client_id": pl.Utf8,
    "event_count": pl.UInt32,
    "mcps_at_threshold": pl.Float64,
    "max_MSCI": pl.Float64,
    "mean_MSCI": pl.Float64,
    "matched_event_count": pl.UInt32,
    "matched_event_share": pl.Float64,
    "max_WMSCI_event": pl.Float64,
    "mean_WMSCI_event": pl.Float64,
    "positive_fpm_mid_share": pl.Float64,
    "positive_reversion_mid_share": pl.Float64,
    "mean_execution_price_advantage_vs_posture_mid": pl.Float64,
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
    prepared_risk = risk_features
    for column, default in {
        "matched_event_share": None,
        "max_WMSCI_event": None,
        "positive_fpm_mid_share": None,
        "positive_reversion_mid_share": None,
    }.items():
        if column not in prepared_risk.columns:
            prepared_risk = prepared_risk.with_columns(pl.lit(default).alias(column))
    joined = prepared_risk.join(legitimacy_features, on="client_id", how="left")
    alerts = (
        joined.with_columns(
            (
                pl.max_horizontal("mcps_at_threshold", "matched_event_share").fill_null(0.0) * 0.35
                + pl.col("max_WMSCI_event").fill_null(0.0).log1p().clip(0.0, 5.0) / 5.0 * 0.25
                + pl.col("positive_fpm_mid_share").fill_null(0.0).clip(0.0, 1.0) * 0.1
                + pl.col("positive_reversion_mid_share").fill_null(0.0).clip(0.0, 1.0) * 0.1
                + (1.0 - pl.col("side_symmetry_score").fill_null(0.5)).clip(0.0, 1.0) * 0.2
            ).alias("alert_score")
        )
        .filter((pl.col("event_count") >= min_events) & (pl.max_horizontal("mcps_at_threshold", "matched_event_share") >= min_mcps))
        .with_columns(pl.lit("human_review").alias("recommended_action"))
        .sort(["alert_score", "event_count"], descending=[True, True])
    )
    if alerts.is_empty():
        return empty_alerts()
    for column, dtype in ALERT_SCHEMA.items():
        if column not in alerts.columns:
            alerts = alerts.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return alerts.select(list(ALERT_SCHEMA))
