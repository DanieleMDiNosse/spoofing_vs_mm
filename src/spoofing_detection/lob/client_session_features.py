from __future__ import annotations

import polars as pl

CLIENT_SESSION_FEATURE_SCHEMA = {
    "client_id": pl.Utf8,
    "event_count": pl.UInt32,
    "msci_exceedance_count": pl.UInt32,
    "mcps_at_threshold": pl.Float64,
    "max_MSCI": pl.Float64,
    "mean_MSCI": pl.Float64,
    "matched_event_count": pl.UInt32,
    "matched_event_share": pl.Float64,
    "max_WMSCI_event": pl.Float64,
    "mean_WMSCI_event": pl.Float64,
    "mean_withdrawal_to_fill_ratio": pl.Float64,
    "positive_fpm_mid_share": pl.Float64,
    "positive_reversion_mid_share": pl.Float64,
    "mean_execution_price_advantage_vs_posture_mid": pl.Float64,
    "mean_SCI": pl.Float64,
    "mean_opposite_collapse": pl.Float64,
    "mean_same_side_collapse": pl.Float64,
    "mean_matched_cancel_fraction": pl.Float64,
    "total_fill_qty": pl.Float64,
}


def empty_client_session_features() -> pl.DataFrame:
    return pl.DataFrame(schema=CLIENT_SESSION_FEATURE_SCHEMA)


def compute_client_session_features(executions: pl.DataFrame, *, msci_threshold: float) -> pl.DataFrame:
    if executions.is_empty() or "client_id" not in executions.columns:
        return empty_client_session_features()
    required = [
        "client_id",
        "MSCI",
        "SCI",
        "collapse_opposite_side",
        "collapse_same_side",
        "matched_deceptive_cancel_fraction_window",
        "fill_qty",
    ]
    missing = [column for column in required if column not in executions.columns]
    if missing:
        raise ValueError(f"missing execution metric columns: {missing}")
    optional_defaults = {
        "has_matched_deceptive_cancel_window": False,
        "WMSCI_event": None,
        "withdrawal_to_fill_ratio": None,
        "favorable_mid_move_pre_fill": None,
        "post_cancel_mid_reversion": None,
        "execution_price_advantage_vs_posture_mid": None,
    }
    prepared = executions
    for column, default in optional_defaults.items():
        if column not in prepared.columns:
            prepared = prepared.with_columns(pl.lit(default).alias(column))
    return (
        prepared.with_columns(
            [
                pl.col("client_id").cast(pl.Utf8),
                (pl.col("MSCI") > msci_threshold).cast(pl.UInt8).alias("msci_exceeds_threshold"),
                pl.col("has_matched_deceptive_cancel_window").fill_null(False).cast(pl.UInt8).alias("matched_event"),
                (pl.col("favorable_mid_move_pre_fill") > 0).cast(pl.UInt8).alias("positive_fpm_mid"),
                (pl.col("post_cancel_mid_reversion") > 0).cast(pl.UInt8).alias("positive_reversion_mid"),
            ]
        )
        .group_by("client_id")
        .agg(
            [
                pl.len().cast(pl.UInt32).alias("event_count"),
                pl.col("msci_exceeds_threshold").sum().cast(pl.UInt32).alias("msci_exceedance_count"),
                pl.col("MSCI").max().alias("max_MSCI"),
                pl.col("MSCI").mean().alias("mean_MSCI"),
                pl.col("matched_event").sum().cast(pl.UInt32).alias("matched_event_count"),
                pl.col("WMSCI_event").max().alias("max_WMSCI_event"),
                pl.col("WMSCI_event").mean().alias("mean_WMSCI_event"),
                pl.col("withdrawal_to_fill_ratio").mean().alias("mean_withdrawal_to_fill_ratio"),
                pl.col("positive_fpm_mid").mean().alias("positive_fpm_mid_share"),
                pl.col("positive_reversion_mid").mean().alias("positive_reversion_mid_share"),
                pl.col("execution_price_advantage_vs_posture_mid").mean().alias(
                    "mean_execution_price_advantage_vs_posture_mid"
                ),
                pl.col("SCI").mean().alias("mean_SCI"),
                pl.col("collapse_opposite_side").mean().alias("mean_opposite_collapse"),
                pl.col("collapse_same_side").mean().alias("mean_same_side_collapse"),
                pl.col("matched_deceptive_cancel_fraction_window").mean().alias("mean_matched_cancel_fraction"),
                pl.col("fill_qty").sum().alias("total_fill_qty"),
            ]
        )
        .with_columns(
            [
                (pl.col("msci_exceedance_count") / pl.col("event_count")).alias("mcps_at_threshold"),
                (pl.col("matched_event_count") / pl.col("event_count")).alias("matched_event_share"),
            ]
        )
        .select(list(CLIENT_SESSION_FEATURE_SCHEMA))
        .sort(["matched_event_share", "max_WMSCI_event", "event_count"], descending=[True, True, True])
    )
