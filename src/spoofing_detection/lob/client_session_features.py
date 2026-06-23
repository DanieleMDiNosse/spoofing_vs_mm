from __future__ import annotations

import polars as pl

CLIENT_SESSION_FEATURE_SCHEMA = {
    "client_id": pl.Utf8,
    "event_count": pl.UInt32,
    "msci_exceedance_count": pl.UInt32,
    "mcps_at_threshold": pl.Float64,
    "max_MSCI": pl.Float64,
    "mean_MSCI": pl.Float64,
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
    return (
        executions.with_columns(
            [
                pl.col("client_id").cast(pl.Utf8),
                (pl.col("MSCI") > msci_threshold).cast(pl.UInt8).alias("msci_exceeds_threshold"),
            ]
        )
        .group_by("client_id")
        .agg(
            [
                pl.len().cast(pl.UInt32).alias("event_count"),
                pl.col("msci_exceeds_threshold").sum().cast(pl.UInt32).alias("msci_exceedance_count"),
                pl.col("MSCI").max().alias("max_MSCI"),
                pl.col("MSCI").mean().alias("mean_MSCI"),
                pl.col("SCI").mean().alias("mean_SCI"),
                pl.col("collapse_opposite_side").mean().alias("mean_opposite_collapse"),
                pl.col("collapse_same_side").mean().alias("mean_same_side_collapse"),
                pl.col("matched_deceptive_cancel_fraction_window").mean().alias("mean_matched_cancel_fraction"),
                pl.col("fill_qty").sum().alias("total_fill_qty"),
            ]
        )
        .with_columns((pl.col("msci_exceedance_count") / pl.col("event_count")).alias("mcps_at_threshold"))
        .select(list(CLIENT_SESSION_FEATURE_SCHEMA))
        .sort(["mcps_at_threshold", "max_MSCI", "event_count"], descending=[True, True, True])
    )
