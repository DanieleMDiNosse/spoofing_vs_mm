from __future__ import annotations

import polars as pl

LEGITIMACY_FEATURE_SCHEMA = {
    "client_id": pl.Utf8,
    "event_count": pl.UInt32,
    "bid_event_share": pl.Float64,
    "ask_event_share": pl.Float64,
    "side_symmetry_score": pl.Float64,
    "matched_cancel_event_share": pl.Float64,
    "execution_event_share": pl.Float64,
    "mean_displayed_qty": pl.Float64,
    "total_executed_qty": pl.Float64,
}


def empty_legitimacy_features() -> pl.DataFrame:
    return pl.DataFrame(schema=LEGITIMACY_FEATURE_SCHEMA)


def compute_legitimacy_features(events: pl.DataFrame) -> pl.DataFrame:
    if events.is_empty() or "client_id" not in events.columns:
        return empty_legitimacy_features()
    required = ["client_id", "side", "is_execution_order", "is_matched_deceptive_cancel_order", "displayed_qty", "last_shares"]
    missing = [column for column in required if column not in events.columns]
    if missing:
        raise ValueError(f"missing legitimacy feature columns: {missing}")
    return (
        events.with_columns(
            [
                pl.col("client_id").cast(pl.Utf8),
                (pl.col("side") == "bid").cast(pl.UInt8).alias("is_bid"),
                (pl.col("side") == "ask").cast(pl.UInt8).alias("is_ask"),
                pl.col("is_execution_order").fill_null(False).cast(pl.UInt8).alias("execution_flag"),
                pl.col("is_matched_deceptive_cancel_order").fill_null(False).cast(pl.UInt8).alias("matched_cancel_flag"),
            ]
        )
        .group_by("client_id")
        .agg(
            [
                pl.len().cast(pl.UInt32).alias("event_count"),
                pl.col("is_bid").mean().alias("bid_event_share"),
                pl.col("is_ask").mean().alias("ask_event_share"),
                pl.col("matched_cancel_flag").mean().alias("matched_cancel_event_share"),
                pl.col("execution_flag").mean().alias("execution_event_share"),
                pl.col("displayed_qty").mean().alias("mean_displayed_qty"),
                pl.col("last_shares").fill_null(0).sum().alias("total_executed_qty"),
            ]
        )
        .with_columns((1.0 - (pl.col("bid_event_share") - pl.col("ask_event_share")).abs()).alias("side_symmetry_score"))
        .select(list(LEGITIMACY_FEATURE_SCHEMA))
        .sort(["side_symmetry_score", "event_count"], descending=[True, True])
    )
