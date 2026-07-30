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


ACTOR_LEGITIMACY_FEATURE_SCHEMA = {
    "actor_key": pl.Utf8,
    "actor_id": pl.Utf8,
    "identity_level": pl.Utf8,
    "identity_source": pl.Utf8,
    "identity_fallback_flag": pl.Boolean,
    "execution_anchor_mode": pl.Utf8,
    **{key: value for key, value in LEGITIMACY_FEATURE_SCHEMA.items() if key != "client_id"},
}


def empty_actor_legitimacy_features() -> pl.DataFrame:
    return pl.DataFrame(schema=ACTOR_LEGITIMACY_FEATURE_SCHEMA)


def compute_actor_legitimacy_features(events: pl.DataFrame) -> pl.DataFrame:
    """Compute descriptive legitimacy features by actor and execution anchor."""

    if events.is_empty():
        return empty_actor_legitimacy_features()
    identity_columns = [
        "actor_key",
        "actor_id",
        "identity_level",
        "identity_source",
        "identity_fallback_flag",
        "execution_anchor_mode",
    ]
    missing = [column for column in identity_columns if column not in events.columns]
    if missing:
        raise ValueError(f"missing actor legitimacy feature columns: {missing}")
    actor_key = pl.col("actor_key").cast(pl.Utf8, strict=False).fill_null("").str.strip_chars()
    events = events.filter(
        actor_key.str.starts_with("client_original:") | actor_key.str.starts_with("firm:")
    )
    if events.is_empty():
        return empty_actor_legitimacy_features()

    rows: list[dict[str, object]] = []
    for group in events.partition_by(["actor_key", "execution_anchor_mode"], maintain_order=True):
        identity: dict[str, object] = {}
        for column in identity_columns:
            values = group.get_column(column).drop_nulls().unique().to_list()
            if len(values) != 1:
                raise ValueError(
                    f"{column} must be unique and non-null within each "
                    "(actor_key, execution_anchor_mode) group"
                )
            identity[column] = values[0]
        legacy_input = group.with_columns(pl.col("actor_key").alias("client_id"))
        metrics = compute_legitimacy_features(legacy_input)
        metric_row = metrics.row(0, named=True)
        metric_row.pop("client_id", None)
        rows.append({**identity, **metric_row})

    return pl.DataFrame(rows, schema=ACTOR_LEGITIMACY_FEATURE_SCHEMA).sort(
        ["identity_level", "execution_anchor_mode", "side_symmetry_score", "event_count", "actor_key"],
        descending=[False, False, True, True, False],
    )
