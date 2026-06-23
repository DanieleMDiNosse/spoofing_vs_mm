from __future__ import annotations

import polars as pl


def add_time_shift_placebo(events: pl.DataFrame, *, shift_events: int) -> pl.DataFrame:
    if "sort_index" not in events.columns:
        raise ValueError("events must contain sort_index for time-shift placebo")
    return events.with_columns(
        [
            pl.lit("time_shift").alias("placebo_type"),
            (pl.col("sort_index") + shift_events).alias("placebo_sort_index"),
        ]
    )


def add_wrong_side_placebo(events: pl.DataFrame) -> pl.DataFrame:
    if "deceptive_side" not in events.columns:
        raise ValueError("events must contain deceptive_side for wrong-side placebo")
    return events.with_columns(
        [
            pl.lit("wrong_side").alias("placebo_type"),
            pl.when(pl.col("deceptive_side") == "bid")
            .then(pl.lit("ask"))
            .when(pl.col("deceptive_side") == "ask")
            .then(pl.lit("bid"))
            .otherwise(None)
            .alias("placebo_deceptive_side"),
        ]
    )
