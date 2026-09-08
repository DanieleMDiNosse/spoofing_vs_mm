"""Broad post-execution cancellation statistic used to motivate sequence reconstruction."""

from __future__ import annotations

import bisect
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Any

import polars as pl

from .config import LOBConfig
from .normalize import normalize_event
from .panel import (
    _apply_event,
    _fill_group_key,
    _flush_pending_aggressive_residuals,
    _partition_id,
    sort_events,
)
from .spoofing_metrics import _direct_cancel_row, choose_event_timestamp

DIRECT_CANCELLATION_SCHEMA: dict[str, pl.DataType] = {
    "partition_id": pl.String,
    "sort_index": pl.Int64,
    "event_ts": pl.Datetime("us"),
    "actor_key": pl.String,
    "actor_id": pl.String,
    "identity_level": pl.String,
    "identity_source": pl.String,
    "identity_fallback_flag": pl.Boolean,
    "client_original_id": pl.String,
    "firm_id": pl.String,
    "side": pl.String,
    "ORDERID": pl.String,
    "visible_qty_pre_cancel": pl.Float64,
}

LINK_SCHEMA: dict[str, pl.DataType] = {
    "execution_cluster_id": pl.String,
    "partition_id": pl.String,
    "actor_key": pl.String,
    "actor_id": pl.String,
    "identity_level": pl.String,
    "identity_source": pl.String,
    "identity_fallback_flag": pl.Boolean,
    "execution_anchor_mode": pl.String,
    "execution_side": pl.String,
    "opposite_side": pl.String,
    "cluster_end_ts": pl.Datetime("us"),
    "cluster_last_sort_index": pl.Int64,
    "cancel_order_id": pl.String,
    "cancel_event_ts": pl.Datetime("us"),
    "cancel_sort_index": pl.Int64,
    "cancel_delay_seconds": pl.Float64,
}

CLUSTER_INDICATOR_SCHEMA: dict[str, pl.DataType] = {
    "execution_cluster_id": pl.String,
    "partition_id": pl.String,
    "actor_key": pl.String,
    "actor_id": pl.String,
    "identity_level": pl.String,
    "identity_source": pl.String,
    "identity_fallback_flag": pl.Boolean,
    "execution_anchor_mode": pl.String,
    "execution_side": pl.String,
    "opposite_side": pl.String,
    "cluster_end_ts": pl.Datetime("us"),
    "cluster_last_sort_index": pl.Int64,
    "qualifying_cancellation_count": pl.UInt32,
    "has_opposite_side_cancellation_within_2s": pl.Boolean,
    "first_qualifying_cancel_order_id": pl.String,
    "first_qualifying_cancel_ts": pl.Datetime("us"),
    "first_qualifying_cancel_sort_index": pl.Int64,
    "first_qualifying_cancel_delay_seconds": pl.Float64,
}

ACTOR_RATE_SCHEMA: dict[str, pl.DataType] = {
    "instrument": pl.String,
    "actor_key": pl.String,
    "actor_id": pl.String,
    "identity_level": pl.String,
    "identity_scope": pl.String,
    "identity_source": pl.String,
    "identity_fallback_flag": pl.Boolean,
    "execution_cluster_count": pl.UInt32,
    "qualifying_execution_cluster_count": pl.UInt32,
    "cancellation_rate_2s": pl.Float64,
    "passive_execution_cluster_count": pl.UInt32,
    "aggressive_execution_cluster_count": pl.UInt32,
    "rank_within_instrument": pl.UInt32,
}


def _coerce_schema(frame: pl.DataFrame, schema: Mapping[str, pl.DataType]) -> pl.DataFrame:
    if frame.is_empty():
        return pl.DataFrame(schema=schema)
    missing = [
        pl.lit(None).cast(dtype).alias(column)
        for column, dtype in schema.items()
        if column not in frame.columns
    ]
    if missing:
        frame = frame.with_columns(missing)
    return frame.select(pl.col(column).cast(dtype, strict=False) for column, dtype in schema.items())


def reconstruct_direct_cancellations(
    raw_events: pl.DataFrame,
    *,
    max_rows: int | None = None,
) -> pl.DataFrame:
    """Replay the existing LOB lifecycle and retain visible cancellation records.

    The replay uses the repository's canonical event sort, normalization, trading
    partition, and active-order mutation logic. Identity and side are read from
    the active order immediately before cancellation, not inferred from a later
    detector candidate set.
    """
    sorted_events = sort_events(raw_events)
    if max_rows is not None:
        sorted_events = sorted_events.head(max_rows)
    config = LOBConfig(top_n=1, snapshot_mode="none")
    normalized = (
        normalize_event(raw_row, sort_index=index, config=config)
        for index, raw_row in enumerate(sorted_events.iter_rows(named=True), start=1)
    )

    active_orders: dict[str, Any] = {}
    pending_aggressive_residuals: dict[str, tuple[dict[str, Any], tuple[Any, ...] | None]] = {}
    non_resting_order_ids: set[str] = set()
    current_partition_id: str | None = None
    rows: list[dict[str, Any]] = []

    event = next(normalized, None)
    while event is not None:
        next_event = next(normalized, None)
        partition_id = _partition_id(event)
        if current_partition_id is None:
            current_partition_id = partition_id
        elif partition_id != current_partition_id:
            _flush_pending_aggressive_residuals(
                active_orders,
                pending_aggressive_residuals,
                keep_group=None,
            )
            active_orders = {}
            pending_aggressive_residuals = {}
            non_resting_order_ids = set()
            current_partition_id = partition_id

        cancellation = _direct_cancel_row(
            event,
            active_orders,
            event_ts=choose_event_timestamp(event),
            partition_id=partition_id,
        )
        if cancellation is not None:
            rows.append(cancellation)

        _apply_event(
            active_orders,
            event,
            pending_aggressive_residuals=pending_aggressive_residuals,
            non_resting_order_ids=non_resting_order_ids,
        )
        _flush_pending_aggressive_residuals(
            active_orders,
            pending_aggressive_residuals,
            keep_group=_fill_group_key(next_event) if next_event is not None else None,
        )
        event = next_event

    return _coerce_schema(
        pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(),
        DIRECT_CANCELLATION_SCHEMA,
    )


def _require_columns(frame: pl.DataFrame, required: set[str], *, name: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {', '.join(missing)}")


def _validate_execution_anchor_modes(frame: pl.DataFrame, *, name: str) -> None:
    observed = set(frame.get_column("execution_anchor_mode").drop_nulls().unique().to_list())
    invalid = sorted(observed - {"passive", "aggressive"})
    if invalid:
        raise ValueError(f"{name} contains unsupported execution_anchor_mode values: {invalid}")


def match_post_execution_opposite_side_cancellations(
    executions: pl.DataFrame,
    cancellations: pl.DataFrame,
    *,
    window_seconds: float = 2.0,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Match every cluster to later same-actor opposite-side cancellations.

    A link requires the same trading partition and canonical actor, the side
    opposite to the cluster's execution side, a strictly later event timestamp,
    a strictly larger canonical source index, and delay at most ``window_seconds``.
    No candidate-age, rank, quantity, price-response, WMSCI, MSCI, or strict-gate
    condition is applied. One cancellation may therefore qualify for more than
    one nearby cluster; the actor numerator remains a cluster-level indicator.
    """
    if window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    _require_columns(
        executions,
        {
            "execution_cluster_id",
            "partition_id",
            "actor_key",
            "actor_id",
            "identity_level",
            "identity_source",
            "identity_fallback_flag",
            "execution_anchor_mode",
            "execution_side",
            "deceptive_side",
            "cluster_end_ts",
            "cluster_last_sort_index",
        },
        name="executions",
    )
    _require_columns(
        cancellations,
        {"partition_id", "actor_key", "side", "event_ts", "sort_index", "ORDERID"},
        name="cancellations",
    )
    if executions.get_column("execution_cluster_id").is_duplicated().any():
        raise ValueError("executions contains duplicate execution_cluster_id values")
    _validate_execution_anchor_modes(executions, name="executions")

    grouped_cancellations: dict[
        tuple[str | None, str, str], tuple[list[datetime], list[dict[str, Any]]]
    ] = {}
    raw_groups: dict[tuple[str | None, str, str], list[dict[str, Any]]] = defaultdict(list)
    for cancellation in cancellations.iter_rows(named=True):
        raw_groups[
            (
                cancellation.get("partition_id"),
                str(cancellation.get("actor_key")),
                str(cancellation.get("side")),
            )
        ].append(cancellation)
    for key, rows in raw_groups.items():
        rows.sort(
            key=lambda row: (
                row["event_ts"],
                int(row["sort_index"]),
                str(row.get("ORDERID") or ""),
            )
        )
        grouped_cancellations[key] = ([row["event_ts"] for row in rows], rows)

    cluster_rows: list[dict[str, Any]] = []
    link_rows: list[dict[str, Any]] = []
    window = timedelta(seconds=window_seconds)
    for execution in executions.iter_rows(named=True):
        end_ts = execution.get("cluster_end_ts")
        last_sort_index = execution.get("cluster_last_sort_index")
        if end_ts is None or last_sort_index is None:
            raise ValueError("every execution cluster must have an end timestamp and final source index")
        execution_side = execution.get("execution_side")
        opposite_side = execution.get("deceptive_side")
        expected_opposite = "ask" if execution_side == "bid" else "bid" if execution_side == "ask" else None
        if opposite_side != expected_opposite:
            raise ValueError(
                f"cluster {execution['execution_cluster_id']} has inconsistent execution/opposite sides"
            )

        times, candidates = grouped_cancellations.get(
            (
                execution.get("partition_id"),
                str(execution.get("actor_key")),
                str(opposite_side),
            ),
            ([], []),
        )
        upper_ts = end_ts + window
        lower = bisect.bisect_right(times, end_ts)
        upper = bisect.bisect_right(times, upper_ts)
        qualifying = [
            cancellation
            for cancellation in candidates[lower:upper]
            if int(cancellation["sort_index"]) > int(last_sort_index)
        ]
        qualifying.sort(
            key=lambda row: (
                row["event_ts"],
                int(row["sort_index"]),
                str(row.get("ORDERID") or ""),
            )
        )
        for cancellation in qualifying:
            link_rows.append(
                {
                    "execution_cluster_id": execution.get("execution_cluster_id"),
                    "partition_id": execution.get("partition_id"),
                    "actor_key": execution.get("actor_key"),
                    "actor_id": execution.get("actor_id"),
                    "identity_level": execution.get("identity_level"),
                    "identity_source": execution.get("identity_source"),
                    "identity_fallback_flag": execution.get("identity_fallback_flag"),
                    "execution_anchor_mode": execution.get("execution_anchor_mode"),
                    "execution_side": execution_side,
                    "opposite_side": opposite_side,
                    "cluster_end_ts": end_ts,
                    "cluster_last_sort_index": last_sort_index,
                    "cancel_order_id": cancellation.get("ORDERID"),
                    "cancel_event_ts": cancellation.get("event_ts"),
                    "cancel_sort_index": cancellation.get("sort_index"),
                    "cancel_delay_seconds": (
                        cancellation["event_ts"] - end_ts
                    ).total_seconds(),
                }
            )
        first = qualifying[0] if qualifying else None
        cluster_rows.append(
            {
                "execution_cluster_id": execution.get("execution_cluster_id"),
                "partition_id": execution.get("partition_id"),
                "actor_key": execution.get("actor_key"),
                "actor_id": execution.get("actor_id"),
                "identity_level": execution.get("identity_level"),
                "identity_source": execution.get("identity_source"),
                "identity_fallback_flag": execution.get("identity_fallback_flag"),
                "execution_anchor_mode": execution.get("execution_anchor_mode"),
                "execution_side": execution_side,
                "opposite_side": opposite_side,
                "cluster_end_ts": end_ts,
                "cluster_last_sort_index": last_sort_index,
                "qualifying_cancellation_count": len(qualifying),
                "has_opposite_side_cancellation_within_2s": bool(qualifying),
                "first_qualifying_cancel_order_id": first.get("ORDERID") if first else None,
                "first_qualifying_cancel_ts": first.get("event_ts") if first else None,
                "first_qualifying_cancel_sort_index": first.get("sort_index") if first else None,
                "first_qualifying_cancel_delay_seconds": (
                    (first["event_ts"] - end_ts).total_seconds() if first else None
                ),
            }
        )

    clusters = _coerce_schema(
        pl.DataFrame(cluster_rows, infer_schema_length=None) if cluster_rows else pl.DataFrame(),
        CLUSTER_INDICATOR_SCHEMA,
    )
    links = _coerce_schema(
        pl.DataFrame(link_rows, infer_schema_length=None) if link_rows else pl.DataFrame(),
        LINK_SCHEMA,
    )
    return clusters, links


def aggregate_actor_cancellation_rates(
    cluster_indicators: pl.DataFrame,
    *,
    instrument: str,
) -> pl.DataFrame:
    """Aggregate the binary cluster outcome and rank actors within one instrument."""
    if not instrument.strip():
        raise ValueError("instrument must be non-empty")
    _require_columns(
        cluster_indicators,
        {
            "execution_cluster_id",
            "actor_key",
            "actor_id",
            "identity_level",
            "identity_source",
            "identity_fallback_flag",
            "execution_anchor_mode",
            "has_opposite_side_cancellation_within_2s",
        },
        name="cluster_indicators",
    )
    if cluster_indicators.is_empty():
        return pl.DataFrame(schema=ACTOR_RATE_SCHEMA)
    if cluster_indicators.get_column("execution_cluster_id").is_duplicated().any():
        raise ValueError("cluster_indicators contains duplicate execution_cluster_id values")
    _validate_execution_anchor_modes(cluster_indicators, name="cluster_indicators")

    identity_consistency = cluster_indicators.group_by("actor_key").agg(
        pl.col("actor_id").n_unique().alias("actor_id_count"),
        pl.col("identity_level").n_unique().alias("identity_level_count"),
        pl.col("identity_source").n_unique().alias("identity_source_count"),
        pl.col("identity_fallback_flag").n_unique().alias("fallback_count"),
    )
    if identity_consistency.select(
        pl.any_horizontal(pl.exclude("actor_key") != 1).any()
    ).item():
        raise ValueError("canonical actor fields are inconsistent within actor_key")

    grouped = (
        cluster_indicators.group_by(
            [
                "actor_key",
                "actor_id",
                "identity_level",
                "identity_source",
                "identity_fallback_flag",
            ]
        )
        .agg(
            pl.len().cast(pl.UInt32).alias("execution_cluster_count"),
            pl.col("has_opposite_side_cancellation_within_2s")
            .sum()
            .cast(pl.UInt32)
            .alias("qualifying_execution_cluster_count"),
            (pl.col("execution_anchor_mode") == "passive")
            .sum()
            .cast(pl.UInt32)
            .alias("passive_execution_cluster_count"),
            (pl.col("execution_anchor_mode") == "aggressive")
            .sum()
            .cast(pl.UInt32)
            .alias("aggressive_execution_cluster_count"),
        )
        .with_columns(
            pl.lit(instrument).alias("instrument"),
            pl.when(pl.col("identity_level") == "client_original")
            .then(pl.lit("client-level"))
            .when(pl.col("identity_level") == "firm")
            .then(pl.lit("firm-fallback"))
            .otherwise(pl.lit("unknown"))
            .alias("identity_scope"),
            (
                pl.col("qualifying_execution_cluster_count")
                / pl.col("execution_cluster_count")
            ).alias("cancellation_rate_2s"),
        )
        .sort(
            ["cancellation_rate_2s", "execution_cluster_count", "actor_key"],
            descending=[True, True, False],
        )
        .with_row_index("rank_within_instrument", offset=1)
        .with_columns(pl.col("rank_within_instrument").cast(pl.UInt32))
    )
    return _coerce_schema(grouped, ACTOR_RATE_SCHEMA)
