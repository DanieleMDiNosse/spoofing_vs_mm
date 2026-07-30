from __future__ import annotations

import math
from collections.abc import Mapping

import polars as pl

CLIENT_SESSION_FEATURE_SCHEMA = {
    "client_id": pl.Utf8,
    "event_count": pl.UInt32,
    "raw_fill_message_count": pl.UInt32,
    "msci_threshold": pl.Float64,
    "msci_threshold_applicable": pl.Boolean,
    "msci_exceedance_count": pl.UInt32,
    "mcps_at_threshold": pl.Float64,
    "max_MSCI": pl.Float64,
    "mean_MSCI": pl.Float64,
    "max_MSCI_resting_profile": pl.Float64,
    "mean_MSCI_resting_profile": pl.Float64,
    "matched_event_count": pl.UInt32,
    "matched_event_share": pl.Float64,
    "max_WMSCI_event": pl.Float64,
    "mean_WMSCI_event": pl.Float64,
    "max_withdrawal_profile_scale_event": pl.Float64,
    "mean_withdrawal_profile_scale_event": pl.Float64,
    "mean_withdrawal_to_fill_ratio": pl.Float64,
    "mean_withdrawal_to_execution_ratio": pl.Float64,
    "positive_fpm_mid_share": pl.Float64,
    "positive_reversion_mid_share": pl.Float64,
    "mean_execution_price_advantage_vs_posture_mid": pl.Float64,
    "mean_SCI": pl.Float64,
    "mean_opposite_collapse": pl.Float64,
    "mean_same_side_collapse": pl.Float64,
    "mean_matched_cancel_fraction": pl.Float64,
    "total_fill_qty": pl.Float64,
    "total_execution_quantity": pl.Float64,
}


def empty_client_session_features() -> pl.DataFrame:
    return pl.DataFrame(schema=CLIENT_SESSION_FEATURE_SCHEMA)


def compute_client_session_features(
    executions: pl.DataFrame,
    *,
    msci_threshold: float | None,
) -> pl.DataFrame:
    if executions.is_empty() or "client_id" not in executions.columns:
        return empty_client_session_features()
    threshold_applicable = msci_threshold is not None
    threshold_value: float | None = None
    if threshold_applicable:
        try:
            threshold_value = float(msci_threshold)
        except (TypeError, ValueError) as exc:
            raise ValueError("msci_threshold must be finite or null") from exc
        if not math.isfinite(threshold_value):
            raise ValueError("msci_threshold must be finite or null")
    msci_exceeds_threshold = (
        (pl.col("MSCI_resting_profile") > threshold_value).cast(pl.UInt8)
        if threshold_applicable
        else pl.lit(None).cast(pl.UInt8)
    ).alias("msci_exceeds_threshold")
    msci_exceedance_count = (
        pl.col("msci_exceeds_threshold").sum().cast(pl.UInt32)
        if threshold_applicable
        else pl.lit(None).cast(pl.UInt32)
    ).alias("msci_exceedance_count")
    mcps_at_threshold = (
        pl.col("msci_exceedance_count") / pl.col("event_count")
        if threshold_applicable
        else pl.lit(None).cast(pl.Float64)
    ).alias("mcps_at_threshold")
    prepared = executions
    canonical_aliases = {
        "MSCI_resting_profile": "MSCI",
        "withdrawal_profile_scale_event": "WMSCI_event",
        "withdrawal_to_execution_ratio": "withdrawal_to_fill_ratio",
        "execution_quantity": "fill_qty",
    }
    for canonical, legacy in canonical_aliases.items():
        if canonical not in prepared.columns and legacy in prepared.columns:
            prepared = prepared.with_columns(pl.col(legacy).alias(canonical))
    required = [
        "client_id",
        "MSCI_resting_profile",
        "SCI",
        "collapse_opposite_side",
        "collapse_same_side",
        "matched_deceptive_cancel_fraction_window",
        "execution_quantity",
    ]
    missing = [column for column in required if column not in prepared.columns]
    if missing:
        raise ValueError(f"missing execution metric columns: {missing}")
    if "execution_cluster_id" in executions.columns:
        cluster_ids = executions.filter(pl.col("execution_cluster_id").is_not_null()).get_column(
            "execution_cluster_id"
        )
        if cluster_ids.is_duplicated().any():
            raise ValueError("execution_metrics must contain at most one row per execution_cluster_id")
    optional_defaults = {
        "child_fill_count": 1,
        "has_matched_deceptive_cancel_window": False,
        "withdrawal_profile_scale_event": None,
        "withdrawal_to_execution_ratio": None,
        "favorable_mid_move_pre_fill": None,
        "post_cancel_mid_reversion": None,
        "execution_price_advantage_vs_posture_mid": None,
    }
    for column, default in optional_defaults.items():
        if column not in prepared.columns:
            prepared = prepared.with_columns(pl.lit(default).alias(column))
    return (
        prepared.with_columns(
            [
                pl.col("client_id").cast(pl.Utf8),
                msci_exceeds_threshold,
                pl.col("has_matched_deceptive_cancel_window").fill_null(False).cast(pl.UInt8).alias("matched_event"),
                (pl.col("favorable_mid_move_pre_fill") > 0).cast(pl.UInt8).alias("positive_fpm_mid"),
                (pl.col("post_cancel_mid_reversion") > 0).cast(pl.UInt8).alias("positive_reversion_mid"),
            ]
        )
        .with_columns(
            [
                pl.when(pl.col("matched_event") == 1)
                .then(pl.col("withdrawal_profile_scale_event"))
                .otherwise(None)
                .alias("matched_withdrawal_profile_scale_event"),
                pl.when(pl.col("matched_event") == 1)
                .then(pl.col("withdrawal_to_execution_ratio"))
                .otherwise(None)
                .alias("matched_withdrawal_to_execution_ratio"),
                pl.when(pl.col("matched_event") == 1)
                .then(pl.col("positive_fpm_mid"))
                .otherwise(None)
                .alias("matched_positive_fpm_mid"),
                pl.when(pl.col("matched_event") == 1)
                .then(pl.col("positive_reversion_mid"))
                .otherwise(None)
                .alias("matched_positive_reversion_mid"),
                pl.when(pl.col("matched_event") == 1)
                .then(pl.col("execution_price_advantage_vs_posture_mid"))
                .otherwise(None)
                .alias("matched_execution_price_advantage_vs_posture_mid"),
            ]
        )
        .group_by("client_id")
        .agg(
            [
                pl.len().cast(pl.UInt32).alias("event_count"),
                pl.col("child_fill_count").fill_null(1).sum().cast(pl.UInt32).alias("raw_fill_message_count"),
                msci_exceedance_count,
                pl.col("MSCI_resting_profile").max().alias("max_MSCI"),
                pl.col("MSCI_resting_profile").mean().alias("mean_MSCI"),
                pl.col("MSCI_resting_profile").max().alias("max_MSCI_resting_profile"),
                pl.col("MSCI_resting_profile").mean().alias("mean_MSCI_resting_profile"),
                pl.col("matched_event").sum().cast(pl.UInt32).alias("matched_event_count"),
                pl.col("withdrawal_profile_scale_event").max().alias("max_WMSCI_event"),
                pl.col("matched_withdrawal_profile_scale_event").mean().alias("mean_WMSCI_event"),
                pl.col("withdrawal_profile_scale_event")
                .max()
                .alias("max_withdrawal_profile_scale_event"),
                pl.col("matched_withdrawal_profile_scale_event")
                .mean()
                .alias("mean_withdrawal_profile_scale_event"),
                pl.col("matched_withdrawal_to_execution_ratio")
                .mean()
                .alias("mean_withdrawal_to_fill_ratio"),
                pl.col("matched_withdrawal_to_execution_ratio")
                .mean()
                .alias("mean_withdrawal_to_execution_ratio"),
                pl.col("matched_positive_fpm_mid").mean().alias("positive_fpm_mid_share"),
                pl.col("matched_positive_reversion_mid").mean().alias("positive_reversion_mid_share"),
                pl.col("matched_execution_price_advantage_vs_posture_mid").mean().alias(
                    "mean_execution_price_advantage_vs_posture_mid"
                ),
                pl.col("SCI").mean().alias("mean_SCI"),
                pl.col("collapse_opposite_side").mean().alias("mean_opposite_collapse"),
                pl.col("collapse_same_side").mean().alias("mean_same_side_collapse"),
                pl.col("matched_deceptive_cancel_fraction_window").mean().alias("mean_matched_cancel_fraction"),
                pl.col("execution_quantity").sum().alias("total_fill_qty"),
                pl.col("execution_quantity").sum().alias("total_execution_quantity"),
            ]
        )
        .with_columns(
            [
                pl.lit(threshold_value).cast(pl.Float64).alias("msci_threshold"),
                pl.lit(threshold_applicable).alias("msci_threshold_applicable"),
                mcps_at_threshold,
                (pl.col("matched_event_count") / pl.col("event_count")).alias("matched_event_share"),
            ]
        )
        .select(list(CLIENT_SESSION_FEATURE_SCHEMA))
        .sort(
            [
                "max_WMSCI_event",
                "mean_WMSCI_event",
                "max_MSCI",
                "mean_MSCI",
                "matched_event_count",
                "event_count",
                "client_id",
            ],
            descending=[True, True, True, True, True, True, False],
            nulls_last=True,
        )
    )


ACTOR_SESSION_FEATURE_SCHEMA = {
    "actor_key": pl.Utf8,
    "actor_id": pl.Utf8,
    "identity_level": pl.Utf8,
    "identity_source": pl.Utf8,
    "identity_fallback_flag": pl.Boolean,
    "execution_anchor_mode": pl.Utf8,
    **{key: value for key, value in CLIENT_SESSION_FEATURE_SCHEMA.items() if key != "client_id"},
}


def empty_actor_session_features() -> pl.DataFrame:
    return pl.DataFrame(schema=ACTOR_SESSION_FEATURE_SCHEMA)


def filter_attributable_actor_rows(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep rows carrying a canonical client- or firm-level actor key."""

    if frame.is_empty() or "actor_key" not in frame.columns:
        return frame
    actor_key = pl.col("actor_key").cast(pl.Utf8, strict=False).fill_null("").str.strip_chars()
    return frame.filter(
        actor_key.str.starts_with("client_original:") | actor_key.str.starts_with("firm:")
    )


def compute_actor_session_features(
    executions: pl.DataFrame,
    *,
    msci_threshold: float | Mapping[str, float | None],
) -> pl.DataFrame:
    """Aggregate execution metrics without pooling actors or execution anchors."""

    if executions.is_empty():
        return empty_actor_session_features()
    identity_columns = [
        "actor_key",
        "actor_id",
        "identity_level",
        "identity_source",
        "identity_fallback_flag",
        "execution_anchor_mode",
    ]
    missing = [column for column in identity_columns if column not in executions.columns]
    if missing:
        raise ValueError(f"missing actor execution metric columns: {missing}")
    executions = filter_attributable_actor_rows(executions)
    if executions.is_empty():
        return empty_actor_session_features()
    if "execution_cluster_id" in executions.columns:
        duplicate_keys = executions.select("execution_cluster_id", "execution_anchor_mode")
        if duplicate_keys.is_duplicated().any():
            raise ValueError(
                "execution_metrics must contain at most one row per "
                "(execution_cluster_id, execution_anchor_mode)"
            )

    threshold_by_anchor: dict[str, float | None] | None = None
    if isinstance(msci_threshold, Mapping):
        threshold_by_anchor = {
            str(anchor).strip().lower(): value
            for anchor, value in msci_threshold.items()
        }
        unknown_anchors = sorted(set(threshold_by_anchor) - {"passive", "aggressive"})
        if unknown_anchors:
            raise ValueError(
                "msci_threshold contains unknown execution anchor mode(s): "
                + ", ".join(unknown_anchors)
            )

    rows: list[dict[str, object]] = []
    for group in executions.partition_by(["actor_key", "execution_anchor_mode"], maintain_order=True):
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
        anchor_mode = str(identity["execution_anchor_mode"])
        group_threshold = (
            threshold_by_anchor.get(anchor_mode)
            if threshold_by_anchor is not None
            else msci_threshold
        )
        metrics = compute_client_session_features(
            legacy_input,
            msci_threshold=group_threshold,
        )
        metric_row = metrics.row(0, named=True)
        metric_row.pop("client_id", None)
        rows.append({**identity, **metric_row})

    return pl.DataFrame(rows, schema=ACTOR_SESSION_FEATURE_SCHEMA).sort(
        [
            "identity_level",
            "execution_anchor_mode",
            "max_withdrawal_profile_scale_event",
            "mean_withdrawal_profile_scale_event",
            "max_MSCI_resting_profile",
            "mean_MSCI_resting_profile",
            "actor_key",
        ],
        descending=[False, False, True, True, True, True, False],
        nulls_last=True,
    )
