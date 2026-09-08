from __future__ import annotations

import polars as pl

ALERT_SCHEMA = {
    "client_id": pl.Utf8,
    "episode_count": pl.UInt32,
    "matched_episode_count": pl.UInt32,
    "matched_episode_share": pl.Float64,
    "strict_episode_count": pl.UInt32,
    "event_count": pl.UInt32,
    "matched_event_count": pl.UInt32,
    "matched_event_share": pl.Float64,
    "max_withdrawal_profile_scale_event": pl.Float64,
    "mean_withdrawal_profile_scale_event": pl.Float64,
    "max_MSCI_resting_profile": pl.Float64,
    "mean_MSCI_resting_profile": pl.Float64,
    "max_WMSCI_event": pl.Float64,
    "mean_WMSCI_event": pl.Float64,
    "max_MSCI": pl.Float64,
    "mean_MSCI": pl.Float64,
    "msci_threshold": pl.Float64,
    "msci_threshold_applicable": pl.Boolean,
    "mcps_at_threshold": pl.Float64,
    "positive_fpm_mid_share": pl.Float64,
    "positive_reversion_mid_share": pl.Float64,
    "mean_execution_price_advantage_vs_posture_mid": pl.Float64,
    "side_symmetry_score": pl.Float64,
    "matched_cancel_event_share": pl.Float64,
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
        "msci_threshold": None,
        "msci_threshold_applicable": True,
        "matched_event_count": 0,
        "matched_event_share": None,
        "max_WMSCI_event": None,
        "mean_WMSCI_event": None,
        "positive_fpm_mid_share": None,
        "positive_reversion_mid_share": None,
    }.items():
        if column not in prepared_risk.columns:
            prepared_risk = prepared_risk.with_columns(pl.lit(default).alias(column))
    canonical_aliases = {
        "max_withdrawal_profile_scale_event": "max_WMSCI_event",
        "mean_withdrawal_profile_scale_event": "mean_WMSCI_event",
        "max_MSCI_resting_profile": "max_MSCI",
        "mean_MSCI_resting_profile": "mean_MSCI",
    }
    missing_canonical = [
        pl.col(legacy).alias(canonical)
        for canonical, legacy in canonical_aliases.items()
        if canonical not in prepared_risk.columns
    ]
    if missing_canonical:
        prepared_risk = prepared_risk.with_columns(missing_canonical)
    prepared_risk = prepared_risk.with_columns(
        pl.col(canonical).alias(legacy)
        for canonical, legacy in canonical_aliases.items()
    )
    joined = prepared_risk.join(legitimacy_features, on="client_id", how="left")
    repeat_support = (
        pl.coalesce("matched_episode_count", "matched_event_count")
        if "matched_episode_count" in joined.columns else pl.col("matched_event_count")
    )
    episode_aware = "episode_count" in joined.columns and "matched_episode_share" in joined.columns
    event_support = pl.coalesce("episode_count", "event_count") if episode_aware else pl.col("event_count")
    share_support = (
        pl.coalesce("matched_episode_share", "matched_event_share")
        if episode_aware
        else pl.col("matched_event_share")
    )
    alerts = (
        joined
        .filter(
            pl.col("msci_threshold_applicable").fill_null(False)
            & (event_support.fill_null(0) >= min_events)
            & (repeat_support.fill_null(0) >= min_events)
            & (share_support.fill_null(0.0) >= min_mcps)
            & (pl.col("max_withdrawal_profile_scale_event").fill_null(0.0) > 0.0)
        )
        .with_columns(pl.lit("human_review").alias("recommended_action"))
        .sort(
            [
                "max_withdrawal_profile_scale_event",
                "mean_withdrawal_profile_scale_event",
                "max_MSCI_resting_profile",
                "mean_MSCI_resting_profile",
                "matched_event_count",
                "event_count",
                "client_id",
            ],
            descending=[True, True, True, True, True, True, False],
            nulls_last=True,
        )
    )
    if alerts.is_empty():
        return empty_alerts()
    for column, dtype in ALERT_SCHEMA.items():
        if column not in alerts.columns:
            alerts = alerts.with_columns(pl.lit(None, dtype=dtype).alias(column))
    return alerts.select(list(ALERT_SCHEMA))


ACTOR_ALERT_SCHEMA = {
    "partition_id": pl.String,
    "event_date": pl.Date,
    "actor_key": pl.Utf8,
    "actor_id": pl.Utf8,
    "identity_level": pl.Utf8,
    "identity_source": pl.Utf8,
    "identity_fallback_flag": pl.Boolean,
    "identity_scope_warning": pl.Utf8,
    "execution_anchor_mode": pl.Utf8,
    **{key: value for key, value in ALERT_SCHEMA.items() if key != "client_id"},
}


def empty_actor_alerts() -> pl.DataFrame:
    return pl.DataFrame(schema=ACTOR_ALERT_SCHEMA)


def build_actor_session_alerts(
    risk_features: pl.DataFrame,
    legitimacy_features: pl.DataFrame,
    *,
    min_events: int,
    min_mcps: float,
) -> pl.DataFrame:
    """Build human-review alerts stratified by actor identity and execution anchor."""

    if risk_features.is_empty():
        return empty_actor_alerts()
    identity_columns = [
        "actor_key",
        "actor_id",
        "identity_level",
        "identity_source",
        "identity_fallback_flag",
        "execution_anchor_mode",
    ]
    missing = [column for column in identity_columns if column not in risk_features.columns]
    if missing:
        raise ValueError(f"missing actor risk feature columns: {missing}")

    rows: list[dict[str, object]] = []
    day_keys = [k for k in ("partition_id", "event_date") if k in risk_features.columns]
    for risk_group in risk_features.partition_by(
        [*day_keys, "actor_key", "execution_anchor_mode"], maintain_order=True
    ):
        identity: dict[str, object] = {}
        for column in identity_columns:
            values = risk_group.get_column(column).drop_nulls().unique().to_list()
            if len(values) != 1:
                raise ValueError(
                    f"{column} must be unique and non-null within each "
                    "(actor_key, execution_anchor_mode) group"
                )
            identity[column] = values[0]

        actor_key = identity["actor_key"]
        anchor_mode = identity["execution_anchor_mode"]
        day_identity = {k: risk_group[k][0] for k in day_keys}
        identity.update(partition_id=day_identity.get("partition_id"), event_date=day_identity.get("event_date"))
        legacy_risk = risk_group.with_columns(pl.col("actor_key").alias("client_id"))
        if {"actor_key", "execution_anchor_mode"}.issubset(legitimacy_features.columns):
            legacy_legitimacy = legitimacy_features.filter(
                (pl.col("actor_key") == actor_key)
                & (pl.col("execution_anchor_mode") == anchor_mode)
            ).with_columns(pl.col("actor_key").alias("client_id"))
            for day_key in day_keys:
                if day_key not in legacy_legitimacy.columns:
                    legacy_legitimacy = pl.DataFrame(schema={"client_id": pl.Utf8})
                    break
                legacy_legitimacy = legacy_legitimacy.filter(pl.col(day_key).eq_missing(day_identity[day_key]))
        else:
            legacy_legitimacy = pl.DataFrame(schema={"client_id": pl.Utf8})
        legacy_alert = build_client_session_alerts(
            legacy_risk,
            legacy_legitimacy,
            min_events=min_events,
            min_mcps=min_mcps,
        )
        for alert_row in legacy_alert.to_dicts():
            alert_row.pop("client_id", None)
            warning = (
                "firm_fallback_may_aggregate_multiple_clients"
                if identity["identity_level"] == "firm"
                else None
            )
            rows.append(
                {
                    **identity,
                    "identity_scope_warning": warning,
                    **alert_row,
                }
            )

    if not rows:
        return empty_actor_alerts()
    return pl.DataFrame(rows, schema=ACTOR_ALERT_SCHEMA).sort(
        [
            "identity_level",
            "execution_anchor_mode",
            "max_WMSCI_event",
            "mean_WMSCI_event",
            "max_MSCI",
            "mean_MSCI",
            "actor_key",
        ],
        descending=[False, False, True, True, True, True, False],
        nulls_last=True,
    )
