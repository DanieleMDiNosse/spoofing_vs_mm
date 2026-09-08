from datetime import datetime

import polars as pl

from spoofing_detection.lob.alert_objects import build_actor_session_alerts
from spoofing_detection.lob.client_session_features import compute_actor_session_features


def executions():
    return pl.DataFrame([dict(partition_id=partition, event_ts=datetime(2024,1,day,12),
        actor_key='client_original:C',actor_id='C',identity_level='client_original',
        identity_source='NMSC_ORIGINALCLIENTIDSHORTCODE',identity_fallback_flag=False,
        execution_anchor_mode='passive',execution_cluster_id=f'E{i}',episode_id=f'EP{day}',
        episode_has_matched_withdrawal=True,spoofing_compatible_episode=True,
        MSCI_resting_profile=.4,SCI=.2,collapse_opposite_side=.5,collapse_same_side=.1,
        matched_deceptive_cancel_fraction_window=1.,execution_quantity=10.,
        has_matched_deceptive_cancel_window=True,withdrawal_profile_scale_event=2.)
        for i,partition,day in [(1,'P1',1),(2,'P1',1),(3,'P1',1),(4,'P2',2)]])


def test_actor_day_features_preserve_partition_and_count_unique_episodes():
    features=compute_actor_session_features(executions(),msci_threshold=.1)
    assert features.height==2
    first=features.filter(pl.col('partition_id')=='P1').row(0,named=True)
    assert first['event_count']==3
    assert first['episode_count']==1
    assert first['matched_episode_count']==1
    assert first['matched_episode_share']==1.0
    assert first['strict_episode_count']==1


def test_same_partition_different_dates_still_separate():
    e=executions().with_columns(pl.lit('BAD_PARTITION').alias('partition_id'))
    assert compute_actor_session_features(e,msci_threshold=.1).height==2


def test_three_cluster_fragments_of_one_episode_do_not_trigger_repetition_alert():
    risk=compute_actor_session_features(executions(),msci_threshold=.1)
    alerts=build_actor_session_alerts(risk,pl.DataFrame(),min_events=3,min_mcps=0.)
    assert alerts.is_empty()


def test_alert_preserves_actor_day_and_episode_support():
    risk=compute_actor_session_features(executions(),msci_threshold=.1)
    alerts=build_actor_session_alerts(risk,pl.DataFrame(),min_events=1,min_mcps=0.)
    assert alerts.height==2
    assert alerts['partition_id'].n_unique()==2
    assert alerts['matched_episode_count'].to_list()==[1,1]


def test_alert_uses_episode_share_not_cluster_share():
    rows = executions().filter(pl.col("partition_id") == "P1")
    unmatched = rows.head(1).with_columns(
        pl.lit("E-unmatched").alias("execution_cluster_id"),
        pl.lit("EP-unmatched").alias("episode_id"),
        pl.lit(False).alias("episode_has_matched_withdrawal"),
        pl.lit(False).alias("spoofing_compatible_episode"),
        pl.lit(False).alias("has_matched_deceptive_cancel_window"),
    )
    risk = compute_actor_session_features(
        pl.concat([rows, unmatched], how="vertical"), msci_threshold=.1
    )

    assert risk.item(0, "matched_event_share") == .75
    assert risk.item(0, "matched_episode_share") == .5
    assert build_actor_session_alerts(
        risk, pl.DataFrame(), min_events=1, min_mcps=.6
    ).is_empty()
