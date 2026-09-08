from datetime import datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal


def inputs(quantity=60.):
    t = datetime(2024, 1, 1, 12)
    identity = dict(actor_key='client_original:C', actor_id='C', identity_level='client_original',
                    identity_source='NMSC_ORIGINALCLIENTIDSHORTCODE', identity_fallback_flag=False)
    executions = [dict(**identity, partition_id='P', execution_cluster_id=f'E{i}',
        execution_anchor_mode=anchor, execution_side='ask', deceptive_side='bid',
        cluster_first_sort_index=i*10, cluster_last_sort_index=i*10,
        cluster_start_ts=t+timedelta(seconds=i), cluster_end_ts=t+timedelta(seconds=i),
        execution_quantity=quantity, execution_vwap=100.+i,
        favorable_mid_move_pre_fill=.1)
        for i, anchor in [(1,'passive'), (2,'aggressive')]]
    candidates = [dict(**identity, partition_id='P', execution_cluster_id=e['execution_cluster_id'],
        execution_anchor_mode=e['execution_anchor_mode'], execution_side='ask', deceptive_side='bid',
        deceptive_order_id='D', deceptive_order_first_seen_sort_index=1,
        deceptive_order_first_seen_ts=t, deceptive_order_visible_qty_pre=100.) for e in executions]
    links = [dict(**identity, partition_id='P', execution_cluster_id=e['execution_cluster_id'],
        execution_anchor_mode=e['execution_anchor_mode'], execution_side='ask', deceptive_side='bid',
        candidate_order_id='D', cancel_sort_index=30, cancel_event_ts=t+timedelta(seconds=2.5),
        attributed_cancel_visible_qty=100., cancel_visible_qty=100., assigned_flag=i==1,
        post_cancel_mid_reversion=.05, has_cancel_reversion_state=True, cancel_reversion_weight=95.)
        for i,e in enumerate(executions)]
    return pl.DataFrame(executions), pl.DataFrame(candidates), pl.DataFrame(links)


def build(e,c,l):
    from spoofing_detection.lob.candidate_episodes import build_candidate_episodes
    return build_candidate_episodes(e,c,l)


def test_joint_smallness_includes_nonwinning_cluster_and_no_double_withdrawal():
    result = build(*inputs())
    assert result.episodes.height == 1
    episode = result.episodes.row(0,named=True)
    assert episode['total_execution_quantity'] == pytest.approx(120.)
    assert episode['withdrawn_quantity'] == pytest.approx(100.)
    assert episode['execution_vwap'] == pytest.approx(101.5)
    assert episode['spoofing_compatible_episode'] is False
    assert result.members.height == 2 and result.withdrawals.height == 1
    assert set(result.anchor_summary['execution_anchor_mode']) == {'passive','aggressive'}
    assert result.anchor_summary['execution_quantity'].sum() == pytest.approx(120.)
    assert result.actor_day_summary['episode_count'].sum() == 1


def test_joint_positive_episode_and_stable_ids():
    e,c,l = inputs(10.)
    a,b = build(e,c,l), build(e.reverse(),c.reverse(),l.reverse())
    assert a.episodes['spoofing_compatible_episode'][0] is True
    for name in ('episodes','members','withdrawals','anchor_summary','actor_day_summary'):
        assert_frame_equal(getattr(a,name),getattr(b,name))


@pytest.mark.parametrize('field,value', [('actor_key','firm:C'),('partition_id','OTHER'),('execution_anchor_mode','passive')])
def test_cancellation_identity_and_anchor_mismatch_rejected(field,value):
    e,c,l = inputs()
    l = l.with_columns(pl.when(pl.col('assigned_flag')).then(pl.lit(value)).otherwise(pl.col(field)).alias(field))
    with pytest.raises(ValueError, match='provenance'):
        build(e,c,l)


def test_no_cross_day_episode_even_same_order_key():
    e,c,l = inputs()
    e = e.with_columns(*[pl.when(pl.col('execution_cluster_id')=='E2').then(pl.col(k)+pl.duration(days=1)).otherwise(pl.col(k)).alias(k) for k in ('cluster_start_ts','cluster_end_ts')])
    c = c.with_columns(pl.when(pl.col('execution_cluster_id')=='E2').then(pl.col('deceptive_order_first_seen_ts')+pl.duration(days=1)).otherwise(pl.col('deceptive_order_first_seen_ts')).alias('deceptive_order_first_seen_ts'))
    result = build(e,c,l.head(0))
    assert result.episodes.height == 2
    assert result.actor_day_summary.height == 2


def test_reused_order_lifecycle_and_disjoint_candidate_do_not_merge():
    e,c,l = inputs()
    c = c.with_columns(pl.when(pl.col('execution_cluster_id')=='E2').then(2).otherwise(1).alias('deceptive_order_first_seen_sort_index'))
    assert build(e,c,l.head(0)).episodes.height == 2


@pytest.mark.parametrize('outcome', ['fpm','reversion'])
def test_missing_outcomes_are_unavailable_not_positive(outcome):
    e,c,l = inputs(10.)
    if outcome == 'fpm':
        e=e.with_columns(pl.when(pl.col('execution_cluster_id')=='E1').then(None).otherwise(.1).alias('favorable_mid_move_pre_fill'))
    else:
        l=l.with_columns(pl.lit(None,dtype=pl.Float64).alias('post_cancel_mid_reversion'))
    result=build(e,c,l)
    assert result.episodes['spoofing_compatible_episode'][0] is False
    assert result.episodes['price_path_observed'][0] is False


def test_no_candidate_empty_output_has_stable_schema():
    e,c,l=inputs()
    empty=build(e.head(0),c.head(0),l.head(0))
    no_profile=build(e,c.head(0),l.head(0))
    for name in ('episodes','members','withdrawals','anchor_summary','actor_day_summary'):
        assert getattr(empty,name).height == 0
        assert getattr(empty,name).schema == getattr(no_profile,name).schema
        assert len(getattr(empty,name).columns)>0


def test_double_assignment_rejected():
    e,c,l=inputs()
    l=l.with_columns(pl.lit(True).alias('assigned_flag'))
    with pytest.raises(ValueError,match='assigned'):
        build(e,c,l)
