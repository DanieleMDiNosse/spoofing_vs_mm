from datetime import datetime, timedelta
from dataclasses import replace
import math

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


def build(e, c, l, **kwargs):
    from spoofing_detection.lob.candidate_episodes import build_candidate_episodes
    return build_candidate_episodes(e, c, l, **kwargs)


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


def test_shared_candidate_episode_scores_unique_lifecycle_not_repeated_snapshots():
    executions, candidates, links = inputs()
    executions = executions.with_columns(
        pl.when(pl.col('execution_cluster_id') == 'E1').then(10.0).otherwise(20.0)
        .alias('execution_quantity')
    )
    # E1 and E2 see the same live candidate at different snapshots.  Its
    # exposure must be one lifecycle cap (max pre-cancel visible quantity),
    # rather than the sum of repeated per-execution snapshots.
    candidates = candidates.with_columns(
        pl.when(pl.col('execution_cluster_id') == 'E2').then(80.0).otherwise(100.0)
        .alias('deceptive_order_visible_qty_pre')
    )
    # Attribution is a legacy winner-level diagnostic and may be capped or
    # otherwise transformed.  The primary episode quantity must instead use
    # the raw pre-cancel displayed quantity of the unique physical removal.
    links = links.with_columns(
        pl.when(pl.col('assigned_flag')).then(150.0).otherwise(pl.col('attributed_cancel_visible_qty'))
        .alias('attributed_cancel_visible_qty')
    ).with_columns(
        pl.when(pl.col('assigned_flag')).then(150.0).otherwise(pl.col('cancel_visible_qty'))
        .alias('cancel_visible_qty')
    )

    result = build(executions, candidates, links)
    episode = result.episodes.row(0, named=True)

    assert episode['total_execution_quantity'] == pytest.approx(30.0)
    assert episode['primary_lifecycle_union_exposure_quantity'] == pytest.approx(100.0)
    # Raw removal is preserved for audit, but cannot claim more quantity than
    # the maximum admissible pre-cancel snapshot supports for this lifecycle.
    assert episode['primary_raw_unique_cancellation_quantity'] == pytest.approx(150.0)
    assert episode['primary_supported_unique_cancellation_quantity'] == pytest.approx(100.0)
    assert episode['primary_withdrawal_to_lifecycle_exposure_ratio'] == pytest.approx(1.0)
    assert episode['primary_quantity_only_wmsci'] == pytest.approx(
        math.log1p(100.0 / 30.0) * math.log1p(100.0 / 30.0)
    )
    assert result.members.height == 2  # zero-score members must remain auditable.
    assert result.withdrawals.height == 1  # one physical cancellation once.
    lifecycle = result.candidate_lifecycles.row(0, named=True)
    assert lifecycle['candidate_snapshot_count'] == 2
    assert lifecycle['admissible_pre_cancel_snapshot_count'] == 2
    assert lifecycle['lifecycle_union_exposure_quantity'] == pytest.approx(100.0)
    assert lifecycle['raw_unique_cancellation_quantity'] == pytest.approx(150.0)
    assert lifecycle['supported_unique_cancellation_quantity'] == pytest.approx(100.0)
    assert lifecycle['legacy_attributed_cancellation_quantity'] == pytest.approx(150.0)
    withdrawal = result.withdrawals.row(0, named=True)
    assert withdrawal['raw_cancel_visible_qty'] == pytest.approx(150.0)
    assert withdrawal['supported_cancel_visible_qty'] == pytest.approx(100.0)
    assert withdrawal['cancel_reversion_weight'] == pytest.approx(100.0)


def test_quantity_only_episode_score_has_piecewise_zero_semantics():
    from spoofing_detection.lob.candidate_episodes import quantity_only_episode_wmsci

    assert quantity_only_episode_wmsci(100.0, 10.0, 0.0) == 0.0
    assert quantity_only_episode_wmsci(0.0, 10.0, 10.0) == 0.0
    # Q<=0 violates the execution-cluster quantity invariant; it is undefined,
    # not a zero-risk observation.
    assert quantity_only_episode_wmsci(100.0, 0.0, 10.0) is None
    assert quantity_only_episode_wmsci(100.0, -1.0, 10.0) is None
    assert quantity_only_episode_wmsci(None, 10.0, 10.0) is None


def test_episode_artifact_validator_registers_lifecycle_audit_and_rejects_a_gt_one():
    from spoofing_detection.lob.episode_artifacts import (
        episode_frames,
        episode_metadata,
        validate_episode_result,
    )

    result = build(*inputs())
    assert 'candidate_lifecycles' in episode_frames(result)
    semantics = episode_metadata()['episode_semantics']
    assert semantics['primary_cancellation_quantity'].startswith('supported')
    assert 'not_simultaneous_depth' in semantics['primary_exposure_quantity']
    validate_episode_result(result)

    invalid = replace(
        result,
        episodes=result.episodes.with_columns(
            pl.lit(1.01).alias('primary_withdrawal_to_lifecycle_exposure_ratio')
        ),
    )
    with pytest.raises(ValueError, match='A=U/P'):
        validate_episode_result(invalid)


def test_post_cancel_candidate_snapshot_is_not_admissible_for_lifecycle_exposure():
    executions, candidates, links = inputs(10.0)
    # The second execution is after the first physical cancellation.  Its
    # repeated snapshot cannot enlarge the lifecycle's exposure denominator.
    executions = executions.with_columns(
        pl.when(pl.col('execution_cluster_id') == 'E2').then(40).otherwise(pl.col('cluster_first_sort_index'))
        .alias('cluster_first_sort_index'),
        pl.when(pl.col('execution_cluster_id') == 'E2').then(40).otherwise(pl.col('cluster_last_sort_index'))
        .alias('cluster_last_sort_index'),
        pl.when(pl.col('execution_cluster_id') == 'E2').then(pl.col('cluster_start_ts') + pl.duration(seconds=3)).otherwise(pl.col('cluster_start_ts'))
        .alias('cluster_start_ts'),
        pl.when(pl.col('execution_cluster_id') == 'E2').then(pl.col('cluster_end_ts') + pl.duration(seconds=3)).otherwise(pl.col('cluster_end_ts'))
        .alias('cluster_end_ts'),
    )
    candidates = candidates.with_columns(
        pl.when(pl.col('execution_cluster_id') == 'E2').then(1_000.0).otherwise(pl.col('deceptive_order_visible_qty_pre'))
        .alias('deceptive_order_visible_qty_pre'),
    )
    # There is no second candidate-to-cancel link: the candidate-order
    # observation at E2 is after E1's physical cancellation and must not be
    # allowed to enlarge the original lifecycle exposure.
    links = links.filter(pl.col('execution_cluster_id') == 'E1').with_columns(
        pl.lit(True).alias('assigned_flag')
    )
    result = build(executions, candidates, links)
    lifecycle = result.candidate_lifecycles.row(0, named=True)

    assert lifecycle['candidate_snapshot_count'] == 2
    assert lifecycle['admissible_pre_cancel_snapshot_count'] == 1
    assert lifecycle['lifecycle_union_exposure_quantity'] == pytest.approx(100.0)


def test_shared_candidate_transitive_links_merge_without_cross_scope_merges():
    executions, candidates, links = inputs(10.0)
    e3 = executions.filter(pl.col('execution_cluster_id') == 'E2').with_columns(
        pl.lit('E3').alias('execution_cluster_id'),
        (pl.col('cluster_first_sort_index') + 10).alias('cluster_first_sort_index'),
        (pl.col('cluster_last_sort_index') + 10).alias('cluster_last_sort_index'),
        (pl.col('cluster_start_ts') + pl.duration(seconds=1)).alias('cluster_start_ts'),
        (pl.col('cluster_end_ts') + pl.duration(seconds=1)).alias('cluster_end_ts'),
    )
    executions = pl.concat([executions, e3])
    # E1--B--E2--C--E3 is one connected component; a same-named B for another
    # actor remains a separate episode because the scope is part of the link.
    c1 = candidates.filter(pl.col('execution_cluster_id') == 'E1')
    c2b = candidates.filter(pl.col('execution_cluster_id') == 'E2')
    c2c = c2b.with_columns(pl.lit('C').alias('deceptive_order_id'))
    c3 = c2c.with_columns(pl.lit('E3').alias('execution_cluster_id'))
    other = c1.with_columns(
        pl.lit('E4').alias('execution_cluster_id'), pl.lit('client_original:OTHER').alias('actor_key'),
        pl.lit('OTHER').alias('actor_id'),
    )
    e4 = executions.filter(pl.col('execution_cluster_id') == 'E1').with_columns(
        pl.lit('E4').alias('execution_cluster_id'), pl.lit('client_original:OTHER').alias('actor_key'),
        pl.lit('OTHER').alias('actor_id'),
    )
    result = build(pl.concat([executions, e4]), pl.concat([c1, c2b, c2c, c3, other]), links.head(0))

    assert sorted(result.episodes['cluster_count'].to_list()) == [1, 3]


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


def v3_inputs():
    executions, candidates, links = inputs()
    replay_start_ns = 1_704_110_400_000_000_000
    executions = executions.with_columns(
        (pl.lit(replay_start_ns) + (pl.col('cluster_first_sort_index') * 100_000_000)).alias(
            'cluster_start_replay_time_ns'
        ),
        (pl.lit(replay_start_ns) + (pl.col('cluster_last_sort_index') * 100_000_000)).alias(
            'cluster_end_replay_time_ns'
        ),
        (pl.lit(replay_start_ns) + (pl.col('cluster_last_sort_index') * 100_000_000)).alias(
            'cluster_end_fill_anchor_ns'
        ),
    )
    candidates = candidates.with_columns(
        pl.lit(replay_start_ns - 100_000_000).alias('deceptive_order_first_seen_replay_ns')
    )
    links = links.with_columns(
        pl.lit(replay_start_ns + 2_500_000_000).alias('cancellation_removal_ns')
    )
    return executions, candidates, links


def test_v3_candidate_prior_check_uses_replay_ns_not_display_datetime():
    executions, candidates, links = v3_inputs()
    candidates = candidates.with_columns(
        pl.when(pl.col('execution_cluster_id') == 'E1')
        .then(1_704_110_401_000_000_000)
        .otherwise(1_704_110_402_000_000_000)
        .alias('deceptive_order_first_seen_replay_ns')
    )

    with pytest.raises(ValueError, match='candidate provenance is not prior') as exc_info:
        build(executions, candidates, links)

    message = str(exc_info.value)
    assert "v3_candidate=True" in message
    assert "candidate_first_sort_index=1" in message
    assert "cluster_first_sort_index=10" in message


def test_v3_cancellation_causality_uses_removal_ns_not_display_datetime():
    executions, candidates, links = v3_inputs()
    links = links.with_columns(
        pl.when(pl.col('execution_cluster_id') == 'E1')
        .then(1_704_110_400_999_999_999)
        .otherwise(1_704_110_401_999_999_999)
        .alias('cancellation_removal_ns')
    )

    with pytest.raises(ValueError, match='withdrawal provenance is not causally after'):
        build(executions, candidates, links)


def test_v3_episode_scope_uses_replay_ns_not_display_datetime():
    executions, candidates, links = v3_inputs()
    executions = executions.with_columns(
        pl.when(pl.col('execution_cluster_id') == 'E2')
        .then(pl.col('cluster_start_ts') + pl.duration(days=1))
        .otherwise(pl.col('cluster_start_ts'))
        .alias('cluster_start_ts'),
        pl.when(pl.col('execution_cluster_id') == 'E2')
        .then(pl.col('cluster_end_ts') + pl.duration(days=1))
        .otherwise(pl.col('cluster_end_ts'))
        .alias('cluster_end_ts'),
    )
    candidates = candidates.with_columns(
        pl.when(pl.col('execution_cluster_id') == 'E2')
        .then(pl.col('deceptive_order_first_seen_ts') + pl.duration(days=1))
        .otherwise(pl.col('deceptive_order_first_seen_ts'))
        .alias('deceptive_order_first_seen_ts'),
    )

    result = build(executions, candidates, links.head(0))

    assert result.episodes.height == 1


def test_v3_missing_candidate_replay_clock_is_excluded_without_display_fallback():
    executions, candidates, links = v3_inputs()
    candidates = candidates.with_columns(
        pl.when(pl.col("execution_cluster_id") == "E1")
        .then(None)
        .otherwise(pl.col("deceptive_order_first_seen_replay_ns"))
        .alias("deceptive_order_first_seen_replay_ns")
    )

    result = build(
        executions.filter(pl.col("execution_cluster_id") == "E1"),
        candidates.filter(pl.col("execution_cluster_id") == "E1"),
        links.filter(pl.col("execution_cluster_id") == "E1"),
        require_v3_clocks=True,
    )

    assert result.episodes.height == 0


def test_v3_producer_lifecycle_clock_column_yields_episode():
    executions, candidates, links = v3_inputs()
    result = build(
        executions.filter(pl.col('execution_cluster_id') == 'E1'),
        candidates.filter(pl.col('execution_cluster_id') == 'E1'),
        links.filter(pl.col('execution_cluster_id') == 'E1'),
        require_v3_clocks=True,
    )
    assert result.episodes.height == 1
    assert result.members.height == 1


def test_v3_rejects_missing_candidate_clock_column():
    executions, candidates, links = v3_inputs()
    candidates = candidates.drop('deceptive_order_first_seen_replay_ns')
    with pytest.raises(ValueError, match='deceptive_order_first_seen_replay_ns'):
        build(executions, candidates, links, require_v3_clocks=True)
