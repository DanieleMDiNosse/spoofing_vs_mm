from datetime import datetime, timedelta

import polars as pl
import pytest

from spoofing_detection.lob.spoofing_metrics import (
    _attach_direct_cancellation_window,
    attach_sci_window_metrics,
)


def fixture():
    t = datetime(2024, 1, 1, 12)
    rows = [
        dict(partition_id='P', actor_key='client_original:C', sort_index=i,
             event_ts=t + timedelta(seconds=i), DWI=dwi,
             L_bid_topN=10., L_ask_topN=qty, market_mid=mid)
        for i, dwi, qty, mid in [(1, .8, 90., 99.), (3, .8, 90., 101.),
                                  (4, .8, 90., 100.), (5, 0., 0., 100.),
                                  (8, 0., 0., 100.)]
    ]
    execution = dict(partition_id='P', actor_key='client_original:C', sort_index=5,
        cluster_first_sort_index=5, cluster_last_sort_index=5,
        event_ts=t + timedelta(seconds=5), cluster_end_ts=t + timedelta(seconds=5),
        execution_side='ask', deceptive_side='bid', fill_qty=5., execution_quantity=5.,
        event_price=100., candidate_deceptive_first_seen_sort_index_min=1,
        candidate_deceptive_first_seen_sort_index_max=3,
        candidate_deceptive_placement_observed=True)
    return pl.DataFrame([execution]), pl.DataFrame(rows)


def test_fpm_starts_after_last_candidate_posted_not_earliest():
    executions, states = fixture()
    out = attach_sci_window_metrics(executions, states, window_seconds=2.)
    assert out['posture_state_sort_index'][0] == 3
    assert out['favorable_mid_move_pre_fill'][0] == pytest.approx(-1.)


@pytest.mark.parametrize('placement', [4, 5, 6])
def test_missing_or_nonprior_placement_is_not_imputed(placement):
    executions, states = fixture()
    executions = executions.with_columns(pl.lit(placement).alias('candidate_deceptive_first_seen_sort_index_max'))
    states = states.filter(pl.col('sort_index') != 4)
    out = attach_sci_window_metrics(executions, states, window_seconds=2.)
    assert out['favorable_mid_move_pre_fill'][0] is None


def test_unknown_reload_placement_does_not_supply_favorable_move():
    executions, states = fixture()
    executions = executions.with_columns(pl.lit(False).alias('candidate_deceptive_placement_observed'))
    assert attach_sci_window_metrics(executions, states, window_seconds=2.)['favorable_mid_move_pre_fill'][0] is None


def test_right_censored_sci_null_but_pre_execution_fpm_preserved():
    executions, states = fixture()
    out = attach_sci_window_metrics(executions, states.filter(pl.col('sort_index') <= 5), window_seconds=2.)
    assert out['has_post_window_state'][0] is False
    assert out['MSCI_resting_profile'][0] is None
    assert out['favorable_mid_move_pre_fill'][0] == pytest.approx(-1.)


def test_next_day_cannot_supply_horizon_coverage():
    executions, states = fixture()
    states = states.with_columns(pl.when(pl.col('sort_index') == 8)
        .then(pl.col('event_ts') + pl.duration(days=1)).otherwise(pl.col('event_ts')).alias('event_ts'))
    assert attach_sci_window_metrics(executions, states, window_seconds=2.)['has_post_window_state'][0] is False


def test_broad_cancellation_requires_source_order_and_same_day():
    executions, _ = fixture()
    t = executions['event_ts'][0]
    cancels = pl.DataFrame([dict(partition_id='P', actor_key='client_original:C', side='bid',
        ORDERID=str(i), sort_index=i, event_ts=ts, visible_qty_pre_cancel=50.)
        for i, ts in [(2, t + timedelta(seconds=.5)), (6, t + timedelta(seconds=1)),
                      (7, t + timedelta(days=1))]])
    out = _attach_direct_cancellation_window(executions, cancels, window_seconds=90000.)
    assert out['direct_opposite_cancel_count_window'][0] == 1
    assert out['direct_opposite_cancel_visible_qty_window'][0] == pytest.approx(50.)
