"""Scientific contracts for the bounded withdrawal_multiscale_v1 core."""
import importlib
import math

import polars as pl
import pytest

NS = 1_000_000_000
KEYS = ['instrument', 'partition_id', 'actor_key', 'execution_cluster_id']
POLICIES = [
    {'id': 'legacy', 'max_age_ns': 30*NS, 'half_life_ns': None},
    {'id': 'all', 'max_age_ns': None, 'half_life_ns': None},
    {'id': 'exp30', 'max_age_ns': None, 'half_life_ns': 30*NS},
]

@pytest.fixture
def core():
    return importlib.import_module('spoofing_detection.lob.withdrawal_multiscale')


def frames(*, delay=1_200_000_000, age=60*NS):
    e = dict(instrument='X', partition_id='P', actor_key='A', execution_cluster_id='E',
             execution_quantity=10., opposite_market_visible_qty_pre_topN=100.,
             deceptive_side='ask', execution_side='bid', execution_anchor_mode='passive',
             cluster_first_sort_index=10, cluster_last_sort_index=11,
             cluster_start_replay_time_ns=100*NS, cluster_end_replay_time_ns=100*NS,
             cluster_end_fill_anchor_ns=100*NS)
    c = {k:e[k] for k in KEYS}
    c.update(deceptive_order_id='O', deceptive_order_visible_qty_pre=60.,
             deceptive_order_first_seen_sort_index=1,
             deceptive_order_first_seen_replay_ns=100*NS-age)
    x = dict(partition_id='P', actor_key='A', ORDERID='O', side='ask', sort_index=20,
             cancel_order_first_seen_sort_index=1, cancellation_removal_ns=100*NS+delay,
             visible_qty_pre_cancel=30.)
    l = {k:e[k] for k in KEYS}
    l.update(candidate_order_id='O', cancel_sort_index=20,
             cluster_first_sort_index=10, cluster_last_sort_index=11,
             cluster_end_fill_anchor_ns=100*NS, cancellation_removal_ns=100*NS+delay,
             cancel_visible_qty=30., assigned_flag=True)
    return tuple(pl.DataFrame([r]) for r in (e,c,l,x))


def test_score_identity_depth_scale_and_zero(core):
    s = core.score_pair(60,30,10,100)
    expected = math.log1p(6)*math.log1p(3)*.5
    assert s['WMSCI_self'] == pytest.approx(expected)
    assert s['WMSCI_market'] == pytest.approx(expected*.6)
    assert s['removal_share_age_weighted'] == .5
    assert s['candidate_market_share_age_weighted'] == .6
    assert core.score_pair(600,300,100,1000) == pytest.approx(s)
    assert core.score_pair(60,30,10,200)['WMSCI_market'] == pytest.approx(s['WMSCI_market']/2)
    assert core.score_pair(0,0,10,0)['WMSCI_self'] == 0
    assert core.score_pair(0,0,10,0)['removal_share_age_weighted'] is None
    assert core.score_pair(60,0,10,100)['WMSCI_market'] == 0
    assert core.score_pair(60,30,10,None)['WMSCI_market'] is None


@pytest.mark.parametrize('args', [(-1,0,1,2),(1,2,1,2),(1,1,0,2),(1,1,1,0),(3,1,1,2),(1,1,1,float('nan')),(True,0,1,2)])
def test_score_invalid_domain(core,args):
    with pytest.raises(ValueError): core.score_pair(*args)


def test_age_half_life_and_unknown(core):
    assert core.age_weight(60*NS,30*NS) == pytest.approx(.25)
    assert core.age_weight(0,30*NS) == 1
    assert core.age_weight(60*NS,None) == 1
    for age, half in [(None,NS),(-1,NS),(1.0,NS),(True,NS),(NS,0),(NS,1.0)]:
        with pytest.raises(ValueError): core.age_weight(age,half)


def test_candidate_groups_preserve_execution_and_input_order(core):
    first = ('X', 'P', 'A', 'E1')
    second = ('X', 'P', 'A', 'E2')
    info = {(first, 'O1'): (10.0, 0), (second, 'O2'): (20.0, 1),
            (first, 'O3'): (30.0, 2)}
    original = dict(info)
    grouped = core._by_execution(info)
    assert grouped[first] == [(10.0, 0), (30.0, 2)]
    assert grouped[second] == [(20.0, 1)]
    assert info == original


def test_age_horizon_cells_preserve_population(core):
    p,a = core.evaluate_cells(*frames(),[NS,2*NS],POLICIES)
    assert p.height == 6
    def cell(age,h): return p.filter((pl.col('age_policy')==age)&(pl.col('horizon_ns')==h)).row(0,named=True)
    assert cell('legacy',2*NS)['candidate_qty_raw'] == 0
    assert cell('all',NS)['withdrawal_qty_raw'] == 0
    s = cell('all',2*NS)
    assert (s['candidate_qty_raw'],s['withdrawal_qty_raw'],s['is_candidate']) == (60,30,True)
    s = cell('exp30',2*NS)
    assert (s['candidate_qty_age_weighted'],s['withdrawal_qty_age_weighted']) == pytest.approx((15,7.5))
    assert a.filter(pl.col('assigned_flag')).height == 2


def test_exact_horizon_and_underflow_raw_candidate(core):
    p,_ = core.evaluate_cells(*frames(delay=NS),[NS],POLICIES)
    assert p['withdrawal_qty_raw'].sum() == 0
    p,_ = core.evaluate_cells(*frames(age=100000*NS),[2*NS],[{'id':'tiny','max_age_ns':None,'half_life_ns':NS}])
    s=p.row(0,named=True)
    assert s['is_candidate'] is True
    assert s['WMSCI_self'] == 0
    assert s['age_weight_underflow_count'] == 1
    assert s['removal_share_age_weighted'] is None


def test_reassign_after_age_eligibility_and_ignore_original_winner(core):
    e,c,l,x=frames(age=20*NS)
    newer=e.with_columns(pl.lit('E2').alias('execution_cluster_id'),pl.lit(12,dtype=pl.Int64).alias('cluster_first_sort_index'),pl.lit(13,dtype=pl.Int64).alias('cluster_last_sort_index'))
    c2=c.with_columns(pl.lit('E2').alias('execution_cluster_id'),pl.lit(0,dtype=pl.Int64).alias('deceptive_order_first_seen_replay_ns'))
    l2=l.with_columns(pl.lit('E2').alias('execution_cluster_id'),pl.lit(12,dtype=pl.Int64).alias('cluster_first_sort_index'),pl.lit(13,dtype=pl.Int64).alias('cluster_last_sort_index'))
    e=pl.concat([e,newer]); c=pl.concat([c,c2]); l=pl.concat([l.with_columns(pl.lit(False).alias('assigned_flag')),l2])
    p,a=core.evaluate_cells(e,c,l,x,[2*NS],POLICIES[:2])
    winners=a.filter(pl.col('assigned_flag')).select('age_policy','execution_cluster_id').rows()
    assert set(winners)=={('legacy','E'),('all','E2')}
    assert p['withdrawal_qty_raw'].sum()==60


def test_reused_order_id_lifecycle_not_attributed(core):
    e,c,l,x=frames()
    x=x.with_columns(pl.lit(15).alias('cancel_order_first_seen_sort_index'))
    p,a=core.evaluate_cells(e,c,l,x,[2*NS],POLICIES)
    assert p['withdrawal_qty_raw'].sum()==0
    assert a.height==0


def test_multiple_cancels_capped_cumulatively(core):
    e,c,l,x=frames()
    x2=x.with_columns(pl.lit(21,dtype=pl.Int64).alias('sort_index'),pl.lit(50.).alias('visible_qty_pre_cancel'))
    l2=l.with_columns(pl.lit(21,dtype=pl.Int64).alias('cancel_sort_index'),pl.lit(50.).alias('cancel_visible_qty'))
    p,a=core.evaluate_cells(e,c,pl.concat([l,l2]),pl.concat([x,x2]),[2*NS],POLICIES[1:2])
    assert p['withdrawal_qty_raw'].to_list()==[60]
    assert a['attributed_cancel_visible_qty'].sum()==60


@pytest.mark.parametrize('which,col,value',[(0,'cluster_end_fill_anchor_ns',1.0),(1,'deceptive_order_first_seen_replay_ns',None),(2,'cancel_sort_index',True),(3,'cancellation_removal_ns',1.0)])
def test_reject_bad_clock_schema(core,which,col,value):
    f=list(frames()); f[which]=f[which].with_columns(pl.lit(value).alias(col))
    with pytest.raises(ValueError): core.evaluate_cells(*f,[2*NS],POLICIES)


@pytest.mark.parametrize('which',[0,1,2,3])
def test_duplicate_identities_rejected(core,which):
    f=list(frames()); f[which]=pl.concat([f[which],f[which]])
    with pytest.raises(ValueError): core.evaluate_cells(*f,[2*NS],POLICIES)


def test_empty_typed_frames(core):
    f=frames()
    p,a=core.evaluate_cells(*(x.clear() for x in f),[NS],POLICIES)
    assert p.is_empty() and a.is_empty()
    assert 'WMSCI_market' in p.columns and 'assigned_flag' in a.columns
    p,a=core.evaluate_cells(f[0],*(x.clear() for x in f[1:]),[NS],POLICIES)
    assert p.height==3 and not p['is_candidate'].any()


def test_priorities_ties_j_zero_missing_and_full_population(core):
    rows=[]
    for i in range(11):
        rows.append(dict(instrument='X',partition_id='P',actor_key='A',execution_cluster_id=f'E{i:02}',horizon_ns=NS,age_policy='all',is_candidate=True,WMSCI_self=1.,WMSCI_market=1.,J=None if i==0 else 0.,complete_grid=i!=0))
    p=core.add_priority_flags(pl.DataFrame(rows),.1)
    assert p.height==11
    assert p.filter(pl.col('priority_top_w_self'))['execution_cluster_id'].to_list()==['E00','E01']
    assert p.filter(pl.col('priority_top_j_given_self'))['execution_cluster_id'].to_list()==['E01']
    assert p['priority_top_w_market'].to_list()==p['priority_top_w_self'].to_list()
    assert p['priority_top_j_given_market'].to_list()==p['priority_top_j_given_self'].to_list()


def test_profiles_short_supported_long_incomplete(core):
    e=frames()[0]
    states=pl.DataFrame(dict(partition_id=['P']*3,actor_key=['A']*3,sort_index=[9,11,30],replay_time_ns=[100*NS-1,100*NS,102*NS],DWI=[0.,0.,0.],L_bid_topN=[.5]*3,L_ask_topN=[.5]*3))
    coverage={'segments':[{'partition_id':'P','start_ns':100*NS-1,'end_ns':101*NS,'attested':True,'source':'test'}]}
    s=core.profile_summaries(e,states,coverage,[NS,2*NS]).sort('horizon_ns')
    assert s.height==2
    assert s['complete_grid'].to_list()==[True,False]
    assert s['J'].to_list()==[0.,None]
    assert s[0,'MSCI_t0']==0 and s[0,'MSCI_horizon']==0 and s[0,'MSCI_max']==0
    assert s[1,'reason'] is not None


def test_profile_invalid_later_clock_does_not_invalidate_short_prefix(core):
    e=frames()[0]
    states=pl.DataFrame(dict(partition_id=['P']*4,actor_key=['A']*4,sort_index=[9,11,30,31],replay_time_ns=[100*NS-1,100*NS,101*NS+1,102*NS],DWI=[0.]*4,L_bid_topN=[.5]*4,L_ask_topN=[.5]*4)).with_columns(pl.when(pl.col('sort_index')==31).then(None).otherwise(pl.col('replay_time_ns')).alias('replay_time_ns'))
    coverage={'segments':[{'partition_id':'P','start_ns':100*NS-1,'end_ns':103*NS,'attested':True,'source':'test'}]}
    s=core.profile_summaries(e,states,coverage,[NS,2*NS]).sort('horizon_ns')
    assert s['complete_grid'].to_list()==[True,False]
