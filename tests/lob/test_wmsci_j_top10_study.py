"""Contract tests for the new pre-price conditional double decile."""
import importlib
import math
import polars as pl
import pytest


def api():
    try:
        return importlib.import_module('scripts.wmsci_j_top10_study')
    except ModuleNotFoundError:
        pytest.fail('new double-decile implementation absent')


def population(n=21):
    return pl.DataFrame({'instrument':['X']*n, 'partition_id':['p']*n,
        'actor_key':['client:x']*n,
        'execution_cluster_id':[f'E{i:03d}' for i in range(n)],
        'WMSCI_event':[float(i+1) for i in range(n)],
        'withdrawal_to_execution_ratio':[0.1]*n,
        'favorable_mid_move_pre_fill':[None]*n,
        'post_cancel_mid_reversion':[None]*n})


def test_exact_ceil_ranking_and_execution_ascending_ties_without_uq_gate():
    m=api(); p=population().with_columns(pl.lit(1.).alias('WMSCI_event'))
    s=m.first_stage(p)
    assert s['execution_cluster_id'].to_list()==['E000','E001','E002']
    assert s['withdrawal_to_execution_ratio'].max()<1


def test_positive_finite_and_instrument_local_denominators():
    m=api();p=population(4).with_columns(pl.Series('WMSCI_event',[math.nan,math.inf,0.,2.]))
    q=population(11).with_columns(pl.lit('Y').alias('instrument'))
    s=m.first_stage(pl.concat([p,q]))
    assert s.group_by('instrument').len().sort('instrument')['len'].to_list()==[1,2]


def test_conditional_j_uses_only_complete_finite_first_stage_and_includes_zero():
    m=api();p=population(21)
    first=m.first_stage(p)
    summary=first.select(m.KEYS).with_columns(pl.Series('J',[None,math.inf,0.]),pl.Series('complete_grid',[False,True,True]))
    s=m.second_stage(first,summary)
    assert s.height==1 and s['execution_cluster_id'][0]=='E018'
    assert s['J'][0]==0
    with pytest.raises(ValueError):m.second_stage(first,summary.head(2))


def test_price_fields_never_select_and_complete_reversion_required():
    m=api();p=population(11)
    s=m.first_stage(p)
    summary=s.select(m.KEYS).with_columns(pl.lit(1.).alias('J'),pl.lit(True).alias('complete_grid'))
    a=m.second_stage(s,summary)
    altered=s.with_columns(pl.lit(-999.).alias('favorable_mid_move_pre_fill'),pl.lit(999.).alias('post_cancel_mid_reversion'))
    b=m.second_stage(altered,summary)
    assert a.select(m.KEYS).equals(b.select(m.KEYS))
    f=pl.DataFrame({'favorable_mid_move_pre_fill':[1.,0.,-1.,None],
        'post_cancel_mid_reversion':[1.,0.,-1.,1.],
        'post_cancel_mid_reversion_complete':[True,True,True,False]})
    facts=m.price_counts(f)
    assert facts['n']==4 and facts['FPM_positive']==1 and facts['FPM_zero']==1 and facts['FPM_negative']==1 and facts['FPM_missing']==1
    assert facts['REV_positive']==1 and facts['REV_zero']==1 and facts['REV_negative']==1 and facts['REV_missing_or_incomplete']==1


def test_missing_support_is_not_zero_j_and_exact_post_fill_required():
    m=api();cfg=m.kernel_config()
    ex=pl.DataFrame({'instrument':['X'],'partition_id':['p'],'actor_key':['client:x'],
        'execution_cluster_id':['E'], 'deceptive_side':['bid'],
        'cluster_first_sort_index':[2],'cluster_last_sort_index':[2],'cluster_end_replay_time_ns':[100]})
    st=pl.DataFrame({'partition_id':['p','p'],'actor_key':['client:x']*2,'sort_index':[1,2],
        'replay_time_ns':[90,100],'DWI':[.5,.1],'L_bid_topN':[1.,.5],'L_ask_topN':[1.,1.]})
    t,s=m.compute_and_verify(ex,st,None,cfg)
    assert s['J'][0] is None and not s['complete_grid'][0]
    cov={'segments':[{'partition_id':'p','start_ns':0,'end_ns':2000000000,'attested':True,'source':'test'}]}
    t,s=m.compute_and_verify(ex,st.head(1),cov,cfg)
    assert s['J'][0] is None and set(t['reason'])=={'missing_post_execution_state'}


def test_verifier_rejects_corrupted_trajectory():
    m=api();cfg=m.kernel_config()
    ex=pl.DataFrame({'instrument':['X'],'partition_id':['p'],'actor_key':['client:x'],
        'execution_cluster_id':['E'],'deceptive_side':['bid'],'cluster_first_sort_index':[2],
        'cluster_last_sort_index':[2],'cluster_end_replay_time_ns':[100]})
    st=pl.DataFrame({'partition_id':['p','p'],'actor_key':['client:x']*2,'sort_index':[1,2],
        'replay_time_ns':[90,100],'DWI':[.5,.1],'L_bid_topN':[1.,.5],'L_ask_topN':[1.,1.]})
    cov={'segments':[{'partition_id':'p','start_ns':0,'end_ns':2000000000,'attested':True,'source':'test'}]}
    t,s=m.compute_and_verify(ex,st,cov,cfg)
    assert s['complete_grid'][0]
    from spoofing_detection.lob.msci_grid_runner import verify_partition_grids
    with pytest.raises(ValueError):verify_partition_grids(ex,t.with_columns(pl.lit(0.).alias('MSCI')),s,cfg,states=st,coverage=cov)


def test_selected_trajectory_figure_contains_endpoint_tick_labels(tmp_path):
    m=api()
    selected=population(3).with_columns(pl.Series('J',[0.1,1.0,2.0]))
    aliases=selected.select(m.KEYS).with_columns(pl.lit('A001').alias('actor_alias'),pl.col('execution_cluster_id').alias('execution_alias'))
    trajectory=pl.concat([selected.select(m.KEYS).with_columns(pl.lit(offset).alias('grid_offset_ns'),pl.lit(value).alias('MSCI'))
        for offset,value in [(0,-1.0),(50000000,1.0),(1000000000,2.0)]])
    m.plot_selected(selected,trajectory,aliases,tmp_path)
    assert (tmp_path/'selected_msci_paths.pdf').is_file()


@pytest.mark.parametrize('key,value',[('first_fraction',0.05),('second_fraction',0.05),('price_selection',True),('uq_selection',True),('grid_step_ns',True)])
def test_scientific_config_rejects_changed_selection(key,value):
    import json
    m=api()
    config=json.loads((m.ROOT/'configs/wmsci_j_top10_study.json').read_text())
    m.validate_study_config(config)
    config[key]=value
    with pytest.raises(ValueError):m.validate_study_config(config)
