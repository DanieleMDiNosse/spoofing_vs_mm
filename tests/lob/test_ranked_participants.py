"""Participant aggregation contracts: no score/price-based membership."""
import polars as pl
import pytest
from scripts import generate_rank_only_manuscript as frozen


def fixtures():
    rows = []
    for i, (instrument, client, firm, candidate, w, j) in enumerate([
        ('A', '7', '7', True, 0., 9.),
        ('A', '7', '8', True, None, None),
        ('A', None, '7', True, 2., 3.),
        ('A', '9', '8', False, 0., 100.),
        ('B', '7', None, True, 4., 0.),
        ('B', None, '7', False, None, None),
    ]):
        actor = 'client_original:'+client if client else 'firm:'+firm
        row = dict(instrument=instrument, partition_id='p'+str(i % 2), actor_key=actor,
            execution_cluster_id=str(i), age_policy='exp_h30', horizon_ns=10_000_000_000,
            event_client_original_id=client, event_firm_id=firm,
            identity_level='client_original' if client else 'firm',
            identity_fallback_flag=client is None,
            execution_anchor_mode='passive' if i % 2 == 0 else 'aggressive',
            candidate_qty_raw=1. if candidate else 0., withdrawal_qty_raw=1.,
            execution_quantity=1., candidate_qty_age_weighted=1.,
            withdrawal_qty_age_weighted=1., opposite_market_visible_qty_pre_topN=10.,
            is_candidate=candidate, WMSCI_market=w, J=j, complete_grid=i != 1,
            child_fill_count=1)
        rows.append(row)
    population = pl.DataFrame(rows, infer_schema_length=None)
    members = pl.DataFrame([{**{k:r[k] for k in frozen.KEYS},
        'child_sort_index':i, 'child_event_firm_id':r['event_firm_id']} for i,r in enumerate(rows)])
    prices = population.filter(pl.col('is_candidate')).with_columns(
        pl.lit(1.).alias('FPM'), pl.lit(.01).alias('FPM_zero_tolerance'), pl.lit(1).alias('FPM_sign'),
        pl.lit(None,dtype=pl.Float64).alias('REV'), pl.lit(.01).alias('REV_zero_tolerance'),
        pl.lit(None,dtype=pl.Int8).alias('REV_sign'), pl.lit(False).alias('REV_complete'),
        pl.lit(1).alias('REV_cancel_count'), pl.lit(1).alias('REV_missing_cancel_count'),
        pl.lit(0.).alias('ADV'), pl.lit(.01).alias('ADV_zero_tolerance'), pl.lit(0).alias('ADV_sign'))
    return population, prices, members


def aggregate(*args):
    from scripts.summarize_ranked_participants import aggregate_participants
    return aggregate_participants(*args)


def test_nonadditive_complete_populations_aliases_and_accounting():
    result = aggregate(*fixtures())
    tables = result['tables']; entities = tables['entities']; summary = tables['summary']
    assert summary.height == 4
    a = summary.filter(pl.col('instrument')=='A')
    clients = a.filter(pl.col('perspective')=='identified_client').row(0,named=True)
    firms = a.filter(pl.col('perspective')=='intermediary').row(0,named=True)
    assert (clients['entity_count'], clients['candidate_entity_count'],clients['execution_count'],clients['candidate_count']) == (2,1,3,2)
    assert (firms['entity_count'],firms['execution_count'],firms['candidate_count']) == (2,4,3)
    assert clients['repeat_candidate_entity_count'] == 1
    assert clients['largest_candidate_group_count'] == 2
    assert clients['largest_candidate_group_denominator'] == 2
    assert entities.filter(pl.col('candidate_count')==0).height == 2
    mapping = result['identity_mapping']
    c7 = mapping.filter(pl.col('original_key')=='client_original:7')['entity_alias'][0]
    assert entities.filter(pl.col('entity_alias')==c7).height == 2
    assert mapping.filter(pl.col('original_key')=='firm:7')['entity_alias'][0] != c7
    r = entities.filter((pl.col('instrument')=='A')&(pl.col('entity_alias')==c7)).row(0,named=True)
    assert (r['candidate_count'],r['supported_W_count'],r['zero_W_count'],r['unsupported_W_count']) == (2,1,1,1)
    assert r['W_median']==0. and r['W_max']==0. and r['J_median']==9.
    assert r['complete_finite_J_count']==1 and r['J_missing_or_incomplete_count']==1
    assert r['REV_missing_count']==2 and r['ADV_zero_count']==2
    assert r['FPM_positive_count']==2 and r['FPM_REV_positive_count']==0
    assert r['candidate_fraction']==1. and r['observed_partition_count']==2
    assert entities.filter(pl.col('firm_missing_bucket')).height==1
    for table in tables.values():
        assert not {'original_key','actor_key','event_firm_id','partition_id'} & set(table.columns)
    for r in entities.to_dicts():
        assert r['passive_execution_count']+r['aggressive_execution_count']==r['execution_count']
        assert r['source_client_execution_count']+r['source_fallback_execution_count']==r['execution_count']
        for m in ['FPM','REV','ADV']:
            assert sum(r[m+'_'+s+'_count'] for s in ['positive','zero','negative','missing'])==r['candidate_count']


@pytest.mark.parametrize('which', ['population','prices'])
def test_duplicate_full_keys_rejected(which):
    p, q, m = fixtures()
    if which=='population': p=pl.concat([p,p.head(1)])
    else: q=pl.concat([q,q.head(1)])
    with pytest.raises(ValueError,match='key'): aggregate(p,q,m)


def test_multifirm_execution_rejected_but_separate_executions_allowed():
    p,q,m=fixtures()
    aggregate(p,q,m)  # same client across firms, separate clusters: valid
    m=m.with_columns(pl.when(pl.col('execution_cluster_id')=='0').then(pl.lit('8')).otherwise(pl.col('child_event_firm_id')).alias('child_event_firm_id'))
    with pytest.raises(ValueError,match='firm'): aggregate(p,q,m)


@pytest.mark.parametrize('change', ['missing_member','extra_member','duplicate_member','child_count'])
def test_exact_member_coverage(change):
    p,q,m=fixtures()
    if change=='missing_member': m=m.slice(1)
    if change=='extra_member': m=pl.concat([m,m.head(1).with_columns(pl.lit('extra').alias('execution_cluster_id'))])
    if change=='duplicate_member': m=pl.concat([m,m.head(1)])
    if change=='child_count': p=p.with_columns(pl.lit(2).alias('child_fill_count'))
    with pytest.raises(ValueError,match='member'): aggregate(p,q,m)


def test_missing_price_rows_stop_but_missing_values_retained():
    p,q,m=fixtures()
    with pytest.raises(ValueError,match='membership'): aggregate(p,q.slice(1),m)
    assert aggregate(p,q,m)['tables']['entities']['REV_missing_count'].sum()>0


def test_j_never_membership_and_aliases_stable_to_row_order():
    p,q,m=fixtures()
    a=aggregate(p,q,m)
    p=p.with_columns(pl.lit(None,dtype=pl.Float64).alias('J'))
    q=q.with_columns(pl.lit(None,dtype=pl.Float64).alias('J'))
    b=aggregate(p.reverse(),q.reverse(),m.reverse())
    assert a['tables']['summary'].equals(b['tables']['summary'])
    assert a['identity_mapping'].equals(b['identity_mapping'])


@pytest.mark.parametrize('field,value', [('actor_key','firm:7'),('identity_fallback_flag',True)])
def test_namespaces_and_attribution_validated(field,value):
    p,q,m=fixtures()
    p=p.with_columns(pl.when(pl.col('execution_cluster_id')=='0').then(pl.lit(value)).otherwise(pl.col(field)).alias(field))
    with pytest.raises(ValueError,match='identity'): aggregate(p,q,m)


def test_saved_price_signs_not_reclassified_or_silently_trusted():
    p,q,m=fixtures()
    q=q.with_columns(pl.lit(-1).alias('FPM_sign'))
    with pytest.raises(ValueError,match='sign'): aggregate(p,q,m)


@pytest.mark.parametrize('bindings', [{}, {'/native/population.parquet':'b'*64}, {'/native/population.parquet':'invalid'}])
def test_price_input_bindings_cannot_drift_from_authenticated_native(bindings):
    from scripts.summarize_ranked_participants import validate_price_bindings
    with pytest.raises(ValueError,match='binding'):
        validate_price_bindings({'input_bindings':bindings},{'/native/population.parquet':'a'*64})
    validate_price_bindings({'input_bindings':{'/native/population.parquet':'a'*64}},
                            {'/native/population.parquet':'a'*64})


def test_loader_exposes_read_only_api():
    from scripts.summarize_ranked_participants import load_native_participants
    assert callable(load_native_participants)


def test_all_zero_candidates_keep_all_entities():
    p,q,m=fixtures()
    p=p.with_columns(pl.lit(False).alias('is_candidate'),pl.lit(0.).alias('candidate_qty_raw'))
    r=aggregate(p,q.head(0),m)
    assert r['tables']['entities'].height==7
    assert r['tables']['entities']['candidate_count'].sum()==0
    assert r['tables']['summary']['largest_candidate_group_fraction'].null_count()==4
    assert r['tables']['entities']['W_median'].null_count()==7


def test_no_identified_clients_does_not_invent_a_client():
    p,q,m=fixtures()
    p=p.filter(pl.col('identity_level')=='firm')
    q=q.filter(pl.col('identity_level')=='firm')
    m=m.join(p.select(frozen.KEYS),on=frozen.KEYS)
    r=aggregate(p,q,m)
    s=r['tables']['summary'].filter(pl.col('perspective')=='identified_client')
    assert s.height==2 and s['entity_count'].sum()==0 and s['execution_count'].sum()==0
    assert r['identity_mapping'].filter(pl.col('perspective')=='identified_client').is_empty()


def test_real_multifirm_child_cluster_stops():
    p,q,m=fixtures()
    p=p.with_columns(pl.when(pl.col('execution_cluster_id')=='0').then(2).otherwise(pl.col('child_fill_count')).alias('child_fill_count'))
    extra=m.head(1).with_columns(pl.lit(99,dtype=pl.Int64).alias('child_sort_index'),pl.lit('8').alias('child_event_firm_id'))
    with pytest.raises(ValueError,match='multi-firm'): aggregate(p,q,pl.concat([m,extra]))


def test_null_anchor_firm_does_not_accept_known_child_firm():
    p,q,m=fixtures()
    m=m.with_columns(pl.when(pl.col('execution_cluster_id')=='4').then(pl.lit('7')).otherwise(pl.col('child_event_firm_id')).alias('child_event_firm_id'))
    with pytest.raises(ValueError,match='firm'): aggregate(p,q,m)


def test_all_price_sign_partitions_and_joint_positive_use_full_candidates():
    p,q,m=fixtures()
    q=q.with_columns(
        pl.Series('FPM',[1.,0.,-1.,None]),pl.Series('FPM_sign',[1,0,-1,None]),
        pl.lit(1.).alias('REV'),pl.lit(1).alias('REV_sign'),
        pl.lit(True).alias('REV_complete'),pl.lit(0).alias('REV_missing_cancel_count'),
        pl.lit(1.).alias('ADV'),pl.lit(1).alias('ADV_sign'))
    r=aggregate(p,q,m)['tables']['entities'].filter((pl.col('instrument')=='A')&(pl.col('perspective')=='identified_client')&(pl.col('candidate_count')>0)).row(0,named=True)
    assert r['candidate_count']==2 and r['price_denominator']==2
    assert r['FPM_positive_count']==1 and r['FPM_zero_count']==1
    assert r['FPM_REV_positive_count']==1 and r['FPM_REV_ADV_positive_count']==1


def test_loader_authentication_failure_stops_before_read(monkeypatch):
    from scripts.summarize_ranked_participants import load_native_participants
    def reject(_):
        raise ValueError('manifest not authenticated')
    monkeypatch.setattr(frozen,'validate_root',reject)
    with pytest.raises(ValueError,match='manifest'): load_native_participants()
