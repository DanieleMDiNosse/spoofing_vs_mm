"""Contract tests for the fresh all-candidate publication consumer."""
import importlib.util
from pathlib import Path
import math
import polars as pl
import pytest

PATH = Path(__file__).resolve().parents[2] / 'scripts/generate_rank_only_manuscript.py'


def module():
    assert PATH.exists(), 'new rank-only consumer is not implemented'
    spec = importlib.util.spec_from_file_location('rank_only', PATH)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def population():
    rows = []
    for instrument in ['NEXI', 'RISANAMENTO']:
        for cluster, w, j in [('b', 2., 9.), ('a', 2., 0.), ('c', 0., None), ('d', float('nan'), 8.)]:
            rows.append(dict(instrument=instrument, partition_id='secret-partition', actor_key='secret-actor',
                execution_cluster_id=cluster, age_policy='exp_h30', horizon_ns=10_000_000_000,
                execution_anchor_mode='passive', identity_level='client_original',
                candidate_qty_raw=10., withdrawal_qty_raw=3., execution_quantity=2.,
                candidate_qty_age_weighted=5., withdrawal_qty_age_weighted=1.,
                opposite_market_visible_qty_pre_topN=20., WMSCI_market=w, J=j,
                complete_grid=j is not None, is_candidate=True))
    return pl.DataFrame(rows)


def prices():
    return population().with_columns(pl.lit(1.).alias('FPM'), pl.lit(0.).alias('REV'),
        pl.lit(-2.).alias('ADV'), pl.lit(True).alias('REV_complete'),
        pl.lit(1).alias('REV_cancel_count'), pl.lit(0).alias('REV_missing_cancel_count'),
        *[pl.lit(1e-12).alias(m+'_zero_tolerance') for m in ['FPM','REV','ADV']],
        pl.lit(1).alias('FPM_sign'), pl.lit(0).alias('REV_sign'), pl.lit(-1).alias('ADV_sign'))


def test_rank_per_instrument_ties_zero_and_undefined_retained():
    m=module(); ranked=m.rank_candidates(population())
    assert ranked.height == 8
    for frame in ranked.partition_by('instrument'):
        assert frame['execution_cluster_id'].to_list() == ['a','b','c','d']
        assert frame['W_rank'].to_list() == [1,2,3,None]
        assert frame['W_status'].to_list() == ['finite','finite','finite','unsupported']
    changed=population().with_columns(pl.lit(None,dtype=pl.Float64).alias('J'),pl.lit(False).alias('complete_grid'))
    assert m.rank_candidates(changed).select(m.KEYS+['W_rank']).equals(ranked.select(m.KEYS+['W_rank']))


@pytest.mark.parametrize('field',['partition_id','actor_key','execution_cluster_id'])
def test_null_keys_rejected(field):
    m=module()
    with pytest.raises(ValueError,match='key'):
        m.rank_candidates(population().with_columns(pl.lit(None,dtype=pl.String).alias(field)))


def test_duplicate_keys_and_raw_definition_rejected():
    m=module(); p=population()
    with pytest.raises(ValueError,match='key'): m.rank_candidates(pl.concat([p,p.head(1)]))
    for field in ['candidate_qty_raw','withdrawal_qty_raw','execution_quantity']:
        with pytest.raises(ValueError,match='raw candidate'):
            m.rank_candidates(p.with_columns(pl.lit(0.).alias(field)))
    with pytest.raises(ValueError,match='raw candidate'):
        m.rank_candidates(p.with_columns(pl.lit(None,dtype=pl.Float64).alias('withdrawal_qty_raw')))


def test_exact_all_candidate_prices_and_original_values():
    m=module(); p=population(); q=prices()
    m.verify_price_membership(q,p)
    with pytest.raises(ValueError,match='key'): m.verify_price_membership(q.head(7),p)
    with pytest.raises(ValueError,match='key'): m.verify_price_membership(pl.concat([q,q.head(1)]),p)
    with pytest.raises(ValueError,match='native value'): m.verify_price_membership(q.with_columns(pl.lit(0.).alias('J')),p)
    with pytest.raises(ValueError,match='native value'): m.verify_price_membership(q.with_columns(pl.lit(4.).alias('withdrawal_qty_raw')),p)
    with pytest.raises(ValueError,match='native value'): m.verify_price_membership(q.with_columns(pl.lit(False).alias('is_candidate')),p)


def test_approved_case_figures_reconcile_native_read_only():
    m=module()
    columns=m.KEYS+['age_policy','horizon_ns','is_candidate',*m.NATIVE]
    native=pl.concat([pl.read_parquet(m.REPO/r/'analysis/population.parquet',columns=columns) for r in m.DEFAULT_ROOTS])
    cases=m.authenticate_reuse(native,m.cell_summary(native))
    assert cases.height==40 and cases['case_id'].n_unique()==2
    assert not any('priority' in c or 'self' in c for c in cases.columns)


def test_price_partitions_strata_and_complete_reversion():
    m=module(); q=prices()
    s=m.summarize_prices(q); strata=m.summarize_prices(q,roles=True)
    assert s.height == 2
    for row in s.to_dicts():
        assert row['n']==4 and row['FPM_positive']==4 and row['REV_zero']==4 and row['ADV_negative']==4
        assert row['both_positive_complete']==0
        for metric in ['FPM','REV','ADV']:
            assert sum(row[metric+'_'+x] for x in ['positive','zero','negative','missing_or_incomplete' if metric=='REV' else 'missing'])==row['n']
    m.verify_strata(s,strata)
    bad=q.with_columns(pl.lit(False).alias('REV_complete'))
    with pytest.raises(ValueError,match='REV'): m.validate_prices(bad)
    with pytest.raises(ValueError,match='sign'):
        m.validate_prices(q.with_columns(pl.lit(0).alias('ADV_sign')))
    tiny=q.with_columns(pl.lit(1e-13).alias('FPM'),pl.lit(0).alias('FPM_sign'))
    assert m.summarize_prices(tiny,factor=0)['FPM_positive'].sum()==8
    assert m.summarize_prices(tiny,factor=8)['FPM_zero'].sum()==8


def test_public_allowlist_alias_multiplicity_and_distribution():
    m=module(); ranked=m.rank_candidates(population()); public=m.public_ranking(ranked,prices())
    assert public.height==8 and public['execution_alias'].n_unique()==8
    assert public.equals(m.public_ranking(m.rank_candidates(population().reverse()),prices().reverse()))
    assert not set(m.KEYS[1:]) & set(public.columns)
    assert not any(any(t in c.lower() for t in ['priority','stage','timestamp','self','rank_j']) for c in public.columns)
    assert 'secret-' not in str(public.to_dicts())
    assert {'WMSCI_market','J','J_status','W_rank','FPM','REV','ADV','execution_anchor_mode','identity_level'} <= set(public.columns)
    counts=m.cell_summary(population())
    assert counts['candidate_executions'].to_list()==[4,4]
    assert counts['finite_W'].to_list()==[3,3] and counts['zero_W'].to_list()==[1,1]
    assert counts['complete_finite_J'].to_list()==[3,3]
    assert counts['J_missing_or_incomplete'].to_list()==[1,1]
    assert counts['W_median'].to_list()==[2.,2.]


def test_full_key_tie_breaks_not_actor_or_cluster_deduplication():
    m=module(); p=population().head(1)
    p=pl.concat([p.with_columns(pl.lit(part).alias('partition_id'),pl.lit(actor).alias('actor_key'))
                 for part,actor in [('p2','a'),('p1','b'),('p1','a')]])
    ranked=m.rank_candidates(p)
    assert ranked.select('partition_id','actor_key').rows()==[('p1','a'),('p1','b'),('p2','a')]
    assert ranked['W_rank'].to_list()==[1,2,3]
    assert m.public_ranking(ranked,p.join(prices().head(1).select(m.KEYS+[m+s for m in m.METRICS for s in ['', '_sign', '_zero_tolerance']]+['REV_complete','REV_cancel_count','REV_missing_cancel_count']),on=['instrument','execution_cluster_id'],suffix='_source').drop('partition_id_source','actor_key_source')).height==3


def test_native_frozen_primary_and_all_sixty_cells_read_only():
    m=module(); frozen=pl.read_parquet(m.REPO/m.REVIEW/'expected_candidate_memberships.parquet')
    columns=[*frozen.columns,'age_policy','horizon_ns','is_candidate']
    population=pl.concat([pl.read_parquet(m.REPO/r/'analysis/population.parquet',columns=columns) for r in m.DEFAULT_ROOTS])
    primary=m.primary(population).filter(pl.col('is_candidate'))
    m.verify_values(primary,frozen,[c for c in frozen.columns if c not in m.KEYS])
    counts=m.cell_summary(population)
    assert counts.height==60
    assert primary.height==6914 and m.rank_candidates(primary).height==6914
    assert {r['instrument']:r['len'] for r in primary.group_by('instrument').len().to_dicts()}==m.EXPECTED
    assert primary['WMSCI_market'].is_finite().all() and primary['complete_grid'].all() and primary['J'].is_finite().all()
    assert primary.filter((pl.col('instrument')=='RISANAMENTO')&(pl.col('WMSCI_market')==0)).height==17


def test_summary_reconciliation_rejects_counts_groups_and_old_stages():
    m=module(); summary=m.summarize_prices(prices())
    m.reconcile_summary(summary,summary,['instrument'])
    with pytest.raises(ValueError,match='mismatch'):
        m.reconcile_summary(summary,summary.with_columns(pl.lit(90).alias('FPM_positive')),['instrument'])
    with pytest.raises(ValueError,match='membership'):
        m.reconcile_summary(summary,summary.head(1),['instrument'])
    with pytest.raises(ValueError,match='schema'):
        m.reconcile_summary(summary,summary.with_columns(pl.lit('W').alias('stage')),['instrument'])
    with pytest.raises(ValueError,match='partition'):
        m.verify_strata(summary,m.summarize_prices(prices(),roles=True).with_columns(pl.lit(90).alias('n')))


def test_actual_producer_raw_sign_columns_are_reconciled():
    m=module();actual=m.summarize_prices(prices())
    assert actual['FPM_raw_positive'].to_list()==[4,4]
    assert actual['REV_raw_zero'].to_list()==[4,4]
    assert actual['ADV_raw_negative'].to_list()==[4,4]
    m.reconcile_summary(actual,actual,['instrument'])


def test_existing_output_refused_before_any_source_read(tmp_path,monkeypatch):
    m=module(); out=tmp_path/'exists'; out.mkdir()
    def forbidden(*args,**kwargs): pytest.fail('source read before overwrite refusal')
    monkeypatch.setattr(m,'validate_root',forbidden)
    monkeypatch.setattr(m,'load_json',forbidden)
    with pytest.raises(FileExistsError): m.build(out,prices_root=tmp_path/'absent')


def test_render_contract_synthetic(tmp_path):
    m=module(); p=population(); rank=m.rank_candidates(p); count=m.cell_summary(p)
    count=count.with_columns(pl.lit(1.782e-6).alias('W_median'))
    totals=pl.DataFrame([dict(instrument=i,execution_count=4,fill_member_count=4) for i in ['NEXI','RISANAMENTO']])
    composition=p.group_by(['instrument','execution_anchor_mode','identity_level']).len().rename({'len':'executions'}).with_columns(pl.lit('Primary candidates').alias('cohort'))
    cases=p.head(1).with_columns(pl.lit('R').alias('case_id'))
    m.render_tables(tmp_path,totals,count,composition,cases,m.summarize_prices(prices()),m.summarize_prices(prices(),roles=True))
    expected={'population.tex','ranking_summary.tex','horizon_counts.tex','age_policy_summary.tex','composition.tex','anonymouscase_table.tex','candidate_prices.tex','candidate_advantage.tex','candidate_price_strata.tex'}
    assert {x.name for x in tmp_path.iterdir()}==expected
    text='\n'.join(x.read_text() for x in tmp_path.iterdir())
    assert '1.782e-06' in (tmp_path/'ranking_summary.tex').read_text()
    for label in ['tab:rank-summary','tab:candidate-prices','tab:candidate-advantage','tab:candidate-price-strata']:
        assert label in text
    assert not any(token in text for token in ['priority_top','W+J','top decile','secret-'])
