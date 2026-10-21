"""Market-only publication contract; frozen sources are never rewritten."""
import importlib.util
import json
from pathlib import Path
import polars as pl
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / 'scripts/generate_market_only_manuscript.py'


def module():
    assert SCRIPT.exists(), 'market-only renderer is missing'
    spec = importlib.util.spec_from_file_location('market_publication', SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_existing_output_refused_before_source_reads(tmp_path, monkeypatch):
    m = module()
    def forbidden(*args, **kwargs):
        pytest.fail('read before existing-output guard')
    monkeypatch.setattr(m, 'validate_root', forbidden)
    monkeypatch.setattr(m, 'validate_stage', forbidden)
    with pytest.raises(FileExistsError):
        m.build(tmp_path)


def cell():
    return pl.DataFrame({'instrument':['X']*3, 'partition_id':['p']*3,
        'actor_key':['a']*3, 'execution_cluster_id':['1','2','3'],
        'is_candidate':[True]*3, 'WMSCI_market':[1.,0.,0.],
        'complete_grid':[True]*3, 'J':[0.,1.,2.],
        'priority_top_w_market':[True,False,False],
        'priority_top_j_given_market':[True,False,False]})


def test_market_support_zero_j_and_ceiling_budgets():
    r = module().market_metrics(cell())
    assert r == {'candidate_executions':3,'positive_market_support':1,
        'market_w_budget':1,'market_w_count':1,'conditional_finite_J_support':1,
        'market_j_budget':1,'market_j_count':1}


@pytest.mark.parametrize('flag', ['is_candidate','complete_grid','priority_top_w_market','priority_top_j_given_market'])
def test_null_membership_flags_rejected(flag):
    with pytest.raises(ValueError, match='Boolean'):
        module().market_metrics(cell().with_columns(pl.lit(None, dtype=pl.Boolean).alias(flag)))


def test_non_nested_j_rejected():
    with pytest.raises(ValueError):
        module().market_metrics(cell().with_columns(pl.Series('priority_top_j_given_market',[False,True,False])))


def test_same_count_wrong_price_key_rejected():
    m = module()
    native = cell().filter(pl.col(m.WFLAG))
    prices = native.with_columns(pl.lit('wrong').alias('execution_cluster_id'))
    with pytest.raises(ValueError, match='key'):
        m.verify_price_membership(prices, native)


def test_wrong_price_flag_rejected():
    m = module()
    native = cell().filter(pl.col(m.WFLAG))
    with pytest.raises(ValueError, match='flag'):
        m.verify_price_membership(native.with_columns(pl.lit(False).alias(m.JFLAG)), native)


def test_public_package_contract():
    m = module()
    out = REPO / m.OUTPUT
    assert out.is_dir(), 'real publication package has not been generated'
    m.validate_stage(out, 'publication_package_verified')
    counts = pl.read_csv(out/'cell_counts.csv')
    assert counts.height == 60
    primary = counts.filter((pl.col('age_policy')=='exp_h30') & (pl.col('horizon_ns')==10_000_000_000)).sort('instrument')
    assert primary['candidate_executions'].to_list()==[3526,2393,995]
    assert primary['market_w_count'].to_list()==[353,240,98]
    assert primary['market_j_count'].to_list()==[36,24,10]
    cases = pl.read_csv(out/'cases.csv')
    assert cases.height==40 and set(cases['case_id'])=={'R','N'}
    summary=pl.read_csv(out/'primary_price_summary.csv')
    assert summary.height==6 and set(summary['ranking'])=={'market'}
    sensitivity=pl.read_csv(out/'primary_price_sign_sensitivity.csv')
    assert sensitivity.height==24 and set(sensitivity['ulp_factor'])=={4,8,16,32}
    for path in out.iterdir():
        assert not any(s in path.name for s in ['self','baseline','ranking_overlap'])
        if path.suffix=='.csv':
            columns=pl.read_csv(path).columns
            assert not any('self' in c or c in m.KEYS[1:] or c.endswith('_id') and c!='case_id' for c in columns)
        if path.suffix=='.tex':
            text=path.read_text().lower()
            assert 'self' not in text and 'alert' not in text
    facts=json.loads((out/'facts.json').read_text())
    assert facts['scope']=='market_only'
    assert facts['primary_candidate_executions']==6914
    assert facts['primary_price_union_executions']==691
    assert len(facts['reused_assets'])==2
    assert facts['market_formula'] == 'W = log(1+P_g/Q)*log(1+U_g/Q)*(U_g/D); D>0; null if unsupported'
    for path in out.glob('*.json'):
        text = path.read_text().lower()
        assert 'alert' not in text
        assert '900000000000664' not in text and '900000000001520' not in text
    for source, expected in facts['input_renderer_bindings'].items():
        assert m.digest(REPO/source)==expected
