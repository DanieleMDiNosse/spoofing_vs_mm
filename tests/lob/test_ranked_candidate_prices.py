"""All raw candidates, not the native priority union, define the price cohort."""
import importlib.util
from pathlib import Path
import polars as pl
import pytest


def load(name):
    path=Path(__file__).parents[2]/'scripts'/f'{name}.py'
    assert path.exists(), 'all-candidate implementation missing'
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def fixture():
    path=Path(__file__).with_name('test_primary_kernel_prices.py')
    spec=importlib.util.spec_from_file_location('price_fixture',path)
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    args=mod.fixture()
    args[0]=args[0].with_columns(pl.lit(False).alias(f) for f in load('study_primary_kernel_prices').FLAGS).with_columns(
        pl.lit(4.).alias('candidate_qty_raw'),pl.lit(2.).alias('opposite_market_visible_qty_pre_topN'),
        pl.lit(0.).alias('WMSCI_market'),pl.lit(0.).alias('J'))
    return args


@pytest.mark.parametrize('side,h',[('ask',1),('bid',-1)])
def test_zero_w_zero_j_and_false_flags_still_compute(side,h):
    m=load('study_ranked_candidate_prices');args=fixture()
    args[0]=args[0].with_columns(pl.lit(side).alias('execution_side'))
    e,c=m.compute_all(*args)
    assert e.height==1 and c.height==2
    assert e['FPM'][0]==2*h and e['REV'][0]==6.25*h
    assert e['J'][0]==0 and e['WMSCI_market'][0]==0
    assert not any(x.startswith('priority_') for x in e.columns+c.columns)
    assert e['FPM_sign'][0]==h and c['REV_cancel_sign'].to_list()==[h,h]


def test_complete_rev_and_invalid_predecessor_are_not_imputed():
    m=load('study_ranked_candidate_prices');args=fixture()
    args[-1]=args[-1].filter(pl.col('sort_index')<16).with_columns(
        pl.when(pl.col('sort_index')==9).then(101).otherwise(pl.col('replay_time_ns')).alias('replay_time_ns'))
    e,c=m.compute_all(*args)
    assert e['FPM'][0] is None and e['FPM_sign'][0] is None
    assert e['ADV'][0]==3 and e['REV'][0] is None and not e['REV_complete'][0]
    assert e['REV_available_only'][0]==4 and c['REV_cancel_sign'].null_count()==1


def test_frozen_validation_is_exact_and_null_safe():
    m=load('study_ranked_candidate_prices');d=fixture()[0].drop(load('study_primary_kernel_prices').FLAGS)
    m.validate_frozen(d,d)
    for field in ['J','execution_side','candidate_qty_raw']:
        altered=d.with_columns(pl.lit(None).alias(field))
        with pytest.raises(ValueError,match='frozen'):m.validate_frozen(altered,d)
    with pytest.raises(ValueError,match='frozen'):m.validate_frozen(pl.concat([d,d]),d)


def test_summary_has_one_row_per_instrument_and_separate_raw_zero_reporting():
    m=load('study_ranked_candidate_prices');e,c=m.compute_all(*fixture())
    e=e.with_columns(pl.lit(1e-16).alias('FPM'),pl.lit(0,dtype=pl.Int8).alias('FPM_sign'))
    s=m.summarize_all(e)
    assert s.height==1 and s['n'][0]==1
    assert 'stage' not in s.columns and 'ranking' not in s.columns
    assert s['FPM_zero'][0]==1 and s['FPM_raw_positive'][0]==1
    assert s['FPM_median'][0]==0 and s['FPM_raw_median'][0]==1e-16
    assert m.summarize_all(e,roles=True)['n'].sum()==1


@pytest.mark.parametrize('depth',[0.,None])
def test_price_candidacy_does_not_require_a_supported_market_score(depth):
    m=load('study_ranked_candidate_prices');args=fixture()
    args[0]=args[0].with_columns(pl.lit(depth,dtype=pl.Float64).alias('opposite_market_visible_qty_pre_topN'),pl.lit(None,dtype=pl.Float64).alias('WMSCI_market'))
    e,_=m.compute_all(*args)
    assert e.height==1 and e['FPM'][0]==2 and e['WMSCI_market'][0] is None


def test_nonpositive_execution_quantity_is_rejected_as_structural():
    m=load('study_ranked_candidate_prices');args=fixture()
    args[0]=args[0].with_columns(pl.lit(0.).alias('execution_quantity'))
    with pytest.raises(ValueError,match='structural candidate quantity: execution_quantity'):
        m.compute_all(*args)


def test_refuse_output_before_loading_any_source(tmp_path,monkeypatch):
    m=load('study_ranked_candidate_prices');out=tmp_path/'exists';out.mkdir()
    monkeypatch.setattr(m.base,'load_json',lambda *a:pytest.fail('read source before refusal'))
    with pytest.raises(FileExistsError):m.run(out)


def test_physical_duplicates_and_original_quantities_reject():
    m=load('study_ranked_candidate_prices');args=fixture()
    args[2]=pl.concat([args[2],args[2].head(1)])
    with pytest.raises(ValueError,match='duplicate physical'):m.compute_all(*args)
    args=fixture();args[0]=args[0].with_columns(pl.lit(5.).alias('withdrawal_qty_raw'))
    with pytest.raises(ValueError,match='quantity'):m.compute_all(*args)


def test_real_completed_artifacts():
    m=load('study_ranked_candidate_prices');out=m.OUTPUT
    if not (out/'COMPLETE.json').exists():pytest.skip('production run not complete')
    m.base.verify_manifest(out)
    e=pl.read_parquet(out/'private_execution_prices.parquet');c=pl.read_parquet(out/'private_cancel_prices.parquet')
    m.validate_frozen(e,pl.read_parquet(m.EXPECTED))
    assert e.height==6914 and not any(x.startswith('priority_') for x in e.columns)
    assert e.filter(pl.col('WMSCI_market')==0).height==17
    assert c.select(m.PHYSICAL).unique().height==c.height
    assert pl.read_csv(out/'summary.csv').height==3
    assert pl.read_csv(out/'sign_sensitivity.csv').height==15
    assert pl.read_csv(out/'reference_parity.csv')['passed'].all()
