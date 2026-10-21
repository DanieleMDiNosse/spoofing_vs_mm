"""Numerical equality is not a price threshold or a selection rule."""
import importlib.util
from pathlib import Path
import math
import polars as pl


def module():
    path=Path(__file__).parents[2]/'scripts/refine_primary_price_signs.py'
    assert path.exists(), 'numerical sign-accounting policy missing'
    spec=importlib.util.spec_from_file_location('price_signs',path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def fixture():
    key={'instrument':'TEST','partition_id':'P','actor_key':'A','execution_cluster_id':'E'}
    row={**key,'posture_mid':100.,'pre_mid':100.,'execution_vwap':100.,
         'FPM':math.ulp(100.),'ADV':-math.ulp(100.),'REV':math.ulp(100.),'REV_complete':True,
         'priority_top_w_self':True,'priority_top_w_market':True,
         'priority_top_j_given_self':True,'priority_top_j_given_market':False,
         'execution_anchor_mode':'passive','identity_level':'client_original'}
    cancels=[{**key,'raw_quantity_weight':q,'cancel_pre_mid':100.,'cancel_post_mid':100.,'REV_cancel':math.ulp(100.)}
             for q in [1.,3.]]
    return pl.DataFrame([row]),pl.DataFrame(cancels)


def test_roundoff_signs_are_zero_but_raw_values_and_membership_stay():
    m=module();e,c=fixture();a=m.annotate(e,c)
    for name in ['FPM','REV','ADV']:
        assert a[name][0]==e[name][0]
        assert a[name+'_sign'][0]==0
        assert a[name+'_zero_tolerance'][0]>abs(e[name][0])
    assert a['priority_top_j_given_self'][0] is True
    s=m.summarize(a)
    assert s['FPM_positive'].sum()==0 and s['REV_positive'].sum()==0
    assert s['both_positive_complete'].sum()==0


def test_genuine_small_price_difference_is_not_an_economic_floor():
    m=module();e,c=fixture();e=e.with_columns(pl.lit(1e-10).alias('FPM'))
    a=m.annotate(e,c)
    assert a['FPM_sign'][0]==1
    assert a['FPM'][0]==1e-10


def test_missing_reversion_is_not_zero():
    m=module();e,c=fixture()
    e=e.with_columns(pl.lit(None,dtype=pl.Float64).alias('REV'),pl.lit(False).alias('REV_complete'))
    a=m.annotate(e,c);s=m.summarize(a)
    assert a['REV_sign'][0] is None
    assert s.filter((pl.col('ranking')=='self')&(pl.col('stage')=='W'))['REV_missing_or_incomplete'][0]==1


def test_reversion_tolerance_uses_physical_not_age_quantities():
    m=module();e,c=fixture()
    c=c.with_columns(pl.Series('cancel_pre_mid',[100.,1000.]),pl.Series('cancel_post_mid',[100.,1000.]),pl.Series('age_weight',[1.,.001]))
    a=m.annotate(e,c)
    expected=8*(math.ulp(100.)+3*math.ulp(1000.))/4+8*math.ulp(e['REV'][0])
    assert math.isclose(a['REV_zero_tolerance'][0],expected,rel_tol=1e-15)


def test_completed_real_numeric_stage_preserves_raw_and_is_stable():
    import pytest
    m=module();out=m.OUTPUT
    if not out.exists():pytest.skip('private price artifacts unavailable')
    source=m.source_module();source.verify_manifest(out)
    meta=source.load_json(out/'metadata.json')
    for path,h in meta['input_bindings'].items():source.checked_hash(path,h)
    raw=pl.read_parquet(m.SOURCE/'private_execution_prices.parquet')
    actual=pl.read_parquet(out/'private_execution_prices.parquet')
    assert raw.equals(actual.select(raw.columns),null_equal=True)
    assert raw.height==906
    sensitivity=pl.read_csv(out/'sign_sensitivity.csv')
    cols=[c for c in sensitivity.columns if c!='ulp_factor']
    eight=sensitivity.filter(pl.col('ulp_factor')==8).select(cols)
    for factor in [4,16,32]:
        assert eight.equals(sensitivity.filter(pl.col('ulp_factor')==factor).select(cols),null_equal=True)
    strict=sensitivity.filter(pl.col('ulp_factor')==0).select(cols)
    assert strict.equals(pl.read_csv(m.SOURCE/'summary.csv'),null_equal=True)
    assert strict['both_positive_complete'].equals(eight['both_positive_complete'])
    for metric in ['FPM','REV','ADV']:
        assert actual[metric].null_count()==actual[metric+'_sign'].null_count()
