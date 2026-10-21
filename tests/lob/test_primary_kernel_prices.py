"""Price estimands and the real artifact writer, independent of raw replay."""
import importlib.util
from pathlib import Path
import json
import polars as pl
import pytest


def module():
    path = Path(__file__).parents[2] / 'scripts/study_primary_kernel_prices.py'
    assert path.exists(), 'cache-only price implementation is missing'
    spec = importlib.util.spec_from_file_location('primary_prices', path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fixture(side='ask'):
    key = dict(instrument='TEST', partition_id='P', actor_key='A', execution_cluster_id='E')
    ex = dict(**key, execution_side=side, cluster_first_sort_index=10,
              cluster_start_replay_time_ns=100, cluster_end_fill_anchor_ns=100,
              execution_quantity=2., execution_vwap=103., withdrawal_qty_raw=4.,
              execution_anchor_mode='passive', identity_level='client_original',
              priority_top_w_self=True, priority_top_w_market=True,
              priority_top_j_given_self=True, priority_top_j_given_market=False)
    ca = [dict(**key, execution_sort_index=10, deceptive_order_first_seen_sort_index=2,
               deceptive_order_placement_observed=True, deceptive_order_first_seen_replay_ns=20)]
    au = [dict(**key, candidate_order_id=str(i), cancel_sort_index=i, assigned_flag=True,
               attributed_cancel_visible_qty=q, age_weight=w) for i,q,w in [(12,1.,.01),(14,3.,1.)]]
    cn = [dict(partition_id='P',actor_key='A',ORDERID=str(i),sort_index=i,cancellation_removal_ns=t)
          for i,t in [(12,200),(14,300)]]
    me = [dict(partition_id='P',actor_key='A',execution_cluster_id='E',child_fill_qty=2.,child_fill_price=103.)]
    st = [dict(partition_id='P',actor_key='A',sort_index=i,replay_time_ns=t,market_mid=v)
          for i,t,v in [(2,20,100.),(9,100,102.),(11,150,105.),(13,250,107.),
                         (15,2_000_000_200,101.),(16,2_000_000_300,100.),(17,2_000_000_301,999.)]]
    return [pl.DataFrame(x) for x in [[ex],ca,au,cn,me,st]]


@pytest.mark.parametrize('side,sign',[('ask',1),('bid',-1)])
def test_prices_and_raw_quantity_weighted_reversion(side,sign):
    m=module(); e,c=m.compute(*fixture(side))
    r=e.row(0,named=True)
    assert r['FPM']==2*sign and r['ADV']==3*sign
    assert r['REV']==pytest.approx(6.25*sign)
    assert c['cancel_target_ns'].to_list()==[2_000_000_200,2_000_000_300]
    assert c['cancel_post_sort_index'].to_list()==[15,16]
    assert r['pre_sort_index']==9  # equal-time canonical predecessor is valid


def test_missing_one_cancel_never_imputes_zero_or_available_only():
    m=module(); args=fixture(); args[-1]=args[-1].filter(pl.col('sort_index')<16)
    e,c=m.compute(*args);r=e.row(0,named=True)
    assert r['REV'] is None and r['REV_missing_cancel_count']==1
    assert r['REV_available_only']==4.
    assert c['REV_cancel'].null_count()==1


@pytest.mark.parametrize('mutation',['missing','unobserved','future'])
def test_exact_placement_no_substitution(mutation):
    m=module(); args=fixture()
    if mutation=='missing': args[-1]=args[-1].filter(pl.col('sort_index')!=2)
    elif mutation=='unobserved': args[1]=args[1].with_columns(pl.lit(False).alias('deceptive_order_placement_observed'))
    else: args[-1]=args[-1].with_columns(pl.when(pl.col('sort_index')==2).then(101).otherwise(pl.col('replay_time_ns')).alias('replay_time_ns'))
    e,_=m.compute(*args)
    assert e['FPM'][0] is None and e['ADV'][0] is None


def test_partition_actor_separation_and_same_time_tie():
    m=module();args=fixture()
    extras=pl.DataFrame([dict(partition_id='P',actor_key='A',sort_index=18,replay_time_ns=2_000_000_300,market_mid=98.),
                        dict(partition_id='Q',actor_key='A',sort_index=19,replay_time_ns=2_000_000_300,market_mid=1.),
                        dict(partition_id='P',actor_key='B',sort_index=19,replay_time_ns=2_000_000_300,market_mid=1.)])
    args[-1]=pl.concat([args[-1],extras]);e,c=m.compute(*args)
    assert c['cancel_post_sort_index'].to_list()==[15,18]
    assert e['REV'][0]==pytest.approx(7.75)


@pytest.mark.parametrize('flag',['priority_top_w_self','priority_top_w_market','priority_top_j_given_self','priority_top_j_given_market'])
def test_native_flags_reject_null_and_nonboolean(flag):
    m=module()
    for value in [None,1]:
        args=fixture();args[0]=args[0].with_columns(pl.lit(value).alias(flag))
        with pytest.raises(ValueError,match='Boolean'): m.compute(*args)


def test_quantity_and_conditional_membership_fail_closed():
    m=module();args=fixture();args[0]=args[0].with_columns(pl.lit(5.).alias('withdrawal_qty_raw'))
    with pytest.raises(ValueError,match='quantity'): m.compute(*args)
    args=fixture();args[0]=args[0].with_columns(pl.lit(False).alias('priority_top_w_self'))
    with pytest.raises(ValueError,match='subset'): m.compute(*args)


def test_real_writer_manifest_summary_and_no_overwrite(tmp_path):
    m=module();e,c=m.compute(*fixture());out=tmp_path/'bundle'
    m.write_bundle(out,e,c,{'fixture':True},pl.DataFrame({'metric':['FPM'],'n':[1]}))
    summary=pl.read_csv(out/'summary.csv');assert summary.height==4
    assert summary.filter((pl.col('stage')=='J')&(pl.col('ranking')=='market'))['n'][0]==0
    assert summary['both_positive_complete'].sum()==3
    marker=json.loads((out/'COMPLETE.json').read_text())
    assert marker['status']=='primary_prices_verified'
    assert marker['artifact_manifest_sha256']==m.sha256(out/'artifact_manifest.json')
    m.verify_manifest(out)
    with pytest.raises(FileExistsError):m.write_bundle(out,e,c,{},pl.DataFrame())


def test_epoch_ns_and_future_canonical_index_never_round_or_leak():
    m=module();args=fixture();epoch=1_713_862_844_678_104_463
    args[0]=args[0].with_columns((pl.col('cluster_start_replay_time_ns')+epoch),
                                (pl.col('cluster_end_fill_anchor_ns')+epoch))
    args[1]=args[1].with_columns(pl.col('deceptive_order_first_seen_replay_ns')+epoch)
    args[3]=args[3].with_columns(pl.col('cancellation_removal_ns')+epoch)
    args[-1]=args[-1].with_columns(pl.col('replay_time_ns')+epoch)
    e,c=m.compute(*args)
    assert c['cancel_target_ns'].to_list()==[epoch+2_000_000_200,epoch+2_000_000_300]
    assert c['cancel_post_sort_index'].to_list()==[15,16]
    assert e['REV'][0]==6.25
    # A future-clock canonical predecessor must not fall back to an older valid row.
    args[-1]=args[-1].with_columns(pl.when(pl.col('sort_index')==9).then(epoch+101)
                                 .otherwise(pl.col('replay_time_ns')).alias('replay_time_ns'))
    e,_=m.compute(*args)
    assert e['FPM'][0] is None and e['ADV'][0]==3.


def test_zero_j_does_not_change_conditional_membership():
    m=module();args=fixture();args[0]=args[0].with_columns(pl.lit(0.).alias('J'))
    e,c=m.compute(*args)
    assert m.summary(e).filter((pl.col('ranking')=='self')&(pl.col('stage')=='J'))['n'][0]==1


def test_manifest_tampering_and_invalid_digest_fail(tmp_path):
    m=module();e,c=m.compute(*fixture());out=tmp_path/'bundle'
    m.write_bundle(out,e,c,{},pl.DataFrame({'n':[1]}))
    (out/'summary.csv').write_text('tampered')
    with pytest.raises(ValueError,match='hash mismatch'):m.verify_manifest(out)
    with pytest.raises(ValueError,match='invalid SHA256'):m.checked_hash(out/'summary.csv','z'*64)


def test_real_completed_artifacts_native_membership_and_all_summaries():
    m=module();out=m.DEFAULT_OUT
    if not out.exists():pytest.skip('run the cache-only production stage first')
    m.verify_manifest(out)
    e=pl.read_parquet(out/'private_execution_prices.parquet')
    c=pl.read_parquet(out/'private_cancel_prices.parquet')
    s=pl.read_csv(out/'summary.csv')
    assert s.height==12 and e.height==906 and c.height==2482
    assert e.select(m.KEYS).unique().height==e.height
    assert s.to_dicts()==m.summary(e).to_dicts()
    for ins,root in m.ROOTS.items():
        native=pl.read_parquet(root/'analysis/population.parquet').filter(
            (pl.col('age_policy')=='exp_h30')&(pl.col('horizon_ns')==10_000_000_000))
        for flag in m.FLAGS:
            assert set(native.filter(pl.col(flag)).select(m.KEYS).iter_rows())==set(
                e.filter((pl.col('instrument')==ins)&pl.col(flag)).select(m.KEYS).iter_rows())
    for metric,missing in [('FPM','FPM_missing'),('ADV','ADV_missing'),('REV','REV_missing_or_incomplete')]:
        assert s.select((pl.col(metric+'_positive')+pl.col(metric+'_zero')+
                         pl.col(metric+'_negative')+pl.col(missing)==pl.col('n')).all()).item()
    assert c.filter(pl.col('cancel_target_ns')!=pl.col('cancellation_removal_ns')+2_000_000_000).height==0
    supported=c.filter(pl.col('REV_cancel').is_not_null())
    assert supported.filter((pl.col('cancel_pre_sort_index')>=pl.col('cancel_sort_index'))|
        (pl.col('cancel_pre_replay_time_ns')>pl.col('cancellation_removal_ns'))|
        (pl.col('cancel_post_sort_index')<pl.col('cancel_sort_index'))|
        (pl.col('cancel_post_replay_time_ns')>pl.col('cancel_target_ns'))|
        ~pl.col('actor_coverage_through_target')).height==0
    parity=pl.read_csv(out/'historical_reference_parity.csv')
    assert parity.height==18 and parity['passed'].all() and parity['membership_exact'].all()
    assert parity['max_abs_error'].max()<1e-12
    # Historical source CSVs are also authenticated to their original producer manifest.
    old=Path('outputs/new_kernel_h1_top10_study')
    manifest=json.loads((old/'artifact_manifest.json').read_text())
    for name in ['private_first_stage.csv','private_second_stage.csv']:
        m.checked_hash(old/name,manifest[name])
