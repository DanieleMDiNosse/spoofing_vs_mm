"""Small executable feed, reused by CLI and scientific integration checks."""
import hashlib
import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

from spoofing_detection.lob.spoofing_metrics import compute_exploratory_metrics

ROOT = Path(__file__).resolve().parents[2]
KERNEL = {side: {i: 1. for i in range(1, 11)} for side in ('bid', 'ask')}


def raw_fixture(reload=False):
    rows=[]
    # A coherent candidate-posture episode with both execution routes.
    events=[(1,1,'B0',1,99.,100.,'OTHER',None), (2,1,'A0',2,103.,100.,'OTHER',None),
            (3,11 if reload else 1,'D1',1,98.,100.,'C',None),
            (4,1,'P',2,102.,20.,'C',None), (5,1,'D2',1,97.,20.,'C',None),
            (6,1,'B1',1,100.,100.,'OTHER',None), (7,3,'P',2,102.,14.,'C','passive'),
            (8,3,'A',2,100.,0.,'C','aggressive'), (9,4,'D1',1,98.,0.,'C',None),
            (10,4,'D2',1,97.,0.,'C',None), (11,4,'B1',1,100.,0.,'OTHER',None),
            (13,1,'COVER',1,98.,1.,'C',None)]
    for seq,event,oid,side,price,qty,actor,anchor in events:
        stamp=f'2024-01-02 09:30:{seq:02d}'
        rows.append({'TRADEDATE':'2024-01-02','MIC':'XMIL','MARKETCODE':'MTA','SYMBOLINDEX':123,'EMM (*)':1,
            'SEQUENCETIME':stamp,'BOOKIN':stamp,'BOOKOUTTIME':stamp,'TRADETIME':stamp if anchor else None,
            'HDR_APPLKEYSEQUENCENUMBER':seq,'HDR_HWMSEQUENCENUMBER':seq,'HDR_OFFSETID':seq,'ROW_NUMBER':seq,
            'ORDEREVENTTYPE (*)':event,'ORDERID':oid,'ORDERPRIORITY':str(seq),'ORDERSIDE (*)':side,
            'ORDERPX':price,'ORDERQTY':qty,'DISPLAYEDQTY':qty,'LEAVESQTY':qty,
            'LASTSHARES':6. if anchor else None,'LASTTRADEDPX':price if anchor else None,
            'ORDERTYPE (*)':2,'TIMEINFORCE (*)':0,'FIRMID':'F','NMSC_ORIGINALCLIENTIDSHORTCODE':actor,
            'ORDER_TRADINGCAPACITY (*)':3,'PASSIVEORDER':'Y' if anchor=='passive' else 'N',
            'AGGRESSIVEORDER':'Y' if anchor=='aggressive' else 'N'})
    return pl.DataFrame(rows)


def compute(raw):
    return compute_exploratory_metrics(raw,top_n=10,tick_size=1.,window_seconds=2.,
        withdrawal_window_seconds=3.,reversion_horizon_seconds=2.,empirical_kernel_weights=KERNEL)


def test_core_emits_joint_episode_membership_and_latest_posture():
    result=compute(raw_fixture())
    assert result.execution_metrics.height==2
    assert result.execution_metrics['posture_state_sort_index'].to_list()==[5,5]
    assert result.episode_result.episodes.height==1
    row=result.episode_result.episodes.row(0,named=True)
    assert row['total_execution_quantity']==pytest.approx(12.)
    assert row['withdrawn_quantity']==pytest.approx(120.)
    assert row['spoofing_compatible_episode'] is True
    assert result.execution_metrics['episode_id'].n_unique()==1
    assert result.execution_metrics['episode_strict_detection'].sum()==1
    assert result.spoofing_compatible_events.height==1
    assert result.spoofing_compatible_events['episode_id'].n_unique()==1


def test_reload_only_observation_does_not_establish_placement():
    result=compute(raw_fixture(reload=True))
    assert result.execution_metrics['favorable_mid_move_pre_fill'].null_count()==2
    assert result.episode_result.episodes['spoofing_compatible_episode'].to_list()==[False]


def test_raw_two_days_never_pool_episode_or_actor_day():
    raw=raw_fixture()
    second=raw.with_columns(pl.lit('2024-01-03').alias('TRADEDATE'),
        *[pl.col(k).str.replace('2024-01-02','2024-01-03').alias(k)
          for k in ('SEQUENCETIME','BOOKIN','BOOKOUTTIME','TRADETIME')])
    result=compute(pl.concat([raw,second]))
    assert result.episode_result.episodes.height==2
    assert result.episode_result.actor_day_summary.height==2


def test_cli_persists_episode_artifacts_and_semantics(tmp_path):
    spec=importlib.util.spec_from_file_location('episode_cli',ROOT/'scripts/compute_spoofing_metrics.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw=tmp_path/'input.parquet'; raw_fixture().write_parquet(raw)
    kernel=tmp_path/'kernel.csv'
    pl.DataFrame([dict(side=s,rank=r,kernel_weight=w) for s,weights in KERNEL.items() for r,w in weights.items()]).write_csv(kernel)
    config=json.loads((ROOT/'configs/spoofing_detection_parameters.json').read_text())
    config['metrics'].update(empirical_depth_kernel=str(kernel),tick_size=1.,window_seconds=2.,withdrawal_window_seconds=3.)
    cfg=tmp_path/'config.json'; cfg.write_text(json.dumps(config))
    out=tmp_path/'run'
    module.main(['--config',str(cfg),'--input',str(raw),'--output-dir',str(out)])
    metadata=json.loads((out/'metadata.json').read_text())
    assert metadata['episode_semantics']['primary_unit']=='candidate_posture_episode'
    assert metadata['execution_role_provenance']['verification_status']=='unverified_upstream_mapping'
    for name in ('candidate_episodes','episode_cluster_members','episode_withdrawals','episode_anchor_summary','actor_day_episode_summary'):
        path=out/(name+'.parquet')
        assert pl.read_parquet(path).height>0
        assert hashlib.sha256(path.read_bytes()).hexdigest()==metadata['artifact_hashes'][name]


def test_grid_writes_episodes_and_rejects_modified_cache(tmp_path):
    spec=importlib.util.spec_from_file_location('episode_grid',ROOT/'scripts/run_multilevel_spoofing_grid.py')
    assert spec is not None and spec.loader is not None
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw=tmp_path/'input.parquet'; raw_fixture().write_parquet(raw)
    kernel=tmp_path/'kernel.csv'
    pl.DataFrame([dict(side=s,rank=r,kernel_weight=w) for s,weights in KERNEL.items() for r,w in weights.items()]).write_csv(kernel)
    config=json.loads((ROOT/'configs/spoofing_detection_parameters.json').read_text())
    config['grid'].update(empirical_depth_kernel=str(kernel),tick_size=1.,depth_grid=[10],
        window_seconds=2.,withdrawal_window_seconds=3.,reversion_horizon_seconds=2.,max_deceptive_order_age_seconds=90.)
    cfg=tmp_path/'config.json'; cfg.write_text(json.dumps(config))
    out=tmp_path/'run'
    module.main(['--config',str(cfg),'--input',str(raw),'--output-dir',str(out)])
    paths=module._depth_output_paths(out,10)
    assert pl.read_parquet(out/'topn_10/candidate_episodes.parquet').height==1
    metadata_path=out/'metadata.json'
    metadata=json.loads(metadata_path.read_text())
    expected={'analysis_semantics_version':metadata['analysis_semantics_version']}
    assert module._can_reuse_depth_outputs(paths,metadata_path=metadata_path,expected_metadata=expected,top_n=10)
    original=pl.read_parquet(paths['execution_metrics'])
    original.with_columns(pl.lit(-10.).alias('MSCI')).write_parquet(paths['execution_metrics'])
    assert not module._can_reuse_depth_outputs(paths,metadata_path=metadata_path,expected_metadata=expected,top_n=10)
