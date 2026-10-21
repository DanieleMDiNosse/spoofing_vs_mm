#!/usr/bin/env python3
"""Cache-only, pre-price conditional WMSCI -> J double-decile release.

No historical consumer is used. The separately named numerical compatibility
config is an API adapter to the unchanged validated pure kernel, not eligibility.
Private outputs retain source identities; publication outputs use local aliases.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import resource
import sys
import time
from pathlib import Path
import numpy as np
import polars as pl
from spoofing_detection.lob.msci_grid_runner import (
    compute_partition_grids, verify_partition_grids, summarize_grid,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = ['instrument','partition_id','actor_key','execution_cluster_id']
STATE_COLUMNS = ['partition_id','actor_key','sort_index','replay_time_ns','DWI','L_bid_topN','L_ask_topN']
EXEC_COLUMNS = KEYS + ['deceptive_side','cluster_first_sort_index','cluster_last_sort_index','cluster_end_replay_time_ns']
SCHEMA = 'wmsci_j_conditional_double_top_decile_v1'


def validate_study_config(config):
    expected = {'schema': SCHEMA, 'first_fraction': 0.1, 'second_fraction': 0.1,
                'tie_break': 'execution_cluster_id_ascending',
                'grid_step_ns': 10000000, 'lag_ns': 50000000, 'horizon_ns': 1000000000,
                'complete_grid_required': True, 'price_selection': False, 'uq_selection': False}
    for key, value in expected.items():
        if key not in config or type(config[key]) is not type(value) or config[key] != value:
            raise ValueError(f'unsupported scientific contract: {key}')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def dump(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,sort_keys=True,allow_nan=False)+'\n')


def kernel_config():
    """Numerical-only legacy API adapter; not this release's cohort contract."""
    return {'schema':'msci_grid_study_v1',
        'source_clock_policy':'spoofing_detection_sequence_replay_trade_fill_sequence_cancel_v3_utc_provisional',
        'grid_policy':'fixed_pre_exact_post_execution_closed_grid_v1',
        'grid_step_ns':10000000,'lag_ns':50000000,'horizon_ns':1000000000,
        'complete_grid_required':True,
        'cohort':'exact_top_positive_W_5pct_then_U_over_Q_gt_1_before_prices',
        'grid_sensitivity_ns':[10000000],'lag_sensitivity_ns':[50000000],
        'lag_sensitivity_grid_ns':10000000,'selection_threshold':None}


def unique(frame,keys=KEYS):
    if frame.select(keys).null_count().sum_horizontal().sum()!=0:
        raise ValueError('null execution identity')
    if frame.select(keys).unique().height!=frame.height:
        raise ValueError('duplicate execution identity')


def exact_top(frame,score):
    chunks=[]
    for part in frame.partition_by('instrument',maintain_order=True):
        k=math.ceil(part.height/10)
        if k:chunks.append(part.sort([score,'execution_cluster_id'],descending=[True,False]).head(k))
    return pl.concat(chunks) if chunks else frame.head(0)


def first_stage(population):
    unique(population)
    # Stable execution ID is unambiguous within an instrument in this source.
    unique(population,['instrument','execution_cluster_id'])
    return exact_top(population.filter(pl.col('WMSCI_event').is_finite() & (pl.col('WMSCI_event')>0)), 'WMSCI_event')


def second_stage(first,summary):
    unique(first);unique(summary)
    if first.select(KEYS).sort(KEYS).equals(summary.select(KEYS).sort(KEYS)) is False:
        raise ValueError('exact first-stage/summary membership mismatch')
    if summary['complete_grid'].dtype!=pl.Boolean or summary['complete_grid'].null_count():
        raise ValueError('complete_grid must be non-null Boolean')
    joined=first.join(summary,on=KEYS,how='left',validate='1:1')
    return exact_top(joined.filter(pl.col('complete_grid') & pl.col('J').is_finite()),'J')


def price_counts(frame):
    result={'n':frame.height}
    for prefix,col in [('FPM','favorable_mid_move_pre_fill'),('REV','post_cancel_mid_reversion')]:
        good=pl.col(col).is_finite().fill_null(False)
        if prefix=='REV':good=good & pl.col('post_cancel_mid_reversion_complete').fill_null(False)
        count=int(frame.select(good.sum()).item())
        result[prefix+'_finite'+('_complete' if prefix=='REV' else '')]=count
        result[prefix+('_missing_or_incomplete' if prefix=='REV' else '_missing')]=frame.height-count
        for label,expr in [('positive',pl.col(col)>0),('zero',pl.col(col)==0),('negative',pl.col(col)<0)]:
            result[prefix+'_'+label]=int(frame.select((good & expr).sum()).item())
    both=(pl.col('favorable_mid_move_pre_fill').is_finite() & (pl.col('favorable_mid_move_pre_fill')>0) &
          pl.col('post_cancel_mid_reversion').is_finite() & (pl.col('post_cancel_mid_reversion')>0) &
          pl.col('post_cancel_mid_reversion_complete'))
    result['both_positive_complete']=int(frame.select(both.sum()).item())
    return result


def compute_and_verify(ex,st,cov,cfg):
    t,s=compute_partition_grids(ex,st,cov,cfg)
    verify_partition_grids(ex,t,s,cfg,states=st,coverage=cov)
    return t,s


def verify_binding(binding):
    p=Path(binding['path'])
    if sha256(p)!=binding['sha256']:raise ValueError(f'input changed: {p}')
    return {'path':str(p.resolve()),'sha256':binding['sha256']}


def public_table(frame,path):
    frame.write_csv(path.with_suffix('.csv'))
    def tex(v):
        if v is None:return '--'
        if isinstance(v,float):return f'{v:.6g}'
        return str(v).replace('_',r'\_').replace('%',r'\%')
    lines=[r'\begin{tabular}{'+'l'*frame.width+'}',r'\hline',
           ' & '.join(tex(x) for x in frame.columns)+r' \\',r'\hline']
    lines+=[' & '.join(tex(x) for x in row)+r' \\' for row in frame.iter_rows()]
    lines += [r'\hline',r'\end{tabular}']
    path.with_suffix('.tex').write_text('\n'.join(lines)+'\n')


def package(pop,first,summary,selected,trajectory,out,pub,cfg):
    pub.mkdir(parents=True,exist_ok=False)
    actors=sorted(first['actor_key'].unique().to_list())
    actor_map={a:f'A{i+1:03d}' for i,a in enumerate(actors)}
    exec_order=first.select(KEYS).sort(KEYS).to_dicts()
    exec_map={tuple(r[k] for k in KEYS):f'E{i+1:03d}' for i,r in enumerate(exec_order)}
    aliases=pl.DataFrame([{**r,'actor_alias':actor_map[r['actor_key']],
        'execution_alias':exec_map[tuple(r[k] for k in KEYS)]} for r in exec_order])
    aliases.write_csv(out/'private_alias_map.csv')
    joined=first.join(summary,on=KEYS,validate='1:1')
    selected_keys=set(tuple(r) for r in selected.select(KEYS).iter_rows())
    all_members=pop.select(KEYS).with_columns(
        pl.Series('first_selected',[tuple(r) in set(tuple(x) for x in first.select(KEYS).iter_rows()) for r in pop.select(KEYS).iter_rows()]),
        pl.Series('second_selected',[tuple(r) in selected_keys for r in pop.select(KEYS).iter_rows()]))
    all_members.write_csv(out/'private_memberships.csv')
    # Reconstruct ranking independently with Python, not the producer sorter.
    for instrument in pop['instrument'].unique().sort():
        rows=pop.filter(pl.col('instrument')==instrument).to_dicts()
        positive=[r for r in rows if r['WMSCI_event'] is not None and math.isfinite(r['WMSCI_event']) and r['WMSCI_event']>0]
        expected=sorted(positive,key=lambda r:(-r['WMSCI_event'],r['execution_cluster_id']))[:math.ceil(len(positive)/10)]
        observed=first.filter(pl.col('instrument')==instrument)
        assert {tuple(r[k] for k in KEYS) for r in expected}==set(observed.select(KEYS).iter_rows())
        valid=[r for r in joined.filter(pl.col('instrument')==instrument).to_dicts() if r['complete_grid'] and r['J'] is not None and math.isfinite(r['J'])]
        expected=sorted(valid,key=lambda r:(-r['J'],r['execution_cluster_id']))[:math.ceil(len(valid)/10)]
        assert {tuple(r[k] for k in KEYS) for r in expected}==set(selected.filter(pl.col('instrument')==instrument).select(KEYS).iter_rows())
    counts=[];prices=[];stats=[];roles=[]
    metrics=['WMSCI_event','J','MSCI_t0','MSCI_horizon','MSCI_max']
    for instrument in pop['instrument'].unique().sort():
        p=pop.filter(pl.col('instrument')==instrument)
        a=joined.filter(pl.col('instrument')==instrument)
        b=selected.filter(pl.col('instrument')==instrument)
        counts.append({'instrument':instrument,'all_executions':p.height,
            'finite_positive_W':p.filter(pl.col('WMSCI_event').is_finite() & (pl.col('WMSCI_event')>0)).height,
            'first_stage':a.height,'complete_finite_J':a.filter(pl.col('complete_grid') & pl.col('J').is_finite()).height,
            'missing_or_incomplete_J':a.filter(~pl.col('complete_grid') | ~pl.col('J').is_finite().fill_null(False)).height,
            'second_stage':b.height,'first_UQ_le_1_or_missing':a.filter(~(pl.col('withdrawal_to_execution_ratio').is_finite() & (pl.col('withdrawal_to_execution_ratio')>1)).fill_null(False)).height,
            'second_UQ_le_1_or_missing':b.filter(~(pl.col('withdrawal_to_execution_ratio').is_finite() & (pl.col('withdrawal_to_execution_ratio')>1)).fill_null(False)).height})
        for stage,frame in [('first_stage',a),('second_stage',b)]:
            prices.append({'instrument':instrument,'stage':stage,**price_counts(frame)})
            for metric in metrics:
                values=frame[metric].drop_nulls().to_numpy();values=values[np.isfinite(values)]
                stats.append({'instrument':instrument,'stage':stage,'metric':metric,'n':len(values),
                    **{label:float(value) if len(values) else None for label,value in
                       zip(['min','q25','median','q75','max'],np.quantile(values,[0,.25,.5,.75,1]) if len(values) else [None]*5)}})
            for row in frame.group_by(['execution_anchor_mode','identity_level']).len().sort(['execution_anchor_mode','identity_level']).to_dicts():
                roles.append({'instrument':instrument,'stage':stage,'role':row['execution_anchor_mode'],
                              'attribution':row['identity_level'],'n':row['len'],'denominator':frame.height})
    public_table(pl.DataFrame(counts),pub/'population_funnel')
    public_table(pl.DataFrame(prices),pub/'conditional_price_outcomes')
    public_table(pl.DataFrame(stats),pub/'profile_summaries')
    public_table(pl.DataFrame(roles),pub/'role_attribution_composition')
    columns=['instrument','actor_alias','execution_alias','execution_anchor_mode','identity_level',
        'WMSCI_event','J','MSCI_t0','MSCI_horizon','MSCI_max','favorable_mid_move_pre_fill',
        'post_cancel_mid_reversion','post_cancel_mid_reversion_complete']
    display=selected.join(aliases,on=KEYS,validate='1:1').select(columns).sort(['instrument','J'],descending=[False,True])
    public_table(display,pub/'selected_executions')
    plot_selected(selected,trajectory,aliases,pub)
    facts={'schema':SCHEMA,'selection_contract':cfg,'counts':counts,'price_outcomes':prices,
           'profile_summaries':stats,'role_attribution':roles,'totals':{
               'all_executions':pop.height,'first_stage':first.height,'second_stage':selected.height,
               'complete_finite_J':summary.filter(pl.col('complete_grid') & pl.col('J').is_finite()).height},
           'verification':{'independent_state_aware_kernel':'all_first_stage_trajectories',
               'independent_python_ranking':'both_stages_all_instruments','exact_source_membership_join':True},
           'sensitivities':'No new-cohort sensitivities calculated. Historical cohort sensitivity claims cannot be reused.',
           'limitations':['Ferrari and Nexi each cover one day; Risanamento covers multiple trading days; descriptive, not causal or evidence of intent.',
               'Conditional observed-event coverage witness; external feed completeness not verified; protected cache integrity assumed.',
               'Provisional common UTC origin of mixed TRADETIME and SEQUENCETIME; integer ns comparisons do not imply effective ns precision.',
               'Fixed canonical pre-fill reference has no staleness bound; exact post-fill state at zero, later nodes canonical as-of.',
               'J uses replay-time cluster end, W/price quantities retain their frozen source mixed-clock policy.',
               'No matched controls, uncertainty estimates, counterfactual, or new sensitivity analyses.']}
    dump(pub/'facts.json',facts)
    (pub/'README.md').write_text('''# Conditional WMSCI → J double-decile package

Population: executions, per instrument. First select exactly ceil(0.1 n) among finite positive WMSCI; then select ceil(0.1 m) among complete finite J on the first-stage population. Ties use execution ID ascending. Zero J remains eligible. No U/Q or price selection.

J = max over grid starts of max(MSCI(t+50 ms)-MSCI(t),0), on the closed [0,1 s] 10 ms grid (101 nodes, 96 lag pairs), only when every node is valid. Invalid/incomplete J is missing, not zero. The reference is the latest actor state with source sort_index strictly preceding the first fill, fixed across the path; t=0 uses the exact last-fill source state. Later nodes use canonical as-of state. Source support is conditional on observed normalized events, not external feed completeness.

FPM is the source favorable_mid_move_pre_fill (signed price difference, source price units). REV is post_cancel_mid_reversion, the quantity-weighted complete finite reversion across all assigned positive-quantity cancellations, with the frozen 2 s cancellation-anchored horizon. REV is missing unless every assigned positive-quantity cancellation has a finite outcome; available-only REV is not substituted. Positive means strictly >0; zero and negative outcomes remain. Columns named n are execution denominators. Price counts are unconditional within each stated selected stage, never an additional selection. Both-positive counts are descriptive intersections.

profile_summaries uses observed finite values and NumPy linear quantiles; no uncertainty intervals. role_attribution_composition preserves execution role and client/firm-fallback attribution. All paper actor/execution labels are report-local aliases, not claims of global anonymity; original keys and alias map remain in private outputs. Trajectory figure: all second-stage paths faint, highlighted minimum/lower-middle/maximum J representatives; fixed common MSCI axis [-1,2], no confidence bands. No sensitivity from the historical cohort is valid for this release.

CSV files are machine-readable companions; TeX fragments are unstyled tables for parent integration (not a compiled manuscript). facts.json contains exact counts and limitations.
''')
    return facts


def plot_selected(selected,trajectory,aliases,pub):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':11,'axes.labelsize':12,'axes.titlesize':13,'legend.fontsize':10})
    instruments=selected['instrument'].unique().sort().to_list()
    fig,axes=plt.subplots(len(instruments),1,figsize=(9,3.25*len(instruments)),sharex=True,squeeze=False)
    reps=[]
    for ax,instrument in zip(axes[:,0],instruments):
        cases=selected.filter(pl.col('instrument')==instrument).sort(['J','execution_cluster_id'])
        for row in cases.to_dicts():
            t=trajectory.filter(pl.all_horizontal([pl.col(k)==row[k] for k in KEYS])).sort('grid_offset_ns')
            ax.step(t['grid_offset_ns'].to_numpy()/1e9,t['MSCI'].to_numpy(),where='post',color='#64748b',alpha=.22,lw=1)
        positions=sorted(set([0,(cases.height-1)//2,cases.height-1]))
        for position,color in zip(positions,['#2563eb','#d97706','#dc2626']):
            row=cases.row(position,named=True)
            t=trajectory.filter(pl.all_horizontal([pl.col(k)==row[k] for k in KEYS])).sort('grid_offset_ns')
            alias=aliases.filter(pl.all_horizontal([pl.col(k)==row[k] for k in KEYS])).row(0,named=True)
            label=f"{alias['execution_alias']} / {alias['actor_alias']}  J={row['J']:.3f}"
            ax.step(t['grid_offset_ns'].to_numpy()/1e9,t['MSCI'].to_numpy(),where='post',color=color,lw=2,label=label)
            reps.append({'instrument':instrument,'execution_alias':alias['execution_alias'],'actor_alias':alias['actor_alias'],'J':row['J'],'position_sorted_J':position})
        ax.set_title(f'{instrument.title()} — {cases.height} conditional top-J executions',loc='left',pad=64)
        ax.set_ylabel('MSCI (dimensionless)');ax.set_ylim(-1.05,2.05);ax.set_xlim(-.015,1.015)
        ax.set_xticks(np.linspace(0,1,6));ax.set_yticks([-1,0,1,2])
        ax.grid(alpha=.22);ax.axhline(0,color='black',lw=.5)
        ax.legend(loc='lower left',bbox_to_anchor=(0,1.01),ncol=1,frameon=False)
    axes[-1,0].set_xlabel('Time after exact last-fill replay state (s)')
    fig.tight_layout(h_pad=3)
    fig.canvas.draw();renderer=fig.canvas.get_renderer()
    for ax in axes[:,0]:
        for artist in [ax._left_title,ax.xaxis.label,ax.yaxis.label,ax.get_legend(),*ax.get_xticklabels(),*ax.get_yticklabels()]:
            if artist.get_visible() and (not hasattr(artist,'get_text') or artist.get_text()):
                box=artist.get_window_extent(renderer)
                if box.x0 < -1 or box.y0 < -1 or box.x1>fig.bbox.x1+1 or box.y1>fig.bbox.y1+1:
                    raise ValueError(f'figure label clipped: {artist}')
        if ax._left_title.get_window_extent(renderer).overlaps(ax.get_legend().get_window_extent(renderer)):
            raise ValueError('figure title and legend overlap')
    for ext in ['pdf','png','svg']:fig.savefig(pub/f'selected_msci_paths.{ext}',dpi=180)
    plt.close(fig)
    pl.DataFrame(reps).write_csv(pub/'figure_representatives.csv')
    dump(pub/'figure_geometry_verification.json',{'canvas_bounds_checked':True,'fonts_pt':{'body':11,'axes':12,'titles':13,'legend':10},'common_MSCI_axis':[-1.05,2.05]})


def run(args):
    start=time.perf_counter();out=Path(args.output_root).resolve();pub=Path(args.paper_root).resolve()
    if pub.exists():raise ValueError('paper root already exists')
    out.mkdir(parents=True,exist_ok=True)
    if (out/'prelaunch_manifest.json').exists():raise ValueError('analysis root already initialized; preserve historic run')
    config_path=Path(args.config).resolve();cfg=json.loads(config_path.read_text())
    frozen=json.loads((ROOT/'configs/wmsci_j_top10_study.json').read_text())
    if cfg!=frozen or cfg['schema']!=SCHEMA:raise ValueError('unsupported selection config')
    validate_study_config(cfg)
    prepath=ROOT/cfg['preflight'];covpath=ROOT/cfg['coverage_witness']
    pre=json.loads(prepath.read_text());cov=json.loads(covpath.read_text())
    if pre['format']!='msci_grid_preflight_v1' or cov['schema']!='msci_grid_observed_event_coverage_witness_v1':raise ValueError('invalid frozen preflight/coverage contract')
    if cov['external_feed_completeness_verified'] is not False:raise ValueError('unexpected external coverage claim')
    bindings={}
    for key,b in pre['input_digests'].items():bindings[key]=verify_binding(b)
    for entry in pre['instruments']:
        name=entry['instrument']
        if Path(entry['metadata']['path']).parent != (ROOT/cfg['source_root']/name).resolve():raise ValueError('source root mismatch')
        for key,b in entry['artifacts'].items():
            bindings[name+':'+key]=verify_binding(b)
            if cov['instruments'][name]['artifacts'][key]!=b:raise ValueError('coverage source identity mismatch')
        for key,b in entry['coverage_historical_evidence']['identity']['source_bindings'].items():
            verify_binding({'path':b['current_path'],'sha256':b['sha256']})
    qroot=ROOT/cfg['quantity_root']
    for name in ['metadata.json','refreshed_execution_diagnostics.csv','step_memberships.csv']:
        p=qroot/name;bindings[name]={'path':str(p.resolve()),'sha256':sha256(p)}
    for p in [prepath,covpath,config_path,Path(__file__),ROOT/'tests/lob/test_wmsci_j_top10_study.py',
              ROOT/'src/spoofing_detection/lob/msci_grid.py',ROOT/'src/spoofing_detection/lob/msci_grid_runner.py',ROOT/'src/spoofing_detection/lob/msci_grid_store.py',ROOT/'src/spoofing_detection/lob/spoofing_metrics.py']:
        bindings[str(p.relative_to(ROOT))]={'path':str(p),'sha256':sha256(p)}
    manifest={'schema':SCHEMA,'command':sys.argv,'inputs_and_code':bindings,'actual_selection_config':cfg,
              'numerical_kernel_compatibility_config':kernel_config(),
              'numerical_adapter_is_not_selection_contract':True,'python':platform.python_version(),
              'packages':{p:importlib.metadata.version(p) for p in ['polars','numpy','matplotlib','pyarrow','pytest']}}
    dump(out/'prelaunch_manifest.json',manifest)
    pop=pl.read_csv(qroot/'refreshed_execution_diagnostics.csv',infer_schema_length=10000)
    unique(pop);unique(pop,['instrument','execution_cluster_id'])
    step=pl.read_csv(qroot/'step_memberships.csv',infer_schema_length=10000)
    unique(step,['instrument','execution_cluster_id'])
    if pop.select(['instrument','execution_cluster_id']).sort(['instrument','execution_cluster_id']).equals(step.select(['instrument','execution_cluster_id']).sort(['instrument','execution_cluster_id'])) is False:
        raise ValueError('quantity membership universe mismatch')
    check=pop.join(step.select('instrument','execution_cluster_id','positive_W'),on=['instrument','execution_cluster_id'],validate='1:1')
    if check['positive_W'].null_count() or check.filter(pl.col('positive_W').ne_missing(pl.col('WMSCI_event').is_finite() & (pl.col('WMSCI_event')>0))).height:raise ValueError('positive W flags disagree')
    first=first_stage(pop);first.write_csv(out/'private_first_stage.csv')
    dump(out/'checkpoint.json',{'phase':'first_stage_frozen','n':first.height})
    trajectories=[];summaries=[];timings=[]
    for entry in pre['instruments']:
        instrument=entry['instrument'];tstart=time.perf_counter()
        a=first.filter(pl.col('instrument')==instrument)
        ex=pl.read_parquet(entry['artifacts']['execution_metrics']['path'],columns=EXEC_COLUMNS[1:]).with_columns(pl.lit(instrument).alias('instrument'))
        unique(ex)
        source=ex.join(a.select(KEYS),on=KEYS,how='semi')
        if source.height!=a.height:raise ValueError('source exact membership mismatch')
        for field in ['deceptive_side','cluster_first_sort_index','cluster_last_sort_index']:
            compare=source.select(KEYS+[field]).join(a.select(KEYS+[field]),on=KEYS,suffix='_quantity',validate='1:1')
            if compare.filter(pl.col(field).ne_missing(pl.col(field+'_quantity'))).height:raise ValueError(f'quantity/source {field} mismatch')
        source.write_parquet(out/f'{instrument}_private_execution_anchors.parquet')
        # Project seven fields and filter requested partition/actors BEFORE materialization.
        st=pl.scan_parquet(entry['artifacts']['state_time_series']['path']).select(STATE_COLUMNS).join(
            a.select('partition_id','actor_key').unique().lazy(),on=['partition_id','actor_key'],how='semi').collect(engine='streaming')
        support=cov['instruments'][instrument]['partition_coverage_witness']
        its=[];iss=[]
        for local in st.partition_by(['partition_id','actor_key'],maintain_order=True):
            pid=local['partition_id'][0];actor=local['actor_key'][0]
            e=source.filter((pl.col('partition_id')==pid)&(pl.col('actor_key')==actor))
            t,s=compute_and_verify(e,local,support,kernel_config())
            rich=[]
            for frame in t.partition_by(KEYS,maintain_order=True):
                values=summarize_grid(frame,lag_ns=50000000)
                rich.append({**frame.select(KEYS).row(0,named=True),**{k:values[k] for k in ['MSCI_t0','MSCI_horizon','MSCI_max','MSCI_max_offset_ns']}})
            s=s.join(pl.DataFrame(rich),on=KEYS,validate='1:1')
            its.append(t);iss.append(s)
        t=pl.concat(its);s=pl.concat(iss)
        if s.height!=source.height:raise ValueError('an execution actor lacks retained states')
        t.write_parquet(out/f'{instrument}_private_trajectories.parquet');s.write_csv(out/f'{instrument}_private_summary.csv')
        trajectories.append(t);summaries.append(s)
        timings.append({'instrument':instrument,'seconds':time.perf_counter()-tstart,'projected_selected_actor_state_rows':st.height,
                        'trajectories':s.height,'grid_nodes':t.height,'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss})
        del st
        dump(out/'checkpoint.json',{'phase':'primary_computed_and_independently_verified','completed_instruments':[x['instrument'] for x in timings],'timings':timings})
        print(json.dumps(timings[-1]),flush=True)
    trajectory=pl.concat(trajectories);summary=pl.concat(summaries)
    selected=second_stage(first,summary)
    summary.write_csv(out/'private_all_first_stage_summary.csv');selected.write_csv(out/'private_second_stage.csv')
    facts=package(pop,first,summary,selected,trajectory,out,pub,cfg)
    # Verify persisted primary products, public counts and membership before receipt.
    for entry in pre['instruments']:
        name=entry['instrument'];stored=pl.read_parquet(out/f'{name}_private_trajectories.parquet')
        expected=trajectory.filter(pl.col('instrument')==name)
        if not stored.equals(expected):raise ValueError('persisted trajectory differs')
    census=pl.read_csv(pub/'population_funnel.csv')
    assert census['first_stage'].sum()==first.height and census['second_stage'].sum()==selected.height
    for key,b in bindings.items():verify_binding(b)
    receipt={'schema':SCHEMA,'status':'verified','elapsed_seconds':time.perf_counter()-start,
        'peak_process_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'timings':timings,
        'all_primary_trajectories_state_aware_verified':summary.height,'all_primary_nodes':trajectory.height,
        'membership_exact':True,'counts':facts['totals'],'source_hashes_rechecked_after_run':True}
    dump(out/'verification.json',receipt)
    dump(out/'checkpoint.json',{'phase':'complete_verified','counts':facts['totals']})
    for directory in [out,pub]:
        dump(directory/'artifact_manifest.json',{str(p.relative_to(directory)):sha256(p) for p in sorted(directory.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','run.log','resource_usage.txt','exit_status.txt']})
    print(json.dumps(receipt),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default=str(ROOT/'configs/wmsci_j_top10_study.json'))
    parser.add_argument('--output-root',required=True)
    parser.add_argument('--paper-root',required=True)
    args=parser.parse_args();run(args)

if __name__=='__main__':main()
