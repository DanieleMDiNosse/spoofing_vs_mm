#!/usr/bin/env python3
"""Render verified double-decile observations for the simplified manuscript."""
from pathlib import Path
import argparse
import hashlib
import json
import math
import shutil
import sys
import numpy as np
import polars as pl

KEYS = ['instrument','partition_id','actor_key','execution_cluster_id']


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()


def verify_manifest(root):
    declared=json.loads((root/'artifact_manifest.json').read_text())
    for name,digest in declared.items():
        assert sha(root/name)==digest, name


def table(path,caption,label,headers,rows,alignment):
    lines=[r'\begin{table}[!htbp]',r'\centering',r'\small',r'\setlength{\tabcolsep}{4pt}',
           r'\caption{'+caption+'}',r'\label{'+label+'}',r'\begin{tabular}{'+alignment+'}',r'\toprule',
           ' & '.join(headers)+r' \\',r'\midrule']
    lines += [' & '.join(str(x) for x in row)+r' \\' for row in rows]
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    path.write_text('\n'.join(lines)+'\n')


def render_paths(analysis, selected, output):
    """Style-only render from frozen paths; preserve representative membership."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    aliases=pl.read_csv(analysis/'private_alias_map.csv',infer_schema_length=10000)
    representatives=[]
    with plt.rc_context({'font.size':13,'axes.labelsize':14,'axes.titlesize':14,'legend.fontsize':11}):
        instruments=sorted(selected['instrument'].unique())
        fig,axes=plt.subplots(len(instruments),1,figsize=(7,8.5),sharex=True,squeeze=False)
        for ax,instrument in zip(axes[:,0],instruments):
            cases=selected.filter(pl.col('instrument')==instrument).sort(['J','execution_cluster_id'])
            trajectories=pl.read_parquet(analysis/f'{instrument}_private_trajectories.parquet')
            def path(row):
                return trajectories.filter(pl.all_horizontal([pl.col(k)==row[k] for k in KEYS])).sort('grid_offset_ns')
            for row in cases.to_dicts():
                frame=path(row);values=frame['MSCI'].to_numpy()
                assert np.all((values>=-1.05)&(values<=2.05))
                ax.step(frame['grid_offset_ns'].to_numpy()/1e9,values,where='post',color='#64748b',alpha=.22,lw=1)
            for position,color in zip(sorted(set([0,(cases.height-1)//2,cases.height-1])),['#2563eb','#d97706','#dc2626']):
                row=cases.row(position,named=True);frame=path(row)
                alias=aliases.filter(pl.all_horizontal([pl.col(k)==row[k] for k in KEYS])).row(0,named=True)
                label=f"{alias['execution_alias']} / {alias['actor_alias']}  J={row['J']:.3f}"
                ax.step(frame['grid_offset_ns'].to_numpy()/1e9,frame['MSCI'].to_numpy(),where='post',color=color,lw=2,label=label)
                representatives.append({'instrument':instrument,'execution_alias':alias['execution_alias'],'actor_alias':alias['actor_alias'],'J':row['J'],'position_sorted_J':position})
            noun='execution' if cases.height==1 else 'executions'
            ax.set_title(f'{instrument.title()} — {cases.height} conditional top-J {noun}',loc='left',pad=66)
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
                    bounds=artist.get_window_extent(renderer)
                    assert bounds.x0>=-1 and bounds.y0>=-1 and bounds.x1<=fig.bbox.x1+1 and bounds.y1<=fig.bbox.y1+1
            assert not ax._left_title.get_window_extent(renderer).overlaps(ax.get_legend().get_window_extent(renderer))
        for ext in ['pdf','png']:fig.savefig(output/f'selected_msci_paths.{ext}',dpi=180)
        plt.close(fig)
    pl.DataFrame(representatives).write_csv(output/'figure_representatives.csv')
    geometry={'canvas_bounds_checked':True,'title_legend_nonoverlap_checked':True,'figsize_inches':[7,8.5],
              'fonts_pt':{'body':13,'axes':14,'titles':14,'legend':11},'common_MSCI_axis':[-1.05,2.05],
              'observed_path_range_inside_axis_checked':True,'all_selected_paths':selected.height}
    (output/'figure_geometry_verification.json').write_text(json.dumps(geometry,indent=2)+'\n')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis-root',type=Path,required=True)
    p.add_argument('--package-root',type=Path,required=True)
    p.add_argument('--output-root',type=Path,required=True)
    args=p.parse_args()
    a,g,o=args.analysis_root,args.package_root,args.output_root
    assert not o.exists()
    verify_manifest(a);verify_manifest(g)
    receipt=json.loads((a/'verification.json').read_text());assert receipt['status']=='verified'
    pre=json.loads((a/'prelaunch_manifest.json').read_text())
    for binding in pre['inputs_and_code'].values():assert sha(binding['path'])==binding['sha256']
    qroot=Path(pre['actual_selection_config']['quantity_root'])
    qmeta=json.loads((qroot/'metadata.json').read_text())
    for name in ['refreshed_execution_diagnostics.csv','step_memberships.csv']:
        assert sha(qroot/name)==qmeta['outputs'][name]
    pop=pl.read_csv(qroot/'refreshed_execution_diagnostics.csv',infer_schema_length=10000)
    first=pl.read_csv(a/'private_first_stage.csv',infer_schema_length=10000)
    summary=pl.read_csv(a/'private_all_first_stage_summary.csv',infer_schema_length=10000)
    selected=pl.read_csv(a/'private_second_stage.csv',infer_schema_length=10000)
    joined=first.join(summary,on=KEYS,validate='1:1')
    full_summary={tuple(r[k] for k in KEYS):r for r in summary.to_dicts()}
    rechecked=0
    for instrument in sorted(pop['instrument'].unique()):
        values=[r for r in pop.filter(pl.col('instrument')==instrument).to_dicts()
                if r['WMSCI_event'] is not None and math.isfinite(r['WMSCI_event']) and r['WMSCI_event']>0]
        expected=sorted(values,key=lambda r:(-r['WMSCI_event'],r['execution_cluster_id']))[:math.ceil(len(values)/10)]
        assert {tuple(r[k] for k in KEYS) for r in expected}==set(first.filter(pl.col('instrument')==instrument).select(KEYS).iter_rows())
        valid=[r for r in joined.filter(pl.col('instrument')==instrument).to_dicts() if r['complete_grid'] and r['J'] is not None and math.isfinite(r['J'])]
        expected=sorted(valid,key=lambda r:(-r['J'],r['execution_cluster_id']))[:math.ceil(len(valid)/10)]
        assert {tuple(r[k] for k in KEYS) for r in expected}==set(selected.filter(pl.col('instrument')==instrument).select(KEYS).iter_rows())
        traj=pl.read_parquet(a/f'{instrument}_private_trajectories.parquet')
        assert set(traj.select(KEYS).unique().iter_rows())==set(first.filter(pl.col('instrument')==instrument).select(KEYS).iter_rows())
        for frame in traj.partition_by(KEYS):
            frame=frame.sort('grid_offset_ns');r=full_summary[tuple(frame.select(KEYS).row(0))]
            assert frame.height==101 and frame['valid'].all()
            assert frame['grid_offset_ns'].to_list()==list(range(0,1000000001,10000000))
            v=frame['MSCI'].to_numpy();inc=np.maximum(v[5:]-v[:-5],0)
            assert np.isclose(np.max(inc),r['J'],rtol=1e-12,atol=1e-12)
            assert np.allclose([v[0],v[-1],v.max()],[r['MSCI_t0'],r['MSCI_horizon'],r['MSCI_max']],rtol=1e-12,atol=1e-12)
            assert np.allclose(frame['MSCI'].to_numpy(),0.5*frame['SCI'].to_numpy()+frame['C_opposite'].to_numpy()-frame['C_same'].to_numpy(),rtol=1e-12,atol=1e-12)
            if r['J']>0:
                index=int(np.argmax(inc));assert r['argmax_start_offset_ns']==index*10000000
                assert r['argmax_end_offset_ns']==(index+5)*10000000
            rechecked+=1
    assert rechecked==receipt['all_primary_trajectories_state_aware_verified']
    o.mkdir()
    census=pl.read_csv(g/'population_funnel.csv');prices=pl.read_csv(g/'conditional_price_outcomes.csv')
    facts={'recomputed_trajectories':rechecked,'cohort':first.height,'selected':selected.height,'instruments':{},'prices':prices.to_dicts()}
    profile_rows=[];adv_rows=[]
    for instrument in sorted(first['instrument'].unique()):
        title=instrument.title();facts['instruments'][instrument]={}
        for stage,frame in [('W tail',joined.filter(pl.col('instrument')==instrument)),('W then J',selected.filter(pl.col('instrument')==instrument))]:
            r={'n':frame.height,'actors':frame['actor_key'].n_unique(),'partitions':frame['partition_id'].n_unique(),
               'J_median':frame['J'].median(),'J_min':frame['J'].min(),'J_max':frame['J'].max(),
               'MSCI_H_median':frame['MSCI_horizon'].median(),
               'hidden':frame.filter((pl.col('J')>0)&(pl.col('MSCI_horizon')<=pl.col('MSCI_t0'))).height,
               'peakfall':frame.filter(pl.col('MSCI_horizon')<pl.col('MSCI_max')-1e-12).height,
               'argmax_end_ms_median':frame['argmax_end_offset_ns'].median()/1e6,
               'UQ_le_one_or_missing':frame.filter(~((pl.col('withdrawal_to_execution_ratio')>1)&pl.col('withdrawal_to_execution_ratio').is_finite()).fill_null(False)).height}
            v=frame['execution_price_advantage_vs_posture_mid'].drop_nulls().to_numpy();v=v[np.isfinite(v)]
            r['ADV']={'n':len(v),'positive':int((v>0).sum()),'zero':int((v==0).sum()),'negative':int((v<0).sum()),'missing':frame.height-len(v),'median':float(np.median(v)) if len(v) else None}
            facts['instruments'][instrument][stage]=r
            profile_rows.append([title,stage,r['n'],f"{r['J_median']:.3f}",f"{r['J_min']:.3f}--{r['J_max']:.3f}",f"{r['MSCI_H_median']:.3f}",r['hidden'],r['peakfall']])
            av=r['ADV'];adv_rows.append([title,stage,r['n'],f"{av['positive']}/{av['zero']}/{av['negative']}/{av['missing']}",f"{av['median']:.5f}" if av['median'] is not None else '--'])
    table(o/'population.tex','Execution populations and the two within-instrument rankings. W denotes finite positive WMSCI. The W tail retains exactly the top 10\\% of W; W then J retains exactly the top 10\\% of complete finite J within that tail. Counts are rounded upward and ties use stable execution identifiers. All first-stage trajectories are complete; neither U/Q nor prices enter selection.', 'tab:wmsci-j-population',
          ['Instrument','All','W positive','W tail','Complete J','W then J'],
          [[r['instrument'].title(),r['all_executions'],r['finite_positive_W'],r['first_stage'],r['complete_finite_J'],r['second_stage']] for r in census.to_dicts()],'lrrrrr')
    table(o/'profile_summary.tex','Observed profile changes in the W tail and its conditional top-J subset (W then J). J uses a 50 ms comparison lag on a 10 ms grid over one second. J and MSCI are dimensionless; the central values are sample medians. Hidden counts J>0 with endpoint MSCI no larger than its immediate post-execution value. Below peak counts endpoint MSCI below the highest observed level, using $10^{-12}$ only as an equality tolerance. No uncertainty intervals are estimated.','tab:wmsci-j-profile',
          ['Instrument','Stage','$N$','Median $J$','$J$ range','Median MSCI$_H$','Hidden','Below peak'],profile_rows,'llrrrrrr')
    price_rows=[]
    for r in prices.to_dicts():
        stage='W tail' if r['stage']=='first_stage' else 'W then J'
        price_rows.append([r['instrument'].title(),stage,r['n'],f"{r['FPM_positive']}/{r['FPM_zero']}/{r['FPM_negative']}/{r['FPM_missing']}",f"{r['REV_positive']}/{r['REV_zero']}/{r['REV_negative']}/{r['REV_missing_or_incomplete']}",r['both_positive_complete']])
    table(o/'prices.tex','Midprice outcomes after selection by WMSCI and J, without price filtering. Each four-part entry gives positive/zero/negative/missing counts, with N as its execution denominator. FPM is favorable pre-execution movement; REV is cancellation-anchored reversion over two seconds, missing unless every assigned positive-quantity cancellation has a finite outcome. Both counts positive FPM and complete positive REV together; this intersection is descriptive, not a selection condition.','tab:wmsci-j-prices',
          ['Instrument','Stage','$N$','FPM (+/0/--/?)','REV (+/0/--/?)','Both $>0$'],price_rows,'llrrrr')
    table(o/'advantage.tex','Execution advantage relative to the completed-display midprice benchmark in the two selected populations. Each four-part entry gives positive/zero/negative/missing execution counts. The median uses finite observations and retains source price units (EUR per share); positive advantage is not realized profit.','tab:wmsci-j-advantage',
          ['Instrument','Stage','$N$','ADV (+/0/--/?)','Median ADV (EUR)'],adv_rows,'llrrr')
    roles=pl.read_csv(g/'role_attribution_composition.csv');role_rows=[]
    for instrument in sorted(first['instrument'].unique()):
        r=roles.filter((pl.col('instrument')==instrument)&(pl.col('stage')=='second_stage'))
        def count(role,identity):
            return r.filter((pl.col('role')==role)&(pl.col('attribution')==identity))['n'].sum()
        role_rows.append([instrument.title(),count('passive','client_original'),count('aggressive','client_original'),count('passive','firm'),count('aggressive','firm')])
    table(o/'composition.tex','Execution role and recorded attribution in the conditional top-J subset. Original-client attribution is distinct from firm fallback, which can pool several clients. Counts concern executions, not independent participants or confirmed strategies.','tab:wmsci-j-composition',
          ['Instrument','Client passive','Client aggressive','Firm passive','Firm aggressive'],role_rows,'lrrrr')
    from wmsci_j_manuscript_additions import render_additions
    aliases = pl.read_csv(a/'private_alias_map.csv', infer_schema_length=10000)
    facts['additions'] = render_additions(pop, joined, selected, aliases, o, table)
    render_paths(a,selected,o)
    assert (o/'figure_representatives.csv').read_bytes()==(g/'figure_representatives.csv').read_bytes()
    for name in ['population_funnel.csv','conditional_price_outcomes.csv','profile_summaries.csv','role_attribution_composition.csv']:
        shutil.copy2(g/name,o/name)
    (o/'facts.json').write_text(json.dumps(facts,indent=2,allow_nan=False)+'\n')
    metadata={'analysis_root':str(a.resolve()),'analysis_manifest_sha256':sha(a/'artifact_manifest.json'),'package_root':str(g.resolve()),'package_manifest_sha256':sha(g/'artifact_manifest.json'),'renderer_sha256':sha(__file__),'independently_recomputed_trajectories':rechecked}
    import matplotlib
    metadata['additions_renderer_sha256'] = sha(Path(__file__).with_name('wmsci_j_manuscript_additions.py'))
    metadata['versions']={'python':sys.version,'numpy':np.__version__,'polars':pl.__version__,'matplotlib':matplotlib.__version__}
    metadata['command']=sys.argv
    metadata['artifacts']={x.name:sha(x) for x in sorted(o.iterdir()) if x.is_file()}
    (o/'metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(facts,indent=2))

if __name__=='__main__':main()
