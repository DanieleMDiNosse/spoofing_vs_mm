"""Reproducible exploratory paper package from frozen MSCI grids; no raw replay.

Validate final closure bytes and existing campaign/partition integrity before
extracting. This is NOT independent authentication of raw source data.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
from scipy.stats import rankdata
from scripts.msci_grid_study_report import KEYS, load_campaign, members, sha256

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['primary', 'resolution5', 'resolution25', 'lag25', 'lag100']
DISPLAY = {'FERRARI': 'Ferrari', 'NEXI': 'Nexi', 'RISANAMENTO': 'Risanamento'}


def rank_corr(x, y):
    pairs = [(a, b) for a, b in zip(x, y) if a is not None and b is not None and math.isfinite(a) and math.isfinite(b)]
    if len(pairs) < 2:
        return None
    a, b = map(np.asarray, zip(*pairs))
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(np.corrcoef(rankdata(a), rankdata(b))[0, 1])


def verify_receipt(root, receipt):
    receipt = Path(receipt)
    data = json.loads(receipt.read_text())
    hashes = {}
    for relative, digest in data['artifacts'].items():
        path = Path(root)/relative
        if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest) or sha256(path) != digest:
            raise ValueError(f'closure artifact hash mismatch: {relative}')
        hashes[relative] = digest
    hashes[str(receipt.relative_to(root))] = sha256(receipt)
    return hashes


def representatives(frame):
    reps = []
    for instrument in sorted(frame['instrument'].unique()):
        eligible = frame.filter((pl.col('instrument') == instrument) & (pl.col('report_status') == 'success')).to_dicts()
        eligible = sorted((r for r in eligible if r['J'] is not None and math.isfinite(r['J'])), key=lambda r: (r['J'], *(r[k] for k in KEYS)))
        if not eligible:
            continue
        for label, i in [('minimum', 0), ('central-ranked', (len(eligible)-1)//2), ('maximum', len(eligible)-1)]:
            value = eligible[i]['J']
            selected = next(r for r in eligible if r['J'] == value)
            reps.append({**{k: selected[k] for k in KEYS}, 'representative': label, 'J': value})
    return reps


def statistics(frame):
    members(frame)
    success = frame.filter((pl.col('report_status') == 'success') & pl.col('J').is_finite())
    hidden = success.filter((pl.col('J') > 0) & (pl.col('MSCI_H') <= pl.col('MSCI_t0')))
    stats = dict(N=frame.height, Positive=success.filter(pl.col('J') > 0).height,
                 Hidden=hidden.height, HiddenPct=100*hidden.height/frame.height,
                 HiddenMedianJ=hidden['J'].median(), HiddenMaxJ=hidden['J'].max())
    aggregates = []
    for instrument in sorted(frame['instrument'].unique()):
        full = frame.filter(pl.col('instrument') == instrument)
        sub = success.filter(pl.col('instrument') == instrument)
        h = hidden.filter(pl.col('instrument') == instrument)
        prefix = DISPLAY.get(instrument, instrument.title())
        fall = sub.filter(pl.col('post_max_change') < -1e-12).height
        positive_fall = sub.filter((pl.col('J') > 0) & (pl.col('post_max_change') < -1e-12)).height
        vals = dict(N=full.height, Complete=sub.height, Excluded=full.height-sub.height,
                    Actors=full['actor_key'].n_unique(), Partitions=full['partition_id'].n_unique(),
                    JMin=sub['J'].min(), JMedian=sub['J'].median(), JMax=sub['J'].max(),
                    Hidden=h.height, PeakFall=fall, PositivePeakFall=positive_fall,
                    RhoW=rank_corr(sub['J'], sub['WMSCI_event']))
        stats.update({prefix+k: v for k, v in vals.items()})
        aggregates.append(dict(instrument=instrument, **vals,
                               endpoint_nonpositive=sub.filter(pl.col('MSCI_H') <= pl.col('MSCI_t0')).height,
                               endpoint_missing=sub.filter(pl.col('MSCI_H').is_null() | pl.col('MSCI_t0').is_null()).height))
    # Report-local aliases preserve rows and nulls; exact private keys remain only in frozen input.
    rows = []
    for i, row in enumerate(frame.sort(KEYS).to_dicts(), 1):
        retained = {k: v for k, v in row.items() if k in ['instrument', 'identity_level', 'execution_anchor_mode', 'J', 'report_status', 'exclusion_reason', 'peak_MSCI_offset_ns', 'post_max_change', 'post_max_min_MSCI'] or k.endswith('_t0') or k.endswith('_H')}
        rows.append(dict(execution_alias=f'execution-{i:03d}', **retained,
                         endpoint_hidden=(row['J'] > 0 and row['MSCI_H'] <= row['MSCI_t0']) if row['J'] is not None and row['MSCI_H'] is not None and row['MSCI_t0'] is not None else None))
    return stats, pl.DataFrame(aggregates), pl.DataFrame(rows, infer_schema_length=None)


def paired(base, case):
    a, b = base['summary'], case['summary']
    if members(a) != members(b):
        raise ValueError('sensitivity identity mismatch')
    joined = a.select(KEYS+['J', 'report_status']).join(b.select(KEYS+['J', 'report_status']), on=KEYS, suffix='_case')
    joined = joined.filter((pl.col('report_status') == 'success') & (pl.col('report_status_case') == 'success') & pl.col('J').is_finite() & pl.col('J_case').is_finite())
    delta = joined['J_case']-joined['J']
    return dict(n_matched=joined.height, rho=rank_corr(joined['J'], joined['J_case']), median_delta_J=delta.median(), mean_delta_J=delta.mean()), joined


def sensitivity(campaigns):
    expected = {'primary': (10,50), 'resolution5': (5,50), 'resolution25': (25,50), 'lag25': (5,25), 'lag100': (5,100)}
    for name, (step, lag) in expected.items():
        setting = campaigns[name]['settings']
        if (setting['grid_step_ns'], setting['lag_ns']) != (step*1000000, lag*1000000) or setting['horizon_ns'] != campaigns['primary']['settings']['horizon_ns']:
            raise ValueError('sensitivity setting mismatch')
    rows = []
    for case_name, base_name in [('resolution5','primary'), ('resolution25','primary'), ('lag25','resolution5'), ('lag100','resolution5')]:
        values, _ = paired(campaigns[base_name], campaigns[case_name])
        bs, cs = campaigns[base_name]['settings'], campaigns[case_name]['settings']
        rows.append(dict(comparison=case_name, base=base_name,
                         base_step_ms=bs['grid_step_ns']/1e6, case_step_ms=cs['grid_step_ns']/1e6,
                         base_lag_ms=bs['lag_ns']/1e6, case_lag_ms=cs['lag_ns']/1e6, **values))
    return pl.DataFrame(rows)


def trajectory_figure(result):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    frame = result['summary']
    instruments = sorted(frame['instrument'].unique())
    with plt.rc_context({'font.size': 13, 'axes.titlesize': 14, 'axes.labelsize': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13, 'legend.fontsize': 13, 'pdf.fonttype': 42}):
        fig, axes = plt.subplots(len(instruments), 1, figsize=(8,8.5), squeeze=False)
        fig.subplots_adjust(left=.13, right=.95, bottom=.09, top=.88, hspace=.90)
        colors = ['#0072B2', '#D55E00', '#009E73']
        for ax, instrument in zip(axes[:,0], instruments):
            sub = frame.filter(pl.col('instrument') == instrument)
            for color, rep in zip(colors, representatives(sub)):
                path = result['trajectory'].filter(pl.all_horizontal([pl.col(k) == rep[k] for k in KEYS])).sort('grid_offset_ns')
                values = [v if valid else np.nan for v, valid in zip(path['MSCI'], path['valid'])]
                ax.step(path['grid_offset_ns'].to_numpy()/1e6, values, where='post', color=color, linewidth=1.8, label=f"{rep['representative']}  J={rep['J']:.3f}")
            excluded = sub.filter(pl.col('report_status') != 'success').height
            ax.set_title(f"{DISPLAY.get(instrument, instrument)}: N={sub.height}, excluded={excluded}", loc='center', pad=42)
            ax.set_ylabel('MSCI')
            ax.set_xlim(0,1000)
            ax.set_xlabel('Time after execution (ms)')
            ax.grid(alpha=.2)
            ax.legend(loc='lower center', bbox_to_anchor=(.5,1.01), borderaxespad=0, ncol=3, frameon=False, handlelength=1.2, columnspacing=.75, fontsize=13)
        fig.canvas.draw()
        return fig


def associations(frame):
    rows = []
    for keys in [['instrument'], ['instrument','identity_level','execution_anchor_mode']]:
        for group, sub in sorted(frame.partition_by(keys, as_dict=True).items()):
            good = sub.filter((pl.col('report_status') == 'success') & pl.col('J').is_finite() & pl.col('WMSCI_event').is_finite())
            rows.append(dict(scope='instrument' if len(keys)==1 else 'stratum', **dict(zip(keys,group)), n_cohort=sub.height, n_pairs=good.height,
                             rho_J_W=rank_corr(good['J'],good['WMSCI_event']),
                             rho_J_UQ=rank_corr(good['J'],good['withdrawal_to_execution_ratio'])))
    return pl.DataFrame(rows, infer_schema_length=None)


def tex_table(caption, label, headers, rows, alignment):
    return '\n'.join([r'\begin{table}[htbp]', r'\centering', '\\caption{'+caption+'}', '\\label{'+label+'}', '\\begin{tabular}{'+alignment+'}', r'\hline', ' & '.join(headers)+r' \\', r'\hline', *[' & '.join(row)+r' \\' for row in rows], r'\hline', r'\end{tabular}', r'\end{table}', ''])


def write_package(campaigns, output, metadata):
    output = Path(output)
    if output.exists():
        raise ValueError('output directory must be new')
    stats, aggregate, endpoints = statistics(campaigns['primary']['summary'])
    sens = sensitivity(campaigns)
    assoc = associations(campaigns['primary']['summary'])
    bycase = {r['comparison']: r for r in sens.to_dicts()}
    macros = {k: v for k, v in stats.items() if k != 'N' and k not in ['HiddenMedianJ','HiddenMaxJ']}
    macros.update(Total=stats['N'], HiddenMedian=stats['HiddenMedianJ'], HiddenMax=stats['HiddenMaxJ'])
    for case, name in [('resolution5','ResolutionFive'), ('resolution25','ResolutionTwentyFive'), ('lag25','LagTwentyFive'), ('lag100','LagHundred')]:
        macros['Rho'+name] = bycase[case]['rho']
    output.mkdir(parents=True)
    aggregate.write_csv(output/'summary.csv'); endpoints.write_csv(output/'endpoints_postmax.csv')
    sens.write_csv(output/'sensitivity.csv'); assoc.write_csv(output/'associations.csv')
    (output/'statistics.json').write_text(json.dumps(stats, indent=2, allow_nan=False)+'\n')
    def fmt(k,v):
        return 'NA' if v is None else f'{v:.1f}' if k=='HiddenPct' else f'{v:.3f}' if isinstance(v,float) else str(v)
    (output/'macros.tex').write_text('% Generated; do not hand-edit.\n'+''.join('\\newcommand{\\MSCIG'+k+'}{'+fmt(k,v)+'}\n' for k,v in macros.items()))
    rows = [[DISPLAY.get(r['instrument'],r['instrument']), str(r['N']), f"{r['JMedian']:.3f}", f"{r['JMax']:.3f}", str(r['endpoint_nonpositive'])] for r in aggregate.to_dicts()]
    caption = r'Execution-level MSCI-grid summaries for the selected cohort: retain the deterministic top $5\%$ of finite positive event scores $W$ within each instrument, with ties resolved by stable execution identifier, then retain executions with $U/Q>1$, before conditioning on price outcomes. $N$ includes all selected executions; $J$ is dimensionless. The final column counts complete executions with $\operatorname{MSCI}(H_J)\leq\operatorname{MSCI}(0)$, including $J=0$; $u=0$ is the exact post-execution state. No uncertainty estimates are supplied.'
    (output/'summary.tex').write_text(tex_table(caption,'tab:msci-grid-summary', ['Instrument',r'$N$',r'Median $J$',r'Maximum $J$',r'$\operatorname{MSCI}(H_J)\leq\operatorname{MSCI}(0)$'],rows,'lrrrr'))
    rows = [[f"{int(r['base_step_ms'])}/{int(r['base_lag_ms'])}",f"{int(r['case_step_ms'])}/{int(r['case_lag_ms'])}",str(r['n_matched']),f"{r['rho']:.3f}",f"{r['median_delta_J']:.3f}",f"{r['mean_delta_J']:.3f}"] for r in sens.to_dicts()]
    caption = r'Exploratory matched-execution sensitivity of $J$ in the same selected cohort as Table~\ref{tab:msci-grid-summary}. Base and case columns give grid step/lag in ms; $H=1000$ ms throughout. $N$ counts complete finite pairs matched on instrument, partition, actor and execution. $\rho$ is Spearman correlation with average ranks for ties; $\Delta J=J_{\rm case}-J_{\rm base}$ is dimensionless. Lag comparisons hold the step fixed at 5 ms. The primary 10/50 setting and the shared 5/50 reference are not independent duplicate evidence. No hypothesis tests or uncertainty estimates are reported.'
    (output/'sensitivity.tex').write_text(tex_table(caption,'tab:msci-grid-sensitivity', ['Base', 'Case', r'$N$',r'$\rho$',r'Median $\Delta J$',r'Mean $\Delta J$'], rows, 'rrrrrr'))
    fig = trajectory_figure(campaigns['primary'])
    fig.savefig(output/'trajectories.pdf', metadata={'CreationDate':None,'ModDate':None})
    import matplotlib.pyplot as plt
    plt.close(fig)
    caption = ('MSCI trajectories for minimum, central-ranked (lower order statistic), and maximum saved J within each instrument, among complete executions in the selected cohort. Ties use lexicographically first full private key; no private identifiers are displayed. Lines are forward-filled steps on the saved 10 ms grid (lag 50 ms; H=1000 ms); independent panel MSCI scales. Legends report each execution\'s J, not the numerical sample median. MSCI and J are dimensionless. No uncertainty bands or causal interpretation are supplied.')
    (output/'figure_caption.txt').write_text(caption+'\n')
    rep_rows = [{k:v for k,v in r.items() if k not in KEYS or k=='instrument'} for r in representatives(campaigns['primary']['summary'])]
    pl.DataFrame(rep_rows).write_csv(output/'representatives.csv')
    correlations = []
    for case, base in [('resolution5','primary'),('resolution25','primary'),('lag25','resolution5'),('lag100','resolution5')]:
        _, joined = paired(campaigns[base], campaigns[case])
        for instrument in sorted(joined['instrument'].unique()):
            sub = joined.filter(pl.col('instrument')==instrument)
            correlations.append(dict(comparison=case, instrument=instrument, n_pairs=sub.height, rho=rank_corr(sub['J'],sub['J_case'])))
    pl.DataFrame(correlations).write_csv(output/'sensitivity_by_instrument.csv')
    (output/'README.md').write_text('# Exploratory MSCI paper package\n\n'+caption+'\n\nEndpoint-hidden means exactly J>0 and MSCI_H<=MSCI_t0, not zero horizon score. PeakFall counts all complete executions with post_max_change < -1e-12; PositivePeakFall restricts to positive J; this tolerance is only an equality diagnostic. Null endpoints remain null. Associations condition on W-selected actors and U/Q>1; source-derived roles and firm/client attribution are kept separate in associations.csv. Counts are executions, not independent observations. No p-values, confidence intervals, raw-data authentication, or preregistration is claimed. Report-local execution aliases are sorted frozen-key row positions; consult frozen inputs privately for original keys.\n')
    metadata = dict(metadata, code_hashes={str(Path(__file__).relative_to(ROOT)):sha256(__file__), 'scripts/msci_grid_study_report.py':sha256(ROOT/'scripts/msci_grid_study_report.py')},
                    software_versions={'numpy':np.__version__,'polars':pl.__version__},
                    artifacts={p.name:sha256(p) for p in sorted(output.iterdir()) if p.is_file()},
                    scope='saved-artifact integrity only; not independent raw source authentication')
    (output/'metadata.json').write_text(json.dumps(metadata, indent=2, sort_keys=True)+'\n')
    return stats


def generate(root, output):
    root = Path(root).resolve()
    hashes = verify_receipt(root, root/'outputs/msci_grid_final_closure_receipt.json')
    campaigns = {}
    for label in LABELS:
        directory = root/f'outputs/msci_grid_final_{label}_report_integrated'
        run = json.loads((directory/'run_manifest.json').read_text())
        command = run['command']
        cohort = root/command[command.index('--cohort-root')+1]
        campaigns[label] = load_campaign(directory,cohort)
        for path in sorted(directory.rglob('*')):
            if path.is_file():
                hashes[str(path.relative_to(root))] = sha256(path)
        for filename in ['cohort.csv','cohort_provenance.json']:
            hashes[str((cohort/filename).relative_to(root))] = sha256(cohort/filename)
    frozen = pl.read_csv(root/'outputs/msci_grid_final_report_complete/summary.csv',infer_schema_length=None)
    loaded = campaigns['primary']['summary']
    if members(frozen) != members(loaded) or not frozen.sort(KEYS).equals(loaded.select(frozen.columns).sort(KEYS)):
        raise ValueError('frozen report differs from verified campaign')
    if frozen.height != json.loads((root/'outputs/msci_grid_final_closure_receipt.json').read_text())['cohort_exact_ids']:
        raise ValueError('closure cohort count mismatch')
    return write_package(campaigns,output,dict(input_hashes=hashes, run_manifest_hashes={k:v['run_manifest_sha256'] for k,v in campaigns.items()}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root',type=Path,default=ROOT)
    parser.add_argument('--output-root',type=Path,required=True)
    args = parser.parse_args()
    print(json.dumps(generate(args.repo_root,args.output_root),indent=2))
