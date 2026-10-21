"""Task10 descriptive reports from immutable saved grids; never replay raw data.

Scientific adapter/consumer verification is a caller prerequisite. This module
checks artifact integrity and optionally caller-supplied frozen identities, NOT
whether current source code produced or scientifically verified historical data.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl

from spoofing_detection.lob.msci_grid_store import _load_verified, _validate_identity

KEYS = ['instrument', 'partition_id', 'actor_key', 'execution_cluster_id']
CASES = {'resolution_5ms_lag50ms': (5,50), 'resolution_10ms_lag50ms': (10,50),
         'resolution_25ms_lag50ms': (25,50), 'lag_25ms_step5ms': (5,25),
         'lag_50ms_step5ms': (5,50), 'lag_100ms_step5ms': (5,100)}
COMPONENTS = ['MSCI', 'DWI', 'SCI', 'C_opposite', 'C_same', 'L_bid_topN', 'L_ask_topN']


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def members(frame, keys=KEYS):
    if any(k not in frame.columns for k in keys):
        raise ValueError('missing identity columns')
    rows = frame.select(keys).rows()
    if any(any(v is None for v in row) for row in rows) or len(set(rows)) != len(rows):
        raise ValueError('null or duplicate keys')
    return set(rows)


def describe(cohort, trajectory, summary, *, horizon_ns):
    """Pure table derivation. Keep exclusions; representative ties use full keys."""
    expected = members(cohort)
    if members(summary) != expected or set(trajectory.select(KEYS).rows()) != expected:
        raise ValueError('missing or extra cohort execution IDs')
    members(trajectory, KEYS + ['grid_offset_ns'])
    if trajectory.schema['valid'] != pl.Boolean or trajectory['valid'].null_count():
        raise ValueError('valid must be non-null Boolean')
    if summary.schema['complete_grid'] != pl.Boolean or summary['complete_grid'].null_count():
        raise ValueError('complete_grid must be non-null Boolean')
    metadata = {tuple(r[k] for k in KEYS):r for r in cohort.to_dicts()}
    paths = {}
    for row in trajectory.sort(KEYS+['grid_offset_ns']).to_dicts():
        paths.setdefault(tuple(row[k] for k in KEYS), []).append(row)
    rows = []
    for saved in summary.sort(KEYS).to_dicts():
        key = tuple(saved[k] for k in KEYS); path = paths[key]
        ok = saved['complete_grid'] and saved['J'] is not None and math.isfinite(saved['J'])
        reasons = sorted({r['reason'] or 'invalid_node' for r in path if not r['valid']})
        row = {**metadata[key], **saved, 'report_status': 'success' if ok else 'excluded',
               'exclusion_reason': '' if ok else '|'.join(reasons or ['incomplete_or_nonfinite_J'])}
        for suffix, offset in [('t0',0),('H',horizon_ns)]:
            endpoint = next((r for r in path if r['grid_offset_ns']==offset),None)
            if ok and (endpoint is None or not endpoint['valid']):
                raise ValueError('complete execution lacks valid endpoint')
            for component in COMPONENTS:
                row[f'{component}_{suffix}'] = endpoint[component] if endpoint and endpoint['valid'] else None
        valid = [r for r in path if r['valid'] and r['MSCI'] is not None and math.isfinite(r['MSCI'])]
        peak = min(valid, key=lambda r:(-r['MSCI'],r['grid_offset_ns'])) if valid else None
        row['peak_MSCI_offset_ns'] = peak['grid_offset_ns'] if peak else None
        row['post_max_change'] = row['MSCI_H']-peak['MSCI'] if peak and row['MSCI_H'] is not None else None
        row['post_max_min_MSCI'] = min((r['MSCI'] for r in valid if r['grid_offset_ns'] >= peak['grid_offset_ns']), default=None) if peak else None
        rows.append(row)
    table = pl.DataFrame(rows, infer_schema_length=None)
    representatives = []
    for instrument in sorted({r['instrument'] for r in rows}):
        eligible = sorted((r for r in rows if r['instrument']==instrument and r['report_status']=='success'),key=lambda r:(r['J'],*(r[k] for k in KEYS)))
        if eligible:
            for label,index in [('minimum',0),('median',(len(eligible)-1)//2),('maximum',len(eligible)-1)]:
                # Pick lexicographically first ID for any selected J tie.
                value = eligible[index]['J']; chosen = next(r for r in eligible if r['J']==value)
                representatives.append({**{k:chosen[k] for k in KEYS},'representative':label,'J':value})
    n = sum(r['report_status']=='success' for r in rows)
    return {'summary':table,'trajectory':trajectory,'representatives':representatives,
            'counts':{'cohort':len(rows),'success':n,'excluded':len(rows)-n,
                      'missing_baseline':sum('missing_baseline' in r['exclusion_reason'] for r in rows)}}


def compare_sensitivities(primary, labeled):
    """Paired differences among exact same IDs with complete support in both runs."""
    base = primary['summary']; rows=[]
    for label, result in labeled:
        other = result['summary']
        if members(base)!=members(other):
            raise ValueError('sensitivity identity alignment mismatch')
        paired = base.select(KEYS+['J','report_status']).join(other.select(KEYS+['J','report_status']),on=KEYS,suffix='_case').filter((pl.col('report_status')=='success') & (pl.col('report_status_case')=='success'))
        delta = paired['J_case']-paired['J']
        rows.append({'setting':label,'n_cohort':base.height,'n_case_success':result['counts']['success'],
                     'n_same_id_support':paired.height,'n_not_jointly_supported':base.height-paired.height,
                     'median_delta_J':delta.median(),'mean_delta_J':delta.mean(),
                     'shared_reference':'resolution_5ms_lag50ms' if label=='lag_50ms_step5ms' else 'primary' if label=='resolution_10ms_lag50ms' else ''})
    return pl.DataFrame(rows, infer_schema_length=None) if rows else pl.DataFrame(schema={'setting':pl.String,'n_same_id_support':pl.Int64})


def load_campaign(primary_root, cohort_root, *, expected_identities=None):
    """Read immutable artifacts; expected identities are supplied by caller, not current code."""
    root, co = Path(primary_root), Path(cohort_root)
    run = json.loads((root/'run_manifest.json').read_text())
    provenance = json.loads((co/'cohort_provenance.json').read_text())
    for label, file in [('cohort','cohort.csv'),('cohort_provenance','cohort_provenance.json')]:
        if run['input_hashes'][label] != sha256(co/file):
            raise ValueError(f'{label} hash mismatch')
    if provenance['artifacts']['cohort']['sha256'] != sha256(co/'cohort.csv'):
        raise ValueError('cohort provenance hash mismatch')
    cohort = pl.read_csv(co/'cohort.csv',infer_schema_length=None)
    expected = members(cohort)
    if provenance['artifacts']['cohort']['row_count'] != cohort.height:
        raise ValueError('cohort count mismatch')
    if members(pl.DataFrame(run['requested_execution_ids']))!=expected:
        raise ValueError('run requested IDs differ from cohort')
    trajectories=[]; summaries=[]; observed={}
    config_key=run['settings']['configuration_key']
    for instrument, partition in sorted({(k[0],k[1]) for k in expected}):
        label=f'{instrument}::{partition}'
        directory=root/'partitions'/instrument/partition/config_key/'complete'
        manifest=json.loads((directory/'manifest.json').read_text())
        identity=_validate_identity(manifest['identity'])
        if expected_identities is not None and identity != expected_identities.get(label):
            raise ValueError('expected frozen identity mismatch')
        if identity['code_hashes'] != run['code_hashes']:
            raise ValueError('partition/run frozen code identity mismatch')
        for item in ['cohort','cohort_provenance']:
            if identity['input_hashes'][item] != run['input_hashes'][item]:
                raise ValueError('partition/run cohort identity mismatch')
        requested={k for k in expected if k[:2]==(instrument,partition)}
        completion=_load_verified(directory,identity,requested)
        trajectories.append(completion.trajectory);summaries.append(completion.summary)
        observed[label]=len(requested)
    if observed!=run['expected_partitions']:
        raise ValueError('partition count mismatch')
    actual=set((root/'partitions').glob('*/*/*/complete'))
    if len(actual)!=len(observed):
        raise ValueError('extra completed partitions')
    result=describe(cohort,pl.concat(trajectories),pl.concat(summaries),horizon_ns=run['settings']['horizon_ns'])
    result['settings']=run['settings']; result['run_manifest_sha256']=sha256(root/'run_manifest.json')
    result['identity_verification']='caller-supplied frozen identity checked' if expected_identities is not None else 'artifact integrity only; scientific verification is a prerequisite'
    return result


def render(result, output_root, *, sensitivity, complete, missing=()):
    """Write fresh tables/PDFs. Actor IDs are retained only in machine-readable table."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    out=Path(output_root);out.mkdir(parents=True,exist_ok=True);figdir=out/'figures';figdir.mkdir(exist_ok=True)
    result['summary'].write_csv(out/'summary.csv');sensitivity.write_csv(out/'sensitivity.csv')
    # Normalize only the presentation copy, never the saved cohort/summary.
    strata_keys = ['instrument', 'identity_level', 'execution_anchor_mode']
    table = result['summary'].with_columns([
        pl.col(k).fill_null('unknown') if k in result['summary'].columns
        else pl.lit('unknown').alias(k) for k in strata_keys[1:]])
    strata = table.partition_by(strata_keys, as_dict=True)

    def stratum_label(key):
        return f'{key[0]} / identity_level={key[1]} / execution_anchor_mode={key[2]}'

    def finite_pairs(subset, column):
        return [(r['J'], r[column]) for r in subset.to_dicts()
                if r['report_status'] == 'success'
                and r['J'] is not None and math.isfinite(r['J'])
                and r[column] is not None and math.isfinite(r[column])
                and (column != 'post_cancel_mid_reversion'
                     or r.get('post_cancel_mid_reversion_complete') is True)]
    text=['# MSCI grid descriptive report',f"Completion: {'complete' if complete else 'PARTIAL'}; missing settings: {', '.join(missing) or 'none'}.",
          f"Counts (execution denominator): {result['counts']}.",
          'Surveillance/triage cues only: no causal, confirmatory, manipulation-intent or prevalence claim.',
          'MSCI/J/DWI/SCI/collapse are dimensionless; relative time is ms. U/Q is a quantity ratio. Price summaries retain source price units (not ticks/bps).',
          'Missing completed partitions are fatal input errors, not scientific exclusions. Missing baseline counts are recorded below among retained excluded executions.',
          'REV primary association requires finite pairs and post_cancel_mid_reversion_complete=True; available-only observations are separate diagnostics, never selection gates.',
          'Success requires complete_grid and finite saved J; excluded rows are retained with reasons. No transient threshold is imposed.',
          'Median representative is the lower order statistic; ties select lexicographically first full execution key.',
          'Post-max statistics follow the earliest absolute MSCI maximum through H, not the lagged-J maximum.',
          'Resolution10/lag50 shares the primary numerical setting; it is not independent scientific evidence.',
          'The lag50/step5 case is the shared resolution5/lag50 reference, not an independent sensitivity run.',
          result.get('identity_verification','Pure saved-table derivation; no scientific source verification asserted.'),
          f"Frozen run manifest SHA256: {result.get('run_manifest_sha256','not supplied')}", '## Counts by attribution and source-derived execution role']
    groups=table.group_by(['instrument','identity_level','execution_anchor_mode','report_status']).len().sort(['instrument','identity_level','execution_anchor_mode','report_status'])
    text.append(str(groups.to_dicts()))
    text.append('## Excluded reasons')
    text.append(str(table.filter(pl.col('report_status')!='success').group_by('exclusion_reason').len().to_dicts()))
    text.append('## Descriptive associations (finite pairs; no hypothesis tests)')
    text.append('Associations are separate by instrument, attribution and execution role; missing metadata is unknown. Sparse strata (fewer than 3 finite pairs) show observations, not fitted models.')
    for key, subset in sorted(strata.items()):
        label = stratum_label(key)
        n_success = subset.filter(pl.col('report_status') == 'success').height
        text.append(f'{label}: cohort={subset.height}, success={n_success}, excluded={subset.height-n_success}.')
        for column in ['withdrawal_to_execution_ratio','favorable_mid_move_pre_fill','post_cancel_mid_reversion', 'post_cancel_mid_reversion_available_only']:
            if column not in subset.columns:
                text.append(f'{label}: {column} unavailable in saved cohort.');continue
            suffix = ' (complete-only)' if column == 'post_cancel_mid_reversion' else ' (diagnostic available-only)' if column == 'post_cancel_mid_reversion_available_only' else ''
            pairs = finite_pairs(subset, column)
            x,y=zip(*pairs) if pairs else ([],[])
            corr=float(np.corrcoef(x,y)[0,1]) if len(x)>2 and np.std(x)>0 and np.std(y)>0 else None
            text.append(f'{label}: J vs {column}{suffix}: N={len(pairs)}, Pearson r={corr}; median={float(np.median(y)) if pairs else None}.')
            if len(pairs) < 3:
                text.append(f'{label}: observations (J, value)={pairs} for {column}{suffix}.')
    (out/'report.md').write_text('\n\n'.join(text)+'\n')
    with PdfPages(figdir/'msci_trajectories.pdf') as pdf:
        instruments=sorted(table['instrument'].unique().to_list())
        for instrument in instruments:
            fig,ax=plt.subplots(figsize=(8,4))
            for rep in result['representatives']:
                if rep['instrument']!=instrument:continue
                path=result['trajectory'].filter(pl.all_horizontal([pl.col(k)==rep[k] for k in KEYS])).sort('grid_offset_ns')
                ax.plot(path['grid_offset_ns'].to_numpy()/1e6,[v if valid else np.nan for v,valid in zip(path['MSCI'],path['valid'])],label=f"{rep['representative']} J={rep['J']:.3g}")
            count=table.filter(pl.col('instrument')==instrument).height
            excluded=table.filter((pl.col('instrument')==instrument)&(pl.col('report_status')!='success')).height
            ax.set(title=f'{instrument}: N={count}, excluded={excluded}',xlabel='Time after execution (ms)',ylabel='MSCI (dimensionless)');ax.axvline(0,color='gray',linewidth=.5)
            if ax.lines and result['representatives']:ax.legend()
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4))
    n_pairs = 0
    for index, (key, subset) in enumerate(sorted(strata.items())):
        if 'withdrawal_to_execution_ratio' in subset.columns:
            pairs = finite_pairs(subset, 'withdrawal_to_execution_ratio')
            j, uq = zip(*pairs) if pairs else ([], [])
            n_pairs += len(pairs)
            ax.scatter(uq, j, marker=['o', 's', '^', 'D', 'v', 'P', 'X'][index % 7],
                       label=f'{stratum_label(key)} N={len(pairs)}')
    ax.set(xlabel='U/Q (dimensionless)',ylabel='J (dimensionless)',title=f'Finite pairs across labeled strata N={n_pairs}')
    if ax.collections:
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.35), fontsize=8)
    fig.tight_layout();fig.savefig(figdir/'j_vs_withdrawal.pdf', bbox_inches='tight');plt.close(fig)


def generate_report(primary_root, cohort_root, labeled_sensitivity_roots, output_root, *, expected_identities=None, allow_partial=False, verification_metadata=None):
    """Callable future CLI API; no raw replay, never overwrite report artifacts.

expected_identities maps setting label ('primary' included) to partition-label
identity maps. Missing cases fail before writing unless allow_partial=True.
"""
    labeled=list(labeled_sensitivity_roots)
    labels=[label for label,_ in labeled]
    if len(set(labels))!=len(labels) or set(labels)-set(CASES):
        raise ValueError('duplicate or unknown sensitivity labels')
    missing=sorted(set(CASES)-set(labels))
    if missing and not allow_partial:
        raise ValueError(f'missing sensitivity settings: {missing}')
    if Path(output_root).exists():
        raise ValueError('output root must be new')
    identities=expected_identities or {}
    primary=load_campaign(primary_root,cohort_root,expected_identities=identities.get('primary'))
    if verification_metadata is not None:
        primary['identity_verification'] = ('All primary/sensitivity inputs scientifically verified against current consumer/source before rendering. '
            + json.dumps(verification_metadata, sort_keys=True)
            + '; report module SHA256: ' + verification_metadata['report_module_sha256'])
    loaded=[]
    for label,root in labeled:
        case=load_campaign(root,cohort_root,expected_identities=identities.get(label))
        step,lag=CASES[label]
        if (case['settings']['grid_step_ns'],case['settings']['lag_ns'])!=(step*1_000_000,lag*1_000_000) or case['settings']['horizon_ns']!=primary['settings']['horizon_ns']:
            raise ValueError('sensitivity settings mismatch')
        loaded.append((label,case))
    bylabel=dict(loaded)
    if all(label in bylabel for label in ['resolution_5ms_lag50ms','lag_50ms_step5ms']):
        a,b=bylabel['resolution_5ms_lag50ms'],bylabel['lag_50ms_step5ms']
        if not a['summary'].sort(KEYS).equals(b['summary'].sort(KEYS)) or not a['trajectory'].sort(KEYS+['grid_offset_ns']).equals(b['trajectory'].sort(KEYS+['grid_offset_ns'])):
            raise ValueError('shared reference outputs differ')
    sensitivity=compare_sensitivities(primary,loaded)
    render(primary,output_root,sensitivity=sensitivity,complete=not missing,missing=missing)
    return {'complete':not missing,'missing':missing,'counts':primary['counts'],'output_root':str(output_root)}
