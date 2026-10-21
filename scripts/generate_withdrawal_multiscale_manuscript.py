#!/usr/bin/env python
"""Publication consumer for the frozen v4 multiscale campaign; never replays LOB.

Run with PYTHONPATH=.:src python scripts/generate_withdrawal_multiscale_manuscript.py.
Only hashes and reads completed artifacts. Outputs are descriptive, not labels.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import platform
from pathlib import Path
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]
DEFAULT_ROOTS = [Path('outputs/withdrawal_multiscale_20261013_v4/RISANAMENTO'),
                 Path('outputs/withdrawal_multiscale_20261013_v4/NEXI'),
                 Path('outputs/withdrawal_multiscale_20261013_v4_ferrari_extended/FERRARI')]
SUMMARY = Path('outputs/withdrawal_multiscale_20261013_v4/campaign_summary')
DIAGNOSTICS = Path('outputs/withdrawal_multiscale_20261013_v4/diagnostics')
OUTPUT = Path('paper/generated/withdrawal_multiscale_20261014_primary_prices_v1')
PRIMARY_PRICES = Path('outputs/primary_kernel30_h10_prices_20261014_213056_numeric_v2')
PRIMARY_POLICY = 'exp_h30'
PRIMARY_HORIZON_NS = 10_000_000_000
KEYS = ['instrument', 'partition_id', 'actor_key', 'execution_cluster_id']
HORIZONS = [1_000_000_000, 2_000_000_000, 5_000_000_000, 10_000_000_000]
POLICIES = ['legacy_age30', 'all_ages_unweighted', 'exp_h30', 'exp_h120', 'exp_h300']
SHORT = {'RISANAMENTO': 'Risanamento', 'NEXI': 'Nexi', 'FERRARI': 'Ferrari'}
FLAGS = ['priority_top_w_self', 'priority_top_w_market', 'priority_top_j_given_self', 'priority_top_j_given_market']


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def load_json(path):
    return json.loads(Path(path).read_text())


def verify_manifest(directory, name='artifact_manifest.json'):
    manifest = load_json(directory / name)
    if not manifest.get('artifacts'):
        raise ValueError(f'empty artifacts: {directory}')
    for filename, expected in manifest['artifacts'].items():
        path = directory / filename
        if Path(filename).is_absolute() or '..' in Path(filename).parts:
            raise ValueError('unsafe manifest path')
        if not isinstance(expected, str) or len(expected) != 64 or any(c not in '0123456789abcdef' for c in expected):
            raise ValueError(f'invalid hash: {path}')
        if not path.is_file() or digest(path) != expected:
            raise ValueError(f'artifact hash mismatch: {path}')
    return manifest


def validate_stage(directory, status):
    marker = directory / 'COMPLETE.json'
    if not marker.is_file() or load_json(marker).get('status') != status:
        raise ValueError(f'invalid completion: {directory}')
    if load_json(marker).get('artifact_manifest_sha256') != digest(directory / 'artifact_manifest.json'):
        raise ValueError(f'completion manifest hash mismatch: {directory}')
    return verify_manifest(directory)


def validate_root(root):
    validate_stage(root / 'prepare', 'prepared')
    validate_stage(root / 'analysis', 'verified')
    validate_stage(root / 'report', 'report_verified')
    derived = verify_manifest(root / 'analysis', 'derived_manifest.json')
    marker = load_json(root / 'analysis/DERIVED.json')
    if marker.get('status') != 'derived' or marker.get('artifact_manifest_sha256') != digest(root / 'analysis/derived_manifest.json'):
        raise ValueError('derived completion hash mismatch')
    verification = load_json(root / 'analysis/verification.json')
    if verification.get('status') != 'verified':
        raise ValueError('analysis independent verification missing')
    provenance = load_json(root / 'report/provenance.json')
    if provenance['analysis_manifest']['sha256'] != digest(root / 'analysis/artifact_manifest.json'):
        raise ValueError('report/analysis provenance mismatch')
    if provenance['config']['sha256'] != digest(REPO / 'configs/withdrawal_multiscale_v1.json'):
        raise ValueError('report config mismatch')
    return {'root': str(root.relative_to(REPO)), 'population_sha256': derived['artifacts']['population.parquet'],
            'profiles_sha256': derived['artifacts']['profiles.parquet'],
            'audit_sha256': derived['artifacts']['cancellation_audit.parquet'],
            'report_provenance': provenance,
            'complete_sha256': {stage: digest(root / stage / 'COMPLETE.json') for stage in ['prepare', 'analysis', 'report']}}


def overlap_metrics(a, b):
    union = len(a | b)
    return {'intersection': len(a & b), 'union': union, 'jaccard': len(a & b) / union if union else None}


def keyset(frame):
    return set(frame.select(KEYS).iter_rows())


def ranking_metrics(cell):
    """Compare stored lists without restricting or reranking their supports."""
    c = cell.filter(pl.col('is_candidate'))
    positive_self = c.filter((pl.col('WMSCI_self') > 0) & pl.col('WMSCI_self').is_finite())
    positive_market = c.filter((pl.col('WMSCI_market') > 0) & pl.col('WMSCI_market').is_finite())
    a = keyset(cell.filter(pl.col(FLAGS[0])))
    b = keyset(cell.filter(pl.col(FLAGS[1])))
    return {'raw_candidates': c.height, 'positive_self_scores': positive_self.height,
            'positive_market_scores': positive_market.height,
            'common_eligible_candidates': len(keyset(positive_self) & keyset(positive_market)),
            'self_ceiling_budget': math.ceil(positive_self.height / 10),
            'market_ceiling_budget': math.ceil(positive_market.height / 10),
            'top_self': len(a), 'top_market': len(b), **overlap_metrics(a, b)}


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def number(value):
    if value is None:
        return '--'
    if isinstance(value, float):
        return f'{value:.3f}'
    return f'{value:,}' if isinstance(value, int) else str(value).replace('_', r'\_')


def table(out, name, label, caption, headers, rows, *, floating=True):
    text = ('\\begin{table}[!htbp]\n' if floating else '\\noindent\\begin{minipage}{\\textwidth}\n')
    text += '\\centering\n\\small\n'
    text += ('\\caption{' if floating else '\\captionof{table}{') + caption + '}\n\\label{' + label + '}\n'
    text += '\\begin{tabular}{' + 'l' + 'r' * (len(headers) - 1) + '}\n\\toprule\n'
    text += ' & '.join(headers) + ' \\\\\n\\midrule\n'
    text += ''.join(' & '.join(number(v) for v in row) + ' \\\\\n' for row in rows)
    text += '\\bottomrule\n\\end{tabular}\n'
    text += '\\end{table}\n' if floating else '\\end{minipage}\\par\\medskip\n'
    (out / name).write_text(text)


def save_figure(fig, out, filename, caption, data):
    fig.canvas.draw()
    # Matplotlib allocates locator ticks outside axis limits but does not draw
    # them. Hide their labels explicitly so the geometry audit checks the PDF.
    for ax in fig.axes:
        for axis, limits in [(ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())]:
            lower, upper = sorted(limits)
            for tick in axis.get_major_ticks() + axis.get_minor_ticks():
                if not lower <= tick.get_loc() <= upper:
                    tick.label1.set_visible(False)
                    tick.label2.set_visible(False)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    w, h = fig.canvas.get_width_height()
    visible_text = [t for t in fig.findobj(matplotlib.text.Text) if t.get_visible() and t.get_text()]
    minimum = min(t.get_fontsize() for t in visible_text)
    boxes = [t.get_window_extent(renderer) for t in visible_text]
    geometry = all(b.x0 >= -1 and b.y0 >= -1 and b.x1 <= w + 1 and b.y1 <= h + 1 for b in boxes)
    if not geometry or minimum < 13:
        raise ValueError(f'figure text geometry/font failure: {filename}; minimum={minimum}; outside=' + str([(t.get_text(), tuple(b.bounds)) for t, b in zip(visible_text, boxes) if b.x0 < -1 or b.y0 < -1 or b.x1 > w + 1 or b.y1 > h + 1]))
    fig.savefig(out / filename, metadata={'CreationDate': None, 'ModDate': None})
    result = {'path': filename, 'caption': caption, 'companion_data': data,
              'width_inches': float(fig.get_figwidth()), 'height_inches': float(fig.get_figheight()),
              'minimum_font_pt': minimum, 'geometry_ok': geometry,
              'axis_limits': [{'x': list(ax.get_xlim()), 'y': list(ax.get_ylim())} for ax in fig.axes],
              'suggested_latex_width': r'\textwidth', 'axes': 'linear; no confidence intervals'}
    plt.close(fig)
    return result


def render_primary_price_tables(out, root, counts):
    """Render authenticated outcomes on the native pre-price priority lists."""
    source_manifest = validate_stage(root, 'primary_prices_verified')
    if not {'summary.csv', 'metadata.json'} <= set(source_manifest['artifacts']):
        raise ValueError('price summary and metadata must be manifest-bound')
    summary = pl.read_csv(root / 'summary.csv')
    sign_fields = {
        'FPM': ['FPM_positive', 'FPM_zero', 'FPM_negative', 'FPM_missing'],
        'REV': ['REV_positive', 'REV_zero', 'REV_negative', 'REV_missing_or_incomplete'],
        'ADV': ['ADV_positive', 'ADV_zero', 'ADV_negative', 'ADV_missing']}
    fields = ['instrument', 'ranking', 'stage', 'n', *sum(sign_fields.values(), []), 'both_positive_complete']
    if set(fields) - set(summary.columns) or summary.height != 12:
        raise ValueError('primary price summary schema or group membership mismatch')
    expected = {(i, r, s) for i in SHORT for r in ['self', 'market'] for s in ['W', 'J']}
    if set(summary.select(['instrument', 'ranking', 'stage']).iter_rows()) != expected:
        raise ValueError('primary price group membership mismatch')
    native = counts.filter((pl.col('age_policy') == PRIMARY_POLICY) &
                           (pl.col('horizon_ns') == PRIMARY_HORIZON_NS))
    by_group = {}
    for row in summary.iter_rows(named=True):
        group = (row['instrument'], row['ranking'], row['stage'])
        values = [row['n'], row['both_positive_complete'], *[row[f] for cols in sign_fields.values() for f in cols]]
        if any(type(v) is not int or v < 0 for v in values):
            raise ValueError('price summary counts must be nonnegative integers')
        flag = 'priority_top_' + ('w_' if row['stage'] == 'W' else 'j_given_') + row['ranking'] + '_count'
        source = native.filter(pl.col('instrument') == row['instrument'])
        if source.height != 1 or row['n'] != source[flag][0]:
            raise ValueError('primary price membership denominator differs from native flags')
        if any(sum(row[f] for f in cols) != row['n'] for cols in sign_fields.values()):
            raise ValueError('price sign counts do not partition the list denominator')
        if row['both_positive_complete'] > min(row['FPM_positive'], row['REV_positive']):
            raise ValueError('joint positive price count exceeds its marginals')
        by_group[group] = row
    for instrument in SHORT:
        for ranking in ['self', 'market']:
            outer, inner = (by_group[(instrument, ranking, stage)] for stage in ['W', 'J'])
            for field in ['n', 'both_positive_complete', *sum(sign_fields.values(), [])]:
                if inner[field] > outer[field]:
                    raise ValueError('conditional price counts are not nested within the W list')
    # Only declared aggregate fields reach publication; private keys never do.
    public = summary.select([f for f in summary.columns if f in fields])
    public.write_csv(out / 'primary_price_summary.csv')
    signs = lambda r, name: '/'.join(str(r[f]) for f in sign_fields[name])
    for ranking in ['self', 'market']:
        rows = [by_group[(i, ranking, stage)] for i in SHORT for stage in ['W', 'J']]
        table(out, f'primary_{ranking}_prices.tex', f'tab:primary-{ranking}-prices',
              f'Midprice outcomes in the {ranking}-normalized WMSCI review lists and their conditional top-$J$ subsets, with no price filtering. '
              'All observable order ages enter with a 30-second exponential half-life and $H=10$ seconds. '
              'W is the top withdrawal decile; W+J is its conditional profile decile. Each four-part entry gives positive/zero/negative/missing execution counts, with $N$ as the full list denominator. '
              'FPM measures favorable movement before execution; REV measures quantity-weighted reversion over two seconds after each assigned cancellation and is missing unless every positive-quantity cancellation has a finite outcome. '
              'Both counts jointly positive FPM and complete REV, not a further selection. Zero includes numerical equality under the endpoint-scaled floating-point roundoff convention in the methods.',
              ['Instrument', 'Stage', '$N$', 'FPM (+/0/--/?)', 'REV (+/0/--/?)', 'Both $>0$'],
              [[SHORT[r['instrument']], 'W' if r['stage'] == 'W' else 'W+J', r['n'], signs(r, 'FPM'), signs(r, 'REV'), r['both_positive_complete']] for r in rows])
    rows = [by_group[(i, ranking, stage)] for i in SHORT for ranking in ['self', 'market'] for stage in ['W', 'J']]
    table(out, 'primary_advantage.tex', 'tab:primary-advantage',
          'Execution-price advantage relative to the completed-display midprice in the primary specification ($H=10$ seconds, exponential age half-life 30 seconds). '
          'Self and market denote the withdrawal normalization; W+J is the conditional profile-priority subset of W. '
          'Four-part entries give positive/zero/negative/missing execution counts; zero includes numerical equality under the endpoint-scaled roundoff convention. Positive means selling above or buying below the benchmark, not realized profit. Prices do not determine list membership.',
          ['Instrument', 'Ranking', 'Stage', '$N$', 'ADV (+/0/--/?)'],
          [[SHORT[r['instrument']], r['ranking'].title(), 'W' if r['stage'] == 'W' else 'W+J', r['n'], signs(r, 'ADV')] for r in rows])
    if 'sign_sensitivity.csv' in source_manifest['artifacts']:
        sensitivity = pl.read_csv(root / 'sign_sensitivity.csv')
        safe = [c for c in sensitivity.columns if c in fields or c == 'ulp_factor']
        sensitivity.select(safe).write_csv(out / 'primary_price_sign_sensitivity.csv')
    if 'role_identity_summary.csv' in source_manifest['artifacts']:
        strata = pl.read_csv(root / 'role_identity_summary.csv')
        strata.select([c for c in strata.columns if c in fields or c in ['execution_anchor_mode', 'identity_level']]).write_csv(out / 'primary_price_strata.csv')
        strata_rows = []
        for instrument in SHORT:
            for ranking in ['self', 'market']:
                for stage in ['W', 'J']:
                    local = strata.filter((pl.col('instrument') == instrument) & (pl.col('ranking') == ranking) & (pl.col('stage') == stage))
                    if local['n'].sum() != by_group[(instrument, ranking, stage)]['n']:
                        raise ValueError('price role strata do not reconcile to list denominator')
                    cells = []
                    for role, identity in [('passive', 'client_original'), ('aggressive', 'client_original'), ('passive', 'firm'), ('aggressive', 'firm')]:
                        sub = local.filter((pl.col('execution_anchor_mode') == role) & (pl.col('identity_level') == identity))
                        cells.append(f"{int(sub['n'].sum())} / {int(sub['both_positive_complete'].sum())}")
                    strata_rows.append([SHORT[instrument], ranking.title() + (' W' if stage == 'W' else ' W+J'), *cells])
        table(out, 'primary_price_strata.tex', 'tab:primary-price-strata',
              'Price evidence by economic role and attribution in the primary review lists. Each entry is the stratum execution count / count with jointly positive FPM and complete REV. W+J is conditional on the corresponding self or market W list. Client denotes original-client attribution; firm denotes fallback attribution, which may pool multiple clients. Strata and lists overlap across rankings and stages; these are descriptive counts, not independent participants or confirmed strategies.',
              ['Instrument', 'List', r'\shortstack{Client\\passive}', r'\shortstack{Client\\aggressive}', r'\shortstack{Firm\\passive}', r'\shortstack{Firm\\aggressive}'], strata_rows, floating=False)
    return {'scope': 'exp_h30_H10_self_market_W_and_conditional_J_no_price_selection',
            'sign_policy': load_json(root / 'metadata.json').get('sign_policy'),
            'summary': public.to_dicts(), 'reversion_horizon_ns': 2_000_000_000,
            'source_complete_sha256': digest(root / 'COMPLETE.json'),
            'source_manifest_sha256': digest(root / 'artifact_manifest.json'),
            'source_metadata_sha256': digest(root / 'metadata.json')}


def build(out):
    if out.exists():
        raise FileExistsError(f'publication output exists: {out}; use a fresh directory')
    validate_stage(REPO / PRIMARY_PRICES, 'primary_prices_verified')
    evidence = [validate_root(REPO / r) for r in DEFAULT_ROOTS]
    validate_stage(REPO / SUMMARY, 'descriptive_campaign_summary_verified')
    totals_frame = pl.read_csv(REPO / SUMMARY / 'instrument_totals.csv')
    reference_counts = pl.read_csv(REPO / SUMMARY / 'cell_counts.csv')
    out.mkdir(parents=True, exist_ok=False)
    cell_rows, composition_rows, ranking_rows, score_rows, age_rows, support_rows = [], [], [], [], [], []
    populations = {}
    for relative, ev in zip(DEFAULT_ROOTS, evidence):
        root = REPO / relative
        # Projection prevents carrying order IDs/price fields into publication outputs.
        columns = KEYS + ['horizon_ns', 'age_policy', 'execution_anchor_mode', 'identity_level',
            'child_fill_count', 'is_candidate', 'candidate_qty_raw', 'withdrawal_qty_raw',
            'candidate_qty_age_weighted', 'withdrawal_qty_age_weighted', 'execution_quantity',
            'opposite_market_visible_qty_pre_topN', 'WMSCI_self', 'WMSCI_market', 'J',
            'complete_grid', 'age_weight_underflow_count'] + FLAGS
        p = pl.read_parquet(root / 'analysis/population.parquet', columns=columns)
        instrument = p['instrument'][0]
        populations[instrument] = p
        if p.select(KEYS + ['horizon_ns', 'age_policy']).is_duplicated().any():
            raise ValueError('duplicate execution/cell key')
        if p.select(KEYS).null_count().row(0) != (0, 0, 0, 0):
            raise ValueError('null execution key')
        raw = (pl.col('candidate_qty_raw') > 0) & (pl.col('withdrawal_qty_raw') > 0) & (pl.col('execution_quantity') > 0)
        if p.filter(pl.col('is_candidate') != raw).height:
            raise ValueError('raw candidate definition mismatch')
        units = p.unique(KEYS)
        candidates = p.filter(pl.col('is_candidate')).unique(KEYS)
        native = totals_frame.filter(pl.col('instrument') == instrument).row(0, named=True)
        observed = {'execution_count': units.height, 'execution_cells': p.height,
                    'fill_member_count': units['child_fill_count'].sum(),
                    'unique_candidate_executions_any_cell': candidates.height,
                    'candidate_cells': p['is_candidate'].sum()}
        if any(observed[k] != native[k] for k in observed) or p.height != units.height * 20:
            raise ValueError('campaign totals mismatch')
        primary_candidates = p.filter((pl.col('age_policy') == PRIMARY_POLICY) & (pl.col('horizon_ns') == PRIMARY_HORIZON_NS) & pl.col('is_candidate'))
        for cohort, frame in [('All executions', units), ('Primary candidates', primary_candidates)]:
            for row in frame.group_by(['execution_anchor_mode', 'identity_level']).len().sort(['execution_anchor_mode', 'identity_level']).iter_rows(named=True):
                composition_rows.append({'instrument': instrument, 'cohort': cohort, 'role': row['execution_anchor_mode'],
                                         'identity_level': row['identity_level'], 'executions': row['len'], 'denominator': frame.height})
        for (horizon, policy), cell in p.partition_by(['horizon_ns', 'age_policy'], as_dict=True).items():
            c = cell.filter(pl.col('is_candidate'))
            row = {'instrument': instrument, 'horizon_ns': horizon, 'horizon_seconds': horizon / 1e9,
                   'age_policy': policy, 'execution_count': cell.height, 'candidate_executions': c.height}
            row.update({f + '_count': int(cell[f].sum()) for f in FLAGS})
            expected = reference_counts.filter((pl.col('instrument') == instrument) & (pl.col('horizon_ns') == horizon) & (pl.col('age_policy') == policy)).row(0, named=True)
            if any(row[k] != expected[k] for k in expected):
                raise ValueError('cell counts disagree with validated campaign summary')
            cell_rows.append(row)
            score_rows.append({'instrument': instrument, 'horizon_seconds': horizon / 1e9, 'age_policy': policy,
                'raw_candidates': c.height, 'positive_self_scores': c.filter(pl.col('WMSCI_self') > 0).height,
                'positive_market_scores': c.filter(pl.col('WMSCI_market') > 0).height,
                'weighted_zero_self_raw_candidates': c.filter(pl.col('WMSCI_self') == 0).height,
                'underflow_affected_raw_candidates': c.filter(pl.col('age_weight_underflow_count') > 0).height,
                'median_P_raw_shares': c['candidate_qty_raw'].median(), 'median_U_raw_shares': c['withdrawal_qty_raw'].median(),
                'median_P_weighted_shares': c['candidate_qty_age_weighted'].median(), 'median_U_weighted_shares': c['withdrawal_qty_age_weighted'].median(),
                'median_W_self': c['WMSCI_self'].median(), 'median_W_market': c['WMSCI_market'].median(),
                'finite_complete_J_candidates': c.filter(pl.col('complete_grid') & pl.col('J').is_finite()).height,
                'zero_complete_J_candidates': c.filter(pl.col('complete_grid') & (pl.col('J') == 0)).height})
            if policy == PRIMARY_POLICY:
                metrics = ranking_metrics(cell)
                if metrics['top_self'] != metrics['self_ceiling_budget'] or metrics['top_market'] != metrics['market_ceiling_budget']:
                    raise ValueError('stored primary lists disagree with ceiling budgets')
                ranking_rows.append({'instrument': instrument, 'horizon_seconds': horizon / 1e9,
                    'age_policy': policy, **metrics})
        support = pl.read_csv(root / 'report/common_support_counts.csv')
        # Check the saved horizon-common complete-profile support, rather than inventing a sampler.
        for policy in POLICIES:
            sub = p.filter(pl.col('age_policy') == policy)
            union = keyset(sub.filter(pl.col('is_candidate')))
            complete_sets = [keyset(sub.filter((pl.col('horizon_ns') == h) & pl.col('complete_grid') & pl.col('J').is_finite())) & union for h in HORIZONS]
            common = set.intersection(*complete_sets)
            for h, supported in zip(HORIZONS, complete_sets):
                saved = support.filter((pl.col('age_policy') == policy) & (pl.col('horizon_ns') == h)).row(0, named=True)
                if (len(union), len(supported), len(common)) != (saved['union_candidates'], saved['complete_support_at_horizon'], saved['common_complete_support']):
                    raise ValueError('saved common profile support mismatch')
                support_rows.append(saved)
        for h in HORIZONS:
            base = p.filter((pl.col('horizon_ns') == h) & (pl.col('age_policy') == 'all_ages_unweighted'))
            for policy in POLICIES:
                if policy == 'all_ages_unweighted':
                    continue
                other = p.filter((pl.col('horizon_ns') == h) & (pl.col('age_policy') == policy))
                joined = base.join(other, on=KEYS, suffix='_other').filter(pl.col('is_candidate') & pl.col('is_candidate_other') & pl.col('complete_grid') & pl.col('complete_grid_other') & pl.col('J').is_finite() & pl.col('J_other').is_finite())
                if joined.filter(pl.col('J') != pl.col('J_other')).height:
                    raise ValueError('J changes with age policy on common complete profile support')
                for score, flag in [('WMSCI_self', FLAGS[0]), ('WMSCI_market', FLAGS[1])]:
                    shared = joined.filter((pl.col(score) > 0) & (pl.col(score + '_other') > 0) & pl.col(score).is_finite() & pl.col(score + '_other').is_finite())
                    correlation = shared.select(pl.corr(pl.col(score).rank(), pl.col(score + '_other').rank())).item() if shared.height > 1 else None
                    if correlation is not None and not math.isfinite(correlation):
                        correlation = None
                    a = keyset(shared.filter(pl.col(flag)))
                    b = keyset(shared.filter(pl.col(flag + '_other')))
                    age_rows.append({'instrument': instrument, 'horizon_seconds': h / 1e9, 'comparison_policy': policy,
                        'score': score, 'common_complete_raw_candidates': joined.height, 'common_positive_score_support': shared.height,
                        'spearman_average_ties': correlation, 'baseline_top_on_common_support': len(a),
                        'comparison_top_on_common_support': len(b), **overlap_metrics(a, b)})
    counts = pl.DataFrame(cell_rows).sort(['instrument', 'horizon_ns', 'age_policy'])
    ranking = pl.DataFrame(ranking_rows).sort(['instrument', 'horizon_seconds'])
    for name, frame in [('cell_counts.csv', counts), ('ranking_overlap.csv', ranking), ('composition.csv', pl.DataFrame(composition_rows)),
                        ('age_common_support.csv', pl.DataFrame(age_rows)), ('score_quantities.csv', pl.DataFrame(score_rows)),
                        ('common_support_counts.csv', pl.DataFrame(support_rows))]:
        frame.write_csv(out / name)
    totals_frame.drop('root').write_csv(out / 'instrument_totals.csv')
    # Retained real diagnostics, strict public allow-list; no actor/order IDs in CSV or TeX.
    case_provenance = load_json(REPO / DIAGNOSTICS / 'case_provenance.json')
    case_frames = []
    allowed = ['instrument', 'execution_anchor_mode', 'execution_quantity', 'horizon_ns', 'age_policy',
               'candidate_qty_raw', 'withdrawal_qty_raw', 'candidate_qty_age_weighted', 'withdrawal_qty_age_weighted',
               'opposite_market_visible_qty_pre_topN', 'WMSCI_self', 'WMSCI_market', 'J', 'is_candidate'] + FLAGS
    for case_id, filename in [('R', 'RISANAMENTO_17295_multiscale.csv'), ('N', 'NEXI_31425_multiscale.csv')]:
        path = REPO / DIAGNOSTICS / filename
        records = case_provenance['cases']
        record = next(r for r in records if Path(r['output']).name == filename)
        if digest(path) != record['output_sha256'] or digest(REPO / record['source']) != record['source_sha256']:
            raise ValueError('retained case diagnostic hash mismatch')
        original = pl.read_csv(path, infer_schema_length=None)
        if original.height != 20 or original.select(KEYS).unique().height != 1:
            raise ValueError('case must be one execution repeated over 20 cells')
        native = populations[original['instrument'][0]].join(original.select(KEYS).unique(), on=KEYS)
        for r in original.iter_rows(named=True):
            nr = native.filter((pl.col('horizon_ns') == r['horizon_ns']) & (pl.col('age_policy') == r['age_policy'])).row(0, named=True)
            if any(r[k] != nr[k] for k in allowed):
                raise ValueError('case diagnostic differs from frozen population')
        case_frames.append(original.select(allowed).with_columns(pl.lit(case_id).alias('case_id'), (pl.col('horizon_ns') / 1e9).alias('horizon_seconds')))
    cases = pl.concat(case_frames).sort(['case_id', 'age_policy', 'horizon_ns'])
    cases.write_csv(out / 'cases.csv')
    totals = {k: int(totals_frame[k].sum()) for k in ['execution_count', 'fill_member_count', 'execution_cells', 'unique_candidate_executions_any_cell', 'candidate_cells']}
    if totals != load_json(REPO / SUMMARY / 'COMPLETE.json')['totals']:
        raise ValueError('summary completion totals mismatch')
    table(out, 'population.tex', 'tab:wmsci-j-population',
          'Execution population, child fills, and raw candidates under the primary exponential age kernel (30-second half-life) at $H=10$ s. A raw candidate has positive unweighted displayed quantity, attributed withdrawal, and execution quantity; candidature does not require a positive weighted score. Counts are distinct execution keys, not repeated design cells.',
          ['Instrument', 'Executions', 'Fills', 'Candidates ($H=10$ s)'],
          [[SHORT[r['instrument']], r['execution_count'], r['fill_member_count'], counts.filter((pl.col('instrument') == r['instrument']) & (pl.col('age_policy') == PRIMARY_POLICY) & (pl.col('horizon_ns') == PRIMARY_HORIZON_NS))['candidate_executions'][0]] for r in totals_frame.iter_rows(named=True)])
    table(out, 'horizon_counts.tex', 'tab:multiscale-horizon-counts',
          'Raw candidate executions at each withdrawal horizon. Entries are 30-second age cap / all ages without weighting. Both definitions use the same execution population; neither is a percentile selection. Horizons are evaluated separately, so counts need not be monotone under exclusive cancellation attribution.',
          ['Instrument', '$H=1$ s', '$H=2$ s', '$H=5$ s', '$H=10$ s'],
          [[SHORT[i]] + [str(counts.filter((pl.col('instrument') == i) & (pl.col('horizon_ns') == h) & (pl.col('age_policy') == 'legacy_age30'))['candidate_executions'][0]) + ' / ' + str(counts.filter((pl.col('instrument') == i) & (pl.col('horizon_ns') == h) & (pl.col('age_policy') == 'all_ages_unweighted'))['candidate_executions'][0]) for h in HORIZONS] for i in SHORT], floating=False)
    table(out, 'ranking_comparison.tex', 'tab:multiscale-ranking-comparison',
          'Top-decile withdrawal priorities with all observable ages and the primary 30-second exponential half-life. Positive support reports finite positive-score candidates for self / market WMSCI. Each list uses its own support and upward-rounded budget $\\lceil 0.1n\\rceil$; numerical score zeros can make the supports differ. Self and Market count the respective review lists, Both their intersection, and Jaccard their intersection divided by their union. Lists are not reranked on shared support. Conditional $J$ flags are separate.',
          ['Instrument', '$H$ (s)', r'\shortstack{Positive support\\self / market}', 'Self', 'Market', 'Both', 'Jaccard'],
          [[SHORT[r['instrument']], int(r['horizon_seconds']), f"{r['positive_self_scores']:,} / {r['positive_market_scores']:,}", r['top_self'], r['top_market'], r['intersection'], r['jaccard']] for r in ranking.iter_rows(named=True)], floating=False)
    table(out, 'composition.tex', 'tab:multiscale-composition',
          'Execution composition by economic role and attribution level. Each entry reports all executions / primary raw candidates under the 30-second half-life exponential kernel at $H=10$ s. Client attribution uses original client identity; firm attribution is a fallback. These strata are descriptive and have different coverage across instruments.',
          ['Instrument', 'Aggressive/client', 'Aggressive/firm', 'Passive/client', 'Passive/firm'],
          [[SHORT[i]] + [str(next(r['executions'] for r in composition_rows if r['instrument'] == i and r['cohort'] == 'All executions' and r['role'] == role and r['identity_level'] == ident)) + ' / ' + str(next((r['executions'] for r in composition_rows if r['instrument'] == i and r['cohort'] == 'Primary candidates' and r['role'] == role and r['identity_level'] == ident), 0)) for role, ident in [('aggressive', 'client_original'), ('aggressive', 'firm'), ('passive', 'client_original'), ('passive', 'firm')]] for i in SHORT])
    transparent = cases.filter(pl.col('age_policy') == PRIMARY_POLICY)
    table(out, 'anonymouscase_table.tex', 'tab:multiscale-anonymous-cases',
          'Two post-hoc diagnostic examples, labelled R (Risanamento) and N (Nexi), evaluated at four discrete horizons under the primary 30-second half-life exponential kernel. $P$ and $U$ are raw share quantities, distinct from the age-weighted quantities used in $W$; $J$ is the complete unweighted profile statistic. These are repeated evaluations of two executions, not eight independent cases or time trajectories.',
          ['Case', '$H$ (s)', '$P$ (shares)', '$U$ (shares)', r'$W^{\mathrm{self}}$', r'$W^{\mathrm{market}}$', '$J$'],
          [[r['case_id'], int(r['horizon_seconds']), int(r['candidate_qty_raw']), int(r['withdrawal_qty_raw']), r['WMSCI_self'], r['WMSCI_market'], r['J']] for r in transparent.iter_rows(named=True)])
    age_names = {'legacy_age30': 'Cap 30 s', 'all_ages_unweighted': 'All ages', 'exp_h30': 'Half-life 30 s', 'exp_h120': 'Half-life 120 s', 'exp_h300': 'Half-life 300 s'}
    age_summary_rows = []
    for instrument in SHORT:
        for policy in POLICIES:
            endpoints = [counts.filter((pl.col('instrument') == instrument) & (pl.col('age_policy') == policy) & (pl.col('horizon_ns') == h)).row(0, named=True) for h in [HORIZONS[0], HORIZONS[-1]]]
            age_summary_rows.append([SHORT[instrument], age_names[policy], endpoints[0]['candidate_executions'], endpoints[1]['candidate_executions'], endpoints[0][FLAGS[1] + '_count'], endpoints[1][FLAGS[1] + '_count']])
    table(out, 'age_policy_summary.tex', 'tab:multiscale-age-policies',
          'Age-policy sensitivity at the shortest and longest withdrawal horizons. All ages means no age cap and no weighting. Half-life policies retain all ages but exponentially weight quantities with the stated half-life; the cap policy retains orders aged at most 30 seconds without weighting. Raw counts do not depend on weighted-score underflow. Market counts are stored top-decile market-score priorities and can change with both ranking and positive-score support. All intermediate horizons are retained in the companion data.',
          ['Instrument', 'Age policy', 'Raw 1 s', 'Raw 10 s', 'Market 1 s', 'Market 10 s'], age_summary_rows, floating=False)
    plt.rcParams.update({'font.size': 15, 'axes.labelsize': 15, 'axes.titlesize': 16,
                         'xtick.labelsize': 15, 'ytick.labelsize': 15, 'legend.fontsize': 15,
                         'pdf.fonttype': 42, 'font.family': 'DejaVu Sans'})
    figures = []
    fig, axes = plt.subplots(3, 1, figsize=(6.8, 8.7), layout='constrained')
    for ax, instrument in zip(axes, SHORT):
        for policy, label, color, marker in [(PRIMARY_POLICY, 'Half-life 30 s', '#0072B2', 'o')]:
            subset = counts.filter((pl.col('instrument') == instrument) & (pl.col('age_policy') == policy)).sort('horizon_ns')
            ax.plot(subset['horizon_seconds'], subset['candidate_executions'], color=color, marker=marker, label=label)
        ax.set_title(SHORT[instrument]); ax.set_ylabel('Raw candidates (count)'); ax.set_xticks([1, 2, 5, 10]); ax.set_ylim(bottom=0)
        ax.grid(alpha=.2)
    axes[0].legend(loc='best'); axes[-1].set_xlabel('Withdrawal horizon H (seconds)')
    figures.append(save_figure(fig, out, 'candidate_counts.pdf', 'Raw candidate execution counts across four withdrawal horizons. Panels use only the primary exponential kernel with 30-second half-life; $H=10$ s is primary and shorter horizons are sensitivity checks. Membership uses positive raw displayed, withdrawal, and execution quantities. Lines connect design points, not an event-time trajectory; counts are not prevalence estimates. All 20 cells per execution reuse the same records.', 'cell_counts.csv'))
    fig, ax = plt.subplots(figsize=(6.8, 4.4), layout='constrained')
    for instrument, marker in zip(SHORT, ['o', 's', '^']):
        subset = ranking.filter(pl.col('instrument') == instrument).sort('horizon_seconds')
        ax.plot(subset['horizon_seconds'], subset['jaccard'], marker=marker, label=SHORT[instrument])
    ax.set_xticks([1, 2, 5, 10]); ax.set_ylim(0, 1); ax.set_xlabel('Withdrawal horizon H (seconds)'); ax.set_ylabel('Top-W intersection / union'); ax.legend(); ax.grid(alpha=.2)
    figures.append(save_figure(fig, out, 'ranking_overlap.pdf', 'Overlap of stored self- and market-normalized top-decile withdrawal priorities under the primary 30-second half-life exponential kernel. Each list uses its own positive finite-score candidate support and ceiling budget; numerical zeros can make supports and budgets unequal (see ranking_overlap.csv). Lists are not reranked on shared support. The plotted Jaccard index is intersection divided by union of selected execution keys. Differences describe review prioritization, not detection accuracy.', 'ranking_overlap.csv'))
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 6.8), layout='constrained')
    for col, case_id in enumerate(['R', 'N']):
        subset = transparent.filter(pl.col('case_id') == case_id).sort('horizon_ns')
        ratios = (subset['withdrawal_qty_raw'] / subset['candidate_qty_raw']).to_list()
        axes[0, col].scatter(subset['horizon_seconds'], ratios, color='#0072B2', s=45)
        axes[1, col].scatter(subset['horizon_seconds'], subset['J'], color='#D55E00', s=45)
        axes[0, col].set_title('Case ' + case_id); axes[0, col].set_ylim(-.05, 1.05)
        axes[1, col].set_ylim(-.08, 2.0); axes[1, col].set_xlabel('Discrete H (seconds)')
        for ax in axes[:, col]:
            ax.set_xticks([1, 2, 5, 10]); ax.grid(alpha=.2)
    axes[0, 0].set_ylabel('Raw withdrawal U / P'); axes[1, 0].set_ylabel('Positive MSCI increment J')
    figures.append(save_figure(fig, out, 'withdrawal_and_profile_summary.pdf', 'Discrete-horizon summaries for post-hoc diagnostic examples R (Risanamento) and N (Nexi) under the primary 30-second half-life exponential kernel. Upper panels show raw attributed withdrawal divided by raw displayed quantity; lower panels show the complete unweighted profile statistic J. Points are evaluations of the same execution at H=1,2,5,10 seconds, not observed trajectories. A positive J does not replace the raw withdrawal condition.', 'cases.csv'))
    baseline_root = REPO / 'paper/generated/new_kernel_h1_top10_manuscript'
    baseline_metadata = load_json(baseline_root / 'metadata.json')
    baseline_bindings = {}
    for filename in ['prices.tex', 'advantage.tex', 'conditional_price_outcomes.csv']:
        source = baseline_root / filename
        if digest(source) != baseline_metadata['artifacts'][filename]:
            raise ValueError(f'reference baseline artifact hash mismatch: {filename}')
        baseline_bindings[filename] = digest(source)
        if source.suffix == '.tex':
            text = source.read_text()
            text = text.replace(r'\caption{', r'\caption{Reference baseline: one-second withdrawal and profile horizons, a 30-second candidate-age limit, and self-normalized WMSCI. ', 1)
            (out / ('baseline_' + filename)).write_text(text)
        else:
            (out / ('baseline_' + filename)).write_bytes(source.read_bytes())
    primary_prices = render_primary_price_tables(out, REPO / PRIMARY_PRICES, counts)
    facts = {'version': 'withdrawal_multiscale_publication_consumer_primary_prices_v1',
             'primary_prices': primary_prices,
             'primary_policy': PRIMARY_POLICY, 'primary_horizon_ns': PRIMARY_HORIZON_NS,
             'primary_population': counts.filter((pl.col('age_policy') == PRIMARY_POLICY) & (pl.col('horizon_ns') == PRIMARY_HORIZON_NS)).to_dicts(),
             'primary_candidate_executions': int(counts.filter((pl.col('age_policy') == PRIMARY_POLICY) & (pl.col('horizon_ns') == PRIMARY_HORIZON_NS))['candidate_executions'].sum()),
             'raw_grid_union_totals': totals,
             'primary_choice': 'provisional shortest tested half-life; not calibrated, optimal, or independently validated', 'totals': totals, 'figures': figures,
             'reference_price_baseline': {'scope': 'H1_age30_self_priority_only_not_full_multiscale_prices',
                 'source_artifacts': baseline_bindings, 'source_metadata_sha256': digest(baseline_root / 'metadata.json'),
                 'parity_evidence': 'paper/reviews/20261014_091222_multiscale_simple/baseline_membership_evidence.json'},
             'horizons_seconds': [1, 2, 5, 10], 'age_policies': POLICIES,
             'units': {'counts': 'distinct execution keys per declared cohort/cell; cells repeated, not independent',
                       'P_raw_U_raw_Q_D': 'shares', 'P_g_U_g': 'age-weighted shares', 'W_self_W_market_J': 'dimensionless', 'horizons': 'seconds; source comparisons integer nanoseconds'},
             'keys': KEYS, 'candidate_rule': 'P_raw > 0 and U_raw > 0 and Q > 0; no J exclusion or percentile',
             'market_identity': 'W_market = W_self * P_g / D on supported market denominator',
             'conditional_J': 'stored conditional top-J flags use complete finite J including zero; separate from top-W flags',
             'age_common_support': 'age_common_support.csv compares scores on shared raw candidates with complete finite J and positive finite score under both policies; Spearman uses average tie ranks; top flags are original within-cell selections restricted to that support, NOT reranked top deciles',
             'J_age_invariance': 'verified equality at fixed execution/horizon on common complete-profile support',
             'common_horizon_profile_support': 'common_support_counts.csv validated against saved report/common_support_counts.csv using candidate union and complete finite profiles at all four horizons',
             'cases': 'two post-hoc diagnostic examples repeated over 20 cells each; identifiers omitted from public CSV/TeX; provenance retained internally',
             'primary_case_quantities': transparent.to_dicts(),
             'source_evidence': evidence, 'summary_manifest_sha256': digest(REPO / SUMMARY / 'artifact_manifest.json'),
             'diagnostic_provenance_sha256': digest(REPO / DIAGNOSTICS / 'case_provenance.json'),
             'generator_sha256': digest(Path(__file__)), 'config_sha256': digest(REPO / 'configs/withdrawal_multiscale_v1.json'),
             'environment': {'python': platform.python_version(), 'polars': pl.__version__, 'matplotlib': matplotlib.__version__},
             'selection_counts': cell_rows, 'ranking_overlap': ranking_rows, 'age_priority_overlap': age_rows,
             'raw_score_quantities': score_rows, 'limitations': ['descriptive triage signals, no intent inference', 'no price evidence generated', 'no horizon declared optimal', 'instrument coverage differs', 'no replay, new sampler, bootstrap or confidence intervals']}
    write_json(out / 'facts.json', facts)
    write_json(out / 'style_metadata.json', {'fonts': 'DejaVu Sans, embedded TrueType', 'minimum_font_pt': 13,
        'maximum_width_inches': 6.8, 'publication_width': 'use textwidth near 6.8 inches; narrow-column scaling reduces readability',
        'geometry': 'all visible text bounding boxes inside canvas verified before save', 'figures': figures})
    artifacts = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file() and p.name not in ['artifact_manifest.json', 'COMPLETE.json']}
    write_json(out / 'artifact_manifest.json', {'version': facts['version'], 'artifacts': artifacts})
    write_json(out / 'COMPLETE.json', {'status': 'publication_package_verified', 'artifact_manifest_sha256': digest(out / 'artifact_manifest.json'), 'totals': totals})
    verify_manifest(out)
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output if args.output.is_absolute() else REPO / args.output
    # This command owns only its fresh package, never legacy assets or the manuscript.
    if output.resolve().parent != (REPO / 'paper/generated').resolve() or not output.name.startswith('withdrawal_multiscale_'):
        parser.error('output must be a fresh withdrawal_multiscale_* package under paper/generated')
    facts = build(output)
    print(json.dumps({'output': str(output), 'totals': facts['totals'], 'figures': [f['path'] for f in facts['figures']]}, indent=2))


if __name__ == '__main__':
    main()
