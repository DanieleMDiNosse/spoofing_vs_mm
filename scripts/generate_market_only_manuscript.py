#!/usr/bin/env python3
"""Read-only market-normalized publication consumer for authenticated v4 artifacts.

No replay, fitting, reranking, or price selection. Existing outputs are refused.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import platform
import shutil
from pathlib import Path
import polars as pl

REPO = Path(__file__).resolve().parents[1]
OUTPUT = Path('paper/generated/withdrawal_market_20261015_v1')
PREVIOUS = Path('paper/generated/withdrawal_multiscale_20261014_primary_prices_v1')
PRIMARY_PRICES = Path('outputs/primary_kernel30_h10_prices_20261014_213056_numeric_v2')
PRIMARY_POLICY = 'exp_h30'
PRIMARY_HORIZON_NS = 10_000_000_000
KEYS = ['instrument', 'partition_id', 'actor_key', 'execution_cluster_id']
WFLAG = 'priority_top_w_market'
JFLAG = 'priority_top_j_given_market'
FLAGS = [WFLAG, JFLAG]
HORIZONS = [1_000_000_000, 2_000_000_000, 5_000_000_000, 10_000_000_000]
POLICIES = ['legacy_age30', 'all_ages_unweighted', 'exp_h30', 'exp_h120', 'exp_h300']
SHORT = {'RISANAMENTO': 'Risanamento', 'NEXI': 'Nexi', 'FERRARI': 'Ferrari'}

# Import only unchanged authentication/rendering primitives, never its build.
_spec = importlib.util.spec_from_file_location('frozen_multiscale_helpers', Path(__file__).with_name('generate_withdrawal_multiscale_manuscript.py'))
assert _spec is not None and _spec.loader is not None
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
validate_root = _old.validate_root
validate_stage = _old.validate_stage
digest = _old.digest
load_json = _old.load_json
write_json = _old.write_json
table = _old.table
DEFAULT_ROOTS = _old.DEFAULT_ROOTS
SUMMARY = _old.SUMMARY
DIAGNOSTICS = _old.DIAGNOSTICS


def keyset(frame):
    return set(frame.select(KEYS).iter_rows())


def boolean_flags(frame, fields):
    for field in fields:
        if frame.schema[field] != pl.Boolean or frame[field].null_count():
            raise ValueError(f'non-null Boolean required: {field}')


def market_metrics(cell):
    """Validate stored market memberships without re-estimating their ranks."""
    boolean_flags(cell, ['is_candidate', 'complete_grid', *FLAGS])
    eligible = cell.filter(pl.col('is_candidate') & pl.col('WMSCI_market').is_finite() & (pl.col('WMSCI_market') > 0))
    w = cell.filter(pl.col(WFLAG))
    finite_j = w.filter(pl.col('complete_grid') & pl.col('J').is_finite())
    j = cell.filter(pl.col(JFLAG))
    wb, jb = math.ceil(eligible.height / 10), math.ceil(finite_j.height / 10)
    if not keyset(w) <= keyset(eligible) or not keyset(j) <= keyset(finite_j):
        raise ValueError('market priority flag outside eligible or nested support')
    if w.height != wb or j.height != jb:
        raise ValueError('market flag counts disagree with ceiling budgets')
    return {'candidate_executions': int(cell['is_candidate'].sum()),
            'positive_market_support': eligible.height, 'market_w_budget': wb,
            'market_w_count': w.height, 'conditional_finite_J_support': finite_j.height,
            'market_j_budget': jb, 'market_j_count': j.height}


def verify_price_membership(prices, native):
    """Full execution-key equality and null-safe flag agreement, not count parity."""
    for frame in [prices, native]:
        boolean_flags(frame, FLAGS)
        if frame.select(KEYS).is_duplicated().any() or any(frame.select(KEYS).null_count().row(0)):
            raise ValueError('duplicate or null price key')
    if keyset(prices) != keyset(native):
        raise ValueError('market price key set differs from native selection')
    joined = prices.select(KEYS + FLAGS).join(native.select(KEYS + FLAGS), on=KEYS, suffix='_native')
    if any(joined.filter(pl.col(f).ne_missing(pl.col(f + '_native'))).height for f in FLAGS):
        raise ValueError('market price flag differs from native selection')


def primary(frame):
    return frame.filter((pl.col('age_policy') == PRIMARY_POLICY) & (pl.col('horizon_ns') == PRIMARY_HORIZON_NS))


def summarize_prices(frame, *, roles=False, factor=8):
    """Recount all signs and normalized medians from unchanged raw price rows."""
    rows = []
    for instrument in sorted(SHORT):
        for stage, flag in [('W', WFLAG), ('J', JFLAG)]:
            selected = frame.filter((pl.col('instrument') == instrument) & pl.col(flag))
            groups = selected.partition_by(['execution_anchor_mode', 'identity_level'], as_dict=True) if roles else {(): selected}
            for stratum, sub in groups.items():
                row = {'instrument': instrument, 'ranking': 'market', 'stage': stage, 'n': sub.height}
                if roles:
                    row.update(dict(zip(['execution_anchor_mode', 'identity_level'], stratum)))
                signs = {}
                for metric in ['FPM', 'REV', 'ADV']:
                    values, normalized = [], []
                    for raw, tolerance in sub.select(metric, metric + '_zero_tolerance').iter_rows():
                        sign = None if raw is None or not math.isfinite(raw) else (0 if abs(raw) <= tolerance * factor / 8 else (1 if raw > 0 else -1))
                        values.append(sign)
                        if sign is not None:
                            normalized.append(0. if sign == 0 else raw)
                    signs[metric] = values
                    for name, value in [('positive', 1), ('zero', 0), ('negative', -1)]:
                        row[metric + '_' + name] = values.count(value)
                    missing = 'REV_missing_or_incomplete' if metric == 'REV' else metric + '_missing'
                    row[missing] = values.count(None)
                    row[metric + '_defined'] = len(normalized)
                    row[metric + '_median'] = pl.Series(normalized, dtype=pl.Float64).median()
                row['both_positive_complete'] = sum(a == 1 and b == 1 for a, b in zip(signs['FPM'], signs['REV']))
                rows.append(row)
    return pl.DataFrame(rows)


def assert_same_records(actual, expected, keys):
    if set(actual.columns) != set(expected.columns):
        raise ValueError('summary schema mismatch')
    a = actual.select(expected.columns).sort(keys)
    b = expected.sort(keys)
    if a.height != b.height:
        raise ValueError('summary row membership mismatch')
    for left, right in zip(a.iter_rows(named=True), b.iter_rows(named=True)):
        for name in left:
            x, y = left[name], right[name]
            if x != y and not (isinstance(x, float) and isinstance(y, float) and math.isclose(x, y, rel_tol=1e-12, abs_tol=1e-15)):
                raise ValueError(f'summary reconciliation failed: {name}: {x} != {y}')


def render_prices(out, prices, summary, strata):
    signs = lambda r, m: '/'.join(str(r[m + '_' + suffix]) for suffix in ['positive', 'zero', 'negative', 'missing_or_incomplete' if m == 'REV' else 'missing'])
    ordered = [summary.filter((pl.col('instrument') == i) & (pl.col('stage') == s)).row(0, named=True) for i in SHORT for s in ['W', 'J']]
    caption = ('Midprice outcomes in the market-normalized WMSCI priorities, without price filtering. '
        'The age half-life is 30 seconds and withdrawal horizon $H=10$ seconds. W is the top withdrawal decile on finite positive-score candidates; W+J is the conditional top profile decile among W executions with complete finite $J$, including zero. '
        'Each entry is positive/zero/negative/missing, with the full list size $N$ as denominator. FPM is favorable pre-execution movement. REV is raw-quantity-weighted reversion over two seconds after each attributed cancellation, not the ten-second withdrawal horizon; it is missing unless every positive-quantity cancellation has a finite outcome. Both counts jointly positive FPM and complete REV. Zero uses endpoint-scaled floating-point roundoff, not an economic threshold.')
    table(out, 'primary_market_prices.tex', 'tab:primary-market-prices', caption,
        ['Instrument', 'Stage', '$N$', 'FPM (+/0/--/?)', 'REV (+/0/--/?)', 'Both $>0$'],
        [[SHORT[r['instrument']], 'W' if r['stage'] == 'W' else 'W+J', r['n'], signs(r, 'FPM'), signs(r, 'REV'), r['both_positive_complete']] for r in ordered])
    table(out, 'primary_advantage.tex', 'tab:primary-advantage',
        'Execution-price advantage relative to the completed-display midprice for the market-normalized review priorities (30-second age half-life, $H=10$ seconds). W+J is the conditional profile decile within W. Entries count positive/zero/negative/missing observations with full list denominator $N$. Zero uses endpoint-scaled roundoff. Positive means selling above or buying below the benchmark, not realized profit. Prices do not select executions.',
        ['Instrument', 'Stage', '$N$', 'ADV (+/0/--/?)'],
        [[SHORT[r['instrument']], 'W' if r['stage'] == 'W' else 'W+J', r['n'], signs(r, 'ADV')] for r in ordered])
    rows = []
    for r in ordered:
        local = strata.filter((pl.col('instrument') == r['instrument']) & (pl.col('stage') == r['stage']))
        entries = []
        for role, identity in [('passive', 'client_original'), ('aggressive', 'client_original'), ('passive', 'firm'), ('aggressive', 'firm')]:
            sub = local.filter((pl.col('execution_anchor_mode') == role) & (pl.col('identity_level') == identity))
            entries.append(f"{int(sub['n'].sum())} / {int(sub['both_positive_complete'].sum())}")
        rows.append([SHORT[r['instrument']], 'W' if r['stage'] == 'W' else 'W+J', *entries])
    table(out, 'primary_price_strata.tex', 'tab:primary-price-strata',
        'Primary market-priority price evidence by source-derived execution role and attribution. Entries are stratum executions / executions with jointly positive FPM and complete two-second REV. W+J is conditional on W; the withdrawal horizon is ten seconds. Client denotes original-client attribution; firm fallback may pool clients. Strata partition each list, while W and W+J overlap. These are descriptive counts, not independent participants or confirmed strategies.',
        ['Instrument', 'List', r'\shortstack{Client\\passive}', r'\shortstack{Client\\aggressive}', r'\shortstack{Firm\\passive}', r'\shortstack{Firm\\aggressive}'], rows, floating=False)


def build(out):
    out = Path(out)
    if out.exists():
        raise FileExistsError(f'publication output exists: {out}; use a fresh directory')
    # All authentication happens before creating output, including reused assets.
    evidence = []
    for root in DEFAULT_ROOTS:
        authenticated = validate_root(REPO / root)
        # Public provenance uses hashes, not the producer's broad code inventory.
        # The latter includes unrelated modules outside this publication's scope.
        evidence.append({k: v for k, v in authenticated.items() if k != 'report_provenance'})
        evidence[-1]['report_provenance_sha256'] = digest(REPO / root / 'report/provenance.json')
    validate_stage(REPO / SUMMARY, 'descriptive_campaign_summary_verified')
    validate_stage(REPO / PRIMARY_PRICES, 'primary_prices_verified')
    validate_stage(REPO / PREVIOUS, 'publication_package_verified')
    totals = pl.read_csv(REPO / SUMMARY / 'instrument_totals.csv').drop('root')
    reference = pl.read_csv(REPO / SUMMARY / 'cell_counts.csv')
    previous_counts = pl.read_csv(REPO / PREVIOUS / 'cell_counts.csv')
    previous_cases = pl.read_csv(REPO / PREVIOUS / 'cases.csv')
    populations, count_rows, composition = {}, [], []
    columns = KEYS + ['horizon_ns', 'age_policy', 'execution_anchor_mode', 'identity_level',
        'child_fill_count', 'is_candidate', 'candidate_qty_raw', 'withdrawal_qty_raw',
        'candidate_qty_age_weighted', 'withdrawal_qty_age_weighted', 'execution_quantity',
        'opposite_market_visible_qty_pre_topN', 'WMSCI_market', 'J', 'complete_grid', *FLAGS]
    for relative in DEFAULT_ROOTS:
        p = pl.read_parquet(REPO / relative / 'analysis/population.parquet', columns=columns)
        instrument = p['instrument'][0]
        populations[instrument] = p
        if any(p.select(KEYS).null_count().row(0)) or p.select(KEYS + ['horizon_ns', 'age_policy']).is_duplicated().any():
            raise ValueError('null or duplicate native full key')
        if any(p[c].null_count() for c in ['execution_anchor_mode', 'identity_level']):
            raise ValueError('missing role or identity stratum')
        boolean_flags(p, ['is_candidate', 'complete_grid', *FLAGS])
        raw = (pl.col('candidate_qty_raw') > 0) & (pl.col('withdrawal_qty_raw') > 0) & (pl.col('execution_quantity') > 0)
        if p.filter(pl.col('is_candidate').ne_missing(raw)).height:
            raise ValueError('native raw candidate definition mismatch')
        units = p.unique(KEYS)
        observed = {'execution_count': units.height, 'execution_cells': p.height,
            'fill_member_count': int(units['child_fill_count'].sum()),
            'unique_candidate_executions_any_cell': p.filter(pl.col('is_candidate')).unique(KEYS).height,
            'candidate_cells': int(p['is_candidate'].sum())}
        expected = totals.filter(pl.col('instrument') == instrument).row(0, named=True)
        if any(observed[k] != expected[k] for k in observed) or p.height != units.height * 20:
            raise ValueError('native campaign totals mismatch')
        for (horizon, policy), cell in p.partition_by(['horizon_ns', 'age_policy'], as_dict=True).items():
            r = {'instrument': instrument, 'horizon_ns': horizon, 'horizon_seconds': horizon / 1e9,
                 'age_policy': policy, 'execution_count': cell.height, **market_metrics(cell)}
            saved = reference.filter((pl.col('instrument') == instrument) & (pl.col('horizon_ns') == horizon) & (pl.col('age_policy') == policy)).row(0, named=True)
            if any(r[k] != saved[k] for k in ['execution_count', 'candidate_executions']) or any(r[name] != saved[flag + '_count'] for name, flag in [('market_w_count', WFLAG), ('market_j_count', JFLAG)]):
                raise ValueError('native market counts differ from campaign summary')
            prev = previous_counts.filter((pl.col('instrument') == instrument) & (pl.col('horizon_ns') == horizon) & (pl.col('age_policy') == policy)).row(0, named=True)
            if r['candidate_executions'] != prev['candidate_executions']:
                raise ValueError('reused candidate figure companion mismatch')
            count_rows.append(r)
        for cohort, frame in [('All executions', units), ('Primary candidates', primary(p).filter(pl.col('is_candidate')))]:
            for row in frame.group_by(['execution_anchor_mode', 'identity_level']).len().iter_rows(named=True):
                composition.append({'instrument': instrument, 'cohort': cohort, 'role': row['execution_anchor_mode'], 'identity_level': row['identity_level'], 'executions': row['len'], 'denominator': frame.height})
    counts = pl.DataFrame(count_rows).sort(['instrument', 'age_policy', 'horizon_ns'])
    native = pl.concat([primary(p) for p in populations.values()])
    native_selection = native.filter(pl.col(WFLAG) | pl.col(JFLAG))
    price_columns = KEYS + ['horizon_ns', 'age_policy', 'execution_anchor_mode', 'identity_level', *FLAGS,
        'FPM', 'REV', 'ADV', 'REV_complete', 'REV_cancel_count', 'REV_missing_cancel_count',
        *[m + suffix for m in ['FPM', 'REV', 'ADV'] for suffix in ['_sign', '_zero_tolerance']]]
    prices = pl.read_parquet(REPO / PRIMARY_PRICES / 'private_execution_prices.parquet', columns=price_columns)
    boolean_flags(prices, FLAGS)
    prices = prices.filter(pl.col(WFLAG) | pl.col(JFLAG))
    verify_price_membership(prices, native_selection)
    if prices.height != primary(prices).height:
        raise ValueError('price policy or horizon mismatch')
    paired = prices.join(native_selection.select(KEYS + ['execution_anchor_mode', 'identity_level']), on=KEYS, suffix='_native')
    if any(paired.filter(pl.col(c).ne_missing(pl.col(c + '_native'))).height for c in ['execution_anchor_mode', 'identity_level']):
        raise ValueError('price role or attribution mismatch')
    summary = pl.read_csv(REPO / PRIMARY_PRICES / 'summary.csv').filter(pl.col('ranking') == 'market')
    strata = pl.read_csv(REPO / PRIMARY_PRICES / 'role_identity_summary.csv').filter(pl.col('ranking') == 'market')
    assert_same_records(summarize_prices(prices), summary, ['instrument', 'stage'])
    assert_same_records(summarize_prices(prices, roles=True), strata, ['instrument', 'stage', 'execution_anchor_mode', 'identity_level'])
    sensitivity = pl.read_csv(REPO / PRIMARY_PRICES / 'sign_sensitivity.csv').filter((pl.col('ranking') == 'market') & pl.col('ulp_factor').is_in([4, 8, 16, 32]))
    for factor in [4, 8, 16, 32]:
        assert_same_records(summarize_prices(prices, factor=factor), sensitivity.filter(pl.col('ulp_factor') == factor).drop('ulp_factor'), ['instrument', 'stage'])
    # Check source sign flags themselves against the exact saved roundoff bounds.
    for metric in ['FPM', 'REV', 'ADV']:
        for value, tolerance, sign in prices.select(metric, metric + '_zero_tolerance', metric + '_sign').iter_rows():
            expected = None if value is None or not math.isfinite(value) else (0 if abs(value) <= tolerance else (1 if value > 0 else -1))
            if sign != expected:
                raise ValueError('saved price sign differs from numeric policy')
    case_frames = []
    case_provenance = load_json(REPO / DIAGNOSTICS / 'case_provenance.json')
    allowed = ['instrument', 'execution_anchor_mode', 'execution_quantity', 'horizon_ns', 'age_policy',
        'candidate_qty_raw', 'withdrawal_qty_raw', 'candidate_qty_age_weighted', 'withdrawal_qty_age_weighted',
        'opposite_market_visible_qty_pre_topN', 'WMSCI_market', 'J', 'is_candidate', *FLAGS]
    for alias, filename in [('R', 'RISANAMENTO_17295_multiscale.csv'), ('N', 'NEXI_31425_multiscale.csv')]:
        path = REPO / DIAGNOSTICS / filename
        record = next(r for r in case_provenance['cases'] if Path(r['output']).name == filename)
        if digest(path) != record['output_sha256'] or digest(REPO / record['source']) != record['source_sha256']:
            raise ValueError('diagnostic provenance hash mismatch')
        original = pl.read_csv(path, infer_schema_length=None)
        if original.height != 20 or original.select(KEYS).unique().height != 1:
            raise ValueError('case is not one native execution over 20 cells')
        matching = populations[original['instrument'][0]].join(original.select(KEYS).unique(), on=KEYS)
        assert_same_records(matching.select(allowed), original.select(allowed), ['horizon_ns', 'age_policy'])
        public = original.select(allowed).with_columns(pl.lit(alias).alias('case_id'), (pl.col('horizon_ns') / 1e9).alias('horizon_seconds'))
        assert_same_records(public, previous_cases.filter(pl.col('case_id') == alias).select(public.columns), ['horizon_ns', 'age_policy'])
        case_frames.append(public)
    cases = pl.concat(case_frames).sort(['case_id', 'age_policy', 'horizon_ns'])
    primary_counts = primary(counts).sort('instrument')
    if primary_counts['candidate_executions'].to_list() != [3526, 2393, 995] or primary_counts['market_w_count'].to_list() != [353, 240, 98] or primary_counts['market_j_count'].to_list() != [36, 24, 10]:
        raise ValueError('primary publication contract changed')
    if summary.height != 6 or sensitivity.height != 24:
        raise ValueError('price group coverage mismatch')
    for row in summary.iter_rows(named=True):
        group = strata.filter((pl.col('instrument') == row['instrument']) & (pl.col('stage') == row['stage']))
        for field in ['n', 'both_positive_complete', *[c for c in summary.columns if c.endswith(('_positive', '_zero', '_negative', '_missing', '_missing_or_incomplete', '_defined'))]]:
            if int(group[field].sum()) != row[field]:
                raise ValueError('role/identity strata do not partition price counts')
    out.mkdir(parents=True, exist_ok=False)
    for filename, frame in [('cell_counts.csv', counts), ('market_priorities.csv', primary_counts),
        ('market_horizon_priorities.csv', counts.filter(pl.col('age_policy') == PRIMARY_POLICY)),
        ('instrument_totals.csv', totals), ('composition.csv', pl.DataFrame(composition)), ('cases.csv', cases),
        ('primary_price_summary.csv', summary), ('primary_price_strata.csv', strata), ('primary_price_sign_sensitivity.csv', sensitivity)]:
        frame.write_csv(out / filename)
    # Public per-execution ledger has no participant, order, cluster or partition identity.
    prices.drop(KEYS[1:]).write_csv(out / 'primary_market_price_records.csv')
    table(out, 'population.tex', 'tab:wmsci-j-population',
        'Execution population, constituent child fills, and raw candidate executions in the primary specification (30-second age half-life, $H=10$ seconds). A raw candidate requires positive unweighted displayed quantity, attributed withdrawal and execution quantity, independently of weighted-score positivity. Counts are distinct execution keys, not repeated design cells.',
        ['Instrument', 'Executions', 'Fills', 'Candidates ($H=10$ s)'],
        [[SHORT[i], totals.filter(pl.col('instrument') == i)['execution_count'][0], totals.filter(pl.col('instrument') == i)['fill_member_count'][0], primary_counts.filter(pl.col('instrument') == i)['candidate_executions'][0]] for i in SHORT])
    priority_caption = ('Market-normalized withdrawal and conditional profile priorities under the 30-second exponential age half-life. Positive support is the number of raw candidates with finite positive WMSCI; W is its top decile with upward-rounded budget $\\lceil n/10\\rceil$. Finite $J$ counts complete finite profiles within W, including $J=0$; W+J is its conditional top decile, again upward-rounded. These are nested review lists, not confirmed manipulation or price-selected observations.')
    headers = ['Instrument', 'Positive support', 'W', r'Finite $J$ in W', 'W+J']
    to_rows = lambda frame: [[SHORT[r['instrument']], r['positive_market_support'], r['market_w_count'], r['conditional_finite_J_support'], r['market_j_count']] for r in frame.iter_rows(named=True)]
    ordered_primary = pl.concat([primary_counts.filter(pl.col('instrument') == i) for i in SHORT])
    table(out, 'market_priorities.tex', 'tab:market-priorities', 'Primary review priorities at $H=10$ seconds. ' + priority_caption, headers, to_rows(ordered_primary))
    horizon_priorities = pl.concat([counts.filter((pl.col('instrument') == i) & (pl.col('age_policy') == PRIMARY_POLICY)).sort('horizon_ns') for i in SHORT])
    table(out, 'market_horizon_priorities.tex', 'tab:market-horizon-priorities', priority_caption + ' The ten-second horizon is primary; shorter horizons are sensitivity checks.',
        ['Instrument', '$H$ (s)', *headers[1:]],
        [[SHORT[r['instrument']], int(r['horizon_seconds']), r['positive_market_support'], r['market_w_count'], r['conditional_finite_J_support'], r['market_j_count']] for r in horizon_priorities.iter_rows(named=True)], floating=False)
    def count(i, h, policy, field):
        return counts.filter((pl.col('instrument') == i) & (pl.col('horizon_ns') == h) & (pl.col('age_policy') == policy))[field][0]
    table(out, 'horizon_counts.tex', 'tab:multiscale-horizon-counts',
        'Raw candidate executions at each withdrawal horizon: 30-second age cap / all ages without weighting. The execution population is fixed and neither entry is a percentile selection. Horizons are evaluated separately, so exclusive cancellation attribution need not produce monotone counts.',
        ['Instrument', '$H=1$ s', '$H=2$ s', '$H=5$ s', '$H=10$ s'],
        [[SHORT[i]] + [f"{count(i,h,'legacy_age30','candidate_executions')} / {count(i,h,'all_ages_unweighted','candidate_executions')}" for h in HORIZONS] for i in SHORT], floating=False)
    age_names = {'legacy_age30': 'Cap 30 s', 'all_ages_unweighted': 'All ages', 'exp_h30': 'Half-life 30 s', 'exp_h120': 'Half-life 120 s', 'exp_h300': 'Half-life 300 s'}
    table(out, 'age_policy_summary.tex', 'tab:multiscale-age-policies',
        'Age-policy sensitivity at one- and ten-second withdrawal horizons. All ages uses no weighting or age cap; cap retains ages at most 30 seconds; half-life policies exponentially weight all observable ages. Raw counts use unweighted quantities. Market W counts are stored withdrawal priorities with cell-specific positive finite-score support and upward-rounded decile budgets. Intermediate horizons and conditional W+J counts are in the companion CSV.',
        ['Instrument', 'Age policy', 'Raw 1 s', 'Raw 10 s', 'Market W 1 s', 'Market W 10 s'],
        [[SHORT[i], age_names[p], *[count(i, h, p, f) for f in ['candidate_executions', 'market_w_count'] for h in [HORIZONS[0], HORIZONS[-1]]]] for i in SHORT for p in POLICIES], floating=False)
    def comp(i, cohort, role, identity):
        return sum(r['executions'] for r in composition if (r['instrument'],r['cohort'],r['role'],r['identity_level']) == (i,cohort,role,identity))
    table(out, 'composition.tex', 'tab:multiscale-composition',
        'Execution composition by source-derived economic role and attribution: all executions / primary raw candidates (30-second age half-life, $H=10$ seconds). Client means original-client attribution and firm means fallback attribution, potentially pooling clients. Strata partition each instrument population; coverage differs across instruments.',
        ['Instrument', 'Aggressive/client', 'Aggressive/firm', 'Passive/client', 'Passive/firm'],
        [[SHORT[i]] + [f"{comp(i,'All executions',r,d)} / {comp(i,'Primary candidates',r,d)}" for r,d in [('aggressive','client_original'),('aggressive','firm'),('passive','client_original'),('passive','firm')]] for i in SHORT])
    transparent = cases.filter(pl.col('age_policy') == PRIMARY_POLICY)
    table(out, 'anonymouscase_table.tex', 'tab:multiscale-anonymous-cases',
        'Two post-hoc diagnostic examples R (Risanamento) and N (Nexi), evaluated at four discrete withdrawal horizons under the 30-second age half-life. $P$ and $U$ are raw displayed and attributed-withdrawal shares, distinct from weighted score inputs. $W$ is market-normalized WMSCI and $J$ the complete unweighted profile statistic. These are repeated evaluations of two executions, not eight independent cases or trajectories.',
        ['Case', '$H$ (s)', '$P$ (shares)', '$U$ (shares)', '$W$', '$J$'],
        [[r['case_id'], int(r['horizon_seconds']), int(r['candidate_qty_raw']), int(r['withdrawal_qty_raw']), r['WMSCI_market'], r['J']] for r in transparent.iter_rows(named=True)])
    render_prices(out, prices, summary, strata)
    styles = load_json(REPO / PREVIOUS / 'style_metadata.json')
    reused = []
    figures = []
    for filename, companion in [('candidate_counts.pdf', 'cell_counts.csv'), ('withdrawal_and_profile_summary.pdf', 'cases.csv')]:
        source = REPO / PREVIOUS / filename
        shutil.copyfile(source, out / filename)
        if digest(out / filename) != digest(source):
            raise ValueError('copied figure hash mismatch')
        reused.append({'asset': filename, 'source_sha256': digest(source), 'companion': companion,
            'companion_validation': 'native values equal authenticated prior companion values; only market columns exported'})
        figures.append(next(f for f in styles['figures'] if f['path'] == filename))
    missingness = [{k: r[k] for k in ['instrument', 'stage', 'n', 'FPM_missing', 'REV_missing_or_incomplete', 'ADV_missing']} for r in summary.iter_rows(named=True)]
    bindings = {str(Path(__file__).relative_to(REPO)): digest(__file__),
        'scripts/generate_withdrawal_multiscale_manuscript.py': digest(Path(__file__).with_name('generate_withdrawal_multiscale_manuscript.py')),
        str(PRIMARY_PRICES / 'artifact_manifest.json'): digest(REPO / PRIMARY_PRICES / 'artifact_manifest.json'),
        str(PRIMARY_PRICES / 'COMPLETE.json'): digest(REPO / PRIMARY_PRICES / 'COMPLETE.json'),
        str(PRIMARY_PRICES / 'metadata.json'): digest(REPO / PRIMARY_PRICES / 'metadata.json'),
        str(PREVIOUS / 'artifact_manifest.json'): digest(REPO / PREVIOUS / 'artifact_manifest.json'),
        str(SUMMARY / 'artifact_manifest.json'): digest(REPO / SUMMARY / 'artifact_manifest.json'),
        str(DIAGNOSTICS / 'case_provenance.json'): digest(REPO / DIAGNOSTICS / 'case_provenance.json'),
        'configs/withdrawal_multiscale_v1.json': digest(REPO / 'configs/withdrawal_multiscale_v1.json')}
    facts = {'version': 'withdrawal_market_publication_v1', 'scope': 'market_only',
        'primary_policy': PRIMARY_POLICY, 'primary_horizon_ns': PRIMARY_HORIZON_NS,
        'primary_choice': 'market normalization chosen for simpler exposition, not empirically demonstrated superiority; half-life and horizon not optimized',
        'primary_candidate_executions': int(primary_counts['candidate_executions'].sum()),
        'primary_price_union_executions': prices.height, 'primary_population': primary_counts.to_dicts(),
        'selection_counts': counts.to_dicts(), 'primary_prices': summary.to_dicts(),
        'price_missingness_by_instrument_and_stage': missingness,
        'conditional_J': 'complete finite J within stored W, including zero; ceiling decile budget; no price filtering',
        'market_formula': 'W = log(1+P_g/Q)*log(1+U_g/Q)*(U_g/D); D>0; null if unsupported',
        'candidate_rule': 'P_raw>0 and U_raw>0 and Q>0, independent of weighted-score positivity',
        'units': {'counts':'distinct native execution keys in each declared cohort; 20 repeated cells per execution',
            'P_U_Q_D':'shares', 'P_g_U_g':'age-weighted shares', 'W_J':'dimensionless',
            'FPM_REV_ADV':'instrument price units per share; not profit', 'horizons':'integer nanoseconds in source; seconds in tables'},
        'reversion_horizon_ns': 2_000_000_000,
        'price_sign_policy': load_json(REPO / PRIMARY_PRICES / 'metadata.json')['sign_policy'],
        'price_membership_verification': 'exact full native key-set equality and non-null Boolean W/J flag equality; roles and attribution equal native',
        'price_recomputation': 'six summary rows, all role/identity strata, 4/8/16/32 sensitivity recounted from unchanged raw values and saved endpoint roundoff bounds; equal authenticated sources',
        'cases':'public aliases R/N; twenty native cells each, original identities omitted; source hashes verified',
        'source_evidence': evidence, 'input_renderer_bindings': bindings, 'reused_assets': reused,
        'figures': figures, 'environment': {'python': platform.python_version(), 'polars': pl.__version__},
        'limitations':['descriptive triage, not manipulation labels or intent', 'no uniform price enrichment claim', 'instrument coverage differs', 'no replay, fitting, reranking or new price measurement'],
        'totals': {c:int(totals[c].sum()) for c in totals.columns if c != 'instrument'}}
    write_json(out / 'facts.json', facts)
    write_json(out / 'style_metadata.json', {**{k:v for k,v in styles.items() if k != 'figures'}, 'figures': figures, 'asset_policy':'two authenticated PDFs reused unchanged; native companions checked'})
    artifacts = {p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()}
    write_json(out / 'artifact_manifest.json', {'version':facts['version'], 'scope':'market_only', 'artifacts':artifacts, 'input_renderer_bindings':bindings})
    write_json(out / 'COMPLETE.json', {'status':'publication_package_verified', 'artifact_manifest_sha256':digest(out / 'artifact_manifest.json'), 'scope':'market_only', 'primary_candidate_executions':6914, 'primary_price_union_executions':prices.height})
    validate_stage(out, 'publication_package_verified')
    return facts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    args = parser.parse_args()
    out = args.output if args.output.is_absolute() else REPO / args.output
    if out.resolve().parent != (REPO / 'paper/generated').resolve() or not out.name.startswith('withdrawal_market_'):
        parser.error('output must be a fresh withdrawal_market_* package under paper/generated')
    facts = build(out)
    print(json.dumps({'output':str(out), 'scope':facts['scope'], 'primary_candidates':facts['primary_candidate_executions'], 'market_price_union':facts['primary_price_union_executions'], 'primary_population':facts['primary_population']}, indent=2))


if __name__ == '__main__':
    main()
