#!/usr/bin/env python3
"""Fresh, authenticated all-candidate market-W ranking publication consumer.

No replay, price gating, percentile selection, or J ranking. Production builds
require a completed all-candidate price stage. Historical packages are read only.
"""
from __future__ import annotations
import argparse
import importlib.util
import json
import math
import shutil
from pathlib import Path
import polars as pl

REPO = Path(__file__).resolve().parents[1]
OUTPUT = Path('paper/generated/withdrawal_rank_20261015_v2')
PRICES = Path('outputs/ranked_candidate_prices_20261015_101002_v2')
APPROVED = Path('paper/generated/withdrawal_market_20261015_v1')
REVIEW = Path('paper/reviews/20261015_101002_rank_only')
KEYS = ['instrument', 'partition_id', 'actor_key', 'execution_cluster_id']
CELL = ['instrument', 'age_policy', 'horizon_ns']
PRIMARY_POLICY = 'exp_h30'
PRIMARY_HORIZON_NS = 10_000_000_000
EXPECTED = {'FERRARI': 3526, 'NEXI': 2393, 'RISANAMENTO': 995}
ROLES = ['execution_anchor_mode', 'identity_level']
QUANTITIES = ['candidate_qty_raw', 'withdrawal_qty_raw', 'execution_quantity',
              'candidate_qty_age_weighted', 'withdrawal_qty_age_weighted',
              'opposite_market_visible_qty_pre_topN']
NATIVE = QUANTITIES + ROLES + ['WMSCI_market', 'J', 'complete_grid']
METRICS = ['FPM', 'REV', 'ADV']
_spec = importlib.util.spec_from_file_location('rank_only_frozen_primitives',
    Path(__file__).with_name('generate_withdrawal_multiscale_manuscript.py'))
_old = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_old)
validate_root, validate_stage = _old.validate_root, _old.validate_stage
digest, load_json, write_json, table = _old.digest, _old.load_json, _old.write_json, _old.table
DEFAULT_ROOTS, SUMMARY, DIAGNOSTICS = _old.DEFAULT_ROOTS, _old.SUMMARY, _old.DIAGNOSTICS


def finite(value):
    return value is not None and math.isfinite(value)


def primary(frame):
    return frame.filter((pl.col('age_policy') == PRIMARY_POLICY) &
                        (pl.col('horizon_ns') == PRIMARY_HORIZON_NS))


def validate_keys(frame, *, cells=False):
    fields = KEYS + (['age_policy', 'horizon_ns'] if cells else [])
    if set(fields) - set(frame.columns) or any(frame.select(fields).null_count().row(0)) or frame.select(fields).is_duplicated().any():
        raise ValueError('null, duplicate or missing full key')


def validate_population(frame):
    validate_keys(frame, cells=True)
    for field in ['is_candidate', 'complete_grid']:
        if frame.schema[field] != pl.Boolean or frame[field].null_count():
            raise ValueError('non-null Boolean required: ' + field)
    for field in ROLES:
        if frame[field].null_count():
            raise ValueError('missing role or identity')
    if set(frame['execution_anchor_mode'].unique()) - {'passive', 'aggressive'} or set(frame['identity_level'].unique()) - {'client_original', 'firm'}:
        raise ValueError('unsupported source role or attribution')
    raw = pl.all_horizontal([pl.col(c).is_finite() & (pl.col(c) > 0) for c in QUANTITIES[:3]]).fill_null(False)
    if frame.filter(pl.col('is_candidate').ne_missing(raw)).height:
        raise ValueError('native raw candidate definition mismatch')


def rank_candidates(frame):
    """Retain every raw candidate; finite W descending, full-key stable ties.

    Undefined W is unsupported with null rank, never imputed zero. J is untouched
    and never enters membership or ordering. Ranking is independent in each cell.
    """
    validate_population(frame)
    result = []
    candidates = frame.filter(pl.col('is_candidate'))
    for _, cell in candidates.partition_by(CELL, as_dict=True).items():
        records = cell.to_dicts()
        records.sort(key=lambda r: (not finite(r['WMSCI_market']),
            -r['WMSCI_market'] if finite(r['WMSCI_market']) else 0,
            r['execution_cluster_id'], r['partition_id'], r['actor_key']))
        for index, row in enumerate(records, 1):
            supported = finite(row['WMSCI_market'])
            row.update(W_rank=index if supported else None,
                       W_status='finite' if supported else 'unsupported',
                       J_status='complete_finite' if row['complete_grid'] and finite(row['J']) else 'missing_or_incomplete')
        result.extend(records)
    if not result:
        return candidates.with_columns(pl.lit(None, dtype=pl.Int64).alias('W_rank'),
            pl.lit(None, dtype=pl.String).alias('W_status'), pl.lit(None, dtype=pl.String).alias('J_status'))
    return pl.DataFrame(result, schema_overrides={'W_rank': pl.Int64}).sort(CELL + ['W_rank', 'execution_cluster_id', 'partition_id', 'actor_key'], nulls_last=True)


def equal_value(x, y):
    if x is None or y is None:
        return x is y
    if isinstance(x, float) and isinstance(y, float) and math.isnan(x) and math.isnan(y):
        return True
    return x == y


def verify_values(left, right, fields, *, cells=False):
    validate_keys(left, cells=cells)
    validate_keys(right, cells=cells)
    keys = KEYS + (['age_policy', 'horizon_ns'] if cells else [])
    a = {tuple(r[k] for k in keys): r for r in left.select(keys + fields).to_dicts()}
    b = {tuple(r[k] for k in keys): r for r in right.select(keys + fields).to_dicts()}
    if a.keys() != b.keys():
        raise ValueError('full key membership differs')
    for key in a:
        for field in fields:
            if not equal_value(a[key][field], b[key][field]):
                raise ValueError('native value mismatch: ' + field)


def verify_price_membership(prices, native):
    verify_values(prices, native, NATIVE + ['is_candidate'])
    if prices.height != primary(prices).height:
        raise ValueError('price primary policy/horizon mismatch')


def cell_summary(frame):
    validate_population(frame)
    rows = []
    for group, cell in frame.partition_by(CELL, as_dict=True).items():
        c = cell.filter(pl.col('is_candidate'))
        w = [v for v in c['WMSCI_market'] if finite(v)]
        j = [r['J'] for r in c.select('J', 'complete_grid').to_dicts() if r['complete_grid'] and finite(r['J'])]
        row = dict(zip(CELL, group))
        row.update(execution_count=cell.height, candidate_executions=c.height,
            finite_W=len(w), positive_W=sum(v > 0 for v in w), zero_W=w.count(0.),
            negative_W=sum(v < 0 for v in w), unsupported_W=c.height-len(w),
            complete_finite_J=len(j), J_missing_or_incomplete=c.height-len(j), zero_J=j.count(0.))
        for label, values in [('W', w), ('J', j)]:
            s = pl.Series(values, dtype=pl.Float64)
            for name, value in [('min', s.min()), ('q25', s.quantile(.25, interpolation='linear')),
                                ('median', s.median()), ('q75', s.quantile(.75, interpolation='linear')), ('max', s.max())]:
                row[label + '_' + name] = value
        rows.append(row)
    return pl.DataFrame(rows).sort(CELL)


def price_sign(value, tolerance, factor=8):
    if not finite(value):
        return None
    if not finite(tolerance) or tolerance < 0:
        raise ValueError('invalid price zero tolerance')
    return 0 if abs(value) <= tolerance * factor / 8 else (1 if value > 0 else -1)


def validate_prices(frame):
    validate_keys(frame)
    for r in frame.to_dicts():
        if type(r['REV_complete']) is not bool:
            raise ValueError('REV_complete must be non-null Boolean')
        complete = (r['REV_cancel_count'] > 0 and r['REV_missing_cancel_count'] == 0)
        if r['REV_complete'] != complete or finite(r['REV']) != complete:
            raise ValueError('REV complete two-second outcome mismatch')
        for metric in METRICS:
            if r[metric+'_sign'] != price_sign(r[metric], r[metric+'_zero_tolerance']):
                raise ValueError('saved price sign mismatch: ' + metric)


def summarize_prices(frame, *, roles=False, factor=8):
    fields = ['instrument'] + (ROLES if roles else [])
    rows = []
    for group, sub in frame.partition_by(fields, as_dict=True).items():
        row = dict(zip(fields, group)); row['n'] = sub.height
        signs = {}
        for metric in METRICS:
            values = [price_sign(v, t, factor) for v, t in sub.select(metric, metric+'_zero_tolerance').iter_rows()]
            signs[metric] = values
            for label, sign in [('positive', 1), ('zero', 0), ('negative', -1)]:
                row[metric+'_'+label] = values.count(sign)
            row[metric+('_missing_or_incomplete' if metric == 'REV' else '_missing')] = values.count(None)
            row[metric+'_defined'] = sub.height-values.count(None)
            raw = [v for v in sub[metric] if finite(v)]
            for label, sign in [('positive', 1), ('zero', 0), ('negative', -1)]:
                row[metric+'_raw_'+label] = sum((1 if v > 0 else -1 if v < 0 else 0) == sign for v in raw)
            normalized = [0. if sign == 0 else v for v, sign in zip(sub[metric], values) if sign is not None]
            row[metric+'_raw_median'] = pl.Series(raw, dtype=pl.Float64).median()
            row[metric+'_median'] = pl.Series(normalized, dtype=pl.Float64).median()
        row['both_positive_complete'] = sum(a == 1 and b == 1 for a, b in zip(signs['FPM'], signs['REV']))
        rows.append(row)
    return pl.DataFrame(rows).sort(fields)


def verify_strata(summary, strata):
    for row in summary.to_dicts():
        group = strata.filter(pl.col('instrument') == row['instrument'])
        for name in summary.columns:
            if name in ['n', 'both_positive_complete'] or name.endswith(('_positive', '_negative', '_zero', '_missing', '_missing_or_incomplete', '_defined')):
                if group[name].sum() != row[name]:
                    raise ValueError('price strata do not partition counts: ' + name)


def reconcile_summary(actual, saved, keys):
    """Match every producer summary field; extra recomputed raw medians allowed."""
    if {'stage', 'ranking'} & set(saved.columns) or set(saved.columns)-set(actual.columns):
        raise ValueError('unsupported producer summary schema: ' + str(saved.columns))
    required = {'n', 'both_positive_complete', *[m+'_'+s for m in METRICS for s in ['positive','zero','negative']],
                'FPM_missing','REV_missing_or_incomplete','ADV_missing'}
    if required-set(saved.columns):
        raise ValueError('producer summary missing sign partitions')
    a = actual.select(saved.columns).sort(keys); b = saved.sort(keys)
    if a.height != b.height or a.select(keys).is_duplicated().any() or b.select(keys).is_duplicated().any():
        raise ValueError('summary group membership mismatch')
    for x, y in zip(a.iter_rows(named=True), b.iter_rows(named=True)):
        for field in x:
            if not equal_value(x[field], y[field]) and not (finite_number(x[field]) and finite_number(y[field]) and math.isclose(x[field], y[field], rel_tol=1e-12, abs_tol=1e-15)):
                raise ValueError('producer summary mismatch: ' + field)


def finite_number(v):
    return isinstance(v, (int, float)) and finite(v)


def public_ranking(ranked, prices):
    """Explicit allowlist; aliases represent full keys, not just cluster IDs."""
    validate_keys(ranked); validate_keys(prices)
    ordered = ranked.sort(['instrument', 'W_rank', 'execution_cluster_id', 'partition_id', 'actor_key'], nulls_last=True)
    ordered = ordered.with_columns(pl.Series('execution_alias', [f'E{n:05d}' for n in range(1, ordered.height+1)]))
    price_fields = [m+s for m in METRICS for s in ['', '_sign', '_zero_tolerance']] + ['REV_complete','REV_cancel_count','REV_missing_cancel_count']
    result = ordered.join(prices.select(KEYS+price_fields), on=KEYS, how='left', validate='1:1', maintain_order='left')
    return result.select(['instrument','execution_alias',*ROLES,'WMSCI_market','W_rank','W_status','J','J_status','complete_grid',*QUANTITIES,*price_fields])


def render_tables(out, totals, counts, composition, cases, summary, strata):
    instruments = sorted(totals['instrument'].unique())
    def title(i): return i.title()
    def formatted(value):
        if isinstance(value, float):
            return '--' if not finite(value) else (int(value) if value.is_integer() else r'\num{' + format(value, '.4g') + '}')
        return value
    def tab(name, label, caption, headers, rows, floating=True):
        table(out,name,label,caption,headers,[[formatted(v) for v in row] for row in rows],floating=floating)
    def count(i, p, h, field='candidate_executions'):
        sub=counts.filter((pl.col('instrument')==i)&(pl.col('age_policy')==p)&(pl.col('horizon_ns')==h))
        return sub[field][0] if sub.height else None
    tab('population.tex','tab:wmsci-j-population',
        'Execution population, constituent child fills and all raw candidates at the primary 30-second age half-life and ten-second withdrawal horizon. Raw candidates require positive physical displayed, withdrawal and execution quantities; no score or price selection.',
        ['Instrument','Executions','Fills','Candidates'],
        [[title(r['instrument']),r['execution_count'],r['fill_member_count'],count(r['instrument'],PRIMARY_POLICY,PRIMARY_HORIZON_NS)] for r in totals.to_dicts()])
    rank_rows=[]
    for r in primary(counts).to_dicts():
        for metric, valid, missing, zero in [('W','finite_W','unsupported_W','zero_W'),('J','complete_finite_J','J_missing_or_incomplete','zero_J')]:
            interval=f"{formatted(r[metric+'_q25'])}/{formatted(r[metric+'_q75'])}"
            rank_rows.append([title(r['instrument']),f'${metric}$',r['candidate_executions'],f"{r[valid]}/{r[missing]}",r[zero],r[metric+'_median'],interval])
    tab('ranking_summary.tex','tab:rank-summary',
        'All primary raw candidates, ordered within instrument by descending market-normalized WMSCI. Zero scores remain ranked; undefined scores remain with unsupported status and no rank. W and complete finite J distributions are descriptive (median and interquartile interval); J never selects or reorders executions. Missing J includes incomplete profiles. No cross-instrument ranking is implied.',
        ['Instrument','Metric','$N$',r'\shortstack{Finite/\\missing}','Zero','Median','Q1/Q3'],rank_rows)
    tab('horizon_counts.tex','tab:multiscale-horizon-counts',
        'Raw candidate executions at each withdrawal horizon: 30-second age cap / all ages unweighted. Horizons are evaluated separately and counts need not be monotone; neither entry is a score selection.',
        ['Instrument',*[f'$H={h}$ s' for h in [1,2,5,10]]],
        [[title(i),*[f"{count(i,'legacy_age30',h*10**9)} / {count(i,'all_ages_unweighted',h*10**9)}" for h in [1,2,5,10]]] for i in instruments],False)
    tab('age_policy_summary.tex','tab:multiscale-age-policies',
        'Age-policy sensitivity: raw candidates / finite market-score support at one- and ten-second withdrawal horizons. Exponential policies weight all observable ages; raw eligibility is independent of weighted positivity. All sixty instrument-policy-horizon cells are retained in the companion CSV.',
        ['Instrument','Age policy','$H=1$ s','$H=10$ s'],
        [[title(i),p.replace('_',' '),*[f"{count(i,p,h)} / {count(i,p,h,'finite_W')}" for h in [10**9,10*10**9]]] for i in instruments for p in sorted(counts['age_policy'].unique())],False)
    tab('composition.tex','tab:multiscale-composition',
        'All executions / primary raw candidates by source-derived role and attribution. Original-client attribution differs from firm fallback, which may pool clients. Counts refer to executions, not independent participants.',
        ['Instrument','Passive/client','Aggressive/client','Passive/firm','Aggressive/firm'],
        [[title(i),*[ '/'.join(str(composition.filter((pl.col('instrument')==i)&(pl.col('cohort')==c)&(pl.col('execution_anchor_mode')==r)&(pl.col('identity_level')==d))['executions'].sum()) for c in ['All executions','Primary candidates']) for r,d in [('passive','client_original'),('aggressive','client_original'),('passive','firm'),('aggressive','firm')]]] for i in instruments])
    tab('anonymouscase_table.tex','tab:multiscale-anonymous-cases',
        'Post-hoc diagnostic examples R and N at four withdrawal horizons, with the 30-second age half-life. P and U are physical shares; W is market-normalized WMSCI and J the unweighted profile statistic. These are repeated evaluations of two executions, not eight independent observations.',
        ['Case','$H$ (s)','$P$ (shares)','$U$ (shares)','$W$','$J$'],
        [[r['case_id'],r['horizon_ns']//10**9,r['candidate_qty_raw'],r['withdrawal_qty_raw'],r['WMSCI_market'],r['J']] for r in cases.filter(pl.col('age_policy')==PRIMARY_POLICY).to_dicts()])
    def signs(r,m): return '/'.join(str(r[m+'_'+s]) for s in ['positive','zero','negative','missing_or_incomplete' if m=='REV' else 'missing'])
    tab('candidate_prices.tex','tab:candidate-prices',
        'Price diagnostics for every primary raw candidate, without price filtering. Entries are positive/zero/negative/missing with the entire candidate population N as denominator. FPM is favorable pre-execution midprice movement; REV is physical-quantity-weighted reversion over two seconds after attributed cancellations, missing unless every positive-quantity cancellation is observed. Both counts positive FPM and complete positive REV. Numerical zero uses the saved endpoint-scaled eight-ULP convention, not an economic threshold.',
        ['Instrument','$N$','FPM (+/0/--/?)','REV (+/0/--/?)','Both $>0$'],
        [[title(r['instrument']),r['n'],signs(r,'FPM'),signs(r,'REV'),r['both_positive_complete']] for r in summary.to_dicts()])
    tab('candidate_advantage.tex','tab:candidate-advantage',
        'Execution-price advantage ADV against the completed-display midprice for all primary raw candidates. Positive/zero/negative/missing counts use full N; the median uses finite observations normalized only within the saved roundoff bound. Raw diagnostics remain unchanged in the ranking CSV. Units are price per share; positive advantage is not realized profit and never determines inclusion.',
        ['Instrument','$N$','ADV (+/0/--/?)','Median ADV'],
        [[title(r['instrument']),r['n'],signs(r,'ADV'),r['ADV_median']] for r in summary.to_dicts()])
    tab('candidate_price_strata.tex','tab:candidate-price-strata',
        'All-primary-candidate price evidence by source-derived execution role and attribution. Entries show stratum executions / jointly positive FPM and complete two-second REV; strata partition each instrument, with no score or price exclusion. Firm fallback can pool clients.',
        ['Instrument','Role','Attribution','$N$','Both $>0$'],
        [[title(r['instrument']),r['execution_anchor_mode'],r['identity_level'].replace('_',' '),r['n'],r['both_positive_complete']] for r in strata.to_dicts()],False)


def authenticate_reuse(populations, counts):
    """Authenticate the two approved unchanged PDFs and their native companions."""
    validate_stage(REPO/APPROVED, 'publication_package_verified')
    prior_counts=pl.read_csv(REPO/APPROVED/'cell_counts.csv')
    for row in counts.to_dicts():
        saved=prior_counts.filter(pl.all_horizontal([pl.col(k)==row[k] for k in CELL]))
        if saved.height != 1 or saved['candidate_executions'][0] != row['candidate_executions']:
            raise ValueError('candidate figure companion mismatch')
    provenance=load_json(REPO/DIAGNOSTICS/'case_provenance.json')
    prior_cases=pl.read_csv(REPO/APPROVED/'cases.csv')
    cases=[]
    # Historical companions contain physical quantities and scores, not the
    # complete-grid/identity fields present only in the native ledger.
    fields=QUANTITIES+['execution_anchor_mode','WMSCI_market','J','is_candidate']
    for alias, filename in [('R','RISANAMENTO_17295_multiscale.csv'),('N','NEXI_31425_multiscale.csv')]:
        record=next(r for r in provenance['cases'] if Path(r['output']).name==filename)
        path=REPO/DIAGNOSTICS/filename
        if digest(path)!=record['output_sha256'] or digest(REPO/record['source'])!=record['source_sha256']:
            raise ValueError('diagnostic source provenance mismatch')
        original=pl.read_csv(path,infer_schema_length=None)
        if original.height!=20 or original.select(KEYS).unique().height!=1:
            raise ValueError('case full-key coverage mismatch')
        native=populations.join(original.select(KEYS).unique(),on=KEYS)
        verify_values(original,native,fields,cells=True)
        previous=prior_cases.filter(pl.col('case_id')==alias)
        columns=['instrument','age_policy','horizon_ns',*fields]
        for a,b in zip(original.select(columns).sort(CELL).iter_rows(),previous.select(columns).sort(CELL).iter_rows()):
            if not all(equal_value(x,y) for x,y in zip(a,b)):
                raise ValueError('profile figure companion mismatch')
        if previous.height!=20:
            raise ValueError('profile companion coverage mismatch')
        cases.append(original.select(columns).with_columns(pl.lit(alias).alias('case_id')))
    return pl.concat(cases).sort(['case_id','age_policy','horizon_ns'])


def build(out, *, prices_root=PRICES):
    out=Path(out)
    # This guard deliberately precedes *all* source access.
    if out.exists():
        raise FileExistsError(f'publication output exists: {out}; use a fresh directory')
    prices_root=Path(prices_root)
    if not prices_root.is_absolute(): prices_root=REPO/prices_root
    if out.resolve().parent != (REPO/'paper/generated').resolve() or not out.name.startswith('withdrawal_rank_'):
        raise ValueError('output must be a fresh withdrawal_rank_* under paper/generated')
    baseline=load_json(REPO/REVIEW/'baseline_identity.json')
    for relative, expected in baseline['protected'].items():
        if digest(REPO/relative)!=expected:
            raise ValueError('frozen protected input changed: '+relative)
    evidence=[]
    for relative in DEFAULT_ROOTS:
        auth=validate_root(REPO/relative)
        evidence.append({k:v for k,v in auth.items() if k!='report_provenance'})
    validate_stage(REPO/SUMMARY,'descriptive_campaign_summary_verified')
    price_manifest=validate_stage(prices_root,'ranked_candidate_prices_verified')
    required={'private_execution_prices.parquet','private_cancel_prices.parquet','summary.csv','role_identity_summary.csv','sign_sensitivity.csv','metadata.json'}
    if required-set(price_manifest['artifacts']):
        raise ValueError('price stage missing bound artifacts')
    totals=pl.read_csv(REPO/SUMMARY/'instrument_totals.csv').drop('root')
    reference=pl.read_csv(REPO/SUMMARY/'cell_counts.csv')
    frames=[]; comp=[]
    columns=KEYS+['age_policy','horizon_ns','is_candidate','child_fill_count','execution_vwap',*NATIVE]
    for root in DEFAULT_ROOTS:
        p=pl.read_parquet(REPO/root/'analysis/population.parquet',columns=columns)
        validate_population(p)
        units=p.unique(KEYS)
        instrument=p['instrument'][0]
        if p.height!=20*units.height or set(p['age_policy'].unique())!=set(_old.POLICIES) or set(p['horizon_ns'].unique())!=set(_old.HORIZONS):
            raise ValueError('native twenty-cell grid mismatch')
        expected=totals.filter(pl.col('instrument')==instrument).row(0,named=True)
        observed={'execution_count':units.height,'execution_cells':p.height,'fill_member_count':int(units['child_fill_count'].sum()),'candidate_cells':int(p['is_candidate'].sum()),'unique_candidate_executions_any_cell':p.filter(pl.col('is_candidate')).unique(KEYS).height}
        if any(expected[k]!=v for k,v in observed.items()): raise ValueError('native campaign population mismatch')
        for cohort,sub in [('All executions',units),('Primary candidates',primary(p).filter(pl.col('is_candidate')))]:
            comp.append(sub.group_by(['instrument',*ROLES]).len().rename({'len':'executions'}).with_columns(pl.lit(cohort).alias('cohort')))
        frames.append(p)
    population=pl.concat(frames); counts=cell_summary(population)
    if counts.height!=60: raise ValueError('expected all sixty cells')
    for row in counts.to_dicts():
        saved=reference.filter(pl.all_horizontal([pl.col(k)==row[k] for k in CELL]))
        if saved.height!=1 or any(saved[k][0]!=row[k] for k in ['execution_count','candidate_executions']):
            raise ValueError('raw campaign cell mismatch')
    native=primary(population).filter(pl.col('is_candidate'))
    frozen=pl.read_parquet(REPO/REVIEW/'expected_candidate_memberships.parquet')
    fields=[c for c in frozen.columns if c not in KEYS]
    # Freeze has additional native side/identity fields; read them without export.
    extended=pl.concat([primary(pl.read_parquet(REPO/r/'analysis/population.parquet',
        columns=[*frozen.columns,'age_policy','horizon_ns','is_candidate']))
        .filter(pl.col('is_candidate')).select(frozen.columns) for r in DEFAULT_ROOTS])
    verify_values(extended,frozen,fields)
    if {r['instrument']:r['len'] for r in native.group_by('instrument').len().to_dicts()}!=EXPECTED:
        raise ValueError('primary 6914 candidate contract changed')
    ranked=rank_candidates(native)
    prices=pl.read_parquet(prices_root/'private_execution_prices.parquet')
    verify_price_membership(prices,native); validate_prices(prices)
    if {'priority_top_w_market','priority_top_j_given_market','stage','ranking'} & set(prices.columns):
        raise ValueError('obsolete price ledger selection fields')
    # Original execution VWAP must not drift either.
    verify_values(prices,native,['execution_vwap'])
    summary=summarize_prices(prices); strata=summarize_prices(prices,roles=True)
    verify_strata(summary,strata)
    reconcile_summary(summary,pl.read_csv(prices_root/'summary.csv'),['instrument'])
    reconcile_summary(strata,pl.read_csv(prices_root/'role_identity_summary.csv'),['instrument',*ROLES])
    sensitivity=pl.concat([summarize_prices(prices,factor=f).with_columns(pl.lit(f).alias('ulp_factor')) for f in [0,4,8,16,32]])
    saved_sensitivity=pl.read_csv(prices_root/'sign_sensitivity.csv')
    reconcile_summary(sensitivity,saved_sensitivity,['instrument','ulp_factor'])
    if set(saved_sensitivity['ulp_factor'])!={0,4,8,16,32}: raise ValueError('price sensitivity coverage mismatch')
    cases=authenticate_reuse(population,counts)
    styles=load_json(REPO/APPROVED/'style_metadata.json')
    metadata=load_json(prices_root/'metadata.json')
    public=public_ranking(ranked,prices)
    # All validation above precedes creation. Only a genuinely new package is written.
    out.mkdir(parents=True,exist_ok=False)
    composition=pl.concat(comp)
    for name,frame in [('cell_counts.csv',counts),('ranking_summary.csv',primary(counts)),('composition.csv',composition),('cases.csv',cases),('instrument_totals.csv',totals),('primary_candidate_ranking.csv',public),('candidate_price_summary.csv',summary),('candidate_price_strata.csv',strata),('candidate_price_sign_sensitivity.csv',sensitivity)]:
        frame.write_csv(out/name)
    render_tables(out,totals,counts,composition,cases,summary,strata)
    reused=[]
    figures=[]
    for name,companion in [('candidate_counts.pdf','cell_counts.csv'),('withdrawal_and_profile_summary.pdf','cases.csv')]:
        source=REPO/APPROVED/name
        shutil.copyfile(source,out/name)
        if digest(out/name)!=digest(source): raise ValueError('reused figure bytes differ')
        reused.append({'asset':name,'sha256':digest(source),'companion':companion,'validation':'authenticated native companion equality'})
        figures.append(next(r for r in styles['figures'] if r['path']==name))
    bindings={str(REVIEW/'baseline_identity.json'):digest(REPO/REVIEW/'baseline_identity.json'),str(REVIEW/'expected_candidate_memberships.parquet'):digest(REPO/REVIEW/'expected_candidate_memberships.parquet'),str(Path(__file__).relative_to(REPO)):digest(__file__),'scripts/generate_withdrawal_multiscale_manuscript.py':digest(_old.__file__),str(APPROVED/'artifact_manifest.json'):digest(REPO/APPROVED/'artifact_manifest.json'),str(DIAGNOSTICS/'case_provenance.json'):digest(REPO/DIAGNOSTICS/'case_provenance.json'),'price_artifact_manifest_sha256':digest(prices_root/'artifact_manifest.json'),'price_metadata_sha256':digest(prices_root/'metadata.json')}
    facts={'version':'withdrawal_rank_publication_v1','scope':'all_candidate_market_rank_only','primary_policy':PRIMARY_POLICY,'primary_horizon_ns':PRIMARY_HORIZON_NS,'primary_candidate_executions':native.height,'primary_price_executions':prices.height,'primary_population':primary(counts).to_dicts(),'primary_prices':summary.to_dicts(),'ranking':'descending finite market W within instrument and parameter cell; ties execution_cluster_id, partition_id, actor_key; zero retained; unsupported null rank; J does not select or reorder','candidate_rule':'P_raw>0 and U_raw>0 and Q>0','reversion_horizon_ns':2_000_000_000,'price_sign_policy':metadata.get('sign_policy'),'source_evidence':evidence,'input_renderer_bindings':bindings,'reused_assets':reused,'figures':figures,'limitations':['descriptive triage, not intent or confirmed manipulation','role from source flags; firm fallback may pool clients','instrument coverage differs','J is a diagnostic, not an exclusion or ranking stage','raw physical price diagnostics unchanged; no price gating']}
    write_json(out/'facts.json',facts)
    write_json(out/'style_metadata.json',{**{k:v for k,v in styles.items() if k!='figures'},'figures':figures})
    write_json(out/'artifact_manifest.json',{'version':facts['version'],'artifacts':{p.name:digest(p) for p in sorted(out.iterdir()) if p.is_file()},'input_renderer_bindings':bindings})
    write_json(out/'COMPLETE.json',{'status':'publication_package_verified','scope':facts['scope'],'artifact_manifest_sha256':digest(out/'artifact_manifest.json'),'primary_candidate_executions':native.height,'primary_price_executions':prices.height})
    validate_stage(out,'publication_package_verified')
    return facts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--prices',type=Path,default=PRICES)
    args=parser.parse_args()
    out=args.output if args.output.is_absolute() else REPO/args.output
    facts=build(out,prices_root=args.prices)
    print(json.dumps({'output':str(out),'scope':facts['scope'],'primary_candidates':facts['primary_candidate_executions'],'price_executions':facts['primary_price_executions']},indent=2))


if __name__=='__main__':
    main()
