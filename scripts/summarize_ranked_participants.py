#!/usr/bin/env python3
"""Read-only participant consumer for the frozen primary native execution cohort.

Two NONADDITIVE perspectives: canonical identified clients, and original event
firms over client-attributed AND fallback executions. No W/P/U sums, score gates,
price gates, cross-instrument score pooling, or raw replay. Public tables contain
only report-local Cxxx/Fxxx aliases. Never export ``identity_mapping`` publicly.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import statistics

import polars as pl
from scripts import generate_rank_only_manuscript as frozen
from spoofing_detection.lob.actor_identity import (
    normalize_identity_value, resolve_actor_identity,
)

KEYS = frozen.KEYS
PERSPECTIVES = ('identified_client', 'intermediary')
MISSING_FIRM = 'firm_missing:'
IDENTITY = ['event_client_original_id', 'event_firm_id', 'identity_fallback_flag']


def _key(row):
    return tuple(row[k] for k in KEYS)


def validate_member_firms(population: pl.DataFrame, members: pl.DataFrame) -> dict:
    """Require exact child coverage and every original child firm == anchor firm.

    Null firms match only null firms, which form an explicit unknown-firm bucket.
    A missing/nonmissing mismatch or a multi-firm execution stops attribution.
    Separate executions for one canonical client may legitimately use many firms.
    """
    required = {*KEYS, 'child_sort_index', 'child_event_firm_id'}
    if required - set(members.columns):
        raise ValueError('member schema missing required fields')
    if any(members.select(KEYS + ['child_sort_index']).null_count().row(0)):
        raise ValueError('member null key')
    if members.select(KEYS + ['child_sort_index']).is_duplicated().any():
        raise ValueError('duplicate member key')
    anchors = {_key(r): r for r in population.iter_rows(named=True)}
    counts = Counter()
    for r in members.iter_rows(named=True):
        k = _key(r)
        if k not in anchors:
            raise ValueError('member execution membership differs')
        if normalize_identity_value(r['child_event_firm_id']) != normalize_identity_value(anchors[k]['event_firm_id']):
            raise ValueError('execution child firm differs from anchor: multi-firm or missing-firm mismatch; STOP attribution')
        counts[k] += 1
    if set(counts) != set(anchors) or any(counts[k] != r['child_fill_count'] for k, r in anchors.items()):
        raise ValueError('member coverage/count differs from native execution population')
    return {'checked_execution_count': len(anchors), 'checked_child_member_count': members.height,
            'multifirm_execution_count': 0, 'firm_check': 'all_children_equal_original_anchor_firm'}


def _statistics(rows, price_by_key):
    candidates = [r for r in rows if r['is_candidate']]
    w = [r['WMSCI_market'] for r in candidates if frozen.finite(r['WMSCI_market'])]
    j = [r['J'] for r in candidates if r['complete_grid'] and frozen.finite(r['J'])]
    result = dict(execution_count=len(rows), candidate_count=len(candidates),
        candidate_fraction=len(candidates)/len(rows) if rows else None,
        observed_partition_count=len({r['partition_id'] for r in rows}),
        supported_W_count=len(w), unsupported_W_count=len(candidates)-len(w),
        positive_W_count=sum(v > 0 for v in w), zero_W_count=w.count(0.),
        negative_W_count=sum(v < 0 for v in w),
        W_median=statistics.median(w) if w else None, W_max=max(w) if w else None,
        complete_grid_candidate_count=sum(r['complete_grid'] for r in candidates),
        complete_finite_J_count=len(j), J_missing_or_incomplete_count=len(candidates)-len(j),
        zero_J_count=j.count(0.), J_median=statistics.median(j) if j else None,
        price_denominator=len(candidates))
    for role in ('passive', 'aggressive'):
        result[role+'_execution_count'] = sum(r['execution_anchor_mode']==role for r in rows)
        result[role+'_candidate_count'] = sum(r['execution_anchor_mode']==role for r in candidates)
    for source, level in [('client','client_original'), ('fallback','firm')]:
        result['source_'+source+'_execution_count'] = sum(r['identity_level']==level for r in rows)
        result['source_'+source+'_candidate_count'] = sum(r['identity_level']==level for r in candidates)
    prices = [price_by_key[_key(r)] for r in candidates]
    for metric in frozen.METRICS:
        signs = [r[metric+'_sign'] for r in prices]
        for label, sign in [('positive',1),('zero',0),('negative',-1),('missing',None)]:
            result[metric+'_'+label+'_count'] = signs.count(sign)
        assert sum(result[metric+'_'+s+'_count'] for s in ['positive','zero','negative','missing']) == len(candidates)
    result['FPM_REV_positive_count'] = sum(r['FPM_sign']==1 and r['REV_sign']==1 and r['REV_complete'] for r in prices)
    result['FPM_REV_ADV_positive_count'] = sum(r['FPM_sign']==1 and r['REV_sign']==1 and r['ADV_sign']==1 and r['REV_complete'] for r in prices)
    return result


def aggregate_participants(population: pl.DataFrame, prices: pl.DataFrame,
                           members: pl.DataFrame) -> dict:
    """Pure API: one primary row per ALL native executions; exact candidate prices.

    Members must include instrument (loader adds the known per-root instrument).
    Returns tables: entities (complete, including zero candidates), entity_strata
    (role x source attribution, including empty strata), summary (two perspectives
    per instrument), accounting (disjoint identified/fallback native partitions).
    Identity mapping is deliberately separate/private. No function writes files.
    Partition counts are NOT calendar-day counts: no unverified clock/date mapping.
    """
    if population.is_empty():
        raise ValueError('empty execution population')
    if set(IDENTITY + ['child_fill_count']) - set(population.columns):
        raise ValueError('missing original identity/member count fields')
    for r in population.iter_rows(named=True):
        identity = resolve_actor_identity(client_original_id=r['event_client_original_id'], firm_id=r['event_firm_id'])
        if (identity is None or identity.actor_key != r['actor_key'] or identity.identity_level != r['identity_level']
                or type(r['identity_fallback_flag']) is not bool
                or identity.identity_fallback_flag != r['identity_fallback_flag']):
            raise ValueError('native identity namespace/attribution mismatch')
    frozen.validate_keys(population)
    frozen.validate_population(population)
    if frozen.primary(population).height != population.height:
        raise ValueError('population must contain only primary exp_h30/H10 executions')
    native = population.filter(pl.col('is_candidate'))
    frozen.verify_price_membership(prices, native)
    frozen.verify_values(prices, native, IDENTITY)
    frozen.validate_prices(prices)
    member_evidence = validate_member_firms(population, members)
    price_by_key = {_key(r):r for r in prices.iter_rows(named=True)}
    records = population.to_dicts()
    for r in records:
        firm = normalize_identity_value(r['event_firm_id'])
        r['_firm_key'] = 'firm:'+firm if firm is not None else MISSING_FIRM
    client_keys = sorted({r['actor_key'] for r in records if r['identity_level']=='client_original'})
    firm_keys = sorted({r['_firm_key'] for r in records})
    aliases = {**{k:f'C{i:03d}' for i,k in enumerate(client_keys,1)},
               **{k:f'F{i:03d}' for i,k in enumerate(firm_keys,1)}}
    mapping = pl.DataFrame([dict(perspective=p, entity_alias=aliases[k], original_key=k,
        firm_missing_bucket=k==MISSING_FIRM) for p,keys in zip(PERSPECTIVES,[client_keys,firm_keys]) for k in keys])
    groups = defaultdict(list)
    for r in records:
        groups[(r['instrument'], 'intermediary', r['_firm_key'])].append(r)
        if r['identity_level']=='client_original':
            groups[(r['instrument'], 'identified_client', r['actor_key'])].append(r)
    entities, strata, summaries, accounting = [], [], [], []
    for (instrument, perspective, key), rows in sorted(groups.items()):
        header = dict(instrument=instrument, perspective=perspective, entity_alias=aliases[key], firm_missing_bucket=key==MISSING_FIRM)
        entities.append({**header, **_statistics(rows, price_by_key)})
        for role in ('passive','aggressive'):
            for level in ('client_original','firm'):
                sub = [r for r in rows if r['execution_anchor_mode']==role and r['identity_level']==level]
                strata.append({**header, 'execution_anchor_mode':role, 'identity_level':level,
                               **_statistics(sub, price_by_key)})
    for instrument in sorted({r['instrument'] for r in records}):
        instrument_rows = [r for r in records if r['instrument']==instrument]
        c = [r for r in instrument_rows if r['identity_level']=='client_original']
        f = [r for r in instrument_rows if r['identity_level']=='firm']
        accounting.append(dict(instrument=instrument, execution_count=len(instrument_rows),
            candidate_count=sum(r['is_candidate'] for r in instrument_rows),
            identified_client_execution_count=len(c), fallback_execution_count=len(f),
            identified_client_candidate_count=sum(r['is_candidate'] for r in c),
            fallback_candidate_count=sum(r['is_candidate'] for r in f)))
        for perspective in PERSPECTIVES:
            e = [r for r in entities if r['instrument']==instrument and r['perspective']==perspective]
            denominator = c if perspective=='identified_client' else instrument_rows
            n = sum(r['candidate_count'] for r in e)
            executions = sum(r['execution_count'] for r in e)
            if executions != len(denominator) or n != sum(r['is_candidate'] for r in denominator):
                raise ValueError('entity partitions fail native accounting')
            largest = max((r['candidate_count'] for r in e), default=0)
            summaries.append(dict(instrument=instrument, perspective=perspective,
                entity_count=len(e), candidate_entity_count=sum(r['candidate_count']>0 for r in e),
                execution_count=executions, candidate_count=n,
                largest_candidate_group_count=largest, largest_candidate_group_denominator=n,
                largest_candidate_group_fraction=largest/n if n else None,
                repeat_candidate_entity_count=sum(r['candidate_count']>=2 for r in e)))
    tables = {'entities': pl.DataFrame(entities, infer_schema_length=None),
              'entity_strata': pl.DataFrame(strata, infer_schema_length=None),
              'summary': pl.DataFrame(summaries, infer_schema_length=None),
              'accounting': pl.DataFrame(accounting, infer_schema_length=None)}
    # All count fields, not just total candidates, must partition across strata.
    count_fields = [c for c in tables['entities'].columns if c.endswith('_count') and c != 'observed_partition_count']
    for e in entities:
        sub = [r for r in strata if all(r[k]==e[k] for k in ['instrument','perspective','entity_alias'])]
        if any(sum(r[c] for r in sub) != e[c] for c in count_fields):
            raise ValueError('role/identity count partition mismatch')
    return {'tables': tables, 'identity_mapping': mapping,
            'provenance': {'primary_policy': frozen.PRIMARY_POLICY, 'horizon_ns':frozen.PRIMARY_HORIZON_NS,
                'perspectives_nonadditive':True, 'coverage_unit':'observed_partition_not_calendar_day',
                'roles':'source-derived execution_anchor_mode; native semantics unchanged',
                'alias_scope':'report-local, global across loaded instruments; sorted complete identity union',
                'member_firm_validation':member_evidence}}


def validate_price_bindings(metadata: dict, authenticated_inputs: dict) -> None:
    """Bind consumed native/prepared hashes to the price producer's saved inputs."""
    saved = metadata.get('input_bindings', {})
    for path, expected in authenticated_inputs.items():
        actual = saved.get(path)
        if (not isinstance(actual, str) or len(actual)!=64
                or any(c not in '0123456789abcdef' for c in actual) or actual!=expected):
            raise ValueError('price/native input binding mismatch: '+path)


def load_native_participants(*, roots=None, prices_root=None) -> dict:
    """Authenticate frozen native/prepared/price manifests, then aggregate read-only.

    Defaults are the three frozen DEFAULT_ROOTS and ranked-candidate price v2.
    Native grid/expected counts and price membership/values/signs are reconciled.
    Provenance retains complete authenticated artifact hashes and manifest hashes.
    This API performs NO writes, including CSV/mapping writes; parent owns exports.
    """
    roots = [Path(r) for r in (frozen.DEFAULT_ROOTS if roots is None else roots)]
    roots = [r if r.is_absolute() else frozen.REPO/r for r in roots]
    prices_root = Path(frozen.PRICES if prices_root is None else prices_root)
    if not prices_root.is_absolute():
        prices_root = frozen.REPO/prices_root
    populations, members, sources = [], [], []
    columns = list(dict.fromkeys(KEYS + ['age_policy','horizon_ns','is_candidate',
        'child_fill_count',*frozen.NATIVE,*IDENTITY]))
    for root in roots:
        evidence = frozen.validate_root(root)
        manifests = {}
        for directory, name in [('prepare','artifact_manifest.json'),('analysis','artifact_manifest.json'),
                                ('analysis','derived_manifest.json'),('report','artifact_manifest.json')]:
            path = root/directory/name
            manifests[directory+'/'+name] = {'sha256':frozen.digest(path), 'manifest':frozen.load_json(path)}
        pop = pl.read_parquet(root/'analysis/population.parquet', columns=columns)
        frozen.validate_population(pop)
        units = pop.select(KEYS).unique()
        if (pop.height != 20*units.height or set(pop['age_policy'])!=set(frozen._old.POLICIES)
                or set(pop['horizon_ns'])!=set(frozen._old.HORIZONS)):
            raise ValueError('native twenty-cell execution grid mismatch')
        primary = frozen.primary(pop)
        instruments = primary['instrument'].unique().to_list()
        if len(instruments)!=1:
            raise ValueError('native per-root instrument mismatch')
        member = pl.read_parquet(root/'prepare/execution_cluster_members.parquet',
            columns=[k for k in KEYS if k!='instrument']+['child_sort_index','child_event_firm_id'])
        member = member.with_columns(pl.lit(instruments[0]).alias('instrument'))
        populations.append(primary); members.append(member)
        sources.append({'authentication':evidence, 'manifests':manifests})
    price_manifest = frozen.validate_stage(prices_root, 'ranked_candidate_prices_verified')
    required = {'private_execution_prices.parquet','private_cancel_prices.parquet','summary.csv',
                'role_identity_summary.csv','sign_sensitivity.csv','metadata.json'}
    if required-set(price_manifest['artifacts']):
        raise ValueError('price manifest missing bound artifacts')
    price_metadata = frozen.load_json(prices_root/'metadata.json')
    bindings = {}
    for root, source in zip(roots, sources):
        manifests = source['manifests']
        for relative in ['prepare/artifact_manifest.json','analysis/artifact_manifest.json','analysis/derived_manifest.json']:
            bindings[str(root/relative)] = manifests[relative]['sha256']
        for relative, manifest in [('prepare/execution_cluster_members.parquet','prepare/artifact_manifest.json'),
                                   ('analysis/population.parquet','analysis/derived_manifest.json')]:
            bindings[str(root/relative)] = manifests[manifest]['manifest']['artifacts'][Path(relative).name]
    validate_price_bindings(price_metadata, bindings)
    population = pl.concat(populations, how='vertical_relaxed')
    prices = pl.read_parquet(prices_root/'private_execution_prices.parquet')
    result = aggregate_participants(population, prices, pl.concat(members, how='vertical_relaxed'))
    actual = {r['instrument']:r['len'] for r in population.filter(pl.col('is_candidate')).group_by('instrument').len().to_dicts()}
    if actual != frozen.EXPECTED or population.height != 48_481:
        raise ValueError('frozen native 48481 execution / 6914 candidate contract changed')
    summary = frozen.summarize_prices(prices)
    strata = frozen.summarize_prices(prices, roles=True)
    frozen.verify_strata(summary, strata)
    frozen.reconcile_summary(summary, pl.read_csv(prices_root/'summary.csv'), ['instrument'])
    frozen.reconcile_summary(strata, pl.read_csv(prices_root/'role_identity_summary.csv'), ['instrument',*frozen.ROLES])
    result['provenance'].update(native_sources=sources,
        price_source={'root':str(prices_root.relative_to(frozen.REPO)),
            'manifest_sha256':frozen.digest(prices_root/'artifact_manifest.json'),
            'manifest':price_manifest, 'metadata':price_metadata},
        authenticated_price_native_bindings=bindings,
        observed_execution_count=population.height, candidate_count=prices.height)
    return result


if __name__ == '__main__':
    # Public counts only; private mapping is never printed or written.
    print(load_native_participants()['tables']['summary'])
