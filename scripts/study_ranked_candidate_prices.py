#!/usr/bin/env python3
"""Authenticated cache-only all-raw-candidate exp_h30/H10 price diagnostics.

Native assignments/endpoints are reused unchanged. Compatibility priority flags
exist only inside compute_all, are not eligibility rules, and never exported.
"""
from __future__ import annotations
import argparse
import gc
import importlib.util
import math
from pathlib import Path
import platform
import statistics
import sys
import polars as pl


def _load(name):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(name+'.py'))
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


base=_load('study_primary_kernel_prices')
numeric=_load('refine_primary_price_signs')
KEYS=base.KEYS
PHYSICAL=['instrument','partition_id','cancel_sort_index','candidate_order_id']
OUTPUT=Path('outputs/ranked_candidate_prices_20261015_101002_v2')
EXPECTED=Path('paper/reviews/20261015_101002_rank_only/expected_candidate_memberships.parquet')
REFERENCE=base.DEFAULT_OUT
NUMERIC_REFERENCE=Path(str(REFERENCE)+'_numeric_v2')
FACTORS=[0,4,8,16,32]


def validate_frozen(actual,expected):
    """Exact full-key and every frozen primitive/role/score, null-safe."""
    for d in [actual,expected]:
        if d.select(KEYS).unique().height!=d.height or any(d[k].null_count() for k in KEYS):
            raise ValueError('frozen duplicate/null identity')
    a=actual.select(expected.columns).sort(KEYS);b=expected.sort(KEYS)
    if not a.equals(b,null_equal=True):raise ValueError('frozen membership/primitive/role mismatch')


def compute_all(executions,candidates,audit,cancellations,members,states):
    # Caller supplies structural candidates; never filter on W, J or prices.
    for field in ['candidate_qty_raw','withdrawal_qty_raw','execution_quantity']:
        if executions.filter(~pl.col(field).is_finite()|pl.col(field).is_null()|(pl.col(field)<=0)).height:
            raise ValueError('invalid structural candidate quantity: '+field)
    clean=executions.drop([f for f in executions.columns if f.startswith('priority_')])
    adapter=clean.with_columns(*[pl.lit(f=='priority_top_w_self').alias(f) for f in base.FLAGS])
    e,c=base.compute(adapter,candidates,audit,cancellations,members,states)
    e=e.drop(base.FLAGS)
    c=c.drop([f for f in c.columns if f.startswith('priority_')])
    e=numeric.annotate(e,c,8)
    bounds=[numeric.endpoint_tolerance(r['cancel_pre_mid'],r['cancel_post_mid'],8) for r in c.iter_rows(named=True)]
    signs=[None if not base.finite(v) else (0 if abs(v)<=b else (1 if v>0 else -1)) for v,b in zip(c['REV_cancel'],bounds)]
    c=c.with_columns(pl.Series('REV_cancel_zero_tolerance',bounds,dtype=pl.Float64),pl.Series('REV_cancel_sign',signs,dtype=pl.Int8))
    return e,c


def summarize_all(executions,roles=False):
    fields=['instrument']+(['execution_anchor_mode','identity_level'] if roles else [])
    rows=[]
    for group,d in executions.partition_by(fields,as_dict=True).items():
        r=dict(zip(fields,group));r['n']=d.height
        for m in ['FPM','REV','ADV']:
            raw=[v for v in d[m] if base.finite(v)]
            normalized=[0. if s==0 else v for v,s in zip(d[m],d[m+'_sign']) if base.finite(v)]
            for label,sign in [('positive',1),('zero',0),('negative',-1)]:
                r[m+'_'+label]=sum(s==sign for s in d[m+'_sign'] if s is not None)
                r[m+'_raw_'+label]=sum((1 if v>0 else -1 if v<0 else 0)==sign for v in raw)
            r[m+'_defined']=len(raw);r[m+'_missing_or_incomplete' if m=='REV' else m+'_missing']=d.height-len(raw)
            r[m+'_median']=statistics.median(normalized) if normalized else None
            r[m+'_raw_median']=statistics.median(raw) if raw else None
        r['both_positive_complete']=d.filter((pl.col('FPM_sign')==1)&(pl.col('REV_sign')==1)&pl.col('REV_complete')).height
        rows.append(r)
    return pl.DataFrame(rows,infer_schema_length=None).sort(fields)


def compare_reference(actual,reference,keys,kind):
    """Full common-subset endpoint/raw parity; raw REV may differ <=1e-12."""
    left={tuple(r[k] for k in keys):r for r in actual.iter_rows(named=True)}
    if reference.select(keys).unique().height!=reference.height:raise ValueError('duplicate reference')
    ignored=set(base.FLAGS)
    columns=[x for x in reference.columns if x not in ignored]
    errors={x:0. for x in columns};missing={x:0 for x in columns}
    for old in reference.iter_rows(named=True):
        k=tuple(old[x] for x in keys)
        if k not in left:raise ValueError('missing reference key')
        new=left[k]
        for col in columns:
            a=new[col];b=old[col]
            if a==b or (a is None and b is None):continue
            # Only aggregate raw REV allows roundoff; all endpoints/signs exact.
            if col in ['REV','REV_available_only'] and base.finite(a) and base.finite(b):
                err=abs(a-b);errors[col]=max(errors[col],err)
                if err<=1e-12:continue
            raise ValueError(f'reference parity {kind} {col} {k}: {a} != {b}')
    return [dict(reference=kind,column=c,n=reference.height,max_abs_error=errors[c],passed=True) for c in columns]


def run(out=OUTPUT):
    out=Path(out)
    if out.exists():raise FileExistsError(out)  # Before *any* source reads.
    out.mkdir(parents=True)
    log=(out/'progress.log').open('w')
    def progress(*args):
        text=' '.join(map(str,args));print(text,flush=True);log.write(text+'\n');log.flush()
    bindings={}
    def bind(p):
        p=Path(p);bindings[str(p.resolve())]=base.sha256(p)
    for p in [EXPECTED,base.REGISTRY,Path(__file__),Path(base.__file__),Path(numeric.__file__),Path('tests/lob/test_ranked_candidate_prices.py')]:bind(p)
    expected=pl.read_parquet(EXPECTED)
    if expected.height!=6914:raise ValueError('unexpected frozen denominator')
    registry=base.load_json(base.REGISTRY)
    for folder in [REFERENCE,NUMERIC_REFERENCE]:
        base.verify_manifest(folder)
        for p in [folder/'COMPLETE.json',folder/'artifact_manifest.json',*[folder/n for n in base.load_json(folder/'artifact_manifest.json')['artifacts']]]:bind(p)
    all_e=[];all_c=[];counts={}
    for instrument in ['RISANAMENTO','NEXI','FERRARI']:
        root=base.ROOTS[instrument];progress('authenticate',instrument)
        state_path,b=base.authenticate(instrument,root,registry);bindings.update(b)
        policy=(pl.col('age_policy')=='exp_h30')&(pl.col('horizon_ns')==10_000_000_000)
        pop=pl.read_parquet(root/'analysis/population.parquet').filter(policy)
        selected=pop.filter((pl.col('candidate_qty_raw')>0)&(pl.col('withdrawal_qty_raw')>0)&(pl.col('execution_quantity')>0))
        frozen=expected.filter(pl.col('instrument')==instrument);validate_frozen(selected,frozen)
        original=pl.read_parquet(root/'prepare/executions.parquet')
        fields=['execution_quantity','execution_vwap','execution_side','execution_anchor_mode','identity_level','identity_source','actor_id','identity_fallback_flag']
        joined=selected.join(original.select(base.LOCAL+fields),on=base.LOCAL,how='left',suffix='_original',validate='1:1')
        for f in fields:
            if joined.filter(pl.col(f).ne_missing(pl.col(f+'_original'))).height:raise ValueError('original primitive mismatch '+f)
        wanted=selected.select(base.LOCAL)
        audit=pl.read_parquet(root/'analysis/cancellation_audit.parquet').filter(policy).join(wanted,on=base.LOCAL,how='semi')
        candidates=pl.read_parquet(root/'prepare/candidates.parquet').join(wanted,on=base.LOCAL,how='semi')
        cancels=pl.read_parquet(root/'prepare/cancellations.parquet')
        members=pl.read_parquet(root/'prepare/execution_cluster_members.parquet').join(wanted,on=base.LOCAL,how='semi')
        es=[];cs=[];max_states=0
        for (partition,actor),ex in selected.partition_by(['partition_id','actor_key'],as_dict=True).items():
            states=pl.scan_parquet(state_path).filter((pl.col('partition_id')==partition)&(pl.col('actor_key')==actor)).select('partition_id','actor_key','sort_index','replay_time_ns','market_mid').collect(engine='streaming')
            max_states=max(max_states,states.height)
            local=ex.select(base.LOCAL)
            e,c=compute_all(ex,candidates.join(local,on=base.LOCAL,how='semi'),audit.join(local,on=base.LOCAL,how='semi'),cancels.filter((pl.col('partition_id')==partition)&(pl.col('actor_key')==actor)),members.join(local,on=base.LOCAL,how='semi'),states)
            es.append(e);cs.append(c)
            del states;gc.collect()
        e=pl.concat(es,how='diagonal_relaxed');c=pl.concat(cs,how='diagonal_relaxed')
        validate_frozen(e,frozen)
        assigned=audit.filter(pl.col('assigned_flag')&(pl.col('attributed_cancel_visible_qty')>0))
        auditkeys=['partition_id','cancel_sort_index','candidate_order_id']
        if assigned.select(auditkeys).unique().height!=assigned.height or c.height!=assigned.height:raise ValueError('physical audit uniqueness/count')
        auditview=assigned.select(base.LOCAL+auditkeys[1:]+['attributed_cancel_visible_qty']).sort(base.LOCAL+auditkeys[1:])
        ledgerview=c.select(auditview.columns).sort(base.LOCAL+auditkeys[1:])
        if not auditview.equals(ledgerview):raise ValueError('native audit assignments changed')
        # Per-execution audit raw sum is checked inside base.compute too.
        counts[instrument]={'executions':e.height,'physical_cancels':c.height,'max_actor_state_rows':max_states,'raw_cancel_quantity':c['raw_quantity_weight'].sum(),'zero_W':e.filter(pl.col('WMSCI_market')==0).height,'zero_J':e.filter(pl.col('J')==0).height}
        progress('computed',instrument,counts[instrument])
        e.write_parquet(out/(instrument+'_execution_prices.parquet'));c.write_parquet(out/(instrument+'_cancel_prices.parquet'))
        all_e.append(e);all_c.append(c)
        del pop,selected,original,joined,audit,candidates,cancels,members;gc.collect()
    e=pl.concat(all_e,how='diagonal_relaxed').sort(['instrument','WMSCI_market','execution_cluster_id','partition_id','actor_key'],descending=[False,True,False,False,False])
    c=pl.concat(all_c,how='diagonal_relaxed').sort(PHYSICAL)
    validate_frozen(e,expected)
    if c.select(PHYSICAL).unique().height!=c.height:raise ValueError('duplicate physical cancellation')
    parity=[]
    for folder in [REFERENCE,NUMERIC_REFERENCE]:
        olde=pl.read_parquet(folder/'private_execution_prices.parquet');oldc=pl.read_parquet(folder/'private_cancel_prices.parquet')
        if olde.height!=906 or oldc.height!=2482:raise ValueError('unexpected reference counts')
        parity+=compare_reference(e,olde,KEYS,folder.name+'/execution')
        parity+=compare_reference(c,oldc,PHYSICAL,folder.name+'/cancel')
    e.write_parquet(out/'private_execution_prices.parquet');c.write_parquet(out/'private_cancel_prices.parquet')
    summary=summarize_all(e)
    if summary.height!=3 or summary['n'].sum()!=6914:raise ValueError('summary denominator')
    summary.write_csv(out/'summary.csv');summarize_all(e,True).write_csv(out/'role_identity_summary.csv')
    pl.concat([summarize_all(numeric.annotate(e,c,factor)).with_columns(pl.lit(factor).alias('ulp_factor')) for factor in FACTORS]).write_csv(out/'sign_sensitivity.csv')
    pl.DataFrame(parity).write_csv(out/'reference_parity.csv')
    progress('all checks passed',e.height,c.height)
    log.close()
    for p,h in bindings.items():base.checked_hash(p,h)
    metadata={'version':'ranked_all_candidate_prices_v2','status':'ranked_candidate_prices_verified','age_policy':'exp_h30','horizon_ns':10_000_000_000,'reversion_horizon_ns':2_000_000_000,'counts':counts,'input_bindings':bindings,
        'eligibility':'candidate_qty_raw>0 AND withdrawal_qty_raw>0 AND execution_quantity>0; exact frozen 6914 keys; market-depth support never selects prices; no W/J/price threshold',
        'ordering':'instrument, descending WMSCI_market; equal-W full-key deterministic serialization only; J not used',
        'compatibility_adapter':'all selected supplied with internal self-W=True and other compatibility flags=False; all priority flags dropped before exports; native assignments/quantities untouched',
        'formulas':{'FPM':'h*(pre_mid-posture_mid)','ADV':'h*(original_execution_vwap-posture_mid)','REV':'sum(raw_attributed_qty*REV_cancel)/sum(raw_attributed_qty), defined only if every positive assigned physical cancellation observed','h':'ask +1; bid -1'},
        'endpoints':'unchanged study_primary_kernel_prices StateIndex exact/pre/post; integer ns; no invalid predecessor fallback; actor partition scoped; cancel+2s not H10; coverage is native actor-cache support not feed completeness',
        'sign_policy':{'name':'endpoint_scaled_ulp_roundoff_v1','ulp_factor':8,'sensitivity':FACTORS,'raw_values':'unchanged; numerical signs and medians separately reported; missing never zero'},
        'reference_parity':{'executions':906,'physical_cancels':2482,'endpoints_and_signs':'exact','raw_REV_abs_tolerance':1e-12},
        'software':{'python':sys.version,'polars':pl.__version__,'platform':platform.platform()}}
    base.dump_json(out/'metadata.json',metadata)
    base.dump_json(out/'config.json',{k:metadata[k] for k in ['age_policy','horizon_ns','reversion_horizon_ns','eligibility','ordering','sign_policy']})
    base.dump_json(out/'software.json',metadata['software']);base.dump_json(out/'input_bindings.json',bindings)
    artifacts={p.name:base.sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    base.dump_json(out/'artifact_manifest.json',{'version':metadata['version'],'artifacts':artifacts})
    for n,h in artifacts.items():base.checked_hash(out/n,h)
    base.dump_json(out/'.COMPLETE.pending',{'status':metadata['status'],'artifact_manifest_sha256':base.sha256(out/'artifact_manifest.json')})
    (out/'.COMPLETE.pending').rename(out/'COMPLETE.json');base.verify_manifest(out)
    print('COMPLETE',out,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    run(parser.parse_args().output)
