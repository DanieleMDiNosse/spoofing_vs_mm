#!/usr/bin/env python3
"""Cache-only primary exp_h30/H10 diagnostics; never changes native assignments.

FPM=h*(pre-fill mid - completed-posture mid); ADV=h*(original VWAP-posture).
REV=h*(pre-cancel mid - supported cancel+2s mid), raw attributed-qty weighted.
All event-time selection uses integer ns and full partition/actor identity.
"""
from __future__ import annotations
import argparse
from bisect import bisect_left, bisect_right
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import polars as pl

KEYS = ['instrument','partition_id','actor_key','execution_cluster_id']
LOCAL = KEYS[1:]
FLAGS = ['priority_top_w_self','priority_top_w_market','priority_top_j_given_self','priority_top_j_given_market']
ROOTS = {i: Path('outputs/withdrawal_multiscale_20261013_v4')/i for i in ['RISANAMENTO','NEXI']}
ROOTS['FERRARI'] = Path('outputs/withdrawal_multiscale_20261013_v4_ferrari_extended/FERRARI')
REGISTRY = Path('outputs/new_kernel_h1_verified_sources/verified_sources.json')
DEFAULT_OUT = Path('outputs/primary_kernel30_h10_prices_20261014_213056')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def checked_hash(path, expected):
    if not isinstance(expected,str) or re.fullmatch('[0-9a-f]{64}',expected) is None:
        raise ValueError(f'invalid SHA256: {path}')
    actual=sha256(path)
    if actual!=expected: raise ValueError(f'hash mismatch: {path}')
    return actual


def load_json(path):
    return json.loads(Path(path).read_text())


def dump_json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')


def verify_manifest(root, manifest='artifact_manifest.json', marker='COMPLETE.json'):
    root=Path(root);m=load_json(root/marker)
    checked_hash(root/manifest,m['artifact_manifest_sha256'])
    for name,digest in load_json(root/manifest)['artifacts'].items():
        checked_hash(root/name,digest)


def finite(x):
    return x is not None and math.isfinite(x)


def key(row, fields=LOCAL):
    return tuple(row[x] for x in fields)


class StateIndex:
    """Canonical predecessors and clock-ordered endpoints without ns float casts."""
    def __init__(self, rows):
        self.rows=sorted(rows,key=lambda x:x[0]);self.indices=[r[0] for r in self.rows]
        if len(set(self.indices))!=len(self.indices): raise ValueError('duplicate actor state index')
        self.timed=sorted((r for r in self.rows if r[1] is not None),key=lambda r:(r[1],r[0]))
        self.clocks=[r[1] for r in self.timed]
        self.suffix_max=[-1]*(len(self.rows)+1)
        for n in range(len(self.rows)-1,-1,-1):
            self.suffix_max[n]=max(self.suffix_max[n+1],self.rows[n][1] if self.rows[n][1] is not None else -1)

    def exact(self,index):
        p=bisect_left(self.indices,index)
        return self.rows[p] if p<len(self.rows) and self.indices[p]==index else None

    def pre(self,index,clock):
        p=bisect_left(self.indices,index)-1
        r=self.rows[p] if p>=0 else None
        # Do not skip an invalid latest predecessor and substitute another.
        return r if r and r[1] is not None and r[1]<=clock else None

    def post(self,index,target):
        start=bisect_left(self.indices,index)
        covered=self.suffix_max[start]>=target
        if not covered:return None,False
        p=bisect_right(self.clocks,target)-1
        while p>=0:
            if self.timed[p][0]>=index:return self.timed[p],True
            p-=1
        return None,True


def state_fields(prefix,row):
    return {prefix+'_sort_index':row[0] if row else None,
            prefix+'_replay_time_ns':row[1] if row else None,
            prefix+'_mid':row[2] if row else None}


def compute(executions,candidates,audit,cancellations,members,states,*,legacy=False):
    """Compute once per selected native execution; all four memberships retained."""
    for f in FLAGS:
        if executions.schema[f]!=pl.Boolean or executions[f].null_count():
            raise ValueError(f'native membership must be nonnull Boolean: {f}')
    for ranking in ['self','market']:
        if executions.filter(pl.col('priority_top_j_given_'+ranking)&~pl.col('priority_top_w_'+ranking)).height:
            raise ValueError('conditional J must be subset of W')
    if executions.select(LOCAL).unique().height!=executions.height:raise ValueError('duplicate execution key')
    if any(executions[x].null_count() for x in LOCAL):raise ValueError('null execution identity')
    selected=executions.filter(pl.any_horizontal(FLAGS))
    wanted=set(selected.select(LOCAL).iter_rows())
    cg=defaultdict(list)
    for r in candidates.iter_rows(named=True):
        if key(r) in wanted:cg[key(r)].append(r)
    ag=defaultdict(list)
    physical=set()
    for r in audit.iter_rows(named=True):
        if key(r) not in wanted or not r['assigned_flag']:continue
        q=r['attributed_cancel_visible_qty']
        if not finite(q) or q<0:raise ValueError('invalid attributed cancellation quantity')
        if q<=0:continue
        physical_key=(r['partition_id'],r['cancel_sort_index'],r['candidate_order_id'])
        if physical_key in physical:raise ValueError('duplicate physical assigned cancellation')
        physical.add(physical_key);ag[key(r)].append(r)
    cm={}
    for r in cancellations.iter_rows(named=True):
        k=(r['partition_id'],r['sort_index'],r['ORDERID'])
        if k in cm:raise ValueError('duplicate cancellation source key')
        cm[k]=r
    mg=defaultdict(list)
    for r in members.iter_rows(named=True):
        if key(r) in wanted:mg[key(r)].append(r)
    sg=defaultdict(list)
    for p,a,i,t,v in states.select('partition_id','actor_key','sort_index','replay_time_ns','market_mid').iter_rows():
        sg[(p,a)].append((i,t,v))
    indexes={k:StateIndex(v) for k,v in sg.items()}
    out=[];cancel_out=[]
    for e in selected.iter_rows(named=True):
        k=key(e);h={'ask':1,'bid':-1}.get(e['execution_side'])
        if h is None:raise ValueError('unknown execution side')
        fills=mg[k];qty=sum(x['child_fill_qty'] for x in fills)
        if not fills or not math.isclose(qty,e['execution_quantity'],rel_tol=1e-12,abs_tol=1e-9):
            raise ValueError('child execution quantity mismatch')
        vwap=sum(x['child_fill_qty']*x['child_fill_price'] for x in fills)/qty
        if not math.isclose(vwap,e['execution_vwap'],rel_tol=1e-12,abs_tol=1e-12):raise ValueError('child VWAP mismatch')
        idx=indexes.get((e['partition_id'],e['actor_key']),StateIndex([]))
        first=e['cluster_first_sort_index'];clock=e['cluster_start_replay_time_ns']
        eligible=[x for x in cg[k] if x['execution_sort_index']==first]
        if legacy:
            eligible=[x for x in eligible if x['deceptive_order_first_seen_replay_ns'] is not None
                      and 0<=clock-x['deceptive_order_first_seen_replay_ns']<=30_000_000_000]
        placement=max((x['deceptive_order_first_seen_sort_index'] for x in eligible
                       if x['deceptive_order_first_seen_sort_index'] is not None),default=None)
        observed=bool(eligible) and all(x['deceptive_order_placement_observed'] is True for x in eligible)
        posture=idx.exact(placement) if placement is not None and placement<first and observed else None
        if posture and (posture[1] is None or posture[1]>clock):posture=None
        pre=idx.pre(first,clock)
        fpm=h*(pre[2]-posture[2]) if pre and posture and finite(pre[2]) and finite(posture[2]) else None
        adv=h*(e['execution_vwap']-posture[2]) if posture and finite(posture[2]) and finite(e['execution_vwap']) else None
        assigned=ag[k];total=sum(x['attributed_cancel_visible_qty'] for x in assigned)
        if not math.isclose(total,e['withdrawal_qty_raw'],rel_tol=1e-12,abs_tol=1e-9):raise ValueError('withdrawal quantity mismatch')
        vals=[];missing=0
        for a in sorted(assigned,key=lambda a:(a['cancel_sort_index'],a['candidate_order_id'])):
            source=cm.get((a['partition_id'],a['cancel_sort_index'],a['candidate_order_id']))
            if source is None or source['actor_key']!=e['actor_key']:raise ValueError('cancellation physical join mismatch')
            ns=source['cancellation_removal_ns'];ci=a['cancel_sort_index']
            if not isinstance(ns,int):raise ValueError('cancellation clock must be integer ns')
            target=ns+2_000_000_000
            cp=idx.pre(ci,ns);post,covered=idx.post(ci,target)
            valid_delay=ns>=e['cluster_end_fill_anchor_ns']
            val=h*(cp[2]-post[2]) if cp and post and finite(cp[2]) and finite(post[2]) and valid_delay else None
            q=a['attributed_cancel_visible_qty']
            if val is None:missing+=1
            else:vals.append((val,q))
            cancel_out.append({**{x:e[x] for x in KEYS},
                               **{x:e.get(x) for x in ['execution_side','execution_anchor_mode','actor_id','identity_level','identity_source','identity_fallback_flag']},
                               **a,'cancellation_removal_ns':ns,'cancel_target_ns':target,
                               'raw_quantity_weight':q,'actor_coverage_through_target':covered,
                               'valid_economic_delay':valid_delay,**state_fields('cancel_pre',cp),
                               **state_fields('cancel_post',post),'REV_cancel':val})
        available=sum(v*q for v,q in vals)/sum(q for _,q in vals) if vals else None
        complete=bool(assigned) and missing==0
        out.append({**e,'h':h,'placement_index_requested':placement,'placement_observed_all':observed,
                    'initial_candidate_count':len(eligible),**state_fields('posture',posture),**state_fields('pre',pre),
                    'original_execution_vwap':e['execution_vwap'],'child_verified_vwap':vwap,
                    'FPM':fpm,'ADV':adv,'REV':available if complete else None,
                    'REV_available_only':available,'REV_complete':complete,
                    'REV_cancel_count':len(assigned),'REV_missing_cancel_count':missing,
                    'REV_raw_quantity':total})
    return pl.DataFrame(out,infer_schema_length=None),pl.DataFrame(cancel_out,infer_schema_length=None)


def summary(executions,roles=False):
    rows=[]
    for instrument in sorted(executions['instrument'].unique()):
        for ranking in ['self','market']:
            for stage in ['W','J']:
                flag=('priority_top_w_' if stage=='W' else 'priority_top_j_given_')+ranking
                subset=executions.filter((pl.col('instrument')==instrument)&pl.col(flag))
                groups=subset.partition_by(['execution_anchor_mode','identity_level'],as_dict=True) if roles else {():subset}
                for group,d in groups.items():
                    r=dict(instrument=instrument,ranking=ranking,stage=stage,n=d.height)
                    if roles:r.update(execution_anchor_mode=group[0],identity_level=group[1])
                    for metric in ['FPM','REV','ADV']:
                        values=d[metric].to_list();defined=[v for v in values if finite(v)]
                        for sign,fn in [('positive',lambda v:v>0),('zero',lambda v:v==0),('negative',lambda v:v<0)]:
                            r[metric+'_'+sign]=sum(fn(v) for v in defined)
                        r[metric+('_missing_or_incomplete' if metric=='REV' else '_missing')]=len(values)-len(defined)
                        r[metric+'_defined']=len(defined);r[metric+'_median']=statistics.median(defined) if defined else None
                    r['both_positive_complete']=d.filter((pl.col('FPM')>0)&(pl.col('REV')>0)&pl.col('REV_complete')).height
                    rows.append(r)
    return pl.DataFrame(rows,infer_schema_length=None)


def write_bundle(root,executions,cancels,metadata,parity):
    root=Path(root)
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True)
    executions.write_parquet(root/'private_execution_prices.parquet')
    cancels.write_parquet(root/'private_cancel_prices.parquet')
    summary(executions).write_csv(root/'summary.csv')
    summary(executions,roles=True).write_csv(root/'role_identity_summary.csv')
    parity.write_csv(root/'historical_reference_parity.csv')
    dump_json(root/'metadata.json',metadata)
    artifacts={p.name:sha256(p) for p in sorted(root.iterdir()) if p.is_file()}
    dump_json(root/'artifact_manifest.json',{'version':'primary_kernel_prices_v1','artifacts':artifacts})
    # Read every artifact back before making the atomic completion marker visible.
    for name,h in artifacts.items():checked_hash(root/name,h)
    marker={'status':'primary_prices_verified','artifact_manifest_sha256':sha256(root/'artifact_manifest.json')}
    dump_json(root/'.COMPLETE.pending',marker)
    (root/'.COMPLETE.pending').rename(root/'COMPLETE.json')
    verify_manifest(root)


def authenticate(instrument,root,registry):
    bindings={}
    def bind(p,h):
        p=Path(p).resolve();checked_hash(p,h);bindings[str(p)]=h
    for stage,status in [('prepare','prepared'),('analysis','verified')]:
        folder=root/stage;verify_manifest(folder)
        if load_json(folder/'COMPLETE.json')['status']!=status:raise ValueError('source not complete')
        for name in ['COMPLETE.json','artifact_manifest.json']:
            p=folder/name;bindings[str(p.resolve())]=sha256(p)
        for name,h in load_json(folder/'artifact_manifest.json')['artifacts'].items():bind(folder/name,h)
    analysis=root/'analysis'
    dm=load_json(analysis/'DERIVED.json')
    bind(analysis/'derived_manifest.json',dm['artifact_manifest_sha256'])
    for name,h in load_json(analysis/'derived_manifest.json')['artifacts'].items():bind(analysis/name,h)
    receipt=load_json(root/'prepare/prepare_receipt.json')
    derive=load_json(analysis/'derive_receipt.json')
    if derive['identity']['prepare_identity']!=receipt['identity']:raise ValueError('prepare identity mismatch')
    metadata_path=Path(registry['sources'][instrument]['bundle'])/'metadata.json'
    bind(metadata_path,registry['metadata_hashes'][instrument])
    metadata=load_json(metadata_path);state=Path(metadata['paths']['state_time_series'])
    h=metadata['artifact_hashes']['state_time_series'];bind(state,h)
    binding=receipt['identity']['input_bindings']['state_time_series']
    if Path(binding['path']).resolve()!=state.resolve() or binding['sha256']!=h:raise ValueError('state binding mismatch')
    if derive['identity']['coverage']['state_time_series_sha256']!=h:raise ValueError('derive state hash mismatch')
    return state,bindings


def historical_parity(instrument,root,inputs,states,refs):
    population=pl.read_parquet(root/'analysis/population.parquet').filter((pl.col('age_policy')=='legacy_age30')&(pl.col('horizon_ns')==1_000_000_000))
    audit=pl.read_parquet(root/'analysis/cancellation_audit.parquet').filter((pl.col('age_policy')=='legacy_age30')&(pl.col('horizon_ns')==1_000_000_000))
    # Only historical self lists enter this reference computation; market flags are irrelevant.
    population=population.with_columns(pl.lit(False).alias('priority_top_w_market'),pl.lit(False).alias('priority_top_j_given_market'))
    computed,_=compute(population,inputs[0],audit,inputs[1],inputs[2],states,legacy=True)
    evidence=[]
    for stage,ref in refs.items():
        reference=ref.filter(pl.col('instrument')==instrument)
        flag='priority_top_w_self' if stage=='W' else 'priority_top_j_given_self'
        actual=computed.filter(pl.col(flag));left={key(r,KEYS):r for r in actual.iter_rows(named=True)}
        right={key(r,KEYS):r for r in reference.iter_rows(named=True)}
        if left.keys()!=right.keys():raise ValueError(f'historical membership mismatch: {instrument}/{stage}')
        for metric,old in [('FPM','favorable_mid_move_pre_fill'),('ADV','execution_price_advantage_vs_posture_mid'),('REV','post_cancel_mid_reversion')]:
            errors=[];missing=0
            for k,a in left.items():
                x=a[metric];y=right[k][old]
                if finite(x)!=finite(y):raise ValueError(f'historical missingness mismatch: {instrument}/{stage}/{metric}/{k}')
                if finite(x):
                    error=abs(x-y);errors.append(error)
                    if not math.isclose(x,y,rel_tol=1e-10,abs_tol=1e-12):raise ValueError(f'historical price mismatch: {instrument}/{stage}/{metric}: {x} != {y}')
                else:missing+=1
            evidence.append(dict(instrument=instrument,stage=stage,metric=metric,n=len(left),defined=len(errors),missing=missing,max_abs_error=max(errors,default=0.),membership_exact=True,passed=True))
    return evidence


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=DEFAULT_OUT)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    registry=load_json(REGISTRY)
    refs={s:pl.read_csv('outputs/new_kernel_h1_top10_study/private_'+name+'_stage.csv',infer_schema_length=10000) for s,name in [('W','first'),('J','second')]}
    bindings={str(REGISTRY.resolve()):sha256(REGISTRY),str(Path(__file__).resolve()):sha256(__file__)}
    for name in ['first','second']:
        p=Path('outputs/new_kernel_h1_top10_study')/('private_'+name+'_stage.csv');bindings[str(p.resolve())]=sha256(p)
    all_e=[];all_c=[];parity=[];counts={}
    for instrument,root in ROOTS.items():
        print('authenticate',instrument,flush=True)
        state_path,b=authenticate(instrument,root,registry);bindings.update(b)
        pop=pl.read_parquet(root/'analysis/population.parquet').filter((pl.col('age_policy')=='exp_h30')&(pl.col('horizon_ns')==10_000_000_000))
        audit=pl.read_parquet(root/'analysis/cancellation_audit.parquet').filter((pl.col('age_policy')=='exp_h30')&(pl.col('horizon_ns')==10_000_000_000))
        candidates=pl.read_parquet(root/'prepare/candidates.parquet');cancels=pl.read_parquet(root/'prepare/cancellations.parquet');members=pl.read_parquet(root/'prepare/execution_cluster_members.parquet')
        original=pl.read_parquet(root/'prepare/executions.parquet')
        orig={key(r):r for r in original.iter_rows(named=True)}
        for r in pop.iter_rows(named=True):
            for field in ['execution_quantity','execution_vwap','execution_side','execution_anchor_mode','identity_level','identity_source']:
                if orig[key(r)][field]!=r[field]:raise ValueError(f'original execution field mismatch: {field}')
        old=pl.read_parquet(root/'analysis/population.parquet').filter((pl.col('age_policy')=='legacy_age30')&(pl.col('horizon_ns')==1_000_000_000)&pl.col('priority_top_w_self'))
        needed=pl.concat([pop.filter(pl.any_horizontal(FLAGS)).select('partition_id','actor_key'),old.select('partition_id','actor_key')]).unique()
        states=pl.scan_parquet(state_path).select('partition_id','actor_key','sort_index','replay_time_ns','market_mid').join(needed.lazy(),on=['partition_id','actor_key'],how='semi').collect(engine='streaming')
        print('compute',instrument,'states',states.height,flush=True)
        e,c=compute(pop,candidates,audit,cancels,members,states)
        all_e.append(e);all_c.append(c)
        counts[instrument]={'candidate_n':pop.filter(pl.col('is_candidate')).height,'union_n':e.height,'cancel_n':c.height,'flags':{f:int(pop[f].sum()) for f in FLAGS}}
        parity.extend(historical_parity(instrument,root,(candidates,cancels,members),states,refs))
        print('verified historical parity',instrument,counts[instrument],flush=True)
        del states
    expected={'FERRARI':(353,353,36,36),'NEXI':(240,240,24,24),'RISANAMENTO':(99,98,10,10)}
    for i,values in expected.items():
        if tuple(counts[i]['flags'][f] for f in FLAGS)!=values:raise ValueError('unexpected native cohort counts')
    if sum(x['candidate_n'] for x in counts.values())!=6914:raise ValueError('unexpected primary candidate denominator')
    for p,h in bindings.items():checked_hash(p,h)
    metadata={'status':'primary_prices_verified','version':'primary_kernel_prices_v1','age_policy':'exp_h30','horizon_ns':10_000_000_000,'reversion_horizon_ns':2_000_000_000,
              'native_cohorts':counts,'input_bindings':bindings,'historical_reference':'recomputed legacy_age30/H1 self W and conditional J; tolerance abs1e-12 rel1e-10',
              'formulas':{'FPM':'h*(pre_execution_mid-posture_mid)','ADV':'h*(original_execution_vwap-posture_mid)','REV':'sum(raw_attributed_qty*h*(pre_cancel_mid-post_cancel_mid))/sum(raw_attributed_qty), only if every positive assigned cancellation is supported'},
              'semantics':{'direction':'h=+1 ask/sell, -1 bid/buy','placement':'exact actor state at max first-seen index among all native eligible initial candidates; all placements observed; strictly before first fill',
                           'pre':'latest actor source index strictly before fill/cancel, clock no later than anchor; no invalid-row fallback',
                           'post':'max (replay_time_ns, sort_index) at/before cancel+2s, index>=cancel',
                           'coverage':'native _v3_has_coverage: at least one same partition/actor state index>=cancel and clock>=target; not feed completeness',
                           'membership':'native four nonnull Boolean flags, no price selection; conditional J subset W including J=0; ranking/stage overlap not independent',
                           'roles':'source execution_anchor_mode and identity fields preserved; native assignments and quantities unchanged'},
              'software':{'polars':pl.__version__}}
    write_bundle(args.output,pl.concat(all_e,how='diagonal_relaxed'),pl.concat(all_c,how='diagonal_relaxed'),metadata,pl.DataFrame(parity))
    print('COMPLETE',args.output,flush=True)


if __name__=='__main__':main()
