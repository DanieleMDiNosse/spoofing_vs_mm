"""Pure bounded withdrawal_multiscale_v1 accounting and numerical adapters.

No replay, files, price gates or cohort pruning.  Input links must retain ALL
max-horizon competitors, including previously unassigned rows.  Age is frozen
on the native first-fill sequence baseline, never the economic fill clock.
"""
from __future__ import annotations

import math
from collections import defaultdict
from numbers import Integral, Real

import polars as pl

from .spoofing_metrics import assign_cancellations_to_clusters
from .msci_grid_runner import (
    compute_partition_grids, verify_partition_grids, summarize_grid,
    _sample_local_execution, _state_groups,
)
from .msci_grid_store import SUMMARY_SCHEMA, TRAJECTORY_SCHEMA

KEYS = ['instrument', 'partition_id', 'actor_key', 'execution_cluster_id']
_SCORE_SCHEMA = {name: pl.Float64 for name in (
    'WMSCI_self', 'WMSCI_market', 'removal_share_age_weighted',
    'candidate_market_share_age_weighted',
)}
POPULATION_SCHEMA = {
    'horizon_ns': pl.Int64, 'age_policy': pl.String,
    'candidate_qty_raw': pl.Float64, 'withdrawal_qty_raw': pl.Float64,
    'candidate_qty_age_weighted': pl.Float64, 'withdrawal_qty_age_weighted': pl.Float64,
    'age_weight_underflow_count': pl.Int64, 'is_candidate': pl.Boolean,
    'market_score_supported': pl.Boolean, **_SCORE_SCHEMA,
}
AUDIT_SCHEMA = {
    **{k: pl.String for k in KEYS}, 'candidate_order_id': pl.String,
    'cancel_sort_index': pl.Int64, 'cluster_first_sort_index': pl.Int64,
    'cluster_last_sort_index': pl.Int64, 'horizon_ns': pl.Int64,
    'age_policy': pl.String, 'assigned_flag': pl.Boolean,
    'assignment_rule': pl.String, 'competing_cluster_count': pl.Int64,
    'age_ns': pl.Int64, 'age_weight': pl.Float64,
    'deceptive_order_first_seen_sort_index': pl.Int64,
    'cancel_order_first_seen_sort_index': pl.Int64,
    'cancel_visible_qty': pl.Float64, 'candidate_visible_qty_pre': pl.Float64,
    'attributed_cancel_visible_qty': pl.Float64,
    'attributed_cancel_qty_age_weighted': pl.Float64,
}
PROFILE_SCHEMA = {
    **SUMMARY_SCHEMA, 'horizon_ns': pl.Int64,
    'MSCI_t0': pl.Float64, 'MSCI_horizon': pl.Float64, 'MSCI_max': pl.Float64,
    'MSCI_max_offset_ns': pl.Int64, 'reason': pl.String,
}


def _integer(value, label, *, minimum=None):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f'{label} must be an integer (not bool/float/null)')
    value = int(value)
    if minimum is not None and value < minimum:
        raise ValueError(f'{label} must be >= {minimum}')
    return value


def _quantity(value, label):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value) or value < 0:
        raise ValueError(f'{label} must be a finite nonnegative quantity')
    return float(value)


def score_pair(p, u, q, d):
    """Two dependent normalizations, without epsilon or a U/Q gate."""
    p, u, q = (_quantity(v, n) for v, n in ((p,'P'),(u,'U'),(q,'Q')))
    d = None if d is None else _quantity(d,'D')
    if q <= 0 or u > p or (d is not None and p > d):
        raise ValueError('require Q>0 and U<=P<=D when D is measured')
    share = u/p if p else None
    # Stable log1p ratios even when the ratio itself would overflow.
    def log_ratio(v):
        if not v: return 0.0
        delta = math.log(v)-math.log(q)
        return delta + math.log1p(math.exp(-delta)) if delta > 0 else math.log1p(math.exp(delta))
    product = log_ratio(p)*log_ratio(u)
    return {
        'WMSCI_self': product*share if share is not None else 0.0,
        'WMSCI_market': product*(u/d) if d else None,
        'removal_share_age_weighted': share,
        'candidate_market_share_age_weighted': p/d if d else None,
    }


def age_weight(age_ns, half_life_ns):
    """Frozen native lifecycle age; unknown/negative ages are invalid."""
    age = _integer(age_ns,'age_ns',minimum=0)
    if half_life_ns is None: return 1.0
    half = _integer(half_life_ns,'half_life_ns',minimum=1)
    return math.exp(-math.log(2)*(age/half))


def _frame(frame, required, clocks=()):
    if not isinstance(frame,pl.DataFrame): raise ValueError('expected polars DataFrame')
    missing = set(required)-set(frame.columns)
    if missing: raise ValueError(f'missing columns: {sorted(missing)}')
    for name in clocks:
        if not frame.schema[name].is_integer() or frame[name].null_count():
            raise ValueError(f'{name} must have non-null integer schema')
    return frame.to_dicts()


def _key(row):
    key=tuple(row[k] for k in KEYS)
    if any(not isinstance(v,str) or not v for v in key):
        raise ValueError('execution identity must contain nonempty strings')
    return key


def _order(value):
    if isinstance(value,bool) or not isinstance(value,(str,Integral)) or str(value)=='':
        raise ValueError('order identity must be nonempty string/integer')
    return str(value)


def _unique(rows, key, label):
    out={}
    for row in rows:
        identity=key(row)
        if identity in out: raise ValueError(f'duplicate {label} identity')
        out[identity]=row
    return out


def _horizons(horizons_ns):
    values=[_integer(h,'horizon_ns',minimum=1) for h in horizons_ns]
    if not values or len(set(values))!=len(values):
        raise ValueError('horizons must be nonempty and unique')
    return sorted(values)


def _policies(age_policies):
    seen=set()
    for policy in age_policies:
        if not isinstance(policy,dict) or set(policy)!={'id','max_age_ns','half_life_ns'}:
            raise ValueError('age policy requires id, max_age_ns, half_life_ns')
        ident=policy['id']
        if not isinstance(ident,str) or not ident or ident in seen:
            raise ValueError('age policy identities must be nonempty and unique')
        seen.add(ident)
        if policy['max_age_ns'] is not None: _integer(policy['max_age_ns'],'max_age_ns',minimum=0)
        if policy['half_life_ns'] is not None: _integer(policy['half_life_ns'],'half_life_ns',minimum=1)
    if not seen: raise ValueError('age policies must be nonempty')


def _by_execution(candidate_info):
    """Index immutable candidate values once, preserving their original order."""
    grouped=defaultdict(list)
    for (execution_key,_),value in candidate_info.items():
        grouped[execution_key].append(value)
    return grouped


def evaluate_cells(executions, candidates, max_links, cancellations, horizons_ns, age_policies):
    """Reassign explicit native cancels AFTER each H/age eligibility filter.

    Execution rows (including role/identity/Q/D) are preserved in every cell.
    Gross U is cumulatively capped at the initial candidate quantity per order.
    Audit retains all eligible competitors; losers receive zero attributed U.
    """
    horizons=_horizons(horizons_ns)
    age_policies=list(age_policies)
    _policies(age_policies)
    eclocks=['cluster_first_sort_index','cluster_last_sort_index',
             'cluster_start_replay_time_ns','cluster_end_fill_anchor_ns']
    erows=_frame(executions,KEYS+eclocks+['execution_quantity','opposite_market_visible_qty_pre_topN','deceptive_side'],eclocks)
    cclocks=['deceptive_order_first_seen_sort_index','deceptive_order_first_seen_replay_ns']
    crows=_frame(candidates,KEYS+cclocks+['deceptive_order_id','deceptive_order_visible_qty_pre'],cclocks)
    lclocks=['cancel_sort_index','cluster_first_sort_index','cluster_last_sort_index','cluster_end_fill_anchor_ns','cancellation_removal_ns']
    lrows=_frame(max_links, ['partition_id','actor_key','execution_cluster_id','candidate_order_id']+lclocks,lclocks)
    xclocks=['sort_index','cancel_order_first_seen_sort_index','cancellation_removal_ns']
    xrows=_frame(cancellations,['partition_id','actor_key','ORDERID','side','visible_qty_pre_cancel']+xclocks,xclocks)
    eby=_unique(erows,_key,'execution')
    cby=_unique(crows,lambda r:(_key(r),_order(r['deceptive_order_id'])),'candidate')
    for e in erows:
        score_pair(0,0,e['execution_quantity'],e['opposite_market_visible_qty_pre_topN'])
        if e['deceptive_side'] not in ('bid','ask') or e['cluster_first_sort_index']>e['cluster_last_sort_index']:
            raise ValueError('invalid execution side/indexes')
    candidate_info={}
    for ck,c in cby.items():
        if ck[0] not in eby: raise ValueError('orphan candidate')
        e=eby[ck[0]]
        p=_quantity(c['deceptive_order_visible_qty_pre'],'candidate quantity')
        age=e['cluster_start_replay_time_ns']-c['deceptive_order_first_seen_replay_ns']
        age_weight(age,None)
        if c['deceptive_order_first_seen_sort_index']>=e['cluster_first_sort_index']:
            raise ValueError('candidate lifecycle must precede first fill')
        candidate_info[ck]=(p,age)
    candidate_groups=_by_execution(candidate_info)
    for ek,e in eby.items():
        total=math.fsum(p for p,_ in candidate_groups.get(ek,[]))
        score_pair(total,0,e['execution_quantity'],e['opposite_market_visible_qty_pre_topN'])
    def cancel_key(x):
        if not isinstance(x['partition_id'],str) or not x['partition_id'] or not isinstance(x['actor_key'],str) or not x['actor_key']:
            raise ValueError('invalid cancellation identity')
        return (x['partition_id'],x['sort_index'],_order(x['ORDERID']))
    xby=_unique(xrows,cancel_key,'cancellation')
    for x in xrows:
        _quantity(x['visible_qty_pre_cancel'],'cancel quantity')
        if x['side'] not in ('bid','ask') or x['cancel_order_first_seen_sort_index']>=x['sort_index']:
            raise ValueError('invalid cancellation lifecycle/side')
    local_exec=defaultdict(list)
    for ek in eby: local_exec[ek[1:]].append(ek)
    eligible_links=[]
    seen_links=set()
    for link in lrows:
        if 'instrument' in link:
            ek=_key(link)
        else:
            matches=local_exec[(link['partition_id'],link['actor_key'],link['execution_cluster_id'])]
            if len(matches)!=1: raise ValueError('link instrument cannot be resolved uniquely')
            ek=matches[0]
        ck=(ek,_order(link['candidate_order_id']))
        lk=(ck,link['cancel_sort_index'])
        if lk in seen_links: raise ValueError('duplicate link identity')
        seen_links.add(lk)
        if ek not in eby or ck not in cby: raise ValueError('orphan link')
        x=xby.get((ek[1],link['cancel_sort_index'],ck[1]))
        if x is None: raise ValueError('link cancellation not found')
        e,c=eby[ek],cby[ck]
        for field in ('cluster_first_sort_index','cluster_last_sort_index','cluster_end_fill_anchor_ns'):
            if link[field]!=e[field]: raise ValueError(f'link {field} disagrees with execution')
        if link['cancellation_removal_ns']!=x['cancellation_removal_ns']:
            raise ValueError('link cancel clock disagrees with cancellation')
        if x['actor_key']!=ek[2] or x['side']!=e['deceptive_side']:
            raise ValueError('link actor/side disagrees with cancellation')
        if 'cancel_visible_qty' in link and _quantity(link['cancel_visible_qty'],'link cancel quantity')!=x['visible_qty_pre_cancel']:
            raise ValueError('link quantity disagrees with native cancellation')
        # Reused ORDERID with a different lifecycle is not this candidate.
        if x['cancel_order_first_seen_sort_index']!=c['deceptive_order_first_seen_sort_index']:
            continue
        delay=x['cancellation_removal_ns']-e['cluster_end_fill_anchor_ns']
        if x['sort_index']<=e['cluster_last_sort_index'] or delay<0: continue
        eligible_links.append((ek,ck,x,delay))
    population=[]; audit=[]
    for h in horizons:
        for policy in age_policies:
            selected={ck:(p,age,age_weight(age,policy['half_life_ns']))
                      for ck,(p,age) in candidate_info.items()
                      if policy['max_age_ns'] is None or age<=policy['max_age_ns']}
            selected_groups=_by_execution(selected)
            rows=[]
            for ek,ck,x,delay in eligible_links:
                if ck not in selected or delay>=h: continue
                p,age,weight=selected[ck]; e=eby[ek]
                rows.append({**dict(zip(KEYS,ek)), 'candidate_order_id':ck[1],
                    'cancel_sort_index':x['sort_index'],'cluster_first_sort_index':e['cluster_first_sort_index'],
                    'cluster_last_sort_index':e['cluster_last_sort_index'],'horizon_ns':h,'age_policy':policy['id'],
                    'age_ns':age,'age_weight':weight,
                    'deceptive_order_first_seen_sort_index':cby[ck]['deceptive_order_first_seen_sort_index'],
                    'cancel_order_first_seen_sort_index':x['cancel_order_first_seen_sort_index'],
                    'candidate_visible_qty_pre':p,'cancel_visible_qty':x['visible_qty_pre_cancel']})
            assigned=assign_cancellations_to_clusters(pl.DataFrame(rows,infer_schema_length=None),use_v3_clocks=True).to_dicts() if rows else []
            used=defaultdict(float); raw_u=defaultdict(list); weighted_u=defaultdict(list)
            for r in sorted(assigned,key=lambda r:(r['cancel_sort_index'],r['execution_cluster_id'],r['partition_id'],r['actor_key'])):
                ek=_key(r); ck=(ek,r['candidate_order_id'])
                attributed=min(r['cancel_visible_qty'], max(r['candidate_visible_qty_pre']-used[ck],0.0)) if r['assigned_flag'] else 0.0
                used[ck]+=attributed
                r['attributed_cancel_visible_qty']=attributed
                r['attributed_cancel_qty_age_weighted']=attributed*r['age_weight']
                raw_u[ek].append(attributed); weighted_u[ek].append(attributed*r['age_weight'])
                audit.append(r)
            for ek,e in eby.items():
                local=selected_groups.get(ek,[])
                p=math.fsum(v[0] for v in local); pg=math.fsum(v[0]*v[2] for v in local)
                u=math.fsum(raw_u[ek]); ug=math.fsum(weighted_u[ek])
                # A last-bit summation discrepancy is not extra physical withdrawal.
                ug=min(ug,pg)
                d=e['opposite_market_visible_qty_pre_topN']
                population.append({**e,'horizon_ns':h,'age_policy':policy['id'],
                    'candidate_qty_raw':p,'withdrawal_qty_raw':u,
                    'candidate_qty_age_weighted':pg,'withdrawal_qty_age_weighted':ug,
                    'age_weight_underflow_count':sum(v[0]>0 and v[2]==0 for v in local),
                    'is_candidate':p>0 and u>0,'market_score_supported':d is not None and d>0,
                    **score_pair(pg,ug,e['execution_quantity'],d)})
    return (pl.DataFrame(population,schema={**executions.schema,**POPULATION_SCHEMA},strict=True),
            pl.DataFrame(audit,schema=AUDIT_SCHEMA,strict=True))


def _grid_config(horizon):
    # Numerical adapter ONLY: the historical cohort token is not eligibility.
    return dict(schema='msci_grid_study_v1',source_clock_policy='spoofing_detection_sequence_replay_trade_fill_sequence_cancel_v3_utc_provisional',
        grid_policy='fixed_pre_exact_post_execution_closed_grid_v1',grid_step_ns=10_000_000,lag_ns=50_000_000,
        horizon_ns=horizon,complete_grid_required=True,cohort='exact_top_positive_W_5pct_then_U_over_Q_gt_1_before_prices',
        grid_sensitivity_ns=[10_000_000],lag_sensitivity_ns=[50_000_000],lag_sensitivity_grid_ns=10_000_000,selection_threshold=None)


def profile_summaries(executions,states,coverage,horizons_ns):
    """One max grid, independently summarized and state-verified H prefixes.

    The native runner represents late clock failure as a terminal invalid grid.
    Such grids cannot yield an earlier prefix by slicing: recover only those
    prefixes through the SAME native local sampler (no alternate MSCI math).
    """
    horizons=_horizons(horizons_ns)
    trajectory,_=compute_partition_grids(executions,states,coverage,_grid_config(max(horizons)))
    groups={tuple(f.item(0,k) for k in KEYS):f for f in trajectory.partition_by(KEYS,maintain_order=True)}
    locals_=None; result=[]
    for h in horizons:
        prefix=[]; core_rows=[]
        for e in executions.iter_rows(named=True):
            grid=groups[_key(e)].filter(pl.col('grid_offset_ns')<=h)
            if h<max(horizons) and grid['baseline_sort_index'].null_count()==grid.height:
                if locals_ is None: locals_=_state_groups(states)
                grid=pl.DataFrame(_sample_local_execution(e,locals_.get((e['partition_id'],e['actor_key']),[]),coverage,_grid_config(h)),schema=TRAJECTORY_SCHEMA,strict=False)
            rich=summarize_grid(grid,lag_ns=50_000_000)
            identity={k:e[k] for k in KEYS}
            core_rows.append({**identity,**{k:rich[k] for k in SUMMARY_SCHEMA if k not in KEYS}})
            reasons=grid.filter(~pl.col('valid'))['reason'].drop_nulls().to_list()
            result.append({**identity,**rich,'horizon_ns':h,'reason':reasons[0] if reasons else None})
            prefix.append(grid)
        prefix_grid=pl.concat(prefix) if prefix else pl.DataFrame(schema=TRAJECTORY_SCHEMA)
        summary=pl.DataFrame(core_rows,schema=SUMMARY_SCHEMA,strict=True)
        verify_partition_grids(executions,prefix_grid,summary,_grid_config(h),states=states,coverage=coverage)
    return pl.DataFrame(result,schema=PROFILE_SCHEMA,strict=True)


def add_priority_flags(population,fraction):
    """Matched review budgets; retain raw candidates, null J and J=0 rows.

    J is ranked among complete finite-J members of the corresponding W list.
    Each stage selects ceil(n*fraction); identity tie-breaks are deterministic.
    """
    fraction=_quantity(fraction,'fraction')
    if not 0<fraction<=1: raise ValueError('fraction must be in (0,1]')
    rows=_frame(population,KEYS+['horizon_ns','age_policy','is_candidate','WMSCI_self','WMSCI_market','J','complete_grid'],['horizon_ns'])
    for flag in ('is_candidate','complete_grid'):
        if population.schema[flag]!=pl.Boolean or population[flag].null_count():
            raise ValueError(f'{flag} must be non-null Boolean')
    _unique(rows,lambda r:(_key(r),r['horizon_ns'],r['age_policy']),'population')
    flags={f'priority_top_{s}': [False]*len(rows) for s in ('w_self','w_market','j_given_self','j_given_market')}
    groups=defaultdict(list)
    for i,r in enumerate(rows):
        if not isinstance(r['age_policy'],str) or not r['age_policy']: raise ValueError('invalid age policy identity')
        groups[(r['instrument'],r['horizon_ns'],r['age_policy'])].append(i)
    def finite(v): return isinstance(v,Real) and not isinstance(v,bool) and math.isfinite(v)
    def rank(indices,field):
        return sorted(indices,key=lambda i:(-rows[i][field],rows[i]['execution_cluster_id'],rows[i]['partition_id'],rows[i]['actor_key']))
    for indices in groups.values():
        for suffix in ('self','market'):
            field=f'WMSCI_{suffix}'
            eligible=[i for i in indices if rows[i]['is_candidate'] and finite(rows[i][field]) and rows[i][field]>0]
            top=rank(eligible,field)[:math.ceil(len(eligible)*fraction)]
            for i in top: flags[f'priority_top_w_{suffix}'][i]=True
            js=[i for i in top if rows[i]['complete_grid'] and finite(rows[i]['J']) and rows[i]['J']>=0]
            for i in rank(js,'J')[:math.ceil(len(js)*fraction)]: flags[f'priority_top_j_given_{suffix}'][i]=True
    return population.with_columns([pl.Series(k,v,dtype=pl.Boolean) for k,v in flags.items()])
