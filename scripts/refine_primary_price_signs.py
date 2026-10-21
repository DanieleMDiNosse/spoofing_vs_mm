#!/usr/bin/env python3
"""Roundoff-aware sign accounting; preserve raw primary prices and memberships."""
from pathlib import Path
import argparse
import importlib.util
import json
import math
import shutil
import polars as pl

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'outputs/primary_kernel30_h10_prices_20261014_213056'
OUTPUT=ROOT/'outputs/primary_kernel30_h10_prices_20261014_213056_numeric_v2'
KEYS=['instrument','partition_id','actor_key','execution_cluster_id']


def source_module():
    spec=importlib.util.spec_from_file_location('original_primary_prices',Path(__file__).with_name('study_primary_kernel_prices.py'))
    assert spec is not None and spec.loader is not None
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod


def finite(x):
    return x is not None and math.isfinite(x)


def endpoint_tolerance(a,b,factor):
    if not finite(a) or not finite(b):return None
    return factor*max(math.ulp(a),math.ulp(b))


def annotate(executions,cancels,factor=8):
    if type(factor) is not int or factor<0:raise ValueError('ULP factor must be a nonnegative integer')
    if executions.select(KEYS).is_duplicated().any():raise ValueError('duplicate execution key')
    groups={}
    for row in cancels.iter_rows(named=True):groups.setdefault(tuple(row[k] for k in KEYS),[]).append(row)
    tolerances={m:[] for m in ['FPM','REV','ADV']};signs={m:[] for m in tolerances}
    for row in executions.iter_rows(named=True):
        bounds={'FPM':endpoint_tolerance(row['posture_mid'],row['pre_mid'],factor),
                'ADV':endpoint_tolerance(row['posture_mid'],row['execution_vwap'],factor),'REV':None}
        if finite(row['REV']):
            assigned=groups.get(tuple(row[k] for k in KEYS),[])
            if not row['REV_complete'] or not assigned:raise ValueError('finite complete REV requires assigned cancellations')
            quantities=[];weighted_bounds=[]
            for c in assigned:
                q=c['raw_quantity_weight'];bound=endpoint_tolerance(c['cancel_pre_mid'],c['cancel_post_mid'],factor)
                if not finite(q) or q<=0 or bound is None or not finite(c['REV_cancel']):
                    raise ValueError('complete REV has an unsupported positive-quantity cancellation')
                quantities.append(q);weighted_bounds.append(q*bound)
            bounds['REV']=math.fsum(weighted_bounds)/math.fsum(quantities)+factor*math.ulp(row['REV'])
        for metric in tolerances:
            value=row[metric];bound=bounds[metric]
            if finite(value) and bound is None:raise ValueError('finite price diagnostic lacks finite endpoint scale')
            sign=None if not finite(value) else (0 if abs(value)<=bound else (1 if value>0 else -1))
            tolerances[metric].append(bound);signs[metric].append(sign)
    return executions.with_columns(*[pl.Series(m+'_zero_tolerance',v,dtype=pl.Float64) for m,v in tolerances.items()],
                                   *[pl.Series(m+'_sign',v,dtype=pl.Int8) for m,v in signs.items()])


def summarize(annotated,roles=False):
    # Raw prices stay in the private ledger. Only numerical equality accounting
    # (including medians) uses zero for values within the endpoint roundoff bound.
    normalized=annotated.with_columns(*[
        pl.when(pl.col(m+'_sign')==0).then(pl.lit(0.)).otherwise(pl.col(m)).alias(m)
        for m in ['FPM','REV','ADV']])
    return source_module().summary(normalized,roles=roles)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args();out=args.output
    if out.exists():raise FileExistsError(out)
    src=source_module();src.verify_manifest(SOURCE)
    old=src.load_json(SOURCE/'metadata.json')
    for name,h in old['input_bindings'].items():src.checked_hash(name,h)
    e=pl.read_parquet(SOURCE/'private_execution_prices.parquet');c=pl.read_parquet(SOURCE/'private_cancel_prices.parquet')
    annotated=annotate(e,c);summary=summarize(annotated)
    sensitivity=[]
    for factor in [0,4,8,16,32]:
        s=summarize(annotate(e,c,factor)).with_columns(pl.lit(factor).alias('ulp_factor'))
        sensitivity.append(s)
    sensitivity=pl.concat(sensitivity)
    for col in e.columns:
        assert e[col].equals(annotated[col],null_equal=True),col
    metadata={**old,'version':'primary_kernel_prices_numeric_signs_v2',
              'sign_policy':{'name':'endpoint_scaled_ulp_roundoff_v1','ulp_factor':8,
                  'endpoint_bound':'8*max(ulp(price_a),ulp(price_b)) for FPM/ADV and each cancellation price difference',
                  'REV_bound':'raw-quantity-weighted cancellation endpoint bounds plus 8*ulp(raw aggregate REV)',
                  'zero':'abs(raw_value)<=bound; nonfinite remains missing; no economic minimum movement or price selection',
                  'raw_values':'FPM,REV,ADV unchanged in private execution ledger; *_sign and *_zero_tolerance appended',
                  'sensitivity':'strict signs (factor0), 4,8,16,32; all groups retained in sign_sensitivity.csv'}}
    bindings=metadata['input_bindings']
    for name in ['COMPLETE.json','artifact_manifest.json',*src.load_json(SOURCE/'artifact_manifest.json')['artifacts']]:
        bindings[str(SOURCE/name)]=src.sha256(SOURCE/name)
    bindings[str(Path(__file__).resolve())]=src.sha256(__file__)
    out.mkdir(parents=True)
    annotated.write_parquet(out/'private_execution_prices.parquet')
    for name in ['private_cancel_prices.parquet','historical_reference_parity.csv']:shutil.copy2(SOURCE/name,out/name)
    summary.write_csv(out/'summary.csv');summarize(annotated,roles=True).write_csv(out/'role_identity_summary.csv')
    sensitivity.write_csv(out/'sign_sensitivity.csv');src.dump_json(out/'metadata.json',metadata)
    artifacts={p.name:src.sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    src.dump_json(out/'artifact_manifest.json',{'version':metadata['version'],'artifacts':artifacts})
    for name,h in artifacts.items():src.checked_hash(out/name,h)
    src.dump_json(out/'.COMPLETE.pending',{'status':'primary_prices_verified','artifact_manifest_sha256':src.sha256(out/'artifact_manifest.json')})
    (out/'.COMPLETE.pending').replace(out/'COMPLETE.json');src.verify_manifest(out)
    print(summary.select('instrument','ranking','stage','n','FPM_positive','FPM_zero','REV_positive','REV_zero','ADV_positive','ADV_zero','both_positive_complete'))
    print('COMPLETE',out)

if __name__=='__main__':main()
