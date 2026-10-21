"""Read-only tests of cancel timestamp semantics; no replay changes."""
import json
from pathlib import Path
import hashlib
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/cancel_clock_field_audit'

def main():
    OUT.mkdir(exist_ok=False)
    config = json.loads((ROOT/'configs/spoofing_empirical_controls.json').read_text())
    result = []
    for instrument, run in config['baseline_runs'].items():
        path = ROOT / run['source_paths']['raw_events']
        df = pl.read_parquet(path)
        keys = ['TRADEDATE', 'SYMBOLINDEX', 'EMM (*)', 'ORDERID']
        df = df.with_columns([pl.col(c).str.strptime(pl.Datetime('ns'), format='%Y%m%d %H:%M:%S%.f', strict=False).alias(c+'_parsed') for c in ['BOOKIN','BOOKOUTTIME','SEQUENCETIME']])
        cancel = df.filter(pl.col('ORDEREVENTTYPE (*)') == 4)
        new = df.filter(pl.col('ORDEREVENTTYPE (*)') == 1)
        # Exclude order keys with repeated New messages to avoid ambiguous lifecycles.
        initial = new.group_by(keys).agg(pl.len().alias('new_count'),pl.col('BOOKIN_parsed').first().alias('initial_bookin'),pl.col('SEQUENCETIME_parsed').first().alias('initial_sequence')).filter(pl.col('new_count')==1)
        joined = cancel.join(initial,on=keys,how='inner').filter(pl.col('SEQUENCETIME_parsed')>=pl.col('initial_sequence'))
        def compare(frame, a, b):
            f=frame.filter(pl.col(a).is_not_null() & pl.col(b).is_not_null())
            delta=f.select((pl.col(a).cast(pl.Int64)-pl.col(b).cast(pl.Int64)).alias('ns'))['ns']
            return {'n':f.height,'equal':int((delta==0).sum()),'later':int((delta>0).sum()),'earlier':int((delta<0).sum()),'median_delta_ns':delta.median(),'p99_delta_ns':delta.quantile(.99)}
        r={'instrument':instrument,'raw_rows':df.height,'cancel_rows':cancel.height,'unique_new_matched_cancel_rows':joined.height,
           'cancel_bookin_vs_initial_bookin':compare(joined,'BOOKIN_parsed','initial_bookin'),
           'cancel_bookout_vs_bookin':compare(cancel,'BOOKOUTTIME_parsed','BOOKIN_parsed'),
           'cancel_sequence_vs_bookout':compare(cancel,'SEQUENCETIME_parsed','BOOKOUTTIME_parsed'),
           'cancel_sequence_vs_bookin':compare(cancel,'SEQUENCETIME_parsed','BOOKIN_parsed'),
           'nulls':{c:cancel[c+'_parsed'].null_count() for c in ['BOOKIN','BOOKOUTTIME','SEQUENCETIME']},
           'parse_failures':{c:df.filter(pl.col(c).is_not_null() & pl.col(c+'_parsed').is_null()).height for c in ['BOOKIN','BOOKOUTTIME','SEQUENCETIME']},
           'source':str(path),'source_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        # Anonymous examples: no client/order identifiers exported.
        joined.select('initial_bookin','BOOKIN_parsed','BOOKOUTTIME_parsed','SEQUENCETIME_parsed').head(10).write_csv(OUT/f'{instrument}_examples.csv')
        result.append(r)
        (OUT/'results.json').write_text(json.dumps(result,indent=2))
        print(json.dumps(r,indent=2))
    (OUT/'provenance.json').write_text(json.dumps({'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'polars':pl.__version__,'note':'All Cancel rows; not restricted to client-initiated cancels. Unique-New join excludes repeated New keys; requires cancel sequence >= New sequence. Differences parsed at nanosecond precision; timezone not converted.'},indent=2))

if __name__=='__main__':
    main()
