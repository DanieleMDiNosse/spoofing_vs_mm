"""Read-only cached-risk support audit; no detector rerun or outcome tuning.

Independent integer-microsecond duration accounting, with baseline reconciliation.
Run with project Python; output directory must not exist.
"""
from __future__ import annotations
import argparse
import bisect
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import polars as pl
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
US = 1_000_000
H = 2 * US
STATES = ['no_own_observed', 'own_passive_only', 'own_aggressive_only', 'own_mixed']
BUNDLES = {
    'NEXI': ROOT / 'outputs/empirical_controls_v2_pilot/NEXI_2024-04-23',
    'FERRARI': ROOT / 'outputs/empirical_controls_v2_full/bundles/FERRARI',
    'RISANAMENTO': ROOT / 'outputs/empirical_controls_v2_full/bundles/RISANAMENTO',
}


def load(folder: Path) -> pl.DataFrame:
    return pl.read_parquet(sorted(folder.glob('*.parquet')))


def numeric(df: pl.DataFrame, cols: list[str]) -> pl.DataFrame:
    return df.with_columns([pl.col(c).dt.cast_time_unit('us').cast(pl.Int64) for c in cols])


def durations(lo: int, hi: int, passive: list[int], aggressive: list[int], offset: int = 0) -> np.ndarray:
    """Union overlapping windows; four disjoint states, exact integer durations."""
    out = np.zeros(4, dtype=np.int64)
    if hi <= lo:
        return out
    events = defaultdict(lambda: [0, 0])
    events[lo]; events[hi]
    for mode, anchors in enumerate((passive, aggressive)):
        a = bisect.bisect_right(anchors, lo - offset - H)
        b = bisect.bisect_left(anchors, hi - offset)
        for anchor in anchors[a:b]:
            left, right = max(lo, anchor + offset), min(hi, anchor + offset + H)
            if right > left:
                events[left][mode] += 1
                events[right][mode] -= 1
    active = [0, 0]
    last = lo
    for point in sorted(events):
        state = int(active[0] > 0) + 2 * int(active[1] > 0)
        out[state] += point - last
        active[0] += events[point][0]
        active[1] += events[point][1]
        last = point
    assert out.sum() == hi - lo
    assert np.all(out >= 0)
    return out


def self_test() -> None:
    assert durations(0, 10 * US, [US, 2 * US], []).tolist() == [7 * US, 3 * US, 0, 0]
    assert durations(0, 4 * US, [0], [US]).tolist() == [US, US, US, US]
    assert durations(0, H, [], []).tolist() == [H, 0, 0, 0]
    assert durations(0, H, [0], [], 10 * US).tolist() == [H, 0, 0, 0]


def audit(out: Path) -> None:
    self_test()
    out.mkdir(parents=True, exist_ok=False)
    started = time.time()
    configs = [(b, m) for b in (30, 60, 120) for m in (60, 20)]
    totals = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    contrasts = defaultdict(lambda: {'groups': 0, 'weight_us': 0, 'actor_days': set(), 'actor_weights': defaultdict(int)})
    funnel, epochs_out, reconciliation, examples = [], [], [], []
    manifest_hashes = {}
    partition_count = 0
    for instrument, root in BUNDLES.items():
        manifest_hashes[instrument] = hashlib.sha256((root / 'manifest.json').read_bytes()).hexdigest()
        for part in sorted((root / 'partitions').iterdir()):
            if not part.is_dir():
                continue
            partition_count += 1
            cov = numeric(load(part / 'risk/coverage_epochs'), ['start_ts', 'end_ts'])
            epochs = {r['coverage_epoch_id']: (r['start_ts'], r['end_ts']) for r in cov.to_dicts()}
            ordered_epochs = sorted(epochs.values())
            epoch_starts = [a for a, b in ordered_epochs]
            for r in cov.to_dicts():
                epochs_out.append({'instrument': instrument, 'partition': part.name, 'duration_seconds': (r['end_ts']-r['start_ts']) / US, 'end_reason': r['end_reason']})
            clusters = numeric(pl.read_parquet(part / 'execution_clusters.parquet'), ['cluster_end_ts'])
            schedules = defaultdict(lambda: [[], []])
            counts = defaultdict(lambda: defaultdict(int))
            for r in clusters.to_dicts():
                level, anchor = r['identity_level'], r['cluster_end_ts']
                counts[level]['executions'] += 1
                i = bisect.bisect_right(epoch_starts, anchor) - 1
                if i >= 0 and ordered_epochs[i][0] <= anchor < ordered_epochs[i][1]:
                    counts[level]['anchors_in_coverage'] += 1
                    if ordered_epochs[i][1] - anchor >= H:
                        counts[level]['anchors_with_2s_future_coverage'] += 1
                side = 'ask' if r['event_side'] == 'bid' else 'bid'
                key = (r['actor_key'], r['identity_level'], r['identity_source'], r['identity_fallback_flag'], side)
                schedules[key][int(r['execution_anchor_mode'] == 'aggressive')].append(anchor)
            for windows in schedules.values():
                for w in windows:
                    w.sort()
            risk = numeric(load(part / 'risk/intervals'), ['start_ts', 'end_ts'])
            groups = defaultdict(lambda: np.zeros((6, 4), dtype=np.int64))
            raw_times = defaultdict(lambda: np.zeros(4, dtype=np.int64))
            uncut_times = defaultdict(lambda: np.zeros(4, dtype=np.int64))
            for r in risk.iter_rows(named=True):
                start, end = r['start_ts'], r['end_ts']
                if end <= start:
                    continue
                ident = (r['actor_key'], r['identity_level'], r['identity_source'], r['identity_fallback_flag'], r['side'])
                level = r['identity_level']
                windows = schedules.get(ident, [[], []])
                raw_times[level] += durations(start, end, *windows)
                a, b = epochs[r['coverage_epoch_id']]
                epoch_windows = [w[bisect.bisect_left(w, a):bisect.bisect_left(w, b)] for w in windows]
                uncut_times[level] += durations(max(start, a), min(end, b), *epoch_windows)
                for band in (30, 60, 120):
                    width = band * 60 * US
                    cursor = (max(start, a) // width) * width
                    while cursor < min(end, b):
                        block_a, block_b = max(a, cursor), min(b, cursor + width)
                        block_windows = [w[bisect.bisect_left(w, block_a):bisect.bisect_left(w, block_b)] for w in epoch_windows]
                        for margin in (60, 20):
                            key = (band, margin, *ident, cursor)
                            mat = groups[key]
                            lo, hi = max(start, block_a + H), min(end, block_b)
                            mat[0] += durations(lo, hi, *block_windows)
                            lo, hi = max(start, block_a + H + margin*US), min(end, block_b - margin*US)
                            for j, offset in enumerate((-margin, -margin//2, 0, margin//2, margin)):
                                mat[j+1] += durations(lo, hi, *block_windows, offset*US)
                        cursor += width
            for level, c in counts.items():
                funnel.append({'instrument': instrument, 'partition': part.name, 'identity_level': level,
                    **{name: c[name] for name in ('executions', 'anchors_in_coverage', 'anchors_with_2s_future_coverage')},
                    'risk_seconds': float(raw_times[level].sum()/US),
                    'overlap_ignoring_epoch_anchor_rule_seconds': float(raw_times[level][1:].sum()/US),
                    'overlap_same_epoch_before_history_cut_seconds': float(uncut_times[level][1:].sum()/US)})
            for key, mat in groups.items():
                band, margin, actor, level, source, fallback, side, bucket = key
                for i in range(6):
                    totals[(instrument, level, band, margin, i)] += mat[i]
                for mode, column in [('passive', 1), ('aggressive', 2)]:
                    for kind, indices in [('observed', [0]), ('shift_common', list(range(1, 6)))]:
                        valid = all(mat[i, column] > 0 and mat[i, 0] > 0 for i in indices)
                        if valid:
                            target = contrasts[(instrument, level, band, margin, mode, kind)]
                            weight = int(min(mat[0 if kind == 'observed' else 3, column], mat[0 if kind == 'observed' else 3, 0]))
                            target['groups'] += 1
                            target['weight_us'] += weight
                            target['actor_days'].add((part.name, actor))
                            target['actor_weights'][(part.name, actor)] += weight
            original = pl.read_parquet(root / 'state_statistics.parquet').filter((pl.col('analysis_kind') == 'observed_full') & (pl.col('partition_id') == clusters['partition_id'][0])) if clusters.height else None
            if original is not None:
                expected = original.group_by('identity_level', 'state').agg(pl.col('time_at_risk_seconds').sum())
                for row in expected.to_dicts():
                    level, state = row['identity_level'], row['state']
                    actual = sum(int(mat[0, STATES.index(state)]) for key, mat in groups.items() if key[0:2] == (30, 60) and key[3] == level) / US
                    error = actual - row['time_at_risk_seconds']
                    reconciliation.append({'instrument': instrument, 'partition': part.name, 'identity_level': level, 'state': state, 'published_seconds': row['time_at_risk_seconds'], 'audit_seconds': actual, 'difference_seconds': error})
            pl.DataFrame(funnel).write_csv(out/'funnel.csv')
            pl.DataFrame(reconciliation).write_csv(out/'reconciliation.csv')
            print(instrument, part.name, 'done', flush=True)
    rows = []
    for (inst, level, band, margin, i), values in sorted(totals.items()):
        row = dict(instrument=inst, identity_level=level, band_minutes=band, max_offset_seconds=margin, analysis='observed' if i == 0 else 'shift', offset_seconds=0 if i == 0 else (-margin, -margin//2, 0, margin//2, margin)[i-1])
        rows.append({**row, **{state+'_seconds': float(values[j]/US) for j, state in enumerate(STATES)}})
    pl.DataFrame(rows).write_csv(out/'state_support.csv')
    rows = []
    for inst in BUNDLES:
        for level in ('client_original', 'firm'):
            for band, margin in configs:
                for mode in ('passive', 'aggressive'):
                    for kind in ('observed', 'shift_common'):
                        x = contrasts[(inst, level, band, margin, mode, kind)]
                        rows.append(dict(instrument=inst, identity_level=level, band_minutes=band, max_offset_seconds=margin, mode=mode, analysis=kind, groups=x['groups'], actor_days=len(x['actor_days']), weight_seconds=x['weight_us']/US, largest_actor_day_weight_share=max(x['actor_weights'].values(), default=0)/x['weight_us'] if x['weight_us'] else None))
    pl.DataFrame(rows).write_csv(out/'contrast_support.csv')
    pl.DataFrame(epochs_out).write_csv(out/'coverage_epochs.csv')
    errors = [abs(x['difference_seconds']) for x in reconciliation]
    provenance = {'partition_count': partition_count, 'manifest_sha256': manifest_hashes, 'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'python': platform.python_version(), 'polars': pl.__version__, 'numpy': np.__version__, 'matplotlib': matplotlib.__version__, 'elapsed_seconds': time.time()-started, 'max_baseline_duration_error_seconds': max(errors, default=0), 'synthetic_duration_checks': 'passed', 'notes': 'Support-only independent accounting. Integer microseconds. No outcomes used in grid. Ignoring epoch anchor rule is diagnostic only, not a valid alternative estimate. Top10/90s/2s unchanged.'}
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2))
    assert partition_count == 132, partition_count
    assert max(errors, default=0) < 0.01, 'Baseline reconciliation failed; inspect before interpreting grid'
    plots(out)


def plots(out: Path) -> None:
    plt.rcParams.update({'font.size': 12})
    df = pl.read_csv(out/'state_support.csv')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, inst in zip(axes, BUNDLES):
        for level, label in [('client_original', 'Cliente'), ('firm', 'Firm fallback')]:
            d = df.filter((pl.col('instrument')==inst)&(pl.col('identity_level')==level)&(pl.col('analysis')=='observed')&(pl.col('max_offset_seconds')==20)).sort('band_minutes')
            y = d.select(pl.sum_horizontal('own_passive_only_seconds','own_aggressive_only_seconds','own_mixed_seconds')).to_series().to_numpy()
            ax.plot(d['band_minutes'], y, 'o-', label=label)
        ax.set_title(inst); ax.set_xlabel('Fascia (minuti)'); ax.set_xticks([30,60,120]); ax.grid(alpha=.25)
    axes[0].set_ylabel('Esposizione osservata (postura-secondi)'); axes[-1].legend()
    fig.suptitle('Il tempo post-esecuzione cresce ampliando le fasce?\nUnione degli stati passivo, aggressivo e misto; rischio top-10 / 90 s')
    fig.tight_layout(); fig.savefig(out/'01_exposure_by_band.png', dpi=180); plt.close(fig)
    support = pl.read_csv(out/'contrast_support.csv')
    fig, axes = plt.subplots(1,3,figsize=(15,5))
    for ax, inst in zip(axes, BUNDLES):
        for margin in (60,20):
            d=support.filter((pl.col('instrument')==inst)&(pl.col('analysis')=='shift_common')&(pl.col('max_offset_seconds')==margin)).group_by('band_minutes').agg(pl.col('groups').sum()).sort('band_minutes')
            ax.plot(d['band_minutes'],d['groups'],'o-',label=f'Offset fino a ±{margin} s')
        ax.set_title(inst);ax.set_xlabel('Fascia (minuti)');ax.set_xticks([30,60,120]);ax.grid(alpha=.25)
    axes[0].set_ylabel('Gruppi-contrasto con supporto a tutti gli offset');axes[-1].legend()
    fig.suptitle('Fattibilità degli shift (non numerosità di soggetti indipendenti)')
    fig.tight_layout();fig.savefig(out/'02_shift_support.png',dpi=180);plt.close(fig)
    f=pl.read_csv(out/'funnel.csv').group_by('instrument').agg(pl.col('executions').sum(),pl.col('anchors_in_coverage').sum(),pl.col('anchors_with_2s_future_coverage').sum())
    fig,ax=plt.subplots(figsize=(10,6));x=np.arange(3)
    for j,col in enumerate(['executions','anchors_in_coverage','anchors_with_2s_future_coverage']):
        vals=[f.filter(pl.col('instrument')==i)[col][0] for i in BUNDLES]
        bars=ax.bar(x+(j-1)*.25,vals,.25,label=['Esecuzioni disponibili','Ancora in copertura valida','Ancora con 2 s successivi coperti'][j]);ax.bar_label(bars,fontsize=9)
    ax.set_xticks(x,list(BUNDLES));ax.set_ylabel('Esecuzioni canoniche (conteggio)');ax.set_title('La copertura del clock attorno alle esecuzioni');ax.legend();fig.tight_layout();fig.savefig(out/'03_anchor_coverage.png',dpi=180);plt.close(fig)


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();audit(args.output)
