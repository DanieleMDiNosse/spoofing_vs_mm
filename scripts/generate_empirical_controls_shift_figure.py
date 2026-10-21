#!/usr/bin/env python
"""Plot the validated common-support timing shifts without pooling identity scopes."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import polars as pl

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_empirical_controls_v2_full as workflow
from spoofing_detection.lob.empirical_controls_v2_io import file_hash

INSTRUMENTS = ('NEXI', 'FERRARI', 'RISANAMENTO')
IDENTITIES = ('client_original', 'firm')
MODES = ('passive', 'aggressive')
OFFSETS = (-60, -30, 0, 30, 60)
FILES = ('spoofing_empirical_controls_shift.pdf', 'spoofing_empirical_controls_shift_metadata.json')


def require_new_targets(output_dir: Path) -> None:
    for name in FILES:
        if (output_dir / name).exists():
            raise FileExistsError(f'existing paper artifact must not be overwritten: {name}')


def validate_shift_rows(summary: pl.DataFrame) -> pl.DataFrame:
    """Reject missing, duplicate or incomparable offset strata before plotting."""
    frame = summary.filter(pl.col('analysis_kind') == 'timing_shift_common_support')
    expected = {(instrument, identity, mode, offset)
                for instrument in INSTRUMENTS for identity in IDENTITIES
                for mode in MODES for offset in OFFSETS}
    keys = [(r['instrument'], r['identity_level'], r['comparison_mode'], r['offset_seconds'])
            for r in frame.iter_rows(named=True)]
    if len(keys) != 60 or set(keys) != expected:
        raise ValueError('timing shift plot requires 60 unique strata')
    for instrument in INSTRUMENTS:
        for identity in IDENTITIES:
            for mode in MODES:
                rows = frame.filter((pl.col('instrument') == instrument)
                                    & (pl.col('identity_level') == identity)
                                    & (pl.col('comparison_mode') == mode))
                if (rows['summary_status'].unique().to_list() != ['estimable']
                        or rows['group_count'].n_unique() != 1
                        or rows['weight_seconds'].n_unique() != 1
                        or rows['group_count'].item(0) <= 0
                        or rows['weight_seconds'].item(0) <= 0
                        or not all(math.isfinite(v) for v in rows['weighted_intensity_difference'])):
                    raise ValueError('timing shift plot requires finite estimates on common support')
    return frame


def render_figure(frame: pl.DataFrame):
    fig, axes = plt.subplots(3, 2, figsize=(7.2, 7.3))
    for row, instrument in enumerate(INSTRUMENTS):
        for col, identity in enumerate(IDENTITIES):
            ax = axes[row, col]
            for mode, color, marker in (
                ('passive', '#2878b5', 'o'),
                ('aggressive', '#d97904', '^'),
            ):
                series = frame.filter((pl.col('instrument') == instrument)
                                      & (pl.col('identity_level') == identity)
                                      & (pl.col('comparison_mode') == mode)).sort('offset_seconds')
                ax.plot(series['offset_seconds'].to_list(),
                        series['weighted_intensity_difference'].to_list(),
                        color=color, marker=marker, markersize=4.5, linewidth=1.5,
                        label=mode.title(), zorder=3)
            ax.axhline(0, color='#8b8b8b', linewidth=0.7, zorder=1)
            ax.axvline(0, color='#aaaaaa', linestyle=':', linewidth=0.8, zorder=1)
            ax.set_title(f'{instrument.title()} · {"Original client" if col == 0 else "Firm fallback"}',
                         fontsize=10, pad=6)
            ax.set_xticks(OFFSETS)
            ax.set_xlim(-64, 64)
            ax.tick_params(labelsize=8.5, length=3)
            ax.grid(axis='y', color='#e6e6e6', linewidth=0.5)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    fig.legend(axes[0, 0].lines[:2], ('Passive fill', 'Aggressive execution'),
               loc='upper center', bbox_to_anchor=(0.55, 0.992), ncol=2,
               frameon=False, fontsize=9.5)
    fig.supxlabel('Execution-window shift (seconds)', fontsize=10, y=0.025)
    fig.supylabel('Observed-window minus no-own-observed intensity\n(cancellations/posture-second)',
                 fontsize=9.5, x=0.005)
    fig.subplots_adjust(left=0.15, right=0.98, top=0.925, bottom=0.10,
                        hspace=0.45, wspace=0.28)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--combined', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    workflow._validate_combined_output(args.combined, required_instruments=set(INSTRUMENTS))
    frame = validate_shift_rows(pl.read_parquet(args.combined / 'summary.parquet'))
    require_new_targets(args.output_dir)
    fig = render_figure(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        fig.savefig(args.output_dir / FILES[0], format='pdf',
                    metadata={'CreationDate': datetime(2000, 1, 1, tzinfo=timezone.utc),
                              'Creator': 'generate_empirical_controls_shift_figure.py'})
    finally:
        plt.close(fig)
    manifest = json.loads((args.combined / 'manifest.json').read_text())
    (args.output_dir / FILES[1]).write_text(json.dumps({
        'combined_manifest_sha256': file_hash(args.combined / 'manifest.json'),
        'combined_summary_sha256': file_hash(args.combined / 'summary.parquet'),
        'figure_sha256': file_hash(args.output_dir / FILES[0]),
        'sources': {name: source['manifest_sha256']
                    for name, source in sorted(manifest['sources'].items())},
        'analysis_kind': 'timing_shift_common_support',
        'offset_seconds': list(OFFSETS),
        'panel_scales': 'independent; do not compare heights across panels',
    }, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
