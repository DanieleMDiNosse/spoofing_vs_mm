#!/usr/bin/env python
"""Render the validated v3 descriptive-control summary for the paper."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / 'src'):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts import run_empirical_controls_v2_full as workflow
from spoofing_detection.lob.empirical_controls_v2_io import file_hash

INSTRUMENTS = ('FERRARI', 'NEXI', 'RISANAMENTO')
IDENTITIES = ('client_original', 'firm')
MODES = ('aggressive', 'passive')
PUBLICATION_FILES = ('spoofing_empirical_controls.tex', 'spoofing_empirical_controls_metadata.json')


def require_new_publication_targets(output_dir: Path) -> None:
    for name in PUBLICATION_FILES:
        if (output_dir / name).exists():
            raise FileExistsError(f'existing paper artifact must not be overwritten: {name}')


def render_table(summary: pl.DataFrame) -> str:
    """Show the observed-full strata only, with explicit support and null status."""
    expected = {(instrument, identity, mode) for instrument in INSTRUMENTS
                for identity in IDENTITIES for mode in MODES}
    observed = summary.filter((pl.col('analysis_kind') == 'observed_full')
                              & (pl.col('offset_seconds') == 0))
    keys = [(row['instrument'], row['identity_level'], row['comparison_mode'])
            for row in observed.iter_rows(named=True)]
    if len(keys) != 12 or set(keys) != expected:
        raise ValueError('observed-full summary requires exactly 12 unique strata')
    indexed = {key: row for key, row in zip(keys, observed.iter_rows(named=True))}
    lines = [
        r'\begin{table}[!htbp]',
        r'\centering',
        r'\caption{Descriptive difference in eligible opposite-side physical cancellation intensity '
        r'(cancellations/posture-second) between observed own-execution windows and '
        r'periods with no observed attributed own execution. Group count and weight are '
        r'aggregation support, not independent samples or confidence intervals. '
        r'Client and firm-fallback strata must not be pooled.}',
        r'\label{tab:empirical_controls_v3}',
        r'\begin{tabular}{lllrrr}',
        r'\toprule',
        r'Instrument & Identity & Execution role & Groups & Posture-seconds & Difference \\',
        r'\midrule',
    ]
    for instrument in INSTRUMENTS:
        for identity in IDENTITIES:
            for mode in MODES:
                row = indexed[instrument, identity, mode]
                weight, value, status = (row['weight_seconds'],
                                         row['weighted_intensity_difference'], row['summary_status'])
                if status == 'estimable' and weight is not None and weight > 0 and value is not None:
                    display = f'{value:.3f}'
                elif status == 'not_estimable' and weight == 0 and value is None:
                    display = 'not estimable'
                else:
                    raise ValueError('inconsistent empirical-control status or support')
                display_identity = identity.replace('_', r'\_')
                lines.append(f"{instrument} & {display_identity} & {mode} & {row['group_count']} & "
                             f"{weight:.1f} & {display} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines) + '\n'


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--combined', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.combined / 'manifest.json').read_text())
    workflow._validate_combined_output(args.combined, required_instruments=set(INSTRUMENTS))
    rendered = render_table(pl.read_parquet(args.combined / 'summary.parquet'))
    sources = {instrument: {
        'bundle': entry['bundle'],
        'manifest_sha256': file_hash(Path(entry['bundle']) / 'manifest.json'),
    } for instrument, entry in sorted(manifest['sources'].items())}
    require_new_publication_targets(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / 'spoofing_empirical_controls.tex').write_text(rendered)
    (args.output_dir / 'spoofing_empirical_controls_metadata.json').write_text(
        json.dumps({'clock_policy_id': manifest['clock_policy_id'],
                    'combined_manifest_sha256': file_hash(args.combined / 'manifest.json'),
                    'sources': sources}, indent=2, sort_keys=True) + '\n')


if __name__ == '__main__':
    main()
