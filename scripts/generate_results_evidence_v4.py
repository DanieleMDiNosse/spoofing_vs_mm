#!/usr/bin/env python3
"""Render frozen complete-ranking CSV evidence; no replay or analytical changes.

Fresh output directories only. Source manifests are authenticated before reading.
All observed entities, including zero-candidate groups, are exported in curve order.
Case panels contain only native horizon observations, never invented event paths.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.ticker import PercentFormatter, ScalarFormatter

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'paper/generated/withdrawal_rank_20261015_participants_v3'
OUTPUT = ROOT / 'paper/generated/results_evidence_20261019_v4'
INSTRUMENTS = ['FERRARI', 'NEXI', 'RISANAMENTO']
COLORS = ['#0072B2', '#D55E00', '#009E73']
RC = {'font.family': 'DejaVu Sans', 'font.size': 12, 'axes.labelsize': 12,
      'axes.titlesize': 12, 'xtick.labelsize': 11, 'ytick.labelsize': 11,
      'legend.fontsize': 11, 'pdf.fonttype': 42, 'axes.spines.top': False,
      'axes.spines.right': False, 'savefig.dpi': 160}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(name, source=SOURCE):
    with (source / name).open(newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def authenticate(source=SOURCE):
    manifest = json.loads((source / 'artifact_manifest.json').read_text())
    complete = json.loads((source / 'COMPLETE.json').read_text())
    if complete['artifact_manifest_sha256'] != digest(source / 'artifact_manifest.json'):
        raise ValueError('source completion binding mismatch')
    for name, expected in manifest['artifacts'].items():
        if digest(source / name) != expected:
            raise ValueError('source artifact changed: ' + name)
    for name, expected in manifest['presentation_input_bindings'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('source producer/provenance changed: ' + name)
    return manifest


def evidence(source=SOURCE):
    prices = read_csv('candidate_price_summary.csv', source)
    ranking = read_csv('primary_candidate_ranking.csv', source)
    entities = read_csv('participant_entities.csv', source)
    summaries = read_csv('participant_summary.csv', source)
    totals = read_csv('instrument_totals.csv', source)
    coverage = read_csv('candidate_count_curves.csv', source)
    cases = [r for r in read_csv('cases.csv', source) if r['age_policy'] == 'exp_h30']
    expected = dict(zip(INSTRUMENTS, [3526, 2393, 995]))
    if len(ranking) != 6914 or len(entities) != 2083:
        raise ValueError('full ranking/entity row counts')
    for p in prices:
        i, n = p['instrument'], int(p['n'])
        rows = [r for r in ranking if r['instrument'] == i]
        if n != expected[i] or len(rows) != n:
            raise ValueError('candidate denominator mismatch')
        for metric in ['FPM', 'REV', 'ADV']:
            missing = metric + ('_missing_or_incomplete' if metric == 'REV' else '_missing')
            values = [r[metric + '_sign'] for r in rows]
            for suffix, value in [('positive', '1'), ('zero', '0'), ('negative', '-1')]:
                if values.count(value) != int(p[metric + '_' + suffix]):
                    raise ValueError('ranking price-sign mismatch')
            if values.count('') != int(p[missing]):
                raise ValueError('missing price mismatch')
            if sum(int(p[metric + '_' + s]) for s in ['positive', 'zero', 'negative']) + int(p[missing]) != n:
                raise ValueError('sign partition mismatch')
        if sum(r['FPM_sign'] == r['REV_sign'] == '1' for r in rows) != int(p['both_positive_complete']):
            raise ValueError('joint-positive count mismatch')
        zeros = sum(float(r['WMSCI_market']) == 0 for r in rows)
        if zeros != (17 if i == 'RISANAMENTO' else 0):
            raise ValueError('zero-W retention mismatch')
    concentration = []
    for s in summaries:
        group = sorted([r for r in entities if all(r[k] == s[k] for k in ['instrument', 'perspective'])],
                       key=lambda r: (-int(r['candidate_count']), r['entity_alias']))
        n = int(s['candidate_count'])
        if (len(group) != int(s['entity_count']) or
            sum(int(r['candidate_count']) for r in group) != n or
            sum(int(r['execution_count']) for r in group) != int(s['execution_count']) or
            sum(int(r['candidate_count']) > 0 for r in group) != int(s['candidate_entity_count']) or
            int(group[0]['candidate_count']) != int(s['largest_candidate_group_count'])):
            raise ValueError('entity universe/concentration mismatch')
        cumulative = 0
        for rank, r in enumerate(group, 1):
            cumulative += int(r['candidate_count'])
            concentration.append(dict(instrument=s['instrument'], perspective=s['perspective'],
                entity_alias=r['entity_alias'], entity_rank=rank, candidate_count=int(r['candidate_count']),
                cumulative_candidates=cumulative, candidate_denominator=n,
                cumulative_percent=100*cumulative/n, observed_entities=len(group),
                candidate_entities=int(s['candidate_entity_count'])))
    by_instrument = {r['instrument']: int(r['execution_count']) for r in totals}
    if sum(by_instrument.values()) != 48481:
        raise ValueError('execution universe mismatch')
    if len(coverage) != 12 or len(cases) != 8:
        raise ValueError('native horizon support mismatch')
    for r in coverage:
        r['execution_denominator'] = by_instrument[r['instrument']]
        r['candidate_percent'] = 100*int(r['candidate_executions'])/r['execution_denominator']
    case_points = []
    for r in cases:
        case_points.append(dict(case_id=r['case_id'], instrument=r['instrument'],
            horizon_ns=int(r['horizon_ns']), horizon_seconds=int(r['horizon_ns'])/10**9,
            candidate_qty_raw=float(r['candidate_qty_raw']), withdrawal_qty_raw=float(r['withdrawal_qty_raw']),
            removal_percent=100*float(r['withdrawal_qty_raw'])/float(r['candidate_qty_raw']),
            J=float(r['J']), is_candidate=r['is_candidate']))
    return dict(prices=prices, concentration=concentration, coverage=coverage, cases=case_points)


def geometry(fig):
    """Check actual rendered text and legend bounds without tight-bbox rescue."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    canvas = fig.bbox
    checked = 0
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text():
            continue
        # Matplotlib can create invisible-to-the-domain major tick labels.
        outside = False
        for ax in fig.axes:
            for ticks, labels, limits in [(ax.get_xticks(), ax.get_xticklabels(), ax.get_xlim()),
                                          (ax.get_yticks(), ax.get_yticklabels(), ax.get_ylim())]:
                if any(label is text and not min(limits) <= tick <= max(limits)
                       for tick, label in zip(ticks, labels)):
                    outside = True
        if outside:
            continue
        box = text.get_window_extent(renderer)
        if box.x0 < -1 or box.y0 < -1 or box.x1 > canvas.width+1 or box.y1 > canvas.height+1:
            raise ValueError('clipped figure text: ' + text.get_text())
        checked += 1
    for ax in fig.axes:
        box = ax.get_window_extent(renderer)
        if box.x0 < 0 or box.y0 < 0 or box.x1 > canvas.width or box.y1 > canvas.height:
            raise ValueError('clipped axes')
    return dict(checked_texts=checked, geometry_ok=True,
                width_inches=fig.get_figwidth(), height_inches=fig.get_figheight())


def coverage_figure(data):
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    fig.subplots_adjust(left=.14, right=.84, bottom=.19, top=.83)
    for i, color, marker in zip(INSTRUMENTS, COLORS, ['o', 's', '^']):
        rows = sorted([r for r in data if r['instrument'] == i], key=lambda r: int(r['horizon_ns']))
        x = [int(r['horizon_ns'])/10**9 for r in rows]
        y = [int(r['candidate_executions']) for r in rows]
        ax.plot(x, y, color=color, marker=marker, lw=2, label=i.title())
        ax.annotate(f'{y[-1]:,}', (x[-1], y[-1]), xytext=(7, 0), textcoords='offset points', va='center', color=color)
    ax.grid(alpha=.2)
    ax.set_ylabel('Candidate executions')
    ax.set_ylim(0, 3900)
    ax.set_xlabel('Withdrawal horizon H (seconds)')
    ax.set_xticks([1, 2, 5, 10])
    ax.set_xlim(.5, 10.4)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=3, frameon=False)
    return fig


def cases_figure(data):
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.4), sharex=True, sharey='row')
    fig.subplots_adjust(left=.13, right=.97, bottom=.15, top=.86, hspace=.22, wspace=.16)
    for col, case in enumerate(['R', 'N']):
        rows = sorted([r for r in data if r['case_id'] == case], key=lambda r: r['horizon_ns'])
        for row, field in enumerate(['removal_percent', 'J']):
            ax = axes[row, col]
            ax.plot([r['horizon_seconds'] for r in rows], [r[field] for r in rows],
                    linestyle='none', marker='o', ms=7, color=COLORS[2 if case == 'R' else 1])
            ax.grid(alpha=.2)
            ax.set_xticks([1, 2, 5, 10])
            ax.set_xlim(.4, 10.6)
        axes[0, col].set_title(f'Case {case} — {rows[0]["instrument"].title()}')
        axes[1, col].set_xlabel('Observation horizon H (s)')
    axes[0, 0].set_ylim(-8, 108)
    axes[0, 0].yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axes[0, 0].set_ylabel('Raw removal share (%)')
    axes[1, 0].set_ylim(-.12, 1.95)
    axes[1, 0].set_yticks([0, .5, 1, 1.5])
    axes[1, 0].set_ylabel('Profile increment $J$')
    return fig


def prices_figure(data):
    fig, (ax, joint) = plt.subplots(1, 2, figsize=(7.2, 4.5), gridspec_kw={'width_ratios': [3.2, 1.4]})
    fig.subplots_adjust(left=.23, right=.97, bottom=.15, top=.79, wspace=.20)
    colors = ['#0072B2', '#dedede', '#D55E00', '#f8f8f8']
    for index, r in enumerate(data):
        n = int(r['n'])
        center = 6.5-3*index
        for metric, y in [('FPM', center+.5), ('REV', center-.5)]:
            start = 0
            for sign, color in zip(['positive', 'zero', 'negative', 'missing'], colors):
                key = metric+'_'+('missing_or_incomplete' if metric == 'REV' and sign == 'missing' else sign)
                width = 100*int(r[key])/n
                ax.barh(y, width, left=start, height=.64, color=color,
                        edgecolor='#555555', linewidth=.4, hatch='////' if sign == 'missing' else None,
                        label=sign.title() if index == 0 and metric == 'FPM' else None)
                start += width
        value = 100*int(r['both_positive_complete'])/n
        joint.plot(value, center, 'o', color='#252525', ms=7)
        joint.annotate(f'{int(r["both_positive_complete"]):,}/{n:,}', (value, center),
                       xytext=(0, 13), textcoords='offset points', ha='center', fontsize=11)
    ys = [7, 6, 4, 3, 1, 0]
    ax.set_yticks(ys, [f'{i.title()}  FPM' if k == 0 else 'REV' for i in INSTRUMENTS for k in [0, 1]])
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel('Share of all candidates (%)')
    ax.set_title('Movement and complete reversion', pad=12)
    joint.set_xlim(0, 20)
    joint.set_xticks([0, 10, 20])
    joint.set_xlabel('Share (%)')
    joint.set_title('Both positive', pad=12)
    joint.set_yticks([])
    for a in [ax, joint]:
        a.set_ylim(-.8, 8)
        a.grid(axis='x', alpha=.18)
        a.set_axisbelow(True)
        a.spines['left'].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), loc='upper center', ncol=4, frameon=False)
    return fig


def concentration_figure(data):
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 4.4), sharey=True)
    fig.subplots_adjust(left=.11, right=.97, bottom=.16, top=.90, wspace=.10)
    for ax, perspective, title in zip(axes, ['identified_client', 'intermediary'], ['Identified clients', 'Intermediaries (all activity)']):
        for i, color, marker in zip(INSTRUMENTS, COLORS, ['o', 's', '^']):
            rows = [r for r in data if r['instrument'] == i and r['perspective'] == perspective]
            ax.step([r['entity_rank'] for r in rows], [r['cumulative_percent'] for r in rows],
                    where='post', color=color, lw=2, linestyle=['-', '--', ':'][INSTRUMENTS.index(i)],
                    label=f'{i.title()} ({rows[0]["candidate_entities"]}/{len(rows)})')
            ax.plot(1, rows[0]['cumulative_percent'], marker=marker, color=color, ms=6)
        ax.set_xscale('log')
        ax.set_xlim(.85, 1500 if perspective == 'identified_client' else 65)
        ax.set_xticks([1, 10, 100, 1000] if perspective == 'identified_client' else [1, 5, 10, 50])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylim(-4, 106)
        ax.set_title(title)
        ax.set_xlabel('Entity rank (log scale)')
        ax.grid(alpha=.18)
        ax.legend(loc='lower right', frameon=False, fontsize=10)
    axes[0].set_ylabel('Cumulative candidate share (%)')
    return fig


def build(output=OUTPUT):
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError('immutable package exists; choose a fresh output')
    authenticate()
    data = evidence()
    output.mkdir(parents=True)
    figures = []
    with plt.rc_context(RC):
        for name, function, rows in [
            ('coverage', coverage_figure, data['coverage']),
            ('case_horizons', cases_figure, data['cases']),
            ('price_evidence', prices_figure, data['prices']),
            ('participant_concentration', concentration_figure, data['concentration'])]:
            fig = function(rows)
            info = geometry(fig)
            write_csv(output/(name+'.csv'), rows)
            fig.savefig(output/(name+'.pdf'), metadata={'CreationDate': None, 'ModDate': None, 'Creator': 'frozen-results-evidence-renderer'})
            fig.savefig(output/(name+'.png'))
            plt.close(fig)
            figures.append(dict(name=name, **info))
    manifest = dict(version='frozen_results_evidence_v1', analysis_changed=False,
        source_package=str(SOURCE.relative_to(ROOT)), source_manifest_sha256=digest(SOURCE/'artifact_manifest.json'),
        source_hashes={p.name: digest(p) for p in sorted(SOURCE.iterdir()) if p.is_file()},
        generator_sha256=digest(__file__), figures=figures,
        semantics={'population': '6914 complete-ranked physical candidates; 17 zero W retained',
                   'price': 'frozen 8-ULP sign convention; full denominators; complete REV only',
                   'cases': 'points at native H=1,2,5,10s, not MSCI or withdrawal event paths',
                   'entities': 'candidate-count order, alias ties; all observed entities including zero candidates; overlapping views not additive',
                   'uncertainty': 'descriptive observations, no uncertainty estimates'},
        artifacts={p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(json.dumps({'output': str(output), 'figures': figures}, indent=2))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    build(parser.parse_args().output)
