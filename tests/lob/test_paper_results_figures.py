"""Frozen-result presentation contracts, without replay or score changes."""
from pathlib import Path
import importlib.util
import json
from collections import Counter
import re

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / '.hermes/checkpoints/results_visual_redesign_20261019'
PACKAGE = ROOT / 'paper/generated/results_evidence_20261020_v8'


def module():
    spec = importlib.util.spec_from_file_location('results_evidence_renderer', ROOT/'scripts/generate_results_evidence_figures.py')
    assert spec is not None and spec.loader is not None
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_source_integrity_full_population_and_native_support():
    m = module()
    m.authenticate()
    data = m.evidence()
    assert sum(int(r['n']) for r in data['prices']) == 6914
    assert [int(r['both_positive_complete']) for r in data['prices']] == [397, 308, 44]
    assert [int(r['both_positive_complete_ADV_nonnegative']) for r in data['prices']] == [387, 273, 38]
    assert len(data['concentration']) == 2083
    for instrument in m.INSTRUMENTS:
        for perspective in ['identified_client', 'intermediary']:
            rows = [r for r in data['concentration'] if r['instrument']==instrument and r['perspective']==perspective]
            assert rows[-1]['cumulative_percent'] == pytest.approx(100)
            assert np.all(np.diff([r['cumulative_percent'] for r in rows]) >= 0)
            assert any(r['candidate_count']==0 for r in rows)
    for case in ['R', 'N']:
        rows = [r for r in data['cases'] if r['case_id']==case]
        assert sorted(r['horizon_seconds'] for r in rows) == [1, 2, 5, 10]


def test_joint_price_count_includes_adv_zero_but_not_missing_or_negative():
    m = module()
    rows = [dict(FPM_sign='1', REV_sign='1', ADV_sign=sign) for sign in ['1', '0', '-1', '']]
    rows += [dict(FPM_sign='0', REV_sign='1', ADV_sign='1'),
             dict(FPM_sign='1', REV_sign='', ADV_sign='1'),
             dict(FPM_sign='1', REV_sign='-1', ADV_sign='1')]
    assert m.joint_price_count(rows) == 2


def test_corrupted_joint_count_rejected(monkeypatch):
    m = module()
    original = m.read_csv
    def corrupted(name, source=m.SOURCE):
        rows = original(name, source)
        if name == 'candidate_price_summary.csv':
            rows[0]['both_positive_complete'] = str(int(rows[0]['both_positive_complete'])+1)
        return rows
    monkeypatch.setattr(m, 'read_csv', corrupted)
    with pytest.raises(ValueError, match='joint-positive'):
        m.evidence()


def test_existing_package_rejected_before_reading(tmp_path, monkeypatch):
    m = module()
    def unexpected():
        raise AssertionError('input access before output refusal')
    monkeypatch.setattr(m, 'authenticate', unexpected)
    with pytest.raises(FileExistsError):
        m.build(tmp_path)


def test_all_figures_fit_and_encodings_preserve_observations():
    m = module()
    data = m.evidence()
    with m.plt.rc_context(m.RC):
        for function, key in [(m.coverage_figure,'coverage'), (m.cases_figure,'cases'), (m.prices_figure,'prices'), (m.concentration_figure,'concentration')]:
            fig = function(data[key])
            try:
                assert m.geometry(fig)['geometry_ok']
                if key == 'coverage':
                    assert len(fig.axes) == 1
                    assert fig.axes[0].get_ylabel() == 'Candidate executions'
                    assert len(fig.axes[0].lines) == 3
                    for line, instrument in zip(fig.axes[0].lines, m.INSTRUMENTS):
                        rows = sorted([r for r in data['coverage'] if r['instrument']==instrument], key=lambda r: int(r['horizon_ns']))
                        assert list(line.get_ydata()) == [int(r['candidate_executions']) for r in rows]
                if key == 'cases':
                    assert all(line.get_linestyle() == 'None' for ax in fig.axes for line in ax.lines)
                    assert all(len(line.get_xdata()) == 4 for ax in fig.axes for line in ax.lines)
                if key == 'prices':
                    bars = fig.axes[0].patches
                    assert len(bars)==36
                    for offset in range(0,36,4):
                        assert sum(p.get_height() for p in bars[offset:offset+4]) == pytest.approx(100)
                        assert bars[offset+3].get_hatch() == '////'
                    for index, row in enumerate(data['prices']):
                        for metric_index, metric in enumerate(['FPM', 'REV', 'ADV']):
                            for sign_index, sign in enumerate(['positive', 'zero', 'negative', 'missing']):
                                suffix = 'missing_or_incomplete' if metric == 'REV' and sign == 'missing' else sign
                                bar = bars[12*index + 4*metric_index + sign_index]
                                assert bar.get_height() == pytest.approx(100*int(row[metric+'_'+suffix])/int(row['n']))
                    expected = [100*int(r['both_positive_complete_ADV_nonnegative'])/int(r['n']) for r in data['prices']]
                    assert [bar.get_height() for bar in fig.axes[1].patches] == pytest.approx(expected)
                    assert fig.axes[1].get_title() == 'FPM > 0, REV > 0\nand ADV ≥ 0'
                    renderer = fig.canvas.get_renderer()
                    bounds = fig.axes[1].get_window_extent(renderer)
                    for annotation in fig.axes[1].texts:
                        box = annotation.get_window_extent(renderer)
                        assert bounds.x0 <= box.x0 < box.x1 <= bounds.x1
                        assert bounds.y0 <= box.y0 < box.y1 <= bounds.y1
            finally:
                m.plt.close(fig)




def test_active_package_manifest_preserved():
    m = module()
    manifest = json.loads((PACKAGE/'manifest.json').read_text())
    assert manifest['generator_sha256'] == m.digest(ROOT/'scripts/generate_results_evidence_figures.py')
    for name, expected in manifest['source_hashes'].items():
        assert m.digest(m.SOURCE/name)==expected
    for name, expected in manifest['artifacts'].items():
        assert m.digest(PACKAGE/name)==expected
