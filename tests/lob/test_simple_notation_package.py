"""Presentation-only regression checks against frozen publication assets."""
import importlib.util
import json
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/generate_simple_notation_package.py'


def module():
    assert SCRIPT.exists(), 'notation-only package generator has not been implemented'
    spec = importlib.util.spec_from_file_location('simple_notation', SCRIPT)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def test_notation_is_visible_only_and_preserves_numbers():
    m = module()
    source = (r'\label{tab:WMSCI-R} WMSCI, MSCI, DWI; '
              r'$R$ & $A$ & $P^{(g)}$ & $U^{(g)}$ & $b=30$; 0.0012')
    rendered = m.transform_tex(source)
    assert r'\label{tab:WMSCI-R}' in rendered
    assert 'withdrawal score' in rendered and 'profile shift' in rendered
    assert 'liquidity imbalance' in rendered
    assert '$U/Q$' in rendered and '$U/P$' in rendered
    assert r'P^{\mathrm{age}}' in rendered and r'U^{\mathrm{age}}' in rendered
    assert '$h=30$' in rendered
    assert m.numeric_tokens(source) == m.numeric_tokens(rendered)


def test_calibration_horizon_is_not_withdrawal_horizon():
    m = module()
    assert r'H_{\mathrm{cal}}=1' in m.transform_tex('$H=1$', calibration=True)
    assert m.transform_tex('$H=10$') == '$H=10$'


def test_fresh_package_closure_and_preservation(tmp_path):
    m = module()
    before = m.frozen_hashes()
    out = tmp_path / 'package'
    manifest = m.build(out)
    assert before == m.frozen_hashes()
    assert len(manifest['tex']) == 13
    assert len(manifest['graphics']) == 5
    for row in manifest['tex']:
        original = (ROOT / row['source']).read_text()
        new = (out / row['destination']).read_text()
        assert m.numeric_tokens(original) == m.numeric_tokens(new)
        assert re.findall(r'\\label\{[^}]+\}', original) == re.findall(r'\\label\{[^}]+\}', new)
        visible = re.sub(r'\\label\{[^}]+\}', '', new)
        assert not re.search(r'\b(?:WMSCI|MSCI|DWI)\b', visible)
        for graphic in re.findall(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}', new):
            resolved = Path(graphic) if Path(graphic).is_absolute() else ROOT / 'paper' / graphic
            assert resolved.parent == out.resolve()
            assert resolved.exists()
    for row in manifest['graphics']:
        assert (ROOT / row['source']).read_bytes() == (out / row['destination']).read_bytes()
        assert not re.search(r'\b(?:WMSCI|MSCI|DWI)\b', row['pdf_text'])
    assert manifest['population'] == {'executions': 48481, 'candidates': 6914,
                                     'Ferrari': 3526, 'Nexi': 2393, 'Risanamento': 995}
    assert m.verify(out) == json.loads((out / 'manifest.json').read_text())
    with pytest.raises(FileExistsError):
        m.build(out)


def test_verify_rejects_tampered_asset(tmp_path):
    m = module()
    out = tmp_path / 'package'
    m.build(out)
    with (out / 'population.tex').open('a') as f:
        f.write('\n123')
    with pytest.raises(ValueError, match='artifact changed'):
        m.verify(out)


@pytest.mark.parametrize('field', ['population', 'tex', 'graphics', 'artifacts', 'provenance'])
def test_verify_rejects_tampered_manifest_metadata(tmp_path, field):
    m = module()
    out = tmp_path / 'package'
    m.build(out)
    path = out / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest[field] = [] if field in {'tex', 'graphics'} else {}
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='manifest'):
        m.verify(out)


def test_verify_rejects_changed_renderer_receipt(tmp_path):
    m = module()
    out = tmp_path / 'package'
    m.build(out)
    path = out / 'manifest.json'
    manifest = json.loads(path.read_text())
    manifest['provenance']['results_renderer']['sha256'] = '0' * 64
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='provenance changed'):
        m.verify(out)
