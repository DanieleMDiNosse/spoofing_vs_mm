#!/usr/bin/env python3
"""Notation-only derivative of the ACTIVE spoofing_simple publication assets.

No detector, bootstrap, fitting, plotting or analysis is executed. Existing PDFs
have no obsolete labels and are copied byte-for-byte. Run --check to authenticate
the completed package. Outputs are immutable: build requires a fresh directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
PREFIX = 'generated/simple_notation_20261020'
OUTPUT = ROOT / 'paper' / PREFIX
TABLES = ROOT / 'paper/generated/withdrawal_rank_20261015_participants_v3'
KERNEL = ROOT / 'paper/generated/hit_frequency_iid_se_20261012_160642'
FROZEN_DIRS = [TABLES, KERNEL, ROOT / 'paper/generated/new_kernel_h1_kernel',
               ROOT / 'paper/generated/results_evidence_20261019_v4',
               ROOT / 'paper/generated/results_evidence_20261020_v8']
TABLE_LOCK = {
    'population': '0d9fefdd9181af754e2cb1ab61632674bccbcef788987c7f96f61bd24a627dd3',
    'ranking_summary': '639dce788a45e6b95ca580c02b8f7e43d399ce95ccb4b8d10ced63449f64201e',
    'anonymouscase_table': 'ac440a17dffd6c56e783fbf20e02ec6d410727a8c953398b128aa77abf928911',
    'candidate_prices': '36c89bee04d2dbc23b6feecbc0e1d9df51dcabb3e01dc814e5477b577a9e19cb',
    'composition': 'fb50ac14cc20f7e1cd85ea99dc08b2aecd891218ef23605933226d22676f256f',
    'participant_summary': '1a4867da3604b9f40f0df515cb2779871f8c2d12de5a79fca363491e5b17b1f1',
    'horizon_counts': '5eea2b61d6e3ee4cb7c0c26f0f24cfe1600ec1df68d3ad8436b0c79a13b12f9b',
    'age_policy_summary': '16f725420ba92ef2de066ac05af8475a22b01ddade40f9111849494582fa7d09',
    'candidate_advantage': '3c6f0ed07918043c835722c0150cebcf4e884a1f8ca7d71fdb20c276fae28e1e',
    'candidate_price_strata': 'cd4440774edf2e46607dddc4ca829cfc10f2d18674900512f73c934a373b227e',
    'client_groups': 'b8c04aef0d12c165de926a9c178efae40ea5cc89fa1aa185b22389a9b274ba9c',
    'intermediary_groups': '3aaabd997ba7fd27e4325f7b55f811a5c572400fad6fdba891cd1f1e2ccbe02b',
}
SOURCE_LOCK = {str((TABLES / (name + '.tex')).relative_to(ROOT)): value
               for name, value in TABLE_LOCK.items()}
SOURCE_LOCK.update({
    'paper/generated/hit_frequency_iid_se_20261012_160642/kernel_figures.tex': 'e88c88bac45e02fe0bea55ffd469c8246983b252df20c2ebf1492d7fb5772ab2',
    'paper/generated/new_kernel_h1_kernel/kernel_weights_se.pdf': 'e29452a1f496915c9254427e04331079c46f11f4f0044330ebe7c738e7556280',
    'paper/generated/new_kernel_h1_kernel/kernel_visibility_se.pdf': '5e341a194b69e35d44d12c83277a1e95149797d9737dcab5a1caf36b0aa36006',
    'paper/generated/results_evidence_20261019_v4/coverage.pdf': '13d38fa72bf3149c0a74f0ade30d3e33df43020ef91887df89a048b01d48fc68',
    'paper/generated/results_evidence_20261019_v4/participant_concentration.pdf': '2a34ef10a56fb1b55e377fae3f155637950067ca60be88f9b87fe1b476abc640',
    'paper/generated/results_evidence_20261020_v8/price_evidence.pdf': 'c6294cf0eaca0ae426c22a09c2a193dc9556d58b4c5e6e277306652637b3be24',
})
POPULATION = {'executions': 48481, 'candidates': 6914,
              'Ferrari': 3526, 'Nexi': 2393, 'Risanamento': 995}
OBSOLETE = re.compile(r'\b(?:WMSCI|MSCI|DWI)\b')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def frozen_hashes():
    return {str(p.relative_to(ROOT)): digest(p)
            for directory in FROZEN_DIRS for p in sorted(directory.rglob('*')) if p.is_file()}


def numeric_tokens(text):
    # File paths contain dates/version numbers, not empirical quantities.
    text = re.sub(r'\\includegraphics(?:\[[^\]]*\])?\{[^}]+\}', '', text)
    return re.findall(r'[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?', text)


def transform_tex(text, *, calibration=False):
    # Do not rename identifiers, citations or reference keys: only rendered text.
    protected = re.compile(r'(\\(?:label|ref|eqref|cref|Cref|cite)\{[^}]*\})')
    chunks = protected.split(text)
    for i in range(0, len(chunks), 2):
        s = chunks[i]
        s = s.replace('market-normalized WMSCI', 'withdrawal score $W$')
        s = re.sub(r'\bWMSCI\b', 'withdrawal score', s)
        s = s.replace(r'\operatorname{MSCI}', r'\Phi')
        s = s.replace(r'\operatorname{DWI}', 'I')
        s = re.sub(r'\bMSCI\b', 'profile shift', s)
        s = re.sub(r'\bDWI\b', 'liquidity imbalance', s)
        s = s.replace('$R$', '$U/Q$').replace('$A$', '$U/P$')
        s = s.replace('^{(g)}', r'^{\mathrm{age}}')
        s = re.sub(r'\bb(?=\s*=\s*\d)', 'h', s)
        s = s.replace('g_b', 'g_h')
        # Existing policy names are source keys; expose the actual half-life.
        s = re.sub(r'exp h(\d+)', lambda m: r'exponential $h=' + m[1] + '$ s', s)
        s = s.replace('legacy age30', 'age cap 30 s')
        if calibration:
            s = re.sub(r'\bH\b', lambda m: r'H_{\mathrm{cal}}', s)
        chunks[i] = s
    return ''.join(chunks)


def pdf_text(path):
    return subprocess.check_output(['pdftotext', '-layout', str(path), '-'], text=True)


def authenticate_sources():
    for name, expected in SOURCE_LOCK.items():
        if digest(ROOT / name) != expected:
            raise ValueError('frozen source changed: ' + name)
    # The package must reproduce the exact population, not merely total cells.
    rows = list(csv.DictReader((TABLES / 'instrument_totals.csv').open()))
    if sum(int(r['execution_count']) for r in rows) != POPULATION['executions']:
        raise ValueError('execution universe mismatch')
    population = (TABLES / 'population.tex').read_text()
    for name in ['Ferrari', 'Nexi', 'Risanamento']:
        row = next(line for line in population.splitlines() if line.startswith(name + ' &'))
        count = int(row.split('&')[-1].replace('\\', '').replace(',', '').strip())
        if count != POPULATION[name]:
            raise ValueError('candidate denominator mismatch: ' + name)
    # Authenticate publication receipts, without replaying their analytical inputs.
    complete = json.loads((TABLES / 'COMPLETE.json').read_text())
    if complete['artifact_manifest_sha256'] != digest(TABLES / 'artifact_manifest.json'):
        raise ValueError('completion binding mismatch')
    table_manifest = json.loads((TABLES / 'artifact_manifest.json').read_text())
    for name, expected in table_manifest['artifacts'].items():
        if digest(TABLES / name) != expected:
            raise ValueError('table artifact binding mismatch: ' + name)
    for directory in FROZEN_DIRS[1:]:
        meta_name = 'manifest.json' if directory.name.startswith('results_') else 'metadata.json'
        meta = json.loads((directory / meta_name).read_text())
        for name, expected in meta.get('artifacts', {}).items():
            if digest(directory / name) != expected:
                raise ValueError('publication artifact binding mismatch: ' + str(directory / name))


def build(output=OUTPUT):
    output = Path(output).resolve()
    try:
        prefix = output.relative_to(ROOT / 'paper').as_posix()
    except ValueError:
        prefix = output.as_posix()
    if output.exists():
        raise FileExistsError('immutable package exists; use --check or choose fresh output')
    authenticate_sources()
    before = frozen_hashes()
    tex_rows, graphics = [], []
    # Preflight before writing any output. There is deliberately no silent PDF edit.
    for name in SOURCE_LOCK:
        path = ROOT / name
        if path.suffix == '.pdf':
            text = pdf_text(path)
            if OBSOLETE.search(text):
                raise ValueError('PDF requires generator-level label rerender: ' + name)
            graphics.append({'source': name, 'destination': path.name,
                             'sha256': digest(path), 'action': 'byte-identical copy', 'pdf_text': text})
    output.mkdir(parents=True)
    for name in SOURCE_LOCK:
        source = ROOT / name
        if source.suffix != '.tex':
            shutil.copyfile(source, output / source.name)
            continue
        original = source.read_text()
        new = transform_tex(original, calibration=source.name == 'kernel_figures.tex')
        new = re.sub(r'(\\includegraphics(?:\[[^\]]*\])?\{)([^}]+)(\})',
                     lambda m: m[1] + prefix + '/' + Path(m[2]).name + m[3], new)
        if numeric_tokens(original) != numeric_tokens(new):
            raise ValueError('numeric token sequence changed: ' + name)
        if OBSOLETE.search(re.sub(r'\\label\{[^}]+\}', '', new)):
            raise ValueError('obsolete visible notation remains: ' + name)
        (output / source.name).write_text(new)
        tex_rows.append({'source': name, 'destination': source.name,
                         'numeric_tokens_preserved': True,
                         'numeric_tokens_sha256': hashlib.sha256(json.dumps(numeric_tokens(original)).encode()).hexdigest()})
    if len(tex_rows) != 13 or len(graphics) != 5:
        raise ValueError('active dependency closure mismatch')
    if before != frozen_hashes():
        raise ValueError('historical package changed during presentation build')
    provenance = {
        'kernel_pdf_renderer': 'outputs/kernel_h1_paper_release_20261012_103855/prepare_kernel_figures.py',
        'kernel_pdf_data': 'paper/generated/new_kernel_h1_kernel/kernel_standard_errors.csv',
        'kernel_pdf_receipt': 'paper/generated/new_kernel_h1_kernel/metadata.json',
        'hit_tikz_data': 'paper/generated/hit_frequency_iid_se_20261012_160642/hit_frequency_standard_errors.csv',
        'hit_tikz_receipt': 'paper/generated/hit_frequency_iid_se_20261012_160642/metadata.json',
        'results_renderer': 'scripts/generate_results_evidence_figures.py',
        'coverage_and_concentration_receipt': 'paper/generated/results_evidence_20261019_v4/manifest.json',
        'price_receipt': 'paper/generated/results_evidence_20261020_v8/manifest.json',
    }
    manifest = {'version': 'simple_notation_presentation_v1', 'analysis_changed': False,
                'generator_sha256': digest(__file__), 'population': POPULATION,
                'tex': tex_rows, 'graphics': graphics, 'frozen_source_hashes': before,
                'provenance': {k: {'path': v, 'sha256': digest(ROOT / v)} for k, v in provenance.items()},
                'semantics': 'W is the original market withdrawal score; J is the original largest positive profile increment; no extra selection or joint score.',
                'limitations': 'PDF text checked via pdftotext; PDFs reused byte-for-byte, not re-rendered. Full manuscript build is the integrating editor responsibility.'}
    report = ['# Active notation-only asset mapping', '',
              'All paths below are relative to `paper/`; compile from that directory.',
              'Replace each old generated input with the destination below; main manuscript is not edited.', '',
              '| Source | Parent include path |', '|---|---|']
    for row in tex_rows + graphics:
        report.append(f"| `{row['source'].removeprefix('paper/')}` | `{prefix}/{row['destination']}` |")
    report += ['', '## Preservation and provenance', '',
               '13 TeX inputs and 5 active PDF assets. All numeric token sequences are exact, including TikZ coordinates, axis scales and error-bar endpoints; all PDFs are byte-identical.',
               'Calibration captions and hit-frequency ordinate use $H_{\\mathrm{cal}}$; withdrawal/observation horizons retain $H$. Exponential age policies display $h$ explicitly.',
               'W and J remain separate; labels/reference keys and all empirical cells are preserved. Table R/A and weighted (g) notation are absent from active tables; the wrapper supports their explicit presentation replacements.',
               'The original five publication directories are hashed before/after generation. No raw data, detector, original generators or historical assets are written.', '',
               'Kernel PDFs trace to `outputs/kernel_h1_paper_release_20261012_103855/prepare_kernel_figures.py`, using retained `kernel_standard_errors.csv` (centers and bootstrap standard errors). The active hit TikZ instead comes from the iid-SE package; it is not replaced by the older kernel hit renderer.',
               'Coverage and concentration trace through the v4 manifest and its CSVs; price evidence through v8 and its CSV. `scripts/generate_results_evidence_figures.py` is the current renderer; the v4 manifest records its historical renderer hash, which need not equal the current v8 script.',
               'All five active PDFs were inspected with `pdftotext -layout`; none contains WMSCI, MSCI, DWI or an ambiguous calibration horizon. No PDF regeneration is needed, so every plotted coordinate, scale and uncertainty is preserved at the byte level.',
               '48,481 executions; 6,914 candidates: Ferrari 3,526, Nexi 2,393, Risanamento 995. See manifest.json for source hashes, extracted PDF text and exact mappings.', '',
               '## Verification', '',
               '`/home/danielemdn/miniconda3/envs/main/bin/python scripts/generate_simple_notation_package.py --check`',
               '`/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_simple_notation_package.py -q`', '',
               'This package has no dependency on inactive case_horizons or candidate_counts PDFs. The integrating editor must compile the full manuscript and inspect its final extracted text.']
    (output / 'ASSET_MAPPING.md').write_text('\n'.join(report) + '\n')
    manifest['artifacts'] = {p.name: digest(p) for p in sorted(output.iterdir()) if p.is_file()}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    verify(output)
    return manifest


def verify(output=OUTPUT):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    authenticate_sources()
    expected_tex = {(name, Path(name).name) for name in SOURCE_LOCK if name.endswith('.tex')}
    expected_graphics = {(name, Path(name).name) for name in SOURCE_LOCK if name.endswith('.pdf')}
    if manifest.get('population') != POPULATION:
        raise ValueError('manifest population mismatch')
    for key, expected in [('tex', expected_tex), ('graphics', expected_graphics)]:
        rows = manifest.get(key, [])
        observed = {(r.get('source'), r.get('destination')) for r in rows}
        if len(rows) != len(expected) or observed != expected:
            raise ValueError('manifest dependency mapping mismatch: ' + key)
    expected_artifacts = {dest for _, dest in expected_tex | expected_graphics} | {'ASSET_MAPPING.md'}
    if set(manifest.get('artifacts', {})) != expected_artifacts:
        raise ValueError('manifest artifact closure mismatch')
    for entry in manifest.get('provenance', {}).values():
        if digest(ROOT / entry['path']) != entry['sha256']:
            raise ValueError('recorded provenance changed: ' + entry['path'])
    if len(manifest.get('provenance', {})) != 8:
        raise ValueError('manifest provenance closure mismatch')
    if manifest['generator_sha256'] != digest(__file__):
        raise ValueError('generator changed since build')
    if manifest['frozen_source_hashes'] != frozen_hashes():
        raise ValueError('historical source package changed')
    for name, expected in manifest['artifacts'].items():
        if digest(output / name) != expected:
            raise ValueError('artifact changed: ' + name)
    for row in manifest['tex']:
        if numeric_tokens((ROOT / row['source']).read_text()) != numeric_tokens((output / row['destination']).read_text()):
            raise ValueError('numeric token mismatch')
    for row in manifest['graphics']:
        if digest(ROOT / row['source']) != digest(output / row['destination']):
            raise ValueError('graphic changed')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    result = verify(args.output) if args.check else build(args.output)
    print(json.dumps({'output': str(args.output), 'tex_inputs': len(result['tex']),
                      'byte_identical_pdfs': len(result['graphics']),
                      'population': result['population'], 'verified': True}, indent=2))
