"""Scoped consumer tests; no replay or producer changes."""
import importlib.util
import json
from pathlib import Path
import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / 'scripts/generate_withdrawal_multiscale_manuscript.py'


def module():
    assert SCRIPT.exists(), 'new validated-artifact generator is missing'
    spec = importlib.util.spec_from_file_location('multiscale_manuscript', SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_completed_roots_accept_exact_hashes():
    m = module()
    for root in m.DEFAULT_ROOTS:
        evidence = m.validate_root(REPO / root)
        assert evidence['population_sha256'] == m.digest(REPO / root / 'analysis/population.parquet')


@pytest.mark.parametrize('status', ['complete', 'derived', 'legacy', None])
def test_rejects_legacy_or_incomplete_markers(tmp_path, status):
    m = module()
    (tmp_path / 'prepare').mkdir()
    (tmp_path / 'prepare/COMPLETE.json').write_text(json.dumps({'status': status}))
    with pytest.raises(ValueError, match='completion'):
        m.validate_root(tmp_path)


def test_rejects_manifest_tampering(tmp_path):
    m = module()
    (tmp_path / 'COMPLETE.json').write_text(json.dumps({'status': 'verified', 'artifact_manifest_sha256': '0' * 64}))
    (tmp_path / 'artifact_manifest.json').write_text('{}')
    with pytest.raises(ValueError, match='hash'):
        m.validate_stage(tmp_path, 'verified')


def test_primary_contract_and_stored_support_budgets():
    m = module()
    import polars as pl
    assert m.PRIMARY_POLICY == 'exp_h30'
    assert m.PRIMARY_HORIZON_NS == 10_000_000_000
    assert m.OUTPUT.name == 'withdrawal_multiscale_20261014_primary_prices_v1'
    cell = pl.DataFrame({
        'instrument': ['X'] * 3, 'partition_id': ['p'] * 3,
        'actor_key': ['a'] * 3, 'execution_cluster_id': ['1', '2', '3'],
        'is_candidate': [True] * 3, 'WMSCI_self': [1., 2., 0.],
        'WMSCI_market': [1., 0., 0.],
        'priority_top_w_self': [False, True, False],
        'priority_top_w_market': [True, False, False]})
    row = m.ranking_metrics(cell)
    assert row['positive_self_scores'] == 2
    assert row['positive_market_scores'] == row['common_eligible_candidates'] == 1
    assert row['self_ceiling_budget'] == row['market_ceiling_budget'] == 1
    assert row['top_self'] == row['top_market'] == 1
    assert row['intersection'] == 0  # stored flags, not reranking shared support


def test_overlap_uses_union_denominator():
    m = module()
    assert m.overlap_metrics({1, 2}, {2, 3}) == {'intersection': 1, 'union': 3, 'jaccard': 1 / 3}
    assert m.overlap_metrics(set(), set())['jaccard'] is None


def test_existing_package_is_rejected_before_any_source_reads(tmp_path, monkeypatch):
    m = module()
    sentinel = tmp_path / 'validated.txt'
    sentinel.write_text('preserve validated output')
    def forbidden_read(_):
        raise AssertionError('existing output must be rejected before input reads')
    monkeypatch.setattr(m, 'validate_root', forbidden_read)
    with pytest.raises(FileExistsError, match='exists'):
        m.build(tmp_path)
    assert sentinel.read_text() == 'preserve validated output'


def test_figure_geometry_checks_only_rendered_ticks(tmp_path):
    m = module()
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 13})
    fig, ax = plt.subplots(figsize=(6.8, 4), layout='constrained')
    ax.plot([1, 2], [0, 995])
    ax.set_ylim(0, 1045)
    result = m.save_figure(fig, tmp_path, 'geometry.pdf', 'Discrete design points.', 'counts.csv')
    assert result['geometry_ok']
    assert len(result['axis_limits']) == 1
    assert result['axis_limits'][0]['y'] == [0.0, 1045.0]


def test_appendix_table_can_remain_in_source_order_without_floating(tmp_path):
    m = module()
    m.table(tmp_path, 'counts.tex', 'tab:counts', 'Observed counts.',
            ['Instrument', 'Count'], [['Example', 2]], floating=False)
    rendered = (tmp_path / 'counts.tex').read_text()
    assert r'\captionof{table}' in rendered
    assert r'\begin{minipage}' in rendered
    assert r'\begin{table}' not in rendered


def test_generated_package_roundtrip_geometry_and_anonymity():
    m = module()
    out = REPO / m.OUTPUT
    if not (out / 'facts.json').exists():
        pytest.skip('run real generator before package roundtrip test')
    import polars as pl
    facts = json.loads((out / 'facts.json').read_text())
    counts = pl.read_csv(out / 'cell_counts.csv')
    assert counts.height == 60
    assert facts['totals']['execution_count'] == 48481
    assert facts['totals']['fill_member_count'] == 65471
    assert facts['totals']['execution_cells'] == 969620
    assert facts['totals']['unique_candidate_executions_any_cell'] == 6924
    assert pl.read_csv(out / 'cases.csv').height == 40
    assert counts['candidate_executions'].sum() == facts['totals']['candidate_cells']
    for asset in facts['figures']:
        assert (out / asset['path']).stat().st_size > 1000
        assert asset['minimum_font_pt'] >= 13
        assert asset['geometry_ok']
        assert asset['width_inches'] <= 7
    assert 'Reference baseline' in (out / 'baseline_prices.tex').read_text()
    assert '30-second' in (out / 'baseline_advantage.tex').read_text()
    assert facts['reference_price_baseline']['scope'] == 'H1_age30_self_priority_only_not_full_multiscale_prices'
    case_figure = next(a for a in facts['figures'] if a['path'] == 'withdrawal_and_profile_summary.pdf')
    limits = case_figure['axis_limits']
    assert limits[0]['y'] == limits[1]['y']
    assert limits[2]['y'] == limits[3]['y']
    assert limits[2]['y'][0] < 0
    assert limits[2]['y'][1] > pl.read_csv(out / 'cases.csv')['J'].max()
    assert facts['primary_policy'] == 'exp_h30'
    assert facts['primary_horizon_ns'] == 10_000_000_000
    assert 'external_scope' not in facts
    assert not list(out.glob('*external*')) and not list(out.glob('*alert*'))
    ranking = pl.read_csv(out / 'ranking_overlap.csv')
    assert ranking.height == 12
    assert ranking['age_policy'].unique().to_list() == ['exp_h30']
    primary_total = 0
    for root in m.DEFAULT_ROOTS:
        native = pl.read_parquet(REPO / root / 'analysis/population.parquet')
        for h in m.HORIZONS:
            cell = native.filter((pl.col('age_policy') == m.PRIMARY_POLICY) & (pl.col('horizon_ns') == h))
            observed = ranking.filter((pl.col('instrument') == cell['instrument'][0]) & (pl.col('horizon_seconds') == h / 1e9)).row(0, named=True)
            assert all(observed[k] == v for k, v in m.ranking_metrics(cell).items())
        primary = native.filter((pl.col('age_policy') == m.PRIMARY_POLICY) & (pl.col('horizon_ns') == m.PRIMARY_HORIZON_NS) & pl.col('is_candidate'))
        primary_total += primary.height
        comp = pl.read_csv(out / 'composition.csv').filter((pl.col('instrument') == native['instrument'][0]) & (pl.col('cohort') == 'Primary candidates'))
        assert comp['executions'].sum() == primary.height
    assert facts['primary_candidate_executions'] == primary_total == 6914
    assert {r['instrument']: r['candidate_executions'] for r in facts['primary_population']} == {'FERRARI': 3526, 'NEXI': 2393, 'RISANAMENTO': 995}
    risa = ranking.filter((pl.col('instrument') == 'RISANAMENTO') & (pl.col('horizon_seconds') == 10)).row(0, named=True)
    assert (risa['raw_candidates'], risa['positive_self_scores'], risa['positive_market_scores'], risa['common_eligible_candidates']) == (995, 984, 978, 978)
    assert (risa['top_self'], risa['top_market'], risa['intersection'], risa['union']) == (99, 98, 79, 118)
    population_text = (out / 'population.tex').read_text()
    assert 'Candidate union' not in population_text and 'Cells' not in population_text
    assert '$H=10$ s' in population_text and 'half-life' in population_text
    assert 'candidate-union' not in (out / 'composition.tex').read_text()
    cases = pl.read_csv(out / 'cases.csv').filter(pl.col('age_policy') == m.PRIMARY_POLICY)
    assert facts['primary_case_quantities'] == cases.to_dicts()
    n10 = cases.filter((pl.col('case_id') == 'N') & (pl.col('horizon_seconds') == 10)).row(0, named=True)
    assert [n10[f] for f in m.FLAGS] == [False, True, False, True]
    r2 = cases.filter((pl.col('case_id') == 'R') & (pl.col('horizon_seconds') == 2)).row(0, named=True)
    assert [r2[f] for f in m.FLAGS] == [True, True, False, False]
    for h in [5, 10]:
        r = cases.filter((pl.col('case_id') == 'R') & (pl.col('horizon_seconds') == h)).row(0, named=True)
        assert all(r[f] for f in m.FLAGS)
    case_text = (out / 'anonymouscase_table.tex').read_text()
    assert 'post-hoc' in case_text and 'age-weighted quantities used in $W$' in case_text
    for row in cases.iter_rows(named=True):
        assert ' & '.join(m.number(v) for v in [row['case_id'], int(row['horizon_seconds']), int(row['candidate_qty_raw']), int(row['withdrawal_qty_raw']), row['WMSCI_self'], row['WMSCI_market'], row['J']]) in case_text
    for asset in facts['figures']:
        assert 'without weighting' not in asset['caption']
        assert 'alert' not in asset['caption'].lower()
        assert '30-second half-life' in asset['caption']
    for path in list(out.glob('*.tex')) + [out / 'cases.csv']:
        text = path.read_text()
        assert '900000000000664' not in text and '900000000001520' not in text
        assert 'client_original:' not in text and 'EC-A-' not in text
    manifest = json.loads((out / 'artifact_manifest.json').read_text())
    for name, expected in manifest['artifacts'].items():
        assert m.digest(out / name) == expected


def test_primary_support_table_is_self_contained():
    m = module()
    text = (REPO / m.OUTPUT / 'ranking_comparison.tex').read_text()
    assert '.csv' not in text
    assert '984 / 978' in text
    assert 'Positive support' in text


def test_case_caption_has_no_edit_history():
    m = module()
    text = (REPO / m.OUTPUT / 'anonymouscase_table.tex').read_text()
    assert 'unchanged' not in text


def primary_price_fixture(tmp_path):
    import polars as pl
    m = module()
    root = tmp_path / 'price_source'
    root.mkdir()
    rows, counts = [], []
    for instrument in ['RISANAMENTO', 'NEXI', 'FERRARI']:
        counts.append({'instrument': instrument, 'age_policy': 'exp_h30',
                       'horizon_ns': 10_000_000_000,
                       'priority_top_w_self_count': 3, 'priority_top_w_market_count': 3,
                       'priority_top_j_given_self_count': 1, 'priority_top_j_given_market_count': 1})
        for ranking in ['self', 'market']:
            for stage, n in [('W', 3), ('J', 1)]:
                rows.append({'instrument': instrument, 'ranking': ranking, 'stage': stage, 'n': n,
                             'FPM_positive': 1, 'FPM_zero': n-1, 'FPM_negative': 0, 'FPM_missing': 0,
                             'REV_positive': 1, 'REV_zero': 0, 'REV_negative': 0, 'REV_missing_or_incomplete': n-1,
                             'ADV_positive': 0, 'ADV_zero': n-1, 'ADV_negative': 1, 'ADV_missing': 0,
                             'both_positive_complete': 1})
    pl.DataFrame(rows).write_csv(root / 'summary.csv')
    (root / 'metadata.json').write_text(json.dumps({'schema': 'synthetic_primary_price_fixture'}))
    artifacts = {p.name: m.digest(p) for p in root.iterdir()}
    (root / 'artifact_manifest.json').write_text(json.dumps({'artifacts': artifacts}))
    (root / 'COMPLETE.json').write_text(json.dumps({'status': 'primary_prices_verified',
        'artifact_manifest_sha256': m.digest(root / 'artifact_manifest.json')}))
    return root, pl.DataFrame(counts)


def test_primary_prices_use_native_list_denominators_and_all_signs(tmp_path):
    m = module()
    root, counts = primary_price_fixture(tmp_path)
    output = tmp_path / 'publication'
    output.mkdir()
    facts = m.render_primary_price_tables(output, root, counts)
    assert facts['scope'] == 'exp_h30_H10_self_market_W_and_conditional_J_no_price_selection'
    assert len(facts['summary']) == 12
    for ranking in ['self', 'market']:
        text = (output / f'primary_{ranking}_prices.tex').read_text()
        assert '1/2/0/0' in text and '1/0/0/2' in text
        assert '30-second' in text and '$H=10$' in text and 'two seconds' in text
        assert 'no price filtering' in text
    assert '0/2/1/0' in (output / 'primary_advantage.tex').read_text()
    assert (output / 'primary_price_summary.csv').read_bytes() == (root / 'summary.csv').read_bytes()


def test_primary_prices_reject_wrong_population_before_writing_tables(tmp_path):
    m = module()
    import polars as pl
    root, counts = primary_price_fixture(tmp_path)
    counts = counts.with_columns(pl.lit(4).alias('priority_top_w_self_count'))
    output = tmp_path / 'publication'
    output.mkdir()
    with pytest.raises(ValueError, match='membership|denominator'):
        m.render_primary_price_tables(output, root, counts)
    assert not list(output.iterdir())
