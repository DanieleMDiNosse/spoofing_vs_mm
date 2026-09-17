from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest


def api():
    name = 'spoofing_detection.lob.empirical_controls_v2_io'
    assert importlib.util.find_spec(name) is not None, 'v2 IO contract is not implemented'
    return importlib.import_module(name)


def test_version_and_budget_contract_is_strict():
    m = api()
    good = m.config_template()
    good['resources'] = {'buffer_rows': 10, 'max_rss_mb': 512, 'timeout_seconds': 30}
    m.validate_config(good)
    for key, value in [('design_version', 'empirical_controls_v1'), ('seed', 1), ('placebo_draws', 10)]:
        bad = copy.deepcopy(good)
        bad[key] = value
        with pytest.raises(ValueError):
            m.validate_config(bad)
    for value in [None, True, 0, -1, float('inf')]:
        bad = copy.deepcopy(good)
        bad['resources']['max_rss_mb'] = value
        with pytest.raises(ValueError, match='resource'):
            m.validate_config(bad)
    bad = copy.deepcopy(good)
    bad['offset_seconds'] = [0, 30]
    with pytest.raises(ValueError, match='offset'):
        m.validate_config(bad)


def test_template_requires_explicit_resource_budget():
    with pytest.raises(ValueError, match='resource'):
        api().validate_config(api().config_template())


def test_missing_certificate_fails_without_writes(tmp_path):
    m = api()
    config = m.config_template()
    config['resources'] = {'buffer_rows': 10, 'max_rss_mb': 512, 'timeout_seconds': 30}
    config['output_root'] = str(tmp_path / 'out')
    config['baseline_runs'] = {'TEST': {'certification': str(tmp_path / 'missing.json'), 'selected_event_dates': ['2024-01-01']}}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    before = set(tmp_path.iterdir())
    with pytest.raises(ValueError, match='certification'):
        m.validate_preflight_v2(path)
    assert set(tmp_path.iterdir()) == before


def test_checkpoint_atomic_publish_and_corruption(tmp_path):
    m = api()
    path = tmp_path / 'partition'
    schema = {'x': pl.Int64}
    with m.PartitionWriter(path, cache_key='risk-key', schemas={'intervals': schema}, buffer_rows=1) as writer:
        writer.append('intervals', [{'x': 1}, {'x': 2}])
        assert not (path / 'manifest.json').exists()
        writer.complete()
    assert m.read_checkpoint(path, cache_key='risk-key')['intervals'].to_dicts() == [{'x': 1}, {'x': 2}]
    with pytest.raises(ValueError, match='cache'):
        m.read_checkpoint(path, cache_key='wrong')
    chunk = next(path.glob('intervals/*.parquet'))
    chunk.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='hash'):
        m.read_checkpoint(path, cache_key='risk-key')


def test_incomplete_checkpoint_is_not_complete(tmp_path):
    m = api()
    path = tmp_path / 'partition'
    with pytest.raises(RuntimeError):
        with m.PartitionWriter(path, cache_key='key', schemas={'intervals': {'x': pl.Int64}}, buffer_rows=1) as writer:
            writer.append('intervals', [{'x': 1}])
            raise RuntimeError('interrupted')
    assert not path.exists()
    assert list(tmp_path.glob('.partition.incomplete-*'))
    with pytest.raises(ValueError, match='incomplete'):
        m.read_checkpoint(path, cache_key='key')


def test_checkpoint_empty_schema_and_existing_output(tmp_path):
    m = api()
    path = tmp_path / 'partition'
    with m.PartitionWriter(path, cache_key='key', schemas={'intervals': {'x': pl.Int64}}, buffer_rows=1) as writer:
        writer.complete()
    assert m.read_checkpoint(path, cache_key='key')['intervals'].schema == {'x': pl.Int64}
    with pytest.raises(FileExistsError):
        m.PartitionWriter(path, cache_key='key', schemas={'intervals': {'x': pl.Int64}}, buffer_rows=1)


def test_cache_key_separates_risk_and_analysis():
    m = api()
    config = m.config_template()
    identities = {'raw_events': 'abc', 'execution_metrics': 'def'}
    key = m.risk_cache_key(config, identities, 'source-v1')
    altered = copy.deepcopy(config)
    altered['offset_seconds'] = [0]
    assert m.risk_cache_key(altered, identities, 'source-v1') == key
    altered['eligibility']['max_order_age_seconds'] = 91
    assert m.risk_cache_key(altered, identities, 'source-v1') != key
    assert m.risk_cache_key(config, identities, 'source-v2') != key


def test_checkpoint_completion_does_not_materialize_all_tables(tmp_path, monkeypatch):
    m = api()
    writer = m.PartitionWriter(tmp_path / 'p', cache_key='k', schemas={'a': {'x': pl.Int64}}, buffer_rows=1)
    writer.append('a', [{'x': 1}, {'x': 2}])
    original = pl.LazyFrame.collect

    def only_scalar_collect(self, *args, **kwargs):
        assert self.collect_schema().names() == ['len'], 'checkpoint verification materialized a table'
        return original(self, *args, **kwargs)

    monkeypatch.setattr(pl.LazyFrame, 'collect', only_scalar_collect)
    writer.complete()


def test_checkpoint_requires_expected_table_graph_and_schema(tmp_path):
    m = api()
    schemas = {'a': {'x': pl.Int64}, 'b': {'y': pl.Float64}}
    path = tmp_path / 'p'
    writer = m.PartitionWriter(path, cache_key='k', schemas=schemas, buffer_rows=1)
    writer.complete()
    assert set(m.scan_checkpoint(path, cache_key='k', schemas=schemas)) == set(schemas)
    with pytest.raises(ValueError, match='schema'):
        m.scan_checkpoint(path, cache_key='k', schemas={'a': {'x': pl.String}, 'b': schemas['b']})
    manifest_path = path / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    del manifest['tables']['b']
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='table'):
        m.scan_checkpoint(path, cache_key='k', schemas=schemas)


def certificate_fixture(tmp_path):
    """Schema-valid empty sources for the certification gate, not empirical evidence."""
    from spoofing_detection.lob.panel import SORT_COLUMNS
    from spoofing_detection.lob.spoofing_metrics import EXECUTION_CLUSTER_MEMBER_EMPTY_SCHEMA
    from spoofing_detection.lob.withdrawal_risk import ACTOR_SCHEMA
    m = api()
    raw_schema = {c: pl.String for c in SORT_COLUMNS}
    raw_schema.update({'TRADEDATE': pl.Date, 'ORDEREVENTTYPE (*)': pl.Int64, 'ORDERID': pl.String,
                       'ORDERSIDE (*)': pl.Int64, 'ORDERPX': pl.Float64, 'LEAVESQTY': pl.Float64,
                       'DISPLAYEDQTY': pl.Float64, 'FIRMID': pl.String, 'NMSC_ORIGINALCLIENTIDSHORTCODE': pl.String})
    frames = {
        'raw_events': pl.DataFrame(schema=raw_schema),
        'execution_metrics': pl.DataFrame(schema={
            'execution_cluster_id': pl.String, 'partition_id': pl.String, **ACTOR_SCHEMA,
            'execution_anchor_mode': pl.String, 'event_side': pl.String,
            'cluster_end_ts': pl.Datetime('us'), 'cluster_last_sort_index': pl.Int64,
            'execution_quantity': pl.Float64}),
        'execution_cluster_members': pl.DataFrame(schema=EXECUTION_CLUSTER_MEMBER_EMPTY_SCHEMA),
    }
    artifacts = {}
    for name, frame in frames.items():
        path = tmp_path / f'{name}.parquet'
        frame.write_parquet(path)
        artifacts[name] = {'path': str(path), 'sha256': m.file_hash(path)}
    metrics = {'top_n': 10, 'withdrawal_window_seconds': 2.0, 'max_deceptive_order_age_seconds': 90.0,
               'actor_identity_mode': 'client_then_firm', 'execution_anchor_modes': ['passive', 'aggressive']}
    baseline_config = tmp_path / 'baseline_config.json'
    baseline_config.write_text(json.dumps({'metrics': metrics, 'grid': {'top_n': 999}}))
    artifacts['baseline_config'] = {'path': str(baseline_config), 'sha256': m.file_hash(baseline_config)}
    metadata = tmp_path / 'metadata.json'
    metadata.write_text(json.dumps({'parameter_source': 'json_config_only', 'config_section': 'metrics',
        'output_schema_version': 'actor_execution_anchor_v2', 'actor_identity_mode': 'client_then_firm',
        'execution_anchor_modes': ['passive', 'aggressive'], 'input': artifacts['raw_events']['path'],
        'config': str(baseline_config), 'input_hashes': {'raw_events_sha256': artifacts['raw_events']['sha256'],
        'config_sha256': artifacts['baseline_config']['sha256']},
        'artifact_hashes': {k: artifacts[k]['sha256'] for k in ['execution_metrics', 'execution_cluster_members']}}))
    artifacts['baseline_metadata'] = {'path': str(metadata), 'sha256': m.file_hash(metadata)}
    audit = tmp_path / 'audit.json'
    checks = {k: True for k in ['all_qualifying_executions', 'canonical_roles', 'membership_and_quantity',
                               'original_indices', 'partition_initialization', 'clock_contract']}
    audit.write_text(json.dumps({'status': 'passed', 'checks': checks, 'artifact_hashes': {k: v['sha256'] for k,v in artifacts.items()},
                                'command': 'synthetic empty-input audit fixture', 'producer_source_identity': 'fixture-only'}))
    config = m.config_template()
    cert = {'certificate_version': 'canonical_execution_inputs_v2', 'instrument': 'TEST', 'artifacts': artifacts,
            'clock': config['clock'], 'evidence': {'path': str(audit), 'sha256': m.file_hash(audit)}}
    cert_path = tmp_path / 'certificate.json'
    cert_path.write_text(json.dumps(cert))
    config['resources'] = {'buffer_rows': 10, 'max_rss_mb': 512, 'timeout_seconds': 30}
    config['output_root'] = str(tmp_path / 'out')
    config['baseline_runs'] = {'TEST': {'certification': str(cert_path), 'selected_event_dates': ['2024-01-01']}}
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    return path, cert_path


def test_certificate_readonly_and_effective_metrics(tmp_path):
    path, _ = certificate_fixture(tmp_path)
    before = set(tmp_path.iterdir())
    validated = api().validate_preflight_v2(path)
    assert validated['runs']['TEST']['effective_parameters']['top_n'] == 10
    assert set(tmp_path.iterdir()) == before


def test_existing_cli_dispatches_v2_readonly_without_rescoring(tmp_path, monkeypatch):
    path, _ = certificate_fixture(tmp_path)
    script = Path(__file__).resolve().parents[2] / 'scripts/build_spoofing_empirical_negative_controls.py'
    spec = importlib.util.spec_from_file_location('v2_dispatch_test', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    def forbidden(*args, **kwargs):
        pytest.fail('v2 must not rescore the canonical detector')
    monkeypatch.setattr(module, '_recompute_and_verify_canonical_runs', forbidden)
    module.main(['--config', str(path), '--validate-only'])
    assert not (tmp_path / 'out').exists()
    with pytest.raises(SystemExit):
        module.parse_args([])


@pytest.mark.parametrize('mutation', ['hash', 'schema', 'evidence', 'metrics', 'dates', 'roles'])
def test_certificate_rejects_incompatible_inputs_before_output(tmp_path, mutation):
    m = api()
    path, cp = certificate_fixture(tmp_path)
    cert = json.loads(cp.read_text())
    config = json.loads(path.read_text())
    if mutation == 'hash':
        cert['artifacts']['raw_events']['sha256'] = '0' * 64
    elif mutation == 'evidence':
        cert['evidence']['path'] = str(tmp_path / 'missing')
    elif mutation == 'dates':
        config['baseline_runs']['TEST']['selected_event_dates'] = ['20240101']
    else:
        key = {'schema': 'raw_events', 'metrics': 'baseline_config', 'roles': 'execution_metrics'}[mutation]
        target = Path(cert['artifacts'][key]['path'])
        if mutation == 'metrics':
            target.write_text(json.dumps({'grid': {'top_n': 10}}))
        else:
            pl.DataFrame({'wrong': [1]}).write_parquet(target)
        cert['artifacts'][key]['sha256'] = m.file_hash(target)
    cp.write_text(json.dumps(cert)); path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        m.validate_preflight_v2(path)
    assert not (tmp_path / 'out').exists()
