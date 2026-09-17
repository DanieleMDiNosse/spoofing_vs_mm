"""Strict v2 configuration and atomic, content-verified partition storage.

Certificates are evidence supplied by a separate baseline audit, never issued by
this runner. Hash identity alone does not certify baseline completeness.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import date
from pathlib import Path
import tempfile
from typing import Any, Mapping

import polars as pl

from .panel import SORT_COLUMNS, _partition_id
from .replay_observation import EVENT_TIMESTAMP_PRECEDENCE
from .withdrawal_risk import CLOCK_REGRESSION_POLICY, canonical_execution_cluster_rows, ACTOR_SCHEMA

VERSION = "empirical_controls_v2_light"
SCHEMA_VERSION = "compact_risk_v2_1"


def config_template() -> dict[str, Any]:
    """Intentionally non-runnable until budgets and certificates are supplied."""
    return {
        "design_version": VERSION,
        "output_root": None,
        "clock": {
            "event_timestamp_precedence": list(EVENT_TIMESTAMP_PRECEDENCE),
            "stable_sort_columns": list(SORT_COLUMNS),
            "partition_columns": list(SORT_COLUMNS[:5]),
            "risk_clock_regression_policy": CLOCK_REGRESSION_POLICY,
            "timezone": "source_clock_timezone_unknown",
        },
        "eligibility": {"top_n": 10, "max_order_age_seconds": 90.0, "outcome_selection": "forbidden"},
        "window_seconds": 2.0,
        "time_band_seconds": 1800,
        "offset_seconds": [-60, -30, 0, 30, 60],
        "identity_mode": "client_then_firm",
        "execution_anchor_modes": ["passive", "aggressive"],
        "resources": {"buffer_rows": None, "max_rss_mb": None, "timeout_seconds": None},
        "baseline_runs": {},
    }


def validate_config(config: Mapping[str, Any]) -> None:
    template = config_template()
    if not isinstance(config, Mapping) or set(config) != set(template):
        raise ValueError("v2 config requires exactly the versioned keys; legacy/draw/seed options forbidden")
    for key in ("design_version", "clock", "eligibility", "window_seconds", "time_band_seconds", "offset_seconds", "identity_mode", "execution_anchor_modes"):
        # JSON equality alone treats True as 1; typed serialized comparisons do not.
        if json.dumps(config[key], sort_keys=True) != json.dumps(template[key], sort_keys=True):
            # Permit equivalent finite numeric duration spelling, not booleans.
            if key == 'window_seconds' and type(config[key]) in (int, float) and config[key] == 2:
                continue
            raise ValueError(f"incompatible v2 {key}")
    resources = config.get('resources')
    if not isinstance(resources, dict) or set(resources) != set(template['resources']):
        raise ValueError('explicit resource budgets required')
    for key, value in resources.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'positive finite resource budget required: {key}')
        if key == 'buffer_rows' and type(value) is not int:
            raise ValueError('resource buffer_rows must be an integer')


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def value_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def risk_cache_key(config: Mapping[str, Any], input_identities: Mapping[str, str], source_identity: str) -> str:
    return value_hash({
        'schema': SCHEMA_VERSION, 'inputs': dict(input_identities), 'source': source_identity,
        'clock': config['clock'], 'eligibility': config['eligibility'], 'identity_mode': config['identity_mode'],
    })


def _json(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError(f'JSON object required: {path}')
    return data


def validate_preflight_v2(config_path: Path, *, resume: bool = False) -> dict[str, Any]:
    config = _json(config_path)
    validate_config(config)
    output = config['output_root']
    if not isinstance(output, str) or not output.strip():
        raise ValueError('explicit output_root required')
    output = Path(output).resolve()
    if output.exists():
        raise FileExistsError(f'output root already exists: {output}')
    runs = config['baseline_runs']
    if not isinstance(runs, dict) or not runs:
        raise ValueError('baseline_runs must contain certified input runs')
    validated = {}
    for instrument, run in runs.items():
        if not isinstance(instrument, str) or not instrument.strip() or not isinstance(run, dict):
            raise ValueError('invalid baseline run')
        cert_path = run.get('certification')
        if not isinstance(cert_path, str) or not Path(cert_path).is_file():
            raise ValueError(f'missing baseline certification for {instrument}')
        validated[instrument] = validate_certificate(Path(cert_path), run, config, output)
        if validated[instrument]['instrument'] != instrument:
            raise ValueError('certification instrument mismatch')
    return {'design_version': VERSION, 'output_root': output, 'run_count': len(validated), 'runs': validated, 'config': config}


def validate_certificate(path: Path, run: Mapping, config: Mapping, output: Path) -> dict:
    cert = _json(path)
    if cert.get('certificate_version') != 'canonical_execution_inputs_v2' or cert.get('clock') != config['clock']:
        raise ValueError('incompatible certification version/clock')
    dates = run.get('selected_event_dates')
    if not isinstance(dates, list) or not dates or any(not isinstance(d, str) for d in dates) or len(set(dates)) != len(dates):
        raise ValueError('explicit unique selected_event_dates required')
    for d in dates:
        if date.fromisoformat(d).isoformat() != d:
            raise ValueError('strict ISO selected_event_dates required')
    artifacts = cert.get('artifacts')
    required = {'raw_events', 'execution_metrics', 'execution_cluster_members', 'baseline_config', 'baseline_metadata'}
    if not isinstance(artifacts, dict) or set(artifacts) != required:
        raise ValueError('certification must bind all required artifacts')

    def verify(entry):
        if not isinstance(entry, dict) or not isinstance(entry.get('path'), str):
            raise ValueError('certification artifact path required')
        target = Path(entry['path']).resolve()
        if output == target or output in target.parents:
            raise ValueError('output must be isolated from inputs')
        if not target.is_file() or file_hash(target) != entry.get('sha256'):
            raise ValueError(f'certification path/hash mismatch: {target.name}')
        return target

    sources = {key: verify(entry) for key, entry in artifacts.items()}
    evidence = _json(verify(cert.get('evidence')))
    checks = ['all_qualifying_executions', 'canonical_roles', 'membership_and_quantity',
              'original_indices', 'partition_initialization', 'clock_contract']
    if (evidence.get('status') != 'passed' or any(evidence.get('checks', {}).get(c) is not True for c in checks)
            or not evidence.get('producer_source_identity') or not evidence.get('command')
            or evidence.get('artifact_hashes') != {k: v['sha256'] for k, v in artifacts.items()}):
        raise ValueError('insufficient baseline certification evidence')
    metadata = _json(sources['baseline_metadata'])
    expected = {'parameter_source': 'json_config_only', 'config_section': 'metrics',
                'output_schema_version': 'actor_execution_anchor_v2', 'actor_identity_mode': 'client_then_firm',
                'execution_anchor_modes': ['passive', 'aggressive']}
    if any(metadata.get(k) != v for k, v in expected.items()):
        raise ValueError('incompatible baseline metadata')
    for key, meta_key in [('raw_events', 'input'), ('baseline_config', 'config')]:
        if Path(metadata.get(meta_key, '')).resolve() != sources[key]:
            raise ValueError('baseline metadata source path mismatch')
        hash_key = 'config_sha256' if key == 'baseline_config' else 'raw_events_sha256'
        if metadata.get('input_hashes', {}).get(hash_key) != artifacts[key]['sha256']:
            raise ValueError('baseline metadata source hash mismatch')
    for key in ('execution_metrics', 'execution_cluster_members'):
        if metadata.get('artifact_hashes', {}).get(key) != artifacts[key]['sha256']:
            raise ValueError('baseline metadata artifact hash mismatch')
    metrics = _json(sources['baseline_config']).get('metrics')
    required_metrics = {'top_n': 10, 'withdrawal_window_seconds': 2.0, 'max_deceptive_order_age_seconds': 90.0,
                        'actor_identity_mode': 'client_then_firm', 'execution_anchor_modes': ['passive', 'aggressive']}
    if not isinstance(metrics, dict) or any(metrics.get(k) != v or isinstance(metrics.get(k), bool) for k, v in required_metrics.items()):
        raise ValueError('incompatible effective metrics (grid is not a parameter source)')
    raw = pl.scan_parquet(sources['raw_events'])
    raw_columns = set(raw.collect_schema().names())
    required_raw = set(SORT_COLUMNS) | {'ORDERID', 'ORDERPX', 'LEAVESQTY', 'DISPLAYEDQTY', 'FIRMID', 'NMSC_ORIGINALCLIENTIDSHORTCODE'}
    if (not required_raw <= raw_columns or not {'ORDEREVENTTYPE (*)', 'ORDEREVENTTYPE'} & raw_columns
            or not {'ORDERSIDE (*)', 'ORDERSIDE'} & raw_columns):
        raise ValueError('raw_events schema incompatible')
    if raw.collect_schema()['TRADEDATE'] not in (pl.Date, pl.String):
        raise ValueError('TRADEDATE must be Date or ISO String')
    # Only the tiny partition catalog is materialized. Counts include unselected
    # days to preserve original global canonical one-based sort indices.
    partitions = raw.group_by(SORT_COLUMNS[:5]).agg(pl.len().alias('rows')).sort(SORT_COLUMNS[:5], nulls_last=True).collect()
    offset, catalog = 0, []
    for row in partitions.iter_rows(named=True):
        day = str(row['TRADEDATE'])
        if date.fromisoformat(day).isoformat() != day:
            raise ValueError('non-ISO raw partition date')
        catalog.append({**row, 'partition_id': _partition_id(row), 'event_date': day, 'index_offset': offset})
        offset += row['rows']
    cluster_required = {'execution_cluster_id', 'partition_id', *ACTOR_SCHEMA, 'event_side',
                        'execution_anchor_mode', 'cluster_end_ts', 'cluster_last_sort_index', 'execution_quantity'}
    member_required = {'execution_cluster_id', 'partition_id', *ACTOR_SCHEMA, 'execution_anchor_mode',
                       'child_order_id', 'child_sort_index', 'child_fill_qty', 'child_fill_price'}
    for key, columns in [('execution_metrics', cluster_required), ('execution_cluster_members', member_required)]:
        if not columns <= set(pl.read_parquet_schema(sources[key])):
            raise ValueError(f'{key} schema incompatible')
    # Validate partition-wise, not the full multi-month member table in RAM.
    known_ids = {r['partition_id'] for r in catalog}
    for key in ('execution_metrics', 'execution_cluster_members'):
        ids = set(pl.scan_parquet(sources[key]).select('partition_id').unique().collect()['partition_id'].to_list())
        if not ids <= known_ids:
            raise ValueError('execution artifacts reference unknown partitions')
    for row in catalog:
        if row['event_date'] not in dates:
            continue
        c = pl.scan_parquet(sources['execution_metrics']).filter(pl.col('partition_id') == row['partition_id']).select(sorted(cluster_required)).collect()
        members = pl.scan_parquet(sources['execution_cluster_members']).filter(pl.col('partition_id') == row['partition_id']).select(sorted(member_required)).collect()
        validate_cluster_members(c, members, row)
    return {'instrument': cert.get('instrument'), 'sources': sources, 'artifact_hashes': {k: v['sha256'] for k,v in artifacts.items()},
            'certificate_path': path.resolve(), 'certificate_sha256': file_hash(path), 'effective_parameters': metrics,
            'selected_event_dates': dates, 'partitions': catalog, 'producer_source_identity': evidence['producer_source_identity']}


def validate_cluster_members(clusters: pl.DataFrame, members: pl.DataFrame, partition: Mapping) -> None:
    """Check membership/role identity and quantity without detector rescoring."""
    canonical = canonical_execution_cluster_rows(clusters)
    if len(canonical) != clusters.height:
        raise ValueError('duplicate canonical execution clusters')
    ids = {c['execution_cluster_id']: c for c in canonical}
    by_cluster: dict[str, list[dict]] = {k: [] for k in ids}
    seen = set()
    for member in members.iter_rows(named=True):
        cid = member['execution_cluster_id']
        cluster = ids.get(cid)
        if cluster is None:
            raise ValueError('orphan execution member')
        if any(member[k] != cluster[k] for k in ['partition_id', *ACTOR_SCHEMA, 'execution_anchor_mode']):
            raise ValueError('incompatible member role/identity')
        index = member['child_sort_index']
        if type(index) is not int or not partition['index_offset'] < index <= partition['index_offset'] + partition['rows']:
            raise ValueError('member original index outside partition')
        # A physical child row belongs to only one canonical execution role.
        if index in seen:
            raise ValueError('duplicate member or conflicting execution role')
        seen.add(index)
        qty = member['child_fill_qty']
        if type(qty) not in (int, float) or not math.isfinite(qty) or qty <= 0:
            raise ValueError('invalid child quantity')
        by_cluster[cid].append(member)
    for cid, cluster in ids.items():
        children = by_cluster[cid]
        if (not children or max(m['child_sort_index'] for m in children) != cluster['cluster_last_sort_index']
                or cluster['cluster_end_ts'].date().isoformat() != partition['event_date']
                or not math.isclose(sum(m['child_fill_qty'] for m in children), cluster['execution_quantity'], rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('cluster membership, final index, date or quantity mismatch')


class PartitionWriter:
    """Bounded table chunks; incomplete work remains inspectable but never reusable."""

    def __init__(self, path: Path, *, cache_key: str, schemas: Mapping[str, Mapping], buffer_rows: int):
        if path.exists():
            raise FileExistsError(path)
        if type(buffer_rows) is not int or buffer_rows <= 0:
            raise ValueError('positive buffer_rows required')
        if not schemas or any(not name.isidentifier() for name in schemas):
            raise ValueError('nonempty safe schema table names required')
        self.path, self.cache_key = path, cache_key
        self.schemas = dict(schemas)
        self.buffer_rows = buffer_rows
        path.parent.mkdir(parents=True, exist_ok=True)
        self.temporary = Path(tempfile.mkdtemp(prefix=f'.{path.name}.incomplete-', dir=path.parent))
        self.artifacts: dict[str, list[dict]] = {name: [] for name in schemas}
        self.completed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        # Deliberately preserve incomplete chunks for diagnosis, not resume.
        return False

    def append(self, name: str, rows: list[dict]) -> None:
        if self.completed or name not in self.schemas:
            raise ValueError('closed writer or unknown table')
        for start in range(0, len(rows), self.buffer_rows):
            frame = pl.DataFrame(rows[start:start + self.buffer_rows], schema=self.schemas[name], strict=True)
            self._write(name, frame)

    def _write(self, name: str, frame: pl.DataFrame) -> None:
        relative = f'{name}/{len(self.artifacts[name]):08d}.parquet'
        path = self.temporary / relative
        path.parent.mkdir(exist_ok=True)
        frame.write_parquet(path)
        self.artifacts[name].append({'path': relative, 'sha256': file_hash(path), 'rows': frame.height})

    def complete(self) -> None:
        if self.completed or self.path.exists():
            raise FileExistsError(self.path)
        for name, entries in self.artifacts.items():
            if not entries:
                self._write(name, pl.DataFrame(schema=self.schemas[name]))
        manifest = {'schema_version': SCHEMA_VERSION, 'cache_key': self.cache_key, 'complete': True, 'tables': self.artifacts}
        (self.temporary / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        # Read back all hashes, schemas and counts before publishing the marker.
        scan_checkpoint(self.temporary, cache_key=self.cache_key, schemas=self.schemas)
        os.rename(self.temporary, self.path)
        self.completed = True


def scan_checkpoint(
    path: Path, *, cache_key: str, schemas: Mapping[str, Mapping] | None = None,
) -> dict[str, pl.LazyFrame]:
    """Verify chunks individually and return disk-backed tables without collecting.

    Consumers must supply their expected table graph/schemas for resume. Hashes
    establish file identity, not completeness of a consumer's required graph.
    """
    manifest_path = path / 'manifest.json'
    if not manifest_path.is_file():
        raise ValueError(f'incomplete checkpoint: {path}')
    manifest = _json(manifest_path)
    if manifest.get('complete') is not True or manifest.get('schema_version') != SCHEMA_VERSION:
        raise ValueError('incomplete or incompatible checkpoint')
    if manifest.get('cache_key') != cache_key:
        raise ValueError('checkpoint cache key mismatch')
    tables = manifest.get('tables')
    if not isinstance(tables, dict) or not tables:
        raise ValueError('checkpoint tables missing')
    if schemas is not None and set(tables) != set(schemas):
        raise ValueError('checkpoint table graph mismatch')
    result = {}
    seen_paths = set()
    for name, entries in tables.items():
        paths = []
        expected_schema = dict(schemas[name]) if schemas is not None else None
        for entry in entries:
            chunk = (path / entry['path']).resolve()
            if path.resolve() not in chunk.parents or not chunk.is_file() or file_hash(chunk) != entry['sha256']:
                raise ValueError('checkpoint path/hash mismatch')
            if chunk in seen_paths:
                raise ValueError('duplicate checkpoint chunk path')
            seen_paths.add(chunk)
            scan = pl.scan_parquet(chunk)
            actual_schema = dict(scan.collect_schema())
            if expected_schema is None:
                expected_schema = actual_schema
            if list(actual_schema.items()) != list(expected_schema.items()):
                raise ValueError('checkpoint schema mismatch')
            if scan.select(pl.len()).collect().item() != entry['rows']:
                raise ValueError('checkpoint row count mismatch')
            paths.append(chunk)
        if not paths:
            raise ValueError('checkpoint table has no schema chunk')
        result[name] = pl.scan_parquet(paths)
    return result


def read_checkpoint(
    path: Path, *, cache_key: str, schemas: Mapping[str, Mapping] | None = None,
) -> dict[str, pl.DataFrame]:
    """Eager compatibility reader; large consumers should use scan_checkpoint."""
    return {name: frame.collect() for name, frame in
            scan_checkpoint(path, cache_key=cache_key, schemas=schemas).items()}
