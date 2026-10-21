"""Disk-backed accounting of one certified risk partition, one actor at a time.

The public output contains summary contributions, not a pooled summary. They
must be combined with their original weights, never averaged without weights.
Coverage is shared; market rows are filtered lazily to each actor's epoch/time
support, never retained as an actor-by-market-message panel. A large actor can still require sizable
memory, so the outer runner must enforce the configured process budget.
"""
from __future__ import annotations

import hashlib
from io import BytesIO
from pathlib import Path

import polars as pl

from . import empirical_controls_v2 as accounting
from .empirical_controls_v2 import analyze_controls_v2
from .empirical_controls_v2_io import PartitionWriter, file_hash, scan_checkpoint, value_hash
from .empirical_clock import CLOCK_POLICY_ID
from .withdrawal_risk_v2 import SCHEMAS as RISK_SCHEMAS

SCHEMAS = {
    'state_statistics': accounting.STATE_SCHEMA,
    'contrast_statistics': accounting.CONTRAST_SCHEMA,
    'summary_contributions': accounting.SUMMARY_SCHEMA,
    'coverage': accounting.COVERAGE_SCHEMA,
    'balance': accounting.BALANCE_SCHEMA,
    'shift_support': accounting.SHIFT_SUPPORT_SCHEMA,
}


def _cluster_frame_identity(clusters: pl.DataFrame) -> str:
    encoded = BytesIO()
    clusters.write_ipc(encoded)
    result = hashlib.sha256(encoded.getbuffer()).hexdigest()
    del encoded
    return result


def bound_cache_key(risk_path: Path, *, cluster_identity: str, requested: str) -> str:
    """Bind an analysis checkpoint to exact risk, cluster and source identities."""
    if not isinstance(cluster_identity, str) or not cluster_identity:
        raise ValueError('nonempty cluster identity required')
    result = value_hash({
        'requested': requested,
        'risk_manifest': file_hash(risk_path / 'manifest.json'),
        'clusters': cluster_identity,
        'accounting_source': file_hash(Path(accounting.__file__)),
        'stage_source': file_hash(Path(__file__)),
    })
    return result


def analyze_partition(
    risk_path: Path, *, risk_cache_key: str, clusters: pl.DataFrame,
    output_path: Path, analysis_cache_key: str, buffer_rows: int,
    resume: bool = False, cluster_identity: str | None = None, clock_policy_id: str | None = None,
) -> dict[str, pl.LazyFrame]:
    """Persist per-actor accounting without replay or cross-actor retention."""
    if type(buffer_rows) is not int or buffer_rows <= 0:
        raise ValueError('positive integer buffer_rows required')
    if not isinstance(analysis_cache_key, str) or not analysis_cache_key:
        raise ValueError('analysis cache identity required')
    if clock_policy_id not in {None, CLOCK_POLICY_ID}:
        raise ValueError('unknown clock_policy_id')
    if not accounting.CLUSTER_REQUIRED <= set(clusters.columns):
        raise ValueError('canonical clusters schema incomplete')
    provenance = {'clock_policy_id': clock_policy_id} if clock_policy_id == CLOCK_POLICY_ID else None
    risk = scan_checkpoint(risk_path, cache_key=risk_cache_key, schemas=RISK_SCHEMAS, provenance=provenance)
    coverage = risk['coverage_epochs'].collect()
    domain = ['instrument', 'partition_id', 'event_date']
    if clusters.select(domain).unique().join(coverage.select(domain).unique(), on=domain, how='anti').height:
        raise ValueError('orphan execution cluster partition/date outside risk coverage')
    # Bind resume to concrete dependencies even if a caller mistakenly reuses a
    # human-readable label after changing a schedule or risk checkpoint.
    bound_key = bound_cache_key(
        risk_path,
        cluster_identity=cluster_identity or _cluster_frame_identity(clusters),
        requested=analysis_cache_key,
    )
    if output_path.exists():
        if not resume:
            raise FileExistsError(output_path)
        return scan_checkpoint(output_path, cache_key=bound_key, schemas=SCHEMAS, provenance=provenance)
    # Only the distinct actor catalog is collected globally. All actor risk and
    # state tables are filtered before collection, flushed, then released.
    actors = pl.concat([
        risk['intervals'].select('actor_key'), risk['withdrawal_events'].select('actor_key'),
    ]).unique().sort('actor_key').collect()['actor_key']
    with PartitionWriter(output_path, cache_key=bound_key, schemas=SCHEMAS, buffer_rows=buffer_rows,
                         provenance=provenance) as writer:
        for actor in actors:
            predicate = pl.col('actor_key') == actor
            intervals = risk['intervals'].filter(predicate).collect()
            epoch_keys = ['instrument', 'partition_id', 'event_date', 'coverage_epoch_id']
            start, end = ('start_ns', 'end_ns') if clock_policy_id == CLOCK_POLICY_ID else ('start_ts', 'end_ts')
            bounds = intervals.lazy().group_by(epoch_keys).agg(
                pl.col(start).min().alias('_risk_start'), pl.col(end).max().alias('_risk_end'),
            )
            market = (risk['market_intervals'].join(bounds, on=epoch_keys, how='inner')
                      .filter((pl.col(end) > pl.col('_risk_start')) &
                              (pl.col(start) < pl.col('_risk_end')))
                      .select(list(RISK_SCHEMAS['market_intervals'])).collect())
            inputs = (
                intervals,
                risk['withdrawal_events'].filter(predicate).collect(),
                clusters.filter(predicate), coverage,
            )
            if clock_policy_id is None:
                result = analyze_controls_v2(*inputs, market_intervals=market)
            else:
                result = analyze_controls_v2(
                    *inputs, market_intervals=market, clock_policy_id=clock_policy_id,
                )
            for name in SCHEMAS:
                frame = getattr(result, 'summary' if name == 'summary_contributions' else name)
                for chunk in frame.iter_slices(n_rows=buffer_rows):
                    writer.append(name, chunk.to_dicts())
            del result, intervals, market
        writer.complete()
    return scan_checkpoint(output_path, cache_key=bound_key, schemas=SCHEMAS, provenance=provenance)
