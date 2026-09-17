"""Canonical single-partition replay into verified, resumable compact chunks.

This internal boundary consumes an already-certified, complete partition. It is
not a certification issuer or a real-data CLI; callers remain responsible for
source hashes, complete initialization, resource supervision and provenance.
"""
from __future__ import annotations

from collections.abc import Mapping, Iterator
from pathlib import Path
from typing import Any

import polars as pl

from .config import LOBConfig
from .panel import SORT_COLUMNS, _partition_id
from .withdrawal_risk_v2 import SCHEMAS, compute_compact_risk
from .empirical_controls_v2_io import PartitionWriter, scan_checkpoint


class _OriginalIndices(Mapping[int, int]):
    """Constant-space explicit canonical local-to-global ordinal mapping."""

    def __init__(self, count: int, offset: int):
        self.count, self.offset = count, offset

    def __getitem__(self, key: int) -> int:
        if type(key) is not int or not 1 <= key <= self.count:
            raise KeyError(key)
        return self.offset + key

    def __iter__(self) -> Iterator[int]:
        return iter(range(1, self.count + 1))

    def __len__(self) -> int:
        return self.count


def cache_partition(
    raw_events: pl.DataFrame, *, path: Path, partition: Mapping[str, Any],
    instrument: str, cache_key: str, buffer_rows: int, resume: bool = False,
) -> dict[str, pl.LazyFrame]:
    """Replay one full partition once, publishing only validated schema chunks.

    Index offsets refer to the original globally sorted input, not a filtered
    day ordinal. Incomplete directories are preserved for diagnosis, never
    appended to or treated as resumable. The completed path is reusable only
    with the supplied exact cache identity and required table graph.
    """
    count, offset = partition.get('rows'), partition.get('index_offset')
    if type(count) is not int or count <= 0 or count != raw_events.height:
        raise ValueError('complete nonempty partition row count required')
    if type(offset) is not int or offset < 0:
        raise ValueError('nonnegative original index offset required')
    if not set(SORT_COLUMNS[:5]) <= set(raw_events.columns):
        raise ValueError('partition columns missing')
    domains = raw_events.select(SORT_COLUMNS[:5]).unique()
    if domains.height != 1:
        raise ValueError('exactly one complete partition required')
    row = domains.row(0, named=True)
    if _partition_id(row) != partition.get('partition_id') or str(row['TRADEDATE']) != partition.get('event_date'):
        raise ValueError('partition identity/date mismatch')
    if not isinstance(instrument, str) or not instrument or not isinstance(cache_key, str) or not cache_key:
        raise ValueError('instrument and cache identity required')
    if type(buffer_rows) is not int or buffer_rows <= 0:
        raise ValueError('positive buffer_rows required')
    if path.exists():
        if not resume:
            raise FileExistsError(path)
        return scan_checkpoint(path, cache_key=cache_key, schemas=SCHEMAS)
    with PartitionWriter(path, cache_key=cache_key, schemas=SCHEMAS, buffer_rows=buffer_rows) as writer:
        compute_compact_risk(
            raw_events, instrument=instrument, top_n=10, max_order_age_seconds=90,
            config=LOBConfig(snapshot_mode='none'),
            original_sort_indices=_OriginalIndices(count, offset),
            sink=writer.append, buffer_rows=buffer_rows,
        )
        writer.complete()
    return scan_checkpoint(path, cache_key=cache_key, schemas=SCHEMAS)
