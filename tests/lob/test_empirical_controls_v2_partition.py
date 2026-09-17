"""Synthetic canonical replay/checkpoint integration, no detector metrics forged."""
import importlib

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from spoofing_detection.lob.panel import _partition_id
from spoofing_detection.lob.withdrawal_risk_v2 import SCHEMAS, compute_compact_risk
from spoofing_detection.lob.empirical_controls_v2_io import read_checkpoint


def raw_fixture():
    rows = []
    for seq, order, kind, second in [(1, 'B', 1, '00'), (2, 'A', 1, '03'), (3, 'B', 4, '05')]:
        ts = f'2024-01-02 09:30:{second}'
        rows.append({
            'TRADEDATE': '2024-01-02', 'MIC': 'XMIL', 'MARKETCODE': 'MTA', 'SYMBOLINDEX': 1, 'EMM (*)': 1,
            'SEQUENCETIME': ts, 'BOOKIN': ts, 'BOOKOUTTIME': ts,
            'HDR_APPLKEYSEQUENCENUMBER': seq, 'HDR_HWMSEQUENCENUMBER': seq,
            'HDR_OFFSETID': seq, 'ROW_NUMBER': seq, 'EVENTID': f'e{seq}',
            'ORDEREVENTTYPE (*)': kind, 'ORDERID': order, 'ORDERPRIORITY': str(seq),
            'ORDERSIDE (*)': 1 if order == 'B' else 2, 'ORDERPX': 100.0 if order == 'B' else 101.0,
            'ORDERQTY': 10.0, 'LEAVESQTY': 0.0 if kind == 4 else 10.0,
            'DISPLAYEDQTY': 0.0 if kind == 4 else 10.0, 'ORDERTYPE (*)': 2,
            'TIMEINFORCE (*)': 0, 'PASSIVEORDER': 'N', 'AGGRESSIVEORDER': 'N',
            'FIRMID': 'F', 'NMSC_ORIGINALCLIENTIDSHORTCODE': order, 'ORDER_TRADINGCAPACITY (*)': 3,
        })
    return pl.DataFrame(rows)


def descriptor(raw):
    row = raw.row(0, named=True)
    return {'partition_id': _partition_id(row), 'event_date': row['TRADEDATE'], 'rows': raw.height, 'index_offset': 20}


def api():
    return importlib.import_module('spoofing_detection.lob.empirical_controls_v2_partition')


def test_partition_replay_spills_and_verified_resume_never_replays(tmp_path, monkeypatch):
    m = api()
    raw = raw_fixture()
    meta = descriptor(raw)
    path = tmp_path / 'partition'
    m.cache_partition(raw, path=path, partition=meta, instrument='TEST', cache_key='key', buffer_rows=1)
    stored = read_checkpoint(path, cache_key='key', schemas=SCHEMAS)
    expected = compute_compact_risk(raw, instrument='TEST', original_sort_indices={1: 21, 2: 22, 3: 23})
    for name in SCHEMAS:
        assert_frame_equal(stored[name], getattr(expected, name))
    assert stored['withdrawal_events']['sort_index'].to_list() == [23]
    def forbidden(*args, **kwargs):
        pytest.fail('resume must not replay a validated partition')
    monkeypatch.setattr(m, 'compute_compact_risk', forbidden)
    m.cache_partition(raw, path=path, partition=meta, instrument='TEST', cache_key='key', buffer_rows=1, resume=True)
    with pytest.raises(ValueError, match='cache'):
        m.cache_partition(raw, path=path, partition=meta, instrument='TEST', cache_key='changed', buffer_rows=1, resume=True)


@pytest.mark.parametrize('mutation', ['count', 'partition', 'offset'])
def test_partition_contract_fails_before_writes(tmp_path, mutation):
    raw = raw_fixture()
    meta = descriptor(raw)
    if mutation == 'count':
        meta['rows'] += 1
    elif mutation == 'partition':
        meta['partition_id'] = 'wrong'
    else:
        meta['index_offset'] = -1
    with pytest.raises(ValueError):
        api().cache_partition(raw, path=tmp_path / 'p', partition=meta, instrument='TEST', cache_key='key', buffer_rows=1)
    assert not list(tmp_path.iterdir())


def test_interruption_never_publishes_and_retry_does_not_append(tmp_path, monkeypatch):
    m = api()
    original = m.compute_compact_risk
    def interrupted(raw, **kwargs):
        original(raw, **kwargs)
        raise RuntimeError('simulated interruption after flush')
    monkeypatch.setattr(m, 'compute_compact_risk', interrupted)
    raw = raw_fixture()
    path = tmp_path / 'p'
    args = dict(path=path, partition=descriptor(raw), instrument='TEST', cache_key='key', buffer_rows=1)
    with pytest.raises(RuntimeError, match='interruption'):
        m.cache_partition(raw, **args)
    assert not path.exists()
    monkeypatch.setattr(m, 'compute_compact_risk', original)
    m.cache_partition(raw, **args, resume=True)
    assert read_checkpoint(path, cache_key='key')['withdrawal_events'].height == 1
