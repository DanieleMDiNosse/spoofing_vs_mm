"""Synthetic raw-risk checkpoint integration; not full detector validation."""
from __future__ import annotations

from datetime import datetime
import importlib
from pathlib import Path

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from spoofing_detection.lob.empirical_controls_v2 import (
    BALANCE_SCHEMA,
    CONTRAST_SCHEMA,
    COVERAGE_SCHEMA,
    SHIFT_SUPPORT_SCHEMA,
    STATE_SCHEMA,
    SUMMARY_SCHEMA,
    analyze_controls_v2,
)
from spoofing_detection.lob.empirical_controls_v2_io import PartitionWriter, scan_checkpoint
from spoofing_detection.lob.panel import _partition_id
from spoofing_detection.lob.withdrawal_risk_v2 import SCHEMAS


CLUSTER_SCHEMA = {
    "instrument": pl.String,
    "partition_id": pl.String,
    "event_date": pl.Date,
    "actor_key": pl.String,
    "actor_id": pl.String,
    "identity_level": pl.String,
    "identity_source": pl.String,
    "identity_fallback_flag": pl.Boolean,
    "event_side": pl.String,
    "execution_anchor_mode": pl.String,
    "cluster_end_ts": pl.Datetime("us"),
    "cluster_last_sort_index": pl.Int64,
    "execution_cluster_id": pl.String,
    "execution_quantity": pl.Float64,
}
ANALYSIS_SCHEMAS = {
    "state_statistics": STATE_SCHEMA,
    "contrast_statistics": CONTRAST_SCHEMA,
    "summary_contributions": SUMMARY_SCHEMA,
    "coverage": COVERAGE_SCHEMA,
    "balance": BALANCE_SCHEMA,
    "shift_support": SHIFT_SUPPORT_SCHEMA,
}


def api():
    return importlib.import_module("spoofing_detection.lob.empirical_controls_v2_analysis")


def raw_fixture() -> pl.DataFrame:
    rows = []
    for seq, order, kind, second, client in [
        (1, "B", 1, "00", "C1"),
        (2, "A", 1, "03", "OTHER"),
        (3, "B", 4, "05", "C1"),
        (4, "X", 1, "06", "OTHER"),
    ]:
        ts = f"2024-01-02 09:30:{second}"
        rows.append({
            "TRADEDATE": "2024-01-02", "MIC": "XMIL", "MARKETCODE": "MTA", "SYMBOLINDEX": 1, "EMM (*)": 1,
            "SEQUENCETIME": ts, "BOOKIN": ts, "BOOKOUTTIME": ts,
            "HDR_APPLKEYSEQUENCENUMBER": seq, "HDR_HWMSEQUENCENUMBER": seq, "HDR_OFFSETID": seq,
            "ROW_NUMBER": seq, "EVENTID": f"e{seq}", "ORDEREVENTTYPE (*)": kind, "ORDERID": order,
            "ORDERPRIORITY": str(seq), "ORDERSIDE (*)": 1 if order == "B" else 2,
            "ORDERPX": 100.0 if order == "B" else 101.0, "ORDERQTY": 10.0,
            "LEAVESQTY": 0.0 if kind == 4 else 10.0, "DISPLAYEDQTY": 0.0 if kind == 4 else 10.0,
            "ORDERTYPE (*)": 2, "TIMEINFORCE (*)": 0, "PASSIVEORDER": "N", "AGGRESSIVEORDER": "N",
            "FIRMID": "F", "NMSC_ORIGINALCLIENTIDSHORTCODE": client, "ORDER_TRADINGCAPACITY (*)": 3,
        })
    return pl.DataFrame(rows)


def descriptor(raw: pl.DataFrame) -> dict:
    row = raw.row(0, named=True)
    return {"partition_id": _partition_id(row), "event_date": row["TRADEDATE"], "rows": raw.height, "index_offset": 0}


def clusters(raw: pl.DataFrame) -> pl.DataFrame:
    meta = descriptor(raw)
    return pl.DataFrame([{
        "instrument": "TEST", "partition_id": meta["partition_id"], "event_date": datetime(2024, 1, 2).date(),
        "actor_key": "client_original:C1", "actor_id": "C1", "identity_level": "client_original",
        "identity_source": "NMSC_ORIGINALCLIENTIDSHORTCODE", "identity_fallback_flag": False,
        "event_side": "ask", "execution_anchor_mode": "passive", "cluster_end_ts": datetime(2024, 1, 2, 9, 30, 3),
        "cluster_last_sort_index": 2, "execution_cluster_id": "fixture-cluster", "execution_quantity": 1.0,
    }], schema=CLUSTER_SCHEMA, strict=True)


def cache_risk(tmp_path: Path, raw: pl.DataFrame) -> Path:
    from spoofing_detection.lob.empirical_controls_v2_partition import cache_partition

    path = tmp_path / "risk"
    cache_partition(raw, path=path, partition=descriptor(raw), instrument="TEST", cache_key="risk-key", buffer_rows=1)
    return path


def test_partition_analysis_spills_actor_contributions_matching_per_actor_accounting(tmp_path):
    raw = raw_fixture()
    risk_path = cache_risk(tmp_path, raw)
    out = api().analyze_partition(
        risk_path, risk_cache_key="risk-key", clusters=clusters(raw), output_path=tmp_path / "analysis",
        analysis_cache_key="analysis-key", buffer_rows=1,
    )
    risk = scan_checkpoint(risk_path, cache_key="risk-key", schemas=SCHEMAS)
    actor_keys = sorted(set(
        risk["intervals"].select("actor_key").collect().get_column("actor_key").to_list()
        + risk["withdrawal_events"].select("actor_key").collect().get_column("actor_key").to_list()
    ))
    expected = {name: [] for name in ANALYSIS_SCHEMAS}
    for actor_key in actor_keys:
        result = analyze_controls_v2(
            risk["intervals"].filter(pl.col("actor_key") == actor_key).collect(),
            risk["withdrawal_events"].filter(pl.col("actor_key") == actor_key).collect(),
            clusters(raw).filter(pl.col("actor_key") == actor_key),
            risk["coverage_epochs"].collect(),
            market_intervals=risk["market_intervals"].collect(),
        )
        expected["state_statistics"].append(result.state_statistics)
        expected["contrast_statistics"].append(result.contrast_statistics)
        expected["summary_contributions"].append(result.summary)
        expected["coverage"].append(result.coverage)
        expected["balance"].append(result.balance)
        expected["shift_support"].append(result.shift_support)
    for name, schema in ANALYSIS_SCHEMAS.items():
        got = out[name].collect().sort(schema.keys())
        want = pl.concat(expected[name], how="vertical").sort(schema.keys()) if expected[name] else pl.DataFrame(schema=schema)
        assert_frame_equal(got, want)
    assert set(out["state_statistics"].select("actor_key").collect().get_column("actor_key")) == set(actor_keys)
    assert out["summary_contributions"].collect().height > 0


def test_partition_analysis_resume_verifies_key_without_reaccounting(tmp_path, monkeypatch):
    raw = raw_fixture()
    risk_path = cache_risk(tmp_path, raw)
    output_path = tmp_path / "analysis"
    m = api()
    m.analyze_partition(risk_path, risk_cache_key="risk-key", clusters=clusters(raw), output_path=output_path,
                        analysis_cache_key="analysis-key", buffer_rows=1)

    def forbidden(*args, **kwargs):
        pytest.fail("verified resume must not invoke per-actor accounting")

    monkeypatch.setattr(m, "analyze_controls_v2", forbidden)
    resumed = m.analyze_partition(risk_path, risk_cache_key="risk-key", clusters=clusters(raw), output_path=output_path,
                                 analysis_cache_key="analysis-key", buffer_rows=1, resume=True)
    assert set(resumed) == set(ANALYSIS_SCHEMAS)
    with pytest.raises(ValueError, match="cache"):
        m.analyze_partition(risk_path, risk_cache_key="risk-key", clusters=clusters(raw), output_path=output_path,
                            analysis_cache_key="changed", buffer_rows=1, resume=True)


def test_partition_analysis_rejects_orphan_clusters_even_against_empty_risk(tmp_path):
    risk_path = tmp_path / "empty-risk"
    with PartitionWriter(risk_path, cache_key="risk-key", schemas=SCHEMAS, buffer_rows=1) as writer:
        writer.complete()
    orphan = pl.DataFrame([{
        "instrument": "TEST", "partition_id": "missing", "event_date": datetime(2024, 1, 2).date(),
        "actor_key": "client_original:C1", "actor_id": "C1", "identity_level": "client_original",
        "identity_source": "NMSC_ORIGINALCLIENTIDSHORTCODE", "identity_fallback_flag": False,
        "event_side": "ask", "execution_anchor_mode": "passive", "cluster_end_ts": datetime(2024, 1, 2, 9, 30),
        "cluster_last_sort_index": 1, "execution_cluster_id": "orphan", "execution_quantity": 1.0,
    }], schema=CLUSTER_SCHEMA, strict=True)
    with pytest.raises(ValueError, match="orphan.*cluster"):
        api().analyze_partition(risk_path, risk_cache_key="risk-key", clusters=orphan, output_path=tmp_path / "analysis",
                                analysis_cache_key="analysis-key", buffer_rows=1)
    assert not (tmp_path / "analysis").exists()


def test_empty_analysis_preserves_all_schemas(tmp_path):
    risk_path = tmp_path / 'risk'
    with PartitionWriter(risk_path, cache_key='risk-key', schemas=SCHEMAS, buffer_rows=1) as writer:
        writer.complete()
    output = api().analyze_partition(
        risk_path, risk_cache_key='risk-key', clusters=pl.DataFrame(schema=CLUSTER_SCHEMA),
        output_path=tmp_path / 'out', analysis_cache_key='key', buffer_rows=1,
    )
    for name, schema in ANALYSIS_SCHEMAS.items():
        frame = output[name].collect()
        assert frame.is_empty()
        assert dict(frame.schema) == schema


def test_resume_binds_concrete_schedule_even_if_caller_reuses_cache_label(tmp_path):
    raw = raw_fixture()
    risk_path = cache_risk(tmp_path, raw)
    args = dict(risk_cache_key='risk-key', output_path=tmp_path / 'out',
                analysis_cache_key='same-label', buffer_rows=1)
    api().analyze_partition(risk_path, clusters=clusters(raw), **args)
    changed = clusters(raw).with_columns(pl.lit(9).alias('cluster_last_sort_index'))
    with pytest.raises(ValueError, match='cache'):
        api().analyze_partition(risk_path, clusters=changed, resume=True, **args)


def test_market_input_is_filtered_before_actor_accounting(tmp_path, monkeypatch):
    raw = raw_fixture()
    risk_path = cache_risk(tmp_path, raw)
    m = api()
    original = m.analyze_controls_v2
    def inspect(intervals, *args, market_intervals):
        start = intervals['start_ts'].min()
        end = intervals['end_ts'].max()
        assert market_intervals.filter((pl.col('end_ts') <= start) | (pl.col('start_ts') >= end)).is_empty()
        return original(intervals, *args, market_intervals=market_intervals)
    monkeypatch.setattr(m, 'analyze_controls_v2', inspect)
    m.analyze_partition(risk_path, risk_cache_key='risk-key', clusters=clusters(raw),
                        output_path=tmp_path / 'out', analysis_cache_key='key', buffer_rows=1)


def test_partition_analysis_interruption_never_publishes_or_appends(tmp_path, monkeypatch):
    raw = raw_fixture()
    risk_path = cache_risk(tmp_path, raw)
    m = api()
    original = m.analyze_controls_v2
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return original(*args, **kwargs)

    monkeypatch.setattr(m, "analyze_controls_v2", interrupted)
    args = dict(risk_path=risk_path, risk_cache_key="risk-key", clusters=clusters(raw), output_path=tmp_path / "analysis",
                analysis_cache_key="analysis-key", buffer_rows=1)
    with pytest.raises(RuntimeError, match="interruption"):
        m.analyze_partition(**args)
    assert not args["output_path"].exists()
    assert list(tmp_path.glob(".analysis.incomplete-*"))
    monkeypatch.setattr(m, "analyze_controls_v2", original)
    out = m.analyze_partition(**args, resume=True)
    assert out["state_statistics"].collect().height > 0
