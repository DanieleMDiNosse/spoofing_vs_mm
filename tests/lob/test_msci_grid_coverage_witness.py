"""Synthetic observed-stream witnesses: never read historical raw inputs."""
from __future__ import annotations

from datetime import datetime
import importlib
import json
from pathlib import Path

import polars as pl
import pytest

from scripts import study_msci_grid as cli
from spoofing_detection.lob.panel import _partition_id


def adapter():
    # Assertion failure, rather than collection failure, records the missing API.
    import importlib.util
    assert importlib.util.find_spec("scripts.msci_grid_coverage_witness") is not None, "coverage adapter missing"
    return importlib.import_module("scripts.msci_grid_coverage_witness")


def events():
    return pl.DataFrame({
        "sort_index": [1, 2, 3, 4],
        "TRADEDATE": [datetime(2024, 1, 1)] * 3 + [datetime(2024, 1, 2)],
        "MIC": ["MTAA"] * 4, "MARKETCODE": ["MTA"] * 4,
        "SYMBOLINDEX": [123] * 4, "EMM (*)": [1] * 4,
        "SEQUENCETIME": ["2024-01-01T00:00:00.000000001", "2024-01-01T00:00:00.000000011", "2024-01-01T00:00:00.000000021", "2024-01-02T00:00:00.000000001"],
        "event_class": ["new_order"] * 4,
    })


def scan_fixture(tmp_path, frame=None, states=None):
    frame = events() if frame is None else frame
    ep = tmp_path / "events.parquet"
    sp = tmp_path / "states.parquet"
    frame.write_parquet(ep)
    base = events()
    from spoofing_detection.lob.spoofing_clock import resolve_spoofing_clocks
    anchors = [{"partition_id": _partition_id(row), "sort_index": row["sort_index"], "replay_time_ns": resolve_spoofing_clocks(row).replay_time_ns} for row in base.iter_rows(named=True)]
    (pl.DataFrame(anchors) if states is None else states).write_parquet(sp)
    return ep, sp


def test_scan_exact_partition_bounds_and_all_anchor_agreement(tmp_path):
    ep, sp = scan_fixture(tmp_path)
    result = adapter().scan_observed_events(ep, sp, expected_count=4)
    assert result["event_count"] == 4
    assert result["distinct_state_anchors"] == 4
    assert len(result["partitions"]) == 2
    first = result["partitions"][0]
    assert first["partition_id"] == "2024-01-01 00:00:00|MTAA|MTA|123|1"
    assert first["first_sort_index"] == 1 and first["last_sort_index"] == 3
    assert first["end_ns"] - first["start_ns"] == 20


@pytest.mark.parametrize("indices", [[1, 2, 2, 4], [1, 2, 4, 5], [2, 3, 4, 5], [1, None, 3, 4]])
def test_scan_rejects_duplicate_missing_noncontiguous_indices(tmp_path, indices):
    ep, sp = scan_fixture(tmp_path, events().with_columns(pl.Series("sort_index", indices)))
    with pytest.raises(ValueError, match="indices"):
        adapter().scan_observed_events(ep, sp, expected_count=4)


@pytest.mark.parametrize("clocks", [
    [None, "2024-01-01T00:00:00.000000011", "2024-01-01T00:00:00.000000021", "2024-01-02T00:00:00.000000001"],
    ["2024-01-01T00:00:00.000000011", "2024-01-01T00:00:00.000000001", "2024-01-01T00:00:00.000000021", "2024-01-02T00:00:00.000000001"],
])
def test_scan_rejects_missing_or_regressing_clocks(tmp_path, clocks):
    ep, sp = scan_fixture(tmp_path, events().with_columns(pl.Series("SEQUENCETIME", clocks)))
    with pytest.raises(ValueError, match="clock|canonical"):
        adapter().scan_observed_events(ep, sp, expected_count=4)


@pytest.mark.parametrize("change", ["partition", "clock", "null", "extra_partition"])
def test_scan_requires_exhaustive_exact_state_anchors(tmp_path, change):
    ep, sp = scan_fixture(tmp_path)
    states = pl.read_parquet(sp)
    if change in {"partition", "extra_partition"}:
        states = states.with_columns(pl.when(pl.col("sort_index") == 1).then(pl.lit("2024-01-01 00:00:00.000000|MTAA|MTA|123|1")).otherwise(pl.col("partition_id")).alias("partition_id"))
    elif change == "clock":
        states = states.with_columns((pl.col("replay_time_ns") + 1).alias("replay_time_ns"))
    else:
        states = states.with_columns(pl.lit(None, dtype=pl.Int64).alias("replay_time_ns"))
    states.write_parquet(sp)
    with pytest.raises(ValueError, match="anchor|partition"):
        adapter().scan_observed_events(ep, sp, expected_count=4)


def test_scan_rejects_count_mismatch_and_physical_noncanonical_order(tmp_path):
    ep, sp = scan_fixture(tmp_path)
    with pytest.raises(ValueError, match="count"):
        adapter().scan_observed_events(ep, sp, expected_count=5)
    events().reverse().write_parquet(ep)
    with pytest.raises(ValueError, match="indices|canonical"):
        adapter().scan_observed_events(ep, sp, expected_count=4)


def historical_fixture(tmp_path):
    from tests.lob.test_msci_grid_cli import _write_historical_checkpoint_fixture, CONFIG
    source, quantity, root = _write_historical_checkpoint_fixture(tmp_path)
    bundle = source / "TEST"
    states = bundle / "actor_metric_time_series.parquet"
    rows = events().head(3)
    pid = _partition_id(rows.row(0, named=True))
    from spoofing_detection.lob.spoofing_clock import resolve_spoofing_clocks
    clocks = [resolve_spoofing_clocks(r).replay_time_ns for r in rows.iter_rows(named=True)]
    pl.read_parquet(states).with_columns(pl.Series("sort_index", [1, 2, 3]), pl.Series("replay_time_ns", clocks), pl.lit(pid).alias("partition_id")).write_parquet(states)
    metadata_path = bundle / "metadata.json"
    meta = json.loads(metadata_path.read_text())
    meta["artifact_hashes"]["state_time_series"] = cli.sha256(states)
    metadata_path.write_text(json.dumps(meta))
    checkpoint = Path(meta["command"][meta["command"].index("--checkpoint-dir") + 1])
    import shutil
    shutil.copyfile(states, checkpoint / "replay/frame_000.parquet")
    manifest = json.loads((checkpoint / "replay/manifest.json").read_text())
    manifest["frames"][0]["sha256"] = cli.sha256(states)
    (checkpoint / "replay/manifest.json").write_text(json.dumps(manifest))
    identity_path = checkpoint / "identity.json"
    identity = json.loads(identity_path.read_text())
    identity["resolved_compute_args"]["max_rows"] = None
    identity_path.write_text(json.dumps(identity))
    qm = json.loads((quantity / "metadata.json").read_text())
    qm["source_provenance"]["TEST"]["metadata_sha256"] = cli.sha256(metadata_path)
    qm["source_provenance"]["TEST"]["source_artifact_sha256"]["state_time_series"] = cli.sha256(states)
    (quantity / "metadata.json").write_text(json.dumps(qm))
    report, code = cli.preflight(CONFIG, source, quantity, tmp_path / "unused", command=["test"], historical_root=root)
    assert code == 0, report
    preflight = tmp_path / "preflight.json"
    preflight.write_text(json.dumps(report))
    cache = tmp_path / "cache"
    cache.mkdir()
    normalized = cache / "normalized_events.parquet"
    rows.write_parquet(normalized)
    cachemeta = cache / "metadata.json"
    cachemeta.write_text(json.dumps({"config": {"include_all_events": True}, "max_rows": None, "source_file": identity["inputs"]["raw"]["path"], "source_file_sha256": identity["inputs"]["raw"]["sha256"], "outputs": {"normalized_events": str(normalized)}, "row_counts": {"input": 3, "normalized_events": 3}, "validation": {"events_processed": 3, "partitions_processed": 1}}))
    return preflight, cachemeta


def test_build_freezes_cache_and_fresh_inputs_without_external_completeness(tmp_path):
    preflight, cachemeta = historical_fixture(tmp_path)
    witness = adapter().build_sidecar(preflight, {"TEST": cachemeta}, assume_protected_cache=True)
    assert witness["support_target"] == "represented_event_stream"
    assert witness["external_feed_completeness_verified"] is False
    entry = witness["instruments"]["TEST"]
    assert entry["historical_normalized_sha256"] is None
    assert entry["partition_coverage_witness"]["segments"][0]["attested"] is True
    assert entry["artifacts"]["state_time_series"] == json.loads(preflight.read_text())["instruments"][0]["artifacts"]["state_time_series"]
    path = tmp_path / "witness.json"
    path.write_text(json.dumps(witness))
    assert adapter().load_sidecar(path, preflight)["instruments"]["TEST"] == entry


@pytest.mark.parametrize("tamper", ["raw", "count", "config", "source", "state", "identity", "cache"])
def test_build_or_reuse_rejects_identity_and_hash_mismatch(tmp_path, tamper):
    preflight, cachemeta = historical_fixture(tmp_path)
    mod = adapter()
    witness = mod.build_sidecar(preflight, {"TEST": cachemeta}, assume_protected_cache=True)
    path = tmp_path / "witness.json"
    path.write_text(json.dumps(witness))
    if tamper in {"raw", "count"}:
        data = json.loads(cachemeta.read_text())
        if tamper == "raw":
            data["source_file_sha256"] = "0" * 64
        else:
            data["row_counts"]["input"] = 4
        cachemeta.write_text(json.dumps(data))
        with pytest.raises(ValueError, match="raw|count"):
            mod.build_sidecar(preflight, {"TEST": cachemeta}, assume_protected_cache=True)
    else:
        entry = witness["instruments"]["TEST"]
        key = {"config": "historical_config", "source": "source_metadata", "state": "state_time_series", "identity": "checkpoint_identity", "cache": "normalized_events"}[tamper]
        target = Path(entry["bindings"][key]["path"])
        target.write_bytes(target.read_bytes() + b"tampered")
        with pytest.raises(ValueError, match="hash|SHA|sha256"):
            mod.load_sidecar(path, preflight)


def test_build_requires_explicit_integrity_assumption(tmp_path):
    preflight, cachemeta = historical_fixture(tmp_path)
    with pytest.raises(ValueError, match="protected"):
        adapter().build_sidecar(preflight, {"TEST": cachemeta}, assume_protected_cache=False)


def test_known_gaps_preserved_and_sampler_cannot_extrapolate_or_create_baseline(tmp_path):
    preflight, cachemeta = historical_fixture(tmp_path)
    mod = adapter()
    raw = mod.build_sidecar(preflight, {"TEST": cachemeta}, assume_protected_cache=True)
    segment = raw["instruments"]["TEST"]["partition_coverage_witness"]["segments"][0]
    gap = {"partition_id": segment["partition_id"], "start_ns": segment["start_ns"] + 15, "end_ns": segment["start_ns"] + 16}
    gaps = tmp_path / "gaps.json"
    gaps.write_text(json.dumps({"TEST": [gap]}))
    witness = mod.build_sidecar(preflight, {"TEST": cachemeta}, assume_protected_cache=True, invalid_intervals_path=gaps)
    coverage = witness["instruments"]["TEST"]["partition_coverage_witness"]
    assert coverage["invalid_intervals"] == [gap]
    from spoofing_detection.lob.msci_grid import sample_execution_grid
    config = json.loads((Path(__file__).parents[2] / "configs/msci_grid_study_v1.json").read_text())
    config.update(grid_step_ns=10, lag_ns=10, horizon_ns=20, grid_sensitivity_ns=[5, 10], lag_sensitivity_ns=[5, 10], lag_sensitivity_grid_ns=5)
    states = pl.read_parquet(witness["instruments"]["TEST"]["artifacts"]["state_time_series"]["path"])
    execution = {"partition_id": segment["partition_id"], "actor_key": "firm:A", "execution_cluster_id": "x", "deceptive_side": "ask", "cluster_first_sort_index": 2, "cluster_last_sort_index": 2, "cluster_end_replay_time_ns": segment["start_ns"] + 10}
    sampled = sample_execution_grid(execution, states, coverage, config)
    assert not sampled["valid"].all()
    assert "known_coverage_gap" in sampled["reason"].to_list()
    missing = sample_execution_grid({**execution, "cluster_first_sort_index": 1, "cluster_last_sort_index": 1, "cluster_end_replay_time_ns": segment["start_ns"]}, states, coverage, config)
    assert "missing_pre_execution_state" in missing["reason"].to_list()
