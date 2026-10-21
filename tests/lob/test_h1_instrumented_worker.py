from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from spoofing_detection.lob import spoofing_metrics


ROOT = Path(__file__).resolve().parents[2]
WORKER_PATH = ROOT / "scripts" / "h1_instrumented_worker.py"
COMPUTE_PATH = ROOT / "scripts" / "compute_spoofing_metrics.py"
CONFIG_PATH = ROOT / "configs" / "spoofing_detection_parameters.json"
KERNEL = {side: {rank: 1.0 for rank in range(1, 11)} for side in ("bid", "ask")}


def load_worker():
    spec = importlib.util.spec_from_file_location("h1_instrumented_worker", WORKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_compute_module():
    spec = importlib.util.spec_from_file_location("h1_compute_metrics", COMPUTE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tiny_v3_fixture() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    events = [
        (1, 1, "B0", 1, 99.0, 100.0, "OTHER", None),
        (2, 1, "A0", 2, 103.0, 100.0, "OTHER", None),
        (3, 1, "D1", 1, 98.0, 100.0, "C", None),
        (4, 1, "P", 2, 102.0, 20.0, "C", None),
        (5, 1, "D2", 1, 97.0, 20.0, "C", None),
        (6, 1, "B1", 1, 100.0, 100.0, "OTHER", None),
        (7, 3, "P", 2, 102.0, 14.0, "C", "passive"),
        (8, 3, "A", 2, 100.0, 0.0, "C", "aggressive"),
        (9, 4, "D1", 1, 98.0, 0.0, "C", None),
        (10, 4, "D2", 1, 97.0, 0.0, "C", None),
        (11, 4, "B1", 1, 100.0, 0.0, "OTHER", None),
        (13, 1, "COVER", 1, 98.0, 1.0, "C", None),
    ]
    base_ns = 1_704_186_000_000_000_000
    for seq, event_type, order_id, side, price, quantity, actor, anchor in events:
        timestamp_ns = base_ns + seq * 1_000_000_000
        rows.append(
            {
                "TRADEDATE": "2024-01-02",
                "MIC": "XMIL",
                "MARKETCODE": "MTA",
                "SYMBOLINDEX": 123,
                "EMM (*)": 1,
                "SEQUENCETIME": timestamp_ns,
                "BOOKIN": timestamp_ns,
                "BOOKOUTTIME": timestamp_ns,
                "TRADETIME": timestamp_ns if anchor else None,
                "HDR_APPLKEYSEQUENCENUMBER": seq,
                "HDR_HWMSEQUENCENUMBER": seq,
                "HDR_OFFSETID": seq,
                "ROW_NUMBER": seq,
                "ORDEREVENTTYPE (*)": event_type,
                "ORDERID": order_id,
                "ORDERPRIORITY": str(seq),
                "ORDERSIDE (*)": side,
                "ORDERPX": price,
                "ORDERQTY": quantity,
                "DISPLAYEDQTY": quantity,
                "LEAVESQTY": quantity,
                "LASTSHARES": 6.0 if anchor else None,
                "LASTTRADEDPX": price if anchor else None,
                "ORDERTYPE (*)": 2,
                "TIMEINFORCE (*)": 0,
                "FIRMID": "F",
                "NMSC_ORIGINALCLIENTIDSHORTCODE": actor,
                "ORDER_TRADINGCAPACITY (*)": 3,
                "PASSIVEORDER": "Y" if anchor == "passive" else "N",
                "AGGRESSIVEORDER": "Y" if anchor == "aggressive" else "N",
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col(name).cast(pl.Datetime("ns"))
        for name in ("SEQUENCETIME", "BOOKIN", "BOOKOUTTIME", "TRADETIME")
    )


def test_worker_delegates_exact_argv_emits_json_lifecycle_and_restores_fake_entrypoint():
    worker = load_worker()
    calls: list[list[str]] = []
    observed: list[object] = []

    def replay(*, replay_observer):
        replay_observer({"sort_index": 7})

    def original_compute(*, replay_observer=None):
        assert replay_observer is not None
        metrics._stream_metric_inputs(replay_observer=replay_observer)
        return "computed"

    metrics = SimpleNamespace(compute_exploratory_metrics=original_compute, _stream_metric_inputs=replay)
    compute = SimpleNamespace(compute_exploratory_metrics=original_compute)

    def fake_main(argv):
        calls.append(argv)
        return compute.compute_exploratory_metrics(replay_observer=observed.append)

    compute.main = fake_main
    telemetry = io.StringIO()
    argv = ["--input", "tiny.parquet", "--output-dir", "out", "--clock-policy-v3"]

    assert worker.run(["--progress-seconds", "30", "--", *argv], compute_module=compute, metrics_module=metrics, stderr=telemetry) == "computed"
    assert calls == [argv]
    assert observed == [{"sort_index": 7}]
    assert compute.compute_exploratory_metrics is original_compute
    assert metrics.compute_exploratory_metrics is original_compute

    records = [json.loads(line) for line in telemetry.getvalue().splitlines()]
    assert all(record["format"] == "h1_worker_telemetry_v1" for record in records)
    assert [record["event"] for record in records][0] == "run_enter"
    assert [record["event"] for record in records][-1] == "run_exit"
    assert any(record["event"] == "stage_enter" and record["stage"] == "compute" for record in records)
    assert any(record["event"] == "stage_enter" and record["stage"] == "replay" for record in records)
    assert any(record["event"] == "progress" and record["event_count"] == 1 for record in records)


def test_worker_fails_fast_for_malformed_flags_and_restores_after_exception():
    worker = load_worker()
    with pytest.raises(ValueError, match="separator"):
        worker.parse_worker_args(["--progress-seconds", "30"])
    with pytest.raises(ValueError, match="finite positive"):
        worker.parse_worker_args(["--progress-seconds", "nan", "--", "--input", "x"])

    def original_compute(*, replay_observer=None):
        return None

    metrics = SimpleNamespace(compute_exploratory_metrics=original_compute)
    compute = SimpleNamespace(compute_exploratory_metrics=original_compute)

    def failing_main(argv):
        raise RuntimeError("boom")

    compute.main = failing_main
    with pytest.raises(RuntimeError, match="boom"):
        worker.run(["--progress-seconds", "1", "--", "--input", "x"], compute_module=compute, metrics_module=metrics, stderr=io.StringIO())
    assert compute.compute_exploratory_metrics is original_compute
    assert metrics.compute_exploratory_metrics is original_compute


def test_worker_v3_fixture_matches_direct_cli_outputs_without_bundle_extras(tmp_path: Path):
    worker = load_worker()
    input_path = tmp_path / "tiny-v3.parquet"
    tiny_v3_fixture().write_parquet(input_path)
    kernel_path = tmp_path / "kernel.csv"
    pl.DataFrame(
        [
            {"side": side, "rank": rank, "kernel_weight": weight}
            for side, weights in KERNEL.items()
            for rank, weight in weights.items()
        ]
    ).write_csv(kernel_path)
    config = json.loads(CONFIG_PATH.read_text())
    config["metrics"].update(
        empirical_depth_kernel=str(kernel_path),
        tick_size=1.0,
        window_seconds=2.0,
        withdrawal_window_seconds=3.0,
        reversion_horizon_seconds=2.0,
        max_deceptive_order_age_seconds=90.0,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    direct_dir = tmp_path / "direct"
    worker_dir = tmp_path / "worker"
    args = [
        "--config", str(config_path), "--input", str(input_path), "--output-dir", str(direct_dir), "--clock-policy-v3",
    ]
    load_compute_module().main(args)

    telemetry = io.StringIO()
    worker.run(
        ["--progress-seconds", "30", "--", "--config", str(config_path), "--input", str(input_path), "--output-dir", str(worker_dir), "--clock-policy-v3"],
        stderr=telemetry,
    )

    assert {path.name for path in worker_dir.iterdir()} == {path.name for path in direct_dir.iterdir()}
    assert pl.read_parquet(direct_dir / "execution_metrics.parquet").height > 0
    for path in direct_dir.rglob("*.parquet"):
        assert_frame_equal(pl.read_parquet(worker_dir / path.relative_to(direct_dir)), pl.read_parquet(path), check_exact=True)
    for path in direct_dir.rglob("*.csv"):
        assert (worker_dir / path.relative_to(direct_dir)).read_bytes() == path.read_bytes()
    records = [json.loads(line) for line in telemetry.getvalue().splitlines()]
    assert any(record.get("stage") == "load_parquet" and record["event"] == "stage_exit" for record in records)
    assert any(record.get("stage") == "sci" and record["event"] == "stage_enter" for record in records)
    assert any(record.get("stage") == "cancellation" and record["event"] == "stage_exit" for record in records)
    assert any(record.get("stage") == "episodes" and record["event"] == "stage_exit" for record in records)


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1", "abc"])
def test_invalid_progress_interval(value):
    with pytest.raises(ValueError, match="finite positive"):
        load_worker().parse_worker_args(["--progress-seconds", value, "--", "--input", "x"])


def test_periodic_progress_and_repeated_invocations_restore_all_bindings(monkeypatch):
    worker = load_worker()
    clock = [0.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    seen = []

    def replay(*, replay_observer):
        for value in (1, 2):
            clock[0] += 31.0
            replay_observer(value)

    def original_compute(*, replay_observer=None):
        metrics._stream_metric_inputs(replay_observer=replay_observer)

    metrics = SimpleNamespace(compute_exploratory_metrics=original_compute, _stream_metric_inputs=replay)
    compute = SimpleNamespace(compute_exploratory_metrics=original_compute, pl=pl)
    original_reader = pl.read_parquet
    compute.main = lambda argv: compute.compute_exploratory_metrics(replay_observer=seen.append)
    for _ in range(2):
        stream = io.StringIO()
        worker.run(["--progress-seconds", "30", "--", "x"], compute_module=compute, metrics_module=metrics, stderr=stream)
        records = [json.loads(line) for line in stream.getvalue().splitlines()]
        assert [r["event_count"] for r in records if r["event"] == "progress" and not r["final"]] == [1, 2]
        assert compute.pl is pl
        assert pl.read_parquet is original_reader
        assert metrics._stream_metric_inputs is replay
        assert compute.compute_exploratory_metrics is original_compute
    assert seen == [1, 2, 1, 2]

    def fail_observer(observation):
        raise RuntimeError("observer failed")

    compute.main = lambda argv: compute.compute_exploratory_metrics(replay_observer=fail_observer)
    stream = io.StringIO()
    with pytest.raises(RuntimeError, match="observer failed"):
        worker.run(["--progress-seconds", "30", "--", "x"], compute_module=compute, metrics_module=metrics, stderr=stream)
    assert compute.pl is pl
    assert metrics._stream_metric_inputs is replay
    assert metrics.compute_exploratory_metrics is original_compute
    assert json.loads(stream.getvalue().splitlines()[-1])["status"] == "error"


def test_progress_duration_is_labeled_compute_not_replay(monkeypatch):
    worker = load_worker()
    clock = [0.0]
    monkeypatch.setattr(worker.time, "monotonic", lambda: clock[0])
    stream = io.StringIO()
    progress = worker._ReplayProgress(worker._Telemetry(stream), progress_seconds=30)

    clock[0] = 5.0
    progress.observe(object())
    # The terminal progress message is emitted after the wrapped compute call,
    # so this duration includes post-replay work and must not claim replay-only
    # semantics.
    clock[0] = 12.0
    progress.final()

    record = json.loads(stream.getvalue().splitlines()[-1])
    assert record["event"] == "progress"
    assert record["compute_elapsed_seconds"] == 12.0
    assert "replay_elapsed_seconds" not in record

