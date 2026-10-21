from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest
from polars.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[2]
WRAPPER_PATH = ROOT / "scripts" / "h1_checkpointed_worker.py"
STORE_PATH = ROOT / "scripts" / "h1_stage_checkpoints.py"


def load_wrapper():
    spec = importlib.util.spec_from_file_location("h1_checkpointed_worker", WRAPPER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeStore:
    cache: dict[tuple[str, str], object] = {}
    events: list[tuple[str, str]] = []

    def __init__(self, root: Path, identity: dict[str, object]) -> None:
        self.root = root
        self.identity = identity

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def get_or_compute(self, stage: str, producer):
        key = (json.dumps(self.identity, sort_keys=True), stage)
        if key in self.cache:
            self.events.append((stage, "reused"))
            return self.cache[key]
        value = producer()
        self.cache[key] = value
        self.events.append((stage, "computed"))
        return value


class FakeInstrumentedWorker:
    def __init__(self) -> None:
        self.argv: list[str] | None = None

    def run(self, argv, *, compute_module, metrics_module, stderr):
        self.argv = list(argv)
        assert self.argv[:3] == ["--progress-seconds", "1.0", "--"]
        return compute_module.main(self.argv[3:])


def _files(tmp_path: Path) -> tuple[Path, Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    raw = tmp_path / "raw.parquet"
    pl.DataFrame({"x": [1]}).write_parquet(raw)
    kernel = tmp_path / "kernel.csv"
    kernel.write_text("side,rank,kernel_weight\nbid,1,1.0\n")
    config = tmp_path / "config.json"
    config.write_text('{"metrics": {}}')
    return raw, kernel, config


def _modules(
    tmp_path: Path,
    *,
    fail_direct: bool = False,
    duplicate_sci: bool = False,
    failure: dict[str, bool] | None = None,
):
    raw, kernel, config = _files(tmp_path)
    failure = {"direct": fail_direct} if failure is None else failure
    counts: dict[str, int] = {}

    def mark(name: str, value: object):
        counts[name] = counts.get(name, 0) + 1
        return value

    frame = lambda label: pl.DataFrame({"stage": [label]})

    def replay(*args: Any, **kwargs: Any):
        return mark("replay", (frame("state"), frame("execution"), frame("candidate"), frame("cancel"), frame("rejected"), frame("member")))

    def sci(execution, state, **kwargs):
        return mark("sci", execution)

    def candidates(execution, cancel, candidate, **kwargs):
        return mark("candidates", frame("assigned"))

    def direct(execution, cancel, **kwargs):
        if failure["direct"]:
            raise RuntimeError("forced direct failure")
        return mark("direct", execution)

    def reversion(execution, state, assigned, **kwargs):
        return mark("reversion", (execution, assigned))

    def gate(execution):
        return mark("gate", (execution, frame("gate")))

    metrics = SimpleNamespace(
        _stream_metric_inputs=replay,
        attach_sci_window_metrics=sci,
        _build_execution_cancel_candidates=candidates,
        _attach_direct_cancellation_window=direct,
        attach_cancel_anchored_reversion=reversion,
        attach_spoofing_compatible_sequence_gate=gate,
    )

    def original_compute(*args: Any, **kwargs: Any):
        state, execution, candidate, cancel, rejected, member = metrics._stream_metric_inputs()
        execution = metrics.attach_sci_window_metrics(execution, state)
        if duplicate_sci:
            execution = metrics.attach_sci_window_metrics(execution, state)
        assigned = metrics._build_execution_cancel_candidates(execution, cancel, candidate)
        execution = metrics._attach_direct_cancellation_window(execution, cancel, assigned_candidates=assigned)
        execution, assigned = metrics.attach_cancel_anchored_reversion(execution, state, assigned)
        execution, events = metrics.attach_spoofing_compatible_sequence_gate(execution)
        return execution, events, rejected, member

    parsed = SimpleNamespace(
        config=config,
        input=raw,
        quote_panel=None,
        empirical_depth_kernel=kernel,
        output_dir=tmp_path / "output",
        clock_policy_v3=True,
        top_n=10,
    )
    compute = SimpleNamespace(
        compute_exploratory_metrics=original_compute,
        parse_args=lambda argv: parsed,
        main=lambda argv: original_compute(),
    )
    return compute, metrics, counts, parsed


def test_checkpointed_worker_uses_fake_store_for_each_canonical_stage_and_reuses(tmp_path: Path):
    worker = load_wrapper()
    FakeStore.cache = {}
    FakeStore.events = []
    compute, metrics, counts, parsed = _modules(tmp_path)
    fake_instrumented = FakeInstrumentedWorker()
    checkpoint_dir = tmp_path / "checkpoint"
    output_one = tmp_path / "output-one"
    output_two = tmp_path / "output-two"
    argv_one = [
        "--checkpoint-dir", str(checkpoint_dir), "--progress-seconds", "1", "--",
        "--config", str(parsed.config), "--input", str(parsed.input), "--empirical-depth-kernel", str(parsed.empirical_depth_kernel),
        "--output-dir", str(output_one), "--clock-policy-v3",
    ]

    first = worker.run(
        argv_one,
        compute_module=compute,
        metrics_module=metrics,
        instrumented_worker=fake_instrumented,
        store_factory=FakeStore,
        stderr=io.StringIO(),
    )
    assert counts == {"replay": 1, "sci": 1, "candidates": 1, "direct": 1, "reversion": 1, "gate": 1}
    assert [stage for stage, outcome in FakeStore.events if outcome == "computed"] == [
        "replay", "sci", "cancellation_candidates", "direct_cancellation", "reversion", "sequence_gate",
    ]

    parsed.output_dir = output_two
    second = worker.run(
        argv_one[:-3] + ["--output-dir", str(output_two), "--clock-policy-v3"],
        compute_module=compute,
        metrics_module=metrics,
        instrumented_worker=fake_instrumented,
        store_factory=FakeStore,
        stderr=io.StringIO(),
    )
    assert first[0].equals(second[0])
    assert counts == {"replay": 1, "sci": 1, "candidates": 1, "direct": 1, "reversion": 1, "gate": 1}
    assert [stage for stage, outcome in FakeStore.events if outcome == "reused"] == [
        "replay", "sci", "cancellation_candidates", "direct_cancellation", "reversion", "sequence_gate",
    ]
    assert fake_instrumented.argv is not None


def test_checkpointed_worker_restores_exact_functions_after_error_and_rejects_duplicate_stage(tmp_path: Path):
    worker = load_wrapper()
    compute, metrics, _, parsed = _modules(tmp_path, fail_direct=True)
    originals = {name: getattr(metrics, name) for name in worker.CANONICAL_STAGES}
    fake_instrumented = FakeInstrumentedWorker()
    with pytest.raises(RuntimeError, match="forced direct failure"):
        worker.run(
            ["--checkpoint-dir", str(tmp_path / "cache"), "--progress-seconds", "1", "--", "--input", str(parsed.input), "--output-dir", str(tmp_path / "out")],
            compute_module=compute,
            metrics_module=metrics,
            instrumented_worker=fake_instrumented,
            store_factory=FakeStore,
            stderr=io.StringIO(),
        )
    assert {name: getattr(metrics, name) for name in worker.CANONICAL_STAGES} == originals

    compute, metrics, _, parsed = _modules(tmp_path / "duplicate", duplicate_sci=True)
    originals = {name: getattr(metrics, name) for name in worker.CANONICAL_STAGES}
    with pytest.raises(RuntimeError, match="invoked more than once"):
        worker.run(
            ["--checkpoint-dir", str(tmp_path / "cache-two"), "--progress-seconds", "1", "--", "--input", str(parsed.input), "--output-dir", str(tmp_path / "out-two")],
            compute_module=compute,
            metrics_module=metrics,
            instrumented_worker=FakeInstrumentedWorker(),
            store_factory=FakeStore,
            stderr=io.StringIO(),
        )
    assert {name: getattr(metrics, name) for name in worker.CANONICAL_STAGES} == originals


def test_checkpointed_worker_resumes_after_downstream_failure_without_replaying(tmp_path: Path):
    worker = load_wrapper()
    FakeStore.cache = {}
    FakeStore.events = []
    failure = {"direct": True}
    compute, metrics, counts, parsed = _modules(tmp_path, failure=failure)
    argv = [
        "--checkpoint-dir", str(tmp_path / "cache"), "--progress-seconds", "1", "--",
        "--input", str(parsed.input), "--output-dir", str(tmp_path / "first-output"),
    ]
    with pytest.raises(RuntimeError, match="forced direct failure"):
        worker.run(
            argv,
            compute_module=compute,
            metrics_module=metrics,
            instrumented_worker=FakeInstrumentedWorker(),
            store_factory=FakeStore,
            stderr=io.StringIO(),
        )
    assert counts == {"replay": 1, "sci": 1, "candidates": 1}

    failure["direct"] = False
    parsed.output_dir = tmp_path / "retry-output"
    worker.run(
        argv[:-1] + [str(parsed.output_dir)],
        compute_module=compute,
        metrics_module=metrics,
        instrumented_worker=FakeInstrumentedWorker(),
        store_factory=FakeStore,
        stderr=io.StringIO(),
    )
    assert counts == {"replay": 1, "sci": 1, "candidates": 1, "direct": 1, "reversion": 1, "gate": 1}
    assert ("replay", "reused") in FakeStore.events


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "0", "-1", "abc"])
def test_checkpointed_worker_rejects_invalid_progress_interval(value: str):
    worker = load_wrapper()
    with pytest.raises(ValueError, match="finite positive"):
        worker.parse_worker_args(["--checkpoint-dir", "cache", "--progress-seconds", value, "--", "--input", "x"])


def test_checkpointed_worker_rejects_overlapping_checkpoint_and_output_paths(tmp_path: Path):
    worker = load_wrapper()
    with pytest.raises(ValueError, match="must not contain"):
        worker._validate_separate_paths(tmp_path / "output" / "checkpoints", tmp_path / "output")
    with pytest.raises(ValueError, match="must not contain"):
        worker._validate_separate_paths(tmp_path / "checkpoints", tmp_path / "checkpoints" / "output")


@pytest.mark.skipif(not STORE_PATH.exists(), reason="checkpoint store sibling is not available in this work unit")
def test_real_store_cold_warm_v3_parity_and_resume_reuses_replay(tmp_path: Path, monkeypatch):
    """Integration coverage is enabled only after the sibling-owned durable store lands."""
    from test_h1_instrumented_worker import CONFIG_PATH, KERNEL, load_compute_module, tiny_v3_fixture

    worker = load_wrapper()
    input_path = tmp_path / "tiny-v3.parquet"
    tiny_v3_fixture().write_parquet(input_path)
    kernel_path = tmp_path / "kernel.csv"
    pl.DataFrame(
        [{"side": side, "rank": rank, "kernel_weight": weight} for side, weights in KERNEL.items() for rank, weight in weights.items()]
    ).write_csv(kernel_path)
    config = json.loads(CONFIG_PATH.read_text())
    config["metrics"].update(empirical_depth_kernel=str(kernel_path), tick_size=1.0, window_seconds=2.0, withdrawal_window_seconds=3.0, reversion_horizon_seconds=2.0, max_deceptive_order_age_seconds=90.0)
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    cache = tmp_path / "cache"
    cold = tmp_path / "cold"
    warm = tmp_path / "warm"
    base = ["--config", str(config_path), "--input", str(input_path), "--clock-policy-v3"]
    worker.run(["--checkpoint-dir", str(cache), "--progress-seconds", "1", "--", *base, "--output-dir", str(cold)])
    worker.run(["--checkpoint-dir", str(cache), "--progress-seconds", "1", "--", *base, "--output-dir", str(warm)])
    assert {path.relative_to(cold) for path in cold.rglob("*") if path.is_file()} == {path.relative_to(warm) for path in warm.rglob("*") if path.is_file()}
    for path in cold.rglob("*.parquet"):
        assert_frame_equal(pl.read_parquet(path), pl.read_parquet(warm / path.relative_to(cold)), check_exact=True)
    for path in cold.rglob("*.csv"):
        assert path.read_bytes() == (warm / path.relative_to(cold)).read_bytes()

    from spoofing_detection.lob import spoofing_metrics

    replay_calls = 0
    original_replay = spoofing_metrics._stream_metric_inputs
    original_sci = spoofing_metrics.attach_sci_window_metrics
    fail_sci = {"enabled": True}

    def replay_spy(*args, **kwargs):
        nonlocal replay_calls
        replay_calls += 1
        return original_replay(*args, **kwargs)

    def failing_sci(*args, **kwargs):
        if fail_sci["enabled"]:
            raise RuntimeError("forced downstream failure")
        return original_sci(*args, **kwargs)

    monkeypatch.setattr(spoofing_metrics, "_stream_metric_inputs", replay_spy)
    monkeypatch.setattr(spoofing_metrics, "attach_sci_window_metrics", failing_sci)
    resume_cache = tmp_path / "resume-cache"
    with pytest.raises(RuntimeError, match="forced downstream failure"):
        worker.run(["--checkpoint-dir", str(resume_cache), "--progress-seconds", "1", "--", *base, "--output-dir", str(tmp_path / "failed")])
    fail_sci["enabled"] = False
    worker.run(["--checkpoint-dir", str(resume_cache), "--progress-seconds", "1", "--", *base, "--output-dir", str(tmp_path / "resumed")])
    assert replay_calls == 1
    direct = tmp_path / "direct"
    load_compute_module().main([*base, "--output-dir", str(direct)])
    assert pl.read_parquet(direct / "execution_metrics.parquet").height > 0
    for output in (cold, warm, tmp_path / "resumed"):
        assert {p.relative_to(output) for p in output.rglob("*") if p.is_file()} == {p.relative_to(direct) for p in direct.rglob("*") if p.is_file()}
        for path in direct.rglob("*.parquet"):
            assert_frame_equal(pl.read_parquet(path), pl.read_parquet(output / path.relative_to(direct)), check_exact=True)
        for path in direct.rglob("*.csv"):
            assert path.read_bytes() == (output / path.relative_to(direct)).read_bytes()
