#!/usr/bin/env python
"""Run the existing metrics CLI with in-process, stderr-only H1 telemetry."""
from __future__ import annotations

from contextlib import contextmanager
import functools
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from types import ModuleType
from typing import Any, Callable, Iterator, TextIO


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


class WorkerArguments:
    def __init__(self, *, progress_seconds: float, delegated_argv: list[str]) -> None:
        self.progress_seconds = progress_seconds
        self.delegated_argv = delegated_argv


def parse_worker_args(argv: list[str] | None = None) -> WorkerArguments:
    """Parse only worker flags; all delegated flags must follow ``--`` exactly."""
    values = list(sys.argv[1:] if argv is None else argv)
    if values.count("--") != 1:
        raise ValueError("worker arguments require exactly one `--` separator")
    separator = values.index("--")
    worker_args = values[:separator]
    delegated_argv = values[separator + 1 :]
    if not delegated_argv:
        raise ValueError("worker arguments require delegated compute arguments after `--`")
    if len(worker_args) != 2 or worker_args[0] != "--progress-seconds":
        raise ValueError("worker accepts only `--progress-seconds SECONDS` before `--`")
    try:
        progress_seconds = float(worker_args[1])
    except ValueError as exc:
        raise ValueError("--progress-seconds must be a finite positive number") from exc
    if not math.isfinite(progress_seconds) or progress_seconds <= 0:
        raise ValueError("--progress-seconds must be a finite positive number")
    return WorkerArguments(progress_seconds=progress_seconds, delegated_argv=delegated_argv)


class _Telemetry:
    def __init__(self, stream: TextIO, *, started_at: float | None = None) -> None:
        self.stream = stream
        self.started_at = time.monotonic() if started_at is None else started_at

    def emit(self, event: str, **fields: Any) -> None:
        payload = {
            "format": "h1_worker_telemetry_v1",
            "event": event,
            "elapsed_seconds": round(time.monotonic() - self.started_at, 6),
            **fields,
        }
        print(json.dumps(payload, sort_keys=True, default=str), file=self.stream, flush=True)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        self.emit("stage_enter", stage=name)
        try:
            yield
        except BaseException as exc:
            self.emit("stage_exit", stage=name, status="error", exception=type(exc).__name__)
            raise
        else:
            self.emit("stage_exit", stage=name, status="ok")


class _ReplayProgress:
    def __init__(self, telemetry: _Telemetry, progress_seconds: float) -> None:
        self.telemetry = telemetry
        self.progress_seconds = progress_seconds
        self.event_count = 0
        self.started_at = time.monotonic()
        self.last_report_at = self.started_at

    def observe(self, observation: object) -> None:
        self.event_count += 1
        now = time.monotonic()
        if now - self.last_report_at >= self.progress_seconds:
            self._emit(now, final=False)

    def final(self) -> None:
        if self.event_count:
            self._emit(time.monotonic(), final=True)

    def _emit(self, now: float, *, final: bool) -> None:
        self.telemetry.emit(
            "progress",
            stage="replay",
            event_count=self.event_count,
            # Final progress is emitted after the full compute wrapper returns,
            # so the duration is not replay-only.
            compute_elapsed_seconds=round(now - self.started_at, 6),
            final=final,
        )
        self.last_report_at = now


def _stage_wrapper(
    original: Callable[..., Any],
    *,
    stage: str,
    telemetry: _Telemetry,
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        with telemetry.stage(stage):
            return original(*args, **kwargs)

    return wrapped


def _compute_wrapper(
    original: Callable[..., Any],
    *,
    telemetry: _Telemetry,
    progress_seconds: float,
) -> Callable[..., Any]:
    @functools.wraps(original)
    def wrapped(*args: Any, **kwargs: Any) -> Any:
        existing_observer = kwargs.get("replay_observer")
        progress = _ReplayProgress(telemetry, progress_seconds)

        def combined_observer(observation: object) -> None:
            progress.observe(observation)
            if existing_observer is not None:
                existing_observer(observation)

        kwargs["replay_observer"] = combined_observer
        with telemetry.stage("compute"):
            try:
                return original(*args, **kwargs)
            finally:
                progress.final()

    return wrapped


def _load_compute_module() -> ModuleType:
    path = REPO_ROOT / "scripts" / "compute_spoofing_metrics.py"
    spec = importlib.util.spec_from_file_location("_h1_compute_spoofing_metrics", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load compute entrypoint: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextmanager
def _instrument(
    *,
    compute_module: Any,
    metrics_module: Any,
    telemetry: _Telemetry,
    progress_seconds: float,
) -> Iterator[None]:
    """Temporarily patch only call boundaries; always restore exact originals."""
    restores: list[tuple[Any, str, Any]] = []

    def replace(owner: Any, name: str, value: Any) -> None:
        restores.append((owner, name, getattr(owner, name)))
        setattr(owner, name, value)

    metric_stages = {
        "_stream_metric_inputs": "replay",
        "attach_sci_window_metrics": "sci",
        "_build_execution_cancel_candidates": "cancellation_candidates",
        "_attach_direct_cancellation_window": "cancellation",
        "attach_cancel_anchored_reversion": "reversion",
        "attach_spoofing_compatible_sequence_gate": "sequence_gate",
        "build_candidate_episodes": "episodes",
    }
    try:
        original_compute = getattr(metrics_module, "compute_exploratory_metrics")
        instrumented_compute = _compute_wrapper(
            original_compute,
            telemetry=telemetry,
            progress_seconds=progress_seconds,
        )
        replace(metrics_module, "compute_exploratory_metrics", instrumented_compute)
        if hasattr(compute_module, "compute_exploratory_metrics"):
            replace(compute_module, "compute_exploratory_metrics", instrumented_compute)
        if hasattr(compute_module, "pl"):
            # Proxy only the CLI binding; do not patch the shared Polars module.
            original_polars = compute_module.pl

            class PolarsProxy:
                read_parquet = staticmethod(_stage_wrapper(
                    original_polars.read_parquet, stage="load_parquet", telemetry=telemetry
                ))

                def __getattr__(self, name: str) -> Any:
                    return getattr(original_polars, name)

            replace(compute_module, "pl", PolarsProxy())
        for name, stage in metric_stages.items():
            candidate = getattr(metrics_module, name, None)
            if callable(candidate):
                replace(metrics_module, name, _stage_wrapper(candidate, stage=stage, telemetry=telemetry))
        yield
    finally:
        for owner, name, original in reversed(restores):
            setattr(owner, name, original)


def run(
    argv: list[str] | None = None,
    *,
    compute_module: Any | None = None,
    metrics_module: Any | None = None,
    stderr: TextIO | None = None,
) -> Any:
    """Run the canonical CLI once with ephemeral instrumentation and telemetry."""
    options = parse_worker_args(argv)
    if compute_module is None:
        compute_module = _load_compute_module()
    if metrics_module is None:
        import spoofing_detection.lob.spoofing_metrics as metrics_module

    telemetry = _Telemetry(sys.stderr if stderr is None else stderr)
    telemetry.emit("run_enter", progress_seconds=options.progress_seconds)
    status = "ok"
    try:
        with _instrument(
            compute_module=compute_module,
            metrics_module=metrics_module,
            telemetry=telemetry,
            progress_seconds=options.progress_seconds,
        ):
            with telemetry.stage("compute_main"):
                return compute_module.main(options.delegated_argv)
    except BaseException as exc:
        status = "error"
        telemetry.emit("run_error", exception=type(exc).__name__)
        raise
    finally:
        telemetry.emit("run_exit", status=status)


def main(argv: list[str] | None = None) -> int:
    try:
        run(argv)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
