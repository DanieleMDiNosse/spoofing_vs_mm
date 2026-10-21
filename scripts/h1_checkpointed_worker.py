#!/usr/bin/env python
"""Opt-in deterministic stage checkpoints around the canonical H1 metrics pipeline."""
from __future__ import annotations

from contextlib import contextmanager
import functools
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import platform
import sys
import threading
from types import ModuleType
from typing import Any, Callable, Iterator, TextIO

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

CANONICAL_STAGES = {
    "_stream_metric_inputs": "replay",
    "attach_sci_window_metrics": "sci",
    "_build_execution_cancel_candidates": "cancellation_candidates",
    "_attach_direct_cancellation_window": "direct_cancellation",
    "attach_cancel_anchored_reversion": "reversion",
    "attach_spoofing_compatible_sequence_gate": "sequence_gate",
}
_RUN_LOCK = threading.Lock()


class WorkerArguments:
    def __init__(self, *, checkpoint_dir: Path, progress_seconds: float, delegated_argv: list[str]) -> None:
        self.checkpoint_dir = checkpoint_dir
        self.progress_seconds = progress_seconds
        self.delegated_argv = delegated_argv


def parse_worker_args(argv: list[str] | None = None) -> WorkerArguments:
    """Require worker flags before one separator; delegate the rest unchanged."""
    values = list(sys.argv[1:] if argv is None else argv)
    if values.count("--") != 1:
        raise ValueError("worker arguments require exactly one `--` separator")
    separator = values.index("--")
    worker_args = values[:separator]
    delegated_argv = values[separator + 1 :]
    if not delegated_argv:
        raise ValueError("worker arguments require delegated compute arguments after `--`")
    if len(worker_args) != 4 or worker_args[0] != "--checkpoint-dir" or worker_args[2] != "--progress-seconds":
        raise ValueError("worker accepts only `--checkpoint-dir PATH --progress-seconds SECONDS` before `--`")
    checkpoint_dir = Path(worker_args[1])
    try:
        progress_seconds = float(worker_args[3])
    except ValueError as exc:
        raise ValueError("--progress-seconds must be a finite positive number") from exc
    if not math.isfinite(progress_seconds) or progress_seconds <= 0:
        raise ValueError("--progress-seconds must be a finite positive number")
    return WorkerArguments(
        checkpoint_dir=checkpoint_dir,
        progress_seconds=progress_seconds,
        delegated_argv=delegated_argv,
    )


def _validate_separate_paths(checkpoint_dir: Path, output_dir: Path) -> None:
    checkpoint = checkpoint_dir.expanduser().resolve(strict=False)
    output = output_dir.expanduser().resolve(strict=False)
    if checkpoint == output or checkpoint.is_relative_to(output) or output.is_relative_to(checkpoint):
        raise ValueError("checkpoint directory and output directory must not contain one another")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_hashes() -> dict[str, str]:
    """Hash the whole scientific package plus all process-local execution code."""
    paths = sorted((REPO_ROOT / "src" / "spoofing_detection").rglob("*.py"))
    paths.extend(
        [
            REPO_ROOT / "scripts" / "compute_spoofing_metrics.py",
            REPO_ROOT / "scripts" / "h1_instrumented_worker.py",
            Path(__file__).resolve(),
            REPO_ROOT / "scripts" / "h1_stage_checkpoints.py",
        ]
    )
    hashes: dict[str, str] = {}
    for path in paths:
        key = str(path.resolve().relative_to(REPO_ROOT)) if path.exists() else str(path.resolve())
        hashes[key] = _sha256_file(path) if path.exists() else "missing"
    return hashes


def _normalise(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.expanduser().resolve(strict=False))
    if isinstance(value, dict):
        return {str(key): _normalise(item) for key, item in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (tuple, list)):
        return [_normalise(item) for item in value]
    if isinstance(value, set):
        return sorted((_normalise(item) for item in value), key=lambda item: json.dumps(item, sort_keys=True))
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("resolved compute arguments must be finite")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _file_identity(path: Path | None) -> dict[str, str] | None:
    if path is None:
        return None
    resolved = path.expanduser().resolve(strict=False)
    if not resolved.is_file():
        raise ValueError(f"identity input is not a readable file: {resolved}")
    return {"path": str(resolved), "sha256": _sha256_file(resolved)}


def build_identity(compute_args: Any) -> dict[str, Any]:
    """Bind reuse to resolved scientific settings, inputs, code, and runtime."""
    values = {
        key: _normalise(value)
        for key, value in vars(compute_args).items()
        if key != "output_dir"
    }
    return {
        "format": "h1_checkpoint_identity_v1",
        "resolved_compute_args": dict(sorted(values.items())),
        "inputs": {
            "raw": _file_identity(getattr(compute_args, "input", None)),
            "quote_panel": _file_identity(getattr(compute_args, "quote_panel", None)),
            "empirical_depth_kernel": _file_identity(getattr(compute_args, "empirical_depth_kernel", None)),
            "config": _file_identity(getattr(compute_args, "config", None)),
        },
        "source_sha256": _source_hashes(),
        "runtime": {
            "python": platform.python_version(),
            "polars": pl.__version__,
        },
    }


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_compute_module() -> ModuleType:
    return _load_module(REPO_ROOT / "scripts" / "compute_spoofing_metrics.py", "_h1_checkpoint_compute")


def _load_instrumented_worker() -> ModuleType:
    return _load_module(REPO_ROOT / "scripts" / "h1_instrumented_worker.py", "_h1_checkpoint_instrumented")


def _load_store_factory() -> Callable[[Path, dict[str, Any]], Any]:
    module = _load_module(REPO_ROOT / "scripts" / "h1_stage_checkpoints.py", "_h1_stage_checkpoints")
    factory = getattr(module, "CheckpointStore", None)
    if factory is None:
        raise RuntimeError("h1_stage_checkpoints.py does not expose CheckpointStore")
    return factory


def _emit_checkpoint(stderr: TextIO, *, stage: str, outcome: str) -> None:
    print(
        json.dumps(
            {"format": "h1_checkpoint_wrapper_v1", "event": "checkpoint_stage", "stage": stage, "outcome": outcome},
            sort_keys=True,
        ),
        file=stderr,
        flush=True,
    )


@contextmanager
def _checkpoint_instrument(*, metrics_module: Any, store: Any, stderr: TextIO) -> Iterator[None]:
    """Patch only the six deterministic boundaries and restore exact bindings."""
    restores: list[tuple[Any, str, Any]] = []
    invoked: set[str] = set()

    def replace(name: str, replacement: Callable[..., Any]) -> None:
        original = getattr(metrics_module, name)
        restores.append((metrics_module, name, original))
        setattr(metrics_module, name, replacement)

    try:
        for function_name, stage in CANONICAL_STAGES.items():
            original = getattr(metrics_module, function_name, None)
            if not callable(original):
                raise RuntimeError(f"canonical checkpoint boundary is unavailable: {function_name}")

            @functools.wraps(original)
            def wrapped(*args: Any, __original: Callable[..., Any] = original, __stage: str = stage, **kwargs: Any) -> Any:
                if __stage in invoked:
                    raise RuntimeError(f"checkpoint stage {__stage!r} was invoked more than once in one run")
                invoked.add(__stage)
                computed = False

                def producer() -> Any:
                    nonlocal computed
                    computed = True
                    return __original(*args, **kwargs)

                value = store.get_or_compute(__stage, producer)
                _emit_checkpoint(stderr, stage=__stage, outcome="computed" if computed else "reused")
                return value

            replace(function_name, wrapped)
        yield
    finally:
        for owner, name, original in reversed(restores):
            setattr(owner, name, original)


def run(
    argv: list[str] | None = None,
    *,
    compute_module: Any | None = None,
    metrics_module: Any | None = None,
    instrumented_worker: Any | None = None,
    store_factory: Callable[[Path, dict[str, Any]], Any] | None = None,
    stderr: TextIO | None = None,
) -> Any:
    """Run one canonical compute invocation with reusable completed stage outputs."""
    options = parse_worker_args(argv)
    if not _RUN_LOCK.acquire(blocking=False):
        raise RuntimeError("h1 checkpointed worker does not permit concurrent or reentrant runs")
    try:
        if compute_module is None:
            compute_module = _load_compute_module()
        if metrics_module is None:
            import spoofing_detection.lob.spoofing_metrics as metrics_module
        if instrumented_worker is None:
            instrumented_worker = _load_instrumented_worker()
        using_default_store = store_factory is None
        if using_default_store:
            store_factory = _load_store_factory()
        compute_args = compute_module.parse_args(options.delegated_argv)
        _validate_separate_paths(options.checkpoint_dir, compute_args.output_dir)
        identity = build_identity(compute_args)
        stream = sys.stderr if stderr is None else stderr
        delegated_worker_argv = ["--progress-seconds", str(options.progress_seconds), "--", *options.delegated_argv]
        root = options.checkpoint_dir.expanduser().resolve(strict=False)
        store_context = (
            store_factory(root, identity, stderr=stream)
            if using_default_store
            else store_factory(root, identity)
        )
        with store_context as store:
            with _checkpoint_instrument(metrics_module=metrics_module, store=store, stderr=stream):
                return instrumented_worker.run(
                    delegated_worker_argv,
                    compute_module=compute_module,
                    metrics_module=metrics_module,
                    stderr=stream,
                )
    finally:
        _RUN_LOCK.release()


def main(argv: list[str] | None = None) -> int:
    try:
        run(argv)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr, flush=True)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
