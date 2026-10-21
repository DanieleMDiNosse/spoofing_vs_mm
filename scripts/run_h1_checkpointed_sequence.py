#!/usr/bin/env python3
"""Opt-in checkpointed adapter for the H1 sequential rerun supervisor.

The existing supervisor remains unmodified.  This adapter imports it, narrows
its instrument tuple, preserves its generated per-attempt configs as provenance
copies, and uses byte-stable configs plus external per-instrument checkpoints.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
SUPERVISOR_PATH = REPO_ROOT / "scripts" / "run_h1_age30_quantity_only_shared_sequence.py"
CHECKPOINTED_WORKER_PATH = REPO_ROOT / "scripts" / "h1_checkpointed_worker.py"
CHECKPOINT_STORE_PATH = REPO_ROOT / "scripts" / "h1_stage_checkpoints.py"
BUNDLE_VALIDATOR_PATH = REPO_ROOT / "scripts" / "validate_spoofing_detection_bundle.py"
KNOWN_INSTRUMENTS = ("RISANAMENTO", "NEXI", "FERRARI")
DEFAULT_INSTRUMENTS = KNOWN_INSTRUMENTS


class AdapterArguments:
    def __init__(self, *, checkpoint_root: Path, instruments: tuple[str, ...], supervisor_argv: list[str]) -> None:
        self.checkpoint_root = checkpoint_root
        self.instruments = instruments
        self.supervisor_argv = supervisor_argv


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_module(path: Path, module_name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_supervisor() -> ModuleType:
    """Import the unchanged sequential supervisor only when this adapter runs."""
    return _load_module(SUPERVISOR_PATH, "_h1_checkpointed_sequence_supervisor")


def load_bundle_validator() -> ModuleType:
    return _load_module(BUNDLE_VALIDATOR_PATH, "_h1_checkpointed_sequence_validator")


def parse_adapter_args(argv: list[str] | None = None) -> AdapterArguments:
    values = list(sys.argv[1:] if argv is None else argv)
    if values.count("--") != 1:
        raise ValueError("adapter arguments require exactly one `--` separator")
    separator = values.index("--")
    adapter_argv = values[:separator]
    supervisor_argv = values[separator + 1 :]
    if not supervisor_argv:
        raise ValueError("existing supervisor arguments are required after `--`")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument(
        "--instruments",
        default=",".join(DEFAULT_INSTRUMENTS),
        help="Comma-separated subset in execution order (default: RISANAMENTO,NEXI,FERRARI).",
    )
    args = parser.parse_args(adapter_argv)
    selected = tuple(part.strip() for part in args.instruments.split(",") if part.strip())
    if not selected:
        raise ValueError("--instruments must contain at least one instrument")
    if len(set(selected)) != len(selected):
        raise ValueError("--instruments must contain unique instruments")
    unknown = tuple(item for item in selected if item not in KNOWN_INSTRUMENTS)
    if unknown:
        raise ValueError("--instruments contains unknown instruments: " + ", ".join(unknown))
    return AdapterArguments(
        checkpoint_root=args.checkpoint_root,
        instruments=selected,
        supervisor_argv=supervisor_argv,
    )


def _output_root_from_supervisor_argv(argv: list[str]) -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--output-root", type=Path, required=True)
    args, _ = parser.parse_known_args(argv)
    return args.output_root


def _reject_overlapping_roots(output_root: Path, checkpoint_root: Path) -> None:
    output = output_root.expanduser().resolve(strict=False)
    checkpoint = checkpoint_root.expanduser().resolve(strict=False)
    if output == checkpoint or output.is_relative_to(checkpoint) or checkpoint.is_relative_to(output):
        raise ValueError("output root and checkpoint root must not contain one another")


def _stable_config_path(checkpoint_root: Path, instrument: str) -> Path:
    return checkpoint_root / "configs" / f"{instrument}.json"


def _require_stable_config(path: Path, contents: bytes) -> None:
    """Create a config once, or fail closed if a retry changes its bytes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(contents)
    except FileExistsError:
        if path.read_bytes() != contents:
            raise ValueError(f"stable checkpoint config bytes differ: {path}")


def _replace_config_argument(compute_argv: list[str], stable_config: Path) -> list[str]:
    rewritten = list(compute_argv)
    positions = [index for index, value in enumerate(rewritten) if value == "--config"]
    if len(positions) != 1 or positions[0] + 1 >= len(rewritten):
        raise ValueError("original supervisor command does not contain exactly one --config PATH argument")
    rewritten[positions[0] + 1] = str(stable_config)
    return rewritten


def _canonical_compute_argv(command: list[str]) -> list[str]:
    if command.count("--") != 1:
        raise ValueError("original supervisor command does not have exactly one worker separator")
    separator = command.index("--")
    compute_argv = command[separator + 1 :]
    if not compute_argv:
        raise ValueError("original supervisor command has no compute arguments")
    return compute_argv


def adapt_build_plan(
    *,
    supervisor: Any,
    original_build_plan: Callable[[Path], tuple[dict[str, Any], list[dict[str, Any]]]],
    checkpoint_root: Path,
    output_root: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Reuse supervisor plan construction, then externalize reusable identity."""
    checkpoint_root = checkpoint_root.expanduser().resolve(strict=False)
    supervisor.WORKER_SCRIPT = CHECKPOINTED_WORKER_PATH
    manifest, plan = original_build_plan(output_root)
    for entry in plan:
        instrument = entry["instrument"]
        if instrument not in supervisor.INSTRUMENTS:
            raise ValueError(f"original supervisor planned an unselected instrument: {instrument}")
        provenance_config = Path(entry["config"])
        config_bytes = provenance_config.read_bytes()
        stable_config = _stable_config_path(checkpoint_root, instrument)
        _require_stable_config(stable_config, config_bytes)
        compute_argv = _replace_config_argument(_canonical_compute_argv(entry["command"]), stable_config)
        checkpoint_dir = checkpoint_root / "instruments" / instrument
        entry["generated_config_provenance_copy"] = str(provenance_config)
        entry["generated_config_provenance_copy_sha256"] = sha256(provenance_config)
        entry["config"] = str(stable_config)
        entry["config_sha256"] = sha256(stable_config)
        entry["checkpoint_dir"] = str(checkpoint_dir)
        entry["command"] = [
            str(supervisor.PYTHON),
            str(CHECKPOINTED_WORKER_PATH),
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--progress-seconds",
            "30",
            "--",
            *compute_argv,
        ]
    manifest.update(
        {
            "checkpoint_root": str(checkpoint_root),
            "adapter": str(Path(__file__).resolve()),
            "adapter_sha256": sha256(Path(__file__).resolve()),
            "checkpointed_worker": {
                "path": str(CHECKPOINTED_WORKER_PATH),
                "sha256": sha256(CHECKPOINTED_WORKER_PATH),
            },
            "checkpoint_store": {
                "path": str(CHECKPOINT_STORE_PATH),
                "sha256": sha256(CHECKPOINT_STORE_PATH),
            },
        }
    )
    return manifest, plan


def _install_native_validator(supervisor: Any) -> None:
    original_validate = supervisor.validate_completed_output
    validator = load_bundle_validator()

    def validate_completed_output(output_dir: Path, expected_config_hash: str) -> dict[str, Any]:
        local_result = original_validate(output_dir, expected_config_hash)
        native_result = validator.validate_bundle(output_dir)
        result = dict(local_result)
        result["native_bundle_validator"] = native_result
        return result

    supervisor.validate_completed_output = validate_completed_output


def main(argv: list[str] | None = None) -> int:
    options = parse_adapter_args(argv)
    output_root = _output_root_from_supervisor_argv(options.supervisor_argv)
    _reject_overlapping_roots(output_root, options.checkpoint_root)
    supervisor = load_supervisor()
    original_build_plan = supervisor.build_plan
    supervisor.INSTRUMENTS = options.instruments
    supervisor.WORKER_SCRIPT = CHECKPOINTED_WORKER_PATH
    supervisor.build_plan = lambda root: adapt_build_plan(
        supervisor=supervisor,
        original_build_plan=original_build_plan,
        checkpoint_root=options.checkpoint_root,
        output_root=root,
    )
    _install_native_validator(supervisor)
    original_argv = sys.argv
    try:
        sys.argv = [str(SUPERVISOR_PATH), *options.supervisor_argv]
        return supervisor.main()
    finally:
        sys.argv = original_argv


if __name__ == "__main__":
    raise SystemExit(main())
