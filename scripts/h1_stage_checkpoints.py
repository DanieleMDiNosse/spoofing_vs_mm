#!/usr/bin/env python
"""Atomic sidecar storage for completed deterministic H1 DataFrame stages."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
from typing import Any, Callable, TextIO, TypeAlias
from uuid import uuid4

import polars as pl


_FRAME: TypeAlias = pl.DataFrame
StageResult: TypeAlias = _FRAME | tuple[_FRAME, ...]
_FORMAT = "h1_stage_checkpoint_v1"
_IDENTITY_NAME = "identity.json"
_LOCK_NAME = ".checkpoint-store.lock"
_STAGE_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")


class CheckpointCorruptionError(RuntimeError):
    """A completed checkpoint is malformed or no longer validates."""


class CheckpointStore:
    """Lock, validate, and atomically publish completed Polars stage outputs."""

    def __init__(self, root: Path, identity: dict[str, Any], *, stderr: TextIO | None = None) -> None:
        if not isinstance(root, Path):
            raise TypeError("root must be a pathlib.Path")
        if not isinstance(identity, dict):
            raise TypeError("identity must be a dict")
        self.root = root
        self._identity_bytes = _canonical_json(identity)
        self._stderr = sys.stderr if stderr is None else stderr
        self._lock_file: TextIO | None = None

    def __enter__(self) -> "CheckpointStore":
        self.root.mkdir(parents=True, exist_ok=True)
        if not self.root.is_dir():
            raise ValueError(f"checkpoint root is not a directory: {self.root}")
        self._lock_file = (self.root / _LOCK_NAME).open("a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            self._lock_file.close()
            self._lock_file = None
            raise
        try:
            self._initialize_or_validate_identity()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        if self._lock_file is not None:
            try:
                fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            finally:
                self._lock_file.close()
                self._lock_file = None

    def get_or_compute(self, stage: str, producer: Callable[[], StageResult]) -> StageResult:
        """Reuse one validated completed stage or atomically publish its output."""
        self._require_open()
        self._validate_stage(stage)
        if not callable(producer):
            raise TypeError("producer must be callable")
        destination = self.root / stage
        if destination.exists():
            result = self._load_completed_stage(destination, stage)
            self._emit(stage, "reused")
            return result

        result = producer()
        frames, result_kind = _validate_result(result)
        self._publish(stage, frames, result_kind)
        self._emit(stage, "computed")
        return result

    def _initialize_or_validate_identity(self) -> None:
        destination = self.root / _IDENTITY_NAME
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != self._identity_bytes:
                raise ValueError("checkpoint identity does not match the existing store")
            return
        if any(path.name != _LOCK_NAME for path in self.root.iterdir()):
            raise ValueError("checkpoint store has missing identity but contains existing artifacts")
        _atomic_write(destination, self._identity_bytes)

    def _publish(self, stage: str, frames: tuple[_FRAME, ...], result_kind: str) -> None:
        destination = self.root / stage
        temporary = self.root / f".{stage}-{uuid4().hex}.incomplete"
        temporary.mkdir()
        try:
            manifest_frames: list[dict[str, Any]] = []
            for index, frame in enumerate(frames):
                filename = f"frame_{index:03d}.parquet"
                path = temporary / filename
                frame.write_parquet(path)
                _fsync_file(path)
                manifest_frames.append(
                    {
                        "filename": filename,
                        "row_count": frame.height,
                        "sha256": _sha256(path),
                        "schema": [
                            {"name": name, "dtype": str(dtype)}
                            for name, dtype in frame.schema.items()
                        ],
                    }
                )
            manifest = {
                "format": _FORMAT,
                "stage": stage,
                "result_kind": result_kind,
                "frames": manifest_frames,
            }
            _write_file(temporary / "manifest.json", _canonical_json(manifest))
            _fsync_directory(temporary)
            # The lock means a compatible process cannot race us. A foreign
            # unexpected directory is never overwritten: fail rather than lose evidence.
            if destination.exists():
                raise CheckpointCorruptionError(f"completed stage appeared during publish: {stage}")
            os.replace(temporary, destination)
            _fsync_directory(self.root)
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _load_completed_stage(self, directory: Path, stage: str) -> StageResult:
        if not directory.is_dir():
            raise CheckpointCorruptionError(f"completed stage is not a directory: {stage}")
        manifest_path = directory / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise CheckpointCorruptionError(f"invalid manifest for stage: {stage}") from exc
        if not isinstance(manifest, dict) or set(manifest) != {"format", "stage", "result_kind", "frames"}:
            raise CheckpointCorruptionError(f"invalid manifest fields for stage: {stage}")
        if manifest["format"] != _FORMAT or manifest["stage"] != stage:
            raise CheckpointCorruptionError(f"incompatible manifest for stage: {stage}")
        result_kind = manifest["result_kind"]
        metadata = manifest["frames"]
        if result_kind not in {"data_frame", "tuple"} or not isinstance(metadata, list) or not metadata:
            raise CheckpointCorruptionError(f"invalid result metadata for stage: {stage}")
        if result_kind == "data_frame" and len(metadata) != 1:
            raise CheckpointCorruptionError(f"invalid data-frame result metadata for stage: {stage}")

        expected_names = {"manifest.json"}
        frames: list[_FRAME] = []
        for index, item in enumerate(metadata):
            frame = self._load_frame(directory, stage, index, item)
            expected_names.add(f"frame_{index:03d}.parquet")
            frames.append(frame)
        try:
            actual_names = {path.name for path in directory.iterdir()}
        except OSError as exc:
            raise CheckpointCorruptionError(f"cannot inspect completed stage: {stage}") from exc
        if actual_names != expected_names:
            raise CheckpointCorruptionError(f"unexpected files in completed stage: {stage}")
        if result_kind == "data_frame":
            return frames[0]
        return tuple(frames)

    def _load_frame(self, directory: Path, stage: str, index: int, metadata: object) -> _FRAME:
        filename = f"frame_{index:03d}.parquet"
        if not isinstance(metadata, dict) or set(metadata) != {"filename", "row_count", "sha256", "schema"}:
            raise CheckpointCorruptionError(f"invalid frame metadata for stage: {stage}")
        if metadata["filename"] != filename or not isinstance(metadata["row_count"], int) or metadata["row_count"] < 0:
            raise CheckpointCorruptionError(f"invalid frame descriptor for stage: {stage}")
        if not isinstance(metadata["sha256"], str) or not isinstance(metadata["schema"], list):
            raise CheckpointCorruptionError(f"invalid frame integrity metadata for stage: {stage}")
        path = directory / filename
        try:
            if not path.is_file() or _sha256(path) != metadata["sha256"]:
                raise CheckpointCorruptionError(f"checksum mismatch for stage: {stage}")
            frame = pl.read_parquet(path)
        except CheckpointCorruptionError:
            raise
        except Exception as exc:
            raise CheckpointCorruptionError(f"cannot read frame for stage: {stage}") from exc
        schema = [{"name": name, "dtype": str(dtype)} for name, dtype in frame.schema.items()]
        if frame.height != metadata["row_count"] or schema != metadata["schema"]:
            raise CheckpointCorruptionError(f"frame metadata mismatch for stage: {stage}")
        return frame

    def _emit(self, stage: str, status: str) -> None:
        print(
            json.dumps(
                {"format": _FORMAT, "event": "checkpoint_stage", "stage": stage, "status": status},
                sort_keys=True,
                separators=(",", ":"),
            ),
            file=self._stderr,
            flush=True,
        )

    def _require_open(self) -> None:
        if self._lock_file is None:
            raise RuntimeError("CheckpointStore must be used as a context manager")

    @staticmethod
    def _validate_stage(stage: str) -> None:
        if not isinstance(stage, str) or not _STAGE_PATTERN.fullmatch(stage):
            raise ValueError("stage must be a non-empty simple key without traversal")


def _validate_result(result: object) -> tuple[tuple[_FRAME, ...], str]:
    if isinstance(result, pl.DataFrame):
        return (result,), "data_frame"
    if isinstance(result, tuple) and result and all(isinstance(frame, pl.DataFrame) for frame in result):
        return result, "tuple"
    raise TypeError("producer must return a polars.DataFrame or non-empty tuple of polars.DataFrame values")


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ValueError("identity and metadata must be canonical JSON values") from exc


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_file(path: Path, contents: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(contents)
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_write(destination: Path, contents: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}-", dir=destination.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(contents)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
