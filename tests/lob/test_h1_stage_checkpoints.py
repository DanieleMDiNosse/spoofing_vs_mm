from __future__ import annotations

import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest
from polars.testing import assert_frame_equal


ROOT = Path(__file__).resolve().parents[2]
STORE_PATH = ROOT / "scripts" / "h1_stage_checkpoints.py"


def load_store_module():
    spec = importlib.util.spec_from_file_location("h1_stage_checkpoints", STORE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def nonempty_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "event_time": [
                __import__("datetime").datetime(2024, 1, 2, 3, 4, 5),
                None,
            ],
            "nullable_count": [1, None],
            "label": ["first", None],
        },
        schema={
            "event_time": pl.Datetime("ns"),
            "nullable_count": pl.Int64,
            "label": pl.String,
        },
    )


def test_cold_then_warm_reuses_one_frame_preserving_schema_nulls_and_logs(tmp_path: Path):
    store_module = load_store_module()
    frame = nonempty_frame()
    original = frame.clone()
    calls = 0
    stderr = io.StringIO()

    def produce() -> pl.DataFrame:
        nonlocal calls
        calls += 1
        return frame

    with store_module.CheckpointStore(tmp_path / "checkpoints", {"input": "abc", "version": 1}, stderr=stderr) as store:
        cold = store.get_or_compute("replay", produce)
        warm = store.get_or_compute("replay", produce)

    assert calls == 1
    assert cold is frame
    assert_frame_equal(warm, original, check_dtypes=True)
    assert warm.schema == original.schema
    assert_frame_equal(frame, original, check_dtypes=True)
    records = [json.loads(line) for line in stderr.getvalue().splitlines()]
    assert records == [
        {"event": "checkpoint_stage", "format": "h1_stage_checkpoint_v1", "stage": "replay", "status": "computed"},
        {"event": "checkpoint_stage", "format": "h1_stage_checkpoint_v1", "stage": "replay", "status": "reused"},
    ]


def test_missing_identity_cannot_rebind_existing_completed_stage(tmp_path: Path):
    module = load_store_module()
    root = tmp_path / "checkpoints"
    with module.CheckpointStore(root, {"input": "original"}) as store:
        store.get_or_compute("replay", nonempty_frame)
    (root / "identity.json").unlink()
    with pytest.raises(ValueError, match="missing identity"):
        with module.CheckpointStore(root, {"input": "different"}):
            pass


def test_tuple_round_trip_and_completed_stage_is_not_overwritten(tmp_path: Path):
    store_module = load_store_module()
    frames = (nonempty_frame(), pl.DataFrame({"order": [2, 1], "ok": [True, False]}))
    calls = 0

    def produce() -> tuple[pl.DataFrame, pl.DataFrame]:
        nonlocal calls
        calls += 1
        return frames

    with store_module.CheckpointStore(tmp_path / "checkpoints", {"identity": "same"}) as store:
        first = store.get_or_compute("reversion", produce)
        second = store.get_or_compute("reversion", produce)

    assert calls == 1
    assert isinstance(first, tuple)
    assert isinstance(second, tuple)
    assert len(second) == 2
    for actual, expected in zip(second, frames, strict=True):
        assert_frame_equal(actual, expected, check_dtypes=True)
    manifest = json.loads((tmp_path / "checkpoints" / "reversion" / "manifest.json").read_text())
    assert manifest["result_kind"] == "tuple"
    assert [item["row_count"] for item in manifest["frames"]] == [2, 2]
    assert [item["schema"] for item in manifest["frames"]][0] == [
        {"name": name, "dtype": str(dtype)} for name, dtype in frames[0].schema.items()
    ]


def test_incomplete_directory_is_never_reused_after_interruption(tmp_path: Path):
    store_module = load_store_module()
    root = tmp_path / "checkpoints"
    with store_module.CheckpointStore(root, {"identity": "same"}):
        pass
    (root / ".replay-interrupted.incomplete").mkdir()
    calls = 0

    def interrupted() -> pl.DataFrame:
        nonlocal calls
        calls += 1
        raise KeyboardInterrupt("forced interruption")

    with store_module.CheckpointStore(root, {"identity": "same"}) as store:
        with pytest.raises(KeyboardInterrupt, match="forced interruption"):
            store.get_or_compute("replay", interrupted)
        assert not (root / "replay").exists()
        result = store.get_or_compute("replay", nonempty_frame)

    assert calls == 1
    assert result.height == 2
    assert (root / "replay" / "manifest.json").is_file()


def test_interruption_while_writing_partial_frame_never_publishes_stage(tmp_path: Path, monkeypatch):
    store_module = load_store_module()
    root = tmp_path / "checkpoints"
    original_write = pl.DataFrame.write_parquet

    def write_then_interrupt(self, path, *args, **kwargs):
        original_write(self, path, *args, **kwargs)
        raise KeyboardInterrupt("forced write interruption")

    monkeypatch.setattr(pl.DataFrame, "write_parquet", write_then_interrupt)
    with store_module.CheckpointStore(root, {"identity": "same"}) as store:
        with pytest.raises(KeyboardInterrupt, match="forced write interruption"):
            store.get_or_compute("replay", nonempty_frame)
        assert not (root / "replay").exists()
        assert not list(root.glob(".replay-*.incomplete"))


@pytest.mark.parametrize("target", ["frame", "manifest"])
def test_corrupt_completed_stage_fails_closed(tmp_path: Path, target: str):
    store_module = load_store_module()
    root = tmp_path / "checkpoints"
    with store_module.CheckpointStore(root, {"identity": "same"}) as store:
        store.get_or_compute("replay", nonempty_frame)
    stage = root / "replay"
    if target == "frame":
        (stage / "frame_000.parquet").write_bytes(b"truncated")
    else:
        (stage / "manifest.json").write_text("{not json")

    with store_module.CheckpointStore(root, {"identity": "same"}) as store:
        with pytest.raises(store_module.CheckpointCorruptionError):
            store.get_or_compute("replay", lambda: pytest.fail("corrupt stage must not recompute"))


def test_identity_mismatch_invalid_stage_and_invalid_output_are_rejected(tmp_path: Path):
    store_module = load_store_module()
    root = tmp_path / "checkpoints"
    with store_module.CheckpointStore(root, {"identity": {"raw": "one"}}) as store:
        with pytest.raises(ValueError, match="stage"):
            store.get_or_compute("../escape", nonempty_frame)
        with pytest.raises(TypeError, match="polars.DataFrame"):
            store.get_or_compute("replay", lambda: [nonempty_frame()])
        assert not (root / "replay").exists()
        store.get_or_compute("replay", nonempty_frame)

    identity_path = root / "identity.json"
    identity_path.write_text('{ "raw": "one" }')
    with pytest.raises(ValueError, match="identity"):
        with store_module.CheckpointStore(root, {"raw": "one"}):
            pass
    identity_path.write_text('{"raw":"one"}')
    with pytest.raises(ValueError, match="identity"):
        with store_module.CheckpointStore(root, {"raw": "two"}):
            pass


def test_context_lock_is_exclusive_and_nonblocking(tmp_path: Path):
    root = tmp_path / "checkpoints"
    script = f"""
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('h1_stage_checkpoints_child', {str(STORE_PATH)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
try:
    with module.CheckpointStore(Path({str(root)!r}), {{'identity': 'same'}}):
        pass
except BlockingIOError:
    raise SystemExit(23)
raise SystemExit(0)
"""
    store_module = load_store_module()
    with store_module.CheckpointStore(root, {"identity": "same"}):
        child = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=False)
    assert child.returncode == 23, child.stderr
