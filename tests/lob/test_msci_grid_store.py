from __future__ import annotations

import hashlib
import json
from pathlib import Path

import polars as pl
import pytest

from spoofing_detection.lob.msci_grid_store import (
    TRAJECTORY_SCHEMA,
    SUMMARY_SCHEMA,
    read_or_compute_partition,
)


_HASH = "a" * 64


def _identity(**changes: object) -> dict[str, object]:
    identity: dict[str, object] = {
        "instrument": "NEXI",
        "partition_id": "2024-01-02",
        "configuration_key": "primary",
        "input_hashes": {"executions": _HASH, "states": "b" * 64},
        "config_hashes": {"primary": "c" * 64, "sensitivity": "d" * 64},
        "code_hashes": {
            "msci_grid": "e" * 64,
            "dirty_worktree": "f" * 64,
            "helpers": {"spoofing_metrics": "0" * 64},
        },
    }
    identity.update(changes)
    return identity


def _requested(rows: list[dict[str, object]] | None = None) -> pl.DataFrame:
    return pl.DataFrame(
        rows
        if rows is not None
        else [
            {
                "instrument": "NEXI",
                "partition_id": "2024-01-02",
                "actor_key": "firm:A",
                "execution_cluster_id": "E1",
            }
        ],
        schema={
            "instrument": pl.String,
            "partition_id": pl.String,
            "actor_key": pl.String,
            "execution_cluster_id": pl.String,
        },
        strict=False,
    )


def _frames(*, null_j: bool = False) -> tuple[pl.DataFrame, pl.DataFrame]:
    trajectory = pl.DataFrame(
        [
            {
                "instrument": "NEXI",
                "partition_id": "2024-01-02",
                "actor_key": "firm:A",
                "execution_cluster_id": "E1",
                "grid_offset_ns": 0,
                "target_replay_ns": 100,
                "selected_sort_index": 10,
                "selected_replay_ns": 100,
                "baseline_sort_index": 9,
                "DWI": 0.0,
                "L_bid_topN": 0.5,
                "L_ask_topN": 0.5,
                "SCI": 0.0,
                "C_opposite": 0.0,
                "C_same": 0.0,
                "MSCI": 0.0,
                "valid": True,
                "reason": None,
            }
        ],
        schema=TRAJECTORY_SCHEMA,
        strict=False,
    )
    summary = pl.DataFrame(
        [
            {
                "instrument": "NEXI",
                "partition_id": "2024-01-02",
                "actor_key": "firm:A",
                "execution_cluster_id": "E1",
                "J": None if null_j else 0.0,
                "J_observed_partial": 0.0,
                "argmax_end_offset_ns": None,
                "argmax_start_offset_ns": None,
                "complete_grid": not null_j,
                "valid_pair_count": 0,
            }
        ],
        schema=SUMMARY_SCHEMA,
        strict=False,
    )
    return trajectory, summary


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_known_invalid_execution_retains_trajectory_and_publishes_complete_manifest(tmp_path: Path):
    calls = 0

    def compute() -> tuple[pl.DataFrame, pl.DataFrame]:
        nonlocal calls
        calls += 1
        return _frames(null_j=True)

    result = read_or_compute_partition(tmp_path, _identity(), _requested(), compute)

    assert result.did_compute is True
    assert calls == 1
    assert result.complete_dir == tmp_path / "NEXI" / "2024-01-02" / "primary" / "complete"
    assert (result.complete_dir / "COMPLETE").is_file()
    manifest = json.loads((result.complete_dir / "manifest.json").read_text())
    assert manifest["identity"] == _identity()
    assert manifest["counts"] == {"trajectory_rows": 1, "summary_rows": 1, "requested_executions": 1}
    assert manifest["outputs"]["trajectory"]["sha256"] == _file_sha256(result.complete_dir / "trajectory.parquet")
    assert manifest["outputs"]["summary"]["sha256"] == _file_sha256(result.complete_dir / "summary.parquet")
    assert result.summary.item(0, "J") is None
    assert result.trajectory.select(["instrument", "partition_id", "actor_key", "execution_cluster_id"]).to_dicts() == _requested().to_dicts()


def test_reuses_verified_exact_identity_without_calling_compute(tmp_path: Path):
    first = read_or_compute_partition(tmp_path, _identity(), _requested(), _frames)
    calls = 0

    def must_not_compute() -> tuple[pl.DataFrame, pl.DataFrame]:
        nonlocal calls
        calls += 1
        raise AssertionError("verified output must be reused")

    reused = read_or_compute_partition(tmp_path, _identity(), _requested(), must_not_compute)

    assert first.did_compute is True
    assert reused.did_compute is False
    assert calls == 0
    assert reused.trajectory.equals(first.trajectory)
    assert reused.summary.equals(first.summary)


def test_rejects_changed_input_identity_without_recomputing_existing_unit(tmp_path: Path):
    read_or_compute_partition(tmp_path, _identity(), _requested(), _frames)
    changed = _identity(input_hashes={"executions": "1" * 64, "states": "b" * 64})

    with pytest.raises(ValueError, match="identity mismatch"):
        read_or_compute_partition(tmp_path, changed, _requested(), lambda: (_ for _ in ()).throw(AssertionError()))


@pytest.mark.parametrize(
    "changed",
    [
        _identity(config_hashes={"primary": "2" * 64, "sensitivity": "d" * 64}),
        _identity(code_hashes={"msci_grid": "3" * 64, "dirty_worktree": "f" * 64, "helpers": {"spoofing_metrics": "0" * 64}}),
    ],
)
def test_rejects_changed_configuration_or_code_identity_without_recompute(tmp_path: Path, changed: dict[str, object]):
    read_or_compute_partition(tmp_path, _identity(), _requested(), _frames)

    with pytest.raises(ValueError, match="identity mismatch"):
        read_or_compute_partition(tmp_path, changed, _requested(), lambda: (_ for _ in ()).throw(AssertionError()))


def test_callback_interruption_leaves_unmarked_attempt_and_next_call_uses_fresh_attempt(tmp_path: Path):
    def interrupted() -> tuple[pl.DataFrame, pl.DataFrame]:
        raise RuntimeError("injected interruption")

    with pytest.raises(RuntimeError, match="injected interruption"):
        read_or_compute_partition(tmp_path, _identity(), _requested(), interrupted)

    unit_dir = tmp_path / "NEXI" / "2024-01-02" / "primary"
    failed_attempts = list(unit_dir.glob(".inprogress-*"))
    assert len(failed_attempts) == 1
    assert not (failed_attempts[0] / "COMPLETE").exists()
    assert not (unit_dir / "complete").exists()

    result = read_or_compute_partition(tmp_path, _identity(), _requested(), _frames)

    assert result.did_compute is True
    assert len(list(unit_dir.glob(".inprogress-*"))) == 1
    assert (unit_dir / "complete" / "COMPLETE").is_file()


def test_rejects_corrupted_parquet_even_when_complete_marker_exists(tmp_path: Path):
    result = read_or_compute_partition(tmp_path, _identity(), _requested(), _frames)
    (result.complete_dir / "trajectory.parquet").write_bytes(b"corrupt")

    with pytest.raises(ValueError, match="trajectory output hash mismatch"):
        read_or_compute_partition(tmp_path, _identity(), _requested(), lambda: (_ for _ in ()).throw(AssertionError()))


def test_explicitly_completes_empty_expected_partition_with_typed_empty_frames(tmp_path: Path):
    empty_trajectory = pl.DataFrame(schema=TRAJECTORY_SCHEMA)
    empty_summary = pl.DataFrame(schema=SUMMARY_SCHEMA)

    result = read_or_compute_partition(
        tmp_path,
        _identity(),
        _requested([]),
        lambda: (empty_trajectory, empty_summary),
    )

    assert result.did_compute is True
    assert result.trajectory.schema == TRAJECTORY_SCHEMA
    assert result.summary.schema == SUMMARY_SCHEMA
    assert result.manifest["counts"] == {
        "requested_executions": 0,
        "trajectory_rows": 0,
        "summary_rows": 0,
    }


def test_rejects_typed_empty_trajectory_for_nonempty_requested_execution_before_publication(tmp_path: Path):
    _, summary = _frames()

    with pytest.raises(ValueError, match="trajectory membership"):
        read_or_compute_partition(
            tmp_path,
            _identity(),
            _requested(),
            lambda: (pl.DataFrame(schema=TRAJECTORY_SCHEMA), summary),
        )

    assert not (tmp_path / "NEXI" / "2024-01-02" / "primary" / "complete").exists()


def test_accepts_each_predefined_sensitivity_unit_at_a_distinct_address(tmp_path: Path):
    configuration_keys = (
        "delta_5ms_h50ms",
        "delta_10ms_h50ms",
        "delta_25ms_h50ms",
        "delta_5ms_h25ms",
        "delta_5ms_h100ms",
    )
    config_hashes = {key: f"{index:x}" * 64 for index, key in enumerate(configuration_keys)}
    complete_dirs = []

    for configuration_key in configuration_keys:
        result = read_or_compute_partition(
            tmp_path,
            _identity(configuration_key=configuration_key, config_hashes=config_hashes),
            _requested(),
            _frames,
        )
        complete_dirs.append(result.complete_dir)

    assert len(set(complete_dirs)) == len(configuration_keys)
    assert all((complete_dir / "COMPLETE").is_file() for complete_dir in complete_dirs)


def test_rejects_configuration_key_absent_from_nonempty_config_hashes(tmp_path: Path):
    with pytest.raises(ValueError, match="configuration_key"):
        read_or_compute_partition(
            tmp_path,
            _identity(configuration_key="delta_5ms_h50ms", config_hashes={"delta_10ms_h50ms": "c" * 64}),
            _requested(),
            _frames,
        )


def test_marker_write_failure_keeps_private_attempt_and_fresh_retry_computes_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    original_write_text = Path.write_text

    def fail_complete_marker(path: Path, text: str, *args: str | None, **kwargs: str | None) -> int:
        if path.name == "COMPLETE":
            raise OSError("injected complete-marker failure")
        return original_write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_complete_marker)

    with pytest.raises(OSError, match="complete-marker failure"):
        read_or_compute_partition(tmp_path, _identity(), _requested(), _frames)

    unit_dir = tmp_path / "NEXI" / "2024-01-02" / "primary"
    failed_attempts = list(unit_dir.glob(".inprogress-*"))
    assert len(failed_attempts) == 1
    assert (failed_attempts[0] / "manifest.json").is_file()
    assert not (failed_attempts[0] / "COMPLETE").exists()
    assert not (unit_dir / "complete").exists()

    monkeypatch.setattr(Path, "write_text", original_write_text)
    retry_calls = 0

    def compute_once() -> tuple[pl.DataFrame, pl.DataFrame]:
        nonlocal retry_calls
        retry_calls += 1
        return _frames()

    result = read_or_compute_partition(tmp_path, _identity(), _requested(), compute_once)

    assert retry_calls == 1
    assert result.did_compute is True
    assert (unit_dir / "complete" / "COMPLETE").is_file()
    assert failed_attempts[0].is_dir()


def test_preserves_unmarked_legacy_complete_then_computes_fresh_attempt(tmp_path: Path):
    unit_dir = tmp_path / "NEXI" / "2024-01-02" / "primary"
    legacy_complete = unit_dir / "complete"
    legacy_complete.mkdir(parents=True)
    (legacy_complete / "legacy.txt").write_text("incomplete legacy evidence\n", encoding="utf-8")
    calls = 0

    def compute_once() -> tuple[pl.DataFrame, pl.DataFrame]:
        nonlocal calls
        calls += 1
        return _frames()

    result = read_or_compute_partition(tmp_path, _identity(), _requested(), compute_once)

    preserved_attempts = list(unit_dir.glob(".interrupted-*"))
    assert calls == 1
    assert len(preserved_attempts) == 1
    assert (preserved_attempts[0] / "legacy.txt").read_text(encoding="utf-8") == "incomplete legacy evidence\n"
    assert result.complete_dir == unit_dir / "complete"
    assert (result.complete_dir / "COMPLETE").is_file()


def test_rejects_duplicate_trajectory_node_before_publication(tmp_path: Path):
    trajectory, summary = _frames()
    duplicate_node = pl.concat([trajectory, trajectory])

    with pytest.raises(ValueError, match="duplicate execution/grid node"):
        read_or_compute_partition(tmp_path, _identity(), _requested(), lambda: (duplicate_node, summary))

    assert not (tmp_path / "NEXI" / "2024-01-02" / "primary" / "complete").exists()


def test_rejects_summary_that_omits_requested_execution_before_publication(tmp_path: Path):
    trajectory, summary = _frames()
    missing_summary = summary.clear()

    with pytest.raises(ValueError, match="summary membership"):
        read_or_compute_partition(tmp_path, _identity(), _requested(), lambda: (trajectory, missing_summary))

    assert not (tmp_path / "NEXI" / "2024-01-02" / "primary" / "complete").exists()
