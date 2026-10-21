"""Acceptance tests for the fail-closed MSCI-grid preflight CLI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys

import polars as pl
import pytest

from scripts import msci_grid_study_consumer as consumer
from scripts.study_msci_grid import cohort, preflight
from spoofing_detection.lob.msci_grid import sample_execution_grid


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "study_msci_grid.py"
CONFIG = REPO_ROOT / "configs" / "msci_grid_study_v1.json"
CLOCK_POLICY = "spoofing_detection_sequence_replay_trade_fill_sequence_cancel_v3_utc_provisional"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_fixture_bundle(tmp_path: Path, *, clock_policy: str = CLOCK_POLICY, attest: bool = True) -> tuple[Path, Path]:
    source_root = tmp_path / "source"
    bundle = source_root / "TEST"
    bundle.mkdir(parents=True)
    states = bundle / "states.parquet"
    executions = bundle / "executions.parquet"
    pl.DataFrame(
        {
            "partition_id": ["p", "p", "p"],
            "actor_key": ["firm:A", "firm:A", "firm:A"],
            "sort_index": [9, 10, 11],
            "replay_time_ns": [90, 100, 110],
            "DWI": [0.1, 0.2, 0.3],
            "L_bid_topN": [0.5, 0.5, 0.5],
            "L_ask_topN": [0.5, 0.5, 0.5],
        }
    ).write_parquet(states)
    pl.DataFrame(
        {
            "partition_id": ["p"],
            "actor_key": ["firm:A"],
            "execution_cluster_id": ["e1"],
            "cluster_first_sort_index": [10],
            "cluster_last_sort_index": [10],
            "cluster_end_replay_time_ns": [100],
        }
    ).write_parquet(executions)
    snapshot = bundle / "frozen_producer.py"
    snapshot.write_text("# frozen producer evidence for fixture\n")
    metadata = {
        "clock_policy_id": clock_policy,
        "top_n": 10,
        "window_seconds": 1.0,
        "paths": {"state_time_series": str(states), "execution_metrics": str(executions)},
        "artifact_hashes": {"state_time_series": _sha256(states), "execution_metrics": _sha256(executions)},
        "scientific_source_hashes": {"spoofing_metrics.py": _sha256(snapshot)},
        "row_counts": {"state_time_series": 3, "execution_metrics": 1},
    }
    if attest:
        metadata["state_coverage_attestation"] = {
            "schema": "msci_grid_state_coverage_attestation_v2",
            "state_time_series_sha256": _sha256(states),
            "producer_snapshot": {"path": str(snapshot), "sha256": _sha256(snapshot)},
            "producer_source_key": "spoofing_metrics.py",
            "event_boundary_evaluated": True,
            "transition_zero_emitted": True,
            "sparse_zero_until_reentry": True,
            "state_client_mode": "execution-actors",
        }
    (bundle / "metadata.json").write_text(json.dumps(metadata))

    quantity_root = tmp_path / "quantity"
    quantity_root.mkdir()
    quantity_metadata = {
        "schema": "h1_age30_withdrawal1_wmsci_price_pipeline_v2_complete_finite_rev",
        "contract": {"source_clock_policy_id": CLOCK_POLICY},
        "source_provenance": {
            "TEST": {
                "bundle": str(bundle),
                "metadata_sha256": _sha256(bundle / "metadata.json"),
                "source_artifact_sha256": {
                    "execution_metrics": _sha256(executions),
                    "state_time_series": _sha256(states),
                },
            }
        },
    }
    (quantity_root / "metadata.json").write_text(json.dumps(quantity_metadata))
    return source_root, quantity_root


def _run_preflight(tmp_path: Path, source_root: Path, quantity_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            str(CONFIG),
            "--source-root",
            str(source_root),
            "--quantity-root",
            str(quantity_root),
            "--output-root",
            str(tmp_path / "preflight"),
            "--stage",
            "preflight",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=False,
        env={"PYTHONPATH": "src:.", "PYTHONDONTWRITEBYTECODE": "1"},
    )


def test_preflight_rejects_wrong_clock_or_hash(tmp_path: Path) -> None:
    source_root, quantity_root = _write_fixture_bundle(tmp_path, clock_policy="wrong-clock")
    wrong_clock = _run_preflight(tmp_path, source_root, quantity_root)
    assert wrong_clock.returncode == 2
    clock_report = json.loads((tmp_path / "preflight" / "preflight.json").read_text())
    assert clock_report["status"] == "invalid_input"
    assert any("clock_policy" in blocker for blocker in clock_report["readiness_blockers"])

    source_root, quantity_root = _write_fixture_bundle(tmp_path / "wrong_hash")
    metadata_path = source_root / "TEST" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["artifact_hashes"]["state_time_series"] = "0" * 64
    metadata_path.write_text(json.dumps(metadata))
    wrong_hash = _run_preflight(tmp_path / "wrong_hash", source_root, quantity_root)
    assert wrong_hash.returncode == 2
    hash_report = json.loads((tmp_path / "wrong_hash" / "preflight" / "preflight.json").read_text())
    assert hash_report["status"] == "invalid_input"
    assert any("sha256" in blocker for blocker in hash_report["readiness_blockers"])


def test_preflight_requires_market_update_coverage(tmp_path: Path) -> None:
    source_root, quantity_root = _write_fixture_bundle(tmp_path, attest=False)
    result = _run_preflight(tmp_path, source_root, quantity_root)
    assert result.returncode == 1
    report = json.loads((tmp_path / "preflight" / "preflight.json").read_text())
    assert report["status"] == "blocked_missing_state_evidence"
    assert report["counts"]["market_update_coverage_attested"] == 0
    assert report["counts"]["actor_disappearance_coverage_attested"] == 0
    assert report["counts"]["market_update_coverage_missing"] == 1
    assert report["counts"]["actor_disappearance_coverage_missing"] == 1
    assert report["coverage"]["market_update"]["status"] == "unverified"
    assert report["coverage"]["actor_disappearance"]["status"] == "unverified"
    assert report["diagnostic_execution"] == {
        "status": "not_run",
        "reason": "producer coverage evidence was not verified for every instrument",
    }
    assert report["counts"]["state_rows_observed"] is None
    assert "frozen producer" in " ".join(report["readiness_blockers"]).lower()
    assert "replay" in " ".join(report["minimum_additional_evidence"]).lower()


def test_preflight_scans_exact_t0_and_prior_baseline_without_a_cohort(tmp_path: Path) -> None:
    source_root, quantity_root = _write_fixture_bundle(tmp_path)
    state_path = source_root / "TEST" / "states.parquet"
    pl.read_parquet(state_path).filter(pl.col("sort_index") >= 10).write_parquet(state_path)
    metadata_path = source_root / "TEST" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["artifact_hashes"]["state_time_series"] = _sha256(state_path)
    metadata["state_coverage_attestation"]["state_time_series_sha256"] = _sha256(state_path)
    metadata_path.write_text(json.dumps(metadata))
    quantity_path = quantity_root / "metadata.json"
    quantity = json.loads(quantity_path.read_text())
    quantity["source_provenance"]["TEST"]["metadata_sha256"] = _sha256(metadata_path)
    quantity["source_provenance"]["TEST"]["source_artifact_sha256"]["state_time_series"] = _sha256(state_path)
    quantity_path.write_text(json.dumps(quantity))

    result = _run_preflight(tmp_path, source_root, quantity_root)
    assert result.returncode == 0
    report = json.loads((tmp_path / "preflight" / "preflight.json").read_text())
    assert report["status"] == "ready_for_task6_cohort_gate"
    assert report["counts"]["missing_exact_t0_rows_observed"] == 0
    assert report["counts"]["missing_exact_baseline_rows_observed"] == 1


def test_preflight_and_sampler_allow_equal_clock_canonical_predecessor(tmp_path: Path) -> None:
    """A strictly earlier canonical index may share the execution-end clock."""
    source_root, quantity_root = _write_fixture_bundle(tmp_path)
    state_path = source_root / "TEST" / "states.parquet"
    states = pl.read_parquet(state_path).with_columns(
        pl.Series("replay_time_ns", [100, 100, 120])
    )
    states.write_parquet(state_path)
    _rebind_fixture_artifact(source_root, quantity_root, artifact="state_time_series")

    result = _run_preflight(tmp_path, source_root, quantity_root)

    assert result.returncode == 0
    report = json.loads((tmp_path / "preflight" / "preflight.json").read_text())
    diagnostics = report["instruments"][0]["coverage_diagnostics"]
    assert diagnostics["missing_exact_t0_rows"] == 0
    assert diagnostics["missing_exact_baseline_rows"] == 0
    execution = {
        "partition_id": "p",
        "actor_key": "firm:A",
        "execution_cluster_id": "e1",
        "deceptive_side": "ask",
        "cluster_first_sort_index": 10,
        "cluster_last_sort_index": 10,
        "cluster_end_replay_time_ns": 100,
    }
    config = {
        "schema": "msci_grid_study_v1",
        "source_clock_policy": CLOCK_POLICY,
        "grid_policy": "fixed_pre_exact_post_execution_closed_grid_v1",
        "grid_step_ns": 10,
        "lag_ns": 10,
        "horizon_ns": 20,
        "complete_grid_required": True,
        "cohort": "exact_top_positive_W_5pct_then_U_over_Q_gt_1_before_prices",
        "grid_sensitivity_ns": [5, 10],
        "lag_sensitivity_ns": [5, 10],
        "lag_sensitivity_grid_ns": 5,
        "selection_threshold": None,
    }
    sampled = sample_execution_grid(
        execution,
        states,
        {"segments": [{"partition_id": "p", "start_ns": 0, "end_ns": 200, "attested": True, "source": "test"}]},
        config,
    )
    assert sampled["baseline_sort_index"].unique().to_list() == [9]


@pytest.mark.parametrize(
    ("null_sort_index", "expected_exact_t0", "expected_baseline"),
    [(9, 0, 1), (10, 1, 0)],
)
def test_preflight_diagnoses_null_canonical_anchor_without_fallback(
    tmp_path: Path, null_sort_index: int, expected_exact_t0: int, expected_baseline: int
) -> None:
    source_root, quantity_root = _write_fixture_bundle(tmp_path)
    state_path = source_root / "TEST" / "states.parquet"
    states = pl.read_parquet(state_path)
    if null_sort_index == 9:
        # Index 8 is a valid older candidate, but index 9 is the canonical
        # predecessor and its null clock must not be skipped.
        states = states.with_columns(
            pl.Series("sort_index", [8, 9, 10]),
            pl.Series("replay_time_ns", [80, None, 100]),
        )
    else:
        states = states.with_columns(
            pl.when(pl.col("sort_index") == null_sort_index)
            .then(pl.lit(None, dtype=pl.Int64))
            .otherwise(pl.col("replay_time_ns"))
            .alias("replay_time_ns")
        )
    states.write_parquet(state_path)
    _rebind_fixture_artifact(source_root, quantity_root, artifact="state_time_series")

    result = _run_preflight(tmp_path, source_root, quantity_root)

    assert result.returncode == 0
    diagnostics = json.loads((tmp_path / "preflight" / "preflight.json").read_text())["instruments"][0]["coverage_diagnostics"]
    assert diagnostics["invalid_state_anchor_rows"] == 1
    assert diagnostics["missing_exact_t0_rows"] == expected_exact_t0
    assert diagnostics["missing_exact_baseline_rows"] == expected_baseline


def _rebind_fixture_artifact(
    source_root: Path, quantity_root: Path, *, artifact: str
) -> None:
    """Refresh the fixture's declared hashes after an intentional mutation."""
    bundle = source_root / "TEST"
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    filename = "states.parquet" if artifact == "state_time_series" else "executions.parquet"
    path = bundle / filename
    digest = _sha256(path)
    metadata["artifact_hashes"][artifact] = digest
    if artifact == "state_time_series":
        metadata["state_coverage_attestation"]["state_time_series_sha256"] = digest
    metadata_path.write_text(json.dumps(metadata))
    quantity_path = quantity_root / "metadata.json"
    quantity = json.loads(quantity_path.read_text())
    quantity["source_provenance"]["TEST"]["metadata_sha256"] = _sha256(metadata_path)
    quantity["source_provenance"]["TEST"]["source_artifact_sha256"][artifact] = digest
    quantity_path.write_text(json.dumps(quantity))


def test_preflight_observes_profile_domains_and_exact_t0_clock_mismatch(tmp_path: Path) -> None:
    source_root, quantity_root = _write_fixture_bundle(tmp_path)
    state_path = source_root / "TEST" / "states.parquet"
    pl.DataFrame(
        {
            "partition_id": ["p", "p", "p"],
            "actor_key": ["firm:A", "firm:A", "firm:A"],
            "sort_index": [9, 10, 11],
            "replay_time_ns": [90, 100, 110],
            "DWI": [float("nan"), 1.1, 0.3],
            "L_bid_topN": [0.5, 0.5, float("inf")],
            "L_ask_topN": [0.5, 0.5, 0.5],
        }
    ).write_parquet(state_path)
    execution_path = source_root / "TEST" / "executions.parquet"
    pl.DataFrame(
        {
            "partition_id": ["p"],
            "actor_key": ["firm:A"],
            "execution_cluster_id": ["e1"],
            "cluster_first_sort_index": [10],
            "cluster_last_sort_index": [10],
            "cluster_end_replay_time_ns": [101],
        }
    ).write_parquet(execution_path)
    _rebind_fixture_artifact(source_root, quantity_root, artifact="state_time_series")
    _rebind_fixture_artifact(source_root, quantity_root, artifact="execution_metrics")

    result = _run_preflight(tmp_path, source_root, quantity_root)

    assert result.returncode == 0
    report = json.loads((tmp_path / "preflight" / "preflight.json").read_text())
    diagnostics = report["instruments"][0]["coverage_diagnostics"]
    assert diagnostics["status"] == "observed"
    assert diagnostics["observed_state_rows"] == 3
    assert diagnostics["observed_execution_rows"] == 1
    assert diagnostics["invalid_profile_rows"] == 3
    assert diagnostics["missing_exact_t0_rows"] == 1
    assert report["counts"]["state_rows_observed"] == 3
    assert report["counts"]["execution_rows_observed"] == 1
    assert report["coverage"]["empirical_grid"] == "not_checked_until_sampling"


@pytest.mark.parametrize("mutation", ["boolean_sort_index", "duplicate_sort_index"])
def test_preflight_converts_ambiguous_state_schema_to_invalid_input(tmp_path: Path, mutation: str) -> None:
    source_root, quantity_root = _write_fixture_bundle(tmp_path)
    state_path = source_root / "TEST" / "states.parquet"
    states = pl.read_parquet(state_path)
    if mutation == "boolean_sort_index":
        states = states.with_columns(pl.Series("sort_index", [False, True, True]))
    else:
        states = states.with_columns(pl.Series("sort_index", [9, 10, 10]))
    states.write_parquet(state_path)
    _rebind_fixture_artifact(source_root, quantity_root, artifact="state_time_series")

    result = _run_preflight(tmp_path, source_root, quantity_root)

    assert result.returncode == 2
    report = json.loads((tmp_path / "preflight" / "preflight.json").read_text())
    assert report["status"] == "invalid_input"
    expected = "non-Boolean integral" if mutation == "boolean_sort_index" else "duplicate"
    assert expected in " ".join(report["readiness_blockers"])


def test_preflight_rejects_manifest_path_traversal(tmp_path: Path) -> None:
    source_root, quantity_root = _write_fixture_bundle(tmp_path)
    metadata_path = source_root / "TEST" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["paths"]["state_time_series"] = str(tmp_path / "outside.parquet")
    metadata_path.write_text(json.dumps(metadata))
    result = _run_preflight(tmp_path, source_root, quantity_root)
    assert result.returncode == 2
    report = json.loads((tmp_path / "preflight" / "preflight.json").read_text())
    assert report["status"] == "invalid_input"
    assert any("confined" in blocker for blocker in report["readiness_blockers"])


@pytest.mark.parametrize("missing_column", ["cluster_last_sort_index", "replay_time_ns"])
def test_preflight_accepts_actual_manifest_shape_then_reports_schema_problem(tmp_path: Path, missing_column: str) -> None:
    source_root, quantity_root = _write_fixture_bundle(tmp_path)
    target = source_root / "TEST" / ("executions.parquet" if missing_column.startswith("cluster") else "states.parquet")
    frame = pl.read_parquet(target).drop(missing_column)
    frame.write_parquet(target)
    metadata_path = source_root / "TEST" / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    key = "execution_metrics" if missing_column.startswith("cluster") else "state_time_series"
    metadata["artifact_hashes"][key] = _sha256(target)
    if key == "state_time_series":
        metadata["state_coverage_attestation"]["state_time_series_sha256"] = _sha256(target)
    metadata_path.write_text(json.dumps(metadata))
    quantity = json.loads((quantity_root / "metadata.json").read_text())
    quantity["source_provenance"]["TEST"]["metadata_sha256"] = _sha256(metadata_path)
    quantity["source_provenance"]["TEST"]["source_artifact_sha256"][key] = _sha256(target)
    (quantity_root / "metadata.json").write_text(json.dumps(quantity))
    result = _run_preflight(tmp_path, source_root, quantity_root)
    assert result.returncode == 2
    report = json.loads((tmp_path / "preflight" / "preflight.json").read_text())
    assert report["status"] == "invalid_input"
    assert missing_column in " ".join(report["readiness_blockers"])


def _write_historical_checkpoint_fixture(tmp_path: Path, *, tamper: str | None = None) -> tuple[Path, Path, Path]:
    """Build the frozen H1 evidence shape without changing bundle metadata."""
    historical_root = tmp_path / "historical"
    source_root = historical_root / "run" / "runs"
    bundle = source_root / "TEST"
    bundle.mkdir(parents=True)
    states, executions = bundle / "actor_metric_time_series.parquet", bundle / "execution_metrics.parquet"
    pl.DataFrame({"partition_id": ["p"] * 3, "actor_key": ["firm:A"] * 3, "sort_index": [9, 10, 12], "replay_time_ns": [90, 100, 120], "DWI": [0.4, 0.0, 0.7], "L_bid_topN": [0.2, 0.0, 0.1], "L_ask_topN": [0.6, 0.0, 0.8]}).write_parquet(states)
    pl.DataFrame({"partition_id": ["p"], "actor_key": ["firm:A"], "execution_cluster_id": ["e1"], "cluster_first_sort_index": [10], "cluster_last_sort_index": [10], "cluster_end_replay_time_ns": [100]}).write_parquet(executions)
    checkpoint = historical_root / "checkpoint-cache" / "instruments" / "TEST"
    replay = checkpoint / "replay"
    replay.mkdir(parents=True)
    frame = replay / "frame_000.parquet"
    frame.write_bytes(states.read_bytes())
    source_hashes = {
        "src/spoofing_detection/lob/spoofing_metrics.py": _sha256(REPO_ROOT / "src/spoofing_detection/lob/spoofing_metrics.py"),
        "src/spoofing_detection/lob/replay_observation.py": _sha256(REPO_ROOT / "src/spoofing_detection/lob/replay_observation.py"),
        "scripts/compute_spoofing_metrics.py": _sha256(REPO_ROOT / "scripts/compute_spoofing_metrics.py"),
    }
    config = historical_root / "checkpoint-cache" / "configs" / "TEST.json"
    config.parent.mkdir(parents=True)
    config.write_text('{"metrics":{"top_n":10}}\n')
    raw_sha = "a" * 64
    (checkpoint / "identity.json").write_text(json.dumps({"format": "h1_checkpoint_identity_v1", "source_sha256": source_hashes, "inputs": {"config": {"path": str(config), "sha256": _sha256(config)}, "raw": {"path": "/raw/TEST.parquet", "sha256": raw_sha}}, "resolved_compute_args": {"clock_policy_v3": True, "top_n": 10, "window_seconds": 1.0, "compact_state": True, "state_client_mode": "execution-actors"}}))
    manifest = {"format": "h1_stage_checkpoint_v1", "stage": "replay", "result_kind": "tuple", "frames": [{"filename": "frame_000.parquet", "row_count": 3, "sha256": _sha256(frame), "schema": []}]}
    (replay / "manifest.json").write_text(json.dumps(manifest))
    log = historical_root / "run" / "logs" / "TEST.log"
    log.parent.mkdir(parents=True)
    log.write_text('{"event":"run_exit","status":"ok"}\n')
    metadata = {"clock_policy_id": CLOCK_POLICY, "top_n": 10, "window_seconds": 1.0, "compact_state": True, "state_client_mode": "execution-actors", "config_sha256": _sha256(config), "input_hashes": {"config_sha256": _sha256(config), "raw_events_sha256": raw_sha}, "paths": {"state_time_series": str(states), "execution_metrics": str(executions)}, "artifact_hashes": {"state_time_series": _sha256(states), "execution_metrics": _sha256(executions)}, "scientific_source_hashes": {"spoofing_metrics.py": source_hashes["src/spoofing_detection/lob/spoofing_metrics.py"], "replay_observation.py": source_hashes["src/spoofing_detection/lob/replay_observation.py"]}, "generator_sha256": source_hashes["scripts/compute_spoofing_metrics.py"], "row_counts": {"state_time_series": 3, "execution_metrics": 1}, "command": ["worker", "--checkpoint-dir", str(checkpoint)]}
    if tamper == "frame":
        manifest["frames"][0]["sha256"] = "0" * 64
        (replay / "manifest.json").write_text(json.dumps(manifest))
    elif tamper == "source":
        metadata["scientific_source_hashes"]["spoofing_metrics.py"] = "0" * 64
    elif tamper == "config_raw":
        metadata["input_hashes"]["raw_events_sha256"] = "b" * 64
    elif tamper == "malformed_matching_raw":
        raw_identity = json.loads((checkpoint / "identity.json").read_text())
        raw_identity["inputs"]["raw"]["sha256"] = "not-a-sha256"
        (checkpoint / "identity.json").write_text(json.dumps(raw_identity))
        metadata["input_hashes"]["raw_events_sha256"] = "not-a-sha256"
    elif tamper == "log":
        log.write_text('{"event":"run_exit","status":"failed"}\n')
    elif tamper == "path":
        metadata["command"] = ["worker", "--checkpoint-dir", str(tmp_path / "outside")]
    (bundle / "metadata.json").write_text(json.dumps(metadata))
    quantity_root = tmp_path / "quantity"
    quantity_root.mkdir()
    (quantity_root / "metadata.json").write_text(json.dumps({"schema": "h1_age30_withdrawal1_wmsci_price_pipeline_v2_complete_finite_rev", "contract": {"source_clock_policy_id": CLOCK_POLICY}, "source_provenance": {"TEST": {"bundle": str(bundle), "metadata_sha256": _sha256(bundle / "metadata.json"), "source_artifact_sha256": {"state_time_series": _sha256(states), "execution_metrics": _sha256(executions)}}}}))
    return source_root, quantity_root, historical_root


def test_preflight_accepts_bound_historical_checkpoint_evidence(tmp_path: Path) -> None:
    source_root, quantity_root, historical_root = _write_historical_checkpoint_fixture(tmp_path)
    report, code = preflight(CONFIG, source_root, quantity_root, tmp_path / "unused", command=["test"], historical_root=historical_root)
    assert code == 0
    assert report["status"] == "ready_for_task6_cohort_gate"
    evidence = report["instruments"][0]["coverage_historical_evidence"]
    assert evidence["raw_input"] == {"sha256": "a" * 64}
    assert evidence["semantics"] == {"event_boundary_evaluated": True, "transition_zero_emitted": True, "sparse_zero_until_reentry": True, "state_client_mode": "execution-actors"}
    diagnostics = report["instruments"][0]["coverage_diagnostics"]
    assert diagnostics["status"] == "observed"
    assert (diagnostics["observed_state_rows"], diagnostics["observed_execution_rows"]) == (3, 1)
    assert report["diagnostic_execution"] == {"status": "observed", "instruments_observed": 1}
    assert report["coverage"]["empirical_grid"] == "not_checked_until_sampling"


def test_preflight_rejects_matching_malformed_historical_raw_digests(tmp_path: Path) -> None:
    source_root, quantity_root, historical_root = _write_historical_checkpoint_fixture(tmp_path, tamper="malformed_matching_raw")

    report, code = preflight(CONFIG, source_root, quantity_root, tmp_path / "unused", command=["test"], historical_root=historical_root)

    assert code == 2
    assert report["status"] == "invalid_input"


def test_sparse_producer_zero_is_carried_only_from_transition_until_reentry() -> None:
    """The approved sampler witnesses the historical producer's sparse contract."""
    config = {"schema": "msci_grid_study_v1", "source_clock_policy": CLOCK_POLICY, "grid_policy": "fixed_pre_exact_post_execution_closed_grid_v1", "grid_step_ns": 10, "lag_ns": 10, "horizon_ns": 20, "complete_grid_required": True, "cohort": "exact_top_positive_W_5pct_then_U_over_Q_gt_1_before_prices", "grid_sensitivity_ns": [5, 10], "lag_sensitivity_ns": [5, 10], "lag_sensitivity_grid_ns": 5, "selection_threshold": None}
    states = pl.DataFrame({"partition_id": ["p", "p", "p"], "actor_key": ["firm:A"] * 3, "sort_index": [9, 10, 12], "replay_time_ns": [90, 100, 120], "DWI": [0.4, 0.0, 0.7], "L_bid_topN": [0.2, 0.0, 0.1], "L_ask_topN": [0.6, 0.0, 0.8]})
    execution = {"partition_id": "p", "actor_key": "firm:A", "execution_cluster_id": "e1", "deceptive_side": "ask", "cluster_first_sort_index": 10, "cluster_last_sort_index": 10, "cluster_end_replay_time_ns": 100}
    coverage = {"segments": [{"partition_id": "p", "start_ns": 0, "end_ns": 200, "attested": True, "source": "historical_h1_checkpoint"}]}

    sampled = sample_execution_grid(execution, states, coverage, config)

    absent = sampled.filter(pl.col("grid_offset_ns") == 10).row(0, named=True)
    reentered = sampled.filter(pl.col("grid_offset_ns") == 20).row(0, named=True)
    assert (absent["selected_sort_index"], absent["DWI"]) == (10, 0.0)
    assert (reentered["selected_sort_index"], reentered["DWI"]) == (12, 0.7)


@pytest.mark.parametrize("tamper", ["frame", "source", "config_raw", "log", "path"])
def test_preflight_rejects_tampered_historical_checkpoint_evidence(tmp_path: Path, tamper: str) -> None:
    source_root, quantity_root, historical_root = _write_historical_checkpoint_fixture(tmp_path, tamper=tamper)
    report, code = preflight(CONFIG, source_root, quantity_root, tmp_path / "unused", command=["test"], historical_root=historical_root)
    assert code == 2
    assert report["status"] == "invalid_input"


def _write_cohort_fixture(
    tmp_path: Path, *, mutation: str | None = None, epoch_offset: int = 0
) -> tuple[Path, Path, Path, Path]:
    """Write immutable-like quantity and source artifacts for the Task-6 gate."""
    source_root = tmp_path / "source"
    bundle = source_root / "TEST"
    bundle.mkdir(parents=True)
    executions = bundle / "executions.parquet"
    ids = [f"ec-{index:02d}" for index in range(40)]
    identities = ["client_original" if index == 1 else "firm" for index in range(40)]
    source = pl.DataFrame(
        {
            "partition_id": ["p"] * 40,
            "execution_cluster_id": ids,
            "actor_key": [f"{level}:actor-{index}" for index, level in enumerate(identities)],
            "actor_id": [f"actor-{index}" for index in range(40)],
            "identity_level": identities,
            "identity_source": ["NMSC_ORIGINALCLIENTIDSHORTCODE" if level == "client_original" else "FIRMID" for level in identities],
            "execution_anchor_mode": ["aggressive"] * 40,
            "execution_side": ["bid"] * 40,
            "deceptive_side": ["ask"] * 40,
            "cluster_first_sort_index": list(range(100, 140)),
            "cluster_last_sort_index": list(range(100, 140)),
            "cluster_end_fill_anchor_ns": [epoch_offset + index for index in range(1_000, 1_040)],
        }
    )
    source.write_parquet(executions)
    metadata = {
        "paths": {"execution_metrics": str(executions)},
        "artifact_hashes": {"execution_metrics": _sha256(executions)},
    }
    (bundle / "metadata.json").write_text(json.dumps(metadata))

    quantity_root = tmp_path / "quantity"
    quantity_root.mkdir()
    scores = [10.0, 10.0] + [9.0 - (0.1 * index) for index in range(2, 40)]
    diagnostics = source.with_columns(
        pl.lit("TEST").alias("instrument"),
        pl.Series("WMSCI_event", scores),
        pl.Series("withdrawal_to_execution_ratio", [1.0, 2.0] + [0.5] * 38),
        pl.Series("favorable_mid_move_pre_fill", [None, None] + [-1.0] * 38, dtype=pl.Float64),
        pl.Series("post_cancel_mid_reversion", [None, None] + [-1.0] * 38, dtype=pl.Float64),
        pl.Series("post_cancel_mid_reversion_complete", [False] * 40),
        pl.Series("episode_id", [None, "EP-shared"] + [None] * 38, dtype=pl.String),
    ).select(
        "instrument", "partition_id", "execution_cluster_id", "actor_key", "actor_id", "identity_level", "identity_source",
        "execution_anchor_mode", "execution_side", "deceptive_side", "cluster_first_sort_index", "cluster_last_sort_index",
        "cluster_end_fill_anchor_ns", "WMSCI_event", "withdrawal_to_execution_ratio", "favorable_mid_move_pre_fill",
        "post_cancel_mid_reversion", "post_cancel_mid_reversion_complete", "episode_id",
    )
    memberships = diagnostics.select("instrument", "execution_cluster_id").with_columns(
        pl.lit(True).alias("finite_W"),
        pl.lit(True).alias("positive_W"),
        pl.col("execution_cluster_id").is_in(["ec-00", "ec-01"]).alias("exact_top_positive_W"),
        (pl.col("execution_cluster_id") == "ec-01").alias("U_over_Q_gt_1_finite"),
    ).with_columns(
        (pl.col("exact_top_positive_W") & pl.col("U_over_Q_gt_1_finite")).alias("after_W_then_U_over_Q_gt_1")
    )
    if mutation == "bad_saved_ranking":
        memberships = memberships.with_columns(
            pl.when(pl.col("execution_cluster_id") == "ec-01").then(~pl.col("exact_top_positive_W")).otherwise(pl.col("exact_top_positive_W")).alias("exact_top_positive_W")
        )
    elif mutation is not None and mutation.startswith("null_saved_flag:"):
        flag = mutation.removeprefix("null_saved_flag:")
        memberships = memberships.with_columns(
            pl.when(pl.col("execution_cluster_id") == "ec-01")
            .then(pl.lit(None, dtype=pl.Boolean))
            .otherwise(pl.col(flag))
            .alias(flag)
        )
    elif mutation == "non_boolean_saved_flag":
        memberships = memberships.with_columns(pl.col("finite_W").cast(pl.Int8).alias("finite_W"))
    diagnostics.write_csv(quantity_root / "refreshed_execution_diagnostics.csv")
    memberships.write_csv(quantity_root / "step_memberships.csv")
    output_hashes = {
        name: _sha256(quantity_root / name)
        for name in ("refreshed_execution_diagnostics.csv", "step_memberships.csv")
    }
    quantity_metadata = {
        "schema": "h1_age30_withdrawal1_wmsci_price_pipeline_v2_complete_finite_rev",
        "contract": {
            "top_positive_w_fraction": 0.05,
            "selection_order": ["positive_W_exact_top_5pct", "U_over_Q_gt_1", "FPM_gt_0", "REV_gt_0"],
            "episode_role": "recomputed_shared_order_reporting_only_not_a_selection_gate",
        },
        "outputs": output_hashes,
        "source_provenance": {
            "TEST": {
                "bundle": str(bundle),
                "metadata_sha256": _sha256(bundle / "metadata.json"),
                "source_artifact_sha256": {"execution_metrics": _sha256(executions)},
            }
        },
    }
    (quantity_root / "metadata.json").write_text(json.dumps(quantity_metadata))
    config = tmp_path / "config.json"
    config.write_text(CONFIG.read_text())
    preflight_report = tmp_path / "preflight.json"
    preflight_report.write_text(json.dumps({
        "format": "msci_grid_preflight_v1", "status": "ready_for_task6_cohort_gate",
        "input_digests": {
            "config": {"sha256": _sha256(config)},
            "quantity_metadata": {"sha256": _sha256(quantity_root / "metadata.json")},
            "source_TEST": {"sha256": _sha256(bundle / "metadata.json")},
        },
    }))
    if mutation == "duplicate_source":
        pl.concat([source, source.head(1)]).write_parquet(executions)
    elif mutation == "missing_source":
        source.filter(pl.col("execution_cluster_id") != "ec-01").write_parquet(executions)
    elif mutation == "mismatched_actor":
        source.with_columns(pl.when(pl.col("execution_cluster_id") == "ec-01").then(pl.lit("firm:wrong")).otherwise(pl.col("actor_key")).alias("actor_key")).write_parquet(executions)
    elif mutation == "mismatched_actor_to_null":
        source.with_columns(
            pl.when(pl.col("execution_cluster_id") == "ec-01")
            .then(pl.lit(None, dtype=pl.String))
            .otherwise(pl.col("actor_key"))
            .alias("actor_key")
        ).write_parquet(executions)
    if mutation in {"duplicate_source", "missing_source", "mismatched_actor", "mismatched_actor_to_null"}:
        metadata["artifact_hashes"]["execution_metrics"] = _sha256(executions)
        (bundle / "metadata.json").write_text(json.dumps(metadata))
        quantity_metadata["source_provenance"]["TEST"]["metadata_sha256"] = _sha256(bundle / "metadata.json")
        quantity_metadata["source_provenance"]["TEST"]["source_artifact_sha256"]["execution_metrics"] = _sha256(executions)
        (quantity_root / "metadata.json").write_text(json.dumps(quantity_metadata))
        preflight_report.write_text(json.dumps({
            "format": "msci_grid_preflight_v1", "status": "ready_for_task6_cohort_gate",
            "input_digests": {"config": {"sha256": _sha256(config)}, "quantity_metadata": {"sha256": _sha256(quantity_root / "metadata.json")}, "source_TEST": {"sha256": _sha256(bundle / "metadata.json")}},
        }))
    return source_root, quantity_root, config, preflight_report


def test_cohort_selects_deterministic_top_w_then_strict_quantity_before_prices(tmp_path: Path) -> None:
    source_root, quantity_root, config, preflight_report = _write_cohort_fixture(tmp_path)

    report, code = cohort(config, source_root, quantity_root, tmp_path / "cohort", command=["test"], preflight_report_path=preflight_report)

    assert code == 0
    assert report["status"] == "complete"
    cohort_frame = pl.read_csv(tmp_path / "cohort" / "cohort.csv")
    assert cohort_frame.get_column("execution_cluster_id").to_list() == ["ec-01"]
    row = cohort_frame.row(0, named=True)
    assert row["exact_top_positive_W"] is True
    assert row["U_over_Q_gt_1_finite"] is True
    assert row["identity_level"] == "client_original"
    assert row["identity_source"] == "NMSC_ORIGINALCLIENTIDSHORTCODE"
    assert row["episode_id"] == "EP-shared"
    assert row["favorable_mid_move_pre_fill"] is None
    assert row["post_cancel_mid_reversion"] is None
    assert report["counts"] == {"TEST": {"all_quantity_rows": 40, "positive_finite_W": 40, "exact_top_positive_W": 2, "cohort_after_U_over_Q_gt_1": 1}}


@pytest.mark.parametrize("mutation", ["bad_saved_ranking", "duplicate_source", "missing_source", "mismatched_actor"])
def test_cohort_fails_closed_on_membership_or_one_to_one_source_failure(tmp_path: Path, mutation: str) -> None:
    source_root, quantity_root, config, preflight_report = _write_cohort_fixture(tmp_path, mutation=mutation)

    report, code = cohort(config, source_root, quantity_root, tmp_path / "cohort", command=["test"], preflight_report_path=preflight_report)

    assert code == 2
    assert report["status"] == "invalid_input"
    assert not (tmp_path / "cohort" / "cohort.csv").exists()


@pytest.mark.parametrize(
    "flag",
    [
        "finite_W",
        "positive_W",
        "exact_top_positive_W",
        "U_over_Q_gt_1_finite",
        "after_W_then_U_over_Q_gt_1",
    ],
)
def test_cohort_rejects_each_nullable_saved_membership_flag_without_output(tmp_path: Path, flag: str) -> None:
    source_root, quantity_root, config, preflight_report = _write_cohort_fixture(
        tmp_path, mutation=f"null_saved_flag:{flag}"
    )

    report, code = cohort(config, source_root, quantity_root, tmp_path / "cohort", command=["test"], preflight_report_path=preflight_report)

    assert code == 2
    assert report["status"] == "invalid_input"
    assert "saved quantity membership" in report["error"]
    assert not (tmp_path / "cohort" / "cohort.csv").exists()


def test_cohort_requires_boolean_saved_membership_flag_dtype(tmp_path: Path) -> None:
    source_root, quantity_root, config, preflight_report = _write_cohort_fixture(tmp_path, mutation="non_boolean_saved_flag")

    report, code = cohort(config, source_root, quantity_root, tmp_path / "cohort", command=["test"], preflight_report_path=preflight_report)

    assert code == 2
    assert report["status"] == "invalid_input"
    assert "Boolean" in report["error"]
    assert not (tmp_path / "cohort" / "cohort.csv").exists()


def test_cohort_rejects_required_actor_identity_mismatch_to_null(tmp_path: Path) -> None:
    source_root, quantity_root, config, preflight_report = _write_cohort_fixture(tmp_path, mutation="mismatched_actor_to_null")

    report, code = cohort(config, source_root, quantity_root, tmp_path / "cohort", command=["test"], preflight_report_path=preflight_report)

    assert code == 2
    assert report["status"] == "invalid_input"
    assert "actor_key" in report["error"]
    assert not (tmp_path / "cohort" / "cohort.csv").exists()


def _write_consumer_fixture(
    tmp_path: Path, *, canonical_end_delta: int = 0, coverage_witness: bool = True, epoch_offset: int = 0
) -> tuple[Path, Path, Path, Path, Path]:
    """Build a tiny attested Task-6 cohort for Task-7c CLI integration."""
    source_root, quantity_root, config, preflight = _write_cohort_fixture(tmp_path, epoch_offset=epoch_offset)
    config.write_text(json.dumps({
        "schema": "msci_grid_study_v1", "source_clock_policy": CLOCK_POLICY,
        "grid_policy": "fixed_pre_exact_post_execution_closed_grid_v1",
        "grid_step_ns": 10, "lag_ns": 10, "horizon_ns": 20,
        "complete_grid_required": True,
        "cohort": "exact_top_positive_W_5pct_then_U_over_Q_gt_1_before_prices",
        "grid_sensitivity_ns": [5, 10], "lag_sensitivity_ns": [5, 10],
        "lag_sensitivity_grid_ns": 5, "selection_threshold": None,
    }))
    bundle = source_root / "TEST"
    executions = bundle / "executions.parquet"
    pl.read_parquet(executions).with_columns(
        (pl.col("cluster_end_fill_anchor_ns") + canonical_end_delta).alias("cluster_end_replay_time_ns")
    ).write_parquet(executions)
    states = bundle / "states.parquet"
    pl.DataFrame({
        "partition_id": ["p"] * 4,
        "actor_key": ["client_original:actor-1"] * 4,
        "sort_index": [100, 101, 102, 103],
        "replay_time_ns": [epoch_offset + canonical_end_delta + value for value in [999, 1001, 1011, 1021]],
        "DWI": [0.0, 0.1, 0.2, 0.3],
        "L_bid_topN": [0.5, 0.4, 0.3, 0.2],
        "L_ask_topN": [0.5, 0.6, 0.7, 0.8],
    }).write_parquet(states)
    snapshot = bundle / "producer.py"
    snapshot.write_text("# fixture producer\n")
    metadata_path = bundle / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    metadata.update({
        "clock_policy_id": CLOCK_POLICY, "top_n": 10, "window_seconds": 1.0,
        "paths": {"execution_metrics": str(executions), "state_time_series": str(states)},
        "artifact_hashes": {"execution_metrics": _sha256(executions), "state_time_series": _sha256(states)},
        "scientific_source_hashes": {"producer.py": _sha256(snapshot)},
        "state_coverage_attestation": {
            "schema": "msci_grid_state_coverage_attestation_v2",
            "state_time_series_sha256": _sha256(states),
            "producer_snapshot": {"path": str(snapshot), "sha256": _sha256(snapshot)},
            "producer_source_key": "producer.py", "event_boundary_evaluated": True,
            "transition_zero_emitted": True, "sparse_zero_until_reentry": True,
            "state_client_mode": "execution-actors",
        },
    })
    if coverage_witness:
        metadata["state_coverage_attestation"]["partition_coverage_witness"] = {
            "state_time_series_sha256": _sha256(states),
            "segments": [{
                "partition_id": "p", "start_ns": epoch_offset + 900, "end_ns": epoch_offset + 2_000,
                "attested": True, "source": "fixture_partition_witness",
            }],
            "invalid_intervals": [],
        }
    metadata_path.write_text(json.dumps(metadata))
    quantity_metadata_path = quantity_root / "metadata.json"
    quantity_metadata = json.loads(quantity_metadata_path.read_text())
    quantity_metadata["source_provenance"]["TEST"]["metadata_sha256"] = _sha256(metadata_path)
    quantity_metadata["source_provenance"]["TEST"]["source_artifact_sha256"]["execution_metrics"] = _sha256(executions)
    quantity_metadata_path.write_text(json.dumps(quantity_metadata))
    cohort_root = tmp_path / "cohort"
    report, code = cohort(config, source_root, quantity_root, cohort_root, command=["test"])
    assert code == 0, report
    preflight.write_text(json.dumps({
        "format": "msci_grid_preflight_v1", "status": "ready_for_task6_cohort_gate",
        "input_digests": {
            "config": {"sha256": _sha256(config)},
            "quantity_metadata": {"sha256": _sha256(quantity_metadata_path)},
            "source_TEST": {"sha256": _sha256(metadata_path)},
        },
        "instruments": [{
            "instrument": "TEST", "metadata": {"sha256": _sha256(metadata_path)},
            "artifacts": {"state_time_series": {"sha256": _sha256(states)}, "execution_metrics": {"sha256": _sha256(executions)}},
            "coverage_attestation": metadata["state_coverage_attestation"],
        }],
    }))
    return source_root, quantity_root, config, preflight, cohort_root


def _consumer_command(
    *, source_root: Path, config: Path, preflight: Path, cohort_root: Path,
    output_root: Path, stage: str = "compute", study_mode: str = "full",
    instrument: str = "TEST", max_executions: int | None = None,
) -> list[str]:
    command = [
        sys.executable, str(SCRIPT), "--stage", stage, "--config", str(config),
        "--source-root", str(source_root), "--cohort-root", str(cohort_root),
        "--preflight-report", str(preflight), "--output-root", str(output_root),
        "--study-mode", study_mode, "--instrument", instrument,
    ]
    if max_executions is not None:
        command.extend(["--max-executions", str(max_executions)])
    return command


def test_compute_then_read_only_verify_and_resume_attested_cohort(tmp_path: Path) -> None:
    source_root, _, config, preflight, cohort_root = _write_consumer_fixture(tmp_path)
    output_root = tmp_path / "study"
    command = _consumer_command(
        source_root=source_root, config=config, preflight=preflight,
        cohort_root=cohort_root, output_root=output_root,
    )
    computed = subprocess.run(command, cwd=REPO_ROOT, text=True, capture_output=True, check=False, env={"PYTHONPATH": "src:.", "PYTHONDONTWRITEBYTECODE": "1"})
    assert computed.returncode == 0, computed.stderr
    assert "MSCI_GRID_STUDY_COMPUTED" in computed.stdout
    assert (output_root / "run_manifest.json").is_file()
    manifest = json.loads((output_root / "run_manifest.json").read_text())
    assert manifest["mode"] == "full"
    assert manifest["scope"]["ordering_rule"] == "instrument_then_execution_cluster_id_ascending"
    assert manifest["scope"]["selected_execution_ids"] == manifest["requested_execution_ids"]
    assert set(manifest["software_versions"]) == {"python", "polars"}
    assert set(manifest["partition_validation"]["TEST::p"]) == {
        "coverage_witness_sha256", "validation_contract_sha256"
    }
    verified_command = _consumer_command(
        source_root=source_root, config=config, preflight=preflight,
        cohort_root=cohort_root, output_root=output_root, stage="verify",
    )
    verified = subprocess.run(verified_command, cwd=REPO_ROOT, text=True, capture_output=True, check=False, env={"PYTHONPATH": "src:.", "PYTHONDONTWRITEBYTECODE": "1"})
    assert verified.returncode == 0, verified.stderr
    assert "MSCI_GRID_STUDY_VERIFIED" in verified.stdout
    resumed = subprocess.run([*command, "--resume"], cwd=REPO_ROOT, text=True, capture_output=True, check=False, env={"PYTHONPATH": "src:.", "PYTHONDONTWRITEBYTECODE": "1"})
    assert resumed.returncode == 0, resumed.stderr
    assert "computed=0" in resumed.stdout


@pytest.mark.parametrize("anchor", ["1723024631981001002", "", "1723024631981001001.0"])
def test_compute_rejects_attested_cohort_fill_anchor_mismatch(tmp_path: Path, anchor: str) -> None:
    source_root, _, config, preflight, cohort_root = _write_consumer_fixture(
        tmp_path, canonical_end_delta=318441, epoch_offset=1723024631981000000
    )
    cohort_path = cohort_root / "cohort.csv"
    cohort_path.write_text(cohort_path.read_text().replace("1723024631981001001", anchor))
    provenance_path = cohort_root / "cohort_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["artifacts"]["cohort"]["sha256"] = _sha256(cohort_path)
    provenance_path.write_text(json.dumps(provenance))

    result = subprocess.run(
        _consumer_command(
            source_root=source_root, config=config, preflight=preflight,
            cohort_root=cohort_root, output_root=tmp_path / "study",
        ),
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
        env={"PYTHONPATH": "src:.", "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert result.returncode == 2
    assert "cluster_end_fill_anchor_ns" in result.stderr
    assert not (tmp_path / "study").exists()


@pytest.mark.parametrize("replay_delta", [1, 318441])
def test_epoch_ns_distinct_fill_and_replay_clocks_compute_verify_exactly(tmp_path: Path, replay_delta: int) -> None:
    epoch = 1723024631981000000
    source, _, config, preflight, cohort_root = _write_consumer_fixture(
        tmp_path, canonical_end_delta=replay_delta, epoch_offset=epoch
    )
    context = consumer._load_context(config, source, cohort_root, preflight, "TEST", None)
    assert context["selected"][0]["cluster_end_replay_time_ns"] == epoch + 1001 + replay_delta
    root = tmp_path / "study"
    assert consumer.compute(root, context, resume=False, command=["test"])["computed"] == 1
    assert consumer.verify(root, context, command=["test"])["executions"] == 1


def test_full_mode_rejects_execution_cap_before_loading_inputs(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--stage", "compute", "--study-mode", "full",
            "--config", str(CONFIG), "--source-root", str(tmp_path / "missing"),
            "--cohort-root", str(tmp_path / "missing-cohort"),
            "--preflight-report", str(tmp_path / "missing-preflight.json"),
            "--output-root", str(tmp_path / "study"), "--max-executions", "1",
        ],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
        env={"PYTHONPATH": "src:.", "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert result.returncode == 2
    assert "full" in result.stderr.lower()
    assert "max-executions" in result.stderr


@pytest.mark.parametrize(
    ("instrument", "max_executions"), [("TEST", 1), ("RISANAMENTO", 0), ("RISANAMENTO", 6)]
)
def test_pilot_mode_requires_risanamento_and_bounded_cap(
    tmp_path: Path, instrument: str, max_executions: int
) -> None:
    result = subprocess.run(
        [
            sys.executable, str(SCRIPT), "--stage", "compute", "--study-mode", "pilot",
            "--config", str(CONFIG), "--source-root", str(tmp_path / "missing"),
            "--cohort-root", str(tmp_path / "missing-cohort"),
            "--preflight-report", str(tmp_path / "missing-preflight.json"),
            "--output-root", str(tmp_path / "study"), "--instrument", instrument,
            "--max-executions", str(max_executions),
        ],
        cwd=REPO_ROOT, text=True, capture_output=True, check=False,
        env={"PYTHONPATH": "src:.", "PYTHONDONTWRITEBYTECODE": "1"},
    )

    assert result.returncode == 2
    assert "pilot" in result.stderr.lower()


def test_compute_verifies_before_publication_and_rechecks_reuse(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root, _, config, preflight, cohort_root = _write_consumer_fixture(tmp_path)
    context = consumer._load_context(config, source_root, cohort_root, preflight, "TEST", None, "full")
    output_root = tmp_path / "study"
    consumer.compute(output_root, context, resume=False, command=["test"])
    complete = output_root / "partitions" / "TEST" / "p" / context["config_key"] / "complete" / "COMPLETE"
    assert complete.is_file()

    def fail_verifier(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected verifier failure")

    monkeypatch.setattr(consumer, "verify_partition_grids", fail_verifier)
    with pytest.raises(RuntimeError, match="injected verifier failure"):
        consumer.compute(output_root, context, resume=True, command=["test"])
    assert complete.is_file()


def test_compute_failure_in_verifier_never_publishes_complete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_root, _, config, preflight, cohort_root = _write_consumer_fixture(tmp_path)
    context = consumer._load_context(config, source_root, cohort_root, preflight, "TEST", None, "full")

    def fail_verifier(*args: object, **kwargs: object) -> None:
        raise RuntimeError("injected verifier failure")

    monkeypatch.setattr(consumer, "verify_partition_grids", fail_verifier)
    with pytest.raises(RuntimeError, match="injected verifier failure"):
        consumer.compute(tmp_path / "study", context, resume=False, command=["test"])
    assert not list((tmp_path / "study").glob("partitions/**/COMPLETE"))


def test_same_selected_ids_cannot_resume_full_as_pilot(tmp_path: Path) -> None:
    source_root, _, config, preflight, cohort_root = _write_consumer_fixture(tmp_path)
    full = consumer._load_context(config, source_root, cohort_root, preflight, "TEST", None, "full")
    output_root = tmp_path / "study"
    consumer.compute(output_root, full, resume=False, command=["test", "full"])
    pilot_scope = {**full["scope"], "mode": "pilot", "instrument": "RISANAMENTO", "max_executions": 1}
    pilot = {**full, "scope": pilot_scope, "scope_sha": consumer._canonical_hash(pilot_scope)}

    with pytest.raises(consumer.ConsumerError, match="manifest identity/mode/request mismatch"):
        consumer.compute(output_root, pilot, resume=True, command=["test", "pilot"])


def test_gap_witness_invalidates_later_grid_nodes_without_erasing_gap(tmp_path: Path) -> None:
    source_root, _, config, preflight, cohort_root = _write_consumer_fixture(tmp_path)
    report = json.loads(preflight.read_text())
    witness = report["instruments"][0]["coverage_attestation"]["partition_coverage_witness"]
    witness["invalid_intervals"] = [{"partition_id": "p", "start_ns": 1002, "end_ns": 1004}]
    preflight.write_text(json.dumps(report))
    context = consumer._load_context(config, source_root, cohort_root, preflight, "TEST", None, "full")
    output_root = tmp_path / "study"

    consumer.compute(output_root, context, resume=False, command=["test"])
    completion = consumer._read_verified(
        output_root, consumer._identity(context["selected"][0], context),
        consumer._requested_frame(context["selected"]),
    )

    assert completion.summary.row(0, named=True)["J"] is None
    assert completion.summary.row(0, named=True)["complete_grid"] is False
    assert completion.trajectory.filter(pl.col("grid_offset_ns") > 0).get_column("reason").to_list() == [
        "known_coverage_gap", "known_coverage_gap"
    ]


def test_missing_partition_witness_fails_closed_as_coverage_unverified(tmp_path: Path) -> None:
    source_root, _, config, preflight, cohort_root = _write_consumer_fixture(tmp_path, coverage_witness=False)
    context = consumer._load_context(config, source_root, cohort_root, preflight, "TEST", None, "full")
    output_root = tmp_path / "study"

    consumer.compute(output_root, context, resume=False, command=["test"])
    completion = consumer._read_verified(
        output_root, consumer._identity(context["selected"][0], context),
        consumer._requested_frame(context["selected"]),
    )

    assert completion.summary.row(0, named=True)["complete_grid"] is False
    assert completion.trajectory.get_column("reason").to_list() == ["coverage_unverified"] * 3


def _write_sidecar_consumer_fixture(tmp_path):
    from datetime import datetime
    from scripts import msci_grid_coverage_witness as witness
    from spoofing_detection.lob.panel import _partition_id
    source, _, config, pfpath, cohort_root = _write_consumer_fixture(tmp_path, coverage_witness=False)
    bundle = source / "TEST"
    row = {"TRADEDATE": datetime(2024, 1, 1), "MIC": "MTAA", "MARKETCODE": "MTA", "SYMBOLINDEX": 1, "EMM (*)": 1}
    pid = _partition_id(row)
    mp = bundle / "metadata.json"
    meta = json.loads(mp.read_text())
    for key in ("state_time_series", "execution_metrics"):
        path = Path(meta["paths"][key])
        pl.read_parquet(path).with_columns(pl.lit(pid).alias("partition_id")).write_parquet(path)
        meta["artifact_hashes"][key] = _sha256(path)
    checkpoint = tmp_path / "checkpoint"
    (checkpoint / "replay").mkdir(parents=True)
    frame = checkpoint / "replay/frame_000.parquet"
    frame.write_bytes(Path(meta["paths"]["state_time_series"]).read_bytes())
    (checkpoint / "replay/manifest.json").write_text(json.dumps({"format": "h1_stage_checkpoint_v1", "stage": "replay", "frames": [{"filename": "frame_000.parquet", "sha256": _sha256(frame), "row_count": 4}]}))
    hc = tmp_path / "historical_config.json"
    hc.write_text('{"metrics":{"top_n":10}}')
    hashes = {relative: _sha256(REPO_ROOT / relative) for relative in ("scripts/compute_spoofing_metrics.py", "src/spoofing_detection/lob/spoofing_metrics.py", "src/spoofing_detection/lob/replay_observation.py")}
    raw = {"path": str(tmp_path / "raw-not-opened.parquet"), "sha256": "a" * 64}
    identity = {"format": "h1_checkpoint_identity_v1", "inputs": {"config": {"path": str(hc), "sha256": _sha256(hc)}, "raw": raw}, "source_sha256": hashes, "resolved_compute_args": {"clock_policy_v3": True, "top_n": 10, "window_seconds": 1.0, "compact_state": True, "state_client_mode": "execution-actors", "max_rows": None}}
    (checkpoint / "identity.json").write_text(json.dumps(identity))
    log = source.parent / "logs/TEST.log"
    log.parent.mkdir(parents=True)
    log.write_text('{"event":"run_exit","status":"ok"}\n')
    meta.pop("state_coverage_attestation")
    meta.update(compact_state=True, state_client_mode="execution-actors", config_sha256=_sha256(hc), input_hashes={"config_sha256": _sha256(hc), "raw_events_sha256": raw["sha256"]}, scientific_source_hashes={"spoofing_metrics.py": hashes["src/spoofing_detection/lob/spoofing_metrics.py"], "replay_observation.py": hashes["src/spoofing_detection/lob/replay_observation.py"]}, generator_sha256=hashes["scripts/compute_spoofing_metrics.py"], row_counts={"state_time_series": 4, "execution_metrics": pl.read_parquet(meta["paths"]["execution_metrics"]).height})
    mp.write_text(json.dumps(meta))
    cp = cohort_root / "cohort.csv"
    pl.read_csv(cp).with_columns(pl.lit(pid).alias("partition_id")).write_csv(cp)
    cpp = cohort_root / "cohort_provenance.json"
    provenance = json.loads(cpp.read_text())
    provenance["artifacts"]["cohort"]["sha256"] = _sha256(cp)
    provenance["artifacts"]["source"]["TEST"]["execution_metrics"]["sha256"] = meta["artifact_hashes"]["execution_metrics"]
    provenance["input_digests"]["source_TEST"]["sha256"] = _sha256(mp)
    cpp.write_text(json.dumps(provenance))
    pf = json.loads(pfpath.read_text())
    pf["input_digests"]["source_TEST"] = {"path": str(mp), "sha256": _sha256(mp)}
    entry = pf["instruments"][0]
    entry.pop("coverage_attestation")
    entry["metadata"] = pf["input_digests"]["source_TEST"]
    entry["artifacts"] = {key: {"path": meta["paths"][key], "sha256": meta["artifact_hashes"][key]} for key in ("state_time_series", "execution_metrics")}
    entry["declared_row_counts"] = meta["row_counts"]
    entry["coverage_historical_evidence"] = {"schema": "msci_grid_historical_h1_checkpoint_evidence_v1", "checkpoint": str(checkpoint), "identity": {"path": str(checkpoint / "identity.json"), "source_bindings": {relative: {"sha256": digest, "current_path": str(REPO_ROOT / relative)} for relative, digest in hashes.items()}}, "config": identity["inputs"]["config"], "raw_input": {"sha256": raw["sha256"]}, "replay_frame_0": {"path": str(frame), "sha256": _sha256(frame), "row_count": 4}, "run_log": str(log), "semantics": witness.SEMANTICS}
    pfpath.write_text(json.dumps(pf))
    cache = tmp_path / "cache"
    cache.mkdir()
    normalized = cache / "normalized.parquet"
    pl.DataFrame({**{key: [value] * 104 for key, value in row.items()}, "sort_index": list(range(1, 105)), "SEQUENCETIME": list(range(900, 999)) + [999, 1001, 1011, 1021, 2000], "event_class": ["new_order"] * 104}).write_parquet(normalized)
    cm = cache / "metadata.json"
    cm.write_text(json.dumps({"outputs": {"normalized_events": str(normalized)}, "source_file": raw["path"], "source_file_sha256": raw["sha256"], "config": {"include_all_events": True}, "max_rows": None, "row_counts": {"input": 104, "normalized_events": 104}, "validation": {"events_processed": 104, "partitions_processed": 1}}))
    sidecar = tmp_path / "witness.json"
    assert witness.main(["--preflight-report", str(pfpath), "--cache-metadata", f"TEST={cm}", "--assume-protected-cache", "--output", str(sidecar)]) == 0
    return source, config, pfpath, cohort_root, sidecar


def test_explicit_sidecar_compute_verify_resume_and_cli(tmp_path):
    source, config, pf, cohort_root, sidecar = _write_sidecar_consumer_fixture(tmp_path)
    context = consumer._load_context(config, source, cohort_root, pf, "TEST", None, coverage_witness_path=sidecar)
    root = tmp_path / "study"
    assert consumer.compute(root, context, resume=False, command=["test"])["computed"] == 1
    assert consumer.verify(root, context, command=["test"])["invalid_exclusions"] == 0
    assert consumer.compute(root, context, resume=True, command=["test"])["reused"] == 1
    command = _consumer_command(source_root=source, config=config, preflight=pf, cohort_root=cohort_root, output_root=root, stage="verify")
    result = subprocess.run([*command, "--coverage-witness", str(sidecar)], cwd=REPO_ROOT, text=True, capture_output=True, env={"PYTHONPATH": "src:.", "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    assert "MSCI_GRID_STUDY_VERIFIED" in result.stdout
    no_sidecar = consumer._load_context(config, source, cohort_root, pf, "TEST", None)
    with pytest.raises(consumer.ConsumerError, match="manifest"):
        consumer.compute(root, no_sidecar, resume=True, command=["test"])


@pytest.mark.parametrize("kind,step,lag", [("resolution", 5, 10), ("resolution", 10, 10), ("lag", 5, 5), ("lag", 5, 10)])
def test_sensitivity_exact_setting_preserves_cohort_and_verifies(tmp_path, kind, step, lag):
    source, _, config, pf, cohort_root = _write_consumer_fixture(tmp_path)
    protected = {p: p.read_bytes() for p in (config, pf, cohort_root / "cohort.csv", cohort_root / "cohort_provenance.json")}
    primary = consumer._load_context(config, source, cohort_root, pf, None, None)
    context = consumer._load_context(config, source, cohort_root, pf, None, None,
                                    setting_kind=kind, grid_step_ns=step, lag_ns=lag)
    assert context["scope"] == primary["scope"]
    assert context["cohort_provenance_sha"] == primary["cohort_provenance_sha"]
    assert context["config"]["grid_step_ns"] == step
    assert context["config"]["lag_ns"] == lag
    assert context["config"]["horizon_ns"] == primary["config"]["horizon_ns"]
    root = tmp_path / "sensitivity"
    assert consumer.compute(root, context, resume=False, command=["test"])["computed"] == 1
    assert consumer.verify(root, context, command=["test"])["executions"] == 1
    assert consumer.compute(root, context, resume=True, command=["test"])["computed"] == 0
    manifest = json.loads((root / "run_manifest.json").read_text())
    assert manifest["setting_contract"]["kind"] == kind
    assert manifest["setting_contract"]["primary_config_sha256"] == _sha256(config)
    with pytest.raises(consumer.ConsumerError, match="manifest"):
        consumer.verify(root, primary, command=["test"])
    command = _consumer_command(source_root=source, config=config, preflight=pf, cohort_root=cohort_root,
                                output_root=root, stage="verify")
    index = command.index("--instrument")
    del command[index:index + 2]
    result = subprocess.run([*command, "--setting-kind", kind, "--grid-step-ns", str(step), "--lag-ns", str(lag)],
                            cwd=REPO_ROOT, text=True, capture_output=True,
                            env={"PYTHONPATH": "src:.", "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 0, result.stderr
    assert all(p.read_bytes() == data for p, data in protected.items())


@pytest.mark.parametrize("kind,step,lag", [("primary", 5, 10), ("resolution", 5, 5), ("resolution", 7, 10),
                                         ("lag", 10, 5), ("lag", 5, 15), ("resolution", None, 10),
                                         ("lag", 5, None), ("unknown", 5, 10), ("lag", True, 5)])
def test_sensitivity_rejects_unprespecified_or_incomplete_setting(tmp_path, kind, step, lag):
    source, _, config, pf, cohort_root = _write_consumer_fixture(tmp_path)
    with pytest.raises(consumer.ConsumerError, match="setting|prespecified|primary"):
        consumer._load_context(config, source, cohort_root, pf, None, None,
                               setting_kind=kind, grid_step_ns=step, lag_ns=lag)


def test_sensitivity_real_prespecification_and_distinct_labels():
    raw = json.loads(CONFIG.read_text())
    for kind, step, lag in [("resolution", 5_000_000, 50_000_000),
                            ("resolution", 25_000_000, 50_000_000),
                            ("lag", 5_000_000, 25_000_000),
                            ("lag", 5_000_000, 50_000_000),
                            ("lag", 5_000_000, 100_000_000)]:
        effective, key, digest = consumer._fixed_setting(raw, kind, step, lag)
        assert effective["horizon_ns"] == 1_000_000_000
        assert (effective["grid_step_ns"], effective["lag_ns"]) == (step, lag)
        assert key.startswith(kind + "_")
        assert len(digest) == 64
    resolution = consumer._fixed_setting(raw, "resolution", 5_000_000, 50_000_000)
    lag = consumer._fixed_setting(raw, "lag", 5_000_000, 50_000_000)
    assert resolution[0] == lag[0]
    assert resolution[1:] != lag[1:]


def test_sensitivity_label_cannot_resume_numerically_identical_setting(tmp_path):
    source, _, config, pf, cohort_root = _write_consumer_fixture(tmp_path)
    resolution = consumer._load_context(config, source, cohort_root, pf, None, None,
                                       setting_kind="resolution", grid_step_ns=5, lag_ns=10)
    lag = consumer._load_context(config, source, cohort_root, pf, None, None,
                                setting_kind="lag", grid_step_ns=5, lag_ns=10)
    root = tmp_path / "study"
    consumer.compute(root, resolution, resume=False, command=["test"])
    with pytest.raises(consumer.ConsumerError, match="manifest"):
        consumer.compute(root, lag, resume=True, command=["test"])
    with pytest.raises(consumer.ConsumerError, match="manifest"):
        consumer.verify(root, lag, command=["test"])


@pytest.mark.parametrize("stage", ["preflight", "cohort"])
def test_sensitivity_cli_rejects_setting_outside_compute_verify(tmp_path, stage):
    result = subprocess.run([sys.executable, str(SCRIPT), "--stage", stage,
                             "--config", str(CONFIG), "--source-root", str(tmp_path),
                             "--output-root", str(tmp_path / "out"), "--setting-kind", "lag",
                             "--grid-step-ns", "5000000", "--lag-ns", "25000000"],
                            cwd=REPO_ROOT, text=True, capture_output=True,
                            env={"PYTHONPATH": "src:.", "PYTHONDONTWRITEBYTECODE": "1"})
    assert result.returncode == 2
    assert "only valid with compute/verify" in result.stderr
    assert not (tmp_path / "out").exists()


def test_sensitivity_rejects_changed_primary_config_and_selected_scope(tmp_path):
    source, _, config, pf, cohort_root = _write_consumer_fixture(tmp_path)
    with pytest.raises(consumer.ConsumerError, match="full cohort"):
        consumer._load_context(config, source, cohort_root, pf, "TEST", None,
                               setting_kind="resolution", grid_step_ns=5, lag_ns=10)
    raw = json.loads(config.read_text())
    raw["grid_step_ns"] = 5
    config.write_text(json.dumps(raw))
    with pytest.raises(consumer.ConsumerError, match="config identity"):
        consumer._load_context(config, source, cohort_root, pf, None, None,
                               setting_kind="resolution", grid_step_ns=5, lag_ns=10)




def _report_campaign(tmp_path, count=1, aliases=True):
    from scripts.msci_grid_study_report import CASES
    source, _, config, pf, co = _write_consumer_fixture(tmp_path)
    config.write_text(CONFIG.read_text())
    cpp = co / "cohort_provenance.json"
    provenance = json.loads(cpp.read_text())
    provenance["input_digests"]["config"]["sha256"] = _sha256(config)
    cpp.write_text(json.dumps(provenance))
    evidence = json.loads(pf.read_text())
    evidence["input_digests"]["config"]["sha256"] = _sha256(config)
    evidence["instruments"][0]["coverage_attestation"]["partition_coverage_witness"]["segments"][0]["end_ns"] = 2_000_000_000
    pf.write_text(json.dumps(evidence))
    if count > 1:
        cp = co / "cohort.csv"
        frame = pl.read_csv(cp)
        clones = [frame.with_columns(pl.lit(f"report-ec-{i}").alias("execution_cluster_id")) for i in range(count)]
        pl.concat(clones).write_csv(cp)
        bundle = source / "TEST"; mp = bundle / "metadata.json"
        metadata = json.loads(mp.read_text()); ep = Path(metadata["paths"]["execution_metrics"])
        execution = pl.read_parquet(ep).filter(pl.col("execution_cluster_id") == "ec-01")
        pl.concat([execution.with_columns(pl.lit(f"report-ec-{i}").alias("execution_cluster_id")) for i in range(count)]).write_parquet(ep)
        metadata["artifact_hashes"]["execution_metrics"] = _sha256(ep)
        mp.write_text(json.dumps(metadata))
        provenance["artifacts"]["cohort"].update(sha256=_sha256(cp), row_count=count)
        provenance["artifacts"]["source"]["TEST"]["execution_metrics"]["sha256"] = _sha256(ep)
        provenance["input_digests"]["source_TEST"]["sha256"] = _sha256(mp)
        cpp.write_text(json.dumps(provenance))
        evidence["input_digests"]["source_TEST"]["sha256"] = _sha256(mp)
        evidence["instruments"][0]["artifacts"]["execution_metrics"]["sha256"] = _sha256(ep)
        pf.write_text(json.dumps(evidence))
    primary = tmp_path / "primary"
    ctx = consumer._load_context(config, source, co, pf, None, None)
    consumer.compute(primary, ctx, resume=False, command=["test"])
    roots = {}
    for label, (step, lag) in CASES.items():
        if aliases and label == "resolution_10ms_lag50ms":
            roots[label] = primary
        elif aliases and label == "lag_50ms_step5ms":
            roots[label] = roots["resolution_5ms_lag50ms"]
        else:
            ctx = consumer._load_context(config, source, co, pf, None, None,
                setting_kind=label.split("_")[0], grid_step_ns=step*1_000_000, lag_ns=lag*1_000_000)
            roots[label] = tmp_path / label
            consumer.compute(roots[label], ctx, resume=False, command=["test"])
    args = ["--stage", "report", "--config", str(config), "--source-root", str(source),
            "--cohort-root", str(co), "--preflight-report", str(pf),
            "--primary-run-root", str(primary), "--output-root", str(tmp_path / "report")]
    for label, root in roots.items():
        args += ["--sensitivity-run", f"{label}={root}"]
    return args, roots


def _run_report(args):
    return subprocess.run([sys.executable, str(SCRIPT), *args], cwd=REPO_ROOT,
        text=True, capture_output=True, env={"PYTHONPATH": "src:.", "PYTHONDONTWRITEBYTECODE": "1"})


@pytest.mark.parametrize("count,aliases", [(1,True), (165,False)])
def test_report_cli_scientifically_verified_campaign_generates_pdfs(tmp_path,count,aliases):
    args, _ = _report_campaign(tmp_path,count=count,aliases=aliases)
    result = _run_report(args)
    assert result.returncode == 0, result.stderr
    assert "MSCI_GRID_REPORT_COMPLETE" in result.stdout
    assert pl.read_csv(tmp_path / "report/summary.csv").height == count
    sensitivity = pl.read_csv(tmp_path / "report/sensitivity.csv")
    assert sensitivity.height == 6
    assert sensitivity.filter(pl.col("setting")=="resolution_10ms_lag50ms").item(0,"shared_reference")=="primary"
    for name in ["msci_trajectories.pdf", "j_vs_withdrawal.pdf"]:
        assert (tmp_path / "report/figures" / name).read_bytes().startswith(b"%PDF")
    text = (tmp_path / "report/report.md").read_text()
    assert "scientifically verified" in text
    assert "report module SHA256" in text


@pytest.mark.parametrize("bad", ["missing", "duplicate", "unknown", "wrong_setting", "wrong_source", "coverage", "partial", "extra", "extra_setting", "wrong_primary", "instrument", "resume", "cap", "pilot", "override"])
def test_report_cli_rejects_campaign_before_writing(tmp_path, bad):
    args, roots = _report_campaign(tmp_path)
    if bad == "missing": args = args[:-2]
    elif bad == "duplicate": args += args[-2:]
    elif bad == "unknown": args += ["--sensitivity-run", "unknown=/missing"]
    elif bad in {"wrong_setting", "wrong_source", "coverage"}:
        path = roots["resolution_25ms_lag50ms"] / "run_manifest.json"
        manifest = json.loads(path.read_text())
        if bad == "wrong_setting": manifest["settings"]["lag_ns"] = 25_000_000
        elif bad == "wrong_source": manifest["code_hashes"]["study_cli"] = "a"*64
        else: next(iter(manifest["partition_validation"].values()))["coverage_witness_sha256"] = "a"*64
        path.write_text(json.dumps(manifest))
    elif bad == "partial": next(roots["resolution_25ms_lag50ms"].glob("partitions/**/COMPLETE")).unlink()
    elif bad == "wrong_primary":
        config = Path(args[args.index("--config") + 1])
        raw = json.loads(config.read_text()); raw["grid_step_ns"] = 5_000_000
        config.write_text(json.dumps(raw))
    elif bad == "extra_setting":
        import shutil
        directory = next(roots["resolution_25ms_lag50ms"].glob("partitions/*/*/*/complete"))
        shutil.copytree(directory, directory.parents[1] / "other_setting" / "complete")
    elif bad == "extra":
        import shutil
        directory = next(roots["resolution_25ms_lag50ms"].glob("partitions/*/*/*/complete"))
        shutil.copytree(directory, directory.parents[2] / "extra" / directory.parent.name / "complete")
    else:
        options = {"instrument": ["--instrument", "TEST"], "resume": ["--resume"],
            "cap": ["--max-executions", "1"], "pilot": ["--study-mode", "pilot"],
            "override": ["--setting-kind", "resolution"]}
        args += options[bad]
    result = _run_report(args)
    assert result.returncode == 2
    assert not (tmp_path / "report").exists()
    assert "invalid choice" not in result.stderr


def test_explicit_sidecar_rejects_forged_support_and_external_completeness(tmp_path):
    source, config, pf, cohort_root, sidecar = _write_sidecar_consumer_fixture(tmp_path)
    raw = json.loads(sidecar.read_text())
    raw["instruments"]["TEST"]["partition_coverage_witness"]["segments"][0]["end_ns"] += 100
    sidecar.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="freshly derived"):
        consumer._load_context(config, source, cohort_root, pf, "TEST", None, coverage_witness_path=sidecar)
    raw["external_feed_completeness_verified"] = True
    sidecar.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="scope/schema"):
        consumer._load_context(config, source, cohort_root, pf, "TEST", None, coverage_witness_path=sidecar)

