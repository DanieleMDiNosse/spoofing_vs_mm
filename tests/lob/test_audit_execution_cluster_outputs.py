from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

from spoofing_detection.lob.spoofing_metrics import (
    EXECUTION_CANCEL_CANDIDATE_SCHEMA,
    EXECUTION_CLUSTER_MEMBER_EMPTY_SCHEMA,
    EXECUTION_EMPTY_SCHEMA,
    MCPS_SCORE_SCHEMA,
)


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_execution_cluster_outputs.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("audit_execution_cluster_outputs", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_audit_accepts_official_actor_anchor_artifacts_and_reports_strata(tmp_path: Path):
    module = _load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    paths = {
        "execution_metrics": run_dir / "execution_metrics.parquet",
        "execution_cluster_members": run_dir / "execution_cluster_members.parquet",
        "execution_cancel_candidates": run_dir / "execution_cancel_candidates.parquet",
        "actor_mcps_scores": run_dir / "actor_mcps_scores.parquet",
    }
    pl.DataFrame(
        [
            {
                "execution_cluster_id": "EC1",
                "partition_id": "P1",
                "actor_key": "firm:F1",
                "actor_id": "F1",
                "identity_level": "firm",
                "identity_source": "FIRMID",
                "identity_fallback_flag": True,
                "execution_anchor_mode": "aggressive",
                "child_fill_count": 1,
                "execution_quantity": 5.0,
                "fill_qty": -99.0,
                "cluster_last_sort_index": 10,
                "has_matched_deceptive_cancel_window": True,
                "has_post_window_state": True,
                "MSCI_resting_profile": 0.4,
                "MSCI": -99.0,
            }
        ]
    ).write_parquet(paths["execution_metrics"])
    pl.DataFrame(
        [
            {
                "execution_cluster_id": "EC1",
                "partition_id": "P1",
                "actor_key": "firm:F1",
                "actor_id": "F1",
                "identity_level": "firm",
                "identity_source": "FIRMID",
                "identity_fallback_flag": True,
                "execution_anchor_mode": "aggressive",
                "child_sort_index": 10,
                "child_fill_qty": 5.0,
            }
        ]
    ).write_parquet(paths["execution_cluster_members"])
    pl.DataFrame(
        [
            {
                "execution_cluster_id": "EC1",
                "partition_id": "P1",
                "cancel_sort_index": 11,
                "cluster_last_sort_index": 10,
                "candidate_order_id": "Q1",
                "assigned_flag": True,
                "execution_anchor_mode": "aggressive",
                "actor_key": "firm:F1",
                "actor_id": "F1",
                "identity_level": "firm",
                "identity_source": "FIRMID",
                "identity_fallback_flag": True,
            }
        ]
    ).write_parquet(paths["execution_cancel_candidates"])
    pl.DataFrame(
        [
            {
                "actor_key": "firm:F1",
                "actor_id": "F1",
                "identity_level": "firm",
                "identity_source": "FIRMID",
                "identity_fallback_flag": True,
                "execution_anchor_mode": "aggressive",
                "MCPS_resting_profile": 1.0,
                "MCPS": -99.0,
            }
        ]
    ).write_parquet(paths["actor_mcps_scores"])
    metadata = {
        "output_schema_version": "actor_execution_anchor_v2",
        "actor_identity_mode": "client_then_firm",
        "execution_anchor_modes": ["passive", "aggressive"],
        "observed_execution_anchor_modes": ["aggressive"],
        "paths": {key: str(path) for key, path in paths.items()},
        "artifact_hashes": {key: module.sha256(path) for key, path in paths.items()},
        "analytical_event_population": "all_selected_execution_anchor_clusters",
        "mcps_population": "all_attributable_actor_execution_clusters_stratified_by_anchor",
        "review_event_selection": "canonically_assigned_matched_withdrawal_clusters_only",
        "event_metrics_include_unattributable_rows": True,
        "actor_scores_exclude_unattributable_rows": True,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))

    report = module.audit_run(run_dir)

    assert report["actor_score_rows"] == 1
    assert report["execution_clusters_by_anchor"] == {"aggressive": 1}
    assert report["execution_clusters_by_identity_level"] == {"firm": 1}
    assert report["score_rows_by_anchor"] == {"aggressive": 1}
    assert report["score_rows_by_identity_level"] == {"firm": 1}
    assert report["child_fills_by_execution_anchor_mode"] == {"aggressive": 1}
    assert report["candidate_links_by_execution_anchor_mode"] == {"aggressive": 1}
    assert report["configured_execution_anchor_modes"] == ["passive", "aggressive"]
    assert report["observed_execution_anchor_modes"] == ["aggressive"]

    metadata["observed_execution_anchor_modes"] = []
    (run_dir / "metadata.json").write_text(json.dumps(metadata))
    with pytest.raises(AssertionError, match="observed execution anchors"):
        module.audit_run(run_dir)


def test_audit_accepts_official_empty_actor_anchor_artifacts(tmp_path: Path):
    module = _load_module()
    run_dir = tmp_path / "empty-run"
    run_dir.mkdir()
    paths = {
        "execution_metrics": run_dir / "execution_metrics.parquet",
        "execution_cluster_members": run_dir / "execution_cluster_members.parquet",
        "execution_cancel_candidates": run_dir / "execution_cancel_candidates.parquet",
        "actor_mcps_scores": run_dir / "actor_mcps_scores.parquet",
    }
    for key, schema in (
        ("execution_metrics", EXECUTION_EMPTY_SCHEMA),
        ("execution_cluster_members", EXECUTION_CLUSTER_MEMBER_EMPTY_SCHEMA),
        ("execution_cancel_candidates", EXECUTION_CANCEL_CANDIDATE_SCHEMA),
        ("actor_mcps_scores", MCPS_SCORE_SCHEMA),
    ):
        pl.DataFrame(schema=schema).write_parquet(paths[key])
    metadata = {
        "output_schema_version": "actor_execution_anchor_v2",
        "actor_identity_mode": "client_then_firm",
        "execution_anchor_modes": ["passive", "aggressive"],
        "observed_execution_anchor_modes": [],
        "paths": {key: str(path) for key, path in paths.items()},
        "artifact_hashes": {key: module.sha256(path) for key, path in paths.items()},
        "analytical_event_population": "all_selected_execution_anchor_clusters",
        "mcps_population": "all_attributable_actor_execution_clusters_stratified_by_anchor",
        "review_event_selection": "canonically_assigned_matched_withdrawal_clusters_only",
        "event_metrics_include_unattributable_rows": True,
        "actor_scores_exclude_unattributable_rows": True,
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata))

    report = module.audit_run(run_dir)

    assert report["execution_clusters"] == 0
    assert report["actor_score_rows"] == 0
    assert report["observed_execution_anchor_modes"] == []
    assert report["post_window_coverage"] is None


def test_actor_audit_rejects_inconsistent_identity_source():
    module = _load_module()
    invalid = pl.DataFrame(
        [
            {
                "actor_key": "firm:F1",
                "actor_id": "F1",
                "identity_level": "firm",
                "identity_source": "NMSC_ORIGINALCLIENTIDSHORTCODE",
                "identity_fallback_flag": True,
                "execution_anchor_mode": "passive",
            }
        ]
    )

    with pytest.raises(AssertionError, match="actor identity fields disagree"):
        module._validate_actor_anchor_frame(invalid, artifact="fixture")


def test_actor_audit_accepts_multiple_valid_rows():
    module = _load_module()
    valid = pl.DataFrame(
        [
            {
                "actor_key": "client_original:C1",
                "actor_id": "C1",
                "identity_level": "client_original",
                "identity_source": "NMSC_ORIGINALCLIENTIDSHORTCODE",
                "identity_fallback_flag": False,
                "execution_anchor_mode": "passive",
            },
            {
                "actor_key": "firm:F1",
                "actor_id": "F1",
                "identity_level": "firm",
                "identity_source": "FIRMID",
                "identity_fallback_flag": True,
                "execution_anchor_mode": "aggressive",
            },
        ]
    )

    module._validate_actor_anchor_frame(valid, artifact="fixture")


def test_audit_rejects_score_actor_anchor_absent_from_execution_population():
    module = _load_module()
    executions = pl.DataFrame(
        {
            "actor_key": ["client_original:C1"],
            "execution_anchor_mode": ["passive"],
        }
    )
    scores = pl.DataFrame(
        {
            "actor_key": ["firm:OTHER"],
            "execution_anchor_mode": ["passive"],
        }
    )

    with pytest.raises(AssertionError, match="score provenance"):
        module._validate_score_provenance(executions, scores)


def test_audit_rejects_cross_actor_or_anchor_cluster_provenance():
    module = _load_module()
    executions = pl.DataFrame(
        [
            {
                "execution_cluster_id": "EC1",
                "actor_key": "firm:F1",
                "actor_id": "F1",
                "identity_level": "firm",
                "identity_source": "FIRMID",
                "identity_fallback_flag": True,
                "execution_anchor_mode": "aggressive",
            }
        ]
    )
    linked = pl.DataFrame(
        [
            {
                "execution_cluster_id": "EC1",
                "actor_key": "client_original:C2",
                "actor_id": "C2",
                "identity_level": "client_original",
                "identity_source": "NMSC_ORIGINALCLIENTIDSHORTCODE",
                "identity_fallback_flag": False,
                "execution_anchor_mode": "passive",
            }
        ]
    )

    with pytest.raises(AssertionError, match="cluster provenance disagrees"):
        module._validate_cluster_provenance(
            executions,
            linked,
            artifact="execution_cluster_members",
        )
