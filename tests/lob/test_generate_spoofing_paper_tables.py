from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "generate_spoofing_paper_tables.py"


def load_module():
    spec = importlib.util.spec_from_file_location("generate_spoofing_paper_tables", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_run_counts_clusters_raw_fills_and_assigned_cancellations(tmp_path: Path):
    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    executions = pl.DataFrame(
        {
            "execution_cluster_id": ["EC1", "EC2"],
            "client_id": ["A", "0"],
            "child_fill_count": [3, 1],
            "has_matched_deceptive_cancel_window": [True, False],
            "WMSCI_event": [2.5, 0.0],
            "withdrawal_to_fill_ratio": [10.0, 0.0],
            "favorable_mid_move_pre_fill": [0.1, -0.1],
            "post_cancel_mid_reversion": [0.0, 0.1],
        }
    )
    executions.write_parquet(run_dir / "execution_metrics.parquet")
    pl.DataFrame(
        {
            "partition_id": ["P", "P"],
            "cancel_sort_index": [10, 20],
            "execution_cluster_id": ["EC1", "EC1"],
            "candidate_order_id": ["O1", "O2"],
            "assigned_flag": [True, False],
        }
    ).write_parquet(run_dir / "execution_cancel_candidates.parquet")

    summary, top_clients, hashes = module.summarize_run("Sample", run_dir)

    assert summary["execution_cluster_count"] == 2
    assert summary["raw_fill_message_count"] == 4
    assert summary["matched_cluster_count"] == 1
    assert summary["assigned_cancellation_count"] == 1
    assert summary["unknown_client_matched_cluster_count"] == 0
    assert "alert_count" not in summary
    assert "max_WMSCI" not in summary
    assert top_clients.get_column("client_id").to_list() == ["A"]
    assert "max_WMSCI" not in top_clients.columns
    assert "mean_WMSCI" not in top_clients.columns
    assert set(hashes) == {"execution_metrics", "execution_cancel_candidates"}


def test_assigned_cancellation_count_uses_physical_composite_key(tmp_path: Path):
    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pl.DataFrame(
        {
            "partition_id": ["P1", "P2"],
            "cancel_sort_index": [10, 20],
            "candidate_order_id": ["O1", "O1"],
            "assigned_flag": [True, True],
        }
    ).write_parquet(run_dir / "execution_cancel_candidates.parquet")

    count = module._assigned_cancellation_count(run_dir)

    assert count == 2


def test_assigned_cancellation_count_requires_canonical_candidate_artifact(tmp_path: Path):
    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="execution_cancel_candidates"):
        module._assigned_cancellation_count(run_dir)


def test_summarize_run_ignores_retired_self_controlled_artifacts(tmp_path: Path):
    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pl.DataFrame(
        {
            "execution_cluster_id": ["EC1", "EC2"],
            "client_id": ["A", "B"],
            "child_fill_count": [1, 1],
            "has_matched_deceptive_cancel_window": [True, True],
            "WMSCI_event": [2.5, 1.0],
            "withdrawal_to_fill_ratio": [10.0, 2.0],
            "favorable_mid_move_pre_fill": [0.1, 0.2],
            "post_cancel_mid_reversion": [0.1, 0.2],
        }
    ).write_parquet(run_dir / "execution_metrics.parquet")
    pl.DataFrame(
        {
            "risk_set_id": ["R1", "R1", "R2", "R2"],
            "fill_exposed": [True, False, True, False],
            "withdrawn_within_window": [True, False, False, False],
        }
    ).write_parquet(run_dir / "withdrawal_risk_sets.parquet")
    pl.DataFrame({"withdrawal_excess_supported": [True, False]}).write_parquet(
        run_dir / "client_session_withdrawal_excess.parquet"
    )
    pl.DataFrame({"execution_cluster_id": ["EC1"]}).write_parquet(
        run_dir / "spoofing_compatible_events.parquet"
    )
    pl.DataFrame(
        schema={
            "partition_id": pl.String,
            "cancel_sort_index": pl.Int64,
            "candidate_order_id": pl.String,
            "assigned_flag": pl.Boolean,
        }
    ).write_parquet(run_dir / "execution_cancel_candidates.parquet")

    summary, _, hashes = module.summarize_run("Sample", run_dir)

    assert "matched_pair_count" not in summary
    assert "supported_unit_count" not in summary
    assert "withdrawal_risk_sets" not in hashes
    assert "client_session_withdrawal_excess" not in hashes
    assert "spoofing_compatible_events" not in hashes


def test_summarize_run_rejects_non_clustered_and_duplicate_inputs(tmp_path: Path):
    module = load_module()
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pl.DataFrame({"client_id": ["A"]}).write_parquet(run_dir / "execution_metrics.parquet")

    try:
        module.summarize_run("Sample", run_dir)
    except ValueError as exc:
        assert "not cluster-aware" in str(exc)
    else:
        raise AssertionError("legacy message-level outputs must be rejected")

    pl.DataFrame(
        {
            "execution_cluster_id": ["EC1", "EC1"],
            "client_id": ["A", "A"],
            "has_matched_deceptive_cancel_window": [False, False],
        }
    ).write_parquet(run_dir / "execution_metrics.parquet")
    try:
        module.summarize_run("Sample", run_dir)
    except ValueError as exc:
        assert "duplicate execution_cluster_id" in str(exc)
    else:
        raise AssertionError("duplicate clusters must be rejected")


def test_render_tex_labels_clusters_as_primary_unit():
    module = load_module()
    summary = pl.DataFrame(
        {
            "instrument": ["Sample"],
            "execution_cluster_count": [2],
            "raw_fill_message_count": [4],
            "matched_cluster_count": [1],
            "assigned_cancellation_count": [1],
            "attributable_client_count": [1],
            "matched_attributable_client_count": [1],
            "unknown_client_matched_cluster_count": [0],
            "alert_count": [1],
            "max_WMSCI": [2.5],
            "positive_fpm_mid_share": [1.0],
            "positive_reversion_mid_share": [0.0],
        }
    )
    clients = pl.DataFrame(
        {
            "instrument": ["Sample"],
            "client_id": ["A"],
            "matched_cluster_count": [1],
            "max_WMSCI": [2.5],
            "mean_WMSCI": [2.5],
            "max_withdrawal_to_fill_ratio": [10.0],
            "positive_fpm_mid_share": [1.0],
            "positive_reversion_mid_share": [0.0],
        }
    )

    macros, tables = module.render_tex(summary, clients)

    assert "\\SampleClusters" in macros
    assert "Raw fills" in tables
    assert "Assigned cancels" in tables
    assert "analytical unit" in tables
    assert "Alerts" not in tables
    assert "Self-controlled rapid-withdrawal screening" not in tables
    assert "McNemar" not in tables
    assert "\\SampleCompatibleEvents" not in macros
    assert "WMSCI" not in macros
    assert "WMSCI" not in tables
