from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_execution_cluster_sensitivity.py"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_execution_cluster_sensitivity", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_output_checks_cluster_and_cancellation_cardinality(tmp_path: Path):
    module = load_module()
    pl.DataFrame(
        {
            "execution_cluster_id": ["EC1", "EC2"],
            "has_matched_deceptive_cancel_window": [True, False],
            "WMSCI_event": [4.0, 0.0],
            "withdrawal_to_fill_ratio": [8.0, 0.0],
        }
    ).write_parquet(tmp_path / "execution_metrics.parquet")
    pl.DataFrame(
        {
            "execution_cluster_id": ["EC1", "EC1", "EC2"],
            "child_sort_index": [1, 2, 3],
        }
    ).write_parquet(tmp_path / "execution_cluster_members.parquet")
    pl.DataFrame(
        {
            "execution_cluster_id": ["EC1", "EC2"],
            "partition_id": ["P", "P"],
            "cancel_sort_index": [10, 11],
            "cluster_last_sort_index": [2, 3],
            "candidate_order_id": ["O1", "O2"],
            "assigned_flag": [True, False],
        }
    ).write_parquet(tmp_path / "execution_cancel_candidates.parquet")

    row = module.summarize_output("Sample", 100, tmp_path)

    assert row["execution_cluster_count"] == 2
    assert row["raw_fill_message_count"] == 3
    assert row["matched_cluster_count"] == 1
    assert row["assigned_candidate_count"] == 1
    assert row["max_WMSCI"] == 4.0


def test_summarize_output_rejects_noncausal_cancel_candidate(tmp_path: Path):
    module = load_module()
    pl.DataFrame(
        {
            "execution_cluster_id": ["EC1"],
            "has_matched_deceptive_cancel_window": [True],
            "WMSCI_event": [4.0],
            "withdrawal_to_fill_ratio": [8.0],
        }
    ).write_parquet(tmp_path / "execution_metrics.parquet")
    pl.DataFrame(
        {"execution_cluster_id": ["EC1"], "child_sort_index": [2]}
    ).write_parquet(tmp_path / "execution_cluster_members.parquet")
    pl.DataFrame(
        {
            "execution_cluster_id": ["EC1"],
            "partition_id": ["P"],
            "cancel_sort_index": [2],
            "cluster_last_sort_index": [2],
            "candidate_order_id": ["O1"],
            "assigned_flag": [True],
        }
    ).write_parquet(tmp_path / "execution_cancel_candidates.parquet")

    with pytest.raises(ValueError, match="non-causal"):
        module.summarize_output("Sample", 100, tmp_path)


def test_build_command_passes_operational_cluster_gap(tmp_path: Path):
    module = load_module()
    args = module.parse_args(
        [
            "--input",
            str(tmp_path / "input.parquet"),
            "--instrument",
            "Sample",
            "--output-root",
            str(tmp_path / "out"),
            "--config",
            str(tmp_path / "config.json"),
            "--empirical-kernel",
            str(tmp_path / "kernel.json"),
        ]
    )

    command = module.build_command(args, gap_ms=50, output_dir=tmp_path / "run")

    assert command[command.index("--execution-cluster-max-gap-ms") + 1] == "50"
    assert "--compact-state" in command
    assert command[command.index("--empirical-depth-kernel") + 1].endswith("kernel.json")
