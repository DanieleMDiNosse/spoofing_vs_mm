from __future__ import annotations

import importlib.util
import json
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


def _write_metadata(run_dir: Path) -> None:
    (run_dir / "metadata.json").write_text(
        json.dumps(
            {
                "output_schema_version": "actor_execution_anchor_v2",
                "analytical_unit": "execution_cluster",
                "actor_identity_mode": "client_then_firm",
                "firm_fallback_semantics": "aggregate only when client_original_id is missing",
                "execution_anchor_modes": ["passive", "aggressive"],
            }
        )
    )


def _execution_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "execution_cluster_id": ["EC-P-1", "EC-A-2"],
            "actor_key": ["client_original:A", "firm:F_1"],
            "actor_id": ["A", "F_1"],
            "identity_level": ["client_original", "firm"],
            "identity_fallback_flag": [False, True],
            "client_original_id": ["A", None],
            "firm_id": ["F_1", "F_1"],
            "execution_anchor_mode": ["passive", "aggressive"],
            "child_fill_count": [3, 1],
            "has_matched_deceptive_cancel_window": [True, False],
            "spoofing_compatible_sequence": [True, False],
            "withdrawal_to_execution_ratio": [10.0, None],
            "favorable_mid_move_pre_fill": [0.1, -0.1],
            "post_cancel_mid_reversion": [0.2, 0.1],
        }
    )


def _write_candidates(run_dir: Path) -> None:
    pl.DataFrame(
        {
            "partition_id": ["P", "P"],
            "cancel_sort_index": [10, 20],
            "candidate_order_id": ["O1", "O2"],
            "execution_anchor_mode": ["passive", "passive"],
            "identity_level": ["client_original", "client_original"],
            "assigned_flag": [True, False],
        }
    ).write_parquet(run_dir / "execution_cancel_candidates.parquet")


def test_summarize_run_stratifies_anchor_and_firm_fallback(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_metadata(run_dir)
    _execution_frame().write_parquet(run_dir / "execution_metrics.parquet")
    _write_candidates(run_dir)

    summary, top_actors, hashes = load_module().summarize_run("Sample", run_dir)

    assert summary.select("execution_anchor_mode").to_series().to_list() == ["aggressive", "passive"]
    aggressive, passive = summary.iter_rows(named=True)
    assert aggressive["execution_cluster_count"] == 1
    assert aggressive["raw_fill_message_count"] == 1
    assert aggressive["firm_fallback_cluster_count"] == 1
    assert aggressive["firm_fallback_matched_cluster_count"] == 0
    assert aggressive["assigned_cancellation_count"] == 0
    assert passive["execution_cluster_count"] == 1
    assert passive["raw_fill_message_count"] == 3
    assert passive["matched_cluster_count"] == 1
    assert passive["assigned_cancellation_count"] == 1
    assert passive["compatible_sequence_count"] == 1
    assert top_actors.get_column("actor_key").to_list() == ["client_original:A"]
    assert top_actors.get_column("execution_anchor_mode").to_list() == ["passive"]
    assert set(hashes) == {"execution_metrics", "execution_cancel_candidates", "run_metadata"}


def test_assigned_cancellation_count_uses_physical_composite_key(tmp_path: Path):
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

    assert load_module()._assigned_cancellation_count(run_dir) == 2


def test_assigned_cancellation_count_requires_canonical_candidate_artifact(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    with pytest.raises(FileNotFoundError, match="execution_cancel_candidates"):
        load_module()._assigned_cancellation_count(run_dir)


def test_summarize_run_ignores_retired_self_controlled_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_metadata(run_dir)
    _execution_frame().write_parquet(run_dir / "execution_metrics.parquet")
    _write_candidates(run_dir)
    pl.DataFrame(
        {
            "risk_set_id": ["R1", "R1"],
            "fill_exposed": [True, False],
            "withdrawn_within_window": [True, False],
        }
    ).write_parquet(run_dir / "withdrawal_risk_sets.parquet")
    pl.DataFrame({"execution_cluster_id": ["EC-P-1"]}).write_parquet(
        run_dir / "spoofing_compatible_events.parquet"
    )

    summary, _, hashes = load_module().summarize_run("Sample", run_dir)

    assert "matched_pair_count" not in summary.columns
    assert "supported_unit_count" not in summary.columns
    assert "withdrawal_risk_sets" not in hashes
    assert "spoofing_compatible_events" not in hashes


def test_summarize_run_rejects_non_clustered_and_duplicate_inputs(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_metadata(run_dir)
    _write_candidates(run_dir)
    pl.DataFrame({"actor_key": ["client_original:A"]}).write_parquet(run_dir / "execution_metrics.parquet")

    with pytest.raises(ValueError, match="not cluster-aware"):
        load_module().summarize_run("Sample", run_dir)

    duplicates = pl.concat([_execution_frame().head(1), _execution_frame().head(1)])
    duplicates.write_parquet(run_dir / "execution_metrics.parquet")
    with pytest.raises(ValueError, match="duplicate execution_cluster_id"):
        load_module().summarize_run("Sample", run_dir)


def test_summarize_run_rejects_invalid_firm_fallback_semantics(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_metadata(run_dir)
    _write_candidates(run_dir)
    invalid = _execution_frame().with_columns(
        pl.when(pl.col("identity_level") == "firm")
        .then(pl.lit("SHOULD_BE_MISSING"))
        .otherwise(pl.col("client_original_id"))
        .alias("client_original_id")
    )
    invalid.write_parquet(run_dir / "execution_metrics.parquet")

    with pytest.raises(ValueError, match="firm fallback row has a client identifier"):
        load_module().summarize_run("Sample", run_dir)


def test_render_tex_labels_actor_identity_and_execution_anchor():
    summary = pl.DataFrame(
        {
            "instrument": ["Sample", "Sample"],
            "execution_anchor_mode": ["aggressive", "passive"],
            "execution_cluster_count": [1, 1],
            "raw_fill_message_count": [1, 3],
            "matched_cluster_count": [0, 1],
            "firm_fallback_cluster_count": [1, 0],
            "firm_fallback_matched_cluster_count": [0, 0],
            "assigned_cancellation_count": [0, 1],
            "compatible_sequence_count": [0, 1],
            "fpm_positive_count": [0, 1],
            "fpm_observed_count": [0, 1],
            "reversion_positive_count": [0, 1],
            "reversion_observed_count": [0, 1],
        }
    )
    actors = pl.DataFrame(
        {
            "instrument": ["Sample"],
            "actor_key": ["firm:F_1"],
            "actor_id": ["F_1"],
            "identity_level": ["firm"],
            "execution_anchor_mode": ["aggressive"],
            "matched_cluster_count": [1],
            "max_withdrawal_to_execution_ratio": [10.0],
            "positive_fpm_mid_share": [1.0],
            "positive_reversion_mid_share": [0.0],
        }
    )

    macros, summary_table, actor_table = load_module().render_tex(summary, actors)

    assert "\\newcommand{\\SampleClusters}{2}" in macros
    assert "\\newcommand{\\SampleAggressiveClusters}{1}" in macros
    assert "\\newcommand{\\SamplePassiveClusters}{1}" in macros
    assert "\\newcommand{\\SampleCompatibleSequences}{1}" in macros
    assert "Raw fills" in summary_table
    assert "Assigned cancels" in summary_table
    assert "Firm fallback" in summary_table
    assert "Aggressive" in summary_table
    assert r"F\_1" in actor_table
    assert "W/E max" in actor_table
    assert "top_client_results" not in actor_table
