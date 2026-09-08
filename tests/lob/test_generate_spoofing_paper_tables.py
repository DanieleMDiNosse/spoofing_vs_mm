from __future__ import annotations

import hashlib
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


def test_summarize_run_rejects_zero_client_sentinel_artifacts(tmp_path: Path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    _write_metadata(run_dir)
    _write_candidates(run_dir)
    contaminated = _execution_frame().with_columns(
        pl.when(pl.col("identity_level") == "client_original")
        .then(pl.lit("client_original:0.00"))
        .otherwise(pl.col("actor_key"))
        .alias("actor_key"),
        pl.when(pl.col("identity_level") == "client_original")
        .then(pl.lit("0.00"))
        .otherwise(pl.col("actor_id"))
        .alias("actor_id"),
        pl.when(pl.col("identity_level") == "client_original")
        .then(pl.lit("0.00"))
        .otherwise(pl.col("client_original_id"))
        .alias("client_original_id"),
    )
    contaminated.write_parquet(run_dir / "execution_metrics.parquet")

    with pytest.raises(ValueError, match="zero client sentinel"):
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
    assert "Execution messages" in summary_table
    assert "cluster" not in summary_table.lower()
    assert "fill" not in summary_table.lower()
    assert "cluster" not in actor_table.lower()
    assert "fill" not in actor_table.lower()
    assert "Assigned cancels" in summary_table
    assert "Firm fallback" in summary_table
    assert "Aggressive" in summary_table
    assert r"F\_1" in actor_table
    assert "W/E max" in actor_table
    assert "top_client_results" not in actor_table


def test_external_alert_artifacts_hash_inputs_and_remove_subject_aliases(tmp_path: Path):
    audit_path = tmp_path / "external_alert_audit.json"
    audit = {
        "external_source": {
            "local_path_disclosed": False,
            "sha256": "a" * 64,
            "source_kind": "primary_pdf",
        },
        "verification_tier": "external_subject_temporal_coverage",
        "recall_unit": "unioned timed external-alert window",
        "detector_recall_evaluated": "identity-aligned unioned timed windows",
        "timestamp_precedence": ["TRADETIME", "BOOKOUTTIME"],
        "identifier_policy": "keyed pseudonyms",
        "limitations": ["positive-only source"],
        "overall": {
            "source_external_periods": 37,
            "source_timed_periods": 36,
            "source_date_only_periods": 1,
            "source_periods_with_raw_subject_event": 37,
            "union_timed_periods": 34,
            "union_periods_with_raw_subject_event": 34,
            "union_periods_with_recovered_execution": 34,
            "union_periods_with_matched_withdrawal": 30,
            "union_periods_with_strict_subject_scope_detection": 4,
            "identity_aligned_union_timed_periods": 4,
            "identity_aligned_union_periods_with_strict_detection": 0,
            "identity_unaligned_union_timed_periods": 30,
            "identity_unaligned_union_periods_with_strict_subject_scope_detection": 4,
        },
        "datasets": {
            "SAMPLE": {
                "source_external_periods": 2,
                "source_timed_periods": 2,
                "source_date_only_periods": 0,
                "union_timed_periods": 1,
                "union_periods_with_recovered_execution": 1,
                "union_periods_with_matched_withdrawal": 1,
                "union_periods_with_strict_subject_scope_detection": 1,
                "identity_aligned_union_timed_periods": 0,
                "identity_aligned_union_periods_with_strict_detection": 0,
                "identity_unaligned_union_timed_periods": 1,
                "identity_unaligned_union_periods_with_strict_subject_scope_detection": 1,
                "per_pseudonymized_actor": {
                    "actor_secret": {
                        "detector_actor_key_count_in_dataset": 2,
                        "identity_granularity_aligned": False,
                        "identity_namespace": "firm",
                        "source_external_periods": 2,
                        "source_timed_periods": 2,
                        "source_date_only_periods": 0,
                        "union_timed_periods": 1,
                        "union_periods_with_recovered_execution": 1,
                        "union_periods_with_matched_withdrawal": 1,
                        "union_periods_with_strict_subject_scope_detection": 1,
                        "identity_aligned_union_periods_with_strict_detection": 0,
                    }
                },
                "source_period_results": [
                    {
                        "actor_alias": "actor_secret",
                        "start": "2024-01-01T10:00:00",
                        "recovered_child_fill_rows": 4,
                        "recovered_child_fill_rows_by_anchor": {"aggressive": 4},
                        "recovered_execution_clusters": 1,
                        "recovered_execution_clusters_by_anchor": {"aggressive": 1},
                        "clusters_with_matched_withdrawal": 0,
                        "clusters_with_strict_detection": 0,
                    }
                ],
            }
        },
    }
    payload = json.dumps(audit, sort_keys=True).encode()
    audit_path.write_bytes(payload)

    provenance, public_summary = load_module().external_alert_artifacts(audit_path)

    assert provenance["audit_path"] == str(audit_path)
    assert provenance["audit_sha256"] == hashlib.sha256(payload).hexdigest()
    assert provenance["primary_source"] == audit["external_source"]
    assert provenance["overall_counts"] == audit["overall"]
    assert public_summary["overall_counts"] == audit["overall"]
    assert public_summary["datasets"]["SAMPLE"]["subject_summaries"] == [
        {
            "subject_rank": 1,
            "detector_actor_key_count_in_dataset": 2,
            "identity_granularity_aligned": False,
            "identity_namespace": "firm",
            "source_external_periods": 2,
            "source_timed_periods": 2,
            "source_date_only_periods": 0,
            "union_timed_periods": 1,
            "union_periods_with_recovered_execution": 1,
            "union_periods_with_matched_withdrawal": 1,
            "union_periods_with_strict_subject_scope_detection": 1,
            "identity_aligned_union_periods_with_strict_detection": 0,
        }
    ]
    assert public_summary["datasets"]["SAMPLE"]["source_record_totals"] == {
        "clusters_with_matched_withdrawal": 0,
        "clusters_with_strict_detection": 0,
        "recovered_child_fill_rows": 4,
        "recovered_child_fill_rows_by_anchor": {"aggressive": 4},
        "recovered_execution_clusters": 1,
        "recovered_execution_clusters_by_anchor": {"aggressive": 1},
    }
    assert "actor_secret" not in json.dumps(public_summary)
    assert "2024-01-01" not in json.dumps(public_summary)
