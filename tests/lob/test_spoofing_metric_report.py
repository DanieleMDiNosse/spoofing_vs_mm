from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl


def load_compute_script_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "compute_spoofing_metrics.py"
    spec = importlib.util.spec_from_file_location("compute_spoofing_metrics", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summary_report_explains_msci_mcps_without_old_terms(tmp_path: Path):
    module = load_compute_script_module()
    execution_metrics = pl.DataFrame(
        {
            "sort_index": [1, 2],
            "event_ts": ["2024-01-02 09:30:01", "2024-01-02 09:30:02"],
            "actor_key": ["client_original:C1", "firm:F2"],
            "actor_id": ["C1", "F2"],
            "identity_level": ["client_original", "firm"],
            "execution_anchor_mode": ["passive", "aggressive"],
            "execution_side": ["ask", "bid"],
            "deceptive_side": ["bid", "ask"],
            "execution_quantity": [5.0, 10.0],
            "fill_qty": [-99.0, -99.0],
            "candidate_deceptive_visible_qty_pre": [50.0, 0.0],
            "SCI": [0.7, 0.2],
            "MSCI_resting_profile": [0.4, 0.0],
            "MSCI": [-99.0, -99.0],
            "collapse_opposite_side": [0.8, 0.0],
            "collapse_same_side": [0.2, 0.1],
            "has_direct_opposite_cancel_window": [True, False],
            "direct_opposite_cancel_visible_qty_window": [50.0, 0.0],
            "has_matched_deceptive_cancel_window": [True, False],
            "matched_deceptive_cancel_visible_qty_window": [50.0, 0.0],
            "matched_deceptive_cancel_fraction_window": [1.0, None],
        }
    )
    mcps_scores = pl.DataFrame(
        {
            "actor_key": ["client_original:C1"],
            "actor_id": ["C1"],
            "identity_level": ["client_original"],
            "execution_anchor_mode": ["passive"],
            "gamma": [0.25],
            "executions": [2],
            "finite_msci_executions": [2],
            "msci_above_gamma_count": [1],
            "MCPS_resting_profile": [0.5],
            "max_MSCI_resting_profile": [0.4],
            "MCPS": [-99.0],
            "max_MSCI": [-99.0],
        }
    )
    output = tmp_path / "summary.md"

    module._write_summary_report(
        output_path=output,
        metadata={
            "input": "input.parquet",
            "quote_panel": "panel.parquet",
            "top_n": 3,
            "kappa": 1.0,
            "lambda_": 0.5,
            "ratio_zero_denominator_policy": module.RATIO_ZERO_DENOMINATOR_POLICY,
            "window_seconds": 1.0,
            "max_deceptive_order_age_seconds": 600.0,
            "tick_size": 0.01,
            "gamma_grid": [0.25],
            "output_schema_version": module.OUTPUT_SCHEMA_VERSION,
            "actor_identity_mode": "client_then_firm",
            "execution_anchor_modes": ["passive", "aggressive"],
            "firm_fallback_semantics": module.FIRM_FALLBACK_SEMANTICS,
            "row_counts": {"execution_metrics": 2, "actor_mcps_scores": 1},
        },
        client_audit={"claim_holds": True},
        execution_metrics=execution_metrics,
        state_time_series=pl.DataFrame({"actor_key": ["client_original:C1"]}),
        candidate_deceptive_orders=pl.DataFrame({"deceptive_order_id": ["B1"]}),
        mcps_scores=mcps_scores,
    )

    report = output.read_text()
    populations = module._population_metadata()

    assert populations["analytical_event_population"] == "all_selected_execution_anchor_clusters"
    assert populations["mcps_population"] == "all_attributable_actor_execution_clusters_stratified_by_anchor"
    assert populations["review_event_selection"] == "canonically_assigned_matched_withdrawal_clusters_only"
    assert "Multilevel top-n spoofing surveillance metrics" in report
    assert "DWI" in report
    assert "MSCI" in report
    assert "signed resting-profile contrast" in report
    assert "same-side collapse dominates" in report
    assert "equally weighted mean" not in report
    assert "ranges from 0 to 1" not in report
    assert "MCPS" in report
    assert "Top actors by resting-profile MCPS, stratified by identity level and execution anchor" in report
    assert "candidate deceptive profile" in report
    assert "MCPS population: all attributable actor execution clusters, stratified by anchor" in report
    assert "review-event selection: canonically assigned matched-withdrawal clusters only" in report
    assert "Top execution clusters by resting-profile MSCI, stratified by identity level and execution anchor" in report
    assert "clusters_with_observed_post_window_state" in report
    assert "client × partition_id" not in report
    assert "dependent matched pairs" not in report
    assert "unadjusted exploratory diagnostic" not in report
    assert "no multiple-testing correction" not in report
    assert "no prespecified hard calipers" not in report
    assert "event selection: matched deceptive-order cancellations only" not in report
    assert "fake" not in report.lower()
    assert "old" not in report.lower()