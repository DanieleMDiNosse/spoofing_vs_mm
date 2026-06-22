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
            "client_id": ["C1", "C2"],
            "execution_side": ["ask", "bid"],
            "deceptive_side": ["bid", "ask"],
            "fill_qty": [5.0, 10.0],
            "candidate_deceptive_visible_qty_pre": [50.0, 0.0],
            "SCI": [0.7, 0.2],
            "MSCI": [0.4, 0.0],
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
            "client_id": ["C1"],
            "gamma": [0.25],
            "executions": [2],
            "finite_msci_executions": [2],
            "msci_above_gamma_count": [1],
            "MCPS": [0.5],
            "max_MSCI": [0.4],
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
            "epsilon": 1e-12,
            "window_seconds": 1.0,
            "max_deceptive_order_age_seconds": 600.0,
            "tick_size": 0.01,
            "gamma_grid": [0.25],
            "row_counts": {"execution_metrics": 2, "client_mcps_scores": 1},
        },
        client_audit={"claim_holds": True},
        execution_metrics=execution_metrics,
        state_time_series=pl.DataFrame({"client_id": ["C1"]}),
        candidate_deceptive_orders=pl.DataFrame({"deceptive_order_id": ["B1"]}),
        mcps_scores=mcps_scores,
    )

    report = output.read_text()
    assert "Multilevel top-n spoofing surveillance metrics" in report
    assert "DWI" in report
    assert "MSCI" in report
    assert "MCPS" in report
    assert "Top clients by MCPS" in report
    assert "candidate deceptive profile" in report
    assert "fake" not in report.lower()
    assert "old" not in report.lower()