from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

from spoofing_detection.lob.spoofing_metric_plots import _top_mcps_table, write_spoofing_metric_dashboard


def test_top_mcps_table_prefers_canonical_scores_and_preserves_anchor_strata():
    scores = pl.DataFrame(
        {
            "actor_key": ["client_original:C1", "client_original:C1"],
            "execution_anchor_mode": ["passive", "aggressive"],
            "executions": [4, 3],
            "MCPS_resting_profile": [0.3, 0.8],
            "max_MSCI_resting_profile": [0.5, 0.9],
            "mean_MSCI_resting_profile": [0.2, 0.7],
            "MCPS": [-99.0, -99.0],
            "max_MSCI": [-99.0, -99.0],
        }
    )

    table = _top_mcps_table(scores, pl.DataFrame())

    assert table.height == 2
    assert table.get_column("execution_anchor_mode").to_list() == ["aggressive", "passive"]
    assert table.get_column("MCPS_resting_profile").to_list() == [0.8, 0.3]
    assert "MCPS" not in table.columns


def test_write_spoofing_metric_dashboard_creates_paper_aligned_html(tmp_path: Path):
    execution_metrics = pl.DataFrame(
        {
            "event_ts": ["2024-01-02 09:30:01", "2024-01-02 09:30:02"],
            "sort_index": [1, 2],
            "client_id": ["C1", "C2"],
            "execution_side": ["ask", "bid"],
            "deceptive_side": ["bid", "ask"],
            "fill_qty": [5.0, 3.0],
            "MSCI": [0.75, 0.10],
            "SCI": [0.9, 0.2],
            "collapse_opposite_side": [0.8, 0.1],
            "collapse_same_side": [0.2, 0.2],
            "has_direct_opposite_cancel_window": [True, False],
            "direct_opposite_cancel_visible_qty_window": [50.0, 0.0],
            "candidate_deceptive_visible_qty_pre": [50.0, 0.0],
            "matched_deceptive_cancel_visible_qty_window": [50.0, 0.0],
            "has_matched_deceptive_cancel_window": [True, False],
            "matched_deceptive_cancel_fraction_window": [1.0, 0.0],
            "favorable_mid_move_pre_fill": [0.02, -0.01],
            "post_cancel_mid_reversion": [0.01, 0.0],
            "execution_price_advantage_vs_posture_mid": [0.03, -0.01],
        }
    )
    state_time_series = pl.DataFrame(
        {
            "sort_index": [10, 20],
            "client_id": ["C1", "C1"],
            "DWI": [-0.8, -0.1],
        }
    )
    mcps_scores = pl.DataFrame(
        {
            "client_id": ["C1", "C2"],
            "gamma": [0.5, 0.5],
            "executions": [10, 8],
            "MCPS": [0.4, 0.0],
            "max_MSCI": [0.75, 0.1],
            "mean_favorable_mid_move_pre_fill": [0.02, -0.01],
            "mean_post_cancel_mid_reversion": [0.01, 0.0],
        }
    )
    output = tmp_path / "dashboard.html"

    write_spoofing_metric_dashboard(
        execution_metrics=execution_metrics,
        state_time_series=state_time_series,
        mcps_scores=mcps_scores,
        output_html=output,
        title="Multilevel spoofing metric dashboard",
        client_id="C1",
    )

    html = output.read_text()
    assert "Multilevel spoofing metric dashboard" in html
    assert "MSCI" in html
    assert "signed contrast of normalized SCI plus opposite-side collapse minus same-side collapse" in html
    assert "arithmetic mean" not in html
    assert "positive side-collapse asymmetry" not in html
    assert "becomes large only when" not in html
    assert "MCPS" in html
    assert "DWI" in html
    assert "opposite-side collapse" in html
    assert "candidate deceptive profile" in html
    assert "spoofing-like executions" in html
    assert "Price-response diagnostics" in html
    assert "favorable pre-fill mid-price movement" in html
    assert "mean_favorable_mid_move_pre_fill" in html
    assert "Orange points" not in html
    assert "Blue points" not in html
    assert "#ff7f0e" not in html
    assert "#1f77b4" not in html
    assert "fake" not in html.lower()
    assert "imbalance" not in html.lower()


def test_actor_dashboard_preserves_identity_anchor_and_filters_dwi_by_actor(tmp_path: Path):
    execution_metrics = pl.DataFrame(
        {
            "event_ts": ["2024-01-02 09:30:01", "2024-01-02 09:30:02"],
            "actor_key": ["client_original:C1", "firm:F1"],
            "actor_id": ["C1", "F1"],
            "identity_level": ["client_original", "firm"],
            "identity_fallback_flag": [False, True],
            "execution_anchor_mode": ["passive", "aggressive"],
            "MSCI": [0.75, 0.25],
            "has_matched_deceptive_cancel_window": [True, True],
        }
    )
    state_time_series = pl.DataFrame(
        {
            "sort_index": [10, 20, 30],
            "actor_key": ["client_original:C1", "client_original:C1", "firm:F1"],
            "actor_id": ["C1", "C1", "F1"],
            "identity_level": ["client_original", "client_original", "firm"],
            "DWI": [-0.8, -0.1, 9999.0],
        }
    )
    mcps_scores = pl.DataFrame(
        {
            "actor_key": ["client_original:C1", "firm:F1"],
            "actor_id": ["C1", "F1"],
            "identity_level": ["client_original", "firm"],
            "identity_fallback_flag": [False, True],
            "execution_anchor_mode": ["passive", "aggressive"],
            "top_n": [1, 1],
            "gamma": [0.5, 0.5],
            "executions": [10, 8],
            "MCPS": [0.4, 0.2],
            "max_MSCI": [0.75, 0.25],
        }
    )
    output = tmp_path / "actor_dashboard.html"

    write_spoofing_metric_dashboard(
        execution_metrics=execution_metrics,
        state_time_series=state_time_series,
        mcps_scores=mcps_scores,
        output_html=output,
        title="Actor dashboard",
        actor_key="client_original:C1",
    )

    html = output.read_text()
    assert "Top actors by MCPS" in html
    assert "actor_key" in html
    assert "identity_level" in html
    assert "execution_anchor_mode" in html
    assert "DWI client_original:C1" in html
    assert "9999.0" not in html
    assert "Firm-fallback rows aggregate activity at the firm level" in html

    legacy_selector_output = tmp_path / "legacy_selector_actor_dashboard.html"
    write_spoofing_metric_dashboard(
        execution_metrics=execution_metrics,
        state_time_series=state_time_series,
        mcps_scores=mcps_scores,
        output_html=legacy_selector_output,
        title="Actor dashboard",
        client_id="C1",
    )
    assert "DWI client_original:C1" in legacy_selector_output.read_text()


def test_plot_cli_exposes_actor_key_and_keeps_legacy_client_selector(tmp_path: Path):
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "plot_spoofing_metrics.py"
    spec = importlib.util.spec_from_file_location("plot_spoofing_metrics", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    common = [
        "--execution-metrics",
        str(tmp_path / "execution_metrics.parquet"),
        "--output-html",
        str(tmp_path / "dashboard.html"),
    ]
    assert module.parse_args([*common, "--actor-key", "firm:F1"]).actor_key == "firm:F1"
    assert module.parse_args([*common, "--client-id", "C1"]).client_id == "C1"