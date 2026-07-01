from __future__ import annotations

from pathlib import Path

import polars as pl

from spoofing_detection.lob.spoofing_metric_plots import write_spoofing_metric_dashboard


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