from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_spoofing_event_dossier.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_spoofing_event_dossier", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_event_bundle_filters_all_inputs_by_event_id():
    module = _load_module()
    events = pl.DataFrame(
        [
            {"review_event_id": "S10", "client_id": "C1", "MSCI": 1.2},
            {"review_event_id": "S11", "client_id": "C2", "MSCI": 0.4},
        ]
    )
    log = pl.DataFrame(
        [
            {"review_event_id": "S10", "sort_index": 10, "event_class": "fill"},
            {"review_event_id": "S11", "sort_index": 11, "event_class": "cancel"},
        ]
    )
    queue = pl.DataFrame(
        [
            {"review_event_id": "S10", "snapshot_sort_index": 9, "snapshot_phase": "pre", "side": "bid", "level": 1, "queue_position": 1, "visible_qty": 100},
            {"review_event_id": "S11", "snapshot_sort_index": 9, "snapshot_phase": "pre", "side": "ask", "level": 1, "queue_position": 1, "visible_qty": 200},
        ]
    )

    bundle = module.select_event_bundle("S10", events, log, queue)

    assert bundle.event["review_event_id"] == "S10"
    assert bundle.event_log.height == 1
    assert bundle.queue.height == 1
    assert bundle.event_log["sort_index"].to_list() == [10]


def test_build_stage_depth_summary_aggregates_candidate_and_total_volume():
    module = _load_module()
    queue = pl.DataFrame(
        [
            {
                "snapshot_phase": "pre",
                "side": "bid",
                "level": 1,
                "price": 10.0,
                "level_visible_qty": 1000,
                "visible_qty": 250,
                "is_candidate_deceptive_order": True,
                "is_matched_deceptive_cancel_order": False,
                "client_queue_dict": '{"C1":{"perc_vol":0.25,"priority":1}}',
            },
            {
                "snapshot_phase": "pre",
                "side": "bid",
                "level": 1,
                "price": 10.0,
                "level_visible_qty": 1000,
                "visible_qty": 100,
                "is_candidate_deceptive_order": False,
                "is_matched_deceptive_cancel_order": False,
                "client_queue_dict": '{"C2":{"perc_vol":0.10,"priority":2}}',
            },
        ]
    )

    summary = module.build_stage_depth_summary(queue)

    assert summary.height == 1
    row = summary.row(0, named=True)
    assert row["phase"] == "pre"
    assert row["side"] == "bid"
    assert row["level"] == 1
    assert row["total_visible_qty"] == 1000
    assert row["candidate_visible_qty"] == 250
    assert row["candidate_level_share"] == 0.25


def test_render_dossier_markdown_contains_core_sections():
    module = _load_module()
    event = {
        "review_event_id": "S10",
        "client_id": "C1",
        "event_ts": "2024-01-01T10:00:00",
        "execution_side": "ask",
        "deceptive_side": "bid",
        "fill_qty": 100,
        "DWI_pre_window": -0.7,
        "DWI_post_window": -0.1,
        "SCI": 0.6,
        "MSCI": 0.3,
        "WMSCI_event": 4.2,
        "candidate_deceptive_visible_qty_pre": 1000,
        "matched_deceptive_cancel_visible_qty_window": 800,
        "matched_deceptive_cancel_fraction_window": 0.8,
        "matched_deceptive_cancel_min_delay_seconds": 0.2,
        "matched_deceptive_cancel_max_delay_seconds": 0.7,
        "withdrawal_to_fill_ratio": 8.0,
        "favorable_mid_move_pre_fill": 0.02,
        "post_cancel_mid_reversion": 0.01,
        "execution_price_advantage_vs_posture_mid": 0.03,
    }
    log = pl.DataFrame([{"sort_index": 10, "event_ts": "2024-01-01T10:00:00", "event_class": "fill"}])
    depth = pl.DataFrame(
        [
            {
                "phase": "pre",
                "side": "bid",
                "level": 1,
                "price": 9.99,
                "total_visible_qty": 1000,
                "candidate_visible_qty": 800,
                "candidate_level_share": 0.8,
                "client_queue_dict": "{}",
            }
        ]
    )

    text = module.render_dossier_markdown(event=event, event_log=log, stage_depth=depth, robustness=pl.DataFrame())

    assert "# Event dossier: S10" in text
    assert "## Model scores" in text
    assert "## Stage depth summary" in text
    assert "## Actual event log" in text
    assert "DWI_pre_window" in text
    assert "MSCI" in text
    assert "WMSCI_event: 4.2" in text
    assert "withdrawal_to_fill_ratio: 8.0" in text
    assert "matched_deceptive_cancel_min_delay_seconds: 0.2" in text
    assert "favorable_mid_move_pre_fill: 0.02" in text
    assert "post_cancel_mid_reversion: 0.01" in text
    assert "execution_price_advantage_vs_posture_mid: 0.03" in text


def test_build_parameter_robustness_uses_sort_index_across_runs(tmp_path):
    module = _load_module()
    root = tmp_path / "grid"
    run = root / "kappa_1.0_lambda_2.0"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text('{"kappa": 1.0, "lambda_": 2.0}')
    pl.DataFrame(
        [
            {"sort_index": 10, "MSCI": 0.9, "SCI": 0.8, "collapse_opposite_side": 1.0, "collapse_same_side": 0.1, "has_matched_deceptive_cancel_window": True},
            {"sort_index": 11, "MSCI": 0.2, "SCI": 0.3, "collapse_opposite_side": 0.5, "collapse_same_side": 0.2, "has_matched_deceptive_cancel_window": True},
        ]
    ).write_parquet(run / "execution_metrics.parquet")

    out = module.build_parameter_robustness(event_sort_index=10, parameter_grid_root=root)

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["kappa"] == 1.0
    assert row["lambda"] == 2.0
    assert row["matched"] is True
    assert row["MSCI"] == 0.9
    assert row["rank_by_MSCI"] == 1


def test_main_writes_dossier_files(tmp_path):
    module = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    pl.DataFrame([{"review_event_id": "S10", "sort_index": 10, "client_id": "C1", "MSCI": 0.3}]).write_parquet(
        review_dir / "matched_spoofing_events.parquet"
    )
    pl.DataFrame([{"review_event_id": "S10", "sort_index": 10, "event_class": "fill"}]).write_parquet(
        review_dir / "matched_spoofing_event_log.parquet"
    )
    pl.DataFrame(
        [
            {
                "review_event_id": "S10",
                "snapshot_phase": "pre",
                "snapshot_sort_index": 9,
                "side": "bid",
                "level": 1,
                "price": 10.0,
                "queue_position": 1,
                "level_visible_qty": 100,
                "visible_qty": 100,
                "is_candidate_deceptive_order": True,
                "is_matched_deceptive_cancel_order": False,
                "client_queue_dict": "{}",
            }
        ]
    ).write_parquet(review_dir / "matched_spoofing_lob_queue.parquet")

    out = tmp_path / "out"
    module.main(["--review-dir", str(review_dir), "--event-id", "S10", "--output-dir", str(out)])

    assert (out / "dossier.md").exists()
    assert (out / "dossier.json").exists()
