from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest


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
    assert row["actor_queue_dict"] == '{"C1":{"perc_vol":0.25,"priority":1}}'


def test_build_focal_timeline_excludes_unrelated_prior_fills():
    module = _load_module()
    event = {"sort_index": 20}
    event_log = pl.DataFrame(
        [
            {
                "sort_index": 10,
                "event_class": "fill",
                "is_candidate_deceptive_order": False,
                "is_matched_deceptive_cancel_order": False,
            },
            {
                "sort_index": 12,
                "event_class": "new_order",
                "is_candidate_deceptive_order": True,
                "is_matched_deceptive_cancel_order": True,
            },
            {
                "sort_index": 20,
                "event_class": "fill",
                "is_candidate_deceptive_order": False,
                "is_matched_deceptive_cancel_order": False,
            },
            {
                "sort_index": 25,
                "event_class": "cancel",
                "is_candidate_deceptive_order": True,
                "is_matched_deceptive_cancel_order": True,
            },
        ]
    )

    timeline = module.build_focal_timeline(event, event_log)

    assert timeline["sort_index"].to_list() == [12, 20, 25]
    assert timeline["timeline_role"].to_list() == [
        "candidate_order_before_execution",
        "selected_passive_execution",
        "matched_cancel_after_execution",
    ]


def test_build_focal_timeline_labels_aggressive_execution():
    module = _load_module()
    timeline = module.build_focal_timeline(
        {"sort_index": 20, "execution_anchor_mode": "aggressive"},
        pl.DataFrame(
            [
                {
                    "sort_index": 20,
                    "event_class": "fill",
                    "is_candidate_deceptive_order": False,
                    "is_matched_deceptive_cancel_order": False,
                }
            ]
        ),
    )

    assert timeline.select("timeline_role").item() == "selected_aggressive_execution"


def test_render_dossier_markdown_contains_core_sections():
    module = _load_module()
    event = {
        "review_event_id": "S10",
        "client_id": "C1",
        "event_ts": "2024-01-01T10:00:00",
        "execution_side": "ask",
        "deceptive_side": "bid",
        "execution_quantity": 100,
        "fill_qty": -99,
        "DWI_pre_window": -0.7,
        "DWI_post_window": -0.1,
        "SCI": 0.6,
        "MSCI_resting_profile": 0.3,
        "MSCI": -99.0,
        "withdrawal_profile_scale_event": 8.0,
        "WMSCI_passive": 2.4,
        "WMSCI_aggressive": None,
        "WMSCI_event": -99.0,
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

    focal_timeline = log.with_columns(pl.lit("selected_passive_execution").alias("timeline_role"))
    text = module.render_dossier_markdown(
        event=event,
        event_log=log,
        stage_depth=depth,
        robustness=pl.DataFrame(),
        focal_timeline=focal_timeline,
    )

    assert "# Event dossier: S10" in text
    assert "## Model scores" in text
    assert "## Focal matched-withdrawal timeline" in text
    assert "selected_passive_execution" in text
    assert "## Stage depth summary" in text
    assert "## Actual event log" in text
    assert "DWI_pre_window" in text
    assert "MSCI_resting_profile" in text
    assert "WMSCI_passive" in text
    assert "withdrawal_profile_scale_event: 8.0" in text
    assert "WMSCI_passive: 2.4" in text
    assert "matched_deceptive_cancel_min_delay_seconds: 0.2" in text
    assert "favorable_mid_move_pre_fill: 0.02" in text
    assert "post_cancel_mid_reversion: 0.01" in text
    assert "execution_price_advantage_vs_posture_mid: 0.03" in text


def test_build_parameter_robustness_ranks_within_execution_anchor(tmp_path):
    module = _load_module()
    root = tmp_path / "grid"
    run = root / "kappa_1.0_lambda_2.0"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "kappa": 1.0,
                "lambda_": 2.0,
                "msci_definition": module.MSCI_DEFINITION,
                "msci_range": list(module.MSCI_RANGE),
                "ratio_zero_denominator_policy": module.RATIO_ZERO_DENOMINATOR_POLICY,
            }
        )
    )
    pl.DataFrame(
        [
            {"sort_index": 10, "execution_anchor_mode": "passive", "MSCI_resting_profile": 0.5, "MSCI": -99.0, "SCI": 0.8, "collapse_opposite_side": 1.0, "collapse_same_side": 0.1, "has_matched_deceptive_cancel_window": True},
            {"sort_index": 11, "execution_anchor_mode": "aggressive", "MSCI_resting_profile": 0.9, "MSCI": -99.0, "SCI": 0.3, "collapse_opposite_side": 0.5, "collapse_same_side": 0.2, "has_matched_deceptive_cancel_window": True},
            {"sort_index": 12, "execution_anchor_mode": "passive", "MSCI_resting_profile": 0.2, "MSCI": -99.0, "SCI": 0.3, "collapse_opposite_side": 0.4, "collapse_same_side": 0.2, "has_matched_deceptive_cancel_window": True},
        ]
    ).write_parquet(run / "execution_metrics.parquet")

    out = module.build_parameter_robustness(event_sort_index=10, parameter_grid_root=root)

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["kappa"] == 1.0
    assert row["lambda"] == 2.0
    assert row["matched"] is True
    assert row["MSCI_resting_profile"] == 0.5
    assert row["rank_by_MSCI_resting_profile"] == 1


def test_build_parameter_robustness_rejects_incompatible_msci_provenance(tmp_path):
    module = _load_module()
    run = tmp_path / "grid" / "kappa_1.0_lambda_2.0"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text(
        json.dumps(
            {
                "kappa": 1.0,
                "lambda_": 2.0,
                "msci_definition": "obsolete_definition",
                "msci_range": [0.0, 1.0],
                "ratio_zero_denominator_policy": "epsilon_regularized",
            }
        )
    )
    pl.DataFrame([{"sort_index": 10, "MSCI": 0.9}]).write_parquet(
        run / "execution_metrics.parquet"
    )

    with pytest.raises(ValueError, match="incompatible MSCI definition"):
        module.build_parameter_robustness(10, tmp_path / "grid")


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
    (review_dir / "metadata.json").write_text(
        json.dumps(
            {
                "metric_run_parameters": {
                    "execution_anchor_modes": ["passive", "aggressive"],
                    "observed_execution_anchor_modes": ["passive"],
                    "actor_identity_mode": "client_then_firm",
                    "msci_definition": module.MSCI_DEFINITION,
                    "msci_range": list(module.MSCI_RANGE),
                    "ratio_zero_denominator_policy": module.RATIO_ZERO_DENOMINATOR_POLICY,
                }
            }
        )
    )

    out = tmp_path / "out"
    module.main(["--review-dir", str(review_dir), "--event-id", "S10", "--output-dir", str(out)])

    assert (out / "dossier.md").exists()
    assert (out / "dossier.json").exists()
    payload = json.loads((out / "dossier.json").read_text())
    assert payload["event"]["event_client_original_id"] == "C1"
    assert "client_id" not in payload["event"]
    assert "client_id" not in payload["execution_cluster"]
    assert "actor_queue_dict" in payload["stage_depth"][0]
    assert payload["metric_run_parameters"]["execution_anchor_modes"] == [
        "passive",
        "aggressive",
    ]
    assert payload["metric_run_parameters"]["observed_execution_anchor_modes"] == [
        "passive"
    ]


def test_cluster_bundle_requires_one_cluster_and_keeps_child_and_candidate_provenance():
    module = _load_module()
    cluster_id = "EC000000010-000000011"
    events = pl.DataFrame([
        {"execution_cluster_id": cluster_id, "review_event_id": cluster_id, "cluster_first_sort_index": 10, "cluster_last_sort_index": 11, "child_fill_count": 2, "fill_qty": 30.0, "client_id": "0"},
        {"execution_cluster_id": "EC000000020-000000020", "review_event_id": "EC000000020-000000020"},
    ])
    log = pl.DataFrame({"review_event_id": [cluster_id], "sort_index": [10]})
    queue = pl.DataFrame({"review_event_id": [cluster_id], "level": [1]})
    members = pl.DataFrame([
        {"execution_cluster_id": cluster_id, "child_sort_index": 10, "child_fill_qty": 10.0},
        {"execution_cluster_id": cluster_id, "child_sort_index": 11, "child_fill_qty": 20.0},
    ])
    candidates = pl.DataFrame([{"execution_cluster_id": cluster_id, "assigned_flag": True, "assignment_rule": "canonical_nearest", "competing_cluster_count": 1, "candidate_order_id": "Q1", "original_qty": 100.0, "leaves_qty": 25.0}])

    bundle = module.select_event_bundle(cluster_id, events, log, queue, cluster_members=members, cancel_candidates=candidates)
    text = module.render_dossier_markdown(event=bundle.event, event_log=bundle.event_log, stage_depth=pl.DataFrame(), robustness=pl.DataFrame(), child_members=bundle.child_members, cancel_candidates=bundle.cancel_candidates)

    assert bundle.child_members.height == 2
    assert "## Execution cluster" in text
    assert "## Raw child fills" in text
    assert "## Cancellation assignment" in text
    assert "canonical_nearest" in text
    assert "Client attribution caveat" in text
    assert "partially_executed_fraction" in text


def test_cluster_bundle_rejects_ambiguous_cluster_id():
    module = _load_module()
    events = pl.DataFrame([{"execution_cluster_id": "EC1", "review_event_id": "EC1"}, {"execution_cluster_id": "EC1", "review_event_id": "EC1"}])
    try:
        module.select_event_bundle("EC1", events, pl.DataFrame(), pl.DataFrame())
    except ValueError as exc:
        assert "exactly one" in str(exc)
    else:
        raise AssertionError("expected ambiguous cluster selection to fail")


def test_dossier_exposes_actor_anchor_raw_identity_and_firm_scope_warning():
    module = _load_module()
    event = {
        "execution_cluster_id": "EC1",
        "actor_key": "firm:F1",
        "actor_id": "F1",
        "identity_level": "firm",
        "identity_source": "FIRMID",
        "identity_fallback_flag": True,
        "execution_anchor_mode": "aggressive",
        "event_client_original_id": None,
        "event_firm_id": "F1",
        "cluster_first_sort_index": 10,
        "cluster_last_sort_index": 10,
        "cluster_start_ts": "2024-01-01T10:00:00",
        "cluster_end_ts": "2024-01-01T10:00:00",
        "child_fill_count": 1,
        "fill_qty": 10.0,
    }

    text = module.render_dossier_markdown(
        event=event,
        event_log=pl.DataFrame(),
        stage_depth=pl.DataFrame(),
        robustness=pl.DataFrame(),
    )

    assert "actor_key: firm:F1" in text
    assert "actor_id: F1" in text
    assert "identity_level: firm" in text
    assert "execution_anchor_mode: aggressive" in text
    assert "event_client_original_id: None" in text
    assert "event_firm_id: F1" in text
    assert "Firm-fallback scope warning" in text
    assert "Client attribution caveat" not in text
    assert "- client_id:" not in text
