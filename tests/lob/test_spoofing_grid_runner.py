from __future__ import annotations

import importlib.util
import json
from pathlib import Path


def load_grid_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_multilevel_spoofing_grid.py"
    spec = importlib.util.spec_from_file_location("run_multilevel_spoofing_grid", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_grid_runner_parses_depth_and_gamma_grids():
    module = load_grid_module()

    assert module._parse_int_grid("1,2,3,5,10") == [1, 2, 3, 5, 10]
    assert module._parse_float_grid("0.25,0.5") == [0.25, 0.5]


def test_grid_runner_builds_depth_output_directories(tmp_path: Path):
    module = load_grid_module()

    paths = module._depth_output_paths(tmp_path, 3)

    assert paths["execution_metrics"] == tmp_path / "topn_3" / "execution_metrics.parquet"
    assert paths["candidate_deceptive_orders"] == tmp_path / "topn_3" / "candidate_deceptive_orders.parquet"
    assert paths["spoofing_compatible_events"] == tmp_path / "topn_3" / "spoofing_compatible_events.parquet"
    assert paths["client_mcps_scores"] == tmp_path / "topn_3" / "client_mcps_scores.parquet"


def test_grid_runner_parse_args_loads_parameters_from_config_with_cli_overrides(tmp_path: Path):
    module = load_grid_module()
    config_path = tmp_path / "spoofing_parameters.json"
    config_path.write_text(
        json.dumps(
            {
                "grid": {
                    "depth_grid": [1, 3, 5],
                    "kappa": 2.0,
                    "lambda": 0.5,
                    "window_seconds": 30.0,
                    "withdrawal_window_seconds": 2.5,
                    "reversion_horizon_seconds": 3.5,
                    "execution_cluster_max_gap_ms": 250,
                    "max_deceptive_order_age_seconds": 120.0,
                    "gamma_grid": [0.001, 0.01],
                    "empirical_depth_kernel": str(tmp_path / "kernel.parquet"),
                }
            }
        )
    )

    args = module.parse_args(
        [
            "--config",
            str(config_path),
            "--input",
            str(tmp_path / "input.parquet"),
            "--quote-panel",
            str(tmp_path / "quotes.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--depth-grid",
            "2,4",
        ]
    )

    assert args.config == config_path
    assert args.depth_grid == "2,4"
    assert args.kappa == 2.0
    assert args.lambda_ == 0.5
    assert args.window_seconds == 30.0
    assert args.withdrawal_window_seconds == 2.5
    assert args.reversion_horizon_seconds == 3.5
    assert args.execution_cluster_max_gap_ms == 250
    assert not hasattr(args, "withdrawal_excess_alpha")
    assert args.max_deceptive_order_age_seconds == 120.0
    assert args.gamma_grid == "0.001,0.01"
    assert args.empirical_depth_kernel == tmp_path / "kernel.parquet"

    cli_kernel = tmp_path / "cli_kernel.parquet"
    cli_args = module.parse_args(
        [
            "--input",
            str(tmp_path / "input.parquet"),
            "--quote-panel",
            str(tmp_path / "quotes.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
            "--empirical-depth-kernel",
            str(cli_kernel),
        ]
    )
    assert cli_args.empirical_depth_kernel == cli_kernel


def test_grid_metadata_declares_gate_and_analytical_populations():
    module = load_grid_module()

    assert module._analysis_metadata() == {
        "analytical_unit": "execution_cluster",
        "raw_audit_unit": "child_fill_message",
        "event_selection": "all_passive_execution_clusters",
        "behavioral_gate": (
            "rapid_attributed_cancel AND fill_qty_lt_withdrawn_qty AND favorable_pre_fill_mid_move AND "
            "positive_cancel_anchored_mid_reversion"
        ),
        "analytical_event_population": "all_passive_execution_clusters",
        "mcps_population": "all_attributable_client_execution_clusters",
        "review_event_selection": "canonically_assigned_matched_withdrawal_clusters_only",
    }


def test_grid_metadata_declares_additive_msci_for_provenance_and_cache_invalidation():
    module = load_grid_module()

    assert module._msci_metadata() == {
        "msci_definition": (
            "mean(clip(SCI / 2, 0, 1), clip(C_opposite, 0, 1), "
            "max(clip(C_opposite, 0, 1) - clip(C_same, 0, 1), 0))"
        ),
        "msci_range": [0.0, 1.0],
    }


def test_grid_runner_depth_reuse_is_explicit_opt_in(tmp_path: Path):
    module = load_grid_module()
    required = [
        "--input",
        str(tmp_path / "input.parquet"),
        "--output-dir",
        str(tmp_path / "out"),
        "--tick-size",
        "0.01",
    ]

    assert module.parse_args(required).reuse_depth_outputs is False
    assert module.parse_args([*required, "--reuse-depth-outputs"]).reuse_depth_outputs is True


def test_grid_runner_reuse_rejects_stale_input_hash(tmp_path: Path, monkeypatch):
    module = load_grid_module()
    metadata_path = tmp_path / "metadata.json"
    expected = {
        "input_sha256": "current",
        "kappa": 1.0,
        "lambda_": 0.5,
    }
    metadata_path.write_text(
        json.dumps({**expected, "input_sha256": "stale", "depth_grid": [3]})
    )
    monkeypatch.setattr(module, "_depth_outputs_complete", lambda paths: True)

    assert not module._can_reuse_depth_outputs(
        {},
        metadata_path=metadata_path,
        expected_metadata=expected,
        top_n=3,
    )

    metadata_path.write_text(json.dumps({**expected, "depth_grid": [3]}))
    assert module._can_reuse_depth_outputs(
        {},
        metadata_path=metadata_path,
        expected_metadata=expected,
        top_n=3,
    )


def test_grid_runner_reuse_rejects_stale_msci_definition(tmp_path: Path, monkeypatch):
    module = load_grid_module()
    metadata_path = tmp_path / "metadata.json"
    expected = module._msci_metadata()
    metadata_path.write_text(
        json.dumps(
            {
                **expected,
                "msci_definition": "obsolete_definition",
                "depth_grid": [3],
            }
        )
    )
    monkeypatch.setattr(module, "_depth_outputs_complete", lambda paths: True)

    assert not module._can_reuse_depth_outputs(
        {},
        metadata_path=metadata_path,
        expected_metadata=expected,
        top_n=3,
    )
