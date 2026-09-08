from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest


CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "spoofing_detection_parameters.json"


def load_grid_module():
    script_path = Path(__file__).resolve().parents[2] / "scripts" / "run_multilevel_spoofing_grid.py"
    spec = importlib.util.spec_from_file_location("run_multilevel_spoofing_grid", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_grid_config(tmp_path: Path, **overrides: object) -> Path:
    payload = json.loads(CONFIG_PATH.read_text())
    kernel_path = tmp_path / "empirical_depth_kernel.csv"
    if not kernel_path.exists():
        pl.DataFrame(
            {
                "side": ["bid", "ask"],
                "rank": [1, 1],
                "kernel_weight": [1.0, 1.0],
            }
        ).write_csv(kernel_path)
    overrides.setdefault("empirical_depth_kernel", str(kernel_path))
    payload["grid"].update(overrides)
    config_path = tmp_path / "spoofing_parameters.json"
    config_path.write_text(json.dumps(payload))
    return config_path


def test_grid_runner_parses_depth_and_gamma_grids():
    module = load_grid_module()

    assert module._parse_int_grid("1,2,3,5,10") == [1, 2, 3, 5, 10]
    assert module._parse_float_grid("0.25,0.5") == [0.25, 0.5]


def test_grid_runner_uses_signed_msci_gamma_grid_from_config(tmp_path: Path):
    module = load_grid_module()

    args = module.parse_args(
        [
            "--input",
            str(tmp_path / "input.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert args.gamma_grid == "0.0,0.1,0.25,0.5,1.0,1.5"
    assert args.actor_identity_mode == "client_then_firm"
    assert args.execution_anchor_modes == ("passive", "aggressive")


def test_grid_runner_validates_and_canonicalizes_execution_anchor_modes(tmp_path: Path):
    module = load_grid_module()
    config_path = write_grid_config(tmp_path, execution_anchor_modes=["aggressive", "passive"])
    common = [
        "--config",
        str(config_path),
        "--input",
        str(tmp_path / "input.parquet"),
        "--output-dir",
        str(tmp_path / "out"),
    ]

    assert module.parse_args(common).execution_anchor_modes == (
        "passive",
        "aggressive",
    )
    for invalid in ([], ["passive", "unknown"], ["passive", "passive"]):
        invalid_config = write_grid_config(tmp_path, execution_anchor_modes=invalid)
        with pytest.raises(SystemExit):
            module.parse_args(["--config", str(invalid_config), *common[2:]])


def test_grid_runner_rejects_invalid_actor_identity_mode_from_config(tmp_path: Path):
    module = load_grid_module()
    config_path = write_grid_config(tmp_path, actor_identity_mode="invalid_mode")

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--config",
                str(config_path),
                "--input",
                str(tmp_path / "input.parquet"),
                "--output-dir",
                str(tmp_path / "out"),
            ]
        )


def test_grid_runner_rejects_removed_epsilon_option(tmp_path: Path):
    module = load_grid_module()
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--config",
                str(CONFIG_PATH),
                "--input",
                str(tmp_path / "input.parquet"),
                "--output-dir",
                str(tmp_path / "out"),
                "--epsilon",
                "1e-12",
            ]
        )


def test_grid_runner_builds_depth_output_directories(tmp_path: Path):
    module = load_grid_module()

    paths = module._depth_output_paths(tmp_path, 3)

    assert paths["execution_metrics"] == tmp_path / "topn_3" / "execution_metrics.parquet"
    assert paths["candidate_deceptive_orders"] == tmp_path / "topn_3" / "candidate_deceptive_orders.parquet"
    assert paths["spoofing_compatible_events"] == tmp_path / "topn_3" / "spoofing_compatible_events.parquet"
    assert paths["state_time_series"] == tmp_path / "topn_3" / "actor_metric_time_series.parquet"
    assert paths["actor_mcps_scores"] == tmp_path / "topn_3" / "actor_mcps_scores.parquet"


def test_grid_runner_parse_args_loads_parameters_only_from_config(tmp_path: Path):
    module = load_grid_module()
    config_path = write_grid_config(
        tmp_path,
        depth_grid=[1, 3, 5],

        window_seconds=30.0,
        withdrawal_window_seconds=2.5,
        reversion_horizon_seconds=3.5,
        execution_cluster_max_gap_ms=250,
        max_deceptive_order_age_seconds=120.0,
        gamma_grid=[0.001, 0.01],
        empirical_depth_kernel=str(tmp_path / "kernel.parquet"),
        actor_identity_mode="client_then_firm",
        execution_anchor_modes=["aggressive", "passive"],
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
        ]
    )

    assert args.config == config_path
    assert args.depth_grid == "1,3,5"
    assert not hasattr(args, "kappa")
    assert not hasattr(args, "lambda_")
    assert args.window_seconds == 30.0
    assert args.withdrawal_window_seconds == 2.5
    assert args.reversion_horizon_seconds == 3.5
    assert args.execution_cluster_max_gap_ms == 250
    assert not hasattr(args, "withdrawal_excess_alpha")
    assert args.max_deceptive_order_age_seconds == 120.0
    assert args.gamma_grid == "0.001,0.01"
    assert args.empirical_depth_kernel == tmp_path / "kernel.parquet"
    assert args.actor_identity_mode == "client_then_firm"
    assert args.execution_anchor_modes == ("passive", "aggressive")

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--config",
                str(config_path),
                "--input",
                str(tmp_path / "input.parquet"),
                "--output-dir",
                str(tmp_path / "out"),
                "--depth-grid",
                "2,4",
            ]
        )


def test_grid_metadata_declares_gate_and_analytical_populations():
    module = load_grid_module()

    assert module._analysis_metadata() == {
        "analytical_unit": "execution_cluster",
        "raw_audit_unit": "child_fill_message",
        "event_selection": "selected_execution_anchor_clusters",
        "behavioral_gate": (
            "rapid_attributed_cancel AND fill_qty_lt_withdrawn_qty AND favorable_pre_fill_mid_move AND "
            "positive_cancel_anchored_mid_reversion"
        ),
        "analytical_event_population": "all_selected_execution_anchor_clusters",
        "mcps_population": "all_attributable_actor_execution_clusters_stratified_by_anchor",
        "review_event_selection": "canonically_assigned_matched_withdrawal_clusters_only",
    }


def test_grid_metadata_declares_signed_msci_for_provenance_and_cache_invalidation():
    module = load_grid_module()

    metadata = module._msci_metadata()
    assert metadata["msci_definition"] == "SCI / 2 + C_opposite - C_same"
    assert metadata["msci_range"] == [-1.0, 2.0]
    assert metadata["metric_definitions"]["MSCI_resting_profile"] == metadata["msci_definition"]
    assert metadata["legacy_metric_aliases"]["MSCI"] == "MSCI_resting_profile"
    assert metadata["execution_specific_fields"]["not_applicable_policy"] == "null_not_zero"
    assert metadata["execution_specific_fields"]["aggressive_only"] == [
        "aggressive_execution_quantity",
        "aggressive_execution_vwap",
        "aggressive_child_fill_count",
        "aggressive_execution_price_level_count",
        "aggressive_execution_price_min",
        "aggressive_execution_price_max",
        "aggressive_execution_sweep_id",
        "WMSCI_aggressive",
    ]


def test_grid_runner_depth_reuse_is_explicit_opt_in(tmp_path: Path):
    module = load_grid_module()
    config_path = write_grid_config(tmp_path, tick_size=0.01)
    required = [
        "--config",
        str(config_path),
        "--input",
        str(tmp_path / "input.parquet"),
        "--output-dir",
        str(tmp_path / "out"),
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


def test_grid_runner_main_versions_actor_anchor_artifacts_and_audits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = load_grid_module()
    input_path = tmp_path / "input.parquet"
    output_dir = tmp_path / "out"
    pl.DataFrame(
        {
            "NMSC_ORIGINALCLIENTIDSHORTCODE": [None, None],
            "FIRMID": [None, "F1"],
            "ORDER_TRADINGCAPACITY (*)": [1, 1],
        },
        schema={
            "NMSC_ORIGINALCLIENTIDSHORTCODE": pl.String,
            "FIRMID": pl.String,
            "ORDER_TRADINGCAPACITY (*)": pl.Int64,
        },
    ).write_parquet(input_path)
    empty = pl.DataFrame()
    result = SimpleNamespace(
        state_time_series=pl.DataFrame({"actor_key": ["client_original:C1", "firm:F1"]}),
        execution_metrics=pl.DataFrame(
            {
                "execution_anchor_mode": ["passive", "aggressive"],
                "identity_level": ["client_original", "firm"],
                "has_matched_deceptive_cancel_window": [True, True],
                "spoofing_compatible_sequence": [True, False],
            }
        ),
        candidate_deceptive_orders=empty,
        rejected_executions=pl.DataFrame({"reject_reason": ["missing_actor_identity"]}),
        execution_cluster_members=empty,
        execution_cancel_candidates=empty,
        spoofing_compatible_events=empty,
    )
    compute_kwargs: dict[str, object] = {}

    def fake_compute(*args, **kwargs):
        compute_kwargs.update(kwargs)
        return result

    monkeypatch.setattr(module, "compute_exploratory_metrics", fake_compute)
    monkeypatch.setattr(
        module,
        "compute_mcps_scores",
        lambda *args, **kwargs: pl.DataFrame(
            {
                "actor_key": ["client_original:C1", "firm:F1"],
                "actor_id": ["C1", "F1"],
                "identity_level": ["client_original", "firm"],
                "execution_anchor_mode": ["passive", "aggressive"],
                "top_n": [1, 1],
                "gamma": [0.1, 0.1],
                "executions": [1, 1],
                "finite_msci_executions": [1, 1],
                "MCPS_resting_profile": [1.0, 1.0],
                "max_MSCI_resting_profile": [1.0, 1.0],
                "MCPS": [-99.0, -99.0],
                "max_MSCI": [-99.0, -99.0],
            }
        ),
    )

    module.main(
        [
            "--config",
            str(write_grid_config(tmp_path, tick_size=0.01, depth_grid=[1])),
            "--input",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ]
    )

    metadata = json.loads((output_dir / "metadata.json").read_text())
    assert compute_kwargs["execution_anchor_modes"] == ("passive", "aggressive")
    assert metadata["output_schema_version"] == "actor_execution_anchor_v2"
    assert metadata["actor_identity_mode"] == "client_then_firm"
    assert metadata["execution_anchor_modes"] == ["passive", "aggressive"]
    assert metadata["observed_execution_anchor_modes"] == ["passive", "aggressive"]
    assert metadata["score_grouping"] == ["actor_key", "execution_anchor_mode"]
    assert metadata["market_observation"] == "both_passive_and_aggressive_execution_branches_when_selected"
    assert metadata["firm_fallback_semantics"] == "aggregate only when client_original_id is missing"
    assert metadata["combined_actor_mcps_scores"].endswith("combined_actor_mcps_scores.parquet")
    assert (output_dir / "topn_1" / "actor_metric_time_series.parquet").exists()
    assert (output_dir / "topn_1" / "actor_mcps_scores.parquet").exists()
    assert not (output_dir / "topn_1" / "client_metric_time_series.parquet").exists()
    assert metadata["actor_execution_audit"]["rows_missing_client_and_firm_identity"] == 1
    assert metadata["actor_execution_audit"]["execution_clusters_by_anchor_mode"] == [
        {"execution_anchor_mode": "aggressive", "rows": 1},
        {"execution_anchor_mode": "passive", "rows": 1},
    ]
    assert metadata["actor_execution_audit"]["execution_clusters_by_identity_level"] == [
        {"identity_level": "client_original", "rows": 1},
        {"identity_level": "firm", "rows": 1},
    ]
    assert metadata["actor_execution_audit"]["rejected_executions_by_reason"] == [
        {"reject_reason": "missing_actor_identity", "rows": 1}
    ]
    assert metadata["actor_execution_audit"]["strict_sequence_by_anchor_and_identity"] == [
        {"execution_anchor_mode": "passive", "identity_level": "client_original", "rows": 1}
    ]
    assert metadata["actor_execution_audit"]["matched_withdrawal_by_anchor_and_identity"] == [
        {"execution_anchor_mode": "aggressive", "identity_level": "firm", "rows": 1},
        {"execution_anchor_mode": "passive", "identity_level": "client_original", "rows": 1},
    ]


def test_grid_summary_ranks_mcps_within_identity_and_anchor_strata(tmp_path: Path):
    module = load_grid_module()
    path = tmp_path / "summary.md"
    dominant = pl.DataFrame(
        {
            "actor_key": [f"client_original:C{i}" for i in range(25)],
            "actor_id": [f"C{i}" for i in range(25)],
            "identity_level": ["client_original"] * 25,
            "execution_anchor_mode": ["passive"] * 25,
            "top_n": [1] * 25,
            "gamma": [0.1] * 25,
            "executions": [1] * 25,
            "finite_msci_executions": [1] * 25,
            "MCPS_resting_profile": [float(100 - i) for i in range(25)],
            "max_MSCI_resting_profile": [1.0] * 25,
            "MCPS": [-99.0] * 25,
            "max_MSCI": [-99.0] * 25,
        }
    )
    firm = pl.DataFrame(
        {
            "actor_key": ["firm:F1"],
            "actor_id": ["F1"],
            "identity_level": ["firm"],
            "execution_anchor_mode": ["aggressive"],
            "top_n": [1],
            "gamma": [0.1],
            "executions": [1],
            "finite_msci_executions": [1],
            "MCPS_resting_profile": [-1.0],
            "max_MSCI_resting_profile": [-1.0],
            "MCPS": [-99.0],
            "max_MSCI": [-99.0],
        }
    )
    metadata = {
        "input": "fixture.parquet",
        "depth_grid": [1],
        "kappa": 1.0,
        "lambda_": 1.0,
        "ratio_zero_denominator_policy": "piecewise",
        "window_seconds": 1.0,
        "withdrawal_window_seconds": 2.0,
        "reversion_horizon_seconds": 2.0,
        "execution_cluster_max_gap_ms": 100,
        "max_deceptive_order_age_seconds": 90.0,
        "gamma_grid": [0.1],
        "tick_size": 0.01,
        "actor_identity_mode": "client_then_firm",
        "execution_anchor_modes": ["passive", "aggressive"],
    }

    module._write_grid_summary(
        path,
        metadata=metadata,
        combined_scores=pl.concat([dominant, firm]),
    )

    text = path.read_text()
    assert "## Top actors within identity and execution-anchor strata" in text
    assert "client_original:C0" in text
    assert "firm:F1" in text
    assert "client_id" not in text


def test_grid_summary_keeps_only_best_score_row_per_actor_and_stratum():
    module = load_grid_module()
    repeated_actor = pl.DataFrame(
        {
            "actor_key": ["client_original:C1"] * 30,
            "identity_level": ["client_original"] * 30,
            "execution_anchor_mode": ["passive"] * 30,
            "top_n": list(range(1, 31)),
            "gamma": [0.1] * 30,
            "executions": [10] * 30,
            "MCPS_resting_profile": [float(100 - index) for index in range(30)],
            "max_MSCI_resting_profile": [1.0] * 30,
            "MCPS": [-99.0] * 30,
            "max_MSCI": [-99.0] * 30,
        }
    )
    second_actor = pl.DataFrame(
        {
            "actor_key": ["client_original:C2"],
            "identity_level": ["client_original"],
            "execution_anchor_mode": ["passive"],
            "top_n": [1],
            "gamma": [0.1],
            "executions": [1],
            "MCPS_resting_profile": [-1.0],
            "max_MSCI_resting_profile": [-1.0],
            "MCPS": [-99.0],
            "max_MSCI": [-99.0],
        }
    )

    rows = module._stratified_top_rows(pl.concat([repeated_actor, second_actor]), limit=25)

    assert [row["actor_key"] for row in rows] == ["client_original:C1", "client_original:C2"]
