from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "compute_spoofing_metrics.py"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "spoofing_detection_parameters.json"


def load_module():
    spec = importlib.util.spec_from_file_location("compute_spoofing_metrics", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_config(tmp_path: Path, **metrics_overrides: object) -> Path:
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
    metrics_overrides.setdefault("empirical_depth_kernel", str(kernel_path))
    payload["metrics"].update(metrics_overrides)
    config_path = tmp_path / "spoofing_parameters.json"
    config_path.write_text(json.dumps(payload))
    return config_path


def test_infer_state_client_ids_from_passive_limit_fills_only():
    module = load_module()
    raw = pl.DataFrame(
        {
            "ORDEREVENTTYPE (*)": [3, 3, 3, 1],
            "PASSIVEORDER": ["Y", "Y", None, None],
            "AGGRESSIVEORDER": ["N", "Y", "N", None],
            "ORDERTYPE (*)": [2, 2, 2, 2],
            "NMSC_ORIGINALCLIENTIDSHORTCODE": ["C1", "C2", "C3", "C4"],
        }
    )

    assert module._infer_state_client_ids(raw, mode="passive-fill-clients") == {"C1"}
    assert module._infer_state_client_ids(raw, mode="all") is None


def test_infer_state_client_ids_normalizes_string_enum_codes():
    module = load_module()
    raw = pl.DataFrame(
        {
            "ORDEREVENTTYPE (*)": ["3", "3 : Trade", "1"],
            "PASSIVEORDER": ["Y", "Y", "Y"],
            "AGGRESSIVEORDER": ["N", "N", "N"],
            "ORDERTYPE (*)": ["2", "5 : Market to limit", "2"],
            "NMSC_ORIGINALCLIENTIDSHORTCODE": ["C1", "C2", "C3"],
        }
    )

    assert module._infer_state_client_ids(raw, mode="passive-fill-clients") == {"C1", "C2"}


def test_infer_state_client_ids_excludes_zero_client_sentinel():
    module = load_module()
    raw = pl.DataFrame(
        {
            "ORDEREVENTTYPE (*)": [3, 3],
            "PASSIVEORDER": ["Y", "Y"],
            "AGGRESSIVEORDER": ["N", "N"],
            "ORDERTYPE (*)": [2, 2],
            "NMSC_ORIGINALCLIENTIDSHORTCODE": [0.0, 123.0],
        }
    )

    assert module._infer_state_client_ids(raw, mode="passive-fill-clients") == {"123"}


def test_infer_state_actor_keys_from_selected_execution_fills():
    module = load_module()
    raw = pl.DataFrame(
        {
            "ORDEREVENTTYPE (*)": [3, 3, 3, 3, 3, 1],
            "PASSIVEORDER": ["Y", "N", "Y", "N", "Y", "Y"],
            "AGGRESSIVEORDER": ["N", "Y", "Y", "N", "N", "N"],
            "NMSC_ORIGINALCLIENTIDSHORTCODE": ["C1", None, "C3", "C4", "null", "C6"],
            "FIRMID": ["F1", "F2", "F3", "F4", "F5", "F6"],
        }
    )

    assert module._infer_state_actor_keys(
        raw,
        mode="execution-actors",
        execution_anchor_modes=("passive", "aggressive"),
    ) == {"client_original:C1", "firm:F2", "firm:F5"}
    assert module._infer_state_actor_keys(
        raw,
        mode="execution-actors",
        execution_anchor_modes=("passive",),
    ) == {"client_original:C1", "firm:F5"}
    assert module._infer_state_actor_keys(
        raw,
        mode="all",
        execution_anchor_modes=("passive", "aggressive"),
    ) is None


def test_infer_state_actor_keys_normalizes_string_event_codes():
    module = load_module()
    raw = pl.DataFrame(
        {
            "ORDEREVENTTYPE (*)": ["3", "3 : Trade", "1"],
            "PASSIVEORDER": ["Y", "N", "Y"],
            "AGGRESSIVEORDER": ["N", "Y", "N"],
            "NMSC_ORIGINALCLIENTIDSHORTCODE": ["C1", None, "C3"],
            "FIRMID": ["F1", "F2", "F3"],
        }
    )

    assert module._infer_state_actor_keys(
        raw,
        mode="execution-actors",
        execution_anchor_modes=("passive", "aggressive"),
    ) == {"client_original:C1", "firm:F2"}


def test_parse_args_loads_compact_memory_options_from_config(tmp_path: Path):
    module = load_module()

    args = module.parse_args(
        [
            "--input",
            str(tmp_path / "input.parquet"),
            "--quote-panel",
            str(tmp_path / "quotes.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert args.state_client_mode == "execution-actors"
    assert args.compact_state is True


def test_repository_config_defaults_to_execution_actor_compact_dual_state(tmp_path: Path):
    module = load_module()

    args = module.parse_args(
        [
            "--input",
            str(tmp_path / "input.parquet"),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )

    assert args.state_client_mode == "execution-actors"
    assert args.compact_state is True
    assert args.actor_identity_mode == "client_then_firm"
    assert args.execution_anchor_modes == ("passive", "aggressive")


def test_parse_args_uses_signed_msci_gamma_grid_from_config(tmp_path: Path):
    module = load_module()

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


def test_parse_args_validates_and_canonicalizes_execution_anchor_modes(tmp_path: Path):
    module = load_module()
    config_path = write_config(tmp_path, execution_anchor_modes=["aggressive", "passive"])
    common = [
        "--config",
        str(config_path),
        "--input",
        str(tmp_path / "input.parquet"),
        "--output-dir",
        str(tmp_path / "out"),
    ]

    args = module.parse_args(common)

    assert args.actor_identity_mode == "client_then_firm"
    assert args.execution_anchor_modes == ("passive", "aggressive")

    for invalid in ([], ["passive", "unknown"], ["passive", "passive"]):
        invalid_config = write_config(tmp_path, execution_anchor_modes=invalid)
        with pytest.raises(SystemExit):
            module.parse_args(["--config", str(invalid_config), *common[2:]])


def test_parse_args_rejects_invalid_actor_identity_mode_from_config(tmp_path: Path):
    module = load_module()
    config_path = write_config(tmp_path, actor_identity_mode="invalid_mode")

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

    config_path = write_config(tmp_path, execution_anchor_modes={"passive": True})
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

    config_path = write_config(tmp_path, state_client_mode="invalid-mode")
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


def test_parse_args_loads_spoofing_metric_parameters_only_from_config(tmp_path: Path):
    module = load_module()
    config_path = write_config(
        tmp_path,
        top_n=5,

        window_seconds=30.0,
        withdrawal_window_seconds=2.0,
        reversion_horizon_seconds=3.0,
        execution_cluster_max_gap_ms=250,
        max_deceptive_order_age_seconds=120.0,
        gamma_grid=[0.001, 0.01],
        state_client_mode="passive-fill-clients",
        compact_state=True,
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
    assert args.top_n == 5
    assert not hasattr(args, "kappa")
    assert not hasattr(args, "lambda_")
    assert args.window_seconds == 30.0
    assert args.withdrawal_window_seconds == 2.0
    assert args.reversion_horizon_seconds == 3.0
    assert not hasattr(args, "withdrawal_excess_alpha")
    assert args.execution_cluster_max_gap_ms == 250
    assert args.max_deceptive_order_age_seconds == 120.0
    assert args.gamma_grid == "0.001,0.01"
    assert args.state_client_mode == "passive-fill-clients"
    assert args.compact_state is True
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
                "--top-n",
                "7",
            ]
        )


def test_parse_args_validates_execution_cluster_gap(tmp_path: Path):
    module = load_module()
    config_path = write_config(tmp_path, tick_size=0.01, execution_cluster_max_gap_ms=50)
    common = [
        "--config",
        str(config_path),
        "--input",
        str(tmp_path / "input.parquet"),
        "--output-dir",
        str(tmp_path / "out"),
    ]

    args = module.parse_args(common)
    assert args.execution_cluster_max_gap_ms == 50

    invalid_config = write_config(tmp_path, tick_size=0.01, execution_cluster_max_gap_ms=-1)
    with pytest.raises(SystemExit):
        module.parse_args(["--config", str(invalid_config), *common[2:]])


def test_parse_args_rejects_removed_epsilon_option(tmp_path: Path):
    module = load_module()
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


def test_top_execution_tables_rank_within_identity_and_anchor_strata():
    module = load_module()
    executions = pl.DataFrame(
        {
            "actor_id": ["CP", "CA", "FP", "FA"],
            "identity_level": ["client_original", "client_original", "firm", "firm"],
            "execution_anchor_mode": ["passive", "aggressive", "passive", "aggressive"],
            "MSCI_resting_profile": [4.0, 3.0, 2.0, 1.0],
            "MSCI": [-99.0, -99.0, -99.0, -99.0],
            "has_matched_deceptive_cancel_window": [True, True, True, True],
            "matched_deceptive_cancel_visible_qty_window": [40.0, 30.0, 20.0, 10.0],
        }
    )

    execution_lines = "\n".join(module._top_execution_lines(executions, limit=1))
    cancel_lines = "\n".join(module._top_deceptive_cancel_lines(executions, limit=1))

    for actor_id in ("CP", "CA", "FP", "FA"):
        assert actor_id in execution_lines
        assert actor_id in cancel_lines


def test_main_versions_actor_anchor_artifacts_and_audit_metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    module = load_module()
    input_path = tmp_path / "input.parquet"
    output_dir = tmp_path / "out"
    pl.DataFrame(
        {
            "NMSC_ORIGINALCLIENTIDSHORTCODE": [None, None],
            "FIRMID": [None, "F1"],
            "ORDER_TRADINGCAPACITY (*)": [1, 1],
            "ORDEREVENTTYPE (*)": [3, 3],
            "PASSIVEORDER": ["Y", "N"],
            "AGGRESSIVEORDER": ["N", "Y"],
        },
        schema={
            "NMSC_ORIGINALCLIENTIDSHORTCODE": pl.String,
            "FIRMID": pl.String,
            "ORDER_TRADINGCAPACITY (*)": pl.Int64,
            "ORDEREVENTTYPE (*)": pl.Int64,
            "PASSIVEORDER": pl.String,
            "AGGRESSIVEORDER": pl.String,
        },
    ).write_parquet(input_path)

    empty = pl.DataFrame()
    executions = pl.DataFrame(
        {
            "execution_anchor_mode": ["passive", "aggressive"],
            "identity_level": ["client_original", "firm"],
            "has_matched_deceptive_cancel_window": [True, True],
            "spoofing_compatible_sequence": [True, False],
            "candidate_deceptive_order_count_pre": [1, 1],
            "matched_deceptive_cancel_visible_qty_window": [10.0, 20.0],
            "SCI": [0.2, 0.3],
            "MSCI_resting_profile": [0.4, 0.5],
            "MSCI": [-99.0, -99.0],
        }
    )
    result = SimpleNamespace(
        state_time_series=pl.DataFrame({"actor_key": ["client_original:C1", "firm:F1"]}),
        execution_metrics=executions,
        candidate_deceptive_orders=empty,
        direct_cancellations=empty,
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
    monkeypatch.setattr(module, "compute_mcps_scores", lambda *args, **kwargs: pl.DataFrame())
    monkeypatch.setattr(module, "_write_parquet", lambda df, path: path.write_bytes(b"artifact"))
    monkeypatch.setattr(module, "_write_csv", lambda df, path: path.write_text(""))

    module.main(
        [
            "--config",
            str(write_config(tmp_path, tick_size=0.01)),
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
    assert metadata["execution_branch_applicability"]["aggressive_only"] == [
        "aggressive_execution_quantity",
        "aggressive_execution_vwap",
        "aggressive_child_fill_count",
        "aggressive_execution_price_level_count",
        "aggressive_execution_price_min",
        "aggressive_execution_price_max",
        "aggressive_execution_sweep_id",
        "WMSCI_aggressive",
    ]
    assert metadata["firm_fallback_semantics"] == "aggregate only when client_original_id is missing"
    assert Path(metadata["paths"]["state_time_series"]).name == "actor_metric_time_series.parquet"
    assert Path(metadata["paths"]["actor_mcps_scores"]).name == "actor_mcps_scores.parquet"
    assert metadata["actor_execution_audit"]["rows_missing_client_and_firm_identity"] == 1
    assert metadata["actor_execution_audit"]["execution_clusters_by_anchor_mode"] == [
        {"execution_anchor_mode": "aggressive", "rows": 1},
        {"execution_anchor_mode": "passive", "rows": 1},
    ]
    assert not (output_dir / "client_metric_time_series.parquet").exists()
    assert not (output_dir / "client_mcps_scores.parquet").exists()
