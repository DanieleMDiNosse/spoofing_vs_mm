from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "run_spoofing_production_readiness.py"
CONFIG_PATH = REPO_ROOT / "configs" / "spoofing_detection_parameters.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_spoofing_production_readiness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_readiness_config(tmp_path: Path, **overrides: object) -> Path:
    payload = json.loads(CONFIG_PATH.read_text())
    payload["production_readiness"].update(overrides)
    config_path = tmp_path / "spoofing_parameters.json"
    config_path.write_text(json.dumps(payload))
    return config_path


def test_run_pipeline_writes_alerts(tmp_path):
    module = _load_module()
    executions = pl.DataFrame(
        {
            "review_event_id": ["S1", "S2", "S3"],
            "actor_key": ["client_original:A", "client_original:A", "client_original:A"],
            "actor_id": ["A", "A", "A"],
            "identity_level": ["client_original", "client_original", "client_original"],
            "identity_source": ["NMSC_ORIGINALCLIENTIDSHORTCODE"] * 3,
            "identity_fallback_flag": [False, False, False],
            "execution_anchor_mode": ["passive", "passive", "passive"],
            "MSCI_resting_profile": [0.8, 0.7, 0.1],
            "MSCI": [-99.0, -99.0, -99.0],
            "SCI": [0.9, 0.8, 0.2],
            "collapse_opposite_side": [0.7, 0.6, 0.1],
            "collapse_same_side": [0.1, 0.1, 0.1],
            "matched_deceptive_cancel_fraction_window": [0.9, 0.8, 0.1],
            "execution_quantity": [100.0, 100.0, 100.0],
            "fill_qty": [-99.0, -99.0, -99.0],
            "has_matched_deceptive_cancel_window": [True, True, False],
            "withdrawal_profile_scale_event": [3.0, 2.0, 0.0],
            "WMSCI_event": [-99.0, -99.0, -99.0],
            "favorable_mid_move_pre_fill": [0.01, 0.02, -0.01],
            "post_cancel_mid_reversion": [0.01, -0.01, -0.01],
        }
    )
    event_log = pl.DataFrame(
        {
            "actor_key": ["client_original:A", "client_original:A"],
            "actor_id": ["A", "A"],
            "identity_level": ["client_original", "client_original"],
            "identity_source": ["NMSC_ORIGINALCLIENTIDSHORTCODE"] * 2,
            "identity_fallback_flag": [False, False],
            "execution_anchor_mode": ["passive", "passive"],
            "side": ["bid", "ask"],
            "is_execution_order": [True, False],
            "is_matched_deceptive_cancel_order": [False, True],
            "displayed_qty": [100.0, 100.0],
            "last_shares": [50.0, 0.0],
        }
    )
    execution_path = tmp_path / "execution_metrics.parquet"
    event_log_path = tmp_path / "event_log.parquet"
    output_dir = tmp_path / "readiness"
    executions.write_parquet(execution_path)
    event_log.write_parquet(event_log_path)
    config_path = write_readiness_config(
        tmp_path,
        msci_threshold_by_anchor={"passive": 0.5, "aggressive": 0.5},
        min_events=2,
        min_mcps=0.5,
    )
    outputs = module.run_pipeline(
        execution_metrics_path=execution_path,
        event_log_path=event_log_path,
        output_dir=output_dir,
        msci_threshold=0.5,
        min_events=2,
        min_mcps=0.5,
        actor_identity_mode="client_then_firm",
        execution_anchor_modes=("passive", "aggressive"),
        config_path=config_path,
    )
    assert outputs["alerts"].exists()
    assert outputs["risk"].name == "actor_session_risk_features.parquet"
    assert outputs["legitimacy"].name == "actor_legitimacy_features.parquet"
    assert outputs["alerts"].name == "actor_session_alerts.parquet"
    alerts = pl.read_parquet(outputs["alerts"])
    assert alerts.height == 1
    assert "alert_score" not in alerts.columns
    assert alerts["max_WMSCI_event"].to_list() == [3.0]
    assert alerts["max_MSCI"].to_list() == [0.8]
    metadata = json.loads(outputs["metadata"].read_text())
    assert metadata["output_schema_version"] == "actor_execution_anchor_v2"
    assert metadata["execution_anchor_modes"] == ["passive", "aggressive"]
    assert metadata["observed_execution_anchor_modes"] == ["passive"]
    assert metadata["market_observation"] == (
        "configured_passive_and_aggressive_execution_branches; observed_passive_execution_branch_only"
    )
    assert metadata["score_grouping"] == ["actor_key", "execution_anchor_mode"]
    assert metadata["actor_feature_population"] == "attributable_execution_and_event_rows_only"
    assert metadata["excluded_unattributable_execution_rows"] == 0
    assert metadata["excluded_unattributable_event_rows"] == 0
    assert metadata["parameter_source"] == "json_config_only"

    empty_execution_path = tmp_path / "empty_execution_metrics.parquet"
    empty_event_log_path = tmp_path / "empty_event_log.parquet"
    executions.head(0).write_parquet(empty_execution_path)
    event_log.head(0).write_parquet(empty_event_log_path)
    empty_outputs = module.run_pipeline(
        execution_metrics_path=empty_execution_path,
        event_log_path=empty_event_log_path,
        output_dir=tmp_path / "empty_readiness",
        msci_threshold=0.5,
        min_events=2,
        min_mcps=0.5,
        actor_identity_mode="client_then_firm",
        execution_anchor_modes=("passive", "aggressive"),
    )
    empty_metadata = json.loads(empty_outputs["metadata"].read_text())
    assert empty_metadata["execution_anchor_modes"] == ["passive", "aggressive"]
    assert empty_metadata["observed_execution_anchor_modes"] == []
    assert empty_metadata["market_observation"] == (
        "configured_passive_and_aggressive_execution_branches; no_execution_branches_observed"
    )

    invalid_output_dir = tmp_path / "invalid_readiness"
    with pytest.raises(ValueError, match="unconfigured execution anchor mode"):
        module.run_pipeline(
            execution_metrics_path=execution_path,
            event_log_path=event_log_path,
            output_dir=invalid_output_dir,
            msci_threshold=0.5,
            min_events=2,
            min_mcps=0.5,
            actor_identity_mode="client_then_firm",
            execution_anchor_modes=("aggressive",),
        )
    assert not invalid_output_dir.exists()


def test_config_provenance_cannot_be_attached_to_mismatched_programmatic_parameters(
    tmp_path: Path,
) -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="do not match JSON config"):
        module.run_pipeline(
            execution_metrics_path=tmp_path / "not-read-executions.parquet",
            event_log_path=tmp_path / "not-read-events.parquet",
            output_dir=tmp_path / "output",
            msci_threshold=0.5,
            min_events=999,
            min_mcps=0.0,
            actor_identity_mode="client_then_firm",
            execution_anchor_modes=("passive", "aggressive"),
            config_path=CONFIG_PATH,
        )


def test_parse_args_loads_production_readiness_thresholds_from_config(tmp_path: Path):
    module = _load_module()
    config_path = write_readiness_config(
        tmp_path,
        msci_threshold_by_anchor={"passive": 0.001, "aggressive": None},
        min_events=4,
        min_mcps=0.05,
        actor_identity_mode="client_then_firm",
        execution_anchor_modes=["aggressive", "passive"],
    )

    args = module.parse_args(
        [
            "--config",
            str(config_path),
            "--execution-metrics",
            str(tmp_path / "execution_metrics.parquet"),
            "--event-log",
            str(tmp_path / "event_log.parquet"),
            "--output-dir",
            str(tmp_path / "readiness"),
        ]
    )

    assert args.config == config_path
    assert args.msci_threshold_by_anchor == {"passive": 0.001, "aggressive": None}
    assert args.min_events == 4
    assert args.min_mcps == 0.05
    assert args.actor_identity_mode == "client_then_firm"
    assert args.execution_anchor_modes == ("passive", "aggressive")

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--config",
                str(config_path),
                "--execution-metrics",
                str(tmp_path / "execution_metrics.parquet"),
                "--event-log",
                str(tmp_path / "event_log.parquet"),
                "--output-dir",
                str(tmp_path / "readiness"),
                "--msci-threshold=0.4",
            ]
        )


def test_parse_args_uses_positive_signed_msci_margin_from_config(tmp_path: Path):
    module = _load_module()

    args = module.parse_args(
        [
            "--execution-metrics",
            str(tmp_path / "execution_metrics.parquet"),
            "--event-log",
            str(tmp_path / "event_log.parquet"),
            "--output-dir",
            str(tmp_path / "readiness"),
        ]
    )

    assert args.msci_threshold is None
    assert args.msci_threshold_by_anchor == {"passive": 0.1, "aggressive": None}
    assert args.execution_anchor_modes == ("passive", "aggressive")


def test_parse_args_rejects_invalid_actor_identity_mode_from_config(tmp_path: Path):
    module = _load_module()
    config_path = write_readiness_config(tmp_path, actor_identity_mode="invalid_mode")

    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--config",
                str(config_path),
                "--execution-metrics",
                str(tmp_path / "execution_metrics.parquet"),
                "--event-log",
                str(tmp_path / "event_log.parquet"),
                "--output-dir",
                str(tmp_path / "readiness"),
            ]
        )
