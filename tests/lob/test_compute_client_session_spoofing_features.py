from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "compute_client_session_spoofing_features.py"
CONFIG_PATH = REPO_ROOT / "configs" / "spoofing_detection_parameters.json"


def _load_module():
    spec = importlib.util.spec_from_file_location("compute_client_session_spoofing_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def write_session_config(tmp_path: Path, **overrides: object) -> Path:
    payload = json.loads(CONFIG_PATH.read_text())
    payload["session_features"].update(overrides)
    config_path = tmp_path / "spoofing_parameters.json"
    config_path.write_text(json.dumps(payload))
    return config_path


def test_compute_and_write_features(tmp_path):
    module = _load_module()
    executions = pl.DataFrame(
        {
            "actor_key": ["firm:A"],
            "actor_id": ["A"],
            "identity_level": ["firm"],
            "identity_source": ["FIRMID"],
            "identity_fallback_flag": [True],
            "execution_anchor_mode": ["aggressive"],
            "MSCI_resting_profile": [0.8],
            "MSCI": [-99.0],
            "SCI": [0.7],
            "collapse_opposite_side": [0.6],
            "collapse_same_side": [0.1],
            "matched_deceptive_cancel_fraction_window": [0.9],
            "execution_quantity": [100.0],
            "fill_qty": [-99.0],
        }
    )
    input_path = tmp_path / "execution_metrics.parquet"
    output_dir = tmp_path / "features"
    executions.write_parquet(input_path)
    config_path = write_session_config(
        tmp_path,
        msci_threshold_by_anchor={"passive": 0.5, "aggressive": 0.5},
    )
    output = module.compute_and_write(
        input_path=input_path,
        output_dir=output_dir,
        msci_threshold=0.5,
        actor_identity_mode="client_then_firm",
        execution_anchor_modes=("passive", "aggressive"),
        config_path=config_path,
    )
    assert output["parquet"].exists()
    assert output["csv"].exists()
    assert output["parquet"].name == "actor_session_features.parquet"
    assert output["csv"].name == "actor_session_features.csv"
    assert pl.read_parquet(output["parquet"]).height == 1
    metadata = json.loads(output["metadata"].read_text())
    assert metadata["output_schema_version"] == "actor_execution_anchor_v2"
    assert metadata["actor_identity_mode"] == "client_then_firm"
    assert metadata["execution_anchor_modes"] == ["passive", "aggressive"]
    assert metadata["observed_execution_anchor_modes"] == ["aggressive"]
    assert metadata["score_grouping"] == ["actor_key", "execution_anchor_mode"]
    assert metadata["excluded_unattributable_execution_rows"] == 0
    assert metadata["parameter_source"] == "json_config_only"

    empty_input_path = tmp_path / "empty_execution_metrics.parquet"
    executions.head(0).write_parquet(empty_input_path)
    empty_output = module.compute_and_write(
        input_path=empty_input_path,
        output_dir=tmp_path / "empty_features",
        msci_threshold=0.5,
        actor_identity_mode="client_then_firm",
        execution_anchor_modes=("passive", "aggressive"),
    )
    empty_metadata = json.loads(empty_output["metadata"].read_text())
    assert empty_metadata["execution_anchor_modes"] == ["passive", "aggressive"]
    assert empty_metadata["observed_execution_anchor_modes"] == []

    invalid_output_dir = tmp_path / "invalid_features"
    with pytest.raises(ValueError, match="unconfigured execution anchor mode"):
        module.compute_and_write(
            input_path=input_path,
            output_dir=invalid_output_dir,
            msci_threshold=0.5,
            actor_identity_mode="client_then_firm",
            execution_anchor_modes=("passive",),
        )
    assert not invalid_output_dir.exists()


def test_compute_and_write_excludes_unattributable_execution_rows(tmp_path):
    module = _load_module()
    executions = pl.DataFrame(
        {
            "actor_key": ["client_original:C1", ""],
            "actor_id": ["C1", None],
            "identity_level": ["client_original", None],
            "identity_source": ["NMSC_ORIGINALCLIENTIDSHORTCODE", None],
            "identity_fallback_flag": [False, None],
            "execution_anchor_mode": ["passive", "passive"],
            "MSCI": [0.8, 0.9],
            "SCI": [0.7, 1.0],
            "collapse_opposite_side": [0.6, 0.9],
            "collapse_same_side": [0.1, 0.0],
            "matched_deceptive_cancel_fraction_window": [0.9, 1.0],
            "fill_qty": [100.0, 50.0],
        }
    )
    input_path = tmp_path / "execution_metrics.parquet"
    executions.write_parquet(input_path)

    output = module.compute_and_write(
        input_path=input_path,
        output_dir=tmp_path / "features",
        msci_threshold=0.5,
        execution_anchor_modes=("passive", "aggressive"),
    )

    features = pl.read_parquet(output["parquet"])
    metadata = json.loads(output["metadata"].read_text())
    assert features.get_column("actor_key").to_list() == ["client_original:C1"]
    assert metadata["excluded_unattributable_execution_rows"] == 1
    assert metadata["actor_feature_population"] == "attributable_execution_rows_only"


def test_cli_uses_positive_signed_msci_margin_from_config(tmp_path):
    module = _load_module()

    args = module.parse_args(
        [
            "--execution-metrics",
            str(tmp_path / "execution_metrics.parquet"),
            "--output-dir",
            str(tmp_path / "features"),
        ]
    )

    assert args.msci_threshold is None
    assert args.msci_threshold_by_anchor == {"passive": 0.1, "aggressive": None}
    assert args.actor_identity_mode == "client_then_firm"
    assert args.execution_anchor_modes == ("passive", "aggressive")


def test_config_provenance_cannot_be_attached_to_mismatched_programmatic_parameters(
    tmp_path: Path,
) -> None:
    module = _load_module()
    with pytest.raises(ValueError, match="do not match JSON config"):
        module.compute_and_write(
            input_path=tmp_path / "not-read.parquet",
            output_dir=tmp_path / "output",
            msci_threshold=0.5,
            actor_identity_mode="client_then_firm",
            execution_anchor_modes=("passive", "aggressive"),
            config_path=CONFIG_PATH,
        )


def test_cli_loads_and_validates_actor_anchor_config(tmp_path):
    module = _load_module()
    config_path = write_session_config(
        tmp_path,
        msci_threshold_by_anchor={"passive": 0.25, "aggressive": None},
        actor_identity_mode="client_then_firm",
        execution_anchor_modes=["aggressive", "passive"],
    )
    common = [
        "--config",
        str(config_path),
        "--execution-metrics",
        str(tmp_path / "execution_metrics.parquet"),
        "--output-dir",
        str(tmp_path / "features"),
    ]

    args = module.parse_args(common)
    assert args.msci_threshold_by_anchor == {"passive": 0.25, "aggressive": None}
    assert args.execution_anchor_modes == ("passive", "aggressive")

    with pytest.raises(SystemExit):
        module.parse_args([*common, "--msci-threshold", "0.4"])

    config_path = write_session_config(tmp_path, actor_identity_mode="invalid_mode")
    with pytest.raises(SystemExit):
        module.parse_args(["--config", str(config_path), *common[2:]])
