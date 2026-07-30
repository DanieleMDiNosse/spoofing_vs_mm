from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compute_client_session_spoofing_features.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compute_client_session_spoofing_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
    output = module.compute_and_write(
        input_path=input_path,
        output_dir=output_dir,
        msci_threshold=0.5,
        actor_identity_mode="client_then_firm",
        execution_anchor_modes=("passive", "aggressive"),
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


def test_cli_default_uses_positive_signed_msci_margin(tmp_path):
    module = _load_module()

    args = module.parse_args(
        [
            "--config",
            str(tmp_path / "missing.json"),
            "--execution-metrics",
            str(tmp_path / "execution_metrics.parquet"),
            "--output-dir",
            str(tmp_path / "features"),
        ]
    )

    assert args.msci_threshold == 0.1
    assert args.msci_threshold_by_anchor is None
    assert args.actor_identity_mode == "client_then_firm"
    assert args.execution_anchor_modes == ("passive",)


def test_cli_loads_and_validates_actor_anchor_config(tmp_path):
    module = _load_module()
    config_path = tmp_path / "spoofing_parameters.json"
    config_path.write_text(
        json.dumps(
            {
                "session_features": {
                    "msci_threshold_by_anchor": {
                        "passive": 0.25,
                        "aggressive": None,
                    },
                    "actor_identity_mode": "client_then_firm",
                    "execution_anchor_modes": ["aggressive", "passive"],
                }
            }
        )
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

    scalar_override = module.parse_args([*common, "--msci-threshold", "0.4"])
    assert scalar_override.msci_threshold_by_anchor == {
        "passive": 0.4,
        "aggressive": None,
    }

    map_override = module.parse_args(
        [
            *common,
            "--msci-threshold",
            "0.4",
            "--msci-threshold-by-anchor",
            '{"passive": 0.3, "aggressive": null}',
        ]
    )
    assert map_override.msci_threshold_by_anchor == {
        "passive": 0.3,
        "aggressive": None,
    }

    config_path.write_text(json.dumps({"session_features": {"actor_identity_mode": "invalid_mode"}}))
    with pytest.raises(SystemExit):
        module.parse_args(common)
