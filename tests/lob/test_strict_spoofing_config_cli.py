from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "configs" / "spoofing_detection_parameters.json"


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"strict_config_{name}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _operational_args(name: str, tmp_path: Path) -> list[str]:
    common = ["--config", str(CONFIG_PATH)]
    if name == "compute_spoofing_metrics":
        return [*common, "--input", str(tmp_path / "input.parquet"), "--output-dir", str(tmp_path / "metrics")]
    if name == "run_multilevel_spoofing_grid":
        return [*common, "--input", str(tmp_path / "input.parquet"), "--output-dir", str(tmp_path / "grid")]
    if name == "build_spoofing_event_review_dashboard":
        return [
            *common,
            "--input",
            str(tmp_path / "input.parquet"),
            "--execution-metrics",
            str(tmp_path / "execution_metrics.parquet"),
            "--candidate-deceptive-orders",
            str(tmp_path / "candidate_deceptive_orders.parquet"),
            "--output-dir",
            str(tmp_path / "review"),
        ]
    if name == "compute_client_session_spoofing_features":
        return [
            *common,
            "--execution-metrics",
            str(tmp_path / "execution_metrics.parquet"),
            "--output-dir",
            str(tmp_path / "features"),
        ]
    if name == "run_spoofing_production_readiness":
        return [
            *common,
            "--execution-metrics",
            str(tmp_path / "execution_metrics.parquet"),
            "--event-log",
            str(tmp_path / "event_log.parquet"),
            "--output-dir",
            str(tmp_path / "readiness"),
        ]
    raise AssertionError(name)


@pytest.mark.parametrize(
    ("script_name", "parameter_override"),
    [
        ("compute_spoofing_metrics", ["--top-n", "10"]),
        ("run_multilevel_spoofing_grid", ["--depth-grid", "1,10"]),
        ("build_spoofing_event_review_dashboard", ["--pre-window-seconds", "5"]),
        ("compute_client_session_spoofing_features", ["--msci-threshold", "0.4"]),
        ("run_spoofing_production_readiness", ["--min-events", "1"]),
    ],
)
def test_scientific_parameters_cannot_be_overridden_on_cli(
    script_name: str,
    parameter_override: list[str],
    tmp_path: Path,
):
    module = _load_script(script_name)

    with pytest.raises(SystemExit):
        module.parse_args([*_operational_args(script_name, tmp_path), *parameter_override])


@pytest.mark.parametrize(
    ("script_name", "abbreviated_override"),
    [
        ("compute_spoofing_metrics", ["--top", "2"]),
        ("run_multilevel_spoofing_grid", ["--dep", "1"]),
        ("build_spoofing_event_review_dashboard", ["--max-e", "1"]),
        (
            "compute_client_session_spoofing_features",
            ["--msci-threshold-b", '{"passive": 0.2, "aggressive": null}'],
        ),
        ("run_spoofing_production_readiness", ["--min-e", "7"]),
    ],
)
def test_abbreviated_scientific_parameter_overrides_are_rejected(
    script_name: str,
    abbreviated_override: list[str],
    tmp_path: Path,
):
    module = _load_script(script_name)

    with pytest.raises(SystemExit):
        module.parse_args(
            [*_operational_args(script_name, tmp_path), *abbreviated_override]
        )


@pytest.mark.parametrize(
    "script_name",
    [
        "compute_spoofing_metrics",
        "run_multilevel_spoofing_grid",
        "build_spoofing_event_review_dashboard",
        "compute_client_session_spoofing_features",
        "run_spoofing_production_readiness",
    ],
)
def test_config_file_is_required(script_name: str, tmp_path: Path):
    module = _load_script(script_name)
    args = _operational_args(script_name, tmp_path)
    args[1] = str(tmp_path / "missing.json")

    with pytest.raises(SystemExit):
        module.parse_args(args)


def test_repository_config_drives_metric_detection_parameters(tmp_path: Path):
    module = _load_script("compute_spoofing_metrics")

    args = module.parse_args(_operational_args("compute_spoofing_metrics", tmp_path))

    assert args.top_n == 10
    assert not hasattr(args, "kappa")
    assert not hasattr(args, "lambda_")
    assert args.window_seconds == 10.0
    assert args.withdrawal_window_seconds == 2.0
    assert args.reversion_horizon_seconds == 2.0
    assert args.execution_cluster_max_gap_ms == 100
    assert args.max_deceptive_order_age_seconds == 90.0
    assert args.gamma_grid == "0.0,0.1,0.25,0.5,1.0,1.5"
    assert args.tick_size is None
    assert args.state_client_mode == "execution-actors"
    assert args.compact_state is True
    assert args.empirical_depth_kernel is None
    assert args.actor_identity_mode == "client_then_firm"
    assert args.execution_anchor_modes == ("passive", "aggressive")
