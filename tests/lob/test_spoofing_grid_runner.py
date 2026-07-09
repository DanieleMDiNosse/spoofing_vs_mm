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
