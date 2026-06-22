from __future__ import annotations

import importlib.util
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
