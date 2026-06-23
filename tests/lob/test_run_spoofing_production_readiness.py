from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_spoofing_production_readiness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_spoofing_production_readiness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_pipeline_writes_alerts(tmp_path):
    module = _load_module()
    executions = pl.DataFrame({"review_event_id": ["S1", "S2", "S3"], "client_id": ["A", "A", "A"], "MSCI": [0.8, 0.7, 0.1], "SCI": [0.9, 0.8, 0.2], "collapse_opposite_side": [0.7, 0.6, 0.1], "collapse_same_side": [0.1, 0.1, 0.1], "matched_deceptive_cancel_fraction_window": [0.9, 0.8, 0.1], "fill_qty": [100.0, 100.0, 100.0]})
    event_log = pl.DataFrame({"client_id": ["A", "A"], "side": ["bid", "ask"], "is_execution_order": [True, False], "is_matched_deceptive_cancel_order": [False, True], "displayed_qty": [100.0, 100.0], "last_shares": [50.0, 0.0]})
    execution_path = tmp_path / "execution_metrics.parquet"
    event_log_path = tmp_path / "event_log.parquet"
    output_dir = tmp_path / "readiness"
    executions.write_parquet(execution_path)
    event_log.write_parquet(event_log_path)
    outputs = module.run_pipeline(execution_metrics_path=execution_path, event_log_path=event_log_path, output_dir=output_dir, msci_threshold=0.5, min_events=2, min_mcps=0.5)
    assert outputs["alerts"].exists()
    assert pl.read_parquet(outputs["alerts"]).height == 1
