from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SUPERVISOR_PATH = ROOT / "scripts" / "run_h1_age30_quantity_only_shared_sequence.py"


def load_supervisor():
    spec = importlib.util.spec_from_file_location("h1_supervisor", SUPERVISOR_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_plan(module, output_root: Path, command: list[str]):
    (output_root / "logs").mkdir(exist_ok=True)
    entry = {
        "instrument": "SYNTHETIC",
        "config_sha256": "synthetic-config",
        "output_dir": str(output_root / "runs" / "SYNTHETIC"),
        "log": str(output_root / "logs" / "SYNTHETIC.log"),
        "command": command,
        "status": "planned",
    }
    return {"schema": "synthetic_h1_supervisor_v1", "runs": [entry]}, [entry]


def invoke_main(monkeypatch, module, output_root: Path, command: list[str], *extra: str) -> int:
    monkeypatch.setattr(module, "build_plan", lambda root: fake_plan(module, root, command))
    monkeypatch.setattr(module, "validate_completed_output", lambda *_: {"synthetic": True})
    monkeypatch.setattr(
        sys,
        "argv",
        [str(SUPERVISOR_PATH), "--output-root", str(output_root), "--rss-sample-seconds", "0.01", *extra],
    )
    return module.main()


def test_worker_command_wraps_existing_compute_arguments_exactly():
    supervisor = load_supervisor()

    assert supervisor.worker_command(["--config", "c.json", "--input", "events.parquet"]) == [
        str(supervisor.PYTHON), str(supervisor.WORKER_SCRIPT),
        "--progress-seconds", "30", "--",
        "--config", "c.json", "--input", "events.parquet",
    ]


def test_main_dispatches_synthetic_child_and_records_wrapper_provenance(monkeypatch, tmp_path: Path):
    supervisor = load_supervisor()
    command = [sys.executable, "-c", "import time; time.sleep(0.1); print('synthetic normal child')"]

    assert invoke_main(monkeypatch, supervisor, tmp_path / "normal", command, "--available-memory-floor-bytes", "1") == 0

    manifest = json.loads((tmp_path / "normal" / "manifest.json").read_text())
    result = manifest["runs"][0]
    assert result["status"] == "completed"
    assert manifest["worker_wrapper"] == str(supervisor.WORKER_SCRIPT)
    assert manifest["worker_wrapper_sha256"] == supervisor.sha256(supervisor.WORKER_SCRIPT)
    telemetry = Path(result["supervisor_telemetry"])
    sample = json.loads(telemetry.read_text().splitlines()[0])
    assert sample["child_rss_bytes"] >= 0
    assert sample["mem_available_bytes"] > 0
    assert "swap_free_bytes" in sample


def test_main_terminates_synthetic_process_group_when_memory_floor_is_breached(monkeypatch, tmp_path: Path):
    supervisor = load_supervisor()
    command = [sys.executable, "-c", "import time; time.sleep(60)"]

    assert invoke_main(monkeypatch, supervisor, tmp_path / "floor", command, "--available-memory-floor-bytes", str(2**63)) == 1

    checkpoint = json.loads((tmp_path / "floor" / "checkpoint.json").read_text())
    result = checkpoint["runs"]["SYNTHETIC"]
    assert result["status"] == "failed"
    assert result["failure_type"] == "memory_floor_breach"
    assert result["exit_code"] != 0
    assert result["memory_floor_breach"]["mem_available_bytes"] < 2**63


@pytest.mark.parametrize("failure", ["wall_timeout", "memory_unavailable", "worker_nonzero_exit"])
def test_failure_paths_do_not_certify_output(monkeypatch, tmp_path: Path, failure: str):
    supervisor = load_supervisor()
    command = [sys.executable, "-c", "import time; time.sleep(5)"]
    extra = ["--available-memory-floor-bytes", "1", "--per-instrument-wall-seconds", "1"]
    if failure == "memory_unavailable":
        monkeypatch.setattr(supervisor, "system_memory_snapshot", lambda: {
            "mem_available_bytes": None, "swap_free_bytes": None,
        })
    elif failure == "worker_nonzero_exit":
        command = [sys.executable, "-c", "raise SystemExit(3)"]
    root = tmp_path / failure
    assert invoke_main(monkeypatch, supervisor, root, command, *extra) == 1
    checkpoint = json.loads((root / "checkpoint.json").read_text())
    result = checkpoint["runs"]["SYNTHETIC"]
    assert checkpoint["status"] == "failed"
    assert result["failure_type"] == failure
    assert "validation" not in result
    samples = [json.loads(line) for line in Path(result["supervisor_telemetry"]).read_text().splitlines()]
    assert samples
    assert not Path(f"/proc/{samples[0]['child_pid']}").exists()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf"])
def test_main_rejects_nonpositive_or_nonfinite_memory_floor(monkeypatch, tmp_path: Path, value: str):
    supervisor = load_supervisor()
    monkeypatch.setattr(sys, "argv", [str(SUPERVISOR_PATH), "--output-root", str(tmp_path / value), "--available-memory-floor-bytes", value])

    with pytest.raises(ValueError, match="memory floor"):
        supervisor.main()