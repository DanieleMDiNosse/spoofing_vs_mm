from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
ADAPTER_PATH = ROOT / "scripts" / "run_h1_checkpointed_sequence.py"


def load_adapter():
    spec = importlib.util.spec_from_file_location("h1_checkpointed_sequence", ADAPTER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_supervisor(tmp_path: Path):
    source_build_plan = object()
    supervisor = SimpleNamespace(
        INSTRUMENTS=("FERRARI", "NEXI", "RISANAMENTO"),
        WORKER_SCRIPT=tmp_path / "instrumented.py",
        PYTHON=Path(sys.executable),
        build_plan=source_build_plan,
    )
    calls: list[Path] = []

    def original_build_plan(output_root: Path):
        calls.append(output_root)
        plan = []
        for instrument in supervisor.INSTRUMENTS:
            provenance = output_root / "configs" / f"{instrument}.json"
            provenance.parent.mkdir(parents=True, exist_ok=True)
            provenance.write_bytes(json.dumps({"instrument": instrument}).encode())
            output_dir = output_root / "runs" / instrument
            compute_argv = [
                "--config", str(provenance), "--input", f"/{instrument}.parquet",
                "--output-dir", str(output_dir), "--clock-policy-v3",
            ]
            plan.append({
                "instrument": instrument,
                "config": str(provenance),
                "config_sha256": sha256(provenance),
                "output_dir": str(output_dir),
                "command": [str(supervisor.PYTHON), str(supervisor.WORKER_SCRIPT), "--progress-seconds", "30", "--", *compute_argv],
            })
        return {"schema": "fake", "runs": plan}, plan

    supervisor.build_plan = original_build_plan
    return supervisor, calls


def test_adapted_plan_reuses_stable_config_and_rewrites_checkpoint_command(tmp_path: Path):
    adapter = load_adapter()
    supervisor, calls = fake_supervisor(tmp_path)
    checkpoint_root = tmp_path / "checkpoints"

    first_manifest, first_plan = adapter.adapt_build_plan(
        supervisor=supervisor,
        original_build_plan=supervisor.build_plan,
        checkpoint_root=checkpoint_root,
        output_root=tmp_path / "attempt-one",
    )
    second_manifest, second_plan = adapter.adapt_build_plan(
        supervisor=supervisor,
        original_build_plan=supervisor.build_plan,
        checkpoint_root=checkpoint_root,
        output_root=tmp_path / "attempt-two",
    )

    assert calls == [tmp_path / "attempt-one", tmp_path / "attempt-two"]
    assert first_manifest["checkpoint_root"] == str(checkpoint_root.resolve())
    assert first_manifest["adapter"] == str(ADAPTER_PATH.resolve())
    assert first_manifest["checkpoint_store"]["sha256"] == sha256(ROOT / "scripts" / "h1_stage_checkpoints.py")
    assert supervisor.WORKER_SCRIPT == ROOT / "scripts" / "h1_checkpointed_worker.py"
    for first, second in zip(first_plan, second_plan, strict=True):
        stable_config = checkpoint_root / "configs" / f"{first['instrument']}.json"
        assert Path(first["generated_config_provenance_copy"]) == tmp_path / "attempt-one" / "configs" / f"{first['instrument']}.json"
        assert Path(second["generated_config_provenance_copy"]) == tmp_path / "attempt-two" / "configs" / f"{first['instrument']}.json"
        assert first["config"] == str(stable_config)
        assert second["config"] == str(stable_config)
        assert first["config_sha256"] == second["config_sha256"] == sha256(stable_config)
        assert first["command"] == [
            str(supervisor.PYTHON), str(ROOT / "scripts" / "h1_checkpointed_worker.py"),
            "--checkpoint-dir", str(checkpoint_root / "instruments" / first["instrument"]),
            "--progress-seconds", "30", "--",
            "--config", str(stable_config), "--input", f"/{first['instrument']}.parquet",
            "--output-dir", str(tmp_path / "attempt-one" / "runs" / first["instrument"]), "--clock-policy-v3",
        ]
        assert second["command"][-1:] == ["--clock-policy-v3"]


def test_adapted_plan_rejects_config_byte_mismatch_without_replacing_stable_config(tmp_path: Path):
    adapter = load_adapter()
    supervisor, _ = fake_supervisor(tmp_path)
    checkpoint_root = tmp_path / "checkpoints"
    stable = checkpoint_root / "configs" / "FERRARI.json"
    stable.parent.mkdir(parents=True)
    stable.write_text("different")

    with pytest.raises(ValueError, match="stable checkpoint config bytes differ"):
        adapter.adapt_build_plan(
            supervisor=supervisor,
            original_build_plan=supervisor.build_plan,
            checkpoint_root=checkpoint_root,
            output_root=tmp_path / "attempt",
        )
    assert stable.read_text() == "different"


def test_parse_adapter_args_validates_known_unique_instruments_and_forwards_supervisor_args():
    adapter = load_adapter()

    options = adapter.parse_adapter_args([
        "--checkpoint-root", "/cache", "--instruments", "NEXI,RISANAMENTO", "--",
        "--output-root", "/out", "--per-instrument-wall-seconds", "99",
    ])
    assert options.checkpoint_root == Path("/cache")
    assert options.instruments == ("NEXI", "RISANAMENTO")
    assert options.supervisor_argv == ["--output-root", "/out", "--per-instrument-wall-seconds", "99"]

    for value, message in [("NEXI,NEXI", "unique"), ("UNKNOWN", "unknown"), ("", "at least one")]:
        with pytest.raises(ValueError, match=message):
            adapter.parse_adapter_args(["--checkpoint-root", "/cache", "--instruments", value, "--", "--output-root", "/out"])


def test_main_forwards_supervisor_args_and_requires_native_bundle_validation(monkeypatch, tmp_path: Path):
    adapter = load_adapter()
    supervisor, _ = fake_supervisor(tmp_path)
    observed: dict[str, object] = {}

    def local_validate(output_dir: Path, config_hash: str):
        observed["local"] = (output_dir, config_hash)
        return {"local": True}

    def fake_main():
        observed["argv"] = list(sys.argv)
        manifest, plan = supervisor.build_plan(tmp_path / "out")
        observed["manifest"] = manifest
        observed["plan"] = plan
        observed["validation"] = supervisor.validate_completed_output(tmp_path / "bundle", "config")
        return 17

    supervisor.validate_completed_output = local_validate
    supervisor.main = fake_main
    monkeypatch.setattr(adapter, "load_supervisor", lambda: supervisor)
    monkeypatch.setattr(adapter, "load_bundle_validator", lambda: SimpleNamespace(validate_bundle=lambda bundle: {"bundle": str(bundle), "native": True}))

    assert adapter.main([
        "--checkpoint-root", str(tmp_path / "cache"), "--instruments", "NEXI", "--",
        "--output-root", str(tmp_path / "out"), "--per-instrument-wall-seconds", "99",
    ]) == 17
    assert supervisor.INSTRUMENTS == ("NEXI",)
    assert observed["argv"] == [str(adapter.SUPERVISOR_PATH), "--output-root", str(tmp_path / "out"), "--per-instrument-wall-seconds", "99"]
    assert observed["validation"] == {"local": True, "native_bundle_validator": {"bundle": str(tmp_path / "bundle"), "native": True}}
    assert observed["local"] == (tmp_path / "bundle", "config")


def test_main_rejects_output_and_checkpoint_root_overlap_before_dispatch(monkeypatch, tmp_path: Path):
    adapter = load_adapter()
    supervisor, _ = fake_supervisor(tmp_path)
    imported = {"called": False}
    monkeypatch.setattr(adapter, "load_supervisor", lambda: supervisor)
    monkeypatch.setattr(supervisor, "main", lambda: imported.__setitem__("called", True) or 0, raising=False)

    with pytest.raises(ValueError, match="must not contain"):
        adapter.main([
            "--checkpoint-root", str(tmp_path / "out" / "cache"), "--instruments", "NEXI", "--",
            "--output-root", str(tmp_path / "out"),
        ])
    assert not imported["called"]
