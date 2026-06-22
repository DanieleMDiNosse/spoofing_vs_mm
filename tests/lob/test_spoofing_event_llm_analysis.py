from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_spoofing_event_with_llm.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_spoofing_event_with_llm", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compose_prompt_combines_instruction_and_dossier():
    module = _load_module()
    text = module.compose_prompt("SYSTEM INSTRUCTIONS", "# Event dossier: S10")
    assert "SYSTEM INSTRUCTIONS" in text
    assert "# Event dossier: S10" in text
    assert "Now analyze the event dossier" in text


def test_write_analysis_artifacts_saves_response_and_metadata(tmp_path):
    module = _load_module()
    out = tmp_path / "review"
    module.write_analysis_artifacts(
        output_dir=out,
        prompt_text="PROMPT",
        response_text="RESPONSE",
        metadata={"model": "gemma4", "backend": "ollama"},
    )
    assert (out / "prompt.md").read_text() == "PROMPT"
    assert (out / "response.md").read_text() == "RESPONSE"
    assert json.loads((out / "metadata.json").read_text())["model"] == "gemma4"


def test_clean_response_removes_thinking_and_control_sequences():
    module = _load_module()
    raw = "Thinking...\nnotes\x1b[9D\x1b[K\n# Surveillance review for event S10\nBody\n"

    cleaned = module.clean_llm_response(raw)

    assert cleaned.startswith("# Surveillance review for event S10")
    assert "Thinking" not in cleaned
    assert "\x1b" not in cleaned
