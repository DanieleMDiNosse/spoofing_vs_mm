from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import patch

import polars as pl


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_spoofing_event_with_llm.py"
PROMPT_PATH = Path(__file__).resolve().parents[2] / "prompts" / "spoofing_surveillance_analyst.md"


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


def test_default_prompt_prioritizes_matched_withdrawal_and_price_response_evidence():
    text = PROMPT_PATH.read_text().lower()

    assert "wmsci" in text
    assert "withdrawal-to-fill ratio" in text
    assert "cancellation delay" in text
    assert "favorable pre-fill price movement" in text
    assert "post-cancel reversion" in text
    assert "execution advantage" in text
    assert "msci" in text
    assert "secondary" in text
    assert "not causal evidence" in text
    assert "focal matched-withdrawal timeline" in text


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


def test_openai_compatible_call_uses_chat_completions_and_returns_content():
    module = _load_module()

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"content": "# Surveillance review for event S10\nBody"},
                        }
                    ]
                }
            ).encode()

    with patch.object(module.urllib.request, "urlopen", return_value=_Response()) as urlopen:
        response = module.call_openai_compatible(
            model="Bonsai-27B-Q1_0.gguf",
            prompt_text="PROMPT",
            timeout_seconds=30,
            temperature=0.1,
            api_base="http://127.0.0.1:8080/v1",
            api_key="local",
            max_tokens=3072,
        )

    assert response.startswith("# Surveillance review for event S10")
    request = urlopen.call_args.args[0]
    payload = json.loads(request.data)
    assert request.full_url == "http://127.0.0.1:8080/v1/chat/completions"
    assert payload["model"] == "Bonsai-27B-Q1_0.gguf"
    assert payload["max_tokens"] == 3072


def test_openai_compatible_call_rejects_empty_length_limited_response():
    module = _load_module()

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return None

        def read(self):
            return json.dumps(
                {"choices": [{"finish_reason": "length", "message": {"content": ""}}]}
            ).encode()

    with patch.object(module.urllib.request, "urlopen", return_value=_Response()):
        try:
            module.call_openai_compatible(
                model="Bonsai-27B-Q1_0.gguf",
                prompt_text="PROMPT",
                timeout_seconds=30,
                temperature=0.1,
                api_base="http://127.0.0.1:8080/v1",
                api_key="local",
                max_tokens=128,
            )
        except RuntimeError as exc:
            assert "empty response" in str(exc).lower()
            assert "finish_reason=length" in str(exc)
        else:
            raise AssertionError("Expected an empty response to fail")


def test_prompt_contract_has_ordered_evidence_sections_and_intent_limit():
    text = PROMPT_PATH.read_text()
    headings = ["Observed facts", "Data quality and provenance", "Mechanical matched-withdrawal signal", "Execution risk of the withdrawn order", "Position relative to the touch", "Timing and order duration", "Price response and economic benefit", "Bilateral activity and inventory context", "Alternative legitimate explanations", "Evidence against the spoofing hypothesis", "Surveillance priority and confidence", "Intent limitation"]
    positions = [text.index(heading) for heading in headings]
    assert positions == sorted(positions)
    assert "mechanical_matched_withdrawal_signal" in text
    assert "economically_consistent_with_spoofing" in text
    assert "compatible_with_legitimate_liquidity_provision" in text
    assert "requires_human_review" in text
    assert "must not use any deterministic equation" in text.lower()


def test_analysis_metadata_contains_content_hashes_without_api_key(tmp_path):
    module = _load_module()
    dossier, prompt, out = tmp_path / "dossier.md", tmp_path / "prompt.md", tmp_path / "review"
    dossier.write_text("# Event dossier: EC000000001-000000001\n")
    prompt.write_text("instructions")
    with patch.object(module, "call_ollama", return_value="# Surveillance review for event EC000000001-000000001\n## Intent limitation\nNo intent finding."):
        module.main(["--dossier", str(dossier), "--prompt", str(prompt), "--output-dir", str(out), "--api-key", "secret-value"])
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["prompt_sha256"]
    assert metadata["dossier_sha256"]
    assert "api_key" not in metadata


def test_batch_uses_unique_cluster_directories_and_atomic_cluster_manifest(tmp_path):
    batch_path = SCRIPT_PATH.with_name("batch_analyze_spoofing_events_with_llm.py")
    spec = importlib.util.spec_from_file_location("batch_analyze_spoofing_events_with_llm", batch_path)
    assert spec and spec.loader
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    events = pl.DataFrame([{"execution_cluster_id": "EC2", "review_event_id": "EC2"}, {"execution_cluster_id": "EC1", "review_event_id": "EC1"}, {"execution_cluster_id": "EC2", "review_event_id": "EC2"}])
    assert batch.unique_review_ids(events) == ["EC1", "EC2"]
    path = tmp_path / "manifest.json"
    batch.write_manifest_atomic(path, {"status": "complete", "api_key": "must-not-persist"})
    assert json.loads(path.read_text()) == {"status": "complete"}
    assert not list(tmp_path.glob("*.tmp"))


def test_batch_uses_analyzer_hash_for_exact_composed_prompt(tmp_path):
    batch_path = SCRIPT_PATH.with_name("batch_analyze_spoofing_events_with_llm.py")
    spec = importlib.util.spec_from_file_location("batch_analyze_spoofing_events_with_llm_hash", batch_path)
    assert spec and spec.loader
    batch = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(batch)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"prompt_sha256": "composed-prompt-hash", "dossier_sha256": "dossier-hash"})
    )

    assert batch._analysis_hashes(tmp_path) == {
        "prompt_sha256": "composed-prompt-hash",
        "dossier_sha256": "dossier-hash",
    }
