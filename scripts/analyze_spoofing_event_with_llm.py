#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a spoofing event dossier with a local LLM.")
    parser.add_argument("--dossier", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=Path("prompts/spoofing_surveillance_analyst.md"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gemma4-hermes:latest")
    parser.add_argument("--backend", choices=("ollama", "openai-compatible"), default="ollama")
    parser.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args(argv)


def compose_prompt(instruction_text: str, dossier_text: str) -> str:
    return (
        instruction_text.strip()
        + "\n\n---\n\n"
        + "Now analyze the event dossier below. Use only this dossier as evidence.\n\n"
        + dossier_text.strip()
        + "\n"
    )


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _dossier_identity(dossier_text: str) -> tuple[str | None, str | None]:
    """Read optional cluster ID/schema from dossier JSON-compatible markdown."""
    cluster = re.search(r"^- execution_cluster_id:\s*(\S+)", dossier_text, flags=re.MULTILINE)
    schema = re.search(r"^- dossier_schema_version:\s*(\S+)", dossier_text, flags=re.MULTILINE)
    return (cluster.group(1) if cluster else None, schema.group(1) if schema else None)


def call_ollama(*, model: str, prompt_text: str, timeout_seconds: int, temperature: float = 0.1) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt_text,
            "stream": False,
            "think": False,
            "options": {"temperature": temperature, "num_predict": 1536},
        }
    ).encode()
    request = urllib.request.Request(
        "http://127.0.0.1:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
            body = json.loads(response.read().decode())
        response_text = str(body.get("response", ""))
        if response_text.strip():
            return clean_llm_response(response_text)
    except Exception:
        # Fall back to the CLI when the Ollama HTTP endpoint is unavailable.
        pass
    result = subprocess.run(
        ["ollama", "run", model],
        input="/set parameter num_predict 2048\n" + prompt_text,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ollama failed with code {result.returncode}: {result.stderr}")
    return clean_llm_response(result.stdout if result.stdout.strip() else result.stderr)


def call_openai_compatible(
    *,
    model: str,
    prompt_text: str,
    timeout_seconds: int,
    temperature: float,
    api_base: str,
    api_key: str,
    max_tokens: int,
) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt_text}],
            "stream": False,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
    ).encode()
    request = urllib.request.Request(
        f"{api_base.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = json.loads(response.read().decode())
    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI-compatible endpoint returned no choices")
    choice = choices[0]
    response_text = str((choice.get("message") or {}).get("content") or "")
    if not response_text.strip():
        raise RuntimeError(
            "OpenAI-compatible endpoint returned an empty response "
            f"(finish_reason={choice.get('finish_reason')})"
        )
    return clean_llm_response(response_text)


def clean_llm_response(text: str) -> str:
    """Remove terminal control sequences and keep the requested final markdown review."""
    without_ansi = re.sub(r"\x1b\[[0-9;?]*[A-Za-z]", "", text)
    marker = "# Surveillance review for event"
    marker_index = without_ansi.find(marker)
    if marker_index >= 0:
        without_ansi = without_ansi[marker_index:]
    return without_ansi.strip() + "\n"


def write_analysis_artifacts(
    *,
    output_dir: Path,
    prompt_text: str,
    response_text: str,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.md").write_text(prompt_text)
    (output_dir / "response.md").write_text(response_text)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    instruction_text = args.prompt.read_text()
    dossier_text = args.dossier.read_text()
    prompt_text = compose_prompt(instruction_text, dossier_text)
    if args.backend == "ollama":
        response_text = call_ollama(
            model=args.model,
            prompt_text=prompt_text,
            timeout_seconds=args.timeout_seconds,
            temperature=args.temperature,
        )
    else:
        response_text = call_openai_compatible(
            model=args.model,
            prompt_text=prompt_text,
            timeout_seconds=args.timeout_seconds,
            temperature=args.temperature,
            api_base=args.api_base,
            api_key=args.api_key,
            max_tokens=args.max_tokens,
        )
    cluster_id, dossier_schema_version = _dossier_identity(dossier_text)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": args.backend,
        "provider": args.backend,
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens if args.backend == "openai-compatible" else None,
        "api_base": args.api_base if args.backend == "openai-compatible" else None,
        "dossier": str(args.dossier),
        "prompt_file": str(args.prompt),
        "timeout_seconds": args.timeout_seconds,
        "execution_cluster_id": cluster_id,
        "dossier_schema_version": dossier_schema_version,
        "prompt_sha256": sha256_text(prompt_text),
        "dossier_sha256": sha256_text(dossier_text),
    }
    write_analysis_artifacts(
        output_dir=args.output_dir,
        prompt_text=prompt_text,
        response_text=response_text,
        metadata=metadata,
    )
    print(args.output_dir / "response.md")


if __name__ == "__main__":
    main()
