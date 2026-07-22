#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

_SECRET_KEYS = {"api_key", "token", "authorization", "password", "secret"}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch-generate resumable local LLM reviews for matched spoofing execution clusters.")
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--parameter-grid-root", type=Path, default=None)
    parser.add_argument("--prompt", type=Path, default=Path("prompts/spoofing_surveillance_analyst.md"))
    parser.add_argument("--model", default="gemma4-hermes:latest")
    parser.add_argument("--backend", choices=("ollama", "openai-compatible"), default="ollama")
    parser.add_argument("--api-base", default="http://127.0.0.1:8080/v1")
    parser.add_argument("--api-key", default="local")
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout-seconds", type=int, default=550)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def unique_review_ids(events: pl.DataFrame) -> list[str]:
    """Return sorted unique cluster identifiers, with legacy fallback only when needed."""
    column = "execution_cluster_id" if "execution_cluster_id" in events.columns else "review_event_id"
    if column not in events.columns:
        raise ValueError("execution metrics has neither execution_cluster_id nor review_event_id")
    ids = sorted({str(value) for value in events.get_column(column).drop_nulls().to_list()})
    if "execution_cluster_id" in events.columns and any(item.startswith("S") for item in ids):
        raise ValueError("cluster-capable execution metrics contains stale message-level review IDs")
    return ids


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items() if key.lower() not in _SECRET_KEYS}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    return value


def write_manifest_atomic(path: Path, manifest: dict[str, Any]) -> None:
    """Persist an atomic, credential-free checkpoint for one cluster."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temp.write_text(json.dumps(_sanitize(manifest), indent=2, sort_keys=True, default=str))
    temp.replace(path)


def _read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        value = json.loads(path.read_text())
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _analysis_hashes(output_dir: Path) -> dict[str, str]:
    """Return hashes emitted for the exact dossier and composed prompt analyzed."""
    metadata = _read_manifest(output_dir / "metadata.json")
    hashes: dict[str, str] = {}
    for key in ("prompt_sha256", "dossier_sha256"):
        value = metadata.get(key)
        if isinstance(value, str) and value:
            hashes[key] = value
    return hashes


def _sanitize_error(exc: BaseException) -> str:
    text = str(exc)
    for marker in ("api_key=", "token=", "Bearer "):
        if marker in text:
            before, _, after = text.partition(marker)
            text = before + marker + "[redacted]" + (" " + after.split(" ", 1)[1] if " " in after else "")
    return text[:1000]


def _run(command: list[str], *, dry_run: bool) -> None:
    printable = ["[redacted]" if index and command[index - 1] == "--api-key" else part for index, part in enumerate(command)]
    if dry_run:
        print(f"DRY RUN: {' '.join(printable)}")
        return
    subprocess.run(command, check=True)


def _matching_events(events: pl.DataFrame) -> pl.DataFrame:
    if "has_matched_deceptive_cancel_window" in events.columns:
        return events.filter(pl.col("has_matched_deceptive_cancel_window"))
    return events


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    events = _matching_events(pl.read_parquet(args.review_dir / "matched_spoofing_events.parquet"))
    cluster_ids = unique_review_ids(events)
    if args.limit is not None:
        cluster_ids = cluster_ids[: args.limit]
    for cluster_id in cluster_ids:
        output_dir = args.review_dir / "llm_reviews" / cluster_id
        manifest_path = output_dir / "manifest.json"
        existing = _read_manifest(manifest_path)
        response_path = output_dir / "response.md"
        if response_path.exists() and existing.get("status") == "complete" and not args.overwrite:
            print(f"SKIP {cluster_id}: completed checkpoint")
            continue
        retry_count = int(existing.get("retry_count") or 0) + (1 if existing else 0)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_manifest = {
            "execution_cluster_id": cluster_id,
            "status": "running",
            "retry_count": retry_count,
            "started_at_utc": now_utc(),
            "provider": args.backend,
            "backend": args.backend,
            "model": args.model,
            "prompt_file": str(args.prompt),
            "prompt_template_sha256": sha256_file(args.prompt),
        }
        write_manifest_atomic(manifest_path, base_manifest)
        dossier_cmd = [sys.executable, "scripts/build_spoofing_event_dossier.py", "--review-dir", str(args.review_dir), "--event-id", cluster_id, "--output-dir", str(output_dir)]
        if args.parameter_grid_root is not None:
            dossier_cmd.extend(["--parameter-grid-root", str(args.parameter_grid_root)])
        analyze_cmd = [sys.executable, "scripts/analyze_spoofing_event_with_llm.py", "--dossier", str(output_dir / "dossier.md"), "--prompt", str(args.prompt), "--output-dir", str(output_dir), "--model", args.model, "--backend", args.backend, "--api-base", args.api_base, "--api-key", args.api_key, "--max-tokens", str(args.max_tokens), "--timeout-seconds", str(args.timeout_seconds)]
        print(f"CLUSTER {cluster_id}")
        try:
            _run(dossier_cmd, dry_run=args.dry_run)
            if not args.dry_run:
                base_manifest["dossier_sha256"] = sha256_file(output_dir / "dossier.md")
                write_manifest_atomic(manifest_path, base_manifest)
            _run(analyze_cmd, dry_run=args.dry_run)
            status = "skipped" if args.dry_run else "complete"
            analysis_hashes = {} if args.dry_run else _analysis_hashes(output_dir)
            write_manifest_atomic(
                manifest_path,
                {
                    **base_manifest,
                    **analysis_hashes,
                    "status": status,
                    "completed_at_utc": now_utc(),
                    "dossier_sha256": analysis_hashes.get(
                        "dossier_sha256", sha256_file(output_dir / "dossier.md")
                    ),
                    "response_sha256": sha256_file(response_path),
                },
            )
        except Exception as exc:
            write_manifest_atomic(manifest_path, {**base_manifest, "status": "failed", "failed_at_utc": now_utc(), "error": _sanitize_error(exc), "dossier_sha256": sha256_file(output_dir / "dossier.md")})
            print(f"FAILED {cluster_id}: {_sanitize_error(exc)}", file=sys.stderr)


if __name__ == "__main__":
    main()
