from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate_spoofing_event_reviews.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("validate_spoofing_event_reviews", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _write_complete_review(root: Path, cluster_id: str) -> None:
    review = root / "llm_reviews" / cluster_id
    review.mkdir(parents=True)
    dossier, prompt = f"# Event dossier: {cluster_id}\n", "prompt\n"
    (review / "dossier.md").write_text(dossier)
    (review / "prompt.md").write_text(prompt)
    (review / "response.md").write_text(f"# Surveillance review for event {cluster_id}\n## Observed facts\ntext\n## Intent limitation\nNo intent finding.\n")
    module = _load_module()
    hashes = {"dossier_sha256": module.sha256_text(dossier), "prompt_sha256": module.sha256_text(prompt)}
    (review / "metadata.json").write_text(json.dumps(hashes))
    (review / "manifest.json").write_text(json.dumps({"status": "complete", **hashes}))


def _write_cancel_assignment(
    root: Path,
    cluster_id: str,
    *,
    cluster_last_sort_index: int = 11,
    cancel_sort_index: int = 12,
) -> None:
    pl.DataFrame(
        [
            {
                "partition_id": "P",
                "candidate_order_id": "O1",
                "execution_cluster_id": cluster_id,
                "cluster_last_sort_index": cluster_last_sort_index,
                "cancel_sort_index": cancel_sort_index,
                "assigned_flag": True,
            }
        ]
    ).write_parquet(root / "execution_cancel_candidates.parquet")


def test_validator_accepts_complete_cluster_first_reviews_and_dashboard_timestamp(tmp_path):
    module = _load_module()
    cluster_id = "EC000000010-000000011"
    pl.DataFrame([{"execution_cluster_id": cluster_id, "has_matched_deceptive_cancel_window": True}]).write_parquet(tmp_path / "execution_metrics.parquet")
    _write_complete_review(tmp_path, cluster_id)
    _write_cancel_assignment(tmp_path, cluster_id)
    (tmp_path / "metadata.json").write_text(json.dumps({"dashboard_refreshed_at_utc": "2026-07-16T12:00:00+00:00"}))
    assert module.validate_review_artifacts(tmp_path, tmp_path / "execution_metrics.parquet") == []


def test_validator_rejects_noncausal_cancel_assignment(tmp_path):
    module = _load_module()
    cluster_id = "EC000000010-000000011"
    pl.DataFrame(
        [{"execution_cluster_id": cluster_id, "has_matched_deceptive_cancel_window": True}]
    ).write_parquet(tmp_path / "execution_metrics.parquet")
    _write_complete_review(tmp_path, cluster_id)
    _write_cancel_assignment(
        tmp_path,
        cluster_id,
        cluster_last_sort_index=11,
        cancel_sort_index=10,
    )
    (tmp_path / "metadata.json").write_text(
        json.dumps({"dashboard_refreshed_at_utc": "2026-07-16T12:00:00+00:00"})
    )

    errors = module.validate_review_artifacts(
        tmp_path,
        tmp_path / "execution_metrics.parquet",
    )

    assert any("noncausal cancellation assignment" in error for error in errors)


def test_validator_reports_stale_message_id_and_incomplete_manifest(tmp_path):
    module = _load_module()
    cluster_id = "EC000000010-000000011"
    pl.DataFrame([{"execution_cluster_id": cluster_id, "has_matched_deceptive_cancel_window": True}]).write_parquet(tmp_path / "execution_metrics.parquet")
    _write_complete_review(tmp_path, cluster_id)
    stale = tmp_path / "llm_reviews" / "S10"
    stale.mkdir()
    (stale / "manifest.json").write_text(json.dumps({"status": "failed", "error": "token=abc"}))
    (tmp_path / "metadata.json").write_text(json.dumps({"dashboard_refreshed_at_utc": None}))
    errors = module.validate_review_artifacts(tmp_path, tmp_path / "execution_metrics.parquet")
    assert any("stale message-level" in error for error in errors)
    assert any("failed" in error for error in errors)
    assert any("dashboard_refreshed_at_utc" in error for error in errors)
