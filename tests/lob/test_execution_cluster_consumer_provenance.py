from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_script(name: str):
    path = REPO_ROOT / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _cluster_members() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {"execution_cluster_id": "EC000000010-000000012", "child_sort_index": 10},
            {"execution_cluster_id": "EC000000010-000000012", "child_sort_index": 12},
        ]
    )


def test_dashboard_uses_member_relation_for_noncontiguous_child_fills():
    dashboard = _load_script("build_spoofing_event_review_dashboard")
    metrics = pl.DataFrame(
        [
            {
                "execution_cluster_id": "EC000000010-000000012",
                "cluster_first_sort_index": 10,
                "cluster_last_sort_index": 12,
                "child_fill_count": 2,
                "withdrawal_profile_scale_event": 1.0,
                "has_matched_deceptive_cancel_window": True,
                "execution_anchor_mode": "passive",
            }
        ]
    )

    reviews = dashboard._prepare_review_events(
        metrics,
        None,
        cluster_members=_cluster_members(),
    )

    assert reviews[0]["child_fill_sort_indexes"] == {10, 12}


def test_dossier_timeline_uses_member_relation_for_noncontiguous_child_fills():
    dossier = _load_script("build_spoofing_event_dossier")
    event = {
        "execution_cluster_id": "EC000000010-000000012",
        "cluster_first_sort_index": 10,
        "cluster_last_sort_index": 12,
    }
    event_log = pl.DataFrame(
        [
            {"sort_index": 10, "event_class": "fill"},
            {"sort_index": 11, "event_class": "new_order"},
            {"sort_index": 12, "event_class": "fill"},
        ]
    )

    timeline = dossier.build_focal_timeline(
        event,
        event_log,
        child_members=_cluster_members(),
    )

    assert timeline["sort_index"].to_list() == [10, 12]
    assert timeline["timeline_role"].to_list() == [
        "selected_passive_execution",
        "selected_passive_execution",
    ]
