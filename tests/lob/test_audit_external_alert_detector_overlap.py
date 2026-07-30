from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import polars as pl

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "audit_external_alert_detector_overlap.py"
SPEC = importlib.util.spec_from_file_location("audit_external_alert_detector_overlap", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = module
SPEC.loader.exec_module(module)


def test_parse_external_alerts_recovers_all_dataset_scopes():
    text = """
    RISANAMENTO: giornata del 01/06/2024 relativo al committente 123
    AZIONI  FERRARI .
    111_222
    time interval: [090000000]-[090001000]
    333_444
    time interval: [100000000]-[100001000]
    AZIONI NEXI: committente 456 [23/APR/2024] [110000000]-[110001000]
    """

    alerts = module.parse_external_alerts(text)

    assert len(alerts["RISANAMENTO"]) == 1
    assert alerts["RISANAMENTO"][0].date_level is True
    assert len(alerts["FERRARI"]) == 2
    assert alerts["FERRARI"][1].external_actor_id == "333_444"
    assert len(alerts["NEXI"]) == 1
    assert alerts["NEXI"][0].start == datetime(2024, 4, 23, 11, 0)


def test_compact_string_tradetime_takes_precedence_over_bookouttime():
    raw = pl.DataFrame(
        {
            "TRADETIME": ["20240101 09:00:00.000"],
            "BOOKOUTTIME": ["20240101 09:00:09.000"],
            "BOOKIN": [None],
            "SEQUENCETIME": [None],
            "NMSC_ORIGINALCLIENTIDSHORTCODE": ["C1"],
            "FIRMID": ["F1"],
            "ORDEREVENTTYPE (*)": [3],
            "PASSIVEORDER": ["Y"],
            "AGGRESSIVEORDER": ["N"],
            "LASTSHARES": [1],
            "LASTTRADEDPX": [100.0],
        }
    )

    canonical = module._with_canonical_fields(raw)

    assert canonical.get_column("_event_ts").to_list() == [datetime(2024, 1, 1, 9, 0)]


def test_actor_aliases_are_keyed_per_report():
    first = module._alias("123", b"first-report-key")
    second = module._alias("123", b"second-report-key")

    assert first != second
    assert "123" not in first
    assert "123" not in second


def test_canonical_actor_identity_prefers_client_and_falls_back_to_firm():
    raw = pl.DataFrame(
        {
            "TRADETIME": [datetime(2024, 1, 1, 9, 0), None, None],
            "BOOKOUTTIME": [None, "20240101 09:00:01.000", None],
            "BOOKIN": [None, None, "20240101 09:00:02.000"],
            "SEQUENCETIME": [None, None, None],
            "NMSC_ORIGINALCLIENTIDSHORTCODE": ["C1", None, "null"],
            "FIRMID": ["F1", "F2", "F3"],
            "ORDEREVENTTYPE (*)": [1, 3, 3],
            "PASSIVEORDER": [None, "Y", "N"],
            "AGGRESSIVEORDER": [None, "N", "Y"],
            "LASTSHARES": [None, 1, 2],
            "LASTTRADEDPX": [None, 100.0, 101.0],
        }
    )

    canonical = module._with_canonical_fields(raw)

    assert canonical.get_column("_actor_key").to_list() == [
        "client_original:C1",
        "firm:F2",
        "firm:F3",
    ]
    assert canonical.get_column("_event_ts").to_list() == [
        datetime(2024, 1, 1, 9, 0),
        datetime(2024, 1, 1, 9, 0, 1),
        datetime(2024, 1, 1, 9, 0, 2),
    ]
    resolution = module._resolve_external_identity(canonical, "F2")
    assert resolution.identity_namespace == "firm"
    assert resolution.raw_rows == 1
    assert resolution.detector_actor_key_count == 1


def test_canonical_actor_identity_normalizes_integral_float_storage():
    raw = pl.DataFrame(
        {
            "TRADETIME": [datetime(2024, 1, 1, 9, 0)],
            "BOOKOUTTIME": [None],
            "BOOKIN": [None],
            "SEQUENCETIME": [None],
            "NMSC_ORIGINALCLIENTIDSHORTCODE": pl.Series([123.0], dtype=pl.Float64),
            "FIRMID": pl.Series([456.0], dtype=pl.Float64),
            "ORDEREVENTTYPE (*)": [3],
            "PASSIVEORDER": ["Y"],
            "AGGRESSIVEORDER": ["N"],
            "LASTSHARES": [1],
            "LASTTRADEDPX": [100.0],
        }
    )

    canonical = module._with_canonical_fields(raw)

    assert canonical.get_column("_actor_key").to_list() == ["client_original:123"]


def test_audit_separates_subject_all_actor_union_and_date_only_recall(tmp_path):
    t0 = datetime(2024, 6, 13, 10, 0)
    raw = pl.DataFrame(
        {
            "TRADETIME": [t0, t0.replace(second=1), t0.replace(second=1), None],
            "BOOKOUTTIME": [None, None, None, "20240613 10:00:02.000"],
            "BOOKIN": [None, None, None, None],
            "SEQUENCETIME": [None, None, None, None],
            "NMSC_ORIGINALCLIENTIDSHORTCODE": ["101", "202", "303", "101"],
            "FIRMID": ["F1", "F2", "F1", "F1"],
            "ORDEREVENTTYPE (*)": [3, 3, 3, 4],
            "PASSIVEORDER": ["Y", "N", "N", "N"],
            "AGGRESSIVEORDER": ["N", "Y", "Y", "N"],
            "LASTSHARES": [10, 5, 7, None],
            "LASTTRADEDPX": [100.0, 100.1, 100.2, None],
        }
    )
    raw_path = tmp_path / "raw.parquet"
    raw.write_parquet(raw_path)
    metrics_path = tmp_path / "metrics"
    metrics_path.mkdir()
    pl.DataFrame(
        {
            "actor_key": [
                "client_original:101",
                "client_original:202",
                "client_original:303",
            ],
            "execution_cluster_id": ["c1", "c2", "c3"],
            "execution_anchor_mode": ["passive", "aggressive", "aggressive"],
            "child_event_ts": [t0, t0.replace(second=1), t0.replace(second=1)],
            "child_event_client_original_id": ["101", "202", "303"],
            "child_event_firm_id": ["F1", "F2", "F1"],
        }
    ).write_parquet(metrics_path / "execution_cluster_members.parquet")
    pl.DataFrame(
        {
            "actor_key": [
                "client_original:101",
                "client_original:202",
                "client_original:303",
            ],
            "execution_cluster_id": ["c1", "c2", "c3"],
            "execution_anchor_mode": ["passive", "aggressive", "aggressive"],
            "event_client_original_id": ["101", "202", "303"],
            "event_firm_id": ["F1", "F2", "F1"],
            "has_matched_deceptive_cancel_window": [True, True, True],
            "spoofing_compatible_sequence": [False, True, True],
        }
    ).write_parquet(metrics_path / "execution_metrics.parquet")
    pl.DataFrame(
        {
            "partition_id": ["p1"],
            "sort_index": [3],
            "actor_key": ["client_original:101"],
            "execution_anchor_mode": ["passive"],
            "event_ts": [t0.replace(second=2)],
            "client_original_id": ["101"],
            "firm_id": ["F1"],
            "reject_reason": ["not_fill"],
        }
    ).write_parquet(metrics_path / "rejected_executions.parquet")
    secret_path = "/secret/local/raw.parquet"
    (metrics_path / "metadata.json").write_text(
        json.dumps(
            {
                "input": secret_path,
                "command": ["compute", "--input", secret_path],
                "paths": {"execution_metrics": "/secret/local/out.parquet"},
                "output_schema_version": "actor_execution_anchor_v1",
                "actor_identity_mode": "client_then_firm",
                "execution_anchor_modes": ["passive", "aggressive"],
                "state_client_mode": "execution-actors",
                "compact_state": True,
                "input_hashes": {
                    "raw_events_sha256": "a" * 64,
                    "untrusted_path": secret_path,
                },
                "artifact_hashes": {"execution_metrics": "b" * 64},
                "row_counts": {"execution_metrics": 3},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths = module.DatasetPaths(raw=raw_path, metrics=metrics_path)
    timed_alerts = [
        module.ExternalAlert("TEST", "F1", t0.replace(second=0), t0.replace(second=1)),
        module.ExternalAlert("TEST", "F1", t0, t0.replace(second=3)),
    ]

    timed = module._audit_dataset("TEST", timed_alerts, paths)

    assert timed["source_timed_periods"] == 2
    assert timed["union_timed_periods"] == 1
    assert timed["source_date_only_periods"] == 0
    assert timed["source_period_results"][0]["identity_namespace"] == "firm"
    assert timed["source_period_results"][0]["identity_granularity_aligned"] is False
    assert timed["source_period_results"][0]["detector_actor_key_count_in_period"] == 2
    assert timed["source_period_results"][0]["recovered_execution_clusters"] == 2
    assert timed["source_period_results"][0]["subject_scope_detector_outcome"] == "strict_detection"
    assert timed["source_period_results"][0]["exact_actor_detector_outcome"] == (
        "identity_granularity_not_aligned_exact_actor_recall_not_identifiable"
    )
    assert timed["source_period_results"][0]["all_actor_recovered_execution_clusters"] == 3
    assert timed["source_period_results"][1]["rejected_execution_rows"] == 1
    assert timed["union_timed_results"][0]["source_windows_merged"] == 2
    assert timed["union_timed_results"][0]["recovered_execution_clusters"] == 2
    assert timed["union_periods_with_strict_subject_scope_detection"] == 1
    assert timed["identity_aligned_union_periods_with_strict_detection"] == 0
    assert timed["identity_unaligned_union_periods_with_strict_subject_scope_detection"] == 1
    assert timed["metrics_provenance"]["input_hashes"] == {"raw_events_sha256": "a" * 64}
    assert timed["metrics_provenance"]["artifact_hashes"] == {
        "execution_metrics": "b" * 64
    }
    assert timed["metrics_provenance"]["row_counts"] == {"execution_metrics": 3}
    assert "metadata_sha256" in timed["metrics_provenance"]
    assert secret_path not in json.dumps(timed)

    date_only = module._audit_dataset(
        "TEST",
        [module.ExternalAlert("TEST", "F1", t0.replace(hour=0), t0.replace(hour=0), date_level=True)],
        paths,
    )

    row = date_only["source_period_results"][0]
    assert row["event_recall_identifiable"] is False
    assert row["subject_scope_detector_outcome"] == "date_only_event_recall_not_identifiable"
    assert row["exact_actor_detector_outcome"] == "date_only_event_recall_not_identifiable"
    assert row["recovered_execution_clusters"] is None
    assert row["date_level_recovered_execution_clusters"] == 2
    assert row["recovered_execution_clusters_by_anchor"] is None
    assert row["date_level_recovered_execution_clusters_by_anchor"] == {
        "aggressive": 1,
        "passive": 1,
    }
    assert row["rejected_execution_rows"] is None
    assert row["date_level_rejected_execution_rows"] == 1
    assert row["detector_actor_key_count_in_period"] is None
    assert row["date_level_detector_actor_key_count_in_period"] == 2
    assert date_only["source_timed_periods"] == 0
    assert date_only["union_timed_periods"] == 0
    assert date_only["union_periods_with_recovered_execution"] == 0
