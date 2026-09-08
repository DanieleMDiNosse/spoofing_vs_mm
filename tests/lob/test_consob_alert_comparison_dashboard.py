from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "build_consob_alert_comparison_dashboard.py"
)
_spec = importlib.util.spec_from_file_location(
    "build_consob_alert_comparison_dashboard", SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
_dashboard = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dashboard)


def _window(
    instrument: str,
    *,
    matched_withdrawal: int,
    strict: int,
    identity_aligned: bool,
    anchor: str,
) -> dict[str, object]:
    return {
        "period_id": f"{instrument}-001",
        "union_period_id": f"{instrument}-union-001",
        "start": "2024-04-23T14:40:59",
        "end": "2024-04-23T14:41:14",
        "duration_seconds": 15.0,
        "source_windows_merged": 1,
        "scope": "interval",
        "identity_namespace": "client_original" if identity_aligned else "firm",
        "identity_granularity_aligned": identity_aligned,
        "event_recall_identifiable": True,
        "actor_alias": "actor_secret_should_not_render",
        "raw_actor_rows_in_period": 1,
        "recovered_child_fill_rows": 4,
        "recovered_execution_clusters": 1,
        "recovered_execution_clusters_by_anchor": {anchor: 1},
        "clusters_with_matched_withdrawal": matched_withdrawal,
        "clusters_with_strict_detection": strict,
        "detector_events": [
            {
                "event_alias": "detector_event_" + (
                    "a" if instrument == "FERRARI" else "b"
                ) * 12,
                "cluster_start": "2024-04-23T14:41:03.125000",
                "cluster_end": "2024-04-23T14:41:03.250000",
                "execution_anchor_mode": anchor,
                "execution_side": "bid",
                "execution_quantity": 1250.0,
                "execution_vwap": 7.42,
                "has_matched_withdrawal": bool(matched_withdrawal),
                "gate_rapid_matched_withdrawal": bool(matched_withdrawal),
                "gate_small_fill_relative_to_withdrawal": bool(matched_withdrawal),
                "gate_favorable_pre_fill_move": bool(strict),
                "gate_cancel_anchored_reversion": bool(strict),
                "strict_detection": bool(strict),
            }
        ],
        "subject_scope_detector_outcome": (
            "strict_detection" if strict else "no_matched_withdrawal"
        ),
        "exact_actor_detector_outcome": (
            "strict_detection"
            if strict and identity_aligned
            else "identity_granularity_not_aligned_exact_actor_recall_not_identifiable"
            if strict
            else "no_matched_withdrawal"
        ),
    }


def _report() -> dict[str, Any]:
    ferrari_window = _window(
        "FERRARI",
        matched_withdrawal=1,
        strict=1,
        identity_aligned=False,
        anchor="passive",
    )
    nexi_window = _window(
        "NEXI",
        matched_withdrawal=0,
        strict=0,
        identity_aligned=True,
        anchor="aggressive",
    )
    return {
        "generated_at_utc": "2026-07-31T12:00:00+00:00",
        "verification_tier": "artifact_backed",
        "recall_unit": "merged_subject_instrument_intraday_interval",
        "external_source": {
            "source_kind": "primary_pdf",
            "sha256": "a" * 64,
            "local_path_disclosed": False,
        },
        "identifier_policy": "Report-local pseudonyms; no source path serialized.",
        "limitations": [
            "External alert labels are positive seeds, not independently adjudicated intent.",
            "RISANAMENTO is date-only and excluded from event-level recall.",
        ],
        "overall": {
            "source_external_periods": 3,
            "source_timed_periods": 2,
            "source_date_only_periods": 1,
            "union_timed_periods": 2,
            "union_periods_with_raw_subject_event": 2,
            "union_periods_with_recovered_execution": 2,
            "union_periods_with_matched_withdrawal": 1,
            "union_periods_with_strict_subject_scope_detection": 1,
            "identity_aligned_union_timed_periods": 1,
            "identity_aligned_union_periods_with_strict_detection": 0,
            "identity_unaligned_union_timed_periods": 1,
            "identity_unaligned_union_periods_with_strict_subject_scope_detection": 1,
        },
        "datasets": {
            "FERRARI": {
                "source_external_periods": 1,
                "source_timed_periods": 1,
                "source_date_only_periods": 0,
                "union_timed_periods": 1,
                "union_periods_with_raw_subject_event": 1,
                "union_periods_with_recovered_execution": 1,
                "union_periods_with_matched_withdrawal": 1,
                "union_periods_with_strict_subject_scope_detection": 1,
                "identity_aligned_union_timed_periods": 0,
                "identity_aligned_union_periods_with_strict_detection": 0,
                "identity_unaligned_union_timed_periods": 1,
                "identity_unaligned_union_periods_with_strict_subject_scope_detection": 1,
                "union_timed_results": [ferrari_window],
                "source_period_results": [ferrari_window],
                "metrics_provenance": {
                    "actor_identity_mode": "client_then_firm",
                    "execution_anchor_modes": ["passive", "aggressive"],
                    "output_schema_version": "actor_execution_anchor_v2",
                    "dangerous_path": "/home/research/raw/private.parquet",
                    "dangerous_command": "--input /secret/raw.parquet",
                },
            },
            "NEXI": {
                "source_external_periods": 1,
                "source_timed_periods": 1,
                "source_date_only_periods": 0,
                "union_timed_periods": 1,
                "union_periods_with_raw_subject_event": 1,
                "union_periods_with_recovered_execution": 1,
                "union_periods_with_matched_withdrawal": 0,
                "union_periods_with_strict_subject_scope_detection": 0,
                "identity_aligned_union_timed_periods": 1,
                "identity_aligned_union_periods_with_strict_detection": 0,
                "identity_unaligned_union_timed_periods": 0,
                "identity_unaligned_union_periods_with_strict_subject_scope_detection": 0,
                "union_timed_results": [nexi_window],
                "source_period_results": [nexi_window],
                "metrics_provenance": {
                    "actor_identity_mode": "client_then_firm",
                    "execution_anchor_modes": ["passive", "aggressive"],
                    "output_schema_version": "actor_execution_anchor_v2",
                },
            },
            "RISANAMENTO": {
                "source_external_periods": 1,
                "source_timed_periods": 0,
                "source_date_only_periods": 1,
                "union_timed_periods": 0,
                "union_periods_with_raw_subject_event": 0,
                "union_periods_with_recovered_execution": 0,
                "union_periods_with_matched_withdrawal": 0,
                "union_periods_with_strict_subject_scope_detection": 0,
                "identity_aligned_union_timed_periods": 0,
                "identity_aligned_union_periods_with_strict_detection": 0,
                "identity_unaligned_union_timed_periods": 0,
                "identity_unaligned_union_periods_with_strict_subject_scope_detection": 0,
                "union_timed_results": [],
                "source_period_results": [
                    {
                        "period_id": "RISANAMENTO-001",
                        "scope": "date",
                        "identity_namespace": "client_original",
                        "start": "2024-09-10T00:00:00",
                        "end": None,
                        "event_recall_identifiable": False,
                        "actor_alias": "actor_date_only_secret",
                        "date_level_recovered_execution_clusters": 59,
                        "date_level_recovered_execution_clusters_by_anchor": {
                            "aggressive": 32,
                            "passive": 27,
                        },
                        "date_level_clusters_with_matched_withdrawal": 1,
                        "date_level_clusters_with_strict_detection": 0,
                    }
                ],
                "metrics_provenance": {
                    "actor_identity_mode": "client_then_firm",
                    "execution_anchor_modes": ["passive", "aggressive"],
                    "output_schema_version": "actor_execution_anchor_v2",
                },
            },
        },
    }


def test_build_dashboard_renders_comparison_filters_and_privacy_boundary():
    html = _dashboard.build_dashboard(
        _report(), generated_at_utc="2026-07-31T12:30:00+00:00"
    )

    assert "Confronto alert CONSOB e detector" in html
    assert 'id="instrument-filter"' in html
    assert 'id="outcome-filter"' in html
    assert 'data-instrument="FERRARI"' in html
    assert 'data-instrument="NEXI"' in html
    assert "Finestre CONSOB" in html
    assert "Alert originali ed eventi detector" in html
    assert 'id="source-instrument-filter"' in html
    assert 'id="source-subject-filter"' in html
    assert 'id="source-alert-table"' in html
    assert "detector_event_aaaaaaaaaaaa" in html
    assert "14:41:03.125" in html
    assert "1.250" in html
    assert "7,42" in html
    assert "Soggetto 1" in html
    assert "Ritiro attribuito" in html
    assert "Sequenza stretta" in html
    assert "visible === 1 ? ' alert visualizzato' : ' alert visualizzati'" in html
    assert "Evidenza giornaliera non localizzabile" in html
    assert "59" in html
    assert "32 aggressivi" in html
    assert "27 passivi" in html
    assert "Il 100% delle finestre intraday contiene almeno un cluster di esecuzione" in html
    assert "0/1" in html  # strict exact-actor recall on identity-aligned windows
    assert "actor_secret_should_not_render" not in html
    assert "actor_date_only_secret" not in html
    assert "/home/research/raw/private.parquet" not in html
    assert "--input /secret/raw.parquet" not in html


def test_validate_report_rejects_detector_event_count_mismatch():
    report = _report()
    report["datasets"]["FERRARI"]["source_period_results"][0]["detector_events"] = []

    with pytest.raises(ValueError, match="detector_events.*recovered_execution_clusters"):
        _dashboard.build_dashboard(report)


def test_validate_report_rejects_raw_detector_identifiers_in_event_details():
    report = _report()
    report["datasets"]["FERRARI"]["source_period_results"][0]["detector_events"][0][
        "actor_key"
    ] = "client_original:raw-secret"

    with pytest.raises(ValueError, match="unexpected fields"):
        _dashboard.build_dashboard(report)


def test_validate_report_rejects_unallowlisted_identity_namespace() -> None:
    report = _report()
    report["datasets"]["FERRARI"]["source_period_results"][0][
        "identity_namespace"
    ] = "/home/private/identity.txt"

    with pytest.raises(ValueError, match="identity_namespace"):
        _dashboard.build_dashboard(report)


def test_validate_report_rejects_union_count_mismatch():
    report = _report()
    report["overall"]["union_timed_periods"] = 3

    with pytest.raises(ValueError, match="union_timed_periods"):
        _dashboard.build_dashboard(report)


def test_validate_report_rejects_fractional_window_counts():
    report = _report()
    report["datasets"]["NEXI"]["union_timed_results"][0][
        "recovered_execution_clusters"
    ] = 0.5

    with pytest.raises(ValueError, match="non-negative integer"):
        _dashboard.build_dashboard(report)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("source_total", "source_external_periods"),
        ("raw_subject", "raw-subject"),
        ("identity_windows", "identity aligned/unaligned"),
        ("identity_strict", "strict identity aligned/unaligned"),
    ],
)
def test_validate_report_rejects_impossible_denominator_decompositions(
    case: str, message: str
):
    report = _report()
    if case == "source_total":
        report["overall"]["source_external_periods"] = 4
        report["datasets"]["RISANAMENTO"]["source_external_periods"] = 2
    elif case == "raw_subject":
        report["overall"]["union_periods_with_raw_subject_event"] = 0
        report["datasets"]["FERRARI"]["union_periods_with_raw_subject_event"] = 0
        report["datasets"]["NEXI"]["union_periods_with_raw_subject_event"] = 0
    elif case == "identity_windows":
        report["overall"]["identity_unaligned_union_timed_periods"] = 0
        report["datasets"]["FERRARI"]["identity_unaligned_union_timed_periods"] = 0
    else:
        report["overall"][
            "identity_unaligned_union_periods_with_strict_subject_scope_detection"
        ] = 0
        report["datasets"]["FERRARI"][
            "identity_unaligned_union_periods_with_strict_subject_scope_detection"
        ] = 0

    with pytest.raises(ValueError, match=message):
        _dashboard.build_dashboard(report)


@pytest.mark.parametrize(
    ("field", "unsafe_value"),
    [
        ("actor_identity_mode", "--token top-secret"),
        ("execution_anchor_modes", ["passive", "/home/private/data"]),
        ("output_schema_version", "actor_alias_abc123"),
    ],
)
def test_validate_report_rejects_unsafe_rendered_provenance(
    field: str, unsafe_value: object
):
    report = _report()
    report["datasets"]["FERRARI"]["metrics_provenance"][field] = unsafe_value

    with pytest.raises(ValueError, match=field):
        _dashboard.build_dashboard(report)


def test_validate_report_rejects_unsafe_source_kind():
    report = _report()
    report["external_source"]["source_kind"] = "/home/private/source.pdf"

    with pytest.raises(ValueError, match="source_kind"):
        _dashboard.build_dashboard(report)


def test_validate_report_reconciles_each_dataset_with_its_union_rows():
    report = _report()
    report["datasets"]["FERRARI"]["identity_aligned_union_timed_periods"] = 1
    report["datasets"]["FERRARI"]["identity_unaligned_union_timed_periods"] = 0
    report["datasets"]["FERRARI"][
        "identity_aligned_union_periods_with_strict_detection"
    ] = 1
    report["datasets"]["FERRARI"][
        "identity_unaligned_union_periods_with_strict_subject_scope_detection"
    ] = 0
    report["datasets"]["NEXI"]["identity_aligned_union_timed_periods"] = 0
    report["datasets"]["NEXI"]["identity_unaligned_union_timed_periods"] = 1
    report["overall"]["identity_aligned_union_periods_with_strict_detection"] = 1
    report["overall"][
        "identity_unaligned_union_periods_with_strict_subject_scope_detection"
    ] = 0

    with pytest.raises(ValueError, match="datasets.FERRARI.*does not match union rows"):
        _dashboard.build_dashboard(report)


def test_build_dashboard_rejects_invalid_generated_timestamp():
    with pytest.raises(ValueError, match="generated_at_utc"):
        _dashboard.build_dashboard(_report(), generated_at_utc="not-an-iso-timestamp")


def test_build_dashboard_normalizes_generated_timestamp_to_utc():
    html = _dashboard.build_dashboard(
        _report(), generated_at_utc="2026-07-31T14:30:00+02:00"
    )

    assert "2026-07-31T12:30:00+00:00" in html
    assert "2026-07-31T14:30:00+02:00" not in html


def test_cli_writes_self_contained_html_and_hashed_metadata(tmp_path: Path):
    audit_path = tmp_path / "audit.json"
    output_path = tmp_path / "dashboard" / "consob_alert_comparison_dashboard.html"
    metadata_path = output_path.with_suffix(".metadata.json")
    audit_bytes = json.dumps(_report(), indent=2, sort_keys=True).encode()
    audit_path.write_bytes(audit_bytes)

    exit_code = _dashboard.main(
        [
            "--audit-json",
            str(audit_path),
            "--output-html",
            str(output_path),
            "--generated-at-utc",
            "2026-07-31T12:30:00+00:00",
        ]
    )

    assert exit_code == 0
    html = output_path.read_text()
    metadata = json.loads(metadata_path.read_text())
    assert "<style>" in html
    assert "<script>" in html
    assert "https://" not in html
    assert metadata["audit_sha256"] == hashlib.sha256(audit_bytes).hexdigest()
    assert metadata["dashboard_sha256"] == hashlib.sha256(output_path.read_bytes()).hexdigest()
    assert metadata["union_timed_periods"] == 2
    assert metadata["source_date_only_periods"] == 1
    assert metadata["output_html"] == output_path.name
