from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_spoofing_negative_control_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_spoofing_negative_control_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_report_writes_markdown(tmp_path):
    module = _load_module()
    events = pl.DataFrame({"review_event_id": ["S1"], "sort_index": [10], "deceptive_side": ["bid"], "MSCI": [0.7]})
    events_path = tmp_path / "events.parquet"
    output = tmp_path / "report.md"
    events.write_parquet(events_path)
    module.build_report(events_path=events_path, output_path=output, shift_events=50)
    text = output.read_text()
    assert "# Spoofing Negative-Control Report" in text
    assert "time_shift" in text
    assert "wrong_side" in text
    assert "descriptor_only" in text
    assert "should score real candidate events higher" not in text


def test_report_cli_requires_exactly_one_source_and_legacy_output() -> None:
    module = _load_module()
    args = module.parse_args(["--bundle-root", "bundle"])
    assert args.bundle_root == Path("bundle")
    with pytest.raises(SystemExit):
        module.parse_args(["--bundle-root", "bundle", "--events", "events.parquet"])
    with pytest.raises(ValueError, match="--output is required"):
        module.main(["--events", "events.parquet"])


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_empirical_bundle(tmp_path: Path) -> Path:
    root = tmp_path / "bundle"
    root.mkdir()
    (root / "validation.json").write_text(json.dumps({"status": "pass", "errors": []}))
    pl.DataFrame([
        {
            "instrument": "TEST", "event_date": "2024-01-02", "actor_count": 1,
            "actor_day_count": 1, "risk_spell_count": 2,
            "time_at_risk_seconds": 10.0, "withdrawal_count": 1,
            "censor_count": 1, "censor_reason": "coverage_end",
        }
    ]).write_csv(root / "coverage_and_balance.csv")
    pl.DataFrame([
        {
            "instrument": "TEST", "contrast_label": "own_only",
            "time_at_risk_seconds": 2.0, "withdrawal_count": 1,
            "withdrawal_intensity": 0.5,
        },
        {
            "instrument": "TEST", "contrast_label": "no_qualifying_fill",
            "time_at_risk_seconds": 8.0, "withdrawal_count": 0,
            "withdrawal_intensity": 0.0,
        },
    ]).write_csv(root / "stratum_statistics.csv")
    pl.DataFrame([
        {"draw_id": 1, "accepted": True, "rejection_reason": None},
        {"draw_id": 2, "accepted": False, "rejection_reason": "no_support"},
    ]).write_parquet(root / "placebo_support.parquet")
    statistics_rows = []
    for draw_id in range(3):
        own_withdrawals = 1 if draw_id == 0 else 0
        statistics_rows.extend([
            {
                "draw_id": draw_id, "row_kind": "state", "contrast_label": "own_only",
                "time_at_risk_seconds": 2.0, "withdrawal_count": own_withdrawals,
                "withdrawal_intensity": own_withdrawals / 2.0, "comparison_label": None,
                "comparison_intensity_difference": None,
            },
            {
                "draw_id": draw_id, "row_kind": "state", "contrast_label": "no_qualifying_fill",
                "time_at_risk_seconds": 8.0, "withdrawal_count": 1 - own_withdrawals,
                "withdrawal_intensity": (1 - own_withdrawals) / 8.0, "comparison_label": None,
                "comparison_intensity_difference": None,
            },
            {
                "draw_id": draw_id, "row_kind": "aggregate_contrast", "contrast_label": "own_only",
                "time_at_risk_seconds": 2.0, "withdrawal_count": own_withdrawals,
                "withdrawal_intensity": own_withdrawals / 2.0,
                "comparison_label": "own_minus_no_qualifying_fill",
                "comparison_intensity_difference": own_withdrawals / 2.0 - (1 - own_withdrawals) / 8.0,
            },
        ])
    pl.DataFrame(statistics_rows).write_parquet(root / "placebo_statistics.parquet")
    pl.DataFrame({
        "risk_spell_id": ["S1", "S2"], "instrument": ["TEST", "TEST"],
        "partition_id": ["P", "P"], "event_date": ["2024-01-02", "2024-01-02"],
        "actor_key": ["client_original:C1", "client_original:C2"],
        "posture_side": ["bid", "ask"], "duration_seconds": [2.0, 8.0],
    }).write_parquet(root / "risk_spells.parquet")
    pl.DataFrame({
        "withdrawal_event_id": ["W1"], "physical_order_key": ["P|B"],
        "visible_qty_removed": [10.0],
    }).write_parquet(root / "withdrawal_events.parquet")
    empty_schemas = {
        "risk_membership.parquet": {
            "risk_membership_id": pl.String, "physical_order_key": pl.String,
            "duration_seconds": pl.Float64,
        },

        "episode_risk_links.parquet": {
            "instrument": pl.String, "episode_id": pl.String,
            "execution_cluster_id": pl.String, "risk_spell_id": pl.String,
            "risk_interval_id": pl.String, "physical_order_key": pl.String,
        },
        "placebo_schedules.parquet": {
            "draw_id": pl.Int64, "schedule_block_id": pl.String, "accepted": pl.Boolean,
            "rejection_reason": pl.String,
        },
    }
    for name, schema in empty_schemas.items():
        pl.DataFrame(schema=schema).write_parquet(root / name)
    pl.DataFrame({
        "risk_interval_id": ["I1", "I2"], "risk_spell_id": ["S1", "S2"],
        "instrument": ["TEST", "TEST"], "partition_id": ["P", "P"],
        "event_date": ["2024-01-02", "2024-01-02"],
        "execution_anchor_mode": [None, None], "duration_seconds": [2.0, 8.0],
    }).with_columns(pl.col("event_date").str.to_date()).write_parquet(root / "risk_intervals.parquet")
    pl.DataFrame({
        "execution_anchor_mode": ["passive"], "exposure_mask": ["own_passive"],
        "duration_seconds": [2.0],
    }).write_parquet(root / "exposure_intervals.parquet")
    pl.DataFrame({
        "instrument": ["TEST"], "partition_id": ["P"], "event_date": ["2024-01-02"],
        "total_event_count": [2], "accepted_event_count": [2],
        "missing_timestamp_event_count": [0], "clock_regression_event_count": [0],
        "quarantined_event_count": [0], "quarantine_episode_count": [0],
        "recovered_quarantine_episode_count": [0],
        "recovered_quarantine_duration_seconds": [0.0],
        "unquantified_quarantine_event_count": [0], "open_quarantine_at_end_flag": [False],
        "first_accepted_ts": ["2024-01-02 09:30:00"], "last_accepted_ts": ["2024-01-02 09:30:10"],
        "observed_accepted_span_seconds": [10.0], "time_at_risk_seconds": [10.0],
        "coverage_loss_fraction": [0.0], "terminal_right_censored_spell_count": [1],
        "partition_boundary_censored_spell_count": [0], "clock_ambiguous_censored_spell_count": [0],
    }).with_columns(
        pl.col("event_date").str.to_date(),
        pl.col("first_accepted_ts").str.to_datetime(),
        pl.col("last_accepted_ts").str.to_datetime(),
    ).write_parquet(root / "clock_coverage_audit.parquet")
    (root / "config.json").write_text("{}")
    (root / "versions.txt").write_text("python=test\n")
    (root / "command.txt").write_text("test command\n")
    (root / "run.log").write_text("exit_status=0\n")

    artifact_names = [
        path.name for path in root.iterdir()
    ]
    manifest = {
        "design_version": "empirical_controls_v1",
        "seed": 7,
        "clock": {"timestamp_column": "CONSUMETIME"},
        "risk_policy": {"state_timing": "pre_event"},
        "units": {
            "risk_spells.parquet": {"duration_seconds": "seconds"},
            "risk_intervals.parquet": {"duration_seconds": "seconds"},
            "risk_membership.parquet": {"duration_seconds": "seconds"},
            "exposure_intervals.parquet": {"duration_seconds": "seconds"},
            "withdrawal_events.parquet": {"visible_qty_removed": "shares"},
            "episode_risk_links.parquet": {
                "episode_id": "dimensionless episode identifier",
                "execution_cluster_id": "dimensionless execution-cluster identifier",
                "risk_spell_id": "dimensionless risk-spell identifier",
                "risk_interval_id": "dimensionless risk-interval identifier",
                "physical_order_key": "dimensionless physical lifecycle identifier",
            },
            "placebo_schedules.parquet": {"draw_id": "count", "timestamps": "UTC-naive event clock", "sort_indices": "canonical event order"},
            "placebo_support.parquet": {"draw_id": "count", "support_boundary_count": "count", "supported_start_count": "count"},
            "placebo_statistics.parquet": {"time_at_risk_seconds": "seconds", "withdrawal_count": "physical cancellations", "withdrawal_intensity": "cancellations per second"},
            "coverage_and_balance.csv": {"time_at_risk_seconds": "seconds", "withdrawal_count": "physical cancellations"},
            "clock_coverage_audit.parquet": {
                "total_event_count": "events", "accepted_event_count": "events",
                "missing_timestamp_event_count": "events", "clock_regression_event_count": "events",
                "quarantined_event_count": "events", "quarantine_episode_count": "episodes",
                "recovered_quarantine_episode_count": "episodes",
                "recovered_quarantine_duration_seconds": "seconds",
                "unquantified_quarantine_event_count": "events",
                "observed_accepted_span_seconds": "seconds", "time_at_risk_seconds": "seconds",
                "coverage_loss_fraction": "quantified lower-bound fraction",
                "terminal_right_censored_spell_count": "spells",
                "partition_boundary_censored_spell_count": "spells",
                "clock_ambiguous_censored_spell_count": "spells",
            },
            "stratum_statistics.csv": {"time_at_risk_seconds": "seconds", "withdrawal_count": "physical cancellations", "withdrawal_intensity": "cancellations per second"},
        },
        "exclusion_reasons": {},
        "code_snapshot": {
            "git_head": "test",
            "files": {"config": {"path": str(root / "config.json"), "sha256": _sha256(root / "config.json")}},
        },
        "sources": {
            "TEST": {
                name: {"path": str(root / "config.json"), "sha256": _sha256(root / "config.json")}
                for name in (
                    "raw_events", "baseline_metadata", "execution_metrics", "quote_panel",
                    "empirical_depth_kernel", "baseline_config",
                )
            }
        },
        "artifacts": {
            name: {"path": name, "sha256": _sha256(root / name)}
            for name in artifact_names
        },
        "summary": {
            "risk_spell_count": 2,
            "time_at_risk_seconds": 10.0,
            "withdrawal_count": 1,
            "accepted_placebo_blocks": 1,
            "rejected_placebo_blocks": 1,
            "placebo_statistics_rows": 9,
        },
    }
    (root / "manifest.json").write_text(json.dumps(manifest, sort_keys=True))
    return root


@pytest.mark.parametrize("bad_intensity", [0.25, None])
def test_report_rejects_inconsistent_intensity_even_with_valid_hash(tmp_path, bad_intensity):
    module = _load_module()
    root = _write_empirical_bundle(tmp_path)
    path = root / "stratum_statistics.csv"
    frame = pl.read_csv(path).with_columns(
        pl.when(pl.col("contrast_label") == "own_only")
        .then(pl.lit(bad_intensity, dtype=pl.Float64))
        .otherwise(pl.col("withdrawal_intensity"))
        .alias("withdrawal_intensity")
    )
    frame.write_csv(path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][path.name]["sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="intensity"):
        module.build_empirical_report(bundle_root=root)
    assert not (root / "report.md").exists()


def test_build_empirical_report_validates_bundle_and_reports_support(tmp_path):
    module = _load_module()
    root = _write_empirical_bundle(tmp_path)

    output = module.build_empirical_report(bundle_root=root)

    text = output.read_text()
    assert "descriptive and exploratory" in text
    assert "Numerator" in text and "Denominator" in text
    assert "no_support" in text
    assert "not certified legitimate" in text
    assert "does not estimate false-positive rate, precision, recall, or intent" in text
    assert "not confidence intervals" in text


def test_report_rejects_inconsistent_clock_coverage_audit_even_with_valid_hash(tmp_path):
    module = _load_module()
    root = _write_empirical_bundle(tmp_path)
    path = root / "clock_coverage_audit.parquet"
    pl.read_parquet(path).with_columns(pl.lit(1).alias("accepted_event_count")).write_parquet(path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][path.name]["sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="accepted.*quarantined"):
        module.build_empirical_report(bundle_root=root)


def test_report_reads_lifecycle_episode_risk_linkage_fields(tmp_path):
    module = _load_module()
    root = _write_empirical_bundle(tmp_path)

    _, frames = module._load_validated_bundle(root)

    links = frames["episode_risk_links"]
    assert {
        "instrument", "episode_id", "execution_cluster_id", "risk_spell_id",
        "risk_interval_id", "physical_order_key",
    } <= set(links.columns)


@pytest.mark.parametrize(
    "missing_column",
    ["episode_id", "execution_cluster_id", "physical_order_key"],
)
def test_report_rejects_episode_risk_links_missing_lifecycle_provenance_fields(
    tmp_path, missing_column
):
    module = _load_module()
    root = _write_empirical_bundle(tmp_path)
    path = root / "episode_risk_links.parquet"
    pl.read_parquet(path).drop(missing_column).write_parquet(path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][path.name]["sha256"] = _sha256(path)
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match=missing_column):
        module.build_empirical_report(bundle_root=root)


def test_empirical_report_render_is_independent_of_table_row_order(tmp_path):
    module = _load_module()
    root = _write_empirical_bundle(tmp_path)
    first = module.build_empirical_report(bundle_root=root, output_path=root / "first.md").read_text()
    for name in ("stratum_statistics.csv", "placebo_statistics.parquet"):
        path = root / name
        frame = pl.read_csv(path) if name.endswith(".csv") else pl.read_parquet(path)
        reversed_frame = frame.reverse()
        reversed_frame.write_csv(path) if name.endswith(".csv") else reversed_frame.write_parquet(path)
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name in ("stratum_statistics.csv", "placebo_statistics.parquet"):
        manifest["artifacts"][name]["sha256"] = _sha256(root / name)
    manifest_path.write_text(json.dumps(manifest))

    second = module.build_empirical_report(bundle_root=root, output_path=root / "second.md").read_text()

    assert first == second


@pytest.mark.parametrize(
    "failure", [
        "missing", "graph", "provenance", "schema", "units", "hash",
        "total", "accounting", "validation",
    ]
)
def test_empirical_report_rejects_invalid_bundle_without_writing(tmp_path, failure):
    module = _load_module()
    root = _write_empirical_bundle(tmp_path)
    output = root / "report.md"
    manifest_path = root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if failure == "missing":
        (root / "placebo_statistics.parquet").unlink()
    elif failure == "graph":
        manifest["artifacts"].pop("risk_membership.parquet")
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "provenance":
        manifest.pop("code_snapshot")
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "schema":
        pl.DataFrame(schema={"wrong": pl.String}).write_parquet(root / "risk_membership.parquet")
        manifest["artifacts"]["risk_membership.parquet"]["sha256"] = _sha256(root / "risk_membership.parquet")
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "units":
        manifest["units"]["risk_membership.parquet"] = {}
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "hash":
        manifest["artifacts"]["placebo_support.parquet"]["sha256"] = "0" * 64
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "total":
        manifest["summary"]["withdrawal_count"] = 99
        manifest_path.write_text(json.dumps(manifest))
    elif failure == "accounting":
        coverage = pl.read_csv(root / "coverage_and_balance.csv").with_columns(
            pl.lit(999.0).alias("time_at_risk_seconds")
        )
        coverage.write_csv(root / "coverage_and_balance.csv")
        manifest["summary"]["time_at_risk_seconds"] = 999.0
        manifest["artifacts"]["coverage_and_balance.csv"]["sha256"] = _sha256(
            root / "coverage_and_balance.csv"
        )
        manifest_path.write_text(json.dumps(manifest))
    else:
        (root / "validation.json").write_text(json.dumps({"status": "fail"}))
        manifest["artifacts"]["validation.json"]["sha256"] = _sha256(root / "validation.json")
        manifest_path.write_text(json.dumps(manifest))

    with pytest.raises((ValueError, FileNotFoundError)):
        module.build_empirical_report(bundle_root=root, output_path=output)
    assert not output.exists()
