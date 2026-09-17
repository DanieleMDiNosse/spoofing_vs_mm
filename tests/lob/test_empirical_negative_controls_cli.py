from __future__ import annotations

import hashlib
import importlib.util
import json
from datetime import time
from pathlib import Path

import polars as pl
import pytest


SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "scripts"
    / "build_spoofing_empirical_negative_controls.py"
)


def load_module():
    spec = importlib.util.spec_from_file_location(
        "build_spoofing_empirical_negative_controls", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


CANONICAL_ARTIFACT_NAMES = (
    "execution_metrics",
    "candidate_deceptive_orders",
    "execution_cluster_members",
    "execution_cancel_candidates",
    "rejected_executions",
    "spoofing_compatible_events",
    "candidate_episodes",
    "episode_cluster_members",
    "episode_withdrawals",
    "episode_anchor_summary",
    "actor_day_episode_summary",
)


def canonical_empty_execution_metrics() -> pl.DataFrame:
    """Match the live rescore's explicit empty execution-metrics frame."""
    return pl.DataFrame(schema=load_module().EMPTY_CANONICAL_EXECUTION_METRICS_SCHEMA)


def write_frozen_canonical_artifacts(tmp_path: Path, detector: object | None = None) -> dict[str, str]:
    """Write every strict-gate artifact, using detector schemas when available."""
    frames: dict[str, pl.DataFrame]
    if detector is None:
        frames = {
            "execution_metrics": pl.DataFrame(
                schema={
                    "execution_cluster_id": pl.String, "partition_id": pl.String,
                    "actor_key": pl.String, "actor_id": pl.String, "identity_level": pl.String,
                    "identity_source": pl.String, "identity_fallback_flag": pl.Boolean,
                    "execution_anchor_mode": pl.String, "execution_quantity": pl.Float64,
                    "event_side": pl.String, "cluster_end_ts": pl.Datetime("us"),
                    "cluster_last_sort_index": pl.Int64,
                }
            )
        }
        frames.update({name: pl.DataFrame(schema={"_empty": pl.String}) for name in CANONICAL_ARTIFACT_NAMES if name not in frames})
    else:
        from spoofing_detection.lob.episode_artifacts import episode_frames

        frames = {
            "execution_metrics": (
                canonical_empty_execution_metrics()
                if detector.execution_metrics.is_empty()
                else detector.execution_metrics
            ),
            "candidate_deceptive_orders": detector.candidate_deceptive_orders,
            "execution_cluster_members": detector.execution_cluster_members,
            "execution_cancel_candidates": detector.execution_cancel_candidates,
            "rejected_executions": detector.rejected_executions,
            "spoofing_compatible_events": detector.spoofing_compatible_events,
            **episode_frames(detector.episode_result),
        }
    for name in CANONICAL_ARTIFACT_NAMES:
        frames[name].write_parquet(tmp_path / f"{name}.parquet")
    return {name: _sha256(tmp_path / f"{name}.parquet") for name in CANONICAL_ARTIFACT_NAMES}


def write_preflight_fixture(tmp_path: Path) -> tuple[Path, Path]:
    raw_events = tmp_path / "raw_events.parquet"
    pl.DataFrame(
        schema={
            "TRADEDATE": pl.Date,
            "MIC": pl.String,
            "MARKETCODE": pl.String,
            "SYMBOLINDEX": pl.Int64,
            "EMM (*)": pl.Int64,
            "HDR_APPLKEYSEQUENCENUMBER": pl.Int64,
            "HDR_HWMSEQUENCENUMBER": pl.Int64,
            "HDR_OFFSETID": pl.Int64,
            "ROW_NUMBER": pl.Int64,
            "TRADETIME": pl.Datetime("us"),
            "BOOKOUTTIME": pl.Datetime("us"),
            "BOOKIN": pl.Datetime("us"),
            "SEQUENCETIME": pl.Datetime("us"),
        }
    ).write_parquet(raw_events)
    quote_panel = tmp_path / "quote_panel.parquet"
    pl.DataFrame(
        schema={
            "sort_index": pl.Int64,
            "TRADEDATE": pl.Date,
            "pre_best_bid": pl.Float64,
            "pre_best_ask": pl.Float64,
            "post_best_bid": pl.Float64,
            "post_best_ask": pl.Float64,
        }
    ).write_parquet(quote_panel)
    kernel = tmp_path / "empirical_depth_kernel.parquet"
    pl.DataFrame(
        schema={
            "instrument_id": pl.String,
            "side": pl.String,
            "rank": pl.Int64,
            "kernel_weight": pl.Float64,
        }
    ).write_parquet(kernel)
    from spoofing_detection.lob.spoofing_metrics import compute_exploratory_metrics

    empty_detector = compute_exploratory_metrics(
        pl.read_parquet(raw_events),
        top_n=10,
        tick_size=0.1,
        window_seconds=1.0,
        withdrawal_window_seconds=2.0,
        reversion_horizon_seconds=2.0,
        max_deceptive_order_age_seconds=90.0,
        empirical_kernel_weights={
            "bid": {rank: 1.0 for rank in range(1, 11)},
            "ask": {rank: 1.0 for rank in range(1, 11)},
        },
    )
    artifact_hashes = write_frozen_canonical_artifacts(tmp_path, empty_detector)
    baseline_config = tmp_path / "baseline_config.json"
    baseline_config.write_text(
        json.dumps(
            {
                "metrics": {
                    "top_n": 10,
                    "withdrawal_window_seconds": 2.0,
                    "reversion_horizon_seconds": 2.0,
                    "max_deceptive_order_age_seconds": 90.0,
                    "actor_identity_mode": "client_then_firm",
                    "execution_anchor_modes": ["passive", "aggressive"],
                },
                "grid": {
                    "top_n": 999,
                    "withdrawal_window_seconds": 999.0,
                    "reversion_horizon_seconds": 999.0,
                    "max_deceptive_order_age_seconds": 999.0,
                },
            }
        )
    )

    source_hashes = {
        "raw_events_sha256": _sha256(raw_events),
        "quote_panel_sha256": _sha256(quote_panel),
        "empirical_depth_kernel_sha256": _sha256(kernel),
        "baseline_config_sha256": _sha256(baseline_config),
    }
    baseline_metadata = tmp_path / "baseline_metadata.json"
    baseline_metadata.write_text(
        json.dumps(
            {
                "parameter_source": "json_config_only",
                "config_section": "metrics",
                "output_schema_version": "actor_execution_anchor_v2",
                "actor_identity_mode": "client_then_firm",
                "execution_anchor_modes": ["passive", "aggressive"],
                "input": str(raw_events),
                "quote_panel": str(quote_panel),
                "empirical_depth_kernel": str(kernel),
                "config": str(baseline_config),
                "input_hashes": {
                    "config_sha256": source_hashes["baseline_config_sha256"],
                    **{
                        key: value
                        for key, value in source_hashes.items()
                        if key != "baseline_config_sha256"
                    },
                },
                "artifact_hashes": artifact_hashes,
            }
        )
    )

    output_root = tmp_path / "isolated_empirical_run"
    config = {
        "design_version": "empirical_controls_v1",
        "output_root": str(output_root),
        "baseline_runs": {
            "TEST": {
                "baseline_metadata": str(baseline_metadata),
                "source_paths": {
                    "raw_events": str(raw_events),
                    "quote_panel": str(quote_panel),
                    "empirical_depth_kernel": str(kernel),
                    "baseline_config": str(baseline_config),
                },
                "source_hashes": {
                    "baseline_metadata_sha256": _sha256(baseline_metadata),
                    **source_hashes,
                },
                "selected_event_dates": ["2024-01-02"],
            }
        },
        "clock": {
            "event_timestamp_precedence": ["TRADETIME", "BOOKOUTTIME", "BOOKIN", "SEQUENCETIME"],
            "partition_columns": ["TRADEDATE", "MIC", "MARKETCODE", "SYMBOLINDEX", "EMM (*)"],
            "stable_sort_columns": [
                "TRADEDATE", "MIC", "MARKETCODE", "SYMBOLINDEX", "EMM (*)", "SEQUENCETIME",
                "HDR_APPLKEYSEQUENCENUMBER", "HDR_HWMSEQUENCENUMBER", "HDR_OFFSETID", "BOOKIN",
                "BOOKOUTTIME", "TRADETIME", "ROW_NUMBER",
            ],
            "sort_index_policy": "sort_index_is_derived_after_canonical_stable_sort",
            "risk_clock_regression_policy": "quarantine_until_prior_accepted_high_watermark_never_reorder_or_clamp",
        },
        "horizons": {
            "anchor": "execution_cluster_end",
            "withdrawal_window_seconds": 2.0,
            "reversion_horizon_seconds": 2.0,
            "inclusive_upper_bound": True,
            "boundary_policy": "no_cross_instrument_partition_or_day_horizons",
        },
        "eligibility_policy": {
            "active_visible_only": True,
            "state_timing": "pre_event",
            "top_n": 10,
            "max_order_age_seconds": 90.0,
            "outcome_or_gate_selection": "forbidden",
        },
        "strata": {
            "block_columns": [
                "instrument",
                "HDR_PARTITIONID",
                "TRADEDATE",
                "actor_key",
                "posture_side",
            ],
            "allow_cross_boundary": False,
            "execution_anchor_modes": ["passive", "aggressive"],
            "identity_mode": "client_then_firm",
            "activity_bands": [
                {"name": "full_session", "start_time": "00:00:00", "end_time": "23:59:59.999999"}
            ],
        },
        "seed": 20260909,
        "placebo_draws": 1000,
        "comparability": {
            "version": "pre_anchor_covariates_v1", "anchor": "exposure_contrast_start",
            "quote_boundary_policy": "latest_same_partition_at_or_before_anchor_pre_state_canonical_sort_tiebreak",
            "history_window_seconds": 60.0, "history_interval": "[anchor-60s,anchor)",
            "outcomes_used": False, "matching": "forbidden", "adjusted_effect": "forbidden",
        },
        "output_schema": {
            "risk_spells": [
                "risk_spell_id",
                "instrument",
                "partition_id",
                "event_date",
                "actor_key",
                "posture_side",
            ],
            "risk_intervals": [
                "risk_interval_id",
                "risk_spell_id",
                "execution_anchor_mode",
                "duration_seconds",
            ],
            "exposure_intervals": [
                "execution_anchor_mode",
                "exposure_mask",
                "duration_seconds",
            ],
            "risk_membership": ["risk_membership_id", "physical_order_key", "duration_seconds"],
            "withdrawal_events": ["withdrawal_event_id", "physical_order_key", "visible_qty_removed"],
            "episode_risk_links": [
                "instrument", "episode_id", "execution_cluster_id", "risk_spell_id",
                "risk_interval_id", "physical_order_key",
            ],
            "placebo_schedules": ["draw_id", "schedule_block_id", "accepted", "rejection_reason"],
            "placebo_support": ["draw_id", "schedule_block_id", "accepted", "rejection_reason"],
            "placebo_statistics": ["draw_id", "row_kind", "time_at_risk_seconds", "withdrawal_count"],
            "coverage_and_balance": ["instrument", "event_date", "time_at_risk_seconds", "withdrawal_count"],
            "clock_coverage_audit": [
                "instrument", "partition_id", "event_date", "total_event_count",
                "accepted_event_count", "missing_timestamp_event_count",
                "clock_regression_event_count", "quarantined_event_count",
                "quarantine_episode_count", "recovered_quarantine_episode_count",
                "recovered_quarantine_duration_seconds", "unquantified_quarantine_event_count",
                "open_quarantine_at_end_flag", "first_accepted_ts", "last_accepted_ts",
                "observed_accepted_span_seconds", "time_at_risk_seconds", "coverage_loss_fraction",
                "terminal_right_censored_spell_count", "partition_boundary_censored_spell_count",
                "clock_ambiguous_censored_spell_count",
            ],
            "pre_event_covariates": [
                "risk_interval_id", "contrast_label", "anchor_ts", "eligible_visible_qty", "member_count",
                "mean_member_age_seconds", "oldest_member_age_seconds", "pre_spread_ticks", "pre_total_depth",
                "prior_event_count_60s", "pre_mid_log_return_sd_60s", "activity_band",
                "snapshot_sort_index", "anchor_sort_index", "missing_reasons",
            ],
            "common_support_audit": [
                "instrument", "event_date", "identity_level", "side", "comparison_label", "covariate",
                "total_seconds", "supported_seconds", "outside_support_seconds", "support_established", "supported", "null_fraction",
            ],
            "stratum_statistics": ["instrument", "contrast_label", "time_at_risk_seconds", "withdrawal_count"],
        },
    }
    config_path = tmp_path / "empirical_controls.json"
    config_path.write_text(json.dumps(config))
    return config_path, output_root


def refresh_baseline_config_hashes(config_path: Path) -> None:
    config = json.loads(config_path.read_text())
    run = config["baseline_runs"]["TEST"]
    baseline_config = Path(run["source_paths"]["baseline_config"])
    baseline_metadata = Path(run["baseline_metadata"])
    metadata = json.loads(baseline_metadata.read_text())
    baseline_hash = _sha256(baseline_config)
    run["source_hashes"]["baseline_config_sha256"] = baseline_hash
    metadata["input_hashes"]["config_sha256"] = baseline_hash
    baseline_metadata.write_text(json.dumps(metadata))
    run["source_hashes"]["baseline_metadata_sha256"] = _sha256(baseline_metadata)
    config_path.write_text(json.dumps(config))


def _clock_event(sequence: int, timestamp: str | None, *, partition: int = 1) -> dict[str, object]:
    return {
        "TRADEDATE": "2024-01-02", "MIC": "XMIL", "MARKETCODE": "MTA",
        "SYMBOLINDEX": 1, "EMM (*)": partition,
        "SEQUENCETIME": f"2024-01-02 09:30:{sequence:02d}",
        "HDR_APPLKEYSEQUENCENUMBER": sequence, "HDR_HWMSEQUENCENUMBER": sequence,
        "HDR_OFFSETID": sequence, "BOOKIN": None, "BOOKOUTTIME": None,
        "TRADETIME": timestamp, "ROW_NUMBER": sequence,
    }


def _empty_clock_risk_inputs() -> tuple[pl.DataFrame, pl.DataFrame]:
    return (
        pl.DataFrame(schema={
            "partition_id": pl.String, "event_date": pl.Date, "duration_seconds": pl.Float64,
        }),
        pl.DataFrame(schema={
            "partition_id": pl.String, "sort_index": pl.Int64, "reason": pl.String,
        }),
    )


def test_common_support_audit_handles_overlap_disjoint_missing_and_zero_duration() -> None:
    """Support is a pre-covariate range audit, never a matching procedure."""
    module = load_module()
    rows = []
    for label, value, seconds in (
        ("own_only", 3.0, 3.0), ("other_only", 3.0, 4.0),
        ("no_qualifying_fill", 9.0, 0.0),
    ):
        rows.append({
            "instrument": "TEST", "event_date": "2024-01-02", "identity_level": "client",
            "side": "bid", "contrast_label": label, "duration_seconds": seconds,
            **{name: value for name in module.PRE_EVENT_COVARIATES},
        })
    # A missing state covariate has no finite overlap and a null fraction at zero duration stays null.
    rows[-1]["pre_total_depth"] = None
    audit = module._common_support_audit(pl.DataFrame(rows).with_columns(pl.col("event_date").str.to_date()))
    overlap = audit.filter((pl.col("comparison_label") == "own_only_vs_other_only") & (pl.col("covariate") == "eligible_visible_qty"))
    assert overlap.get_column("support_established").to_list() == [True, True]
    assert overlap.get_column("supported_seconds").to_list() == [3.0, 4.0]
    disjoint = audit.filter((pl.col("comparison_label") == "own_only_vs_no_qualifying_fill") & (pl.col("covariate") == "eligible_visible_qty"))
    assert disjoint.get_column("support_established").to_list() == [False, False]
    assert disjoint.get_column("supported").to_list() == [0, 0]
    missing = audit.filter((pl.col("comparison_label") == "own_only_vs_no_qualifying_fill") & (pl.col("covariate") == "pre_total_depth"))
    assert missing.get_column("support_established").to_list() == [False, False]
    assert missing.filter(pl.col("contrast_label") == "no_qualifying_fill").item(0, "null_fraction") is None
    joint = audit.filter((pl.col("comparison_label") == "own_only_vs_other_only") & (pl.col("covariate") == "__all_required__"))
    assert joint.get_column("support_established").to_list() == [True, True]
    assert joint.get_column("supported_seconds").to_list() == [3.0, 4.0]
    assert all(
        total == supported + outside
        for total, supported, outside in joint.select("total_seconds", "supported_seconds", "outside_support_seconds").iter_rows()
    )


def _covariate_inputs(module: object) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build a selected replay whose 0.1 tick differs from every spread minimum."""
    from test_withdrawal_risk import event

    raw = pl.DataFrame([
        event(1, 1, "A", 1, 100.0, 10, 10, ts="2024-01-02 09:30:00"),
        event(2, 2, "A", 1, 100.1, 10, 10, ts="2024-01-02 09:30:01"),
        event(3, 2, "A", 1, 100.2, 10, 10, ts="2024-01-02 09:30:10"),
    ])
    canonical = module.sort_events(raw)
    quotes = pl.DataFrame({
        "sort_index": [0, 1, 2],
        **{column: canonical.get_column(column) for column in module.CANONICAL_PARTITION_COLUMNS},
        "pre_best_bid": [100.0, 100.1, 100.2],
        "pre_best_ask": [100.3, 100.3, 100.5],
        "post_best_bid": [100.0, 100.1, 100.2],
        "post_best_ask": [100.3, 100.3, 100.5],
        "pre_bid_visible_qty_total": [5.0, 6.0, 999.0],
        "pre_ask_visible_qty_total": [7.0, 8.0, 999.0],
    })
    partition = module._partition_id(canonical.row(0, named=True))
    contrasts = pl.DataFrame({
        "risk_interval_id": ["r"], "partition_id": [partition], "event_date": ["2024-01-02"],
        "actor_key": ["client_original:C1"], "identity_level": ["client"], "side": ["bid"],
        "contrast_label": ["own_only"], "start_ts": ["2024-01-02 09:30:01"],
        "duration_seconds": [2.0],
    }).with_columns(pl.col("event_date").str.to_date(), pl.col("start_ts").str.to_datetime())
    intervals = pl.DataFrame({
        "risk_interval_id": ["r"], "start_sort_index": [1],
        "eligible_visible_qty": [10.0], "member_count": [1.0],
        "mean_member_age_seconds": [2.0], "oldest_member_age_seconds": [2.0],
    })
    return raw, quotes, contrasts, intervals


def test_pre_event_covariates_use_explicit_tick_selected_ordinals_and_no_future() -> None:
    """Canonical tick and tuple boundaries prevent spread and future-data leakage."""
    module = load_module()
    raw, quotes, contrasts, intervals = _covariate_inputs(module)
    output = module._pre_event_covariates(
        "TEST", raw, quotes, contrasts, intervals, canonical_tick_size=0.1,
        activity_bands=[{"name": "session", "start_time": time(9), "end_time": time(10)}],
    )
    row = output.row(0, named=True)
    assert row["pre_spread_ticks"] == pytest.approx(2.0)  # min spread is 0.2, not the 0.1 tick.
    assert row["snapshot_sort_index"] == row["anchor_sort_index"] == 1
    assert row["pre_total_depth"] == 14.0
    assert row["activity_band"] == "2024-01-02:session"
    changed_future = quotes.with_columns(
        pl.when(pl.col("sort_index") == 2).then(1_000_000.0).otherwise(pl.col("pre_bid_visible_qty_total")).alias("pre_bid_visible_qty_total")
    )
    assert module._pre_event_covariates(
        "TEST", raw, changed_future, contrasts, intervals, canonical_tick_size=0.1,
        activity_bands=[{"name": "session", "start_time": time(9), "end_time": time(10)}],
    ).row(0, named=True)["pre_total_depth"] == 14.0


def test_pre_event_synthetic_anchor_and_filtered_panel_use_last_past_selected_boundary() -> None:
    """Filtering an earlier date renumbers panel rows without reintroducing future state."""
    module = load_module()
    raw, quotes, contrasts, intervals = _covariate_inputs(module)
    full_quotes = pl.concat([
        quotes.head(1).with_columns(pl.lit(0, dtype=quotes.schema["sort_index"]).alias("sort_index"), pl.lit("2024-01-01").alias("TRADEDATE")),
        quotes.with_columns((pl.col("sort_index") + 1).alias("sort_index")),
    ])
    selected_quotes = module._selected_date_frame(full_quotes, {"2024-01-02"})
    synthetic = contrasts.with_columns(pl.lit(None, dtype=pl.Int64).alias("unused"))
    output = module._pre_event_covariates(
        "TEST", raw, selected_quotes, synthetic, intervals.with_columns(pl.lit(None, dtype=pl.Int64).alias("start_sort_index")),
        canonical_tick_size=0.1,
        activity_bands=[{"name": "session", "start_time": time(9), "end_time": time(10)}],
    )
    row = output.row(0, named=True)
    assert row["anchor_sort_index"] == row["snapshot_sort_index"] == 1
    assert row["pre_total_depth"] == 14.0


def test_profile_balance_excludes_missing_duration_from_covariate_mean() -> None:
    """Missing covariates are reported separately and never treated as zero."""
    module = load_module()
    rows = [
        {"event_date": "2024-01-02", "identity_level": "client", "side": "bid", "contrast_label": "own_only", "duration_seconds": seconds,
         **{name: (value if name == "pre_total_depth" else 1.0) for name in module.PRE_EVENT_COVARIATES}}
        for seconds, value in ((2.0, 10.0), (8.0, None))
    ]
    balance = module._profile_balance(pl.DataFrame(rows).with_columns(pl.col("event_date").str.to_date()))
    assert balance.item(0, "mean_pre_total_depth") == 10.0
    assert balance.item(0, "pre_total_depth_observed_seconds") == 2.0
    assert balance.item(0, "pre_total_depth_missing_seconds") == 8.0


def test_clock_coverage_audit_recovers_one_episode_after_multiple_regressions() -> None:
    module = load_module()
    intervals, diagnostics = _empty_clock_risk_inputs()
    audit = module.build_clock_coverage_audit(
        "TEST", pl.DataFrame([
            _clock_event(1, "2024-01-02 09:30:10"),
            _clock_event(2, "2024-01-02 09:30:09"),
            _clock_event(3, "2024-01-02 09:30:08"),
            _clock_event(4, "2024-01-02 09:30:11"),
        ]), intervals=intervals, diagnostics=diagnostics,
    )

    row = audit.row(0, named=True)
    assert row["total_event_count"] == 4
    assert row["accepted_event_count"] == 2
    assert row["clock_regression_event_count"] == row["quarantined_event_count"] == 2
    assert row["quarantine_episode_count"] == row["recovered_quarantine_episode_count"] == 1
    assert row["recovered_quarantine_duration_seconds"] == 1.0
    assert row["unquantified_quarantine_event_count"] == 0
    assert row["open_quarantine_at_end_flag"] is False
    assert row["observed_accepted_span_seconds"] == 1.0
    assert row["coverage_loss_fraction"] == pytest.approx(0.5)


def test_clock_coverage_audit_missing_clock_starts_unknown_open_quarantine() -> None:
    module = load_module()
    intervals, diagnostics = _empty_clock_risk_inputs()
    missing = _clock_event(2, None)
    missing["SEQUENCETIME"] = None
    audit = module.build_clock_coverage_audit(
        "TEST", pl.DataFrame([
            _clock_event(1, "2024-01-02 09:30:00"), missing,
            _clock_event(3, "2024-01-02 09:30:02"),
        ]), intervals=intervals, diagnostics=diagnostics,
    )

    row = audit.row(0, named=True)
    assert row["missing_timestamp_event_count"] == row["quarantined_event_count"] == 1
    assert row["quarantine_episode_count"] == 1
    assert row["recovered_quarantine_episode_count"] == 0
    assert row["unquantified_quarantine_event_count"] == 1
    # Canonical null-last ordering places a fully missing clock after valid clocks.
    assert row["open_quarantine_at_end_flag"] is True
    assert row["accepted_event_count"] == 2


def test_clock_coverage_audit_marks_open_regression_unquantified_at_partition_end() -> None:
    module = load_module()
    intervals, diagnostics = _empty_clock_risk_inputs()
    audit = module.build_clock_coverage_audit(
        "TEST", pl.DataFrame([
            _clock_event(1, "2024-01-02 09:30:10"), _clock_event(2, "2024-01-02 09:30:09"),
        ]), intervals=intervals, diagnostics=diagnostics,
    )

    row = audit.row(0, named=True)
    assert row["recovered_quarantine_duration_seconds"] == 0.0
    assert row["unquantified_quarantine_event_count"] == 1
    assert row["open_quarantine_at_end_flag"] is True


def test_clock_coverage_audit_clean_and_empty_inputs_have_explicit_schema() -> None:
    module = load_module()
    intervals, diagnostics = _empty_clock_risk_inputs()
    clean = module.build_clock_coverage_audit(
        "TEST", pl.DataFrame([
            _clock_event(1, "2024-01-02 09:30:00"), _clock_event(2, "2024-01-02 09:30:02"),
        ]), intervals=intervals, diagnostics=diagnostics,
    ).row(0, named=True)
    empty = module.build_clock_coverage_audit(
        "TEST", pl.DataFrame(schema={key: pl.String for key in module.SORT_COLUMNS}),
        intervals=intervals, diagnostics=diagnostics,
    )

    assert clean["accepted_event_count"] == clean["total_event_count"] == 2
    assert clean["quarantined_event_count"] == clean["recovered_quarantine_duration_seconds"] == 0
    assert clean["coverage_loss_fraction"] == 0.0
    assert empty.is_empty()
    assert set(module.CLOCK_COVERAGE_AUDIT_SCHEMA) == set(empty.columns)


def test_validate_only_accepts_empty_schema_valid_input_without_creating_output(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)

    module.main(["--config", str(config_path), "--validate-only"])

    assert "no output created" in capsys.readouterr().out
    assert not output_root.exists()


def test_preflight_requires_every_frozen_canonical_detector_artifact(tmp_path: Path) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    (tmp_path / "candidate_deceptive_orders.parquet").unlink()

    with pytest.raises((ValueError, FileNotFoundError), match="candidate_deceptive_orders"):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


def test_validate_only_rejects_missing_frozen_execution_metrics(tmp_path: Path) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    (tmp_path / "execution_metrics.parquet").unlink()

    with pytest.raises(FileNotFoundError, match="execution_metrics"):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


def test_preflight_retains_effective_detector_parameters_from_metrics(tmp_path: Path) -> None:
    module = load_module()
    config_path, _ = write_preflight_fixture(tmp_path)

    validated = module.validate_preflight(config_path)

    assert validated["effective_parameters"] == {
        "TEST": {
            "top_n": 10,
            "withdrawal_window_seconds": 2.0,
            "reversion_horizon_seconds": 2.0,
            "max_deceptive_order_age_seconds": 90.0,
            "actor_identity_mode": "client_then_firm",
            "execution_anchor_modes": ["passive", "aggressive"],
        }
    }


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda config: config.pop("design_version"), "design_version"),
        (
            lambda config: config["baseline_runs"]["TEST"]["source_paths"].update(
                {"raw_events": "different.parquet"}
            ),
            "baseline metadata",
        ),
    ],
)
def test_preflight_rejects_missing_or_incompatible_contract_metadata_before_writes(
    tmp_path: Path, change, message: str
) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    change(config)
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda metadata: metadata.pop("output_schema_version"), "output_schema_version"),
        (
            lambda metadata: metadata.update({"output_schema_version": "legacy_schema"}),
            "output_schema_version",
        ),
    ],
)
def test_preflight_rejects_missing_or_incompatible_baseline_metadata(
    tmp_path: Path, mutate, message: str
) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    metadata_path = Path(config["baseline_runs"]["TEST"]["baseline_metadata"])
    metadata = json.loads(metadata_path.read_text())
    mutate(metadata)
    metadata_path.write_text(json.dumps(metadata))
    config["baseline_runs"]["TEST"]["source_hashes"]["baseline_metadata_sha256"] = _sha256(
        metadata_path
    )
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


def test_preflight_rejects_split_policy_that_shares_partition_boundaries(tmp_path: Path) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["strata"]["block_columns"].remove("HDR_PARTITIONID")
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="split boundaries"):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda config: config["clock"].update(
                {"event_timestamp_precedence": ["BOOKOUTTIME", "TRADETIME", "BOOKIN", "SEQUENCETIME"]}
            ),
            "event_timestamp_precedence",
        ),
        (
            lambda config: config["clock"]["stable_sort_columns"].pop(),
            "stable_sort_columns",
        ),
        (
            lambda config: config["clock"].update({"sort_index_policy": "raw_row_number"}),
            "sort_index_policy",
        ),
        (
            lambda config: config["clock"].update({"risk_clock_regression_policy": "reorder"}),
            "risk_clock_regression_policy",
        ),
    ],
)
def test_preflight_rejects_noncanonical_clock_before_replay_or_writes(
    tmp_path: Path, mutate, message: str
) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    mutate(config)
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


def test_preflight_requires_all_canonical_stable_sort_columns(tmp_path: Path) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    run = config["baseline_runs"]["TEST"]
    raw_path = Path(run["source_paths"]["raw_events"])
    pl.read_parquet(raw_path).drop("ROW_NUMBER").write_parquet(raw_path)
    raw_hash = _sha256(raw_path)
    run["source_hashes"]["raw_events_sha256"] = raw_hash
    metadata_path = Path(run["baseline_metadata"])
    metadata = json.loads(metadata_path.read_text())
    metadata["input_hashes"]["raw_events_sha256"] = raw_hash
    metadata_path.write_text(json.dumps(metadata))
    run["source_hashes"]["baseline_metadata_sha256"] = _sha256(metadata_path)
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="ROW_NUMBER"):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


def test_preflight_rejects_existing_isolated_output_root(tmp_path: Path) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    output_root.mkdir()

    with pytest.raises(FileExistsError, match="output already exists"):
        module.main(["--config", str(config_path), "--validate-only"])


@pytest.mark.parametrize("malformed_json", ["[]", "null", "42"])
def test_preflight_rejects_non_object_top_level_json(tmp_path: Path, malformed_json: str) -> None:
    module = load_module()
    config_path = tmp_path / "empirical_controls.json"
    config_path.write_text(malformed_json)

    with pytest.raises(ValueError, match="empirical controls config must be a JSON object"):
        module.validate_preflight(config_path)


def test_preflight_requires_exact_empirical_controls_v1_design_version(tmp_path: Path) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["design_version"] = "empirical_controls_v2"
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="design_version must be empirical_controls_v1"):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("seed",), True, "seed must be an integer"),
        (("placebo_draws",), True, "placebo_draws must be positive"),
        (("eligibility_policy", "top_n"), True, "eligibility_policy.top_n must be positive"),
    ],
)
def test_preflight_rejects_boolean_counts(
    tmp_path: Path, path: tuple[str, ...], value: bool, message: str
) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("horizons", "withdrawal_window_seconds"),
            True,
            "horizons.withdrawal_window_seconds must be a finite positive duration",
        ),
        (
            ("horizons", "withdrawal_window_seconds"),
            float("nan"),
            "horizons.withdrawal_window_seconds must be a finite positive duration",
        ),
        (
            ("horizons", "withdrawal_window_seconds"),
            float("inf"),
            "horizons.withdrawal_window_seconds must be a finite positive duration",
        ),
        (
            ("horizons", "reversion_horizon_seconds"),
            True,
            "horizons.reversion_horizon_seconds must be a finite positive duration",
        ),
        (
            ("horizons", "reversion_horizon_seconds"),
            float("nan"),
            "horizons.reversion_horizon_seconds must be a finite positive duration",
        ),
        (
            ("horizons", "reversion_horizon_seconds"),
            float("inf"),
            "horizons.reversion_horizon_seconds must be a finite positive duration",
        ),
        (
            ("eligibility_policy", "max_order_age_seconds"),
            True,
            "eligibility_policy.max_order_age_seconds must be a finite positive duration",
        ),
        (
            ("eligibility_policy", "max_order_age_seconds"),
            float("nan"),
            "eligibility_policy.max_order_age_seconds must be a finite positive duration",
        ),
        (
            ("eligibility_policy", "max_order_age_seconds"),
            float("inf"),
            "eligibility_policy.max_order_age_seconds must be a finite positive duration",
        ),
    ],
)
def test_preflight_rejects_nonfinite_or_boolean_durations(
    tmp_path: Path, path: tuple[str, str], value: float | bool, message: str
) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config[path[0]][path[1]] = value
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


def test_preflight_rejects_actual_source_hash_mismatch(tmp_path: Path) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    raw_events = Path(config["baseline_runs"]["TEST"]["source_paths"]["raw_events"])
    raw_events.write_bytes(b"changed after the configured hash was recorded")

    with pytest.raises(ValueError, match="source hash mismatch"):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


@pytest.mark.parametrize(
    ("source_key", "metadata_hash_key", "message"),
    [
        ("quote_panel", "quote_panel_sha256", "quote panel schema is unreadable"),
        (
            "empirical_depth_kernel",
            "empirical_depth_kernel_sha256",
            "empirical depth kernel schema is unreadable",
        ),
    ],
)
def test_preflight_rejects_hash_consistent_malformed_parquet_sources(
    tmp_path: Path, source_key: str, metadata_hash_key: str, message: str
) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    run = config["baseline_runs"]["TEST"]
    source_path = Path(run["source_paths"][source_key])
    source_path.write_bytes(b"not parquet")
    new_hash = _sha256(source_path)
    run["source_hashes"][metadata_hash_key] = new_hash

    metadata_path = Path(run["baseline_metadata"])
    metadata = json.loads(metadata_path.read_text())
    metadata["input_hashes"][metadata_hash_key] = new_hash
    metadata_path.write_text(json.dumps(metadata))
    run["source_hashes"]["baseline_metadata_sha256"] = _sha256(metadata_path)
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda config: config["horizons"].pop("reversion_horizon_seconds"), "reversion_horizon_seconds"),
        (lambda config: config["horizons"].update({"reversion_horizon_seconds": 3.0}), "reversion_horizon_seconds"),
        (lambda config: config["horizons"].pop("boundary_policy"), "boundary_policy"),
        (lambda config: config["horizons"].update({"boundary_policy": "cross_day"}), "boundary_policy"),
        (lambda config: config["eligibility_policy"].pop("state_timing"), "state_timing"),
        (lambda config: config["eligibility_policy"].update({"state_timing": "post_event"}), "state_timing"),
        (lambda config: config["eligibility_policy"].pop("outcome_or_gate_selection"), "outcome_or_gate_selection"),
        (lambda config: config["eligibility_policy"].update({"outcome_or_gate_selection": "allowed"}), "outcome_or_gate_selection"),
        (lambda config: config["strata"].pop("execution_anchor_modes"), "execution_anchor_modes"),
        (lambda config: config["strata"].update({"execution_anchor_modes": ["passive"]}), "execution_anchor_modes"),
        (lambda config: config["strata"].pop("identity_mode"), "identity_mode"),
        (lambda config: config["strata"].update({"identity_mode": "firm"}), "identity_mode"),
    ],
)
def test_preflight_requires_exact_v1_semantic_controls(tmp_path: Path, mutate, message: str) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    mutate(config)
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match=message):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda baseline: baseline.clear(), "metrics"),
        (lambda baseline: baseline.pop("metrics"), "metrics"),
        (lambda baseline: baseline["metrics"].update({"top_n": 9}), "top_n"),
        (lambda baseline: baseline["metrics"].update({"withdrawal_window_seconds": 3.0}), "withdrawal_window_seconds"),
        (lambda baseline: baseline["metrics"].update({"reversion_horizon_seconds": 3.0}), "reversion_horizon_seconds"),
        (lambda baseline: baseline["metrics"].update({"max_deceptive_order_age_seconds": 30.0}), "max_deceptive_order_age_seconds"),
        (lambda baseline: baseline["metrics"].update({"actor_identity_mode": "firm"}), "actor_identity_mode"),
        (lambda baseline: baseline["metrics"].update({"execution_anchor_modes": ["passive"]}), "execution_anchor_modes"),
    ],
)
def test_preflight_loads_frozen_detector_parameters_from_metrics(tmp_path: Path, mutate, message: str) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    baseline_config = Path(config["baseline_runs"]["TEST"]["source_paths"]["baseline_config"])
    baseline = json.loads(baseline_config.read_text())
    mutate(baseline)
    baseline_config.write_text(json.dumps(baseline))
    refresh_baseline_config_hashes(config_path)

    with pytest.raises(ValueError, match=message):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


@pytest.mark.parametrize("event_date", ["2024-2-03", "2024-02-30", "not-a-date", "2024-02-03T00:00:00"])
def test_preflight_rejects_non_iso_selected_event_dates(tmp_path: Path, event_date: str) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["baseline_runs"]["TEST"]["selected_event_dates"] = [event_date]
    config_path.write_text(json.dumps(config))

    with pytest.raises(ValueError, match="selected_event_dates"):
        module.main(["--config", str(config_path), "--validate-only"])

    assert not output_root.exists()


def test_episode_risk_links_use_physical_lifecycle_and_keep_no_fill_risk(tmp_path):
    from spoofing_detection.lob.withdrawal_risk import compute_withdrawal_risk
    from test_withdrawal_risk import event
    module = load_module()
    risk = compute_withdrawal_risk(pl.DataFrame([
        event(1, 1, "B", 1, 100., 10, 10),
        event(2, 1, "A", 2, 101., 10, 10, client="C2"),
        event(3, 4, "B", 1, 100., 0, 0),
    ]), top_n=1, max_order_age_seconds=60)
    partition = risk.intervals.item(0, "partition_id")
    candidates = pl.DataFrame({"partition_id":[partition], "actor_key":["client_original:C1"],
        "deceptive_side":["bid"], "deceptive_order_id":["B"],
        "deceptive_order_first_seen_sort_index":[1], "execution_cluster_id":["C"]})
    members = pl.DataFrame({"partition_id":[partition],"actor_key":["client_original:C1"],
        "execution_cluster_id":["C"], "episode_id":["E"]})
    links = module._episode_risk_links(risk, candidates, members)
    assert links.get_column("episode_id").unique().to_list() == ["E"]
    assert links.get_column("execution_cluster_id").unique().to_list() == ["C"]
    assert set(links.get_column("risk_interval_id")) == set(risk.intervals.filter(pl.col("actor_key")=="client_original:C1").get_column("risk_interval_id"))
    assert risk.intervals.filter(pl.col("actor_key")=="client_original:C2").height > 0
    wrong_lifecycle = candidates.with_columns(pl.lit(99).alias("deceptive_order_first_seen_sort_index"))
    assert module._episode_risk_links(risk, wrong_lifecycle, members).is_empty()


def test_runner_publishes_empty_schema_valid_bundle(tmp_path: Path) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    config["placebo_draws"] = 2
    config_path.write_text(json.dumps(config))
    assert module.run_empirical_controls(config_path) == output_root
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["summary"]["risk_spell_count"] == 0
    assert manifest["summary"]["time_at_risk_seconds"] == 0.0
    assert manifest["summary"]["withdrawal_count"] == 0
    assert (output_root / "report.md").is_file()


def test_canonical_live_rescore_rejects_nonempty_raw_without_positive_quote_change(
    tmp_path: Path,
) -> None:
    module = load_module()
    raw_events = tmp_path / "raw_events.parquet"
    pl.DataFrame({"TRADEDATE": ["2024-01-02"]}).with_columns(
        pl.col("TRADEDATE").str.to_date()
    ).write_parquet(raw_events)
    quote_panel = tmp_path / "quote_panel.parquet"
    pl.DataFrame({
        "sort_index": [1, 2],
        "TRADEDATE": ["2024-01-02", "2024-01-02"],
        "pre_best_bid": [100.0, 100.0],
        "pre_best_ask": [100.1, 100.1],
        "post_best_bid": [100.0, 100.0],
        "post_best_ask": [100.1, 100.1],
    }).with_columns(pl.col("TRADEDATE").str.to_date()).write_parquet(quote_panel)

    with pytest.raises(ValueError, match="cannot infer tick size.*no positive price changes"):
        module._canonical_live_frames(
            {
                "source_paths": {
                    "raw_events": str(raw_events),
                    "quote_panel": str(quote_panel),
                }
            },
            effective_parameters={
                "top_n": 10,
                "withdrawal_window_seconds": 2.0,
                "reversion_horizon_seconds": 2.0,
                "max_deceptive_order_age_seconds": 90.0,
                "execution_anchor_modes": ["passive", "aggressive"],
            },
            selected_dates={"2024-01-02"},
        )


@pytest.mark.parametrize("raw_date_dtype", [pl.String, pl.Date, pl.Datetime("us")])
@pytest.mark.parametrize("frozen_candidate_mismatch", [False, True])
def test_runner_recomputes_and_reconciles_every_live_canonical_artifact_from_raw_replay(
    tmp_path: Path, raw_date_dtype: pl.DataType, frozen_candidate_mismatch: bool,
) -> None:
    module = load_module()
    config_path, output_root = write_preflight_fixture(tmp_path)
    config = json.loads(config_path.read_text())
    run = config["baseline_runs"]["TEST"]
    raw_path = Path(run["source_paths"]["raw_events"])

    from test_episode_pipeline import raw_fixture

    # This live raw feed contains a verified candidate posture and episode, plus
    # other-actor risk time that remains unlinked.
    raw_fixture().with_columns(pl.lit("P").alias("HDR_PARTITIONID")).write_parquet(raw_path)
    raw = pl.read_parquet(raw_path).with_columns(
        pl.col("TRADEDATE").cast(pl.Date).cast(raw_date_dtype)
    )
    raw.write_parquet(raw_path)
    quote_path = Path(run["source_paths"]["quote_panel"])
    canonical_raw = module.sort_events(raw)
    pl.DataFrame({
        "sort_index": list(range(canonical_raw.height)),
        **{column: canonical_raw.get_column(column) for column in module.CANONICAL_PARTITION_COLUMNS},
        "pre_best_bid": [0.1 + 0.1 * (index % 2) for index in range(canonical_raw.height)],
        "pre_best_ask": [0.2 for _ in range(canonical_raw.height)],
        "post_best_bid": [0.1 + 0.1 * (index % 2) for index in range(canonical_raw.height)],
        "post_best_ask": [0.2 for _ in range(canonical_raw.height)],
    }).write_parquet(quote_path)
    quote_hash = _sha256(quote_path)
    run["source_hashes"]["quote_panel_sha256"] = quote_hash
    kernel_path = Path(run["source_paths"]["empirical_depth_kernel"])
    pl.DataFrame([
        {"instrument_id": "TEST", "side": side, "rank": rank, "kernel_weight": 1.0}
        for side in ("bid", "ask")
        for rank in range(1, 11)
    ]).write_parquet(kernel_path)
    kernel_hash = _sha256(kernel_path)
    run["source_hashes"]["empirical_depth_kernel_sha256"] = kernel_hash
    from spoofing_detection.lob.spoofing_metrics import compute_exploratory_metrics

    detector = compute_exploratory_metrics(
        raw, top_n=10, tick_size=0.1, window_seconds=1.0,
        withdrawal_window_seconds=2.0, reversion_horizon_seconds=2.0,
        max_deceptive_order_age_seconds=90.0,
        empirical_kernel_weights={
            "bid": {rank: 1.0 for rank in range(1, 11)},
            "ask": {rank: 1.0 for rank in range(1, 11)},
        },
    )
    assert detector.execution_metrics.height == 2
    metadata_path = Path(run["baseline_metadata"])
    metadata = json.loads(metadata_path.read_text())
    metadata["artifact_hashes"] = write_frozen_canonical_artifacts(tmp_path, detector)
    if frozen_candidate_mismatch:
        candidate_path = tmp_path / "candidate_deceptive_orders.parquet"
        detector.candidate_deceptive_orders.with_columns(
            pl.lit(True).alias("_forced_mismatch")
        ).write_parquet(candidate_path)
        metadata["artifact_hashes"]["candidate_deceptive_orders"] = _sha256(candidate_path)
    metadata["input_hashes"]["raw_events_sha256"] = _sha256(raw_path)
    metadata["input_hashes"]["quote_panel_sha256"] = quote_hash
    metadata["input_hashes"]["empirical_depth_kernel_sha256"] = kernel_hash
    metadata_path.write_text(json.dumps(metadata))
    run["source_hashes"]["raw_events_sha256"] = _sha256(raw_path)
    run["source_hashes"]["baseline_metadata_sha256"] = _sha256(metadata_path)
    config["placebo_draws"] = 2
    config_path.write_text(json.dumps(config))

    if frozen_candidate_mismatch:
        with pytest.raises(ValueError, match="canonical equivalence.*candidate_deceptive_orders"):
            module.run_empirical_controls(config_path)
        assert not output_root.exists()
        return

    result = module.run_empirical_controls(config_path)

    assert result == output_root
    expected = {
        "config.json", "versions.txt", "command.txt", "manifest.json", "run.log",
        "risk_spells.parquet", "risk_intervals.parquet", "risk_membership.parquet",
        "exposure_intervals.parquet", "withdrawal_events.parquet", "episode_risk_links.parquet",
        "placebo_schedules.parquet", "placebo_support.parquet", "placebo_statistics.parquet",
        "coverage_and_balance.csv", "stratum_statistics.csv", "report.md", "validation.json",
        "canonical_equivalence.parquet",
    }
    assert expected <= {path.name for path in output_root.iterdir()}
    manifest = json.loads((output_root / "manifest.json").read_text())
    assert manifest["summary"]["risk_spell_count"] == pl.read_parquet(output_root / "risk_spells.parquet").height
    withdrawals = pl.read_parquet(output_root / "withdrawal_events.parquet")
    assert manifest["summary"]["withdrawal_count"] == withdrawals.height == 3
    assert withdrawals.item(0, "transition_reason") == "cancellation"
    assert pl.read_parquet(output_root / "risk_membership.parquet").height >= 1
    links = pl.read_parquet(output_root / "episode_risk_links.parquet")
    required_link_columns = {
        "instrument", "episode_id", "execution_cluster_id", "risk_spell_id",
        "risk_interval_id", "physical_order_key",
    }
    assert required_link_columns <= set(links.columns)
    assert not links.is_empty()
    candidate_members = detector.candidate_deceptive_orders.select(
        "partition_id", "actor_key", "execution_cluster_id",
        pl.col("deceptive_order_id").alias("order_id"),
        pl.col("deceptive_order_first_seen_sort_index").alias("first_seen_sort_index"),
    ).join(
        module.episode_frames(detector.episode_result)["episode_cluster_members"].select(
            "partition_id", "actor_key", "execution_cluster_id", "episode_id"
        ),
        on=["partition_id", "actor_key", "execution_cluster_id"], how="inner",
    )
    assert links.join(
        candidate_members,
        on=["episode_id", "execution_cluster_id"], how="anti",
    ).is_empty()
    risk_intervals = pl.read_parquet(output_root / "risk_intervals.parquet")
    assert links.join(
        risk_intervals.select("instrument", "risk_interval_id"),
        on=["instrument", "risk_interval_id"], how="anti",
    ).is_empty()
    assert risk_intervals.height > links.get_column("risk_interval_id").n_unique()
    assert manifest["summary"]["time_at_risk_seconds"] == pytest.approx(39.0)
    assert pl.read_parquet(output_root / "exposure_intervals.parquet").height >= 1
    assert manifest["summary"]["accepted_placebo_blocks"] >= 1
    statistics = pl.read_parquet(output_root / "placebo_statistics.parquet")
    assert statistics.get_column("draw_id").n_unique() == 3
    assert (
        statistics.filter(pl.col("row_kind") == "state")
        .group_by("draw_id")
        .agg(pl.col("withdrawal_count").sum())
        .get_column("withdrawal_count")
        .to_list()
        == [3, 3, 3]
    )
    scientific_tables = {
        name for name in expected if name.endswith((".parquet", ".csv"))
    }
    assert scientific_tables <= set(manifest["units"])
    assert json.loads((output_root / "validation.json").read_text())["status"] == "pass"
    assert "descriptive and exploratory" in (output_root / "report.md").read_text()
    assert (output_root / "risk_diagnostics.parquet").is_file()
    assert (output_root / "risk_transitions.parquet").is_file()
    balance = pl.read_csv(output_root / "profile_balance.csv")
    assert "mean_profile_visible_qty" in balance.columns
    assert manifest["effective_clock"]["event_timestamp_precedence"] == [
        "TRADETIME", "BOOKOUTTIME", "BOOKIN", "SEQUENCETIME"
    ]
    assert manifest["effective_clock"]["stable_sort_columns"] == module.SORT_COLUMNS
    assert manifest["effective_clock"]["sort_index_policy"] == module.SORT_INDEX_POLICY
    assert (
        manifest["effective_clock"]["risk_clock_regression_policy"]
        == module.CLOCK_REGRESSION_POLICY
    )
    assert manifest["configured_execution_anchor_modes"] == ["passive", "aggressive"]
    assert manifest["observed_execution_anchor_modes"] == ["passive", "aggressive"]
    equivalence = pl.read_parquet(output_root / "canonical_equivalence.parquet")
    assert equivalence.filter(~pl.col("equal")).is_empty()
    assert set(equivalence.get_column("artifact")) == set(CANONICAL_ARTIFACT_NAMES)
    assert equivalence.get_column("state_series_exclusion").unique().to_list() == [
        module.STATE_SERIES_EXCLUSION_REASON
    ]
    assert "canonical_equivalence.parquet" in manifest["units"]
    assert "unadjusted" in (output_root / "report.md").read_text()
    assert "clock_regression" in (output_root / "report.md").read_text()
