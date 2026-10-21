"""Contract and read-only preflight tests for the new multiscale CLI."""
from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path

import pytest


def api():
    return importlib.import_module("scripts.study_withdrawal_multiscale")


def config():
    return json.loads(Path("configs/withdrawal_multiscale_v1.json").read_text())


def test_config_contract_and_missing():
    m = api()
    good = config()
    m.validate_config(good)
    bad = copy.deepcopy(good)
    del bad["depth_denominator"]
    with pytest.raises(ValueError):
        m.validate_config(bad)


@pytest.mark.parametrize("field,value", [
    ("horizons_ns", [True]),
    ("horizons_ns", [1_000_000_000, 1_000_000_000]),
    ("horizons_ns", [1_000_000_001]),
    ("horizons_ns", []),
    ("review_fraction", 0),
    ("review_fraction", True),
    ("grid_step_ns", 20_000_000),
    ("price_selection", True),
    ("publication_status", "confirmed_manipulation"),
])
def test_config_rejects_invalid_or_changed_semantics(field, value):
    m = api()
    bad = config()
    bad[field] = value
    with pytest.raises(ValueError):
        m.validate_config(bad)


@pytest.mark.parametrize("policy", [
    {"id": "legacy_age30", "max_age_ns": 29_000_000_000, "half_life_ns": None},
    {"id": "exp_h30", "max_age_ns": None, "half_life_ns": True},
    {"id": "exp_h30", "max_age_ns": None, "half_life_ns": 0},
    {"id": "exp_h30", "max_age_ns": 30_000_000_000, "half_life_ns": 30_000_000_000},
    {"id": "unknown", "max_age_ns": None, "half_life_ns": None},
])
def test_config_rejects_policy_relabeling(policy):
    m = api()
    bad = config()
    bad["age_policies"] = [policy]
    with pytest.raises(ValueError):
        m.validate_config(bad)


def test_config_allows_explicit_smaller_fixture_and_review_budget():
    m = api()
    c = config()
    c["horizons_ns"] = [1_000_000_000, 2_000_000_000]
    c["age_policies"] = c["age_policies"][:2]
    c["review_fraction"] = 0.2
    m.validate_config(c)


def test_validate_only_fails_missing_input_without_writes(tmp_path):
    m = api()
    c = config()
    c["sources"] = str(tmp_path / "missing-sources.json")
    cp = tmp_path / "config.json"
    cp.write_text(json.dumps(c))
    output = tmp_path / "never-created"
    before = set(tmp_path.rglob("*"))
    with pytest.raises((ValueError, FileNotFoundError)):
        m.main(["prepare", "--config", str(cp), "--instrument", "RISANAMENTO",
                "--output-root", str(output), "--validate-only"])
    assert not output.exists()
    assert set(tmp_path.rglob("*")) == before


def test_digest_syntax_is_validated_before_equality(tmp_path):
    m = api()
    p = tmp_path / "input.txt"
    p.write_text("nonsecret fixture")
    with pytest.raises(ValueError, match="digest"):
        m.check_binding(p, "not-a-sha256")
    receipt = m.check_binding(p, m.sha256(p))
    assert receipt["path"] == str(p.resolve())


@pytest.fixture
def source_bundle(tmp_path, request):
    import polars as pl
    from tests.lob.test_withdrawal_multiscale_replay import raw_fixture
    from spoofing_detection.lob.spoofing_metrics import (
        _stream_metric_inputs, _build_execution_cancel_candidates, _attach_direct_cancellation_window)
    from spoofing_detection.lob.depth_kernel_calibration import load_empirical_kernel_weights
    m = api()
    raw = tmp_path / "raw.parquet"
    raw_frame = raw_fixture(two_fills=True)
    variant = getattr(request, "param", "passive")
    if variant == "aggressive":
        raw_frame = raw_frame.with_columns(
            pl.lit("N").alias("PASSIVEORDER"),
            pl.when(pl.col("ORDEREVENTTYPE (*)") == 3).then(pl.lit(101.0)).otherwise(pl.lit(None, dtype=pl.Float64)).alias("LASTTRADEDPX"),
            pl.when(pl.col("ORDEREVENTTYPE (*)") == 3).then(pl.lit("Y")).otherwise(pl.lit("N")).alias("AGGRESSIVEORDER"))
    elif variant == "empty":
        raw_frame = raw_frame.clear()
    elif variant == "no_withdrawal":
        raw_frame = raw_frame.head(raw_frame.height - 1)
    raw_frame.write_parquet(raw)
    kernel = tmp_path / "kernel.parquet"
    pl.DataFrame([{"side": side, "rank": rank, "kernel_weight": 1.0}
                  for side in ("bid", "ask") for rank in range(1, 11)]).write_parquet(kernel)
    frames = _stream_metric_inputs(pl.read_parquet(raw), top_n=10, tick_size=1,
        empirical_kernel_weights=load_empirical_kernel_weights(kernel),
        max_deceptive_order_age_seconds=30, use_v3_clocks=True)
    links = _build_execution_cancel_candidates(frames[1], frames[3], frames[2],
        window_seconds=1.0, use_v3_clocks=True)
    source_executions = _attach_direct_cancellation_window(frames[1], frames[3],
        window_seconds=1.0, assigned_candidates=links, use_v3_clocks=True)
    paths = {}
    for name, frame in [("state_time_series", frames[0]), ("execution_metrics", source_executions),
                        ("execution_cluster_members", frames[5])]:
        paths[name] = str(tmp_path / f"source_{name}.parquet")
        frame.write_parquet(paths[name])
    metadata = {"clock_policy_id": m.CLOCK_POLICY,
        "output_schema_version": "actor_execution_anchor_v3_clocked", "top_n": 10,
        "max_rows": None, "execution_anchor_modes": ["passive", "aggressive"],
        "tick_size": 1.0, "execution_cluster_max_gap_ms": 100,
        "input": str(raw), "empirical_depth_kernel": str(kernel), "paths": paths,
        "input_hashes": {"raw_events_sha256": m.sha256(raw),
                         "empirical_depth_kernel_sha256": m.sha256(kernel)},
        "artifact_hashes": {name: m.sha256(path) for name, path in paths.items()}}
    mp = tmp_path / "metadata.json"
    mp.write_text(json.dumps(metadata))
    registry = tmp_path / "sources.json"
    registry.write_text(json.dumps({"sources": {"RISANAMENTO": {"bundle": str(tmp_path)}},
        "metadata_hashes": {"RISANAMENTO": m.sha256(mp)}}))
    witness = tmp_path / "witness.json"
    witness.write_text(json.dumps({"clock_policy_id": m.CLOCK_POLICY,
        "instruments": {"RISANAMENTO": {"partition_coverage_witness": {
            "state_time_series_sha256": metadata["artifact_hashes"]["state_time_series"],
            "segments": [{"partition_id": group.item(0, "partition_id"),
                "start_ns": group["replay_time_ns"].min(),
                "end_ns": group["replay_time_ns"].max(),
                "attested": True, "source": "synthetic raw_fixture full replay"}
                for group in frames[0].partition_by("partition_id")],
            "invalid_intervals": []}}}}))
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps({"actual_selection_config": {"coverage_witness": str(witness)},
        "inputs_and_code": {"witness": {"path": str(witness), "sha256": m.sha256(witness)}}}))
    c = config()
    c.update(sources=str(registry), baseline_manifest=str(baseline))
    cp = tmp_path / "config.json"
    cp.write_text(json.dumps(c))
    return cp, tmp_path / "prepared", metadata


def run_prepare(bundle, *flags):
    cp, output, _ = bundle
    return api().main(["prepare", "--config", str(cp), "--instrument", "RISANAMENTO",
                       "--output-root", str(output), *flags])


@pytest.mark.parametrize("source_bundle", ["passive", "aggressive"], indirect=True)
def test_prepare_real_replay_preserves_anchors_members_and_unbounded_baseline(source_bundle):
    import polars as pl
    from polars.testing import assert_frame_equal
    cp, out, metadata = source_bundle
    assert run_prepare(source_bundle) == 0
    execution = pl.read_parquet(out / "executions.parquet")
    candidates = pl.read_parquet(out / "candidates.parquet")
    assert_frame_equal(execution.select(api().EXEC_ANCHORS),
                       pl.read_parquet(metadata["paths"]["execution_metrics"]).select(api().EXEC_ANCHORS))
    assert_frame_equal(pl.read_parquet(out / "execution_cluster_members.parquet"),
                       pl.read_parquet(metadata["paths"]["execution_cluster_members"]))
    assert execution.item(0, "opposite_market_visible_qty_pre_topN") == 300
    assert execution["instrument"].to_list() == ["RISANAMENTO"]
    assert candidates["instrument"].to_list() == ["RISANAMENTO"]
    assert candidates["deceptive_order_id"].to_list() == ["OWN"]
    assert candidates.item(0, "deceptive_order_age_seconds_pre") == 60
    assert "cancel_order_first_seen_sort_index" in pl.read_parquet(out / "cancellations.parquet").columns
    receipt = api().read_json(out / "prepare_receipt.json")
    assert receipt["status"] == "prepared"
    assert receipt["identity"]["config_file"]["sha256"] == api().sha256(cp)
    assert receipt["anchor_checks"]["execution_cluster_members"] == "exact"
    assert (out / "artifact_manifest.json").is_file() and (out / "COMPLETE.json").is_file()


@pytest.mark.parametrize("protected", ["source_bundle", "raw", "config", "registry"])
def test_prepare_resume_rejects_input_overlap_before_quarantine(source_bundle, monkeypatch, protected):
    m = api()
    cp, out, metadata = source_bundle
    config_data = m.read_json(cp)
    targets = {"source_bundle": cp.parent, "raw": Path(metadata["input"]),
               "config": cp, "registry": Path(config_data["sources"])}
    target = targets[protected]
    before = snapshot(cp.parent)
    def forbidden(*args, **kwargs):
        pytest.fail("input overlap must be rejected before replay")
    monkeypatch.setattr(m, "prepare_frames", forbidden)
    with pytest.raises(ValueError, match="overlap"):
        m.main(["prepare", "--config", str(cp), "--instrument", "RISANAMENTO",
                "--output-root", str(target), "--resume"])
    assert target.exists()
    assert before == snapshot(cp.parent)


def test_resume_completed_is_read_only_and_does_not_replay(source_bundle, monkeypatch):
    run_prepare(source_bundle)
    out = source_bundle[1]
    before = {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in out.iterdir()}
    def forbidden(*args, **kwargs):
        pytest.fail("completed resume must not replay")
    monkeypatch.setattr(api(), "prepare_frames", forbidden)
    assert run_prepare(source_bundle, "--resume") == 0
    assert before == {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in out.iterdir()}
    with pytest.raises(FileExistsError):
        run_prepare(source_bundle)


@pytest.mark.parametrize("changed", ["config", "code", "raw"])
def test_resume_rejects_changed_identity(source_bundle, monkeypatch, changed):
    m = api()
    run_prepare(source_bundle)
    cp, out, metadata = source_bundle
    if changed == "config":
        cp.write_text(cp.read_text() + "\n")
    elif changed == "code":
        original = m.source_code_bindings()
        original[next(iter(original))]["sha256"] = "0" * 64
        monkeypatch.setattr(m, "source_code_bindings", lambda: original)
    else:
        with Path(metadata["input"]).open("ab") as handle:
            handle.write(b"changed")
    with pytest.raises(ValueError, match="identity|changed input"):
        run_prepare(source_bundle, "--resume")
    assert (out / "COMPLETE.json").exists()


@pytest.mark.parametrize("target", ["executions.parquet", "prepare_receipt.json", "artifact_manifest.json", "COMPLETE.json"])
def test_resume_rejects_corrupt_marked_final(source_bundle, target):
    run_prepare(source_bundle)
    (source_bundle[1] / target).write_text("{}")
    with pytest.raises(ValueError):
        run_prepare(source_bundle, "--resume")


@pytest.mark.parametrize("failure", ["marker", "rename"])
def test_interrupted_private_attempt_is_not_published_or_reused(source_bundle, monkeypatch, failure):
    m = api()
    original = m.write_json if failure == "marker" else m.os.rename
    def fail(*args, **kwargs):
        if failure == "rename" or Path(args[0]).name == "COMPLETE.json":
            raise OSError("injected interruption")
        return original(*args, **kwargs)
    with monkeypatch.context() as patcher:
        if failure == "marker":
            patcher.setattr(m, "write_json", fail)
        else:
            patcher.setattr(m.os, "rename", fail)
        with pytest.raises(OSError, match="injected"):
            run_prepare(source_bundle)
    assert not source_bundle[1].exists()
    assert run_prepare(source_bundle, "--resume") == 0
    assert (source_bundle[1] / "COMPLETE.json").exists()


def test_resume_preserves_unmarked_final_and_validate_only_never_writes(source_bundle):
    out = source_bundle[1]
    out.mkdir()
    (out / "partial.txt").write_text("preserve me")
    before = {str(p): p.read_bytes() for p in out.parent.rglob("*") if p.is_file()}
    assert run_prepare(source_bundle, "--resume", "--validate-only") == 0
    assert before == {str(p): p.read_bytes() for p in out.parent.rglob("*") if p.is_file()}
    with pytest.raises(FileExistsError):
        run_prepare(source_bundle)
    assert run_prepare(source_bundle, "--resume") == 0
    assert len(list(out.parent.glob("prepared.interrupted-*/partial.txt"))) == 1


@pytest.mark.parametrize("adversary", ["anchor", "anchor_null", "duplicate", "member_null", "member_role"])
def test_scientific_mismatch_prevents_complete_publication(source_bundle, adversary):
    import polars as pl
    m = api()
    _, out, metadata = source_bundle
    name = "execution_cluster_members" if adversary.startswith("member") else "execution_metrics"
    path = Path(metadata["paths"][name])
    frame = pl.read_parquet(path)
    if adversary == "duplicate":
        frame = pl.concat([frame, frame])
    else:
        column = {"anchor": "execution_quantity", "anchor_null": "actor_key",
                  "member_null": "child_event_id", "member_role": "execution_anchor_mode"}[adversary]
        value = {"anchor": 999.0, "member_role": "aggressive"}.get(adversary)
        frame = frame.with_columns(pl.lit(value, dtype=frame.schema[column]).alias(column))
    frame.write_parquet(path)
    metadata["artifact_hashes"][name] = m.sha256(path)
    mp = out.parent / "metadata.json"
    mp.write_text(json.dumps(metadata))
    registry = out.parent / "sources.json"
    payload = m.read_json(registry)
    payload["metadata_hashes"]["RISANAMENTO"] = m.sha256(mp)
    registry.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="anchor|member|key"):
        run_prepare(source_bundle)
    assert not out.exists()


def run_analysis(bundle, command="derive", *flags):
    cp, prepared, _ = bundle
    output = prepared.with_name("analysis")
    return api().main([command, "--config", str(cp), "--instrument", "RISANAMENTO",
        "--prepare-root", str(prepared),
        "--output-root" if command == "derive" else "--analysis-root", str(output), *flags])


def stage(bundle):
    return bundle[1].with_name(".analysis.derived")


def snapshot(root):
    return {str(p): (p.stat().st_mtime_ns, p.read_bytes()) for p in root.rglob("*") if p.is_file()}


@pytest.mark.parametrize("source_bundle", ["passive", "aggressive", "no_withdrawal"], indirect=True)
def test_derive_exact_matrix_retains_zero_and_missing_j_then_verify_publishes(source_bundle):
    import polars as pl
    run_prepare(source_bundle)
    assert run_analysis(source_bundle) == 0
    final = source_bundle[1].with_name("analysis")
    assert not final.exists()
    assert (stage(source_bundle) / "DERIVED.json").exists()
    assert not (stage(source_bundle) / "COMPLETE.json").exists()
    pop = pl.read_parquet(stage(source_bundle) / "population.parquet")
    c = api().read_json(source_bundle[0])
    n = pl.read_parquet(source_bundle[1] / "executions.parquet").height
    assert pop.height == n * len(c["horizons_ns"]) * len(c["age_policies"])
    assert pop.filter(pl.col("horizon_ns") == 1_000_000_000)["withdrawal_qty_raw"].sum() == 0
    candidates = pop.filter(pl.col("is_candidate"))
    if candidates.height:
        assert candidates["J"].null_count() == candidates.height
        assert candidates["priority_top_w_self"].all()
        assert not candidates["priority_top_j_given_self"].any()
        assert pop.filter(pl.col("age_policy") == "legacy_age30")["profile_status"].to_list() == ["computed_complete"] + ["computed_invalid"] * (len(c["horizons_ns"]) - 1)
    else:
        assert pop["profile_status"].to_list() == ["not_computed_no_withdrawal"] * pop.height
        assert pop["candidate_qty_raw"].max() > 0
    assert run_analysis(source_bundle, "verify") == 0
    assert final.exists() and not stage(source_bundle).exists()
    receipt = api().read_json(final / "verification.json")
    assert receipt["status"] == "verified"
    assert receipt["accounting"]["execution_cells_checked"] == pop.height
    before = snapshot(final)
    assert run_analysis(source_bundle, "verify") == 0
    assert run_analysis(source_bundle, "derive", "--resume") == 0
    assert before == snapshot(final)


def test_report_requires_verified_analysis_and_keeps_candidates(source_bundle):
    import polars as pl
    m = api()
    cp, prepared, _ = source_bundle
    run_prepare(source_bundle)
    run_analysis(source_bundle)
    analysis = prepared.with_name("analysis")
    report = prepared.with_name("report")
    args = ["report", "--config", str(cp), "--instrument", "RISANAMENTO",
            "--analysis-root", str(analysis), "--output-root", str(report)]
    with pytest.raises((ValueError, FileNotFoundError)):
        m.main(args)
    assert not report.exists()
    run_analysis(source_bundle, "verify")
    assert m.main(args) == 0
    pop = pl.read_parquet(analysis / "population.parquet")
    review = pl.read_csv(report / "review.csv")
    assert review.height == pop.filter(pl.col("is_candidate")).height
    assert set(m.read_json(report / "artifact_manifest.json")["artifacts"]) == {
        "population_counts.csv", "priority_counts.csv", "common_support_counts.csv",
        "score_components.csv", "review.csv", "report.md", "provenance.json"}
    assert (report / "COMPLETE.json").is_file()
    with pytest.raises(FileExistsError):
        m.main(args)


def test_report_verifier_does_not_infer_mixed_identifier_types(tmp_path):
    import polars as pl
    m = api()
    frame = pl.DataFrame({"firm_id": ['900000000000582'] * 100 + ['900000000001574_1'],
                          "client_original_id": ["0001"] * 101})
    frame.write_csv(tmp_path / "review.csv")
    m.write_json(tmp_path / "provenance.json", {"status": "report_verified",
                 "table_rows": {"review": frame.height}})
    m.write_json(tmp_path / "artifact_manifest.json", {"version": "withdrawal_multiscale_report_v1",
                 "artifacts": {name: m.sha256(tmp_path / name) for name in ["review.csv", "provenance.json"]}})
    m.write_json(tmp_path / "COMPLETE.json", {"status": "report_verified",
                 "artifact_manifest_sha256": m.sha256(tmp_path / "artifact_manifest.json")})
    assert m.verify_report(tmp_path)["table_rows"]["review"] == frame.height
    readback = pl.read_csv(tmp_path / "review.csv", infer_schema=False)
    assert readback["client_original_id"].to_list() == frame["client_original_id"].to_list()


def test_profile_batch_resume_is_read_only_with_zero_callback(source_bundle, monkeypatch):
    m = api()
    run_prepare(source_bundle)
    original = m.write_json
    def interrupted(path, payload):
        if path.name == "DERIVED.json":
            raise OSError("injected after completed profiles")
        original(path, payload)
    with monkeypatch.context() as patcher:
        patcher.setattr(m, "write_json", interrupted)
        with pytest.raises(OSError, match="injected"):
            run_analysis(source_bundle)
    batches = stage(source_bundle) / "profiles"
    before = snapshot(batches)
    assert before
    def forbidden(*args, **kwargs):
        pytest.fail("compatible completed profile batch must not recompute")
    monkeypatch.setattr(m, "compute_profiles", forbidden)
    assert run_analysis(source_bundle, "derive", "--resume") == 0
    assert before == snapshot(batches)


@pytest.mark.parametrize("field", ["WMSCI_self", "priority_top_w_self", "execution_quantity"])
def test_independent_verify_rejects_resealed_scientific_corruption(source_bundle, field):
    import polars as pl
    m = api()
    run_prepare(source_bundle)
    run_analysis(source_bundle)
    root = stage(source_bundle)
    path = root / "population.parquet"
    pop = pl.read_parquet(path)
    value = ~pl.col(field) if field.startswith("priority") else pl.col(field) + 1
    pop.with_columns(value.alias(field)).write_parquet(path)
    manifest = m.read_json(root / "derived_manifest.json")
    manifest["artifacts"]["population.parquet"] = m.sha256(path)
    m.write_json(root / "derived_manifest.json", manifest)
    marker = m.read_json(root / "DERIVED.json")
    marker["artifact_manifest_sha256"] = m.sha256(root / "derived_manifest.json")
    m.write_json(root / "DERIVED.json", marker)
    with pytest.raises(ValueError, match="mismatch"):
        run_analysis(source_bundle, "verify")
    assert not source_bundle[1].with_name("analysis").exists()


@pytest.mark.parametrize("changed", ["config", "raw", "batch"])
def test_derive_resume_rejects_changed_bindings_or_corrupt_batch(source_bundle, changed):
    run_prepare(source_bundle)
    run_analysis(source_bundle)
    if changed == "config":
        source_bundle[0].write_text(source_bundle[0].read_text() + "\n")
    elif changed == "raw":
        with Path(source_bundle[2]["input"]).open("ab") as f:
            f.write(b"changed")
    else:
        next((stage(source_bundle) / "profiles").glob("*/summary.parquet")).write_text("bad")
    with pytest.raises(ValueError):
        run_analysis(source_bundle, "derive", "--resume")
