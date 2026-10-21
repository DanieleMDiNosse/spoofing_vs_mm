#!/usr/bin/env python3
"""Derive conditional represented-event support from protected historical caches.

No raw parquet or book replay is opened. A sidecar is not an authentication
signature: historical cache/manifest integrity remains an explicit assumption.
"""
from __future__ import annotations

import argparse
from collections.abc import Mapping
import json
from pathlib import Path
import sys
from typing import Any

import polars as pl

from scripts import study_msci_grid as pre
from spoofing_detection.lob.msci_grid import _coverage_for_partition
from spoofing_detection.lob.panel import SORT_COLUMNS, _partition_id, _preserve_source_clock_nanoseconds
from spoofing_detection.lob.spoofing_clock import resolve_spoofing_clocks

SCHEMA = "msci_grid_observed_event_coverage_witness_v1"
ROOT = Path(__file__).resolve().parents[1]
SEMANTICS = {"event_boundary_evaluated": True, "transition_zero_emitted": True,
             "sparse_zero_until_reentry": True, "state_client_mode": "execution-actors"}
LIMITATION = ("No historical normalized artifact digest is available. Current cache bytes are frozen, "
              "not independently authenticated against historical bytes. Protected historical cache, "
              "checkpoint and preflight integrity is assumed; raw-to-cache derivation is not replayed. "
              "External physical feed completeness and UTC documentation remain unverified.")


def _bind(path: Path, expected: Any = None, *, label: str) -> dict[str, str]:
    path = path.expanduser().resolve(strict=True)
    actual = pre.sha256(path) if expected is None else pre._verify_declared_hash(path, expected, label=label)
    return {"path": str(path), "sha256": actual}


def _entry_binding(entry: Mapping[str, Any], *, label: str) -> dict[str, str]:
    return _bind(Path(entry["path"]), entry["sha256"], label=label)


def _count(value: Any, label: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{label} count must be a nonnegative integer")
    return value


def _adapter_sources() -> dict[str, dict[str, str]]:
    paths = [Path(__file__), ROOT / "scripts/study_msci_grid.py",
             ROOT / "src/spoofing_detection/lob/panel.py",
             ROOT / "src/spoofing_detection/lob/spoofing_clock.py",
             ROOT / "src/spoofing_detection/lob/empirical_clock.py",
             ROOT / "src/spoofing_detection/lob/enums.py"]
    return {str(path.relative_to(ROOT)): _bind(path, label="adapter source") for path in paths}


def scan_observed_events(normalized_path: Path, state_path: Path, *, expected_count: int) -> dict[str, Any]:
    """Project events; use exact shared row semantics; join every unique H1 anchor.

    Only normalized projected rows are iterated in Python. The much larger state
    panel remains a projected lazy unique/anti-join; no state row dicts are built.
    Memory scales with represented events and distinct anchors, not actor rows.
    """
    expected_count = _count(expected_count, "normalized")
    schema = dict(pl.scan_parquet(normalized_path).collect_schema())
    required = set(SORT_COLUMNS[:6]) | {"sort_index", "event_class"}
    if missing := sorted(required - set(schema)):
        raise ValueError(f"normalized events missing canonical columns: {missing}")
    pre._require_non_boolean_integral(schema, {"sort_index"}, label="normalized indices")
    columns = [name for name in SORT_COLUMNS if name in schema]
    projected = list(dict.fromkeys(["sort_index", "event_class", *columns,
                                   *[name for name in ("ORDEREVENTTYPE (*)",) if name in schema]]))
    events = pl.scan_parquet(normalized_path).select(projected).collect(engine="streaming")
    if not events.height or events.height != expected_count:
        raise ValueError("observed event count differs from full normalized metadata count (or empty stream)")
    indices = events.get_column("sort_index")
    if indices.null_count() or indices.to_list() != list(range(1, events.height + 1)):
        raise ValueError("original global indices must equal exactly 1..N in physical canonical order")
    if not events.sort(columns, nulls_last=True).get_column("sort_index").equals(indices):
        raise ValueError("canonical re-sort changes original event indices")
    events = _preserve_source_clock_nanoseconds(events)
    partitions: dict[str, dict[str, Any]] = {}
    pids: list[str] = []
    clocks: list[int] = []
    previous_pid = None
    for row in events.iter_rows(named=True):
        decision = resolve_spoofing_clocks(row)
        clock = decision.replay_time_ns
        if decision.incompatibilities or type(clock) is not int:
            raise ValueError(f"incompatible or missing replay clock at index {row['sort_index']}: {decision.incompatibilities}")
        pid = _partition_id(row)
        assert pid is not None
        index = row["sort_index"]
        if pid in partitions:
            part = partitions[pid]
            if pid != previous_pid or index != part["last_sort_index"] + 1:
                raise ValueError("partition original indices are noncontiguous")
            if clock < part["end_ns"]:
                raise ValueError("replay clock regression in canonical partition")
            part.update(last_sort_index=index, end_ns=clock, event_count=part["event_count"] + 1)
        else:
            partitions[pid] = {"partition_id": pid, "event_count": 1, "first_sort_index": index,
                               "last_sort_index": index, "start_ns": clock, "end_ns": clock}
        previous_pid = pid
        pids.append(pid)
        clocks.append(clock)
    anchors = pl.DataFrame({"partition_id": pids, "sort_index": indices, "replay_time_ns": clocks})
    state_schema = dict(pl.scan_parquet(state_path).collect_schema())
    pre._require_non_boolean_integral(state_schema, {"sort_index", "replay_time_ns"}, label="H1 state anchors")
    states = pl.scan_parquet(state_path).select("partition_id", "sort_index", "replay_time_ns").unique()
    state_partitions = set(states.select("partition_id").unique().collect(engine="streaming").get_column("partition_id").to_list())
    if state_partitions != set(partitions):
        raise ValueError("exact H1 state/event partition-ID sets disagree")
    unmatched = states.join(anchors.lazy(), on=["partition_id", "sort_index", "replay_time_ns"], how="anti").select(pl.len()).collect(engine="streaming").item()
    if unmatched:
        raise ValueError(f"{unmatched} distinct H1 state anchors do not match exact observed event anchors")
    distinct = states.select(pl.len()).collect(engine="streaming").item()
    return {"event_count": events.height, "partitions": list(partitions.values()),
            "distinct_state_anchors": distinct, "unmatched_state_anchors": 0,
            "missing_replay_clocks": 0, "clock_regressions": 0, "canonical_order_verified": True}


def _historical_bindings(entry: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    evidence = entry.get("coverage_historical_evidence")
    if not isinstance(evidence, Mapping) or evidence.get("schema") != "msci_grid_historical_h1_checkpoint_evidence_v1" or evidence.get("semantics") != SEMANTICS:
        raise ValueError("sidecar requires authoritative historical preflight producer evidence")
    bindings = {"source_metadata": _entry_binding(entry["metadata"], label="source metadata"),
                "checkpoint_identity": _bind(Path(evidence["identity"]["path"]), label="checkpoint identity"),
                "historical_config": _entry_binding(evidence["config"], label="historical config")}
    for key in ("state_time_series", "execution_metrics"):
        bindings[key] = _entry_binding(entry["artifacts"][key], label=key)
    metadata = pre._read_json(Path(bindings["source_metadata"]["path"]), label="source metadata")
    identity = pre._read_json(Path(bindings["checkpoint_identity"]["path"]), label="checkpoint identity")
    if identity.get("format") != pre.HISTORICAL_CHECKPOINT_SCHEMA:
        raise ValueError("checkpoint identity format mismatch")
    arguments = identity.get("resolved_compute_args", {})
    expected = {"clock_policy_v3": True, "top_n": 10, "window_seconds": 1.0,
                "compact_state": True, "state_client_mode": "execution-actors", "max_rows": None}
    if any(key not in arguments or arguments[key] != value for key, value in expected.items()):
        raise ValueError("checkpoint identity lacks uncapped v3 execution-actor semantics")
    if metadata.get("clock_policy_id") != pre.EXPECTED_CLOCK_POLICY or metadata.get("state_client_mode") != "execution-actors" or metadata.get("compact_state") is not True:
        raise ValueError("historical source metadata semantics mismatch")
    config = identity["inputs"]["config"]
    if _entry_binding(config, label="checkpoint config") != bindings["historical_config"] or metadata.get("config_sha256") != config["sha256"] or metadata.get("input_hashes", {}).get("config_sha256") != config["sha256"]:
        raise ValueError("checkpoint config identity does not bind current metadata/preflight")
    raw = identity["inputs"]["raw"]
    raw_sha = pre._require_sha256(raw.get("sha256"), label="raw")
    if evidence["raw_input"].get("sha256") != raw_sha or metadata.get("input_hashes", {}).get("raw_events_sha256") != raw_sha:
        raise ValueError("raw identity does not bind metadata/preflight")
    for key in ("state_time_series", "execution_metrics"):
        path = pre._confined_path(metadata["paths"][key], root=Path(bindings["source_metadata"]["path"]).parent, label=key)
        if str(path) != bindings[key]["path"] or metadata["artifact_hashes"].get(key) != bindings[key]["sha256"]:
            raise ValueError(f"current {key} does not bind authoritative preflight identity")
        observed = pl.scan_parquet(path).select(pl.len()).collect(engine="streaming").item()
        if observed != _count(metadata["row_counts"][key], key) or observed != _count(entry["declared_row_counts"][key], key):
            raise ValueError(f"{key} row count mismatch")
    replay = evidence["replay_frame_0"]
    if replay["sha256"] != bindings["state_time_series"]["sha256"] or replay["row_count"] != metadata["row_counts"]["state_time_series"]:
        raise ValueError("historical replay state binding mismatch")
    bindings["replay_manifest"] = _bind(Path(evidence["checkpoint"]) / "replay/manifest.json", label="replay manifest")
    manifest = pre._read_json(Path(bindings["replay_manifest"]["path"]), label="replay manifest")
    if manifest.get("format") != pre.HISTORICAL_REPLAY_SCHEMA or manifest.get("stage") != "replay" or manifest.get("frames", [{}])[0].get("sha256") != replay["sha256"] or manifest["frames"][0].get("row_count") != replay["row_count"]:
        raise ValueError("replay manifest does not bind authoritative state identity")
    # Replay frame bytes equal the independently freshly hashed consumed state;
    # do not hash a second multi-GB copy merely to infer market completeness.
    log = pre._successful_run_log(Path(bindings["source_metadata"]["path"]).parent, instrument=entry["instrument"], historical_root=Path(evidence["run_log"]).parents[2])
    bindings["run_log"] = _bind(log, label="historical run log")
    source_hashes = identity.get("source_sha256", {})
    if not set(pre.HISTORICAL_REQUIRED_SOURCES) <= set(source_hashes):
        raise ValueError("checkpoint identity missing required producer source bindings")
    for relative, expected_sha in source_hashes.items():
        path = (ROOT / relative).resolve()
        path.relative_to(ROOT)
        pre._require_sha256(expected_sha, label=f"historical source {relative}")
        if path.is_file():
            bindings[f"source:{relative}"] = _bind(path, expected_sha, label=f"historical source {relative}")
        elif relative in pre.HISTORICAL_REQUIRED_SOURCES:
            raise ValueError(f"required historical producer source missing: {relative}")
    for relative, key in pre.HISTORICAL_REQUIRED_SOURCES.items():
        recorded = metadata.get("generator_sha256") if key is None else metadata.get("scientific_source_hashes", {}).get(key)
        if recorded != source_hashes[relative] or evidence["identity"]["source_bindings"].get(relative, {}).get("sha256") != source_hashes[relative]:
            raise ValueError("historical producer source identity disagrees with preflight/metadata")
    return bindings, identity


def build_sidecar(preflight_path: Path, cache_metadata: Mapping[str, Path], *,
                  assume_protected_cache: bool, invalid_intervals_path: Path | None = None) -> dict[str, Any]:
    if assume_protected_cache is not True:
        raise ValueError("explicit protected historical cache integrity assumption is required")
    report = pre._read_json(preflight_path, label="preflight")
    if report.get("format") != "msci_grid_preflight_v1" or report.get("status") != "ready_for_task6_cohort_gate":
        raise ValueError("authoritative ready historical preflight required")
    entries = report.get("instruments", [])
    names = [entry["instrument"] for entry in entries]
    if len(set(names)) != len(names) or not names or set(names) != set(cache_metadata):
        raise ValueError("cache metadata must cover exactly the historical preflight instruments")
    gaps = pre._read_json(invalid_intervals_path, label="known invalid intervals") if invalid_intervals_path else {}
    if not set(gaps) <= set(names):
        raise ValueError("known invalid intervals contain an unknown instrument")
    result: dict[str, Any] = {"schema": SCHEMA, "support_target": "represented_event_stream",
        "external_feed_completeness_verified": False, "clock_policy_id": pre.EXPECTED_CLOCK_POLICY,
        "protected_cache_integrity_assumed": True, "integrity_limitation": LIMITATION,
        "preflight": _bind(preflight_path, label="preflight"), "adapter_sources": _adapter_sources(),
        "invalid_intervals_input": _bind(invalid_intervals_path, label="known invalid intervals") if invalid_intervals_path else None,
        "instruments": {}}
    for entry in entries:
        name = entry["instrument"]
        bindings, identity = _historical_bindings(entry)
        source_digest = report.get("input_digests", {}).get(f"source_{name}")
        if source_digest != entry["metadata"]:
            raise ValueError("preflight source metadata identity mismatch")
        cachemeta = cache_metadata[name].expanduser().resolve(strict=True)
        cache = pre._read_json(cachemeta, label="normalized cache metadata")
        if cache.get("config", {}).get("include_all_events") is not True or "max_rows" not in cache or cache["max_rows"] is not None:
            raise ValueError("normalized cache must represent uncapped include_all_events input")
        raw = identity["inputs"]["raw"]
        raw_sha = pre._require_sha256(cache.get("source_file_sha256"), label="cache raw")
        raw_path = Path(cache["source_file"]).expanduser()
        raw_path = raw_path if raw_path.is_absolute() else ROOT / raw_path
        if raw_sha != raw["sha256"] or raw_path.resolve() != Path(raw["path"]).expanduser().resolve():
            raise ValueError("cache raw identity differs from authoritative checkpoint input")
        counts = [cache.get("row_counts", {}).get("input"), cache.get("row_counts", {}).get("normalized_events"), cache.get("validation", {}).get("events_processed")]
        counts = [_count(value, "cache") for value in counts]
        if len(set(counts)) != 1:
            raise ValueError("cache input/normalized/processed count mismatch")
        raw_normalized = Path(cache["outputs"]["normalized_events"]).expanduser()
        normalized = raw_normalized if raw_normalized.is_absolute() else ROOT / raw_normalized
        normalized = normalized.resolve(strict=True)
        normalized.relative_to(cachemeta.parent)
        bindings["cache_metadata"] = _bind(cachemeta, label="cache metadata")
        bindings["normalized_events"] = _bind(normalized, label="normalized events")
        historical_digest = cache.get("artifact_hashes", {}).get("normalized_events")
        if historical_digest is not None:
            pre._verify_declared_hash(normalized, historical_digest, label="historical normalized cache")
        observed = scan_observed_events(normalized, Path(bindings["state_time_series"]["path"]), expected_count=counts[0])
        if cache.get("validation", {}).get("partitions_processed") != len(observed["partitions"]):
            raise ValueError("cache partition count mismatch")
        legacy = entry["coverage_historical_evidence"].get("partition_coverage_witness", {})
        if legacy.get("segments"):
            raise ValueError("sidecar cannot override existing historical support segments")
        invalid = [*legacy.get("invalid_intervals", []), *gaps.get(name, [])]
        segments = [{"partition_id": part["partition_id"], "start_ns": part["start_ns"],
                     "end_ns": part["end_ns"], "attested": True, "source": SCHEMA} for part in observed["partitions"]]
        coverage = {"state_time_series_sha256": bindings["state_time_series"]["sha256"],
                    "segments": segments, "invalid_intervals": invalid}
        partitions = {part["partition_id"] for part in observed["partitions"]}
        if any(not isinstance(gap, Mapping) or gap.get("partition_id") not in partitions for gap in invalid):
            raise ValueError("known gap partition identity is not represented")
        for pid in partitions:
            _coverage_for_partition(coverage, pid)
        result["instruments"][name] = {"bindings": bindings, "semantics": SEMANTICS,
            "artifacts": {key: bindings[key] for key in ("state_time_series", "execution_metrics")},
            "historical_normalized_sha256": historical_digest, "observed": observed,
            "partition_coverage_witness": coverage}
    return result


def load_sidecar(path: Path, preflight_path: Path) -> dict[str, Any]:
    """Re-derive before use: a caller-supplied attested boolean is not authority.

    This costs a projected scan and fresh byte hashes, but no raw book replay.
    Any change to evidence, bounds, gaps, helpers or consumed artifacts fails.
    """
    witness = pre._read_json(path, label="coverage witness sidecar")
    if witness.get("schema") != SCHEMA or witness.get("external_feed_completeness_verified") is not False or witness.get("support_target") != "represented_event_stream":
        raise ValueError("invalid observed-stream sidecar scope/schema")
    if witness.get("preflight") != _bind(preflight_path, label="current preflight"):
        raise ValueError("coverage sidecar/preflight identity mismatch")
    for entry in witness.get("instruments", {}).values():
        for key, binding in entry.get("bindings", {}).items():
            _entry_binding(binding, label=f"sidecar {key}")
    cache_metadata = {name: Path(entry["bindings"]["cache_metadata"]["path"]) for name, entry in witness["instruments"].items()}
    gaps = witness.get("invalid_intervals_input")
    if gaps:
        _entry_binding(gaps, label="sidecar invalid intervals")
    fresh = build_sidecar(preflight_path, cache_metadata,
                          assume_protected_cache=witness.get("protected_cache_integrity_assumed"),
                          invalid_intervals_path=Path(gaps["path"]) if gaps else None)
    if witness != fresh:
        raise ValueError("coverage sidecar differs from freshly derived current-input witness")
    return witness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-report", type=Path, required=True)
    parser.add_argument("--cache-metadata", action="append", required=True, metavar="INSTRUMENT=PATH")
    parser.add_argument("--invalid-intervals", type=Path)
    parser.add_argument("--assume-protected-cache", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.output.exists():
            raise ValueError("output sidecar already exists; refusing overwrite")
        caches: dict[str, Path] = {}
        for item in args.cache_metadata:
            name, separator, value = item.partition("=")
            if not separator or not name or not value or name in caches:
                raise ValueError("cache metadata requires unique INSTRUMENT=PATH entries")
            caches[name] = Path(value)
        witness = build_sidecar(args.preflight_report, caches, assume_protected_cache=args.assume_protected_cache,
                                invalid_intervals_path=args.invalid_intervals)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            handle.write(json.dumps(witness, sort_keys=True, indent=2) + "\n")
        if pre._read_json(args.output, label="published sidecar") != witness:
            raise ValueError("published sidecar readback differs")
        print(f"MSCI_GRID_COVERAGE_WITNESS_CREATED {args.output} sha256={pre.sha256(args.output)}")
        return 0
    except (OSError, ValueError, KeyError, TypeError, pl.exceptions.PolarsError) as exc:
        print(f"MSCI_GRID_COVERAGE_WITNESS_INVALID {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
