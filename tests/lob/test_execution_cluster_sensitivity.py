from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import polars as pl
import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_execution_cluster_sensitivity.py"
METRICS_SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "compute_spoofing_metrics.py"
CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "spoofing_detection_parameters.json"


def load_module():
    spec = importlib.util.spec_from_file_location("analyze_execution_cluster_sensitivity", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_metrics_module():
    spec = importlib.util.spec_from_file_location("compute_spoofing_metrics", METRICS_SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_summarize_output_checks_cluster_and_cancellation_cardinality(tmp_path: Path):
    module = load_module()
    pl.DataFrame(
        {
            "execution_cluster_id": ["EC1", "EC2"],
            "execution_anchor_mode": ["passive", "aggressive"],
            "has_matched_deceptive_cancel_window": [True, True],
            "withdrawal_profile_scale_event": [2.0, 3.0],
            "WMSCI_passive": [4.0, None],
            "WMSCI_aggressive": [None, 6.0],
            "WMSCI_event": [-99.0, -99.0],
            "withdrawal_to_fill_ratio": [-99.0, -99.0],
        }
    ).write_parquet(tmp_path / "execution_metrics.parquet")
    pl.DataFrame(
        {
            "execution_cluster_id": ["EC1", "EC1", "EC2"],
            "child_sort_index": [1, 2, 3],
        }
    ).write_parquet(tmp_path / "execution_cluster_members.parquet")
    pl.DataFrame(
        {
            "execution_cluster_id": ["EC1", "EC2"],
            "partition_id": ["P", "P"],
            "cancel_sort_index": [10, 11],
            "cluster_last_sort_index": [2, 3],
            "candidate_order_id": ["O1", "O2"],
            "assigned_flag": [True, True],
        }
    ).write_parquet(tmp_path / "execution_cancel_candidates.parquet")

    row = module.summarize_output("Sample", 100, tmp_path)

    assert row["execution_cluster_count"] == 2
    assert row["raw_fill_message_count"] == 3
    assert row["matched_cluster_count"] == 2
    assert row["assigned_candidate_count"] == 2
    assert row["matched_cluster_count_passive"] == 1
    assert row["matched_cluster_count_aggressive"] == 1
    assert row["max_WMSCI_passive"] == 4.0
    assert row["max_WMSCI_aggressive"] == 6.0
    assert row["max_withdrawal_profile_scale_event_passive"] == 2.0
    assert row["max_withdrawal_profile_scale_event_aggressive"] == 3.0
    assert "max_WMSCI" not in row


def test_summarize_output_rejects_noncausal_cancel_candidate(tmp_path: Path):
    module = load_module()
    pl.DataFrame(
        {
            "execution_cluster_id": ["EC1"],
            "has_matched_deceptive_cancel_window": [True],
            "WMSCI_event": [4.0],
            "withdrawal_to_fill_ratio": [8.0],
        }
    ).write_parquet(tmp_path / "execution_metrics.parquet")
    pl.DataFrame(
        {"execution_cluster_id": ["EC1"], "child_sort_index": [2]}
    ).write_parquet(tmp_path / "execution_cluster_members.parquet")
    pl.DataFrame(
        {
            "execution_cluster_id": ["EC1"],
            "partition_id": ["P"],
            "cancel_sort_index": [2],
            "cluster_last_sort_index": [2],
            "candidate_order_id": ["O1"],
            "assigned_flag": [True],
        }
    ).write_parquet(tmp_path / "execution_cancel_candidates.parquet")

    with pytest.raises(ValueError, match="non-causal"):
        module.summarize_output("Sample", 100, tmp_path)


@pytest.mark.parametrize("option", ["--gap-ms", "--ga"])
def test_parse_args_rejects_cluster_sensitivity_override(option: str) -> None:
    module = load_module()
    with pytest.raises(SystemExit):
        module.parse_args(
            [
                "--input",
                "events.parquet",
                "--instrument",
                "AIR",
                "--output-root",
                "out",
                "--config",
                str(CONFIG_PATH),
                option,
                "50",
            ]
        )


@pytest.mark.parametrize(
    "changed_input_hash",
    ["raw_events_sha256", "quote_panel_sha256", "empirical_depth_kernel_sha256"],
)
def test_reuse_requires_exact_config_and_input_hashes(
    tmp_path: Path,
    changed_input_hash: str,
) -> None:
    module = load_module()
    effective_config = tmp_path / "effective_config.json"
    module._write_effective_config(
        CONFIG_PATH,
        gap_ms=50,
        destination=effective_config,
    )
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    for name in (
        "execution_metrics.parquet",
        "execution_cluster_members.parquet",
        "execution_cancel_candidates.parquet",
    ):
        (output_dir / name).touch()
    metadata_path = output_dir / "metadata.json"
    metadata = {
        "parameter_source": "json_config_only",
        "config_section": "metrics",
        "config_sha256": module._sha256(effective_config),
        "execution_cluster_max_gap_ms": 50,
        "input_hashes": {
            "raw_events_sha256": "input-hash",
            "quote_panel_sha256": "quote-hash",
            "empirical_depth_kernel_sha256": "kernel-hash",
        },
        "artifact_hashes": {
            "execution_metrics": module._sha256(output_dir / "execution_metrics.parquet"),
            "execution_cluster_members": module._sha256(output_dir / "execution_cluster_members.parquet"),
            "execution_cancel_candidates": module._sha256(output_dir / "execution_cancel_candidates.parquet"),
        },
    }
    metadata_path.write_text(json.dumps(metadata))

    expected_hashes = {
        "input_sha256": "input-hash",
        "quote_panel_sha256": "quote-hash",
        "empirical_depth_kernel_sha256": "kernel-hash",
    }
    assert module._can_reuse_output(
        output_dir,
        effective_config=effective_config,
        gap_ms=50,
        **expected_hashes,
    )

    metadata_path.write_text(json.dumps({**metadata, "config_sha256": "stale"}))
    assert not module._can_reuse_output(
        output_dir,
        effective_config=effective_config,
        gap_ms=50,
        **expected_hashes,
    )

    metadata_path.write_text(
        json.dumps(
            {
                **metadata,
                "input_hashes": {
                    **metadata["input_hashes"],
                    changed_input_hash: "stale",
                },
            }
        )
    )
    assert not module._can_reuse_output(
        output_dir,
        effective_config=effective_config,
        gap_ms=50,
        **expected_hashes,
    )


def test_reuse_rejects_artifacts_that_do_not_match_child_metadata(tmp_path: Path) -> None:
    module = load_module()
    effective_config = tmp_path / "effective_config.json"
    module._write_effective_config(CONFIG_PATH, gap_ms=50, destination=effective_config)
    output_dir = tmp_path / "run"
    output_dir.mkdir()
    artifact_paths = {
        "execution_metrics": output_dir / "execution_metrics.parquet",
        "execution_cluster_members": output_dir / "execution_cluster_members.parquet",
        "execution_cancel_candidates": output_dir / "execution_cancel_candidates.parquet",
    }
    for path in artifact_paths.values():
        path.write_bytes(b"artifact")
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "parameter_source": "json_config_only",
                "config_section": "metrics",
                "config_sha256": module._sha256(effective_config),
                "execution_cluster_max_gap_ms": 50,
                "input_hashes": {
                    "raw_events_sha256": "input-hash",
                    "quote_panel_sha256": None,
                    "empirical_depth_kernel_sha256": None,
                },
                "artifact_hashes": {
                    **{key: module._sha256(path) for key, path in artifact_paths.items()},
                    "execution_metrics": "stale",
                },
            }
        )
    )

    assert not module._can_reuse_output(
        output_dir,
        effective_config=effective_config,
        gap_ms=50,
        input_sha256="input-hash",
        quote_panel_sha256=None,
        empirical_depth_kernel_sha256=None,
    )


def test_build_command_passes_only_derived_config_with_operational_cluster_gap(tmp_path: Path):
    module = load_module()
    source_config = tmp_path / "config.json"
    source_config.write_text(CONFIG_PATH.read_text())
    args = module.parse_args(
        [
            "--input",
            str(tmp_path / "input.parquet"),
            "--instrument",
            "Sample",
            "--output-root",
            str(tmp_path / "out"),
            "--quote-panel",
            str(tmp_path / "quotes.parquet"),
            "--config",
            str(source_config),
        ]
    )

    effective_config = tmp_path / "run" / "effective_config.json"
    effective_config.parent.mkdir()
    module._write_effective_config(source_config, gap_ms=50, destination=effective_config)
    command = module.build_command(args, effective_config=effective_config, output_dir=tmp_path / "run")
    child_args = load_metrics_module().parse_args(command[2:])

    assert command[command.index("--config") + 1] == str(effective_config)
    assert command[command.index("--quote-panel") + 1] == str(tmp_path / "quotes.parquet")
    assert child_args.config == effective_config
    assert child_args.quote_panel == tmp_path / "quotes.parquet"
    assert child_args.execution_cluster_max_gap_ms == 50
    assert "--execution-cluster-max-gap-ms" not in command
    assert "--empirical-depth-kernel" not in command
    assert "--compact-state" not in command
    assert json.loads(effective_config.read_text())["metrics"]["execution_cluster_max_gap_ms"] == 50
    assert json.loads(source_config.read_text())["metrics"]["execution_cluster_max_gap_ms"] == 100
    assert args.gap_ms == [25, 50, 100, 250]


def test_effective_metrics_are_self_contained_and_exclude_comment_keys(tmp_path: Path):
    module = load_module()
    effective_config = tmp_path / "effective_config.json"
    module._write_effective_config(CONFIG_PATH, gap_ms=50, destination=effective_config)

    effective_metrics = module._effective_metrics(effective_config)

    assert effective_metrics["top_n"] == 10
    assert effective_metrics["execution_cluster_max_gap_ms"] == 50
    assert not any(key.startswith("_comment_") for key in effective_metrics)


def test_main_embeds_effective_metrics_and_input_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = load_module()
    input_path = tmp_path / "input.parquet"
    quote_panel = tmp_path / "quotes.parquet"
    input_path.write_bytes(b"input")
    quote_panel.write_bytes(b"quotes")
    commands: list[list[str]] = []
    monkeypatch.setattr(module.subprocess, "run", lambda command, check: commands.append(command))
    monkeypatch.setattr(
        module,
        "summarize_output",
        lambda instrument, gap_ms, output_dir: {
            "instrument": instrument,
            "execution_cluster_max_gap_ms": gap_ms,
            "raw_fill_message_count": 1,
        },
    )
    output_root = tmp_path / "sensitivity"

    module.main(
        [
            "--input",
            str(input_path),
            "--quote-panel",
            str(quote_panel),
            "--instrument",
            "Sample",
            "--output-root",
            str(output_root),
            "--config",
            str(CONFIG_PATH),
        ]
    )

    metadata = json.loads((output_root / "metadata.json").read_text())
    assert metadata["input_sha256"] == module._sha256(input_path)
    assert metadata["quote_panel_sha256"] == module._sha256(quote_panel)
    assert len(metadata["effective_configs"]) == 4
    effective_gaps = [
        item["effective_metrics"]["execution_cluster_max_gap_ms"]
        for item in metadata["effective_configs"]
    ]
    assert effective_gaps == [25, 50, 100, 250]
    assert all(item["effective_metrics"]["top_n"] == 10 for item in metadata["effective_configs"])
    assert all("--quote-panel" in command for command in commands)
