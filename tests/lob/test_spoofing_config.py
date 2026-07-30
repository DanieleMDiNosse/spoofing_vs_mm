from __future__ import annotations

import json
from pathlib import Path

import pytest

from spoofing_detection.lob.spoofing_config import (
    load_spoofing_config_defaults,
    parse_msci_threshold_by_anchor,
)


def test_config_loader_ignores_adjacent_comment_keys(tmp_path: Path):
    config_path = tmp_path / "spoofing.json"
    config_path.write_text(
        json.dumps(
            {
                "metrics": {
                    "top_n": 5,
                    "_comment_top_n": "Number of visible book levels.",
                }
            }
        )
    )

    assert load_spoofing_config_defaults(
        config_path=config_path,
        section="metrics",
        allowed_keys={"top_n"},
    ) == {"top_n": 5}


def test_config_loader_still_rejects_unknown_parameter_keys(tmp_path: Path):
    config_path = tmp_path / "spoofing.json"
    config_path.write_text(json.dumps({"metrics": {"top_m": 5}}))

    with pytest.raises(ValueError, match="unknown key `top_m`"):
        load_spoofing_config_defaults(
            config_path=config_path,
            section="metrics",
            allowed_keys={"top_n"},
        )


def test_threshold_parser_ignores_adjacent_comment_keys():
    assert parse_msci_threshold_by_anchor(
        {
            "passive": 0.1,
            "_comment_passive": "Calibrated passive cutoff.",
            "aggressive": None,
            "_comment_aggressive": "No calibrated aggressive cutoff.",
        }
    ) == {"passive": 0.1, "aggressive": None}


def test_threshold_parser_rejects_map_containing_only_comments():
    with pytest.raises(ValueError, match="non-empty JSON object"):
        parse_msci_threshold_by_anchor({"_comment_passive": "Documentation only."})
