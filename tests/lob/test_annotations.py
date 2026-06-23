from __future__ import annotations

import polars as pl
import pytest

from spoofing_detection.lob.annotations import (
    ALLOWED_EVENT_LABELS,
    AnnotationValidationError,
    default_annotation_frame,
    validate_annotations,
)


def test_default_annotation_frame_has_expected_columns():
    frame = default_annotation_frame()
    assert frame.columns == [
        "review_event_id",
        "analyst_label",
        "confidence",
        "benign_explanation",
        "notes",
        "reviewer",
        "reviewed_at_utc",
    ]


def test_validate_annotations_accepts_known_labels():
    frame = pl.DataFrame(
        {
            "review_event_id": ["S1"],
            "analyst_label": ["weak_spoofing_like"],
            "confidence": [0.7],
            "benign_explanation": ["quote_refresh"],
            "notes": ["fast cancel but low MSCI"],
            "reviewer": ["analyst"],
            "reviewed_at_utc": ["2026-06-23T10:00:00Z"],
        }
    )
    validated = validate_annotations(frame)
    assert validated.height == 1
    assert "weak_spoofing_like" in ALLOWED_EVENT_LABELS


def test_validate_annotations_rejects_unknown_label():
    frame = pl.DataFrame(
        {
            "review_event_id": ["S1"],
            "analyst_label": ["definitely_illegal"],
            "confidence": [0.9],
            "benign_explanation": [""],
            "notes": [""],
            "reviewer": ["analyst"],
            "reviewed_at_utc": ["2026-06-23T10:00:00Z"],
        }
    )
    with pytest.raises(AnnotationValidationError):
        validate_annotations(frame)


def test_validate_annotations_rejects_confidence_outside_unit_interval():
    frame = pl.DataFrame(
        {
            "review_event_id": ["S1"],
            "analyst_label": ["legitimate_market_making"],
            "confidence": [1.2],
            "benign_explanation": ["inventory_rebalancing"],
            "notes": [""],
            "reviewer": ["analyst"],
            "reviewed_at_utc": ["2026-06-23T10:00:00Z"],
        }
    )
    with pytest.raises(AnnotationValidationError):
        validate_annotations(frame)
