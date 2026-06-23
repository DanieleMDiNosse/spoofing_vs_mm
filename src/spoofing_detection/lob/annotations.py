from __future__ import annotations

import polars as pl

ALLOWED_EVENT_LABELS = {
    "strong_spoofing_like",
    "moderate_spoofing_like",
    "weak_spoofing_like",
    "legitimate_market_making",
    "quote_refresh",
    "inventory_rebalancing",
    "stale_quote_cancel",
    "unclear_needs_more_context",
}

ALLOWED_BENIGN_EXPLANATIONS = {
    "",
    "quote_refresh",
    "inventory_rebalancing",
    "adverse_selection_response",
    "stale_quote_cancel",
    "market_wide_move",
    "other",
}

ANNOTATION_COLUMNS = [
    "review_event_id",
    "analyst_label",
    "confidence",
    "benign_explanation",
    "notes",
    "reviewer",
    "reviewed_at_utc",
]


class AnnotationValidationError(ValueError):
    """Raised when analyst annotation files violate the expected schema."""


def default_annotation_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "review_event_id": pl.Utf8,
            "analyst_label": pl.Utf8,
            "confidence": pl.Float64,
            "benign_explanation": pl.Utf8,
            "notes": pl.Utf8,
            "reviewer": pl.Utf8,
            "reviewed_at_utc": pl.Utf8,
        }
    )


def validate_annotations(frame: pl.DataFrame) -> pl.DataFrame:
    missing = [column for column in ANNOTATION_COLUMNS if column not in frame.columns]
    if missing:
        raise AnnotationValidationError(f"missing annotation columns: {missing}")
    frame = frame.select(ANNOTATION_COLUMNS).with_columns(
        [
            pl.col("review_event_id").cast(pl.Utf8),
            pl.col("analyst_label").cast(pl.Utf8),
            pl.col("confidence").cast(pl.Float64),
            pl.col("benign_explanation").fill_null("").cast(pl.Utf8),
            pl.col("notes").fill_null("").cast(pl.Utf8),
            pl.col("reviewer").fill_null("").cast(pl.Utf8),
            pl.col("reviewed_at_utc").fill_null("").cast(pl.Utf8),
        ]
    )
    unknown_labels = sorted(set(frame.get_column("analyst_label").drop_nulls().to_list()) - ALLOWED_EVENT_LABELS)
    if unknown_labels:
        raise AnnotationValidationError(f"unknown analyst labels: {unknown_labels}")
    unknown_benign = sorted(
        set(frame.get_column("benign_explanation").fill_null("").to_list()) - ALLOWED_BENIGN_EXPLANATIONS
    )
    if unknown_benign:
        raise AnnotationValidationError(f"unknown benign explanations: {unknown_benign}")
    invalid_confidence = frame.filter((pl.col("confidence") < 0.0) | (pl.col("confidence") > 1.0))
    if invalid_confidence.height:
        raise AnnotationValidationError("confidence must be in [0, 1]")
    if frame.get_column("review_event_id").is_duplicated().any():
        raise AnnotationValidationError("review_event_id must be unique in annotation file")
    return frame
