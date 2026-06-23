# Production-Ready Spoofing Detection Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Turn the current event-level spoofing review prototype into a calibrated, analyst-reviewable, client-session surveillance pipeline that separates spoofing-like behavior from legitimate market-making patterns with explicit labels, baselines, negative controls, and false-positive management.

**Architecture:** Keep the current event-level DWI/SCI/MSCI/MCPS pipeline and dashboard as forensic drill-down evidence. Add three layers on top: (1) analyst labels saved as structured artifacts, (2) client-session aggregate and legitimacy/baseline features, and (3) calibration/negative-control reports that choose thresholds by false-positive workload rather than by ad hoc score inspection. LLM reviews remain explanatory artifacts, not detection labels.

**Tech Stack:** Python 3.11, Polars, pytest, static HTML dashboard with embedded JavaScript, existing `spoofing_detection.lob` package, existing parquet outputs under `outputs/spoofing_*`, no new runtime dependencies unless a later task proves they are necessary.

---

## Current context and assumptions

- Existing detector outputs are event-level artifacts:
  - `outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/execution_metrics.parquet`
  - `outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/candidate_deceptive_orders.parquet`
  - `outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_events.parquet`
  - `outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_lob_queue.parquet`
  - `outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_event_review_dashboard.html`
- The dashboard already supports event selection, kappa/lambda selection, LOB reconstruction, and precomputed LLM reviews.
- The current model is scientifically best treated as a spoofing-like event detector, not as a legal intent classifier.
- Production readiness means:
  - explicit analyst labels;
  - client/session-level repeated-pattern alerts;
  - market-maker legitimacy baselines;
  - negative controls/placebo checks;
  - threshold calibration with false-positive/workload reporting;
  - reproducible outputs and tests.
- Do not add a database or web backend in this iteration. Use CSV/JSON/parquet files and static dashboard integration first.
- Do not make the LLM determine labels. LLM summaries can be displayed next to observed facts and analyst labels.

---

## Proposed implementation phases

1. Add analyst annotation support to collect labels and review notes.
2. Add client-session aggregate surveillance features.
3. Add legitimate market-maker baseline features.
4. Add negative controls/placebo tests.
5. Add calibration/reporting to select operating thresholds.
6. Promote production alerts from event-level rows to client-session alert objects.
7. Integrate all of the above into the dashboard and docs.

---

## Task 1: Add annotation schema and validation helpers

**Objective:** Define a stable event-label schema so analyst decisions can be saved and validated reproducibly.

**Files:**
- Create: `src/spoofing_detection/lob/annotations.py`
- Test: `tests/lob/test_annotations.py`

**Step 1: Write failing tests**

Create `tests/lob/test_annotations.py`:

```python
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
    frame = default_annotation_frame().vstack(
        pl.DataFrame(
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
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_annotations.py
```

Expected: FAIL because `spoofing_detection.lob.annotations` does not exist.

**Step 3: Implement annotation helpers**

Create `src/spoofing_detection/lob/annotations.py`:

```python
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

    unknown_labels = sorted(set(frame.get_column("analyst_label").drop_nulls().to_list()) - ALLOWED_EVENT_LABELS)
    if unknown_labels:
        raise AnnotationValidationError(f"unknown analyst labels: {unknown_labels}")

    unknown_benign = sorted(
        set(frame.get_column("benign_explanation").fill_null("").to_list()) - ALLOWED_BENIGN_EXPLANATIONS
    )
    if unknown_benign:
        raise AnnotationValidationError(f"unknown benign explanations: {unknown_benign}")

    confidence = frame.get_column("confidence")
    invalid_confidence = frame.filter((pl.col("confidence") < 0.0) | (pl.col("confidence") > 1.0))
    if invalid_confidence.height:
        raise AnnotationValidationError("confidence must be in [0, 1]")

    if frame.get_column("review_event_id").is_duplicated().any():
        raise AnnotationValidationError("review_event_id must be unique in annotation file")

    return frame.select(ANNOTATION_COLUMNS)
```

**Step 4: Run test to verify pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_annotations.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/spoofing_detection/lob/annotations.py tests/lob/test_annotations.py
git commit -m "feat: add analyst annotation schema"
```

---

## Task 2: Add annotation file bootstrap script

**Objective:** Create a reproducible script that initializes an empty annotation CSV for the currently detected dashboard events.

**Files:**
- Create: `scripts/bootstrap_spoofing_event_annotations.py`
- Test: `tests/lob/test_bootstrap_spoofing_event_annotations.py`

**Step 1: Write failing tests**

Create `tests/lob/test_bootstrap_spoofing_event_annotations.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_spoofing_event_annotations.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("bootstrap_spoofing_event_annotations", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_empty_annotations_creates_one_row_per_event():
    module = _load_module()
    events = pl.DataFrame({"review_event_id": ["S2", "S1"]})

    annotations = module.build_empty_annotations(events, reviewer="analyst")

    assert annotations.get_column("review_event_id").to_list() == ["S1", "S2"]
    assert annotations.get_column("analyst_label").to_list() == ["unclear_needs_more_context", "unclear_needs_more_context"]
    assert annotations.get_column("reviewer").to_list() == ["analyst", "analyst"]


def test_merge_existing_annotations_preserves_labels():
    module = _load_module()
    events = pl.DataFrame({"review_event_id": ["S1", "S2"]})
    existing = pl.DataFrame(
        {
            "review_event_id": ["S1"],
            "analyst_label": ["legitimate_market_making"],
            "confidence": [0.8],
            "benign_explanation": ["quote_refresh"],
            "notes": ["looks benign"],
            "reviewer": ["alice"],
            "reviewed_at_utc": ["2026-06-23T10:00:00Z"],
        }
    )

    merged = module.merge_existing_annotations(events, existing, reviewer="bob")

    row_s1 = merged.filter(pl.col("review_event_id") == "S1").row(0, named=True)
    row_s2 = merged.filter(pl.col("review_event_id") == "S2").row(0, named=True)
    assert row_s1["analyst_label"] == "legitimate_market_making"
    assert row_s2["analyst_label"] == "unclear_needs_more_context"
```

**Step 2: Run test to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_bootstrap_spoofing_event_annotations.py
```

Expected: FAIL because the script does not exist.

**Step 3: Implement script**

Create `scripts/bootstrap_spoofing_event_annotations.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from spoofing_detection.lob.annotations import ANNOTATION_COLUMNS, validate_annotations


def build_empty_annotations(events: pl.DataFrame, *, reviewer: str) -> pl.DataFrame:
    ids = sorted(events.get_column("review_event_id").cast(pl.Utf8).unique().to_list())
    now = datetime.now(timezone.utc).isoformat()
    return pl.DataFrame(
        {
            "review_event_id": ids,
            "analyst_label": ["unclear_needs_more_context"] * len(ids),
            "confidence": [0.0] * len(ids),
            "benign_explanation": [""] * len(ids),
            "notes": [""] * len(ids),
            "reviewer": [reviewer] * len(ids),
            "reviewed_at_utc": [now] * len(ids),
        }
    ).select(ANNOTATION_COLUMNS)


def merge_existing_annotations(events: pl.DataFrame, existing: pl.DataFrame, *, reviewer: str) -> pl.DataFrame:
    base = build_empty_annotations(events, reviewer=reviewer)
    existing = validate_annotations(existing)
    existing_ids = set(existing.get_column("review_event_id").to_list())
    new_rows = base.filter(~pl.col("review_event_id").is_in(existing_ids))
    return pl.concat([existing, new_rows], how="vertical").sort("review_event_id")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Initialize or update event annotation CSV for spoofing dashboard events.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reviewer", default="analyst")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    events = pl.read_parquet(args.events)
    if args.output.exists():
        annotations = merge_existing_annotations(events, pl.read_csv(args.output), reviewer=args.reviewer)
    else:
        annotations = build_empty_annotations(events, reviewer=args.reviewer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    annotations.write_csv(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify pass**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_bootstrap_spoofing_event_annotations.py
```

Expected: PASS.

**Step 5: Run script on current review output**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/bootstrap_spoofing_event_annotations.py \
  --events outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_events.parquet \
  --output outputs/spoofing_event_review/risanamento_top3_timing_window/annotations/events.csv \
  --reviewer daniele
```

Expected: creates `outputs/spoofing_event_review/risanamento_top3_timing_window/annotations/events.csv` with 32 rows.

**Step 6: Commit**

```bash
git add scripts/bootstrap_spoofing_event_annotations.py tests/lob/test_bootstrap_spoofing_event_annotations.py
git commit -m "feat: bootstrap spoofing event annotations"
```

---

## Task 3: Display analyst annotations in the dashboard

**Objective:** Show current labels and notes in the event dashboard so analysts see model evidence, LLM explanation, and human label together.

**Files:**
- Modify: `scripts/build_spoofing_event_review_dashboard.py`
- Modify: `tests/lob/test_spoofing_event_review_dashboard.py`

**Step 1: Write failing test**

Add a test to `tests/lob/test_spoofing_event_review_dashboard.py` near the existing dashboard HTML tests:

```python
def test_dashboard_embeds_annotations(tmp_path):
    module = _load_module()
    review_events = pl.DataFrame(
        {
            "review_event_id": ["S1"],
            "event_ts": ["2024-06-10T10:00:00"],
            "client_id": ["C1"],
            "MSCI": [0.2],
            "matched_deceptive_cancel_order_ids_window": ["O1"],
        }
    )
    event_log = pl.DataFrame({"review_event_id": ["S1"], "sort_index": [1]})
    queue = pl.DataFrame({"review_event_id": ["S1"], "phase": ["pre"], "side": ["bid"], "price": [1.0]})
    annotations = pl.DataFrame(
        {
            "review_event_id": ["S1"],
            "analyst_label": ["weak_spoofing_like"],
            "confidence": [0.7],
            "benign_explanation": ["quote_refresh"],
            "notes": ["needs more context"],
            "reviewer": ["alice"],
            "reviewed_at_utc": ["2026-06-23T10:00:00Z"],
        }
    )
    output = tmp_path / "dashboard.html"

    module.write_dashboard(
        output,
        review_events=review_events,
        event_log=event_log,
        queue=queue,
        annotations=annotations,
    )

    html = output.read_text()
    assert "Analyst annotation" in html
    assert "weak_spoofing_like" in html
    assert "needs more context" in html
```

**Step 2: Run test to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_review_dashboard.py::test_dashboard_embeds_annotations
```

Expected: FAIL because `write_dashboard` does not accept `annotations` yet.

**Step 3: Implement dashboard annotation loading**

Modify `scripts/build_spoofing_event_review_dashboard.py`:

- Add CLI argument:

```python
parser.add_argument("--annotations", type=Path, default=None, help="Optional analyst annotation CSV.")
```

- Add helper:

```python
def _load_annotations(path: Path | None) -> pl.DataFrame:
    if path is None or not path.exists():
        return pl.DataFrame({"review_event_id": [], "analyst_label": [], "confidence": [], "benign_explanation": [], "notes": [], "reviewer": [], "reviewed_at_utc": []})
    from spoofing_detection.lob.annotations import validate_annotations
    return validate_annotations(pl.read_csv(path))
```

- Update `write_dashboard` signature:

```python
def write_dashboard(..., annotations: pl.DataFrame | None = None, ...):
```

- Embed JavaScript data:

```python
const annotations = {_json_records(annotations if annotations is not None else pl.DataFrame())};
```

- Add annotation card in HTML:

```html
<div class="card"><h2>Analyst annotation</h2><div id="annotation"></div></div>
```

- Add JavaScript render function:

```javascript
function renderAnnotation(ev) {
  const row = annotations.find(r => r.review_event_id === ev.review_event_id);
  const el = document.getElementById('annotation');
  if (!row) {
    el.innerHTML = '<p>No analyst annotation found for this event.</p>';
    return;
  }
  el.innerHTML = `<table>
    <tr><th>Label</th><td>${escapeHtml(row.analyst_label)}</td></tr>
    <tr><th>Confidence</th><td>${Number(row.confidence).toFixed(2)}</td></tr>
    <tr><th>Benign explanation</th><td>${escapeHtml(row.benign_explanation || '')}</td></tr>
    <tr><th>Reviewer</th><td>${escapeHtml(row.reviewer || '')}</td></tr>
    <tr><th>Reviewed at</th><td>${escapeHtml(row.reviewed_at_utc || '')}</td></tr>
    <tr><th>Notes</th><td>${escapeHtml(row.notes || '')}</td></tr>
  </table>`;
}
```

- Update dashboard `update(id)` to call `renderAnnotation(ev)`.

- In `main`, load annotations and pass to `write_dashboard`.

**Step 4: Run test to verify pass**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_review_dashboard.py::test_dashboard_embeds_annotations
```

Expected: PASS.

**Step 5: Regenerate current dashboard with annotations**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/build_spoofing_event_review_dashboard.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --execution-metrics outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/execution_metrics.parquet \
  --candidate-deceptive-orders outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/candidate_deceptive_orders.parquet \
  --parameter-grid-root outputs/spoofing_metrics/kappa_lambda_sensitivity_top3 \
  --annotations outputs/spoofing_event_review/risanamento_top3_timing_window/annotations/events.csv \
  --output-dir outputs/spoofing_event_review/risanamento_top3_timing_window \
  --top-n 10 \
  --pre-window-seconds 30 \
  --post-window-seconds 5
```

Expected: dashboard regenerates and shows annotation card.

**Step 6: Commit**

```bash
git add scripts/build_spoofing_event_review_dashboard.py tests/lob/test_spoofing_event_review_dashboard.py
git commit -m "feat: show analyst annotations in dashboard"
```

---

## Task 4: Add client-session aggregate surveillance features

**Objective:** Promote evidence from isolated event rows to repeated client-session behavior.

**Files:**
- Create: `src/spoofing_detection/lob/client_session_features.py`
- Test: `tests/lob/test_client_session_features.py`

**Step 1: Write failing tests**

Create `tests/lob/test_client_session_features.py`:

```python
from __future__ import annotations

import polars as pl

from spoofing_detection.lob.client_session_features import compute_client_session_features


def test_compute_client_session_features_aggregates_repeated_events():
    executions = pl.DataFrame(
        {
            "client_id": ["A", "A", "B"],
            "event_ts": ["2024-06-10T10:00:00", "2024-06-10T10:01:00", "2024-06-10T10:02:00"],
            "top_n": [3, 3, 3],
            "MSCI": [0.8, 0.2, 0.0],
            "SCI": [0.9, 0.3, 0.1],
            "collapse_opposite_side": [0.7, 0.1, 0.0],
            "collapse_same_side": [0.1, 0.2, 0.0],
            "matched_deceptive_cancel_fraction_window": [0.9, 0.4, 0.0],
            "fill_qty": [100.0, 50.0, 20.0],
        }
    )

    features = compute_client_session_features(executions, msci_threshold=0.5)

    row_a = features.filter(pl.col("client_id") == "A").row(0, named=True)
    assert row_a["event_count"] == 2
    assert row_a["msci_exceedance_count"] == 1
    assert row_a["mcps_at_threshold"] == 0.5
    assert row_a["max_MSCI"] == 0.8


def test_compute_client_session_features_handles_empty_input():
    features = compute_client_session_features(pl.DataFrame(), msci_threshold=0.5)
    assert features.is_empty()
    assert "client_id" in features.columns
```

**Step 2: Run test to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_client_session_features.py
```

Expected: FAIL because module does not exist.

**Step 3: Implement feature computation**

Create `src/spoofing_detection/lob/client_session_features.py`:

```python
from __future__ import annotations

import polars as pl

CLIENT_SESSION_FEATURE_SCHEMA = {
    "client_id": pl.Utf8,
    "event_count": pl.UInt32,
    "msci_exceedance_count": pl.UInt32,
    "mcps_at_threshold": pl.Float64,
    "max_MSCI": pl.Float64,
    "mean_MSCI": pl.Float64,
    "mean_SCI": pl.Float64,
    "mean_opposite_collapse": pl.Float64,
    "mean_same_side_collapse": pl.Float64,
    "mean_matched_cancel_fraction": pl.Float64,
    "total_fill_qty": pl.Float64,
}


def empty_client_session_features() -> pl.DataFrame:
    return pl.DataFrame(schema=CLIENT_SESSION_FEATURE_SCHEMA)


def compute_client_session_features(executions: pl.DataFrame, *, msci_threshold: float) -> pl.DataFrame:
    if executions.is_empty() or "client_id" not in executions.columns:
        return empty_client_session_features()

    required = [
        "client_id",
        "MSCI",
        "SCI",
        "collapse_opposite_side",
        "collapse_same_side",
        "matched_deceptive_cancel_fraction_window",
        "fill_qty",
    ]
    missing = [column for column in required if column not in executions.columns]
    if missing:
        raise ValueError(f"missing execution metric columns: {missing}")

    return (
        executions.with_columns((pl.col("MSCI") > msci_threshold).cast(pl.UInt8).alias("msci_exceeds_threshold"))
        .group_by("client_id")
        .agg(
            [
                pl.len().alias("event_count"),
                pl.col("msci_exceeds_threshold").sum().alias("msci_exceedance_count"),
                pl.col("MSCI").max().alias("max_MSCI"),
                pl.col("MSCI").mean().alias("mean_MSCI"),
                pl.col("SCI").mean().alias("mean_SCI"),
                pl.col("collapse_opposite_side").mean().alias("mean_opposite_collapse"),
                pl.col("collapse_same_side").mean().alias("mean_same_side_collapse"),
                pl.col("matched_deceptive_cancel_fraction_window").mean().alias("mean_matched_cancel_fraction"),
                pl.col("fill_qty").sum().alias("total_fill_qty"),
            ]
        )
        .with_columns((pl.col("msci_exceedance_count") / pl.col("event_count")).alias("mcps_at_threshold"))
        .select(list(CLIENT_SESSION_FEATURE_SCHEMA))
        .sort(["mcps_at_threshold", "max_MSCI", "event_count"], descending=[True, True, True])
    )
```

**Step 4: Run test to verify pass**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_client_session_features.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/spoofing_detection/lob/client_session_features.py tests/lob/test_client_session_features.py
git commit -m "feat: compute client session spoofing features"
```

---

## Task 5: Add client-session feature script

**Objective:** Generate client-session feature parquet and CSV reports from event-level execution metrics.

**Files:**
- Create: `scripts/compute_client_session_spoofing_features.py`
- Test: `tests/lob/test_compute_client_session_spoofing_features.py`

**Step 1: Write failing smoke test**

Create `tests/lob/test_compute_client_session_spoofing_features.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "compute_client_session_spoofing_features.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("compute_client_session_spoofing_features", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compute_and_write_features(tmp_path):
    module = _load_module()
    executions = pl.DataFrame(
        {
            "client_id": ["A"],
            "MSCI": [0.8],
            "SCI": [0.7],
            "collapse_opposite_side": [0.6],
            "collapse_same_side": [0.1],
            "matched_deceptive_cancel_fraction_window": [0.9],
            "fill_qty": [100.0],
        }
    )
    input_path = tmp_path / "execution_metrics.parquet"
    output_dir = tmp_path / "features"
    executions.write_parquet(input_path)

    output = module.compute_and_write(input_path=input_path, output_dir=output_dir, msci_threshold=0.5)

    assert output["parquet"].exists()
    assert output["csv"].exists()
    assert pl.read_parquet(output["parquet"]).height == 1
```

**Step 2: Run test to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_compute_client_session_spoofing_features.py
```

Expected: FAIL because script does not exist.

**Step 3: Implement script**

Create `scripts/compute_client_session_spoofing_features.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from spoofing_detection.lob.client_session_features import compute_client_session_features


def compute_and_write(*, input_path: Path, output_dir: Path, msci_threshold: float) -> dict[str, Path]:
    executions = pl.read_parquet(input_path)
    features = compute_client_session_features(executions, msci_threshold=msci_threshold)
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / "client_session_features.parquet"
    csv_path = output_dir / "client_session_features.csv"
    metadata_path = output_dir / "metadata.json"
    features.write_parquet(parquet_path)
    features.write_csv(csv_path)
    metadata_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "input_path": str(input_path),
                "msci_threshold": msci_threshold,
                "rows": features.height,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return {"parquet": parquet_path, "csv": csv_path, "metadata": metadata_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute client-session spoofing surveillance features.")
    parser.add_argument("--execution-metrics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--msci-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = compute_and_write(
        input_path=args.execution_metrics,
        output_dir=args.output_dir,
        msci_threshold=args.msci_threshold,
    )
    print(outputs["parquet"])


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify pass**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_compute_client_session_spoofing_features.py
```

Expected: PASS.

**Step 5: Run on current data**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/compute_client_session_spoofing_features.py \
  --execution-metrics outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/execution_metrics.parquet \
  --output-dir outputs/spoofing_production_readiness/risanamento_top3/client_session_features \
  --msci-threshold 0.5
```

Expected:

- `outputs/spoofing_production_readiness/risanamento_top3/client_session_features/client_session_features.parquet`
- `outputs/spoofing_production_readiness/risanamento_top3/client_session_features/client_session_features.csv`
- `outputs/spoofing_production_readiness/risanamento_top3/client_session_features/metadata.json`

**Step 6: Commit**

```bash
git add scripts/compute_client_session_spoofing_features.py tests/lob/test_compute_client_session_spoofing_features.py
git commit -m "feat: write client session spoofing features"
```

---

## Task 6: Add market-maker legitimacy baseline features

**Objective:** Add features that explicitly represent legitimate liquidity provision, not only spoofing-like behavior.

**Files:**
- Create: `src/spoofing_detection/lob/legitimacy_features.py`
- Test: `tests/lob/test_legitimacy_features.py`

**Step 1: Write failing tests**

Create `tests/lob/test_legitimacy_features.py`:

```python
from __future__ import annotations

import polars as pl

from spoofing_detection.lob.legitimacy_features import compute_legitimacy_features


def test_compute_legitimacy_features_measures_symmetry_and_cancel_after_fill():
    events = pl.DataFrame(
        {
            "client_id": ["A", "A", "A", "A"],
            "side": ["bid", "ask", "bid", "ask"],
            "is_execution_order": [False, True, False, False],
            "is_matched_deceptive_cancel_order": [False, False, True, False],
            "displayed_qty": [100.0, 90.0, 100.0, 80.0],
            "last_shares": [0.0, 50.0, 0.0, 0.0],
        }
    )

    features = compute_legitimacy_features(events)

    row = features.row(0, named=True)
    assert row["client_id"] == "A"
    assert row["bid_event_share"] == 0.5
    assert row["ask_event_share"] == 0.5
    assert row["side_symmetry_score"] == 1.0
    assert row["matched_cancel_event_share"] == 0.25


def test_compute_legitimacy_features_handles_empty_input():
    features = compute_legitimacy_features(pl.DataFrame())
    assert features.is_empty()
    assert "client_id" in features.columns
```

**Step 2: Run test to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_legitimacy_features.py
```

Expected: FAIL because module does not exist.

**Step 3: Implement legitimacy features**

Create `src/spoofing_detection/lob/legitimacy_features.py`:

```python
from __future__ import annotations

import polars as pl

LEGITIMACY_FEATURE_SCHEMA = {
    "client_id": pl.Utf8,
    "event_count": pl.UInt32,
    "bid_event_share": pl.Float64,
    "ask_event_share": pl.Float64,
    "side_symmetry_score": pl.Float64,
    "matched_cancel_event_share": pl.Float64,
    "execution_event_share": pl.Float64,
    "mean_displayed_qty": pl.Float64,
    "total_executed_qty": pl.Float64,
}


def empty_legitimacy_features() -> pl.DataFrame:
    return pl.DataFrame(schema=LEGITIMACY_FEATURE_SCHEMA)


def compute_legitimacy_features(events: pl.DataFrame) -> pl.DataFrame:
    if events.is_empty() or "client_id" not in events.columns:
        return empty_legitimacy_features()

    required = ["client_id", "side", "is_execution_order", "is_matched_deceptive_cancel_order", "displayed_qty", "last_shares"]
    missing = [column for column in required if column not in events.columns]
    if missing:
        raise ValueError(f"missing legitimacy feature columns: {missing}")

    return (
        events.with_columns(
            [
                (pl.col("side") == "bid").cast(pl.UInt8).alias("is_bid"),
                (pl.col("side") == "ask").cast(pl.UInt8).alias("is_ask"),
                pl.col("is_execution_order").fill_null(False).cast(pl.UInt8).alias("execution_flag"),
                pl.col("is_matched_deceptive_cancel_order").fill_null(False).cast(pl.UInt8).alias("matched_cancel_flag"),
            ]
        )
        .group_by("client_id")
        .agg(
            [
                pl.len().alias("event_count"),
                pl.col("is_bid").mean().alias("bid_event_share"),
                pl.col("is_ask").mean().alias("ask_event_share"),
                pl.col("matched_cancel_flag").mean().alias("matched_cancel_event_share"),
                pl.col("execution_flag").mean().alias("execution_event_share"),
                pl.col("displayed_qty").mean().alias("mean_displayed_qty"),
                pl.col("last_shares").fill_null(0).sum().alias("total_executed_qty"),
            ]
        )
        .with_columns((1.0 - (pl.col("bid_event_share") - pl.col("ask_event_share")).abs()).alias("side_symmetry_score"))
        .select(list(LEGITIMACY_FEATURE_SCHEMA))
        .sort(["side_symmetry_score", "event_count"], descending=[True, True])
    )
```

**Step 4: Run test to verify pass**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_legitimacy_features.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/spoofing_detection/lob/legitimacy_features.py tests/lob/test_legitimacy_features.py
git commit -m "feat: compute market maker legitimacy features"
```

---

## Task 7: Add negative-control/placebo feature generators

**Objective:** Create placebo controls to test whether scores are truly conditional on the target client/event rather than generic quote refresh.

**Files:**
- Create: `src/spoofing_detection/lob/negative_controls.py`
- Test: `tests/lob/test_negative_controls.py`

**Step 1: Write failing tests**

Create `tests/lob/test_negative_controls.py`:

```python
from __future__ import annotations

import polars as pl

from spoofing_detection.lob.negative_controls import add_time_shift_placebo, add_wrong_side_placebo


def test_add_time_shift_placebo_offsets_sort_index():
    events = pl.DataFrame({"review_event_id": ["S1"], "sort_index": [100], "client_id": ["A"]})
    shifted = add_time_shift_placebo(events, shift_events=50)
    row = shifted.row(0, named=True)
    assert row["placebo_type"] == "time_shift"
    assert row["placebo_sort_index"] == 150


def test_add_wrong_side_placebo_flips_side():
    events = pl.DataFrame({"review_event_id": ["S1"], "execution_side": ["ask"], "deceptive_side": ["bid"]})
    wrong = add_wrong_side_placebo(events)
    row = wrong.row(0, named=True)
    assert row["placebo_type"] == "wrong_side"
    assert row["placebo_deceptive_side"] == "ask"
```

**Step 2: Run test to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_negative_controls.py
```

Expected: FAIL because module does not exist.

**Step 3: Implement simple placebo descriptors**

Create `src/spoofing_detection/lob/negative_controls.py`:

```python
from __future__ import annotations

import polars as pl


def add_time_shift_placebo(events: pl.DataFrame, *, shift_events: int) -> pl.DataFrame:
    if "sort_index" not in events.columns:
        raise ValueError("events must contain sort_index for time-shift placebo")
    return events.with_columns(
        [
            pl.lit("time_shift").alias("placebo_type"),
            (pl.col("sort_index") + shift_events).alias("placebo_sort_index"),
        ]
    )


def add_wrong_side_placebo(events: pl.DataFrame) -> pl.DataFrame:
    if "deceptive_side" not in events.columns:
        raise ValueError("events must contain deceptive_side for wrong-side placebo")
    return events.with_columns(
        [
            pl.lit("wrong_side").alias("placebo_type"),
            pl.when(pl.col("deceptive_side") == "bid")
            .then(pl.lit("ask"))
            .when(pl.col("deceptive_side") == "ask")
            .then(pl.lit("bid"))
            .otherwise(None)
            .alias("placebo_deceptive_side"),
        ]
    )
```

**Step 4: Run test to verify pass**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_negative_controls.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/spoofing_detection/lob/negative_controls.py tests/lob/test_negative_controls.py
git commit -m "feat: add spoofing negative control descriptors"
```

---

## Task 8: Add negative-control report script

**Objective:** Produce an initial production-readiness report comparing real candidate events against simple placebo descriptors.

**Files:**
- Create: `scripts/build_spoofing_negative_control_report.py`
- Test: `tests/lob/test_spoofing_negative_control_report.py`

**Step 1: Write failing test**

Create `tests/lob/test_spoofing_negative_control_report.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

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
```

**Step 2: Run test to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_negative_control_report.py
```

Expected: FAIL because script does not exist.

**Step 3: Implement report script**

Create `scripts/build_spoofing_negative_control_report.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from spoofing_detection.lob.negative_controls import add_time_shift_placebo, add_wrong_side_placebo


def build_report(*, events_path: Path, output_path: Path, shift_events: int) -> None:
    events = pl.read_parquet(events_path)
    time_shift = add_time_shift_placebo(events, shift_events=shift_events)
    wrong_side = add_wrong_side_placebo(events)
    lines = [
        "# Spoofing Negative-Control Report",
        "",
        "This report defines placebo controls that should be scored in a later calibration run.",
        "",
        f"Real candidate events: {events.height}",
        f"time_shift placebo events: {time_shift.height}",
        f"wrong_side placebo events: {wrong_side.height}",
        "",
        "## Interpretation",
        "",
        "A production detector should score real candidate events higher than time-shifted or wrong-side placebo events. If placebo scores are comparable, the model may be detecting generic quote refresh rather than spoofing-like conditional cancellation.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spoofing negative-control report.")
    parser.add_argument("--events", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shift-events", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_report(events_path=args.events, output_path=args.output, shift_events=args.shift_events)
    print(args.output)


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify pass**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_negative_control_report.py
```

Expected: PASS.

**Step 5: Run on current dashboard events**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/build_spoofing_negative_control_report.py \
  --events outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_events.parquet \
  --output outputs/spoofing_production_readiness/risanamento_top3/negative_controls/report.md \
  --shift-events 50
```

Expected: writes negative-control report markdown.

**Step 6: Commit**

```bash
git add scripts/build_spoofing_negative_control_report.py tests/lob/test_spoofing_negative_control_report.py
git commit -m "feat: report spoofing negative controls"
```

---

## Task 9: Add threshold calibration utilities

**Objective:** Convert analyst labels and scores into threshold/workload tables.

**Files:**
- Create: `src/spoofing_detection/lob/calibration.py`
- Test: `tests/lob/test_calibration.py`

**Step 1: Write failing tests**

Create `tests/lob/test_calibration.py`:

```python
from __future__ import annotations

import polars as pl

from spoofing_detection.lob.calibration import build_threshold_table


def test_build_threshold_table_counts_alerts_and_positive_labels():
    scores = pl.DataFrame({"review_event_id": ["S1", "S2", "S3"], "MSCI": [0.9, 0.4, 0.1]})
    labels = pl.DataFrame(
        {
            "review_event_id": ["S1", "S2", "S3"],
            "analyst_label": ["strong_spoofing_like", "legitimate_market_making", "weak_spoofing_like"],
        }
    )

    table = build_threshold_table(scores, labels, score_column="MSCI", thresholds=[0.0, 0.5])

    row = table.filter(pl.col("threshold") == 0.5).row(0, named=True)
    assert row["alert_count"] == 1
    assert row["positive_label_count"] == 1
    assert row["precision_proxy"] == 1.0
```

**Step 2: Run test to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_calibration.py
```

Expected: FAIL because module does not exist.

**Step 3: Implement calibration utility**

Create `src/spoofing_detection/lob/calibration.py`:

```python
from __future__ import annotations

import polars as pl

POSITIVE_LABELS = {"strong_spoofing_like", "moderate_spoofing_like", "weak_spoofing_like"}


def build_threshold_table(
    scores: pl.DataFrame,
    labels: pl.DataFrame,
    *,
    score_column: str,
    thresholds: list[float],
) -> pl.DataFrame:
    if score_column not in scores.columns:
        raise ValueError(f"score column missing: {score_column}")
    joined = scores.join(labels.select(["review_event_id", "analyst_label"]), on="review_event_id", how="left")
    rows = []
    for threshold in thresholds:
        alerted = joined.filter(pl.col(score_column) >= threshold)
        positive_count = alerted.filter(pl.col("analyst_label").is_in(POSITIVE_LABELS)).height
        alert_count = alerted.height
        rows.append(
            {
                "score_column": score_column,
                "threshold": threshold,
                "alert_count": alert_count,
                "positive_label_count": positive_count,
                "precision_proxy": positive_count / alert_count if alert_count else None,
            }
        )
    return pl.DataFrame(rows)
```

**Step 4: Run test to verify pass**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_calibration.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/spoofing_detection/lob/calibration.py tests/lob/test_calibration.py
git commit -m "feat: calibrate spoofing score thresholds"
```

---

## Task 10: Add calibration report script

**Objective:** Generate a markdown and CSV report for score thresholds using analyst labels.

**Files:**
- Create: `scripts/build_spoofing_calibration_report.py`
- Test: `tests/lob/test_spoofing_calibration_report.py`

**Step 1: Write failing smoke test**

Create `tests/lob/test_spoofing_calibration_report.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "build_spoofing_calibration_report.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_spoofing_calibration_report", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_build_calibration_report_writes_outputs(tmp_path):
    module = _load_module()
    scores = pl.DataFrame({"review_event_id": ["S1"], "MSCI": [0.7]})
    labels = pl.DataFrame(
        {
            "review_event_id": ["S1"],
            "analyst_label": ["weak_spoofing_like"],
            "confidence": [0.5],
            "benign_explanation": [""],
            "notes": [""],
            "reviewer": ["a"],
            "reviewed_at_utc": ["2026-06-23T10:00:00Z"],
        }
    )
    scores_path = tmp_path / "scores.parquet"
    labels_path = tmp_path / "labels.csv"
    output_dir = tmp_path / "calibration"
    scores.write_parquet(scores_path)
    labels.write_csv(labels_path)

    outputs = module.build_report(scores_path=scores_path, annotations_path=labels_path, output_dir=output_dir)

    assert outputs["csv"].exists()
    assert outputs["markdown"].exists()
    assert "Threshold Calibration" in outputs["markdown"].read_text()
```

**Step 2: Run test to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_calibration_report.py
```

Expected: FAIL because script does not exist.

**Step 3: Implement script**

Create `scripts/build_spoofing_calibration_report.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
from pathlib import Path

import polars as pl

from spoofing_detection.lob.annotations import validate_annotations
from spoofing_detection.lob.calibration import build_threshold_table


def build_report(*, scores_path: Path, annotations_path: Path, output_dir: Path) -> dict[str, Path]:
    scores = pl.read_parquet(scores_path)
    annotations = validate_annotations(pl.read_csv(annotations_path))
    table = build_threshold_table(scores, annotations, score_column="MSCI", thresholds=[0.0, 0.1, 0.25, 0.5, 0.75, 1.0])
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "threshold_calibration.csv"
    markdown_path = output_dir / "threshold_calibration.md"
    table.write_csv(csv_path)
    lines = [
        "# Threshold Calibration",
        "",
        "This report summarizes alert workload and positive-label concentration by MSCI threshold.",
        "",
        table.to_pandas().to_markdown(index=False),
        "",
        "Interpret precision_proxy cautiously when labels are sparse or exploratory.",
    ]
    markdown_path.write_text("\n".join(lines))
    return {"csv": csv_path, "markdown": markdown_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build spoofing score threshold calibration report.")
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_report(scores_path=args.scores, annotations_path=args.annotations, output_dir=args.output_dir)
    print(outputs["markdown"])


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify pass**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_calibration_report.py
```

Expected: PASS.

**Step 5: Run on current annotations after human labels exist**

Do not expect meaningful calibration until labels are edited manually.

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/build_spoofing_calibration_report.py \
  --scores outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_events.parquet \
  --annotations outputs/spoofing_event_review/risanamento_top3_timing_window/annotations/events.csv \
  --output-dir outputs/spoofing_production_readiness/risanamento_top3/calibration
```

Expected: report files are produced, but with preliminary labels only.

**Step 6: Commit**

```bash
git add scripts/build_spoofing_calibration_report.py tests/lob/test_spoofing_calibration_report.py
git commit -m "feat: report spoofing threshold calibration"
```

---

## Task 11: Build client-session alert objects

**Objective:** Produce production-style client-session alerts that combine spoofing features, legitimacy features, labels, and recommended actions.

**Files:**
- Create: `src/spoofing_detection/lob/alert_objects.py`
- Test: `tests/lob/test_alert_objects.py`

**Step 1: Write failing tests**

Create `tests/lob/test_alert_objects.py`:

```python
from __future__ import annotations

import polars as pl

from spoofing_detection.lob.alert_objects import build_client_session_alerts


def test_build_client_session_alerts_combines_risk_and_legitimacy():
    risk = pl.DataFrame(
        {
            "client_id": ["A"],
            "event_count": [5],
            "mcps_at_threshold": [0.6],
            "max_MSCI": [0.9],
            "mean_MSCI": [0.4],
        }
    )
    legitimacy = pl.DataFrame(
        {
            "client_id": ["A"],
            "side_symmetry_score": [0.2],
            "execution_event_share": [0.1],
            "matched_cancel_event_share": [0.5],
        }
    )

    alerts = build_client_session_alerts(risk, legitimacy, min_events=3, min_mcps=0.5)

    row = alerts.row(0, named=True)
    assert row["client_id"] == "A"
    assert row["recommended_action"] == "human_review"
    assert row["alert_score"] > 0


def test_build_client_session_alerts_filters_low_event_counts():
    risk = pl.DataFrame({"client_id": ["A"], "event_count": [1], "mcps_at_threshold": [1.0], "max_MSCI": [1.0], "mean_MSCI": [1.0]})
    legitimacy = pl.DataFrame({"client_id": ["A"], "side_symmetry_score": [0.5], "execution_event_share": [0.5], "matched_cancel_event_share": [0.5]})
    alerts = build_client_session_alerts(risk, legitimacy, min_events=3, min_mcps=0.5)
    assert alerts.is_empty()
```

**Step 2: Run test to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_alert_objects.py
```

Expected: FAIL because module does not exist.

**Step 3: Implement alert builder**

Create `src/spoofing_detection/lob/alert_objects.py`:

```python
from __future__ import annotations

import polars as pl


def build_client_session_alerts(
    risk_features: pl.DataFrame,
    legitimacy_features: pl.DataFrame,
    *,
    min_events: int,
    min_mcps: float,
) -> pl.DataFrame:
    if risk_features.is_empty():
        return pl.DataFrame(
            schema={
                "client_id": pl.Utf8,
                "alert_score": pl.Float64,
                "recommended_action": pl.Utf8,
            }
        )

    joined = risk_features.join(legitimacy_features, on="client_id", how="left")
    return (
        joined.with_columns(
            (
                pl.col("mcps_at_threshold").fill_null(0.0) * 0.5
                + pl.col("max_MSCI").fill_null(0.0).clip(0.0, 1.0) * 0.3
                + (1.0 - pl.col("side_symmetry_score").fill_null(0.5)).clip(0.0, 1.0) * 0.2
            ).alias("alert_score")
        )
        .filter((pl.col("event_count") >= min_events) & (pl.col("mcps_at_threshold") >= min_mcps))
        .with_columns(pl.lit("human_review").alias("recommended_action"))
        .sort(["alert_score", "event_count"], descending=[True, True])
    )
```

**Step 4: Run test to verify pass**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_alert_objects.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add src/spoofing_detection/lob/alert_objects.py tests/lob/test_alert_objects.py
git commit -m "feat: build client session alert objects"
```

---

## Task 12: Add production-readiness orchestrator script

**Objective:** Run the new production-readiness layer end to end and produce a compact output folder.

**Files:**
- Create: `scripts/run_spoofing_production_readiness.py`
- Test: `tests/lob/test_run_spoofing_production_readiness.py`

**Step 1: Write failing smoke test**

Create `tests/lob/test_run_spoofing_production_readiness.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "run_spoofing_production_readiness.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("run_spoofing_production_readiness", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_run_pipeline_writes_alerts(tmp_path):
    module = _load_module()
    executions = pl.DataFrame(
        {
            "review_event_id": ["S1", "S2", "S3"],
            "client_id": ["A", "A", "A"],
            "MSCI": [0.8, 0.7, 0.1],
            "SCI": [0.9, 0.8, 0.2],
            "collapse_opposite_side": [0.7, 0.6, 0.1],
            "collapse_same_side": [0.1, 0.1, 0.1],
            "matched_deceptive_cancel_fraction_window": [0.9, 0.8, 0.1],
            "fill_qty": [100.0, 100.0, 100.0],
        }
    )
    event_log = pl.DataFrame(
        {
            "client_id": ["A", "A"],
            "side": ["bid", "ask"],
            "is_execution_order": [True, False],
            "is_matched_deceptive_cancel_order": [False, True],
            "displayed_qty": [100.0, 100.0],
            "last_shares": [50.0, 0.0],
        }
    )
    execution_path = tmp_path / "execution_metrics.parquet"
    event_log_path = tmp_path / "event_log.parquet"
    output_dir = tmp_path / "readiness"
    executions.write_parquet(execution_path)
    event_log.write_parquet(event_log_path)

    outputs = module.run_pipeline(
        execution_metrics_path=execution_path,
        event_log_path=event_log_path,
        output_dir=output_dir,
        msci_threshold=0.5,
        min_events=2,
        min_mcps=0.5,
    )

    assert outputs["alerts"].exists()
    assert pl.read_parquet(outputs["alerts"]).height == 1
```

**Step 2: Run test to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_run_spoofing_production_readiness.py
```

Expected: FAIL because script does not exist.

**Step 3: Implement orchestrator**

Create `scripts/run_spoofing_production_readiness.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import polars as pl

from spoofing_detection.lob.alert_objects import build_client_session_alerts
from spoofing_detection.lob.client_session_features import compute_client_session_features
from spoofing_detection.lob.legitimacy_features import compute_legitimacy_features


def run_pipeline(
    *,
    execution_metrics_path: Path,
    event_log_path: Path,
    output_dir: Path,
    msci_threshold: float,
    min_events: int,
    min_mcps: float,
) -> dict[str, Path]:
    executions = pl.read_parquet(execution_metrics_path)
    event_log = pl.read_parquet(event_log_path)
    risk = compute_client_session_features(executions, msci_threshold=msci_threshold)
    legitimacy = compute_legitimacy_features(event_log)
    alerts = build_client_session_alerts(risk, legitimacy, min_events=min_events, min_mcps=min_mcps)

    output_dir.mkdir(parents=True, exist_ok=True)
    risk_path = output_dir / "client_session_risk_features.parquet"
    legitimacy_path = output_dir / "client_legitimacy_features.parquet"
    alerts_path = output_dir / "client_session_alerts.parquet"
    metadata_path = output_dir / "metadata.json"
    risk.write_parquet(risk_path)
    legitimacy.write_parquet(legitimacy_path)
    alerts.write_parquet(alerts_path)
    metadata_path.write_text(
        json.dumps(
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "execution_metrics_path": str(execution_metrics_path),
                "event_log_path": str(event_log_path),
                "msci_threshold": msci_threshold,
                "min_events": min_events,
                "min_mcps": min_mcps,
                "alert_count": alerts.height,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return {"risk": risk_path, "legitimacy": legitimacy_path, "alerts": alerts_path, "metadata": metadata_path}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run production-readiness spoofing surveillance layer.")
    parser.add_argument("--execution-metrics", type=Path, required=True)
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--msci-threshold", type=float, default=0.5)
    parser.add_argument("--min-events", type=int, default=3)
    parser.add_argument("--min-mcps", type=float, default=0.25)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = run_pipeline(
        execution_metrics_path=args.execution_metrics,
        event_log_path=args.event_log,
        output_dir=args.output_dir,
        msci_threshold=args.msci_threshold,
        min_events=args.min_events,
        min_mcps=args.min_mcps,
    )
    print(outputs["alerts"])


if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify pass**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_run_spoofing_production_readiness.py
```

Expected: PASS.

**Step 5: Run on current data**

Use the event log parquet generated by the dashboard:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/run_spoofing_production_readiness.py \
  --execution-metrics outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_events.parquet \
  --event-log outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_event_log.parquet \
  --output-dir outputs/spoofing_production_readiness/risanamento_top3/pipeline \
  --msci-threshold 0.5 \
  --min-events 3 \
  --min-mcps 0.25
```

If `matched_spoofing_event_log.parquet` does not exist, first modify `scripts/build_spoofing_event_review_dashboard.py` to save the event-log dataframe next to the queue parquet, or use the existing event log output path if already generated.

**Step 6: Commit**

```bash
git add scripts/run_spoofing_production_readiness.py tests/lob/test_run_spoofing_production_readiness.py
git commit -m "feat: run spoofing production readiness pipeline"
```

---

## Task 13: Save event-log parquet from dashboard reconstruction if missing

**Objective:** Ensure the production-readiness orchestrator can consume the exact event-log slice used by the dashboard.

**Files:**
- Modify: `scripts/build_spoofing_event_review_dashboard.py`
- Modify: `tests/lob/test_spoofing_event_review_dashboard.py`

**Step 1: Write failing test**

Add a test to `tests/lob/test_spoofing_event_review_dashboard.py` verifying that the dashboard build writes event log parquet when `main` or the relevant writer function runs. If no suitable function exists, extract a helper `write_review_artifacts` and test it.

Suggested test shape:

```python
def test_write_review_artifacts_saves_event_log(tmp_path):
    module = _load_module()
    event_log = pl.DataFrame({"review_event_id": ["S1"], "sort_index": [1]})
    queue = pl.DataFrame({"review_event_id": ["S1"], "phase": ["pre"]})
    outputs = module.write_review_artifacts(output_dir=tmp_path, event_log=event_log, queue=queue)
    assert outputs["event_log"].exists()
    assert outputs["queue"].exists()
```

**Step 2: Run test to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_review_dashboard.py::test_write_review_artifacts_saves_event_log
```

Expected: FAIL if helper does not exist or event log is not written.

**Step 3: Implement helper**

In `scripts/build_spoofing_event_review_dashboard.py`, add:

```python
def write_review_artifacts(*, output_dir: Path, event_log: pl.DataFrame, queue: pl.DataFrame) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    event_log_path = output_dir / "matched_spoofing_event_log.parquet"
    queue_path = output_dir / "matched_spoofing_lob_queue.parquet"
    event_log.write_parquet(event_log_path)
    queue.write_parquet(queue_path)
    return {"event_log": event_log_path, "queue": queue_path}
```

Use this helper in `main` instead of writing only queue directly.

**Step 4: Run test to verify pass**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_review_dashboard.py::test_write_review_artifacts_saves_event_log
```

Expected: PASS.

**Step 5: Regenerate current dashboard artifacts**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/build_spoofing_event_review_dashboard.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --execution-metrics outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/execution_metrics.parquet \
  --candidate-deceptive-orders outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/candidate_deceptive_orders.parquet \
  --parameter-grid-root outputs/spoofing_metrics/kappa_lambda_sensitivity_top3 \
  --annotations outputs/spoofing_event_review/risanamento_top3_timing_window/annotations/events.csv \
  --output-dir outputs/spoofing_event_review/risanamento_top3_timing_window \
  --top-n 10 \
  --pre-window-seconds 30 \
  --post-window-seconds 5
```

Expected:

- `matched_spoofing_event_log.parquet` exists.
- `matched_spoofing_lob_queue.parquet` still exists.

**Step 6: Commit**

```bash
git add scripts/build_spoofing_event_review_dashboard.py tests/lob/test_spoofing_event_review_dashboard.py
git commit -m "feat: save spoofing review event log artifact"
```

---

## Task 14: Add production-readiness dashboard summary card

**Objective:** Display client-session alert summary and calibration links in the existing dashboard.

**Files:**
- Modify: `scripts/build_spoofing_event_review_dashboard.py`
- Modify: `tests/lob/test_spoofing_event_review_dashboard.py`

**Step 1: Write failing test**

Add to `tests/lob/test_spoofing_event_review_dashboard.py`:

```python
def test_dashboard_embeds_client_session_alert_summary(tmp_path):
    module = _load_module()
    review_events = pl.DataFrame({"review_event_id": ["S1"], "event_ts": ["t"], "client_id": ["A"], "MSCI": [0.5], "matched_deceptive_cancel_order_ids_window": [""]})
    event_log = pl.DataFrame({"review_event_id": ["S1"]})
    queue = pl.DataFrame({"review_event_id": ["S1"], "phase": ["pre"]})
    alerts = pl.DataFrame({"client_id": ["A"], "alert_score": [0.8], "recommended_action": ["human_review"]})
    output = tmp_path / "dashboard.html"

    module.write_dashboard(output, review_events=review_events, event_log=event_log, queue=queue, client_session_alerts=alerts)

    html = output.read_text()
    assert "Client-session alerts" in html
    assert "human_review" in html
```

**Step 2: Run test to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_review_dashboard.py::test_dashboard_embeds_client_session_alert_summary
```

Expected: FAIL because `client_session_alerts` argument does not exist.

**Step 3: Implement optional dashboard alert summary**

- Add optional CLI argument:

```python
parser.add_argument("--client-session-alerts", type=Path, default=None)
```

- Add loader:

```python
def _load_optional_parquet(path: Path | None) -> pl.DataFrame:
    if path is None or not path.exists():
        return pl.DataFrame()
    return pl.read_parquet(path)
```

- Update `write_dashboard(..., client_session_alerts: pl.DataFrame | None = None)`.
- Embed JavaScript:

```python
const clientSessionAlerts = {_json_records(client_session_alerts if client_session_alerts is not None else pl.DataFrame())};
```

- Add top card:

```html
<div class="card"><h2>Client-session alerts</h2><div id="clientSessionAlerts"></div></div>
```

- Render table of alert rows or fallback message.

**Step 4: Run test to verify pass**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_review_dashboard.py::test_dashboard_embeds_client_session_alert_summary
```

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/build_spoofing_event_review_dashboard.py tests/lob/test_spoofing_event_review_dashboard.py
git commit -m "feat: show client session alerts in dashboard"
```

---

## Task 15: Add production-readiness documentation

**Objective:** Document the production-readiness workflow, what is scientifically valid, and what remains exploratory.

**Files:**
- Create: `docs/spoofing_production_readiness.md`

**Step 1: Write documentation**

Create `docs/spoofing_production_readiness.md`:

```markdown
# Spoofing Detection Production-Readiness Workflow

## Purpose

This workflow promotes event-level spoofing-like detections into analyst-reviewable client-session alerts.

The detector does not infer legal intent. It produces surveillance cues for human review.

## Layers

1. Event-level DWI/SCI/MSCI/MCPS metrics.
2. Analyst annotations.
3. Client-session repeated-pattern features.
4. Legitimate market-maker baseline features.
5. Negative controls and placebo checks.
6. Threshold calibration by analyst workload and false-positive pressure.
7. Client-session alert objects.

## Recommended order of use

```bash
# 1. Build/refresh event dashboard outputs.
# 2. Bootstrap annotation CSV.
# 3. Analysts edit annotation CSV.
# 4. Compute client-session features.
# 5. Build negative-control report.
# 6. Build calibration report.
# 7. Run production-readiness pipeline.
# 8. Regenerate dashboard with annotation and alert files.
```

## Scientific interpretation

A production alert should be considered stronger when:

- suspicious episodes repeat for the same client/session;
- MCPS remains high across depth choices;
- the pattern is robust over kappa/lambda values;
- opposite-side collapse is stronger than same-side collapse;
- negative controls score lower than real events;
- the behavior is unusual relative to the client's own baseline and peer market makers.

## Non-goals

- The LLM does not decide manipulation.
- The event-level score alone is not a legal conclusion.
- Thresholds are not final until calibrated against labels and analyst workload.
```

**Step 2: Verify documentation exists**

```bash
test -f docs/spoofing_production_readiness.md
```

Expected: exit 0.

**Step 3: Commit**

```bash
git add docs/spoofing_production_readiness.md
git commit -m "docs: describe spoofing production readiness workflow"
```

---

## Task 16: End-to-end validation

**Objective:** Verify the production-readiness layer works with the current Risanamento output and does not break existing tests.

**Files:**
- No source file changes expected.

**Step 1: Run targeted tests**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q \
  tests/lob/test_annotations.py \
  tests/lob/test_bootstrap_spoofing_event_annotations.py \
  tests/lob/test_client_session_features.py \
  tests/lob/test_legitimacy_features.py \
  tests/lob/test_negative_controls.py \
  tests/lob/test_spoofing_negative_control_report.py \
  tests/lob/test_calibration.py \
  tests/lob/test_spoofing_calibration_report.py \
  tests/lob/test_alert_objects.py \
  tests/lob/test_run_spoofing_production_readiness.py \
  tests/lob/test_spoofing_event_review_dashboard.py
```

Expected: all pass.

**Step 2: Run full tests**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q
```

Expected: all pass.

**Step 3: Run current-data production-readiness commands**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/bootstrap_spoofing_event_annotations.py \
  --events outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_events.parquet \
  --output outputs/spoofing_event_review/risanamento_top3_timing_window/annotations/events.csv \
  --reviewer daniele

PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/compute_client_session_spoofing_features.py \
  --execution-metrics outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_events.parquet \
  --output-dir outputs/spoofing_production_readiness/risanamento_top3/client_session_features \
  --msci-threshold 0.5

PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/build_spoofing_negative_control_report.py \
  --events outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_events.parquet \
  --output outputs/spoofing_production_readiness/risanamento_top3/negative_controls/report.md \
  --shift-events 50

PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/build_spoofing_calibration_report.py \
  --scores outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_events.parquet \
  --annotations outputs/spoofing_event_review/risanamento_top3_timing_window/annotations/events.csv \
  --output-dir outputs/spoofing_production_readiness/risanamento_top3/calibration
```

Expected: all output folders are created. Calibration will be preliminary until analyst labels are edited.

**Step 4: Run production-readiness orchestrator**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/run_spoofing_production_readiness.py \
  --execution-metrics outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_events.parquet \
  --event-log outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_event_log.parquet \
  --output-dir outputs/spoofing_production_readiness/risanamento_top3/pipeline \
  --msci-threshold 0.5 \
  --min-events 3 \
  --min-mcps 0.25
```

Expected: writes `client_session_alerts.parquet` and metadata. If no clients meet thresholds, the file may be empty; that is acceptable and should be reported.

**Step 5: Regenerate final dashboard with new files**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/build_spoofing_event_review_dashboard.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --execution-metrics outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/execution_metrics.parquet \
  --candidate-deceptive-orders outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/candidate_deceptive_orders.parquet \
  --parameter-grid-root outputs/spoofing_metrics/kappa_lambda_sensitivity_top3 \
  --annotations outputs/spoofing_event_review/risanamento_top3_timing_window/annotations/events.csv \
  --client-session-alerts outputs/spoofing_production_readiness/risanamento_top3/pipeline/client_session_alerts.parquet \
  --output-dir outputs/spoofing_event_review/risanamento_top3_timing_window \
  --top-n 10 \
  --pre-window-seconds 30 \
  --post-window-seconds 5
```

Expected: dashboard regenerates and includes annotation plus client-session alert cards.

**Step 6: Inspect git diff**

```bash
git status --short
git diff --stat
```

Expected: only intended files are modified/untracked.

**Step 7: Final commit if any validation-only adjustments were made**

```bash
git add docs/spoofing_production_readiness.md scripts src tests
git commit -m "feat: add spoofing production readiness layer"
```

Only commit if there are uncommitted implementation changes not already committed in earlier tasks.

---

## Risks and tradeoffs

1. **Analyst labels are subjective.** Use confidence and notes; do not overfit thresholds to a tiny label set.
2. **Current labels may start as `unclear_needs_more_context`.** Calibration reports are mechanically useful before labels, but scientifically meaningful only after human review.
3. **Legitimacy features are approximate.** Without true inventory, spread capture, and venue-specific queue rules, they are proxies, not proof of legitimate market-making.
4. **Negative controls in this plan are descriptors first.** A later iteration should recompute MSCI under placebo transformations, not only describe them.
5. **Event-log artifact path may not yet exist.** Task 13 handles this by saving the dashboard event-log slice.
6. **Client-session alerts use a simple scoring formula.** Treat it as a transparent baseline, not the final model.
7. **Static dashboard cannot save annotations interactively.** This plan uses CSV editing first. A later server-backed dashboard could support live editing.
8. **LLM reviews are explanations only.** Do not use them as labels or features in threshold calibration.

---

## Open questions for the user / analyst

1. What labels should count as positive for calibration?
   - Current default: strong/moderate/weak spoofing-like all count as positive.
2. Should `weak_spoofing_like` be positive or ambiguous for production thresholds?
3. What daily alert workload is acceptable?
   - e.g. top 5 clients/day, top 20 events/day, or FPR target.
4. Are there known market-maker accounts that should be used as a control group?
5. Are there known enforcement/prosecuted examples that can anchor positive labels?
6. Should client-session aggregation be per instrument/day, per client/day, or per client/instrument/session?
7. Should annotations be event-level only, or should analysts also label client-session alerts?

---

## Later extensions not included in this first implementation

- Interactive dashboard annotation saving via local server.
- Recompute MSCI on full placebo event streams.
- Peer-group normalization by liquidity regime, instrument, and intraday bucket.
- Inventory proxy from signed executions over full session.
- Spread-capture and passive/aggressive execution profitability features.
- Temporal clustering tests for repeated spoofing-like episodes.
- Model card / validation report for regulatory audit.
- CI job that runs the production-readiness smoke pipeline on a tiny fixture dataset.
