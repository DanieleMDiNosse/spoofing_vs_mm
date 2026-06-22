# Local LLM Spoofing Event Analysis Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Add a reproducible local-LLM analysis workflow for one selected candidate spoofing-like event, using Ollama with the already-installed `gemma4` model, and make the generated review visible from the event-review dashboard.

**Architecture:** Start with a reproducible offline/precomputed workflow, not a live browser-to-shell call. A dossier builder creates a compact markdown/JSON evidence packet for a selected event; an analyzer script combines that dossier with a stable surveillance-analyst prompt and calls Ollama; the dashboard displays existing generated reviews and can later be extended to a live server button. This keeps all prompts, inputs, outputs, and model metadata inspectable.

**Tech Stack:** Python, Polars, existing spoofing event-review parquet outputs, Markdown, Ollama CLI/API, local `gemma4` model, static HTML/Plotly dashboard.

---

## Current context / assumptions

- The event-review dashboard already exists at:
  - `scripts/build_spoofing_event_review_dashboard.py`
- Current generated review directory is:
  - `outputs/spoofing_event_review/risanamento_top3_timing_window/`
- Current event-review artifacts include:
  - `matched_spoofing_events.parquet`
  - `matched_spoofing_event_log.parquet`
  - `matched_spoofing_lob_queue.parquet`
  - `matched_spoofing_event_review_dashboard.html`
  - `metadata.json`
- The strict event definition is: matched deceptive-order cancellation only.
- Broad direct opposite-side cancellations should stay out of the main explanation unless explicitly reintroduced as separate diagnostics.
- Ollama is installed.
- A local `gemma4` model is already available.
- First implementation should be offline/precomputed and reproducible. Do not start with a browser button that directly executes local commands.

## Recommended feature shape

### Phase 1: Reproducible offline analysis

Add scripts that produce saved markdown reviews:

```text
outputs/spoofing_event_review/risanamento_top3_timing_window/llm_reviews/
  S12345/
    dossier.md
    dossier.json
    prompt.md
    response.md
    metadata.json
```

The dashboard then shows the saved review for the selected event if it exists.

### Phase 2: Dashboard integration

Add a disabled/non-executing `Analyze` button or panel in the static dashboard:

- If review exists: show it.
- If review does not exist: show the exact command to generate it.

This avoids unsafe static-browser execution while still making the dashboard the analyst’s main review tool.

### Phase 3: Optional local server later

Only after Phase 1 is stable, consider a local server:

```bash
python scripts/serve_spoofing_event_review_dashboard.py \
  --review-dir outputs/spoofing_event_review/risanamento_top3_timing_window \
  --model gemma4
```

Then the dashboard button can call `/api/analyze-event`. This is intentionally not part of the first implementation.

---

## Design principles

1. The LLM is not the detector.
   - The model already selected the event.
   - The LLM explains and critiques the event.

2. The LLM must not infer legal intent.
   - Use “spoofing-like”, “consistent with”, “surveillance cue”.
   - Avoid “the client spoofed” or “manipulation occurred”.

3. Every analysis must be reproducible.
   - Save dossier.
   - Save prompt.
   - Save model name.
   - Save command/backend.
   - Save response.
   - Save timestamp.

4. The local LLM should receive compact evidence, not the full dashboard HTML.

5. The output should be structured and comparable across events.

---

## Files likely to change

### Create

- `prompts/spoofing_surveillance_analyst.md`
- `scripts/build_spoofing_event_dossier.py`
- `scripts/analyze_spoofing_event_with_llm.py`
- `tests/lob/test_spoofing_event_dossier.py`
- `tests/lob/test_spoofing_event_llm_analysis.py`

### Modify

- `scripts/build_spoofing_event_review_dashboard.py`
- `docs/spoofing_multilevel_results.md` maybe, only after implementation, to document the LLM review workflow.

### Generated outputs

- `outputs/spoofing_event_review/<run>/llm_reviews/<event_id>/dossier.md`
- `outputs/spoofing_event_review/<run>/llm_reviews/<event_id>/dossier.json`
- `outputs/spoofing_event_review/<run>/llm_reviews/<event_id>/prompt.md`
- `outputs/spoofing_event_review/<run>/llm_reviews/<event_id>/response.md`
- `outputs/spoofing_event_review/<run>/llm_reviews/<event_id>/metadata.json`

---

## Task 1: Create the stable surveillance-analyst prompt

**Objective:** Add a versioned markdown instruction file that tells the local LLM how to analyze a candidate spoofing-like event without overclaiming intent.

**Files:**
- Create: `prompts/spoofing_surveillance_analyst.md`

**Step 1: Write the prompt file**

Create `prompts/spoofing_surveillance_analyst.md` with this content:

```markdown
# Spoofing Surveillance Analyst Prompt

You are a market-surveillance analyst reviewing one candidate spoofing-like event.

Your task is to explain whether the event is consistent with the provided multilevel DWI/MSCI/MCPS spoofing framework. You are not deciding legal intent, and you must not claim that market manipulation occurred.

## Context

The event has already been selected by a quantitative surveillance model. The model looks for this sequence:

1. A client has recent opposite-side candidate deceptive liquidity visible before a small passive execution.
2. The same client receives a small execution on the opposite side.
3. The same client cancels one of the pre-existing candidate deceptive order IDs shortly after the execution.

The key metrics are:

- DWI: multilevel distance-weighted imbalance of the client's top-N order-book footprint. Positive values mean ask-heavy, negative values mean bid-heavy.
- SCI: absolute change in DWI from before the execution to after the cancellation window.
- Opposite-side collapse: fraction of weighted candidate-side liquidity that disappears after execution.
- Same-side collapse: fraction of weighted execution-side liquidity that disappears after execution.
- MSCI: SCI multiplied by opposite-side collapse and the positive excess of opposite-side over same-side collapse.
- MCPS: client-level frequency of high-MSCI executions above a selected threshold.

## Evidence rules

Use only the event dossier provided below. Do not invent missing facts. If a fact is absent, state that it is absent.

Separate:

- observed facts;
- model-based interpretation;
- uncertainty;
- possible benign explanations.

Use careful language:

- "spoofing-like"
- "consistent with the model"
- "surveillance cue"
- "requires human review"

Do not use accusatory language such as:

- "the client spoofed"
- "manipulation occurred"
- "illegal intent"

## Required output format

Write the review in markdown with exactly these sections:

# Surveillance review for event EVENT_ID

## 1. Short conclusion

Give a 3-5 sentence summary. State whether the event is weak, moderate, strong, or inconclusive as a spoofing-like surveillance cue.

## 2. Observed facts

Bullet the directly observed facts from the dossier: client, execution side, deceptive side, timestamps, quantities, order IDs, best bid/ask if available.

## 3. Model-based interpretation

Explain DWI, SCI, collapse, and MSCI for this event in plain language.

## 4. Spoofing-timeline consistency

Use this table:

| Stage | Evidence in dossier | Assessment |
|---|---|---|
| Pre-execution/posturing | ... | ... |
| Small execution | ... | ... |
| Post-execution cancellation | ... | ... |

## 5. LOB and queue evidence

Explain where the candidate liquidity sat in the book, what share of the level it represented, and whether the queue evidence supports or weakens the suspicious interpretation.

## 6. Parameter robustness

If a kappa/lambda robustness table is present, summarize whether the event is robust across parameter settings. If no robustness table is present, say so.

## 7. Alternative benign explanations

List plausible non-manipulative explanations, such as quote refresh, inventory management, adverse-selection response, market-wide movement, stale quote cancellation, or unrelated same-client activity.

## 8. Recommended human checks

List concrete next checks for a human analyst.

## 9. Final assessment

Choose one label:

- weak spoofing-like cue
- moderate spoofing-like cue
- strong spoofing-like cue
- inconclusive

Then provide a one-paragraph justification.
```

**Step 2: Verify file exists**

Run:

```bash
python - <<'PY'
from pathlib import Path
p = Path('prompts/spoofing_surveillance_analyst.md')
assert p.exists()
assert 'Do not invent missing facts' in p.read_text()
assert 'Required output format' in p.read_text()
print('prompt ok')
PY
```

Expected: `prompt ok`

**Step 3: Commit**

```bash
git add prompts/spoofing_surveillance_analyst.md
git commit -m "docs: add spoofing surveillance analyst prompt"
```

---

## Task 2: Add pure helper functions for event dossier construction

**Objective:** Create testable pure helpers that extract the event rows, event log rows, queue rows, and parameter sensitivity rows for one event id.

**Files:**
- Create: `scripts/build_spoofing_event_dossier.py`
- Test: `tests/lob/test_spoofing_event_dossier.py`

**Step 1: Write failing tests**

Create `tests/lob/test_spoofing_event_dossier.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

import polars as pl

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "build_spoofing_event_dossier.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("build_spoofing_event_dossier", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_select_event_bundle_filters_all_inputs_by_event_id():
    module = _load_module()
    events = pl.DataFrame(
        [
            {"review_event_id": "S10", "client_id": "C1", "MSCI": 1.2},
            {"review_event_id": "S11", "client_id": "C2", "MSCI": 0.4},
        ]
    )
    log = pl.DataFrame(
        [
            {"review_event_id": "S10", "sort_index": 10, "event_class": "fill"},
            {"review_event_id": "S11", "sort_index": 11, "event_class": "cancel"},
        ]
    )
    queue = pl.DataFrame(
        [
            {"review_event_id": "S10", "snapshot_phase": "pre", "side": "bid", "level": 1, "visible_qty": 100},
            {"review_event_id": "S11", "snapshot_phase": "pre", "side": "ask", "level": 1, "visible_qty": 200},
        ]
    )

    bundle = module.select_event_bundle("S10", events, log, queue)

    assert bundle.event["review_event_id"] == "S10"
    assert bundle.event_log.height == 1
    assert bundle.queue.height == 1
    assert bundle.event_log["sort_index"].to_list() == [10]


def test_build_stage_depth_summary_aggregates_candidate_and_total_volume():
    module = _load_module()
    queue = pl.DataFrame(
        [
            {
                "snapshot_phase": "pre",
                "side": "bid",
                "level": 1,
                "price": 10.0,
                "level_visible_qty": 1000,
                "visible_qty": 250,
                "is_candidate_deceptive_order": True,
                "is_matched_deceptive_cancel_order": False,
                "client_queue_dict": '{"C1":{"perc_vol":0.25,"priority":1}}',
            },
            {
                "snapshot_phase": "pre",
                "side": "bid",
                "level": 1,
                "price": 10.0,
                "level_visible_qty": 1000,
                "visible_qty": 100,
                "is_candidate_deceptive_order": False,
                "is_matched_deceptive_cancel_order": False,
                "client_queue_dict": '{"C2":{"perc_vol":0.10,"priority":2}}',
            },
        ]
    )

    summary = module.build_stage_depth_summary(queue)

    assert summary.height == 1
    row = summary.row(0, named=True)
    assert row["phase"] == "pre"
    assert row["side"] == "bid"
    assert row["level"] == 1
    assert row["total_visible_qty"] == 1000
    assert row["candidate_visible_qty"] == 250
    assert row["candidate_level_share"] == 0.25
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_dossier.py
```

Expected: FAIL because `scripts/build_spoofing_event_dossier.py` does not exist.

**Step 3: Create minimal implementation**

Create `scripts/build_spoofing_event_dossier.py` with:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl


@dataclass(frozen=True)
class EventBundle:
    event: dict[str, Any]
    event_log: pl.DataFrame
    queue: pl.DataFrame


def select_event_bundle(
    event_id: str,
    review_events: pl.DataFrame,
    event_log: pl.DataFrame,
    queue: pl.DataFrame,
) -> EventBundle:
    event_rows = review_events.filter(pl.col("review_event_id") == event_id)
    if event_rows.height != 1:
        raise ValueError(f"expected exactly one event for {event_id}, found {event_rows.height}")
    return EventBundle(
        event=event_rows.row(0, named=True),
        event_log=event_log.filter(pl.col("review_event_id") == event_id).sort("sort_index"),
        queue=queue.filter(pl.col("review_event_id") == event_id).sort(
            ["snapshot_sort_index", "side", "level", "queue_position"]
        ),
    )


def build_stage_depth_summary(queue: pl.DataFrame) -> pl.DataFrame:
    if queue.is_empty():
        return pl.DataFrame()
    candidate_expr = (
        pl.when(pl.col("is_candidate_deceptive_order") | pl.col("is_matched_deceptive_cancel_order"))
        .then(pl.col("visible_qty"))
        .otherwise(0)
        .sum()
        .alias("candidate_visible_qty")
    )
    grouped = (
        queue.group_by(["snapshot_phase", "side", "level", "price"])
        .agg(
            pl.max("level_visible_qty").alias("total_visible_qty"),
            candidate_expr,
            pl.first("client_queue_dict").alias("client_queue_dict"),
        )
        .with_columns(
            (pl.col("candidate_visible_qty") / pl.col("total_visible_qty")).fill_nan(0).fill_null(0).alias(
                "candidate_level_share"
            ),
            pl.col("snapshot_phase").alias("phase"),
        )
        .select(
            [
                "phase",
                "side",
                "level",
                "price",
                "total_visible_qty",
                "candidate_visible_qty",
                "candidate_level_share",
                "client_queue_dict",
            ]
        )
        .sort(["phase", "side", "level"])
    )
    return grouped
```

**Step 4: Run tests to verify pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_dossier.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/build_spoofing_event_dossier.py tests/lob/test_spoofing_event_dossier.py
git commit -m "feat: add spoofing event dossier helpers"
```

---

## Task 3: Render a markdown dossier for one event

**Objective:** Add markdown rendering so the LLM receives a compact event dossier rather than raw parquet or dashboard HTML.

**Files:**
- Modify: `scripts/build_spoofing_event_dossier.py`
- Modify: `tests/lob/test_spoofing_event_dossier.py`

**Step 1: Add failing test**

Append to `tests/lob/test_spoofing_event_dossier.py`:

```python
def test_render_dossier_markdown_contains_core_sections():
    module = _load_module()
    event = {
        "review_event_id": "S10",
        "client_id": "C1",
        "event_ts": "2024-01-01T10:00:00",
        "execution_side": "ask",
        "deceptive_side": "bid",
        "fill_qty": 100,
        "DWI_pre_window": -0.7,
        "DWI_post_window": -0.1,
        "SCI": 0.6,
        "MSCI": 0.3,
        "candidate_deceptive_visible_qty_pre": 1000,
        "matched_deceptive_cancel_visible_qty_window": 800,
    }
    log = pl.DataFrame([{"sort_index": 10, "event_ts": "2024-01-01T10:00:00", "event_class": "fill"}])
    depth = pl.DataFrame(
        [
            {
                "phase": "pre",
                "side": "bid",
                "level": 1,
                "price": 9.99,
                "total_visible_qty": 1000,
                "candidate_visible_qty": 800,
                "candidate_level_share": 0.8,
                "client_queue_dict": "{}",
            }
        ]
    )

    text = module.render_dossier_markdown(event=event, event_log=log, stage_depth=depth, robustness=pl.DataFrame())

    assert "# Event dossier: S10" in text
    assert "## Model scores" in text
    assert "## Stage depth summary" in text
    assert "## Actual event log" in text
    assert "DWI_pre_window" in text
    assert "MSCI" in text
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_dossier.py::test_render_dossier_markdown_contains_core_sections
```

Expected: FAIL because `render_dossier_markdown` is missing.

**Step 3: Implement markdown renderer**

Add to `scripts/build_spoofing_event_dossier.py`:

```python
def _markdown_table(df: pl.DataFrame, columns: list[str], limit: int | None = None) -> str:
    if df.is_empty():
        return "No rows.\n"
    shown = df.select([col for col in columns if col in df.columns])
    if limit is not None:
        shown = shown.head(limit)
    cols = shown.columns
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in shown.to_dicts():
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in cols) + " |")
    return "\n".join(lines) + "\n"


def render_dossier_markdown(
    *,
    event: dict[str, Any],
    event_log: pl.DataFrame,
    stage_depth: pl.DataFrame,
    robustness: pl.DataFrame,
) -> str:
    event_id = event.get("review_event_id")
    lines = [
        f"# Event dossier: {event_id}",
        "",
        "## Event identity",
        "",
        f"- event_id: {event_id}",
        f"- client_id: {event.get('client_id')}",
        f"- event_ts: {event.get('event_ts')}",
        f"- execution_side: {event.get('execution_side')}",
        f"- deceptive_side: {event.get('deceptive_side')}",
        f"- fill_qty: {event.get('fill_qty')}",
        "",
        "## Model scores",
        "",
    ]
    score_keys = [
        "DWI_pre_window",
        "DWI_post_window",
        "SCI",
        "collapse_opposite_side",
        "collapse_same_side",
        "MSCI",
        "candidate_deceptive_visible_qty_pre",
        "matched_deceptive_cancel_visible_qty_window",
        "matched_deceptive_cancel_fraction_window",
        "candidate_deceptive_order_ids_pre",
        "matched_deceptive_cancel_order_ids_window",
    ]
    for key in score_keys:
        if key in event:
            lines.append(f"- {key}: {event.get(key)}")
    lines += ["", "## Stage depth summary", ""]
    lines.append(
        _markdown_table(
            stage_depth,
            [
                "phase",
                "side",
                "level",
                "price",
                "total_visible_qty",
                "candidate_visible_qty",
                "candidate_level_share",
                "client_queue_dict",
            ],
        )
    )
    lines += ["", "## Actual event log", ""]
    lines.append(
        _markdown_table(
            event_log,
            [
                "sort_index",
                "event_ts",
                "event_class",
                "side",
                "price",
                "ORDERID",
                "client_id",
                "leaves_qty",
                "displayed_qty",
                "last_shares",
                "is_execution_order",
                "is_candidate_deceptive_order",
                "is_matched_deceptive_cancel_order",
            ],
            limit=200,
        )
    )
    lines += ["", "## Kappa/lambda robustness", ""]
    lines.append(_markdown_table(robustness, robustness.columns if not robustness.is_empty() else []))
    return "\n".join(lines).strip() + "\n"
```

**Step 4: Run tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_dossier.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/build_spoofing_event_dossier.py tests/lob/test_spoofing_event_dossier.py
git commit -m "feat: render spoofing event dossier markdown"
```

---

## Task 4: Add kappa/lambda robustness extraction for a selected event

**Objective:** Include whether the selected event appears across sensitivity runs and how MSCI/rank changes.

**Files:**
- Modify: `scripts/build_spoofing_event_dossier.py`
- Modify: `tests/lob/test_spoofing_event_dossier.py`

**Step 1: Add failing test**

Append:

```python
def test_build_parameter_robustness_uses_sort_index_across_runs(tmp_path):
    module = _load_module()
    root = tmp_path / "grid"
    run = root / "kappa_1.0_lambda_2.0"
    run.mkdir(parents=True)
    (run / "metadata.json").write_text('{"kappa": 1.0, "lambda_": 2.0}')
    pl.DataFrame(
        [
            {"sort_index": 10, "MSCI": 0.9, "has_matched_deceptive_cancel_window": True},
            {"sort_index": 11, "MSCI": 0.2, "has_matched_deceptive_cancel_window": True},
        ]
    ).write_parquet(run / "execution_metrics.parquet")

    out = module.build_parameter_robustness(event_sort_index=10, parameter_grid_root=root)

    assert out.height == 1
    row = out.row(0, named=True)
    assert row["kappa"] == 1.0
    assert row["lambda"] == 2.0
    assert row["matched"] is True
    assert row["MSCI"] == 0.9
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_dossier.py::test_build_parameter_robustness_uses_sort_index_across_runs
```

Expected: FAIL because function is missing.

**Step 3: Implement robustness helper**

Add:

```python
def build_parameter_robustness(event_sort_index: int, parameter_grid_root: Path | None) -> pl.DataFrame:
    if parameter_grid_root is None:
        return pl.DataFrame()
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(parameter_grid_root.glob("kappa_*_lambda_*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        execution_path = metadata_path.parent / "execution_metrics.parquet"
        if not execution_path.exists():
            continue
        metrics = pl.read_parquet(execution_path)
        if "sort_index" not in metrics.columns:
            continue
        hit = metrics.filter(pl.col("sort_index") == event_sort_index)
        if hit.is_empty():
            rows.append(
                {
                    "kappa": metadata.get("kappa"),
                    "lambda": metadata.get("lambda_"),
                    "matched": False,
                    "MSCI": None,
                    "rank_by_MSCI": None,
                }
            )
            continue
        ranked = metrics.with_row_index("rank_by_MSCI", offset=1).sort("MSCI", descending=True)
        ranked_hit = ranked.filter(pl.col("sort_index") == event_sort_index)
        row = hit.row(0, named=True)
        rows.append(
            {
                "kappa": metadata.get("kappa"),
                "lambda": metadata.get("lambda_"),
                "matched": bool(row.get("has_matched_deceptive_cancel_window")),
                "MSCI": row.get("MSCI"),
                "SCI": row.get("SCI"),
                "collapse_opposite_side": row.get("collapse_opposite_side"),
                "collapse_same_side": row.get("collapse_same_side"),
                "rank_by_MSCI": ranked_hit.row(0, named=True).get("rank_by_MSCI") if not ranked_hit.is_empty() else None,
            }
        )
    return pl.DataFrame(rows).sort(["kappa", "lambda"]) if rows else pl.DataFrame()
```

**Implementation note:** The ranking snippet above has a subtle risk: `with_row_index` before sort does not rank after sorting. Implementer should correct it by sorting first, then adding row index:

```python
ranked = metrics.sort("MSCI", descending=True).with_row_index("rank_by_MSCI", offset=1)
```

Use the corrected version in code.

**Step 4: Run tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_dossier.py
```

Expected: PASS.

**Step 5: Commit**

```bash
git add scripts/build_spoofing_event_dossier.py tests/lob/test_spoofing_event_dossier.py
git commit -m "feat: add kappa lambda robustness to event dossiers"
```

---

## Task 5: Add CLI for dossier generation

**Objective:** Allow generating one event dossier from the review directory.

**Files:**
- Modify: `scripts/build_spoofing_event_dossier.py`
- Modify: `tests/lob/test_spoofing_event_dossier.py`

**Step 1: Add CLI behavior**

Add CLI args:

```python
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build an LLM-ready dossier for one spoofing-like event.")
    parser.add_argument("--review-dir", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--parameter-grid-root", type=Path, default=None)
    return parser.parse_args(argv)
```

Add main:

```python
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    review_dir = args.review_dir
    output_dir = args.output_dir or review_dir / "llm_reviews" / args.event_id
    output_dir.mkdir(parents=True, exist_ok=True)

    events = pl.read_parquet(review_dir / "matched_spoofing_events.parquet")
    event_log = pl.read_parquet(review_dir / "matched_spoofing_event_log.parquet")
    queue = pl.read_parquet(review_dir / "matched_spoofing_lob_queue.parquet")

    bundle = select_event_bundle(args.event_id, events, event_log, queue)
    stage_depth = build_stage_depth_summary(bundle.queue)
    robustness = build_parameter_robustness(int(bundle.event["sort_index"]), args.parameter_grid_root)
    markdown = render_dossier_markdown(
        event=bundle.event,
        event_log=bundle.event_log,
        stage_depth=stage_depth,
        robustness=robustness,
    )

    (output_dir / "dossier.md").write_text(markdown)
    (output_dir / "dossier.json").write_text(
        json.dumps(
            {
                "event": bundle.event,
                "event_log": bundle.event_log.to_dicts(),
                "stage_depth": stage_depth.to_dicts(),
                "robustness": robustness.to_dicts(),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
    print(output_dir / "dossier.md")


if __name__ == "__main__":
    main()
```

**Step 2: Add CLI smoke test**

If test fixtures become too verbose, keep CLI test minimal by writing tiny parquet files to `tmp_path` and invoking `main([...])` directly.

Append:

```python
def test_main_writes_dossier_files(tmp_path):
    module = _load_module()
    review_dir = tmp_path / "review"
    review_dir.mkdir()
    pl.DataFrame([{"review_event_id": "S10", "sort_index": 10, "client_id": "C1", "MSCI": 0.3}]).write_parquet(
        review_dir / "matched_spoofing_events.parquet"
    )
    pl.DataFrame([{"review_event_id": "S10", "sort_index": 10, "event_class": "fill"}]).write_parquet(
        review_dir / "matched_spoofing_event_log.parquet"
    )
    pl.DataFrame(
        [
            {
                "review_event_id": "S10",
                "snapshot_phase": "pre",
                "snapshot_sort_index": 9,
                "side": "bid",
                "level": 1,
                "price": 10.0,
                "queue_position": 1,
                "level_visible_qty": 100,
                "visible_qty": 100,
                "is_candidate_deceptive_order": True,
                "is_matched_deceptive_cancel_order": False,
                "client_queue_dict": "{}",
            }
        ]
    ).write_parquet(review_dir / "matched_spoofing_lob_queue.parquet")

    out = tmp_path / "out"
    module.main(["--review-dir", str(review_dir), "--event-id", "S10", "--output-dir", str(out)])

    assert (out / "dossier.md").exists()
    assert (out / "dossier.json").exists()
```

**Step 3: Run tests**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_dossier.py
```

Expected: PASS.

**Step 4: Run real dossier generation**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/build_spoofing_event_dossier.py \
  --review-dir outputs/spoofing_event_review/risanamento_top3_timing_window \
  --event-id S12345 \
  --parameter-grid-root outputs/spoofing_metrics/kappa_lambda_sensitivity_top3
```

Replace `S12345` with an actual event id from:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python - <<'PY'
import polars as pl
print(pl.read_parquet('outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_events.parquet').select('review_event_id').head(5))
PY
```

Expected: prints the dossier path.

**Step 5: Commit**

```bash
git add scripts/build_spoofing_event_dossier.py tests/lob/test_spoofing_event_dossier.py
git commit -m "feat: add spoofing event dossier CLI"
```

---

## Task 6: Add Ollama analyzer script

**Objective:** Combine the stable prompt with a dossier and call local Ollama `gemma4`, saving prompt, response, and metadata.

**Files:**
- Create: `scripts/analyze_spoofing_event_with_llm.py`
- Test: `tests/lob/test_spoofing_event_llm_analysis.py`

**Step 1: Write failing tests using a fake runner**

Create `tests/lob/test_spoofing_event_llm_analysis.py`:

```python
from __future__ import annotations

import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "analyze_spoofing_event_with_llm.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("analyze_spoofing_event_with_llm", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_compose_prompt_combines_instruction_and_dossier():
    module = _load_module()
    text = module.compose_prompt("SYSTEM INSTRUCTIONS", "# Event dossier: S10")
    assert "SYSTEM INSTRUCTIONS" in text
    assert "# Event dossier: S10" in text
    assert "Now analyze the event dossier" in text


def test_write_analysis_artifacts_saves_response_and_metadata(tmp_path):
    module = _load_module()
    out = tmp_path / "review"
    module.write_analysis_artifacts(
        output_dir=out,
        prompt_text="PROMPT",
        response_text="RESPONSE",
        metadata={"model": "gemma4", "backend": "ollama"},
    )
    assert (out / "prompt.md").read_text() == "PROMPT"
    assert (out / "response.md").read_text() == "RESPONSE"
    assert json.loads((out / "metadata.json").read_text())["model"] == "gemma4"
```

**Step 2: Run tests to verify failure**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_llm_analysis.py
```

Expected: FAIL because script does not exist.

**Step 3: Implement analyzer script**

Create `scripts/analyze_spoofing_event_with_llm.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze a spoofing event dossier with a local Ollama model.")
    parser.add_argument("--dossier", type=Path, required=True)
    parser.add_argument("--prompt", type=Path, default=Path("prompts/spoofing_surveillance_analyst.md"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default="gemma4")
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--timeout-seconds", type=int, default=180)
    return parser.parse_args(argv)


def compose_prompt(instruction_text: str, dossier_text: str) -> str:
    return (
        instruction_text.strip()
        + "\n\n---\n\n"
        + "Now analyze the event dossier below. Use only this dossier as evidence.\n\n"
        + dossier_text.strip()
        + "\n"
    )


def call_ollama(*, model: str, prompt_text: str, timeout_seconds: int) -> str:
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt_text,
        text=True,
        capture_output=True,
        timeout=timeout_seconds,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ollama failed with code {result.returncode}: {result.stderr}")
    return result.stdout.strip() + "\n"


def write_analysis_artifacts(
    *,
    output_dir: Path,
    prompt_text: str,
    response_text: str,
    metadata: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "prompt.md").write_text(prompt_text)
    (output_dir / "response.md").write_text(response_text)
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, sort_keys=True, default=str))


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    instruction_text = args.prompt.read_text()
    dossier_text = args.dossier.read_text()
    prompt_text = compose_prompt(instruction_text, dossier_text)
    response_text = call_ollama(model=args.model, prompt_text=prompt_text, timeout_seconds=args.timeout_seconds)
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "backend": "ollama",
        "model": args.model,
        "temperature": args.temperature,
        "dossier": str(args.dossier),
        "prompt_file": str(args.prompt),
        "timeout_seconds": args.timeout_seconds,
    }
    write_analysis_artifacts(
        output_dir=args.output_dir,
        prompt_text=prompt_text,
        response_text=response_text,
        metadata=metadata,
    )
    print(args.output_dir / "response.md")


if __name__ == "__main__":
    main()
```

**Important note:** `ollama run` does not necessarily honor `temperature` directly. Keep `temperature` in metadata for provenance, but if exact decoding control is required, implement the Ollama HTTP API later.

**Step 4: Run unit tests**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_llm_analysis.py
```

Expected: PASS.

**Step 5: Run one real LLM analysis**

After generating a dossier for a real event, run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/analyze_spoofing_event_with_llm.py \
  --dossier outputs/spoofing_event_review/risanamento_top3_timing_window/llm_reviews/S12345/dossier.md \
  --prompt prompts/spoofing_surveillance_analyst.md \
  --output-dir outputs/spoofing_event_review/risanamento_top3_timing_window/llm_reviews/S12345 \
  --model gemma4 \
  --timeout-seconds 180
```

Expected:

```text
outputs/spoofing_event_review/risanamento_top3_timing_window/llm_reviews/S12345/response.md
```

**Step 6: Commit**

```bash
git add scripts/analyze_spoofing_event_with_llm.py tests/lob/test_spoofing_event_llm_analysis.py
git commit -m "feat: add local ollama spoofing event analyzer"
```

---

## Task 7: Add dashboard support for precomputed LLM reviews

**Objective:** Show saved LLM review markdown in the event-review dashboard when available, without making the static browser execute local commands.

**Files:**
- Modify: `scripts/build_spoofing_event_review_dashboard.py`
- Modify: `tests/lob/test_spoofing_event_review_dashboard.py`

**Step 1: Add test expectation**

Modify `tests/lob/test_spoofing_event_review_dashboard.py` to check the generated HTML includes an LLM review container and a safe command hint.

Example assertion:

```python
assert "LLM surveillance review" in html
assert "llmReviews" in html
assert "No precomputed LLM review found" in html
```

**Step 2: Add optional review loading to the dashboard script**

In `scripts/build_spoofing_event_review_dashboard.py`, add a helper:

```python
def _load_llm_reviews(output_dir: Path) -> dict[str, str]:
    root = output_dir / "llm_reviews"
    reviews: dict[str, str] = {}
    if not root.exists():
        return reviews
    for response_path in root.glob("*/response.md"):
        reviews[response_path.parent.name] = response_path.read_text()
    return reviews
```

Add to `write_dashboard` signature:

```python
llm_reviews: dict[str, str] | None = None
```

Add JS data:

```javascript
const llmReviews = {...};
```

Add HTML card after the LOB plot:

```html
<div class="card"><h2>LLM surveillance review</h2><div id="llmReview"></div></div>
```

Add renderer:

```javascript
function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}
function renderLLMReview(ev) {
  const review = llmReviews[ev.review_event_id];
  if (review) {
    document.getElementById('llmReview').innerHTML = `<pre>${escapeHtml(review)}</pre>`;
  } else {
    document.getElementById('llmReview').innerHTML = `<p>No precomputed LLM review found for ${ev.review_event_id}.</p><p>Generate it with <code>scripts/build_spoofing_event_dossier.py</code> and <code>scripts/analyze_spoofing_event_with_llm.py</code>.</p>`;
  }
}
```

Modify `update`:

```javascript
function update(id) {
  const ev = reviewEvents.find(r => r.review_event_id === id);
  renderSummary(ev);
  renderLOB(ev);
  renderLLMReview(ev);
  renderEventTable(ev);
}
```

**Step 3: Run dashboard tests**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_event_review_dashboard.py
```

Expected: PASS.

**Step 4: Regenerate dashboard**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/build_spoofing_event_review_dashboard.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --execution-metrics outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/execution_metrics.parquet \
  --candidate-deceptive-orders outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/candidate_deceptive_orders.parquet \
  --parameter-grid-root outputs/spoofing_metrics/kappa_lambda_sensitivity_top3 \
  --output-dir outputs/spoofing_event_review/risanamento_top3_timing_window \
  --top-n 10 \
  --pre-window-seconds 30 \
  --post-window-seconds 5
```

Expected: dashboard includes an LLM review card.

**Step 5: Commit**

```bash
git add scripts/build_spoofing_event_review_dashboard.py tests/lob/test_spoofing_event_review_dashboard.py
git commit -m "feat: show precomputed llm reviews in spoofing dashboard"
```

---

## Task 8: Add a batch script for all matched events

**Objective:** Generate dossiers and Ollama reviews for all matched events with one command, but keep it explicit and reproducible.

**Files:**
- Create: `scripts/batch_analyze_spoofing_events_with_llm.py`
- Test: optional if mostly orchestration; at least py_compile and a `--dry-run` test.

**Step 1: Implement dry-run friendly batch script**

Create script that:

- reads `matched_spoofing_events.parquet`;
- loops over `review_event_id`;
- for each event:
  - call dossier builder;
  - call analyzer unless `--dry-run`;
- skips events with existing `response.md` unless `--overwrite`.

Suggested args:

```python
parser.add_argument("--review-dir", type=Path, required=True)
parser.add_argument("--parameter-grid-root", type=Path, default=None)
parser.add_argument("--prompt", type=Path, default=Path("prompts/spoofing_surveillance_analyst.md"))
parser.add_argument("--model", default="gemma4")
parser.add_argument("--limit", type=int, default=None)
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--dry-run", action="store_true")
```

**Step 2: Run dry run**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/batch_analyze_spoofing_events_with_llm.py \
  --review-dir outputs/spoofing_event_review/risanamento_top3_timing_window \
  --parameter-grid-root outputs/spoofing_metrics/kappa_lambda_sensitivity_top3 \
  --model gemma4 \
  --limit 3 \
  --dry-run
```

Expected: prints 3 event IDs and commands it would run; no LLM call.

**Step 3: Run one-event batch smoke**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/batch_analyze_spoofing_events_with_llm.py \
  --review-dir outputs/spoofing_event_review/risanamento_top3_timing_window \
  --parameter-grid-root outputs/spoofing_metrics/kappa_lambda_sensitivity_top3 \
  --model gemma4 \
  --limit 1
```

Expected: one `response.md` appears under `llm_reviews/<event_id>/`.

**Step 4: Commit**

```bash
git add scripts/batch_analyze_spoofing_events_with_llm.py
git commit -m "feat: add batch llm analysis for spoofing events"
```

---

## Task 9: Document the workflow

**Objective:** Add concise documentation explaining how to generate and inspect LLM reviews.

**Files:**
- Modify: `docs/spoofing_multilevel_results.md` or create `docs/spoofing_llm_review_workflow.md`

**Recommended:** Create `docs/spoofing_llm_review_workflow.md` to keep it separate.

**Content outline:**

```markdown
# Spoofing Event LLM Review Workflow

## Purpose

The LLM review explains candidate spoofing-like events selected by the DWI/MSCI/MCPS model. It does not decide legal intent.

## Generate one event dossier

[command]

## Analyze one event with Ollama/gemma4

[command]

## Batch analyze events

[command]

## Regenerate dashboard with saved reviews

[command]

## Output files

[list]

## Caveats

- local LLM output is explanatory, not evidence by itself;
- use only as a human-review aid;
- check prompt, dossier, and response together.
```

**Verification:**

```bash
python - <<'PY'
from pathlib import Path
p = Path('docs/spoofing_llm_review_workflow.md')
assert p.exists()
assert 'Ollama' in p.read_text()
assert 'not decide legal intent' in p.read_text()
print('docs ok')
PY
```

**Commit:**

```bash
git add docs/spoofing_llm_review_workflow.md
git commit -m "docs: add spoofing llm review workflow"
```

---

## Task 10: Final integration validation

**Objective:** Verify the whole workflow end-to-end on one real event without overclaiming results.

**Files:**
- No code changes unless failures are found.

**Step 1: Run tests**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q
```

Expected: all tests pass.

**Step 2: Generate one real dossier**

Choose an event id:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python - <<'PY'
import polars as pl
print(pl.read_parquet('outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_events.parquet').select('review_event_id').head(1))
PY
```

Then run dossier builder for that event.

**Step 3: Run one real Ollama analysis**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/analyze_spoofing_event_with_llm.py \
  --dossier outputs/spoofing_event_review/risanamento_top3_timing_window/llm_reviews/<EVENT_ID>/dossier.md \
  --prompt prompts/spoofing_surveillance_analyst.md \
  --output-dir outputs/spoofing_event_review/risanamento_top3_timing_window/llm_reviews/<EVENT_ID> \
  --model gemma4 \
  --timeout-seconds 180
```

Expected:

- `prompt.md` exists;
- `response.md` exists;
- `metadata.json` exists;
- response contains all required sections.

**Step 4: Regenerate dashboard and inspect**

Regenerate event-review dashboard. Open:

```text
outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_event_review_dashboard.html
```

Verify:

- selected event shows the LLM surveillance review;
- events without generated reviews show a clear message;
- no live local command is executed from static HTML.

**Step 5: Final commit**

If all previous commits were made task-by-task, this may only include generated docs or small fixes.

```bash
git status --short
git add <remaining intended files>
git commit -m "test: validate llm spoofing event review workflow"
```

---

## Risks and tradeoffs

### Static dashboard cannot safely run local commands

A standalone HTML file should not directly execute `ollama` or Python scripts. That would require a local server. The recommended first version displays precomputed LLM reviews and provides commands for missing reviews.

### LLM overclaiming

The prompt must strongly forbid legal-intent conclusions. The review should remain a surveillance explanation, not a legal determination.

### Dossier size

The event log and queue can be large. The dossier should stay compact:

- aggregate queue by phase/side/level;
- include full event log only up to a reasonable cap;
- include order IDs and candidate/matched flags;
- keep raw parquet available for deeper inspection.

### Parameter robustness

An event may appear across all kappa/lambda settings but with different MSCI ranks. The LLM should be told whether the event is parameter-robust or parameter-fragile.

### Ollama model behavior

`gemma4` may produce verbose or inconsistent output. Keep temperature low if using the HTTP API later. With CLI, save outputs and inspect. If the format is unstable, add a postprocessing check that verifies required markdown headings exist.

---

## Open questions before implementation

1. Should LLM reviews be generated only for the 32 matched events, or also for top non-matched high-MSCI events as contrast cases?
2. Should the dashboard show raw markdown in a `<pre>` block or render markdown to HTML?
   - First version: `<pre>` is safer and dependency-free.
   - Later version: use a small markdown renderer if desired.
3. Should we use Ollama CLI first or Ollama HTTP API first?
   - Recommended first: CLI for simplicity.
   - Later: HTTP API for temperature/options and streaming.
4. Should `gemma4` be the default hardcoded model, or configurable with default `gemma4`?
   - Recommended: configurable, default `gemma4`.
5. Should all reviews be regenerated when kappa/lambda settings change?
   - Recommended: no; reviews should record selected parameter grid and be regenerated explicitly.

---

## Recommended implementation order

1. Prompt file.
2. Dossier helper functions.
3. Dossier markdown CLI.
4. Robustness table in dossier.
5. Ollama analyzer script.
6. Dashboard display for precomputed reviews.
7. Batch analyzer.
8. Documentation.
9. End-to-end validation on one real event.

Do not implement a live `Analyze` button that directly calls Ollama from static HTML in the first iteration. If a true button is needed later, implement a small local server after this offline workflow is validated.
