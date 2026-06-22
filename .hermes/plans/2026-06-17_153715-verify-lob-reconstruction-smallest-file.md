# Verify LOB Reconstruction on Smallest Parquet Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Verify the current visible limit-order-book reconstruction on one parquet file only, using the smallest available parquet, and save every detected problem with a related possible fix in a Markdown report.

**Architecture:** Add a lightweight verification script that runs the existing reconstruction pipeline on the smallest parquet, computes invariant and quality diagnostics from the generated Parquet outputs, and writes a human-readable problem inventory. The verifier must not change reconstruction semantics; it only observes current behavior and records evidence, severity, and possible fixes.

**Tech Stack:** Python, Polars, existing `spoofing_detection.lob` package, existing `scripts/reconstruct_lob.py` pipeline, pytest.

---

## Current context / assumptions

- Active repository: `/home/danielemdn/Documents/repositories/spoofing_detection`
- Current branch observed during planning: `main`
- Existing reconstruction entry point: `scripts/reconstruct_lob.py`
- Existing core code:
  - `src/spoofing_detection/lob/panel.py`
  - `src/spoofing_detection/lob/normalize.py`
  - `src/spoofing_detection/lob/io.py`
  - `src/spoofing_detection/lob/config.py`
- Existing tests include reconstruction lifecycle tests in `tests/lob/test_reconstruction.py`.
- Existing test command observed to pass during planning:
  - `PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q`
  - Expected current result: `18 passed`
- Available parquet files by size, smallest first:
  1. `data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet` — 20,401,924 bytes
  2. `data/ExportGridData_2026-05-20_102249466_NEXI_23042024_FG.parquet` — 30,320,944 bytes
  3. `data/ExportGridData_2026-05-19_130707032_FERRARI_13062024_FG.parquet` — 56,648,389 bytes
- Verification scope for this plan is only the smallest file:
  - `data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet`
- Do not attempt all-files validation in this plan.
- Do not implement fixes during verification. Save problems and possible fixes in the Markdown report.
- Do not commit unless the user explicitly asks; the repository instruction says not to commit without permission.

---

## Proposed verification output

Create one timestamped verification output folder per run:

```text
outputs/lob_reconstruction/YYYYMMDD_HHMMSS_verify_smallest_risanamento/
  lob_event_state_panel.parquet
  normalized_events.parquet
  agent_event_state_panel.parquet
  active_order_snapshots.parquet
  price_level_depth_snapshots.parquet
  metadata.json
  validation_report.md
  reconstruction_verification_report.md
```

The required final Markdown problem report is:

```text
outputs/lob_reconstruction/YYYYMMDD_HHMMSS_verify_smallest_risanamento/reconstruction_verification_report.md
```

That report must contain a `Problem inventory` section with one row per detected issue:

```markdown
| Severity | Problem | Evidence | Count | Example sort_index | Possible fix | Status |
|---|---|---:|---:|---:|---|---|
```

If no hard failures are found, still write the report and include known limitations as warnings or caveats, for example client ID missingness, shallow validation, approximate queue reconstruction, and unverified all-file generalization.

---

## Verification criteria

For the smallest parquet file, verify at minimum:

1. **Run completeness**
   - `panel_rows == input_rows`
   - `normalized_rows == input_rows`
   - `agent_event_state_rows == input_rows * 2` because current config has firm and client-original dimensions.

2. **Visible book validity**
   - No crossed post state: `book_crossed_post_flag == 0`
   - No locked post state: `book_locked_post_flag == 0`
   - No crossed pre state: `book_crossed_pre_flag == 0`
   - No locked pre state: `book_locked_pre_flag == 0`
   - Best bid and best ask satisfy `best_bid < best_ask` whenever both are present.

3. **Top-N depth consistency**
   - Bid levels are strictly descending by price.
   - Ask levels are strictly ascending by price.
   - Visible quantities at emitted levels are positive or null, never negative.
   - Level 1 price equals the corresponding best bid/ask when present.

4. **Event and normalization issue inventory**
   - Count `lob_issue_flags` by flag.
   - Count `normalization_issue_flags` by flag.
   - For every nonzero issue count, include a problem/limitation row and possible fix.

5. **Marketable-order lifecycle checks**
   - Rows flagged `marketable_order_not_resting` must not produce locked/crossed post state.
   - Positive-residual fill rows that would be marketable should not appear as visible resting liquidity until safe.
   - If violations appear, save examples and suggest improving deferred residual grouping/flush logic.

6. **Stop-limit visibility checks**
   - Stop-limit rows should not seed visible active depth before trigger/transformation.
   - If violations appear, save examples and suggest tightening `is_visible_resting_event()` and lifecycle handling.

7. **Determinism on the same file**
   - Reconstruct the same input twice in memory or into two temporary directories with snapshot mode `none`.
   - Compare deterministic fingerprints of the panel and normalized event output.
   - If fingerprints differ, save the exact mismatch and suggest stabilizing sorting or row ordering.

8. **Scientific limitations**
   - State explicitly that exact queue/FIFO reconstruction is not verified and remains out of scope.
   - State explicitly that all-files validation is not performed in this plan.
   - State whether each detected item is a hard reconstruction error, a data limitation, or an expected caveat.

---

## Step-by-step plan

### Task 1: Add a verifier test skeleton

**Objective:** Create tests that lock the expected behavior of the verification problem inventory without running the full parquet file.

**Files:**
- Create: `tests/lob/test_reconstruction_verifier.py`
- Later create: `scripts/verify_lob_reconstruction.py`

**Step 1: Write failing tests**

Create `tests/lob/test_reconstruction_verifier.py` with tests for the report-building helpers. These tests should use tiny synthetic Polars DataFrames rather than reading the real parquet file.

```python
from __future__ import annotations

import polars as pl

from scripts.verify_lob_reconstruction import (
    Problem,
    check_row_counts,
    check_spread_flags,
    render_markdown_report,
)


def test_check_row_counts_records_mismatch_problem():
    problems = check_row_counts(
        input_rows=3,
        panel_rows=2,
        normalized_rows=3,
        agent_rows=6,
    )

    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert "panel_rows" in problems[0].problem
    assert problems[0].count == 1
    assert "rerun reconstruction" in problems[0].possible_fix.lower()


def test_check_spread_flags_records_crossed_book_problem():
    panel = pl.DataFrame(
        {
            "sort_index": [1, 2],
            "book_crossed_post_flag": [False, True],
            "book_locked_post_flag": [False, False],
            "book_crossed_pre_flag": [False, False],
            "book_locked_pre_flag": [False, False],
            "post_best_bid": [100.0, 101.0],
            "post_best_ask": [101.0, 100.5],
            "pre_best_bid": [None, 100.0],
            "pre_best_ask": [None, 101.0],
        }
    )

    problems = check_spread_flags(panel)

    assert any(p.problem == "Crossed post-book state" for p in problems)
    crossed = next(p for p in problems if p.problem == "Crossed post-book state")
    assert crossed.count == 1
    assert crossed.example_sort_index == 2
    assert "marketable" in crossed.possible_fix.lower()


def test_render_markdown_report_contains_problem_inventory():
    markdown = render_markdown_report(
        source_file="data/example.parquet",
        output_dir="outputs/example",
        summary={"input_rows": 1, "panel_rows": 1},
        problems=[
            Problem(
                severity="warning",
                problem="Example limitation",
                evidence="Synthetic evidence",
                count=1,
                example_sort_index=10,
                possible_fix="Document or fix it.",
                status="open",
            )
        ],
    )

    assert "# LOB Reconstruction Verification Report" in markdown
    assert "## Problem inventory" in markdown
    assert "Example limitation" in markdown
    assert "Document or fix it." in markdown
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_reconstruction_verifier.py
```

Expected: FAIL because `scripts.verify_lob_reconstruction` does not exist yet.

---

### Task 2: Create verifier script and report data model

**Objective:** Add a script that can run reconstruction on one file and write a Markdown verification report.

**Files:**
- Create: `scripts/verify_lob_reconstruction.py`
- Test: `tests/lob/test_reconstruction_verifier.py`

**Step 1: Write minimal implementation**

Create `scripts/verify_lob_reconstruction.py` with this structure:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.io import reconstruct_file
from spoofing_detection.lob.panel import reconstruct_dataframe


@dataclass(frozen=True)
class Problem:
    severity: str
    problem: str
    evidence: str
    count: int
    example_sort_index: int | None
    possible_fix: str
    status: str = "open"


def _bool_sum(df: pl.DataFrame, column: str) -> int:
    if column not in df.columns or df.is_empty():
        return 0
    return int(df.select(pl.col(column).fill_null(False).sum()).item())


def _first_sort_index(df: pl.DataFrame) -> int | None:
    if df.is_empty() or "sort_index" not in df.columns:
        return None
    value = df.select(pl.col("sort_index").first()).item()
    return int(value) if value is not None else None


def check_row_counts(
    *,
    input_rows: int,
    panel_rows: int,
    normalized_rows: int,
    agent_rows: int,
) -> list[Problem]:
    problems: list[Problem] = []
    if panel_rows != input_rows:
        problems.append(
            Problem(
                severity="error",
                problem="panel_rows does not equal input_rows",
                evidence=f"input_rows={input_rows}, panel_rows={panel_rows}",
                count=abs(panel_rows - input_rows),
                example_sort_index=None,
                possible_fix=(
                    "Inspect sort/filter logic in reconstruct_dataframe(); rerun reconstruction "
                    "without max_rows and ensure include_all_events remains true."
                ),
            )
        )
    if normalized_rows != input_rows:
        problems.append(
            Problem(
                severity="error",
                problem="normalized_rows does not equal input_rows",
                evidence=f"input_rows={input_rows}, normalized_rows={normalized_rows}",
                count=abs(normalized_rows - input_rows),
                example_sort_index=None,
                possible_fix="Inspect normalize_event failures and strict enum handling before panel construction.",
            )
        )
    expected_agent_rows = input_rows * 2
    if agent_rows != expected_agent_rows:
        problems.append(
            Problem(
                severity="error",
                problem="agent_event_state_rows does not equal input_rows * 2",
                evidence=f"expected={expected_agent_rows}, observed={agent_rows}",
                count=abs(agent_rows - expected_agent_rows),
                example_sort_index=None,
                possible_fix="Inspect agent_long_rows() and configured agent dimensions in LOBConfig.",
            )
        )
    return problems


def check_spread_flags(panel: pl.DataFrame) -> list[Problem]:
    checks = [
        ("book_crossed_pre_flag", "Crossed pre-book state", "Inspect previous event mutation; active book is already crossed before this event."),
        ("book_crossed_post_flag", "Crossed post-book state", "Inspect marketable order lifecycle handling, deferred residual flushing, and stop-limit visibility."),
        ("book_locked_pre_flag", "Locked pre-book state", "Inspect previous event mutation; active book is already locked before this event."),
        ("book_locked_post_flag", "Locked post-book state", "Inspect marketable New/Fill residual handling and same-price bid/ask active depth."),
    ]
    problems: list[Problem] = []
    for column, label, fix in checks:
        count = _bool_sum(panel, column)
        if count:
            examples = panel.filter(pl.col(column).fill_null(False))
            problems.append(
                Problem(
                    severity="error",
                    problem=label,
                    evidence=f"{column} true on {count} rows",
                    count=count,
                    example_sort_index=_first_sort_index(examples),
                    possible_fix=fix,
                )
            )
    return problems


def check_top_n_depth(panel: pl.DataFrame, *, top_n: int) -> list[Problem]:
    problems: list[Problem] = []
    for prefix in ("pre", "post"):
        for side in ("bid", "ask"):
            for level in range(1, top_n + 1):
                qty_col = f"{prefix}_{side}_level_{level}_visible_qty"
                if qty_col in panel.columns:
                    bad_qty = panel.filter(pl.col(qty_col).is_not_null() & (pl.col(qty_col) < 0))
                    if bad_qty.height:
                        problems.append(
                            Problem(
                                severity="error",
                                problem=f"Negative {prefix} {side} visible quantity at level {level}",
                                evidence=f"{qty_col} < 0 on {bad_qty.height} rows",
                                count=bad_qty.height,
                                example_sort_index=_first_sort_index(bad_qty),
                                possible_fix="Inspect ActiveOrder quantity normalization and fill/cancel mutation logic.",
                            )
                        )
            for level in range(1, top_n):
                left = f"{prefix}_{side}_level_{level}_price"
                right = f"{prefix}_{side}_level_{level + 1}_price"
                if left not in panel.columns or right not in panel.columns:
                    continue
                if side == "bid":
                    bad_order = panel.filter(pl.col(left).is_not_null() & pl.col(right).is_not_null() & (pl.col(left) <= pl.col(right)))
                    expected = "strictly descending"
                else:
                    bad_order = panel.filter(pl.col(left).is_not_null() & pl.col(right).is_not_null() & (pl.col(left) >= pl.col(right)))
                    expected = "strictly ascending"
                if bad_order.height:
                    problems.append(
                        Problem(
                            severity="error",
                            problem=f"{prefix} {side} price levels are not {expected}",
                            evidence=f"{left}/{right} violate order on {bad_order.height} rows",
                            count=bad_order.height,
                            example_sort_index=_first_sort_index(bad_order),
                            possible_fix="Inspect book_summary() grouping/sorting and same-price aggregation before top-N emission.",
                        )
                    )
    return problems


def check_issue_flags(panel: pl.DataFrame, normalized: pl.DataFrame) -> list[Problem]:
    problems: list[Problem] = []
    if "lob_issue_flags" in panel.columns:
        counts = panel.filter(pl.col("lob_issue_flags").is_not_null()).group_by("lob_issue_flags").len().sort("len", descending=True)
        for row in counts.iter_rows(named=True):
            flag = row["lob_issue_flags"]
            count = int(row["len"])
            example = _first_sort_index(panel.filter(pl.col("lob_issue_flags") == flag))
            severity = "warning" if flag in {"marketable_order_not_resting", "modify_for_unseen_order"} else "error"
            if flag == "marketable_order_not_resting":
                fix = "Audit sampled rows; if many are legitimate marketable lifecycles, keep as expected caveat; otherwise improve marketability/deferred-residual classification."
            elif flag == "modify_for_unseen_order":
                fix = "Inspect prior events for the order ID and decide whether missing reload context, partition reset, or hidden-state handling is needed."
            else:
                fix = "Trace the flagged event class and add a targeted state-machine test before changing reconstruction logic."
            problems.append(
                Problem(
                    severity=severity,
                    problem=f"LOB issue flag: {flag}",
                    evidence=f"lob_issue_flags={flag!r} on {count} rows",
                    count=count,
                    example_sort_index=example,
                    possible_fix=fix,
                )
            )
    if "normalization_issue_flags" in normalized.columns:
        exploded = (
            normalized
            .filter(pl.col("normalization_issue_flags").is_not_null())
            .select(["sort_index", pl.col("normalization_issue_flags").str.split(";").alias("flag")])
            .explode("flag")
            .filter(pl.col("flag").is_not_null() & (pl.col("flag") != ""))
        )
        if not exploded.is_empty():
            counts = exploded.group_by("flag").agg(
                pl.len().alias("len"),
                pl.col("sort_index").first().alias("example_sort_index"),
            ).sort("len", descending=True)
            for row in counts.iter_rows(named=True):
                flag = row["flag"]
                count = int(row["len"])
                severity = "warning"
                if flag == "missing_client_original_id":
                    fix = "Carry missingness into client-level features or restrict client-level analysis to rows with reliable client IDs; firm-level analysis remains usable."
                elif flag == "missing_price_for_potential_resting_event":
                    fix = "Audit event/order type mapping; if non-resting by design, update is_visible_resting_event() or enum classification."
                elif flag == "non_resting_unpriced_event":
                    fix = "Expected for market/unpriced non-resting events; keep out of visible depth and document counts."
                else:
                    fix = "Inspect normalize_event() and add a regression test for this normalization issue."
                problems.append(
                    Problem(
                        severity=severity,
                        problem=f"Normalization issue flag: {flag}",
                        evidence=f"normalization_issue_flags contains {flag!r} on {count} rows",
                        count=count,
                        example_sort_index=int(row["example_sort_index"]),
                        possible_fix=fix,
                    )
                )
    return problems


def dataframe_fingerprint(df: pl.DataFrame) -> str:
    payload = df.write_json(row_oriented=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check_determinism(input_path: Path, *, top_n: int) -> list[Problem]:
    df = pl.read_parquet(input_path)
    config = LOBConfig(top_n=top_n, snapshot_mode="none")
    first = reconstruct_dataframe(df, config=config)
    second = reconstruct_dataframe(df, config=config)
    first_panel = dataframe_fingerprint(first.panel)
    second_panel = dataframe_fingerprint(second.panel)
    first_norm = dataframe_fingerprint(first.normalized_events)
    second_norm = dataframe_fingerprint(second.normalized_events)
    if first_panel == second_panel and first_norm == second_norm:
        return []
    return [
        Problem(
            severity="error",
            problem="Non-deterministic reconstruction output",
            evidence=(
                f"panel fingerprints {first_panel} vs {second_panel}; "
                f"normalized fingerprints {first_norm} vs {second_norm}"
            ),
            count=1,
            example_sort_index=None,
            possible_fix="Stabilize sort_events() tie-breakers and avoid iteration over unordered containers in book_summary()/agent aggregation.",
        )
    ]


def render_markdown_report(
    *,
    source_file: str,
    output_dir: str,
    summary: dict[str, Any],
    problems: list[Problem],
) -> str:
    created_at = datetime.now(timezone.utc).isoformat()
    lines = [
        "# LOB Reconstruction Verification Report",
        "",
        f"- created_at_utc: `{created_at}`",
        f"- source_file: `{source_file}`",
        f"- output_dir: `{output_dir}`",
        "- scope: one-file verification on the smallest available parquet only",
        "",
        "## Summary",
        "",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: `{value}`")
    lines.extend([
        "",
        "## Problem inventory",
        "",
        "| Severity | Problem | Evidence | Count | Example sort_index | Possible fix | Status |",
        "|---|---|---:|---:|---:|---|---|",
    ])
    if problems:
        for problem in problems:
            example = "" if problem.example_sort_index is None else str(problem.example_sort_index)
            lines.append(
                "| "
                + " | ".join(
                    [
                        problem.severity,
                        problem.problem.replace("|", "\\|"),
                        problem.evidence.replace("|", "\\|"),
                        str(problem.count),
                        example,
                        problem.possible_fix.replace("|", "\\|"),
                        problem.status,
                    ]
                )
                + " |"
            )
    else:
        lines.append("| info | No hard verification problems found | All implemented checks passed | 0 |  | Keep current tests and rerun on additional files before feature extraction. | closed |")
    lines.extend([
        "",
        "## Scientific caveats",
        "",
        "- This report verifies visible-book reconstruction behavior, not exact FIFO queue position.",
        "- This report uses one parquet file only; it does not prove correctness on all instruments/days.",
        "- Synthetic unit tests remain necessary for specific event-sequence semantics.",
        "- Any warning marked as a data limitation should be propagated into downstream spoofing-feature interpretation.",
        "",
    ])
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify current LOB reconstruction on one parquet file.")
    parser.add_argument("--input", type=Path, required=True, help="Input parquet file to verify")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for reconstruction outputs and verification report")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--snapshot-mode", default="end_of_partition", choices=["none", "every_event_for_sample", "issue_rows_only", "end_of_partition"])
    parser.add_argument("--skip-determinism", action="store_true", help="Skip the in-memory duplicate reconstruction check")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = LOBConfig(top_n=args.top_n, snapshot_mode=args.snapshot_mode)
    paths = reconstruct_file(args.input, args.output_dir, config=config)

    input_rows = pl.scan_parquet(args.input).select(pl.len()).collect().item()
    panel = pl.read_parquet(paths.panel_path)
    normalized = pl.read_parquet(paths.normalized_path)
    agent = pl.read_parquet(paths.agent_panel_path)

    summary: dict[str, Any] = {
        "input_rows": int(input_rows),
        "panel_rows": panel.height,
        "normalized_rows": normalized.height,
        "agent_event_state_rows": agent.height,
        "book_crossed_pre_rows": _bool_sum(panel, "book_crossed_pre_flag"),
        "book_crossed_post_rows": _bool_sum(panel, "book_crossed_post_flag"),
        "book_locked_pre_rows": _bool_sum(panel, "book_locked_pre_flag"),
        "book_locked_post_rows": _bool_sum(panel, "book_locked_post_flag"),
        "top_n": args.top_n,
        "snapshot_mode": args.snapshot_mode,
    }

    problems: list[Problem] = []
    problems.extend(check_row_counts(input_rows=int(input_rows), panel_rows=panel.height, normalized_rows=normalized.height, agent_rows=agent.height))
    problems.extend(check_spread_flags(panel))
    problems.extend(check_top_n_depth(panel, top_n=args.top_n))
    problems.extend(check_issue_flags(panel, normalized))
    if not args.skip_determinism:
        problems.extend(check_determinism(args.input, top_n=args.top_n))

    report = render_markdown_report(
        source_file=str(args.input),
        output_dir=str(args.output_dir),
        summary=summary,
        problems=problems,
    )
    report_path = args.output_dir / "reconstruction_verification_report.md"
    report_path.write_text(report)
    print(f"verification_report: {report_path}")


if __name__ == "__main__":
    main()
```

**Step 2: Run focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_reconstruction_verifier.py
```

Expected: PASS.

**Step 3: If import from `scripts.verify_lob_reconstruction` fails**

If Python cannot import the `scripts` module, use one of these minimal fixes:

Option A, preferred if tests already import scripts elsewhere:
- Add `__init__.py` only if the repo convention allows it:
  - Create: `scripts/__init__.py`

Option B, keep tests path-local:
- In the test file, insert the repository root into `sys.path` before importing:

```python
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
```

Use Option B if adding `scripts/__init__.py` would be an unnecessary packaging change.

---

### Task 3: Run the verifier on the smallest parquet only

**Objective:** Produce the required Markdown problem report from the smallest parquet file.

**Files:**
- Input: `data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet`
- Output directory: `outputs/lob_reconstruction/$(date -u +%Y%m%d_%H%M%S)_verify_smallest_risanamento/`
- Output report: `outputs/lob_reconstruction/.../reconstruction_verification_report.md`

**Step 1: Run the verifier**

Run:

```bash
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_verify_smallest_risanamento"
OUT="outputs/lob_reconstruction/${RUN_ID}"
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/verify_lob_reconstruction.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --output-dir "$OUT" \
  --top-n 10 \
  --snapshot-mode end_of_partition
printf '%s\n' "$OUT"
```

Expected:
- Command exits with code 0.
- Terminal prints `verification_report: outputs/lob_reconstruction/.../reconstruction_verification_report.md`.
- The output directory contains both standard reconstruction files and the verification report.

**Step 2: Read the generated report**

Run:

```bash
REPORT="${OUT}/reconstruction_verification_report.md"
sed -n '1,220p' "$REPORT"
```

Expected:
- The report includes `## Problem inventory`.
- Every nonzero issue count appears as a table row with a possible fix.
- Locked/crossed rows should be zero if the current lifecycle fix behaves as expected.

---

### Task 4: Add marketable lifecycle-specific report checks if missing

**Objective:** Ensure the report explicitly validates the class of bug that previously created artificial locked/crossed books.

**Files:**
- Modify: `scripts/verify_lob_reconstruction.py`
- Test: `tests/lob/test_reconstruction_verifier.py`

**Step 1: Add failing test**

Add this test to `tests/lob/test_reconstruction_verifier.py`:

```python
from scripts.verify_lob_reconstruction import check_marketable_lifecycle_flags


def test_marketable_order_not_resting_rows_must_not_lock_or_cross_post_book():
    panel = pl.DataFrame(
        {
            "sort_index": [1, 2],
            "lob_issue_flags": ["marketable_order_not_resting", "marketable_order_not_resting"],
            "book_crossed_post_flag": [False, True],
            "book_locked_post_flag": [False, False],
        }
    )

    problems = check_marketable_lifecycle_flags(panel)

    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert problems[0].example_sort_index == 2
    assert "deferred residual" in problems[0].possible_fix.lower()
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_reconstruction_verifier.py::test_marketable_order_not_resting_rows_must_not_lock_or_cross_post_book
```

Expected: FAIL because `check_marketable_lifecycle_flags` is not implemented.

**Step 3: Implement helper**

Add this function to `scripts/verify_lob_reconstruction.py`:

```python
def check_marketable_lifecycle_flags(panel: pl.DataFrame) -> list[Problem]:
    required = {"lob_issue_flags", "book_crossed_post_flag", "book_locked_post_flag", "sort_index"}
    if not required.issubset(set(panel.columns)):
        return []
    flagged = panel.filter(pl.col("lob_issue_flags").str.contains("marketable_order_not_resting", literal=True))
    if flagged.is_empty():
        return []
    bad = flagged.filter(pl.col("book_crossed_post_flag").fill_null(False) | pl.col("book_locked_post_flag").fill_null(False))
    if bad.is_empty():
        return []
    return [
        Problem(
            severity="error",
            problem="Marketable non-resting row still creates locked/crossed post-book state",
            evidence=f"{bad.height} marketable_order_not_resting rows have locked/crossed post flags",
            count=bad.height,
            example_sort_index=_first_sort_index(bad),
            possible_fix="Inspect deferred residual handling in _apply_event() and _flush_pending_aggressive_residuals(); add a synthetic lifecycle test for the example sort_index pattern.",
        )
    ]
```

Then call it from `main()` after `check_spread_flags(panel)`:

```python
problems.extend(check_marketable_lifecycle_flags(panel))
```

**Step 4: Run focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_reconstruction_verifier.py
```

Expected: PASS.

---

### Task 5: Add stop-limit visibility-specific report checks if missing

**Objective:** Ensure the verifier records pre-trigger stop-limit visibility regressions.

**Files:**
- Modify: `scripts/verify_lob_reconstruction.py`
- Test: `tests/lob/test_reconstruction_verifier.py`

**Step 1: Add failing test**

Add this test:

```python
from scripts.verify_lob_reconstruction import check_stop_limit_visibility


def test_stop_limit_rows_should_not_have_best_price_equal_to_own_price_when_clean_book_would_not():
    panel = pl.DataFrame(
        {
            "sort_index": [1],
            "event_order_type_label": ["stop_limit_or_stop_limit_on_quote"],
            "event_side": ["ask"],
            "event_price": [99.0],
            "pre_best_bid": [100.0],
            "pre_best_ask": [101.0],
            "post_best_bid": [100.0],
            "post_best_ask": [99.0],
        }
    )

    problems = check_stop_limit_visibility(panel)

    assert len(problems) == 1
    assert problems[0].severity == "error"
    assert problems[0].example_sort_index == 1
    assert "stop-limit" in problems[0].possible_fix.lower()
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_reconstruction_verifier.py::test_stop_limit_rows_should_not_have_best_price_equal_to_own_price_when_clean_book_would_not
```

Expected: FAIL because `check_stop_limit_visibility` is not implemented.

**Step 3: Implement helper**

Add this helper:

```python
def check_stop_limit_visibility(panel: pl.DataFrame) -> list[Problem]:
    required = {
        "event_order_type_label",
        "event_side",
        "event_price",
        "pre_best_bid",
        "pre_best_ask",
        "post_best_bid",
        "post_best_ask",
        "sort_index",
    }
    if not required.issubset(set(panel.columns)):
        return []
    stop_limit = panel.filter(pl.col("event_order_type_label") == "stop_limit_or_stop_limit_on_quote")
    if stop_limit.is_empty():
        return []
    ask_bad = stop_limit.filter(
        (pl.col("event_side") == "ask")
        & pl.col("event_price").is_not_null()
        & (pl.col("post_best_ask") == pl.col("event_price"))
        & ((pl.col("pre_best_ask").is_null()) | (pl.col("pre_best_ask") != pl.col("event_price")))
    )
    bid_bad = stop_limit.filter(
        (pl.col("event_side") == "bid")
        & pl.col("event_price").is_not_null()
        & (pl.col("post_best_bid") == pl.col("event_price"))
        & ((pl.col("pre_best_bid").is_null()) | (pl.col("pre_best_bid") != pl.col("event_price")))
    )
    bad = pl.concat([ask_bad, bid_bad], how="vertical") if not ask_bad.is_empty() or not bid_bad.is_empty() else pl.DataFrame()
    if bad.is_empty():
        return []
    return [
        Problem(
            severity="error",
            problem="Stop-limit row appears to seed visible best price before trigger",
            evidence=f"{bad.height} stop-limit rows changed the visible best price to their own price",
            count=bad.height,
            example_sort_index=_first_sort_index(bad),
            possible_fix="Keep stop-limit orders out of is_visible_resting_event() until a documented trigger or dark-to-lit transformation makes them visible.",
        )
    ]
```

Then call it from `main()`:

```python
problems.extend(check_stop_limit_visibility(panel))
```

**Step 4: Run focused tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_reconstruction_verifier.py
```

Expected: PASS.

---

### Task 6: Rerun verifier and inspect final report

**Objective:** Produce the final one-file verification report with all planned checks.

**Files:**
- Output report: `outputs/lob_reconstruction/<run_id>/reconstruction_verification_report.md`

**Step 1: Run final verifier command**

Run:

```bash
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_verify_smallest_risanamento"
OUT="outputs/lob_reconstruction/${RUN_ID}"
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/verify_lob_reconstruction.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --output-dir "$OUT" \
  --top-n 10 \
  --snapshot-mode end_of_partition
REPORT="${OUT}/reconstruction_verification_report.md"
printf 'REPORT=%s\n' "$REPORT"
```

Expected: exit code 0 and a report path printed.

**Step 2: Check report contains problem inventory**

Run:

```bash
grep -n "Problem inventory" "$REPORT"
grep -n "Possible fix" "$REPORT"
sed -n '1,260p' "$REPORT"
```

Expected:
- `Problem inventory` exists.
- Every row has a possible fix.
- Known warnings such as missing client IDs and non-resting unpriced events are listed.
- Hard locked/crossed spread problems should be absent if current implementation is healthy on the smallest file.

**Step 3: Save summary path for the user**

Record the final path in the response, for example:

```text
outputs/lob_reconstruction/20260617_HHMMSS_verify_smallest_risanamento/reconstruction_verification_report.md
```

---

### Task 7: Run existing test suite after adding verifier

**Objective:** Ensure the verifier script and tests do not break the existing repository.

**Files:**
- Test: entire test suite

**Step 1: Run full tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q
```

Expected: all tests pass. Current baseline during planning was `18 passed`; after adding verifier tests, expected count should increase.

**Step 2: If tests fail**

- If failures are only in the new verifier tests, fix the verifier or tests.
- If existing reconstruction tests fail, stop and inspect whether imports or path changes polluted existing behavior.
- Do not modify core reconstruction logic in this verification task unless a failing test proves a verifier integration issue.

---

## Files likely to change

Planned new files:

- `scripts/verify_lob_reconstruction.py`
- `tests/lob/test_reconstruction_verifier.py`

Generated output files:

- `outputs/lob_reconstruction/<timestamp>_verify_smallest_risanamento/reconstruction_verification_report.md`
- Standard reconstruction outputs in the same output directory.

Core reconstruction files should not change during this verification implementation unless a test reveals the verifier cannot observe required fields:

- Avoid changing `src/spoofing_detection/lob/panel.py` during this task.
- Avoid changing `src/spoofing_detection/lob/normalize.py` during this task.

---

## Tests / validation commands

Run these in order during execution:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_reconstruction_verifier.py
```

```bash
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_verify_smallest_risanamento"
OUT="outputs/lob_reconstruction/${RUN_ID}"
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/verify_lob_reconstruction.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --output-dir "$OUT" \
  --top-n 10 \
  --snapshot-mode end_of_partition
REPORT="${OUT}/reconstruction_verification_report.md"
sed -n '1,260p' "$REPORT"
```

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q
```

---

## Risks, tradeoffs, and open questions

1. **Verifier performance**
   - The determinism check reconstructs the smallest parquet twice in memory. This is acceptable for the selected RISANAMENTO file but may be too slow for larger files.
   - Possible fix: add `--skip-determinism` or compare already-written outputs only.

2. **Markdown report can become too large**
   - If many unique issue strings appear, the problem inventory may become long.
   - Possible fix: aggregate by issue flag and include only first example sort index per issue.

3. **Problem severity classification is partly judgment-based**
   - Missing client IDs are likely data limitations, not reconstruction errors.
   - Locked/crossed states are hard errors for continuous visible-book reconstruction unless explained by auction/phase semantics.
   - Possible fix: add a severity mapping near `check_issue_flags()` and keep it explicit.

4. **Stop-limit heuristic may be too narrow**
   - The planned stop-limit check catches obvious best-price seeding but may miss deeper-level depth pollution.
   - Possible fix: add active snapshot checks for stop-limit `ORDERID`s if issue-row or every-event snapshots are enabled.

5. **Exact FIFO queue is out of scope**
   - Passing this verifier does not prove exact queue reconstruction.
   - Possible fix: add a separate future plan for FIFO/queue reconstruction if needed.

6. **Only one parquet file is tested**
   - This follows the user's scope request.
   - Possible fix: later generalize the same verifier to all files once the smallest-file report is satisfactory.

---

## Final deliverable after execution

When this plan is executed, the user should receive:

1. Path to the generated Markdown verification report.
2. A concise summary of:
   - whether locked/crossed states are present;
   - all issue categories found;
   - which problems are hard errors vs warnings/data limitations;
   - possible fixes listed in the report;
   - tests run and pass/fail output.
3. Explicit caveat that verification used only:
   - `data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet`
