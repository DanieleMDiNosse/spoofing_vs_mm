# Best Bid/Ask Diagnostics Interactive HTML Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build one interactive event-time HTML diagnostic plot showing best bid, best ask, and derived top-of-book diagnostics from one reconstructed LOB panel, using all events and no day/partition split.

**Architecture:** Add a small pure-analysis module under `src/spoofing_detection/lob/` for quote-diagnostic computation and Plotly HTML rendering, plus a thin CLI script under `scripts/`. The first generated artifact should use the post-fix Risanamento panel because it is moderate-sized, already validated, and has 143,018 all-event rows.

**Tech Stack:** Python, Polars, Plotly, pytest. Plotly is available in the `conda run -n main` environment; no new dependency should be added.

---

## Current context / assumptions

- The verified post-fix output root is:
  `outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/`
- Use this panel for the first generated plot:
  `outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG/lob_event_state_panel.parquet`
- Generate one interactive HTML artifact for now, not one per file yet.
- Use event time: x-axis is `sort_index`.
- Use all events: do not filter to quote changes; do not downsample; do not partition per day.
- Plot `post_*` book state, because the question is “evolution of the best bid and best ask” after each event.
- Scientific caveat for the HTML title/subtitle: event-time x-axis is deterministic event order, not physical clock time.
- Do not smooth quotes. Use step-like rendering if feasible; if Plotly performance is poor with SVG `line_shape="hv"`, fall back to `Scattergl` lines and explicitly label the rendering as all-event connected lines. For the first Risanamento run, try step lines first.
- Do not commit automatically. If the user explicitly asks for commits later, commit after implementation and validation.

## Proposed output artifact

Create:

`outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/quote_diagnostics/risanamento_best_quotes_event_time.html`

Optional sidecar metadata:

`outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/quote_diagnostics/risanamento_best_quotes_event_time_metadata.json`

The HTML should contain three stacked panels:

1. Top: `post_best_bid`, `post_best_ask`, and optional `mid_price`.
2. Middle: absolute spread and/or relative spread in basis points.
3. Bottom: top-of-book visible quantity and/or imbalance if these columns exist:
   - `post_bid_level_1_visible_qty`
   - `post_ask_level_1_visible_qty`

Hover data should include at least:

- `sort_index`
- `TRADEDATE` if present
- `event_class` if present
- `event_order_type_label` if present
- `event_side` if present
- `post_best_bid`
- `post_best_ask`
- `spread`
- `relative_spread_bps`
- `lob_issue_flags` if present
- `normalization_issue_flags` if present

---

## Task 1: Add quote-diagnostic computation tests

**Objective:** Define the expected derived columns before adding implementation.

**Files:**
- Create: `tests/lob/test_quote_diagnostics.py`
- Later modify: `src/spoofing_detection/lob/quote_diagnostics.py`

**Step 1: Write failing tests**

Create `tests/lob/test_quote_diagnostics.py` with tests like:

```python
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from spoofing_detection.lob.quote_diagnostics import (
    compute_quote_diagnostics,
    validate_quote_panel_columns,
)


def test_compute_quote_diagnostics_adds_mid_spread_relative_spread_and_imbalance():
    panel = pl.DataFrame(
        {
            "sort_index": [1, 2, 3, 4],
            "post_best_bid": [100.0, 100.5, None, 101.0],
            "post_best_ask": [101.0, 101.5, 102.0, None],
            "post_bid_level_1_visible_qty": [10.0, 20.0, 5.0, 0.0],
            "post_ask_level_1_visible_qty": [30.0, 20.0, 5.0, 10.0],
        }
    )

    out = compute_quote_diagnostics(panel)

    assert out.select("sort_index").to_series().to_list() == [1, 2, 3, 4]
    assert out.select("mid_price").to_series().to_list()[:2] == [100.5, 101.0]
    assert out.select("spread").to_series().to_list()[:2] == [1.0, 1.0]
    assert out.select("relative_spread_bps").to_series().to_list()[:2] == pytest.approx(
        [1.0 / 100.5 * 10_000, 1.0 / 101.0 * 10_000]
    )
    assert out.select("top_of_book_imbalance").to_series().to_list()[:2] == pytest.approx(
        [10.0 / 40.0, 20.0 / 40.0]
    )
    assert out.select("valid_touch").to_series().to_list() == [True, True, False, False]


def test_compute_quote_diagnostics_preserves_all_event_rows_and_order():
    panel = pl.DataFrame(
        {
            "sort_index": [3, 1, 2],
            "post_best_bid": [99.0, 98.0, 98.5],
            "post_best_ask": [100.0, 99.0, 99.5],
        }
    )

    out = compute_quote_diagnostics(panel)

    assert out.height == panel.height
    # Function should not silently reorder rows; event order should be supplied by reconstruction.
    assert out.select("sort_index").to_series().to_list() == [3, 1, 2]


def test_validate_quote_panel_columns_rejects_missing_required_columns():
    panel = pl.DataFrame({"sort_index": [1], "post_best_bid": [100.0]})

    with pytest.raises(ValueError, match="post_best_ask"):
        validate_quote_panel_columns(panel)
```

**Step 2: Run tests to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_quote_diagnostics.py
```

Expected: FAIL because `spoofing_detection.lob.quote_diagnostics` does not exist yet.

---

## Task 2: Implement pure quote-diagnostic computation

**Objective:** Add reusable, testable functions for loading/validating panel columns and computing mid/spread diagnostics.

**Files:**
- Create: `src/spoofing_detection/lob/quote_diagnostics.py`
- Test: `tests/lob/test_quote_diagnostics.py`

**Step 1: Add minimal implementation**

Create `src/spoofing_detection/lob/quote_diagnostics.py` with code along these lines:

```python
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import polars as pl

REQUIRED_QUOTE_COLUMNS = ("sort_index", "post_best_bid", "post_best_ask")
OPTIONAL_HOVER_COLUMNS = (
    "TRADEDATE",
    "event_class",
    "event_order_type_label",
    "event_side",
    "lob_issue_flags",
    "normalization_issue_flags",
)
TOP_OF_BOOK_QTY_COLUMNS = (
    "post_bid_level_1_visible_qty",
    "post_ask_level_1_visible_qty",
)


def validate_quote_panel_columns(panel: pl.DataFrame) -> None:
    missing = [column for column in REQUIRED_QUOTE_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(f"panel is missing required quote columns: {', '.join(missing)}")


def read_panel(panel_path: str | Path) -> pl.DataFrame:
    panel_path = Path(panel_path)
    if panel_path.suffix.lower() != ".parquet":
        raise ValueError(f"expected a parquet panel, got {panel_path}")
    panel = pl.read_parquet(panel_path)
    validate_quote_panel_columns(panel)
    return panel


def compute_quote_diagnostics(panel: pl.DataFrame) -> pl.DataFrame:
    validate_quote_panel_columns(panel)

    out = panel.with_columns(
        [
            (
                pl.col("post_best_bid").is_not_null()
                & pl.col("post_best_ask").is_not_null()
                & (pl.col("post_best_bid") < pl.col("post_best_ask"))
            ).alias("valid_touch"),
            ((pl.col("post_best_bid") + pl.col("post_best_ask")) / 2.0).alias("mid_price"),
            (pl.col("post_best_ask") - pl.col("post_best_bid")).alias("spread"),
        ]
    ).with_columns(
        [
            pl.when(pl.col("mid_price").is_not_null() & (pl.col("mid_price") > 0))
            .then(pl.col("spread") / pl.col("mid_price") * 10_000.0)
            .otherwise(None)
            .alias("relative_spread_bps")
        ]
    )

    if all(column in out.columns for column in TOP_OF_BOOK_QTY_COLUMNS):
        bid_qty, ask_qty = TOP_OF_BOOK_QTY_COLUMNS
        out = out.with_columns(
            [
                (pl.col(bid_qty) + pl.col(ask_qty)).alias("top_of_book_visible_qty"),
            ]
        ).with_columns(
            [
                pl.when(pl.col("top_of_book_visible_qty") > 0)
                .then(pl.col(bid_qty) / pl.col("top_of_book_visible_qty"))
                .otherwise(None)
                .alias("top_of_book_imbalance")
            ]
        )
    else:
        out = out.with_columns(
            [
                pl.lit(None, dtype=pl.Float64).alias("top_of_book_visible_qty"),
                pl.lit(None, dtype=pl.Float64).alias("top_of_book_imbalance"),
            ]
        )

    return out


def write_metadata(
    *,
    output_path: str | Path,
    panel_path: str | Path,
    diagnostics: pl.DataFrame,
    html_path: str | Path,
    command: list[str] | None = None,
) -> None:
    output_path = Path(output_path)
    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "panel_path": str(panel_path),
        "html_path": str(html_path),
        "row_count": diagnostics.height,
        "x_axis": "sort_index",
        "event_time": True,
        "all_events": True,
        "partitioned_by_day": False,
        "columns": diagnostics.columns,
        "valid_touch_rows": int(diagnostics.select(pl.col("valid_touch").fill_null(False).sum()).item()),
        "command": command,
    }
    output_path.write_text(json.dumps(payload, indent=2, sort_keys=True))
```

**Step 2: Run test to verify pass**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_quote_diagnostics.py
```

Expected: PASS for the computation tests.

---

## Task 3: Add Plotly HTML rendering tests

**Objective:** Specify that the renderer writes a self-contained interactive HTML with the expected diagnostics and metadata markers.

**Files:**
- Modify: `tests/lob/test_quote_diagnostics.py`
- Modify: `src/spoofing_detection/lob/quote_diagnostics.py`

**Step 1: Add failing test**

Append to `tests/lob/test_quote_diagnostics.py`:

```python
from spoofing_detection.lob.quote_diagnostics import write_interactive_quote_html


def test_write_interactive_quote_html_creates_expected_file(tmp_path: Path):
    panel = pl.DataFrame(
        {
            "sort_index": [1, 2, 3],
            "TRADEDATE": ["2024-06-03", "2024-06-03", "2024-06-03"],
            "event_class": ["session_reload", "new_order", "cancel"],
            "event_order_type_label": ["limit", "limit", "limit"],
            "event_side": ["bid", "ask", "ask"],
            "post_best_bid": [0.0286, 0.0287, 0.0287],
            "post_best_ask": [0.0292, 0.0292, 0.0293],
            "post_bid_level_1_visible_qty": [100.0, 200.0, 200.0],
            "post_ask_level_1_visible_qty": [300.0, 300.0, 150.0],
            "lob_issue_flags": [None, "marketable_order_not_resting", None],
            "normalization_issue_flags": [None, None, None],
        }
    )
    diagnostics = compute_quote_diagnostics(panel)
    html_path = tmp_path / "quotes.html"

    write_interactive_quote_html(
        diagnostics,
        html_path,
        title="Risanamento best bid/ask event-time diagnostics",
        source_label="test-panel.parquet",
    )

    html = html_path.read_text()
    assert "Risanamento best bid/ask event-time diagnostics" in html
    assert "Best bid" in html
    assert "Best ask" in html
    assert "Spread" in html
    assert "Relative spread" in html
    assert "Top-of-book imbalance" in html
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_quote_diagnostics.py::test_write_interactive_quote_html_creates_expected_file
```

Expected: FAIL because `write_interactive_quote_html` is not implemented.

---

## Task 4: Implement Plotly HTML rendering

**Objective:** Produce a single interactive HTML with all-event event-time best quotes plus derived diagnostics.

**Files:**
- Modify: `src/spoofing_detection/lob/quote_diagnostics.py`
- Test: `tests/lob/test_quote_diagnostics.py`

**Step 1: Add rendering implementation**

Add imports near the top:

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
```

Add functions similar to:

```python
PLOT_COLUMNS = (
    "sort_index",
    "post_best_bid",
    "post_best_ask",
    "mid_price",
    "spread",
    "relative_spread_bps",
    "top_of_book_imbalance",
    "post_bid_level_1_visible_qty",
    "post_ask_level_1_visible_qty",
)


def _available_columns(df: pl.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _customdata(df: pl.DataFrame) -> tuple[list[str], list[list[object]]]:
    hover_columns = _available_columns(
        df,
        (
            "TRADEDATE",
            "event_class",
            "event_order_type_label",
            "event_side",
            "lob_issue_flags",
            "normalization_issue_flags",
        ),
    )
    if not hover_columns:
        return [], []
    return hover_columns, df.select(hover_columns).to_numpy().tolist()


def _hovertemplate(y_label: str, hover_columns: list[str]) -> str:
    lines = [
        "sort_index=%{x}",
        f"{y_label}=%{{y}}",
    ]
    for idx, column in enumerate(hover_columns):
        lines.append(f"{column}=%{{customdata[{idx}]}}")
    return "<br>".join(lines) + "<extra></extra>"


def _add_trace(
    fig: go.Figure,
    *,
    df: pl.DataFrame,
    y_column: str,
    name: str,
    row: int,
    col: int = 1,
    color: str | None = None,
    step_like: bool = False,
) -> None:
    if y_column not in df.columns:
        return
    hover_columns, customdata = _customdata(df)
    line = {}
    if color is not None:
        line["color"] = color
    if step_like:
        line["shape"] = "hv"
    fig.add_trace(
        go.Scatter(
            x=df.get_column("sort_index"),
            y=df.get_column(y_column),
            name=name,
            mode="lines",
            line=line,
            customdata=customdata if customdata else None,
            hovertemplate=_hovertemplate(name, hover_columns),
        ),
        row=row,
        col=col,
    )


def write_interactive_quote_html(
    diagnostics: pl.DataFrame,
    output_html: str | Path,
    *,
    title: str,
    source_label: str,
) -> None:
    validate_quote_panel_columns(diagnostics)
    output_html = Path(output_html)
    output_html.parent.mkdir(parents=True, exist_ok=True)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        subplot_titles=(
            "Best bid / best ask / mid price",
            "Spread diagnostics",
            "Top-of-book depth diagnostics",
        ),
    )

    _add_trace(fig, df=diagnostics, y_column="post_best_bid", name="Best bid", row=1, color="#1f77b4", step_like=True)
    _add_trace(fig, df=diagnostics, y_column="post_best_ask", name="Best ask", row=1, color="#d62728", step_like=True)
    _add_trace(fig, df=diagnostics, y_column="mid_price", name="Mid price", row=1, color="#2ca02c", step_like=True)

    _add_trace(fig, df=diagnostics, y_column="spread", name="Spread", row=2, color="#9467bd", step_like=True)
    _add_trace(fig, df=diagnostics, y_column="relative_spread_bps", name="Relative spread (bps)", row=2, color="#8c564b", step_like=True)

    _add_trace(fig, df=diagnostics, y_column="post_bid_level_1_visible_qty", name="Bid L1 visible qty", row=3, color="#17becf", step_like=True)
    _add_trace(fig, df=diagnostics, y_column="post_ask_level_1_visible_qty", name="Ask L1 visible qty", row=3, color="#ff7f0e", step_like=True)
    _add_trace(fig, df=diagnostics, y_column="top_of_book_imbalance", name="Top-of-book imbalance", row=3, color="#7f7f7f", step_like=True)

    fig.update_layout(
        title=(
            f"{title}<br>"
            f"<sup>Source: {source_label}; x-axis is event time (`sort_index`); all events included; no day partitioning.</sup>"
        ),
        hovermode="x unified",
        template="plotly_white",
        height=950,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_xaxes(title_text="Event time (`sort_index`)", row=3, col=1)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Spread", row=2, col=1)
    fig.update_yaxes(title_text="Qty / imbalance", row=3, col=1)

    fig.write_html(output_html, include_plotlyjs="cdn", full_html=True)
```

**Step 2: Run tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_quote_diagnostics.py
```

Expected: PASS.

**Implementation note:** If the all-event Risanamento HTML feels sluggish, do not silently downsample. Instead add a visible warning in the final response and keep the artifact faithful to the user’s all-event request.

---

## Task 5: Add CLI wrapper

**Objective:** Let the user generate the HTML from a reconstructed panel with an exact command.

**Files:**
- Create: `scripts/plot_best_quotes.py`
- Test manually in Task 6

**Step 1: Create script**

Create `scripts/plot_best_quotes.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.quote_diagnostics import (
    compute_quote_diagnostics,
    read_panel,
    write_interactive_quote_html,
    write_metadata,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot all-event event-time best bid/ask diagnostics from a reconstructed LOB panel."
    )
    parser.add_argument("--panel", type=Path, required=True, help="Input lob_event_state_panel.parquet")
    parser.add_argument("--output-html", type=Path, required=True, help="Output interactive .html path")
    parser.add_argument("--metadata", type=Path, default=None, help="Optional metadata .json path")
    parser.add_argument(
        "--title",
        default="Best bid / best ask event-time diagnostics",
        help="Plot title",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    panel = read_panel(args.panel)
    diagnostics = compute_quote_diagnostics(panel)
    write_interactive_quote_html(
        diagnostics,
        args.output_html,
        title=args.title,
        source_label=str(args.panel),
    )
    if args.metadata is not None:
        write_metadata(
            output_path=args.metadata,
            panel_path=args.panel,
            diagnostics=diagnostics,
            html_path=args.output_html,
            command=sys.argv,
        )
    print(f"html: {args.output_html}")
    if args.metadata is not None:
        print(f"metadata: {args.metadata}")
    print(f"rows_plotted: {diagnostics.height}")


if __name__ == "__main__":
    main()
```

**Step 2: Syntax check**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python -m py_compile scripts/plot_best_quotes.py src/spoofing_detection/lob/quote_diagnostics.py
```

Expected: exit code 0.

---

## Task 6: Generate the first Risanamento interactive HTML

**Objective:** Produce the requested one-file, all-event, event-time interactive plot.

**Files:**
- Read: `outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG/lob_event_state_panel.parquet`
- Create: `outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/quote_diagnostics/risanamento_best_quotes_event_time.html`
- Create: `outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/quote_diagnostics/risanamento_best_quotes_event_time_metadata.json`

**Step 1: Run generator**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/plot_best_quotes.py \
  --panel outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG/lob_event_state_panel.parquet \
  --output-html outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/quote_diagnostics/risanamento_best_quotes_event_time.html \
  --metadata outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/quote_diagnostics/risanamento_best_quotes_event_time_metadata.json \
  --title "Risanamento best bid/ask and top-of-book diagnostics — event time, all events"
```

Expected output includes:

```text
html: outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/quote_diagnostics/risanamento_best_quotes_event_time.html
metadata: outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/quote_diagnostics/risanamento_best_quotes_event_time_metadata.json
rows_plotted: 143018
```

**Step 2: Verify artifact exists and metadata is consistent**

Run:

```bash
python - <<'PY'
from pathlib import Path
import json
html = Path('outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/quote_diagnostics/risanamento_best_quotes_event_time.html')
meta = Path('outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/quote_diagnostics/risanamento_best_quotes_event_time_metadata.json')
print('html_exists', html.exists(), 'html_size', html.stat().st_size if html.exists() else None)
print('metadata_exists', meta.exists())
payload = json.loads(meta.read_text())
print('row_count', payload['row_count'])
print('all_events', payload['all_events'])
print('event_time', payload['event_time'])
print('partitioned_by_day', payload['partitioned_by_day'])
PY
```

Expected:

```text
html_exists True html_size <positive integer>
metadata_exists True
row_count 143018
all_events True
event_time True
partitioned_by_day False
```

---

## Task 7: Run full regression tests

**Objective:** Ensure the plotting addition did not break reconstruction or existing verifier behavior.

**Files:**
- Test: full suite

**Step 1: Run full tests**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q
```

Expected:

```text
28 passed ...
```

The exact count may be higher if more tests are added, but there should be no failures.

---

## Task 8: Final user-facing report

**Objective:** Tell the user exactly what was generated and how to open/regenerate it.

Report:

- Created interactive HTML path.
- Created metadata path.
- Confirmed all-event row count.
- Confirmed x-axis is `sort_index` event time.
- Confirmed no day partitioning and no quote-change filtering.
- Confirmed tests passed.
- Caveat: all-event interactive HTML may be large; if interactive performance is poor, next iteration can add an optional `--quote-changes-only` or `--max-points` mode, but not for this requested artifact.

Suggested final wording:

```text
Generated the first all-event interactive quote diagnostic for Risanamento:
/path/to/risanamento_best_quotes_event_time.html

It plots post-event best bid, best ask, mid price, spread/relative spread, and top-of-book visible quantity/imbalance against event time (`sort_index`). It includes all 143,018 events and does not partition by day.

Validation: pytest passed; metadata confirms row_count=143018, all_events=true, event_time=true, partitioned_by_day=false.
```

---

## Risks, tradeoffs, and open questions

1. **Interactive performance:** All-event Plotly HTML can be heavy. For Risanamento this should be manageable; Ferrari may be much heavier at 481,798 events. Do not downsample unless the user explicitly asks.
2. **Step rendering vs WebGL:** Plotly SVG step traces are scientifically preferable but may be slower. If performance is unacceptable, consider a second implementation with duplicated step coordinates and `Scattergl`, while preserving event-time semantics.
3. **Clock time is intentionally excluded:** User requested event time. Do not use timestamp columns for this first artifact.
4. **No partitioning:** User explicitly requested no day partitioning. The plot title/metadata should state this.
5. **No smoothing:** Smoothing would be misleading for quote paths and should not be added.
6. **Future extension:** After this first artifact, a later task can add a `--all-panels-from-root` option to generate one HTML per file under the post-fix validation root.
