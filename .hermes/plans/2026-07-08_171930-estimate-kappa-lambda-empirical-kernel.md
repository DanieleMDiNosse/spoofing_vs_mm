# Empirical Kappa/Lambda Depth-Kernel Calibration Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Estimate instrument-specific depth-kernel profiles from full LOB samples: \(\kappa\) from level-hit probabilities and \(\lambda\) from level-specific covariance with future mid-price changes, then optionally use the resulting empirical kernel in MSCI/DWI metrics.

**Architecture:** Add a small calibration module that replays the full raw event sample per instrument, emits level-by-level hit-probability and covariance profiles, derives empirical kernel weights, and records implied \(\widehat\kappa\) and \(\widehat\lambda\) as diagnostic exponential summaries. Keep the current parametric kernel as the default path; empirical-kernel use is explicit via a calibration artifact path.

**Tech Stack:** Python, Polars, existing `spoofing_detection.lob` replay/normalization code, pytest, JSON/Parquet/Markdown outputs, LaTeX build for manuscript sync.

---

## Current context / assumptions

### Inspected manuscript context

- `paper/spoofing.tex:532-590` currently defines an **instrument-specific empirical depth kernel**.
- It already states that the kernel is fitted instrument by instrument on the full available sample.
- It defines:
  - hit probability \(\widehat p^{m,s}_k(H)\),
  - protection profile \(\widehat\rho^{m,s}_k(H)=1-\widehat p^{m,s}_k(H)\),
  - visibility profile from absolute covariance,
  - empirical kernel \(\widehat\omega^{m,s}_k(H)\).
- Required manuscript follow-up after implementation: keep the empirical-kernel text, and clarify that \(\widehat\lambda\), if reported, is an implied decay summary of the covariance profile rather than a hand-picked tuning parameter.

### Inspected code context

- `src/spoofing_detection/lob/spoofing_metrics.py:28-33`
  - `ExploratoryMetricsConfig` currently stores scalar `kappa` and `lambda_`.
- `src/spoofing_detection/lob/spoofing_metrics.py:119-129`
  - `depth_kernel_weights(distances, kappa, lambda_)` implements the current parametric kernel.
- `src/spoofing_detection/lob/spoofing_metrics.py:132-152`
  - `_side_depth_metadata(...)` computes per-price kernel weights for current active levels.
- `src/spoofing_detection/lob/spoofing_metrics.py:155-276`
  - `compute_client_top_n_exposures(...)` uses `_side_depth_metadata` and emits per-level diagnostics.
- `src/spoofing_detection/lob/spoofing_metrics.py:377-450`
  - `_candidate_deceptive_order_rows(...)` also uses `_side_depth_metadata` and must receive empirical weights if metrics do.
- `src/spoofing_detection/lob/spoofing_metrics.py:564-676`
  - `_stream_metric_inputs(...)` replays the LOB and is the right pattern to reuse for full-sample calibration.
- `src/spoofing_detection/lob/spoofing_metrics.py:1092-1134`
  - `compute_exploratory_metrics(...)` is the public path used by scripts.
- `scripts/compute_spoofing_metrics.py:28-40, 50-105, 226-236`
  - CLI/config currently expose scalar `--kappa` and `--lambda` only.
- `scripts/run_multilevel_spoofing_grid.py:28-39, 103-144, 226-236`
  - grid runner also exposes scalar `--kappa` and `--lambda`.
- `configs/spoofing_detection_parameters.json:2-24, 36-59`
  - config defaults and parameter descriptions currently describe only scalar `kappa`/`lambda`.
- Existing focused tests:
  - `tests/lob/test_spoofing_metrics.py`
  - `tests/lob/test_compute_spoofing_metrics_cli.py`
  - `tests/lob/test_spoofing_grid_runner.py`

### Important scientific assumptions

- Calibration must use the **full sample for each instrument**, not only suspicious events, alert events, or selected clients.
- Level-hit probability should be estimated from realized aggressive/passive-fill flow over the same horizon `H` used for post-execution cancellation/price-response diagnostics.
- Operational empirical weights should come directly from the empirical components:

  \[
  \widehat\omega^{m,s}_k(H)
  \propto
  \widehat\rho^{m,s}_k(H)\widehat\nu^{m,s}_k(H),
  \quad
  \widehat\rho^{m,s}_k(H)=1-\widehat p^{m,s}_k(H),
  \quad
  \widehat\nu^{m,s}_k(H)=|\widehat{\operatorname{Cov}}(Z^{m,s}_{t,k},\Delta M^m_{t,H})|.
  \]

- If the user still wants scalar \(\kappa\) and \(\lambda\), report them as **instrument/side diagnostic summaries**:
  - \(\widehat\kappa^{m,s}\): weighted log-linear decay slope of hit probability by depth.
  - \(\widehat\lambda^{m,s}\): weighted log-linear decay slope of the absolute covariance profile by depth.
- Do **not** choose parameters by maximizing alert counts.

### Manuscript vs generator/code separation

- Manuscript edits:
  - `paper/spoofing.tex` only, after code artifacts exist.
- Generator/code changes:
  - calibration module and scripts under `src/`, `scripts/`, `tests/`, and `configs/`.
- Generated artifacts to regenerate/sync after implementation:
  - calibration Parquet/CSV/JSON/Markdown under a new `outputs/depth_kernel_calibration/...` run folder;
  - metrics outputs only if empirical kernel is selected for production metrics;
  - `paper/spoofing.pdf` after manuscript sync.
- No manual edits to generated figures are planned.

---

## Proposed approach

1. Add a pure calibration layer that can be tested without running the whole surveillance pipeline.
2. Add a full-sample replay function that emits market-level snapshots and realized future hits.
3. Estimate:
   - `hit_probability` per `(instrument, side, rank)`,
   - `protection_component = 1 - hit_probability`,
   - `visibility_covariance = abs(cov(level_imbalance_contribution, future_mid_change))`,
   - `raw_weight = protection_component * visibility_covariance`,
   - `kernel_weight = raw_weight / sum(raw_weight by instrument+side)`.
4. Store implied `kappa_hat` and `lambda_hat` as summaries, not as the primary operational kernel.
5. Let `compute_exploratory_metrics` optionally consume a calibration artifact. If no artifact is supplied, preserve current scalar `kappa`/`lambda_` behavior exactly.
6. Keep all calibration metadata explicit: input path, instrument id, top-n, horizon, tick size, row count, exposure counts, floors/clips, package versions, command.

---

## Step-by-step plan

### Task 1: Add pure unit tests for empirical kernel normalization

**Objective:** Lock the empirical-kernel math before adding replay logic.

**Files:**
- Create: `tests/lob/test_depth_kernel_calibration.py`
- Later create: `src/spoofing_detection/lob/depth_kernel_calibration.py`

**Step 1: Write failing tests**

Add these initial tests:

```python
from __future__ import annotations

import pytest
import polars as pl

from spoofing_detection.lob.depth_kernel_calibration import (
    build_empirical_depth_kernel,
    fit_log_decay_slope,
)


def test_empirical_kernel_combines_protection_and_covariance_by_side():
    profile = pl.DataFrame(
        {
            "instrument_id": ["ABC", "ABC", "ABC", "ABC"],
            "side": ["bid", "bid", "ask", "ask"],
            "rank": [1, 2, 1, 2],
            "depth_distance_ticks": [1.0, 2.0, 1.0, 2.0],
            "exposure_count": [100, 100, 100, 100],
            "hit_count": [80, 20, 50, 25],
            "hit_probability": [0.8, 0.2, 0.5, 0.25],
            "visibility_covariance": [2.0, 1.0, 3.0, 1.0],
        }
    )

    out = build_empirical_depth_kernel(profile, protection_floor=0.0, visibility_floor=0.0)
    bid = out.filter(pl.col("side") == "bid").sort("rank")
    ask = out.filter(pl.col("side") == "ask").sort("rank")

    # bid raw weights: (1 - .8) * 2 = .4; (1 - .2) * 1 = .8
    assert bid["raw_weight"].to_list() == pytest.approx([0.4, 0.8])
    assert bid["kernel_weight"].to_list() == pytest.approx([1 / 3, 2 / 3])

    # ask raw weights: (1 - .5) * 3 = 1.5; (1 - .25) * 1 = .75
    assert ask["raw_weight"].to_list() == pytest.approx([2 / 3 * 2.25, 1 / 3 * 2.25])
    assert ask["kernel_weight"].to_list() == pytest.approx([2 / 3, 1 / 3])


def test_fit_log_decay_slope_estimates_positive_decay():
    # profile = exp(2 - 0.7 * distance)
    distances = [1.0, 2.0, 3.0, 4.0]
    values = [2.718281828459045 ** (2.0 - 0.7 * d) for d in distances]

    slope = fit_log_decay_slope(distances, values, weights=[10, 10, 10, 10])

    assert slope == pytest.approx(0.7)
```

**Step 2: Run test to verify failure**

Run:

```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_depth_kernel_calibration.py -q
```

Expected: FAIL because `spoofing_detection.lob.depth_kernel_calibration` does not exist yet.

**Step 3: Implement minimal pure helpers**

Create `src/spoofing_detection/lob/depth_kernel_calibration.py` with:

```python
from __future__ import annotations

import math
from collections.abc import Sequence

import polars as pl


def _weighted_mean(values: Sequence[float], weights: Sequence[float]) -> float:
    total = sum(weights)
    if total <= 0:
        raise ValueError("weights must have positive total")
    return sum(value * weight for value, weight in zip(values, weights, strict=True)) / total


def fit_log_decay_slope(
    distances: Sequence[float],
    values: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
    value_floor: float = 1e-12,
) -> float | None:
    rows = [
        (float(distance), math.log(max(float(value), value_floor)), float(weight))
        for distance, value, weight in zip(
            distances,
            values,
            weights if weights is not None else [1.0] * len(values),
            strict=True,
        )
        if math.isfinite(float(distance)) and math.isfinite(float(value)) and float(weight) > 0
    ]
    if len(rows) < 2:
        return None
    xs, ys, ws = zip(*rows, strict=True)
    xbar = _weighted_mean(xs, ws)
    ybar = _weighted_mean(ys, ws)
    denom = sum(weight * (x - xbar) ** 2 for x, weight in zip(xs, ws, strict=True))
    if denom <= 0:
        return None
    beta = sum(weight * (x - xbar) * (y - ybar) for x, y, weight in rows) / denom
    return max(-beta, 0.0)


def build_empirical_depth_kernel(
    profile: pl.DataFrame,
    *,
    protection_floor: float = 0.0,
    visibility_floor: float = 0.0,
) -> pl.DataFrame:
    required = {
        "instrument_id",
        "side",
        "rank",
        "depth_distance_ticks",
        "exposure_count",
        "hit_count",
        "hit_probability",
        "visibility_covariance",
    }
    missing = sorted(required - set(profile.columns))
    if missing:
        raise ValueError(f"missing empirical kernel profile columns: {', '.join(missing)}")
    if profile.is_empty():
        return profile

    with_components = profile.with_columns(
        protection_component=(1.0 - pl.col("hit_probability")).clip(lower_bound=protection_floor),
        visibility_component=pl.col("visibility_covariance").abs().clip(lower_bound=visibility_floor),
    ).with_columns(
        raw_weight=pl.col("protection_component") * pl.col("visibility_component")
    )

    totals = with_components.group_by(["instrument_id", "side"]).agg(pl.col("raw_weight").sum().alias("raw_weight_total"))
    out = with_components.join(totals, on=["instrument_id", "side"], how="left").with_columns(
        kernel_weight=pl.when(pl.col("raw_weight_total") > 0)
        .then(pl.col("raw_weight") / pl.col("raw_weight_total"))
        .otherwise(0.0)
    )
    return out.drop("raw_weight_total").sort(["instrument_id", "side", "rank"])
```

**Step 4: Run test to verify pass**

Run:

```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_depth_kernel_calibration.py -q
```

Expected: PASS.

**Step 5: Checkpoint**

Do not commit unless the user explicitly asks. If commit permission is granted:

```bash
git add src/spoofing_detection/lob/depth_kernel_calibration.py tests/lob/test_depth_kernel_calibration.py
git commit -m "feat: add empirical depth-kernel calibration helpers"
```

---

### Task 2: Add tested hit-probability and covariance estimators from snapshot tables

**Objective:** Estimate \(\widehat p_k\), \(\widehat\nu_k\), \(\widehat\kappa\), and \(\widehat\lambda\) from already-built snapshot/fill tables.

**Files:**
- Modify: `tests/lob/test_depth_kernel_calibration.py`
- Modify: `src/spoofing_detection/lob/depth_kernel_calibration.py`

**Step 1: Write failing tests**

Add tests for helper functions that operate on small DataFrames, independent of LOB replay:

```python
from spoofing_detection.lob.depth_kernel_calibration import (
    estimate_hit_probability_profile,
    estimate_visibility_covariance_profile,
)


def test_estimate_hit_probability_profile_uses_at_or_through_prices():
    snapshots = pl.DataFrame(
        {
            "instrument_id": ["ABC", "ABC", "ABC", "ABC"],
            "snapshot_id": [1, 1, 2, 2],
            "event_ts": [0.0, 0.0, 10.0, 10.0],
            "side": ["ask", "ask", "ask", "ask"],
            "rank": [1, 2, 1, 2],
            "price": [101.0, 102.0, 101.0, 102.0],
            "visible_qty": [10.0, 20.0, 10.0, 20.0],
            "mid": [100.5, 100.5, 100.5, 100.5],
        }
    )
    fills = pl.DataFrame(
        {
            "instrument_id": ["ABC", "ABC"],
            "event_ts": [1.0, 11.0],
            "side": ["ask", "ask"],
            "price": [102.0, 101.0],
        }
    )

    out = estimate_hit_probability_profile(snapshots, fills, horizon_seconds=2.0).sort("rank")

    # Snapshot 1 has an ask fill at 102, so ask ranks 1 and 2 are reached.
    # Snapshot 2 has an ask fill at 101, so only ask rank 1 is reached.
    assert out["exposure_count"].to_list() == [2, 2]
    assert out["hit_count"].to_list() == [2, 1]
    assert out["hit_probability"].to_list() == pytest.approx([1.0, 0.5])


def test_estimate_visibility_covariance_profile_uses_future_mid_change():
    snapshots = pl.DataFrame(
        {
            "instrument_id": ["ABC", "ABC", "ABC", "ABC"],
            "snapshot_id": [1, 1, 2, 2],
            "event_ts": [0.0, 0.0, 10.0, 10.0],
            "side": ["bid", "ask", "bid", "ask"],
            "rank": [1, 1, 1, 1],
            "price": [100.0, 101.0, 100.0, 101.0],
            "visible_qty": [10.0, 30.0, 30.0, 10.0],
            "mid": [100.5, 100.5, 100.5, 100.5],
            "future_mid": [101.0, 101.0, 100.0, 100.0],
        }
    )

    out = estimate_visibility_covariance_profile(snapshots).sort(["side", "rank"])

    assert set(out.columns) >= {"side", "rank", "visibility_covariance", "visibility_observation_count"}
    assert out.filter(pl.col("side") == "ask").item(0, "visibility_covariance") > 0
    assert out.filter(pl.col("side") == "bid").item(0, "visibility_covariance") > 0
```

**Step 2: Run targeted tests and confirm failure**

Run:

```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_depth_kernel_calibration.py -q
```

Expected: FAIL because estimator functions are missing.

**Step 3: Implement estimator helpers**

Implementation notes:

- `estimate_hit_probability_profile(...)`:
  - group snapshots by `(instrument_id, side, rank)`;
  - count only rows with non-null price and positive visible quantity;
  - a side/rank snapshot is hit when at least one future fill in `(t, t + H]` reaches that level:
    - ask: future fill has same instrument, side `ask`, and `fill_price >= snapshot_price`;
    - bid: future fill has same instrument, side `bid`, and `fill_price <= snapshot_price`.
- `estimate_visibility_covariance_profile(...)`:
  - require `future_mid` and `mid`;
  - `delta_mid = future_mid - mid`;
  - use a normalized signed level contribution:
    - ask contribution positive;
    - bid contribution negative;
    - normalize by same-snapshot top-n visible quantity to avoid volume scale dominating covariance.
- `fit_log_decay_slope(...)`:
  - use `hit_probability` for implied `kappa_hat`;
  - use `visibility_covariance` for implied `lambda_hat`;
  - weight by `exposure_count` / `visibility_observation_count`.

**Step 4: Run targeted tests**

Run:

```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_depth_kernel_calibration.py -q
```

Expected: PASS.

**Step 5: Checkpoint**

If commit permission is granted:

```bash
git add src/spoofing_detection/lob/depth_kernel_calibration.py tests/lob/test_depth_kernel_calibration.py
git commit -m "feat: estimate empirical hit and covariance depth profiles"
```

---

### Task 3: Add full-sample LOB replay for calibration snapshots

**Objective:** Build calibration snapshots from raw full-sample instrument events, not from filtered alert outputs.

**Files:**
- Modify: `src/spoofing_detection/lob/depth_kernel_calibration.py`
- Modify: `tests/lob/test_depth_kernel_calibration.py`

**Step 1: Write failing integration-style unit test**

Use the existing raw-event helper pattern from `tests/lob/test_spoofing_metrics.py`.

Test objective:
- feed a small synthetic full sample;
- ensure snapshots include top-n bid/ask levels after each event;
- ensure fill events become future fill rows;
- ensure `calibrate_empirical_depth_kernel(...)` returns one normalized kernel per side.

Sketch:

```python
from spoofing_detection.lob.depth_kernel_calibration import calibrate_empirical_depth_kernel


def test_calibrate_empirical_depth_kernel_replays_full_sample(raw_event):
    df = pl.DataFrame(
        [
            raw_event(1, 1, "B1", 1, 100.0, 10, 10, "C1"),
            raw_event(2, 1, "B2", 1, 99.9, 20, 20, "C2"),
            raw_event(3, 1, "A1", 2, 100.1, 10, 10, "C3"),
            raw_event(4, 1, "A2", 2, 100.2, 20, 20, "C4"),
            raw_event(5, 3, "A2", 2, 100.2, 0, 0, "C4", last_shares=5),
        ]
    )

    out = calibrate_empirical_depth_kernel(
        df,
        instrument_id="ABC",
        top_n=2,
        tick_size=0.1,
        horizon_seconds=10.0,
    )

    assert set(out["side"].to_list()) == {"ask", "bid"}
    assert out.group_by("side").agg(pl.col("kernel_weight").sum()).select("kernel_weight").to_series().to_list() == pytest.approx([1.0, 1.0])
    assert "kappa_hat" in out.columns
    assert "lambda_hat" in out.columns
```

Implementation detail: either copy the `raw_event` helper into the new test file or import only if it is safe. Prefer copying a minimal local helper to keep tests independent.

**Step 2: Run targeted test and confirm failure**

Run:

```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_depth_kernel_calibration.py -q
```

Expected: FAIL because replay calibration is missing.

**Step 3: Implement replay function**

Add functions to `depth_kernel_calibration.py`:

```python
def build_calibration_snapshot_tables(
    raw_events: pl.DataFrame,
    *,
    instrument_id: str,
    top_n: int,
    tick_size: float,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Replay the full sample and return level snapshots plus passive-fill rows."""
```

Implementation requirements:

- Reuse existing primitives:
  - `LOBConfig`
  - `normalize_event`
  - `sort_events`
  - `_apply_event`
  - `_flush_pending_aggressive_residuals`
  - `_fill_group_key`
  - `_partition_id`
  - `_market_levels`
  - `choose_event_timestamp`
- Snapshot after applying each event and flushing pending aggressive residuals, matching the state timing used by `compute_client_metric_time_series`.
- Snapshot schema:
  - `instrument_id`
  - `partition_id`
  - `snapshot_id`
  - `sort_index`
  - `event_ts`
  - `side`
  - `rank`
  - `price`
  - `visible_qty`
  - `mid`
  - `depth_distance_ticks`
- Fill schema:
  - `instrument_id`
  - `partition_id`
  - `sort_index`
  - `event_ts`
  - `side`
  - `price`
  - `last_shares`
- Future mid assignment:
  - for each snapshot, find the last available snapshot mid with `event_ts <= snapshot_ts + horizon_seconds` and `sort_index > snapshot_sort_index`;
  - store it as `future_mid`.

Then add:

```python
def calibrate_empirical_depth_kernel(
    raw_events: pl.DataFrame,
    *,
    instrument_id: str,
    top_n: int,
    tick_size: float,
    horizon_seconds: float,
    protection_floor: float = 0.0,
    visibility_floor: float = 0.0,
) -> pl.DataFrame:
    snapshots, fills = build_calibration_snapshot_tables(...)
    hit_profile = estimate_hit_probability_profile(snapshots, fills, horizon_seconds=horizon_seconds)
    visibility_profile = estimate_visibility_covariance_profile(snapshots)
    profile = hit_profile.join(visibility_profile, on=["instrument_id", "side", "rank", "depth_distance_ticks"], how="outer_coalesce")
    kernel = build_empirical_depth_kernel(profile, protection_floor=protection_floor, visibility_floor=visibility_floor)
    # Attach kappa_hat/lambda_hat per instrument+side as summaries.
    return kernel
```

**Step 4: Run targeted tests**

Run:

```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_depth_kernel_calibration.py -q
```

Expected: PASS.

**Step 5: Checkpoint**

If commit permission is granted:

```bash
git add src/spoofing_detection/lob/depth_kernel_calibration.py tests/lob/test_depth_kernel_calibration.py
git commit -m "feat: calibrate empirical depth kernels from full LOB samples"
```

---

### Task 4: Add a calibration CLI script

**Objective:** Create a reproducible command that estimates the kernel for one full instrument sample and writes inspectable artifacts.

**Files:**
- Create: `scripts/estimate_depth_kernel.py`
- Create: `tests/lob/test_estimate_depth_kernel_cli.py`
- Possibly modify: `configs/spoofing_detection_parameters.json`

**Step 1: Write CLI parsing tests**

Create `tests/lob/test_estimate_depth_kernel_cli.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "estimate_depth_kernel.py"


def load_module():
    spec = importlib.util.spec_from_file_location("estimate_depth_kernel", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_args_requires_full_sample_inputs(tmp_path: Path):
    module = load_module()

    args = module.parse_args(
        [
            "--input",
            str(tmp_path / "raw.parquet"),
            "--quote-panel",
            str(tmp_path / "lob_event_state_panel.parquet"),
            "--instrument-id",
            "RISANAMENTO",
            "--output-dir",
            str(tmp_path / "out"),
            "--top-n",
            "5",
            "--horizon-seconds",
            "10",
        ]
    )

    assert args.instrument_id == "RISANAMENTO"
    assert args.top_n == 5
    assert args.horizon_seconds == 10.0
    assert args.max_rows is None
```

**Step 2: Run test to verify failure**

Run:

```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_estimate_depth_kernel_cli.py -q
```

Expected: FAIL because script does not exist.

**Step 3: Implement CLI**

Create `scripts/estimate_depth_kernel.py` following the style of `scripts/compute_spoofing_metrics.py`.

Required CLI flags:

```text
--input PATH                  full raw instrument parquet
--quote-panel PATH            reconstructed quote panel, used for tick-size inference
--instrument-id TEXT          human/instrument id stored in artifacts
--output-dir PATH             output folder
--top-n INT                   top book levels to calibrate
--horizon-seconds FLOAT       H in the equations
--tick-size FLOAT             optional explicit tick size
--protection-floor FLOAT      default 0.0
--visibility-floor FLOAT      default 0.0
--max-rows INT                only for smoke tests; metadata must mark calibration as non-production if set
```

Required outputs:

```text
<output-dir>/empirical_depth_kernel.parquet
<output-dir>/empirical_depth_kernel.csv
<output-dir>/metadata.json
<output-dir>/summary_report.md
```

Metadata must include:

- command;
- `created_at_utc`;
- input path;
- quote-panel path;
- instrument id;
- top-n;
- horizon;
- tick size;
- raw event row count;
- `max_rows` and `is_full_sample_calibration = max_rows is None`;
- package/runtime versions if cheap to collect.

**Step 4: Add a tiny end-to-end CLI smoke test**

In `tests/lob/test_estimate_depth_kernel_cli.py`, write a small Parquet input and quote panel using the same raw-event helper pattern.

Run the script via `module.main([...])`, then assert:

- output files exist;
- weights sum to one by side;
- metadata records `is_full_sample_calibration` correctly.

**Step 5: Run tests**

Run:

```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_estimate_depth_kernel_cli.py tests/lob/test_depth_kernel_calibration.py -q
```

Expected: PASS.

**Step 6: Checkpoint**

If commit permission is granted:

```bash
git add scripts/estimate_depth_kernel.py tests/lob/test_estimate_depth_kernel_cli.py configs/spoofing_detection_parameters.json
git commit -m "feat: add empirical depth-kernel calibration CLI"
```

---

### Task 5: Let metric computation consume empirical kernel weights explicitly

**Objective:** Use empirical weights in MSCI/DWI calculations while preserving current scalar-parametric behavior by default.

**Files:**
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py`
- Modify: `tests/lob/test_spoofing_metrics.py`

**Step 1: Write failing unit test**

Add to `tests/lob/test_spoofing_metrics.py`:

```python
def test_compute_client_top_n_exposures_can_use_empirical_rank_weights():
    active = {
        "B1": order("B1", "bid", 100.0, 10.0, "C1"),
        "B2": order("B2", "bid", 99.9, 20.0, "C1"),
        "A1": order("A1", "ask", 100.2, 5.0, "C1"),
        "A2": order("A2", "ask", 100.3, 15.0, "C2"),
    }
    empirical_weights = {
        "bid": {1: 0.25, 2: 0.75},
        "ask": {1: 0.80, 2: 0.20},
    }

    rows = compute_client_top_n_exposures(
        active,
        top_n=2,
        tick_size=0.1,
        kappa=1.0,
        lambda_=0.5,
        partition_id="P",
        sort_index=10,
        event_ts=None,
        empirical_kernel_weights=empirical_weights,
    )

    c1 = {row["client_id"]: row for row in rows}["C1"]
    assert c1["bid_level_1_kernel_weight"] == pytest.approx(0.25)
    assert c1["bid_level_2_kernel_weight"] == pytest.approx(0.75)
    assert c1["ask_level_1_kernel_weight"] == pytest.approx(0.80)
```

**Step 2: Run test to verify failure**

Run:

```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_spoofing_metrics.py::test_compute_client_top_n_exposures_can_use_empirical_rank_weights -q
```

Expected: FAIL because `empirical_kernel_weights` is not accepted.

**Step 3: Implement minimal integration**

Change signatures in `src/spoofing_detection/lob/spoofing_metrics.py`:

- `_side_depth_metadata(..., empirical_weights_by_rank: Mapping[int, float] | None = None)`
- `compute_client_top_n_exposures(..., empirical_kernel_weights: Mapping[str, Mapping[int, float]] | None = None)`
- `_candidate_deceptive_order_rows(..., empirical_kernel_weights: Mapping[str, Mapping[int, float]] | None = None)`
- `_stream_metric_inputs(..., empirical_kernel_weights: Mapping[str, Mapping[int, float]] | None = None)`
- `compute_client_metric_time_series(..., empirical_kernel_weights=...)`
- `compute_exploratory_metrics(..., empirical_kernel_weights=...)`

Implementation rule for `_side_depth_metadata`:

```python
if empirical_weights_by_rank is None:
    weights = depth_kernel_weights(distances, kappa=kappa, lambda_=lambda_)
else:
    weights = [float(empirical_weights_by_rank.get(rank, 0.0)) for rank in range(1, len(levels) + 1)]
    total = sum(weights)
    weights = [value / total for value in weights] if total > 0 else [0.0 for _ in weights]
```

This preserves the existing per-observed-level normalization scale. If the manuscript later requires fixed full-top-n weights without renormalizing missing levels, make that a separate explicit option; do not add it now.

**Step 4: Run targeted tests**

Run:

```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_spoofing_metrics.py -q
```

Expected: PASS.

**Step 5: Checkpoint**

If commit permission is granted:

```bash
git add src/spoofing_detection/lob/spoofing_metrics.py tests/lob/test_spoofing_metrics.py
git commit -m "feat: allow empirical depth kernels in spoofing metrics"
```

---

### Task 6: Wire empirical kernel artifacts into metric CLIs

**Objective:** Let existing metric scripts load a calibration artifact without disrupting current default runs.

**Files:**
- Modify: `scripts/compute_spoofing_metrics.py`
- Modify: `scripts/run_multilevel_spoofing_grid.py`
- Modify: `tests/lob/test_compute_spoofing_metrics_cli.py`
- Modify: `tests/lob/test_spoofing_grid_runner.py`
- Modify: `configs/spoofing_detection_parameters.json`

**Step 1: Add parse-arg tests**

In both CLI test files, assert support for:

```text
--empirical-depth-kernel PATH
```

Expected args field:

```python
assert args.empirical_depth_kernel == tmp_path / "empirical_depth_kernel.parquet"
```

**Step 2: Implement loader**

Add a small helper, preferably in `depth_kernel_calibration.py`:

```python
def load_empirical_kernel_weights(path: Path | str, *, instrument_id: str | None = None) -> dict[str, dict[int, float]]:
    df = pl.read_parquet(path) if str(path).endswith(".parquet") else pl.read_csv(path)
    if instrument_id is not None and "instrument_id" in df.columns:
        df = df.filter(pl.col("instrument_id") == instrument_id)
    required = {"side", "rank", "kernel_weight"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"missing empirical kernel columns: {', '.join(missing)}")
    return {
        side: {int(row["rank"]): float(row["kernel_weight"]) for row in rows}
        for side, rows in _group_rows_by_side(df).items()
    }
```

Use an ordinary loop instead of a clever private helper if clearer.

**Step 3: Wire into scripts**

In `scripts/compute_spoofing_metrics.py`:

- add `_CONFIGURABLE_DEFAULT_KEYS` entry: `"empirical_depth_kernel"`;
- parser arg:

```python
parser.add_argument(
    "--empirical-depth-kernel",
    type=Path,
    default=None,
    help="Optional empirical_depth_kernel parquet/csv artifact. When set, rank weights override scalar kappa/lambda in DWI/MSCI weighting.",
)
```

- before `compute_exploratory_metrics`, load weights if path is set;
- pass `empirical_kernel_weights=weights`;
- write `empirical_depth_kernel` path and maybe `kernel_mode` into metadata.

Repeat for `scripts/run_multilevel_spoofing_grid.py`.

**Step 4: Run CLI tests**

Run:

```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest \
  tests/lob/test_compute_spoofing_metrics_cli.py \
  tests/lob/test_spoofing_grid_runner.py \
  tests/lob/test_spoofing_metrics.py \
  tests/lob/test_depth_kernel_calibration.py \
  -q
```

Expected: PASS.

**Step 5: Checkpoint**

If commit permission is granted:

```bash
git add scripts/compute_spoofing_metrics.py scripts/run_multilevel_spoofing_grid.py tests/lob/test_compute_spoofing_metrics_cli.py tests/lob/test_spoofing_grid_runner.py configs/spoofing_detection_parameters.json
git commit -m "feat: wire empirical depth kernels into metric CLIs"
```

---

### Task 7: Produce full-sample calibration artifacts per instrument

**Objective:** Estimate the empirical kernel on the full sample for each instrument and preserve provenance.

**Files / artifacts:**
- Read raw instrument Parquet files.
- Read quote-panel Parquet files for tick-size inference.
- Create output folders under:
  - `outputs/depth_kernel_calibration/<YYYYMMDD_HHMMSS>_<instrument>_top<top_n>_h<horizon>/`

**Step 1: Identify current full-sample inputs**

Use read-only discovery, not guessed paths:

```bash
find data outputs -name '*.parquet' | sort
```

In Hermes, prefer `search_files(pattern="*.parquet", target="files", path="data")` and `search_files(..., path="outputs")` instead of shell `find` when implementing.

For each of Risanamento, Nexi, and Ferrari, identify:

- raw full sample input parquet;
- reconstructed quote panel if needed for tick size.

**Step 2: Run calibration commands**

Template:

```bash
PYTHONPATH=src /home/danielemdn/miniconda3/envs/main/bin/python scripts/estimate_depth_kernel.py \
  --input <RAW_FULL_SAMPLE_PARQUET> \
  --quote-panel <LOB_EVENT_STATE_PANEL_PARQUET> \
  --instrument-id <INSTRUMENT_NAME> \
  --top-n 5 \
  --horizon-seconds 10.0 \
  --output-dir outputs/depth_kernel_calibration/<STAMP>_<INSTRUMENT>_top5_h10
```

Do **not** pass `--max-rows` for production calibration.

**Step 3: Inspect calibration outputs**

For each instrument:

- check weights sum to one by side;
- check `exposure_count` is nontrivial at all used levels;
- check hit probabilities are monotone-ish with rank but do not force monotonicity;
- check covariance values are finite;
- check `metadata.json` says `is_full_sample_calibration: true`.

Suggested verification script:

```bash
PYTHONPATH=src /home/danielemdn/miniconda3/envs/main/bin/python - <<'PY'
from pathlib import Path
import polars as pl
for path in sorted(Path('outputs/depth_kernel_calibration').glob('*/empirical_depth_kernel.parquet')):
    df = pl.read_parquet(path)
    sums = df.group_by(['instrument_id', 'side']).agg(pl.col('kernel_weight').sum().alias('weight_sum'))
    print(path)
    print(sums)
PY
```

Expected: every `weight_sum` is approximately `1.0`.

**Step 4: Checkpoint**

Do not commit large/generated outputs unless the user explicitly wants them versioned. Usually leave outputs uncommitted and report paths.

---

### Task 8: Optional rerun of metrics with empirical kernel

**Objective:** Recompute MSCI/DWI metrics using the calibrated empirical kernel, only after calibration artifacts pass sanity checks.

**Files / artifacts:**
- Existing metric outputs under `outputs/spoofing_metrics/...` should not be overwritten blindly.
- Create a new timestamped output folder if rerunning metrics.

**Step 1: Run one smoke instrument first**

Template:

```bash
PYTHONPATH=src /home/danielemdn/miniconda3/envs/main/bin/python scripts/compute_spoofing_metrics.py \
  --input <RAW_FULL_SAMPLE_PARQUET> \
  --quote-panel <LOB_EVENT_STATE_PANEL_PARQUET> \
  --output-dir outputs/spoofing_metrics/<STAMP>_<INSTRUMENT>_empirical_kernel_top5_h10 \
  --top-n 5 \
  --window-seconds 10.0 \
  --max-deceptive-order-age-seconds 90.0 \
  --empirical-depth-kernel <CALIBRATION_DIR>/empirical_depth_kernel.parquet
```

**Step 2: Compare against parametric baseline**

Compare:

- number of execution metric rows;
- number of candidate deceptive rows;
- top clients by MCPS;
- distribution of `MSCI`, `WMSCI_event`, and candidate weighted liquidity;
- whether matched-withdrawal-first alert counts are stable enough for review.

This is exploratory evidence, not confirmation.

**Step 3: If acceptable, rerun all three instruments**

Use distinct output folders. Do not overwrite the existing reviewed outputs unless explicitly asked.

---

### Task 9: Manuscript sync after implementation

**Objective:** Make `paper/spoofing.tex` match the implemented estimator and artifact schema.

**Files:**
- Modify: `paper/spoofing.tex:532-590`

**Manuscript edits:**

- Keep the current empirical-kernel framing.
- Add one concise sentence that the covariance profile is also the empirical basis for reporting an implied \(\widehat\lambda\):

```tex
When a scalar summary is useful, we report the log-linear decay slope of
$\widehat\nu^{m,s}_{k}(H)$ across depth as an implied $\widehat\lambda^{m,s}$;
the operational kernel, however, uses the covariance profile itself rather than
substituting a common exponential decay.
```

- Add an analogous sentence for \(\widehat\kappa\) if the implementation reports it:

```tex
Analogously, the log-linear decay slope of $\widehat p^{m,s}_{k}(H)$ provides
an implied $\widehat\kappa^{m,s}$ summary of execution risk by depth.
```

**Generated artifacts to sync:**

- If the paper includes tables/values from calibration, generate them from `empirical_depth_kernel.parquet`; do not hand-type numbers except in final LaTeX table text.
- Rebuild:

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error spoofing.tex
```

Expected: exit `0`.

---

### Task 10: Full verification and review

**Objective:** Prove the implementation is correct, reproducible, and does not regress current behavior.

**Run focused tests:**

```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest \
  tests/lob/test_depth_kernel_calibration.py \
  tests/lob/test_estimate_depth_kernel_cli.py \
  tests/lob/test_spoofing_metrics.py \
  tests/lob/test_compute_spoofing_metrics_cli.py \
  tests/lob/test_spoofing_grid_runner.py \
  -q
```

Expected: PASS.

**Run full test suite:**

```bash
/home/danielemdn/miniconda3/envs/main/bin/python -m pytest -q
```

Expected: all tests pass.

**Run LaTeX verification if manuscript changed:**

```bash
cd paper
latexmk -pdf -interaction=nonstopmode -halt-on-error spoofing.tex
```

Expected: exit `0`.

**Scientific review checklist:**

- [ ] Calibration used full samples; no alert/event/client filtering.
- [ ] `metadata.json` records `is_full_sample_calibration: true`.
- [ ] Weights sum to one by `(instrument, side)`.
- [ ] Exposure counts are reported and nonzero.
- [ ] Hit probabilities are finite and within `[0, 1]`.
- [ ] Covariance values are finite; zero-covariance levels are handled explicitly.
- [ ] Implied `kappa_hat`/`lambda_hat` are summaries, not tuning-by-alert-count.
- [ ] Existing parametric path remains default and backwards compatible.
- [ ] Empirical-kernel outputs are written to new timestamped folders, not overwriting reviewed artifacts.

---

## Files likely to change

### New files

- `src/spoofing_detection/lob/depth_kernel_calibration.py`
- `scripts/estimate_depth_kernel.py`
- `tests/lob/test_depth_kernel_calibration.py`
- `tests/lob/test_estimate_depth_kernel_cli.py`

### Modified files

- `src/spoofing_detection/lob/spoofing_metrics.py`
- `scripts/compute_spoofing_metrics.py`
- `scripts/run_multilevel_spoofing_grid.py`
- `tests/lob/test_spoofing_metrics.py`
- `tests/lob/test_compute_spoofing_metrics_cli.py`
- `tests/lob/test_spoofing_grid_runner.py`
- `configs/spoofing_detection_parameters.json`
- `paper/spoofing.tex` only after implementation artifacts exist

### Generated, not necessarily versioned

- `outputs/depth_kernel_calibration/*/empirical_depth_kernel.parquet`
- `outputs/depth_kernel_calibration/*/empirical_depth_kernel.csv`
- `outputs/depth_kernel_calibration/*/metadata.json`
- `outputs/depth_kernel_calibration/*/summary_report.md`
- optional empirical-kernel metric reruns under a new `outputs/spoofing_metrics/*empirical_kernel*` folder
- `paper/spoofing.pdf` if manuscript changes

---

## Risks, tradeoffs, and open questions

### Risks

- **Look-ahead leakage:** calibration deliberately uses future fills/mid changes to estimate a fixed instrument-level kernel. This is acceptable for offline calibration if the same full-sample kernel is then held fixed for all events, but it should not be described as real-time online detection.
- **Sparse deep levels:** deep ranks may have low exposure counts or zero covariance. Use metadata and floors; do not silently extrapolate.
- **Covariance scale:** covariance depends on how `Z` is normalized. Use a stable normalized level contribution and document it.
- **Runtime:** full-sample horizon joins can be expensive. Start with clear implementation; optimize only after profiling.
- **Comparability:** empirical weights can change MSCI scale relative to the parametric baseline. Keep old outputs and compare rather than overwrite.

### Tradeoffs

- Using raw covariance directly is transparent and matches the latest manuscript direction.
- Reporting implied \(\widehat\lambda\) as a log-decay slope of the covariance profile is useful for diagnostics, but the operational kernel should remain empirical.
- Renormalizing empirical weights over observable levels preserves the current metric scale better than treating missing levels as fixed zero mass; if the paper requires fixed-top-n mass later, add a separate explicit option.

### Open questions for the user before production reruns

1. Should empirical weights replace the scalar kernel in the main production outputs, or should we first produce a side-by-side sensitivity appendix?
2. Should \(\widehat\lambda\) be reported per side (`bid`, `ask`) or collapsed to one instrument-level value?
3. Should the calibration horizon exactly equal `metrics.window_seconds = 10.0`, or should we test robustness over `H in {1s, 5s, 10s, 30s}`?
4. Should calibration outputs be committed, or only regenerated and referenced by path?

---

## Recommended execution order

1. Implement Tasks 1-4 to create and validate calibration artifacts.
2. Inspect full-sample calibration outputs for the three instruments.
3. Only then implement Tasks 5-6 to plug empirical weights into metrics.
4. Run one smoke metric rerun with an empirical kernel.
5. Compare against baseline.
6. Update manuscript.
7. Run full verification.

