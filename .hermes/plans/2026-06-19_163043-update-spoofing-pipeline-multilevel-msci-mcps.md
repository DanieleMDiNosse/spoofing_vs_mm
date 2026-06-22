# Multilevel MSCI/MCPS Spoofing Pipeline Update Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Update the current exploratory spoofing pipeline from the old single-score SCI/candidate-fake-order version to the new paper model based on top-$n$ agent-specific depth profiles, the multilevel distance-weighted imbalance (DWI), side collapse, MSCI, and MCPS.

**Architecture:** Keep the existing order-level reconstruction as the source of truth. Replace the old `imbalance` calculation with paper-aligned top-$n$ liquidity profiles: per-level relative agent depth, normalized depth kernel weights, side liquidity `L^{i,s}_n(t)`, DWI, pre/post side collapse, event-level MSCI, and client-level MCPS. Preserve old broad/matched cancellation diagnostics as forensic audit fields, but stop treating them as the primary score.

**Tech Stack:** Python, Polars, existing LOB reconstruction utilities in `src/spoofing_detection/lob/`, Plotly dashboards, pytest.

---

## Current Context / Assumptions

The new `paper/spoofing.tex` model differs from the old implementation in five important ways:

1. The relevant object is the whole top-$n$ agent-specific depth profile, not only one candidate order or one raw side total.
2. Agent depth must be normalized level-by-level:
   `relative_depth = client_visible_qty_at_level / market_visible_qty_at_level`.
3. Distance uses a shifted tick distance so Level 1 gets positive weight:
   `d = raw_same_side_tick_distance + 1`.
4. The depth kernel is:
   `omega_k = exp(-lambda * d_k) * (1 - exp(-kappa * d_k)) / sum_l (...)`.
5. The primary event score is now:
   `MSCI = SCI * C_opposite * max(C_opposite - C_same, 0)`,
   where `C_side` is the fraction of weighted side liquidity that disappears after the fill.

Current code status:

- Main metric file: `src/spoofing_detection/lob/spoofing_metrics.py`
- Current state time series emits old columns such as:
  - `weighted_bid_fraction_topN`
  - `weighted_ask_fraction_topN`
  - `imbalance`
- Current SCI uses old `imbalance`:
  - `attach_sci_window_metrics()`
- Current outputs use old terminology:
  - `SCI`
  - `candidate_fake_visible_qty_pre`
  - broad/matched fake-order cancels
- Current dashboard uses old SCI-centric panels:
  - `src/spoofing_detection/lob/spoofing_metric_plots.py`
- Current compute CLI:
  - `scripts/compute_spoofing_metrics.py`
- Existing tests:
  - `tests/lob/test_spoofing_metrics.py`
  - `tests/lob/test_spoofing_metric_plots.py`
  - `tests/lob/test_spoofing_metric_report.py`

Important constraint:

- Do not remove the matched fake-order cancellation diagnostics. They are still useful forensic evidence, but they should become supporting columns, not the main mathematical score.
- Do not commit unless the user explicitly asks. The workspace already has many modified/untracked files.

---

## Target Output Semantics

### State time series columns

For each client and event-time state, add paper-aligned columns:

```text
lambda_
kappa
top_n
bid_level_1_depth_distance_ticks
bid_level_1_kernel_weight
bid_level_1_client_relative_depth
bid_level_1_weighted_liquidity_contribution
...
ask_level_1_depth_distance_ticks
ask_level_1_kernel_weight
ask_level_1_client_relative_depth
ask_level_1_weighted_liquidity_contribution
...
L_bid_topN
L_ask_topN
DWI
```

Keep compatibility aliases for one transition cycle:

```text
imbalance = DWI
weighted_bid_fraction_topN = L_bid_topN
weighted_ask_fraction_topN = L_ask_topN
```

This allows existing plots/tests to be updated gradually.

### Execution metric columns

For each eligible passive small execution, add:

```text
DWI_pre_window
DWI_post_window
SCI
L_bid_pre_window
L_bid_post_window
L_ask_pre_window
L_ask_post_window
collapse_bid
collapse_ask
collapse_opposite_side
collapse_same_side
MSCI
```

Preserve existing cancellation/audit fields:

```text
candidate_fake_order_count_pre
candidate_fake_visible_qty_pre
candidate_fake_order_ids_pre
has_direct_opposite_cancel_window
has_matched_fake_cancel_window
matched_fake_cancel_order_ids_window
fake_qty_collapse_fraction_window
```

### Candidate fake-order rows

For each pre-existing same-client opposite-side order inside top-$n$, add paper-aligned per-order fields:

```text
fake_order_depth_distance_ticks
fake_order_kernel_weight
fake_order_relative_depth_pre
fake_order_weighted_liquidity_contribution_pre
```

The existing raw fields remain useful:

```text
fake_order_visible_qty_pre
fake_order_level_market_qty_pre
fake_order_level_fraction_pre
fake_order_delta_ticks
```

### Client-level MCPS output

Create a new table:

```text
client_mcps_scores.parquet
```

Columns:

```text
partition_id
client_id
top_n
lambda_
kappa
gamma
executions
finite_msci_executions
msci_above_gamma_count
MCPS
median_MSCI
max_MSCI
mean_MSCI
mean_SCI
mean_collapse_opposite_side
mean_collapse_same_side
matched_fake_cancel_share
direct_opposite_cancel_share
candidate_profile_share
```

---

## Proposed Approach

1. Add small pure functions for the new mathematics first.
2. Write unit tests with hand-checkable examples.
3. Replace state-level DWI calculation while keeping old aliases.
4. Generalize SCI attachment to pull pre/post DWI and side liquidity values.
5. Compute side collapse and MSCI.
6. Add MCPS aggregation.
7. Update CLI arguments, metadata, reports, and dashboard wording.
8. Add a grid runner for `n in {1,2,3,5,10}`.
9. Regenerate Risanamento outputs and compare new counts with old counts.
10. Only after code is stable, update manuscript/report terminology if needed.

---

## Task 1: Add tests for the depth kernel and shifted distance

**Objective:** Lock down the new paper definitions before touching the streaming code.

**Files:**

- Modify: `tests/lob/test_spoofing_metrics.py`
- Modify later: `src/spoofing_detection/lob/spoofing_metrics.py`

**Step 1: Write failing tests**

Add imports for the functions that will be created:

```python
from spoofing_detection.lob.spoofing_metrics import depth_kernel_weights, shifted_depth_distance_ticks
```

Add tests:

```python
def test_shifted_depth_distance_gives_level_one_positive_distance():
    assert shifted_depth_distance_ticks("bid", price=100.0, best_price=100.0, tick_size=0.1) == pytest.approx(1.0)
    assert shifted_depth_distance_ticks("bid", price=99.9, best_price=100.0, tick_size=0.1) == pytest.approx(2.0)
    assert shifted_depth_distance_ticks("ask", price=100.2, best_price=100.2, tick_size=0.1) == pytest.approx(1.0)
    assert shifted_depth_distance_ticks("ask", price=100.3, best_price=100.2, tick_size=0.1) == pytest.approx(2.0)


def test_depth_kernel_weights_are_normalized_and_positive():
    weights = depth_kernel_weights([1.0, 2.0, 3.0], kappa=1.0, lambda_=0.5)

    assert sum(weights) == pytest.approx(1.0)
    assert all(weight > 0 for weight in weights)
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_shifted_depth_distance_gives_level_one_positive_distance tests/lob/test_spoofing_metrics.py::test_depth_kernel_weights_are_normalized_and_positive
```

Expected: FAIL because the functions do not exist yet.

**Step 3: Implement minimal functions**

In `src/spoofing_detection/lob/spoofing_metrics.py`, add near `_distance_ticks`:

```python
def shifted_depth_distance_ticks(side: str, price: float, best_price: float, tick_size: float) -> float:
    """Paper-aligned distance d_k: same-side tick distance shifted so level 1 has d=1."""
    return _distance_ticks(side, price, best_price, tick_size) + 1.0


def depth_kernel_weights(distances: list[float], *, kappa: float, lambda_: float) -> list[float]:
    """Return normalized top-n depth kernel weights from Eq. (depth_kernel)."""
    if kappa <= 0:
        raise ValueError("kappa must be positive")
    if lambda_ <= 0:
        raise ValueError("lambda_ must be positive")
    raw = [math.exp(-lambda_ * d) * (1.0 - math.exp(-kappa * d)) for d in distances]
    total = sum(raw)
    if total <= 0:
        return [0.0 for _ in distances]
    return [value / total for value in raw]
```

**Step 4: Run test to verify pass**

Run same command.

Expected: PASS.

---

## Task 2: Update `ExploratoryMetricsConfig` for lambda and epsilon

**Objective:** Add explicit model parameters for the new paper formulas.

**Files:**

- Modify: `src/spoofing_detection/lob/spoofing_metrics.py`
- Modify: `scripts/compute_spoofing_metrics.py`
- Modify tests as needed.

**Step 1: Write failing test**

In `tests/lob/test_spoofing_metrics.py`, add:

```python
def test_compute_client_metric_time_series_records_lambda_and_epsilon_defaults():
    df = pl.DataFrame([
        raw_event(1, 1, "B1", 1, 100.0, 10, 10, "C1"),
        raw_event(2, 1, "A1", 2, 100.2, 10, 10, "C1"),
    ])

    states = compute_client_metric_time_series(df, top_n=1, tick_size=0.1, kappa=1.0, lambda_=0.5)
    row = states.to_dicts()[-1]

    assert row["lambda_"] == pytest.approx(0.5)
    assert "DWI" in row
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_compute_client_metric_time_series_records_lambda_and_epsilon_defaults
```

Expected: FAIL because `lambda_` is not accepted yet.

**Step 3: Implement config and function signatures**

Update dataclass:

```python
@dataclass(frozen=True)
class ExploratoryMetricsConfig:
    top_n: int = 3
    kappa: float = 1.0
    lambda_: float = 1.0
    epsilon: float = 1e-12
    window_seconds: float = 1.0
```

Update signatures:

```python
def compute_client_top_n_exposures(..., kappa: float, lambda_: float, ...)
def compute_client_metric_time_series(..., kappa: float, lambda_: float = 1.0, ...)
def _stream_metric_inputs(..., kappa: float, lambda_: float, ...)
def compute_exploratory_metrics(..., kappa: float, lambda_: float = 1.0, epsilon: float = 1e-12, ...)
```

Pass `lambda_` through every call site.

**Step 4: Run targeted tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py
```

Expected: existing failures until Task 3 updates DWI semantics.

---

## Task 3: Replace old state imbalance with paper-aligned DWI

**Objective:** Compute `L_bid_topN`, `L_ask_topN`, and `DWI` exactly as in the new paper.

**Files:**

- Modify: `src/spoofing_detection/lob/spoofing_metrics.py:110-198`
- Modify: `tests/lob/test_spoofing_metrics.py`

**Step 1: Update/replace the old state test**

Replace `test_compute_client_top_n_exposures_uses_client_fractions_and_tick_distances` expectations.

Hand-checkable example:

- `top_n = 2`
- `tick_size = 0.1`
- `kappa = 1.0`
- `lambda_ = 0.5`
- C1 bid:
  - level 1: client 10 / market 10 = 1.0, distance 1
  - level 2: client 20 / market 50 = 0.4, distance 2
- C1 ask:
  - level 1: client 5 / market 5 = 1.0, distance 1
  - level 2: client 0 / market 15 = 0.0, distance 2

Expected implementation should compute:

```python
d_bid = [1.0, 2.0]
w_bid = depth_kernel_weights(d_bid, kappa=1.0, lambda_=0.5)
L_bid = w_bid[0] * 1.0 + w_bid[1] * 0.4
L_ask = w_bid[0] * 1.0 + w_bid[1] * 0.0
DWI = (L_ask - L_bid) / (L_ask + L_bid)
```

Update assertions:

```python
assert c1["bid_level_1_depth_distance_ticks"] == pytest.approx(1.0)
assert c1["bid_level_2_depth_distance_ticks"] == pytest.approx(2.0)
assert c1["bid_level_1_client_relative_depth"] == pytest.approx(1.0)
assert c1["bid_level_2_client_relative_depth"] == pytest.approx(0.4)
assert c1["L_bid_topN"] == pytest.approx(L_bid)
assert c1["L_ask_topN"] == pytest.approx(L_ask)
assert c1["DWI"] == pytest.approx(DWI)
assert c1["imbalance"] == pytest.approx(DWI)  # compatibility alias
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_compute_client_top_n_exposures_uses_client_fractions_and_tick_distances
```

Expected: FAIL under old weighting.

**Step 3: Implement DWI in `compute_client_top_n_exposures`**

Inside each side loop:

1. Build per-level distances first.
2. Compute per-side normalized kernel weights over existing levels.
3. Compute per-level relative depth `client_qty / market_qty`.
4. Compute side liquidity `L_side = sum(weight * relative_depth)`.

Implementation pattern:

```python
side_distances: dict[str, list[float]] = {}
side_weights: dict[str, list[float]] = {}
for side in ("bid", "ask"):
    distances = []
    for rank in range(1, top_n + 1):
        if rank <= len(levels[side]) and best[side] is not None:
            price, _ = levels[side][rank - 1]
            distances.append(shifted_depth_distance_ticks(side, price, best[side], tick_size))
        else:
            distances.append(0.0)
    positive_distances = [d for d in distances if d > 0]
    weights = depth_kernel_weights(positive_distances, kappa=kappa, lambda_=lambda_)
    # map weights back by rank; missing levels get 0.
```

For each level write:

```python
row[f"{side}_level_{rank}_depth_distance_ticks"] = depth_distance
row[f"{side}_level_{rank}_kernel_weight"] = kernel_weight
row[f"{side}_level_{rank}_client_relative_depth"] = relative_depth
row[f"{side}_level_{rank}_weighted_liquidity_contribution"] = kernel_weight * relative_depth
```

At side level:

```python
L_side = sum(level_contributions)
row[f"L_{side}_topN"] = L_side
row[f"weighted_{side}_fraction_topN"] = L_side  # compatibility alias
```

At row level:

```python
denom = L_ask + L_bid
row["DWI_denominator"] = denom
row["DWI"] = (L_ask - L_bid) / denom if denom > 0 else None
row["imbalance_denominator"] = denom
row["imbalance"] = row["DWI"]
```

**Step 4: Run targeted tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_compute_client_top_n_exposures_uses_client_fractions_and_tick_distances tests/lob/test_spoofing_metrics.py::test_compute_client_metric_time_series_emits_client_only_top_n_states
```

Expected: PASS after updating expectations.

---

## Task 4: Update candidate fake-order rows with kernel contribution fields

**Objective:** Make candidate profile rows reflect the new top-$n$ depth profile, not just raw order quantities.

**Files:**

- Modify: `src/spoofing_detection/lob/spoofing_metrics.py:298-393`
- Modify: `tests/lob/test_spoofing_metrics.py`

**Step 1: Add failing assertions to existing candidate fake-order test**

In `test_exploratory_metrics_detect_same_client_opposite_side_direct_cancel_window`, add:

```python
assert candidate["fake_order_depth_distance_ticks"] == pytest.approx(2.0)
assert candidate["fake_order_relative_depth_pre"] == pytest.approx(50.0 / 150.0)
assert 0.0 <= candidate["fake_order_kernel_weight"] <= 1.0
assert candidate["fake_order_weighted_liquidity_contribution_pre"] == pytest.approx(
    candidate["fake_order_kernel_weight"] * candidate["fake_order_relative_depth_pre"]
)
```

The exact denominator depends on the test's opposite-side market level. Re-check the synthetic orders before finalizing the expected value.

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_exploratory_metrics_detect_same_client_opposite_side_direct_cancel_window
```

Expected: FAIL because columns do not exist.

**Step 3: Implement fields**

Update `_candidate_fake_order_rows(...)` to accept `kappa` and `lambda_`.

Within the fake side:

```python
distances = [shifted_depth_distance_ticks(fake_side, price, best_price, tick_size) for price, _ in levels]
weights = depth_kernel_weights(distances, kappa=kappa, lambda_=lambda_)
price_to_weight = {price: weights[idx] for idx, (price, _) in enumerate(levels)}
price_to_distance = {price: distances[idx] for idx, (price, _) in enumerate(levels)}
```

Add columns:

```python
relative_depth = qty / market_qty if market_qty > 0 else 0.0
kernel_weight = price_to_weight[price]
weighted_contribution = kernel_weight * relative_depth
```

Row additions:

```python
"fake_order_depth_distance_ticks": price_to_distance[price],
"fake_order_kernel_weight": kernel_weight,
"fake_order_relative_depth_pre": relative_depth,
"fake_order_weighted_liquidity_contribution_pre": weighted_contribution,
```

Keep old columns:

```python
"fake_order_delta_ticks": _distance_ticks(...),
"fake_order_level_fraction_pre": relative_depth,
```

**Step 4: Update candidate summary**

Add:

```python
candidate_fake_weighted_liquidity_pre
candidate_fake_max_weighted_liquidity_contribution_pre
candidate_fake_qty_weighted_depth_distance_ticks_pre
```

Use weighted contributions for the profile interpretation, raw visible quantity for forensic interpretation.

**Step 5: Run test**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_exploratory_metrics_detect_same_client_opposite_side_direct_cancel_window
```

Expected: PASS.

---

## Task 5: Generalize SCI attachment to DWI and side liquidity snapshots

**Objective:** `attach_sci_window_metrics()` must return pre/post DWI and side liquidity values needed by MSCI.

**Files:**

- Modify: `src/spoofing_detection/lob/spoofing_metrics.py:548-627`
- Modify: `tests/lob/test_spoofing_metrics.py`

**Step 1: Replace old state fixture in SCI test**

Update `test_attach_sci_window_metrics_uses_fixed_clock_window` to use:

```python
states = pl.DataFrame(
    {
        "partition_id": ["P", "P", "P"],
        "client_id": ["C1", "C1", "C1"],
        "event_ts": [
            datetime(2024, 1, 2, 9, 30, 8),
            datetime(2024, 1, 2, 9, 30, 9),
            datetime(2024, 1, 2, 9, 30, 11),
        ],
        "DWI": [-0.2, -0.8, -0.1],
        "L_bid_topN": [0.4, 0.9, 0.2],
        "L_ask_topN": [0.2, 0.1, 0.18],
    }
)
executions = pl.DataFrame(
    {
        "partition_id": ["P"],
        "client_id": ["C1"],
        "event_ts": [datetime(2024, 1, 2, 9, 30, 10)],
        "execution_side": ["ask"],
        "fake_side": ["bid"],
    }
)
```

Expected:

```python
assert out.item(0, "DWI_pre_window") == pytest.approx(-0.8)
assert out.item(0, "DWI_post_window") == pytest.approx(-0.1)
assert out.item(0, "SCI") == pytest.approx(0.7)
assert out.item(0, "L_bid_pre_window") == pytest.approx(0.9)
assert out.item(0, "L_bid_post_window") == pytest.approx(0.2)
assert out.item(0, "collapse_bid") == pytest.approx((0.9 - 0.2) / 0.9)
assert out.item(0, "collapse_opposite_side") == pytest.approx((0.9 - 0.2) / 0.9)
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_attach_sci_window_metrics_uses_fixed_clock_window
```

Expected: FAIL because only `imbalance_pre_window` exists.

**Step 3: Implement generic state lookup**

Replace `_group_state_rows()` with a helper that stores multiple values:

```python
def _group_state_rows(states: pl.DataFrame) -> dict[tuple[Any, str], list[dict[str, Any]]]:
    required = {"partition_id", "client_id", "event_ts", "DWI", "L_bid_topN", "L_ask_topN"}
    ...
```

Sort by timestamp.

Add helper:

```python
def _lookup_state_at_or_before(rows: list[dict[str, Any]], target: datetime) -> dict[str, Any] | None:
    # use bisect on parsed event_ts values
```

In `attach_sci_window_metrics()`:

```python
pre_state = lookup(..., pre_target)
post_state = lookup(..., post_target)
pre_dwi = pre_state.get("DWI") if pre_state else None
post_dwi = post_state.get("DWI") if post_state else None
SCI = abs(pre_dwi - post_dwi) if both finite else None
```

Add compatibility aliases:

```python
"imbalance_pre_window": pre_dwi,
"imbalance_post_window": post_dwi,
```

Add side liquidity fields:

```python
"L_bid_pre_window": ...
"L_bid_post_window": ...
"L_ask_pre_window": ...
"L_ask_post_window": ...
```

**Step 4: Run test**

Run targeted test. Expected: PASS.

---

## Task 6: Compute side collapse and MSCI

**Objective:** Implement Eq. (side_collapse) and Eq. (msci) from the new paper.

**Files:**

- Modify: `src/spoofing_detection/lob/spoofing_metrics.py`
- Modify: `tests/lob/test_spoofing_metrics.py`

**Step 1: Add MSCI test**

Add to the SCI window test or a new test:

```python
def test_msci_uses_opposite_side_collapse_minus_same_side_collapse():
    states = pl.DataFrame(
        {
            "partition_id": ["P", "P"],
            "client_id": ["C1", "C1"],
            "event_ts": [datetime(2024, 1, 2, 9, 30, 9), datetime(2024, 1, 2, 9, 30, 11)],
            "DWI": [-0.8, -0.1],
            "L_bid_topN": [0.9, 0.2],
            "L_ask_topN": [0.4, 0.3],
        }
    )
    executions = pl.DataFrame(
        {
            "partition_id": ["P"],
            "client_id": ["C1"],
            "event_ts": [datetime(2024, 1, 2, 9, 30, 10)],
            "execution_side": ["ask"],
            "fake_side": ["bid"],
        }
    )

    out = attach_sci_window_metrics(executions, states, window_seconds=1.0, epsilon=1e-12)

    sci = 0.7
    c_bid = (0.9 - 0.2) / 0.9
    c_ask = (0.4 - 0.3) / 0.4
    expected_msci = sci * c_bid * max(c_bid - c_ask, 0.0)

    assert out.item(0, "collapse_opposite_side") == pytest.approx(c_bid)
    assert out.item(0, "collapse_same_side") == pytest.approx(c_ask)
    assert out.item(0, "MSCI") == pytest.approx(expected_msci)
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_msci_uses_opposite_side_collapse_minus_same_side_collapse
```

Expected: FAIL until implemented.

**Step 3: Implement collapse function**

Add helper:

```python
def _collapse(pre: float | None, post: float | None, *, epsilon: float) -> float | None:
    if pre is None or post is None:
        return None
    if pre + epsilon <= 0:
        return None
    return max(pre - post, 0.0) / (pre + epsilon)
```

In `attach_sci_window_metrics()`:

```python
collapse_bid = _collapse(L_bid_pre, L_bid_post, epsilon=epsilon)
collapse_ask = _collapse(L_ask_pre, L_ask_post, epsilon=epsilon)
if execution_side == "ask":
    c_opposite = collapse_bid
    c_same = collapse_ask
elif execution_side == "bid":
    c_opposite = collapse_ask
    c_same = collapse_bid
else:
    c_opposite = None
    c_same = None
MSCI = sci * c_opposite * max(c_opposite - c_same, 0.0) if all finite else None
```

**Step 4: Run tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py
```

Expected: PASS after updating old assertions.

---

## Task 7: Add MCPS aggregation

**Objective:** Add client-level aggregation of repeated MSCI threshold crossings.

**Files:**

- Modify: `src/spoofing_detection/lob/spoofing_metrics.py`
- Create or modify: `tests/lob/test_spoofing_mcps.py` or append to `tests/lob/test_spoofing_metrics.py`

**Step 1: Write failing test**

Add:

```python
from spoofing_detection.lob.spoofing_metrics import compute_mcps_scores


def test_compute_mcps_scores_groups_by_client_and_gamma():
    executions = pl.DataFrame(
        {
            "partition_id": ["P", "P", "P", "P"],
            "client_id": ["C1", "C1", "C1", "C2"],
            "top_n": [3, 3, 3, 3],
            "kappa": [1.0, 1.0, 1.0, 1.0],
            "lambda_": [0.5, 0.5, 0.5, 0.5],
            "MSCI": [0.2, 0.8, None, 0.9],
            "SCI": [0.4, 0.9, None, 1.0],
            "collapse_opposite_side": [0.5, 0.9, None, 0.8],
            "collapse_same_side": [0.1, 0.2, None, 0.1],
            "has_matched_fake_cancel_window": [False, True, False, True],
            "has_direct_opposite_cancel_window": [True, True, False, True],
            "candidate_fake_order_count_pre": [1, 1, 0, 1],
        }
    )

    scores = compute_mcps_scores(executions, gamma_grid=[0.5])
    c1 = scores.filter(pl.col("client_id") == "C1").to_dicts()[0]

    assert c1["executions"] == 3
    assert c1["finite_msci_executions"] == 2
    assert c1["msci_above_gamma_count"] == 1
    assert c1["MCPS"] == pytest.approx(1 / 3)
```

**Step 2: Run test to verify failure**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_compute_mcps_scores_groups_by_client_and_gamma
```

Expected: FAIL because function does not exist.

**Step 3: Implement `compute_mcps_scores`**

Add to `spoofing_metrics.py`:

```python
def compute_mcps_scores(execution_metrics: pl.DataFrame, *, gamma_grid: list[float]) -> pl.DataFrame:
    if execution_metrics.is_empty():
        return pl.DataFrame()
    rows = []
    group_cols = [col for col in ("partition_id", "client_id", "top_n", "kappa", "lambda_") if col in execution_metrics.columns]
    for gamma in gamma_grid:
        for key, group in execution_metrics.group_by(group_cols):
            row = dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))
            executions = group.height
            finite = group.filter(pl.col("MSCI").is_not_null())
            above = finite.filter(pl.col("MSCI") > gamma).height
            row.update(
                {
                    "gamma": gamma,
                    "executions": executions,
                    "finite_msci_executions": finite.height,
                    "msci_above_gamma_count": above,
                    "MCPS": above / executions if executions > 0 else None,
                    "median_MSCI": finite.select(pl.col("MSCI").median()).item() if finite.height else None,
                    "max_MSCI": finite.select(pl.col("MSCI").max()).item() if finite.height else None,
                    "mean_MSCI": finite.select(pl.col("MSCI").mean()).item() if finite.height else None,
                    "mean_SCI": finite.select(pl.col("SCI").mean()).item() if "SCI" in finite.columns and finite.height else None,
                    "mean_collapse_opposite_side": finite.select(pl.col("collapse_opposite_side").mean()).item() if "collapse_opposite_side" in finite.columns and finite.height else None,
                    "mean_collapse_same_side": finite.select(pl.col("collapse_same_side").mean()).item() if "collapse_same_side" in finite.columns and finite.height else None,
                    "matched_fake_cancel_share": group.select(pl.col("has_matched_fake_cancel_window").cast(pl.Float64).mean()).item() if "has_matched_fake_cancel_window" in group.columns else None,
                    "direct_opposite_cancel_share": group.select(pl.col("has_direct_opposite_cancel_window").cast(pl.Float64).mean()).item() if "has_direct_opposite_cancel_window" in group.columns else None,
                    "candidate_profile_share": group.select((pl.col("candidate_fake_order_count_pre") > 0).cast(pl.Float64).mean()).item() if "candidate_fake_order_count_pre" in group.columns else None,
                }
            )
            rows.append(row)
    return pl.DataFrame(rows, infer_schema_length=None)
```

Polars group key handling may need small adjustment; verify with tests.

**Step 4: Run tests**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py
```

Expected: PASS.

---

## Task 8: Update compute CLI for lambda, epsilon, gamma grid, and MCPS output

**Objective:** Make the command-line pipeline produce all paper-aligned outputs.

**Files:**

- Modify: `scripts/compute_spoofing_metrics.py`
- Modify: `tests/lob/test_spoofing_metric_report.py`

**Step 1: Add CLI arguments**

Add to `parse_args()`:

```python
parser.add_argument("--lambda", dest="lambda_", type=float, default=1.0, help="Visibility-decay parameter for the top-n depth kernel")
parser.add_argument("--epsilon", type=float, default=1e-12, help="Small denominator stabilizer for side-collapse ratios")
parser.add_argument("--gamma-grid", type=str, default="0.25,0.5,0.75,1.0", help="Comma-separated MSCI thresholds for MCPS")
```

Add helper:

```python
def _parse_float_grid(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise ValueError("grid must contain at least one value")
    return values
```

**Step 2: Update compute call**

```python
gamma_grid = _parse_float_grid(args.gamma_grid)
result = compute_exploratory_metrics(..., lambda_=args.lambda_, epsilon=args.epsilon)
mcps_scores = compute_mcps_scores(result.execution_metrics, gamma_grid=gamma_grid)
```

**Step 3: Add output path**

```python
"client_mcps_scores": args.output_dir / "client_mcps_scores.parquet",
```

Write it:

```python
_write_parquet(mcps_scores, paths["client_mcps_scores"])
```

**Step 4: Update metadata**

Add:

```python
"lambda_": args.lambda_,
"epsilon": args.epsilon,
"gamma_grid": gamma_grid,
```

Row counts:

```python
"client_mcps_scores": mcps_scores.height,
```

**Step 5: Update summary report wording**

Replace old report title and explanation with:

```markdown
# Multilevel top-n spoofing surveillance metrics

This report follows the paper's top-n model. It is not a spoofing label.

- DWI: agent ask-minus-bid weighted top-n profile.
- SCI: absolute DWI change around a small passive execution.
- Collapse: fraction of weighted liquidity that disappears after the execution.
- MSCI: high only when DWI jumps and the opposite-side profile collapses more than the same side.
- MCPS: fraction of the client's executions whose MSCI exceeds gamma.
```

Add sections:

```markdown
## Top clients by MCPS
## Top executions by MSCI
## Top matched fake-order cancellations
```

**Step 6: Test report**

Update `tests/lob/test_spoofing_metric_report.py` to assert:

```python
assert "Multilevel top-n spoofing surveillance metrics" in report
assert "MSCI" in report
assert "MCPS" in report
assert "Top clients by MCPS" in report
```

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metric_report.py
```

Expected: PASS.

---

## Task 9: Update dashboard for DWI/MSCI/MCPS semantics

**Objective:** Replace the old SCI-centric dashboard with a paper-aligned dashboard that is still easy to read.

**Files:**

- Modify: `src/spoofing_detection/lob/spoofing_metric_plots.py`
- Modify: `scripts/plot_spoofing_metrics.py`
- Modify: `tests/lob/test_spoofing_metric_plots.py`

**Step 1: Update plot function signature**

Add optional MCPS table:

```python
def write_spoofing_metric_dashboard(
    *,
    execution_metrics: pl.DataFrame,
    state_time_series: pl.DataFrame | None,
    output_html: str | Path,
    title: str,
    client_id: str | None = None,
    mcps_scores: pl.DataFrame | None = None,
) -> None:
```

**Step 2: Update dashboard panels**

Target panels:

1. `MSCI over execution time`
   - y-axis: `MSCI`
   - marker color:
     - red = matched fake-order cancel
     - orange = broad opposite-side cancel only
     - blue = no opposite-side cancel
2. `MSCI distribution`
3. `Opposite-side collapse vs same-side collapse`
   - x: `collapse_same_side`
   - y: `collapse_opposite_side`
   - diagonal line `y=x`
4. `Candidate profile size vs small execution size`
5. `Top clients by MCPS`
6. `Selected-client DWI time series`

**Step 3: Keep simple wording in HTML note**

Use plain-language note:

```html
<p><b>How to read this dashboard:</b> DWI shows whether a client is ask-heavy or bid-heavy in the top-n book. MSCI becomes large only when the client's imbalance changes quickly and the liquidity that disappears is mostly on the side opposite to the small execution. Red points are cases where one of the pre-existing opposite-side candidate orders was directly cancelled after the execution.</p>
```

**Step 4: Update plot CLI**

In `scripts/plot_spoofing_metrics.py`, add optional:

```python
parser.add_argument("--mcps-scores", type=Path, default=None)
```

Load and pass if provided.

**Step 5: Update tests**

In `tests/lob/test_spoofing_metric_plots.py`, change assertions:

```python
assert "MSCI" in html
assert "opposite-side collapse" in html
assert "MCPS" in html
assert "DWI" in html
```

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metric_plots.py
```

Expected: PASS.

---

## Task 10: Add a multidepth grid runner

**Objective:** Implement the paper's recommended multiscale vector `MCPS_n` for `n in {1,2,3,5,10}`.

**Files:**

- Create: `scripts/run_multilevel_spoofing_grid.py`
- Create: `tests/lob/test_spoofing_grid_runner.py` if practical, or add a smoke test around argument parsing only.

**Step 1: CLI design**

Script arguments:

```text
--input PATH
--quote-panel PATH
--output-dir PATH
--depth-grid 1,2,3,5,10
--kappa 1.0
--lambda 1.0
--window-seconds 1.0
--gamma-grid 0.25,0.5,0.75,1.0
--tick-size optional
--max-rows optional
--make-dashboard
```

**Step 2: Implementation outline**

Do not shell out to `compute_spoofing_metrics.py`; import its helpers or import directly from `spoofing_metrics.py`.

Pseudo-code:

```python
for top_n in depth_grid:
    run_dir = output_dir / f"topn_{top_n}"
    result = compute_exploratory_metrics(raw_events, top_n=top_n, tick_size=tick_size, kappa=kappa, lambda_=lambda_, epsilon=epsilon, window_seconds=window_seconds)
    scores = compute_mcps_scores(result.execution_metrics, gamma_grid=gamma_grid)
    write all parquet files under run_dir
    collect scores with top_n column

combined_scores = pl.concat(all_scores)
combined_scores.write_parquet(output_dir / "combined_client_mcps_scores.parquet")
```

Also write:

```text
metadata.json
summary_report.md
```

**Step 3: Summary report content**

Include simple explanations:

```markdown
## Multidepth interpretation

- If MCPS is high at n=1, the suspicious profile is close to the best quote.
- If MCPS is low at n=1 but high at n=5 or n=10, the suspicious profile is deeper in the book.
- Stable high MCPS across several n values is a stronger surveillance cue than a single-depth spike.
```

**Step 4: Smoke command**

After implementation, run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/run_multilevel_spoofing_grid.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --quote-panel outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG/lob_event_state_panel.parquet \
  --output-dir outputs/spoofing_metrics/risanamento_multilevel_grid_smoke \
  --depth-grid 1,3 \
  --kappa 1.0 \
  --lambda 1.0 \
  --window-seconds 1.0 \
  --gamma-grid 0.25,0.5 \
  --max-rows 5000
```

Expected: output directory with per-depth outputs and `combined_client_mcps_scores.parquet`.

---

## Task 11: Update output/report examples from old SCI to new MSCI/MCPS

**Objective:** Make generated reports explain the new paper model in simple wording.

**Files:**

- Modify: `scripts/compute_spoofing_metrics.py`
- Modify: `scripts/run_multilevel_spoofing_grid.py`
- Modify: tests around report text.

**Required report sections:**

```markdown
## How to read this report

DWI tells us whether the client is mostly visible on the ask side or the bid side in the top-n book.

SCI tells us whether this top-n profile changed sharply around a small execution.

Collapse tells us which side disappeared after the execution.

MSCI combines these ideas. It is high only when:
1. DWI changes sharply;
2. the opposite side from the execution collapses;
3. the opposite side collapses more than the execution side.

MCPS is a client-level repetition score. It asks: out of all small executions by this client, how often was MSCI above a chosen threshold?
```

Add explicit warning:

```markdown
These scores are surveillance cues, not labels. A high MSCI/MCPS case needs episode-level review.
```

**Example rows to include:**

1. Top executions by `MSCI`.
2. Top clients by `MCPS`.
3. Top matched fake-order cancellations.
4. For grid runs, top clients whose `MCPS` increases with `n`.

---

## Task 12: Regenerate Risanamento outputs with the new single-depth pipeline

**Objective:** Produce a comparable output for `top_n=3` under the new paper-aligned metrics.

**Command:**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/compute_spoofing_metrics.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --quote-panel outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG/lob_event_state_panel.parquet \
  --output-dir outputs/spoofing_metrics/risanamento_top3_multilevel_msci \
  --top-n 3 \
  --kappa 1.0 \
  --lambda 1.0 \
  --window-seconds 1.0 \
  --gamma-grid 0.25,0.5,0.75,1.0
```

**Dashboard command:**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/plot_spoofing_metrics.py \
  --execution-metrics outputs/spoofing_metrics/risanamento_top3_multilevel_msci/execution_metrics.parquet \
  --state-time-series outputs/spoofing_metrics/risanamento_top3_multilevel_msci/client_metric_time_series.parquet \
  --mcps-scores outputs/spoofing_metrics/risanamento_top3_multilevel_msci/client_mcps_scores.parquet \
  --output-html outputs/spoofing_metrics/risanamento_top3_multilevel_msci/spoofing_metric_dashboard.html \
  --title "Risanamento multilevel spoofing metrics — top n = 3"
```

**Verify:**

```bash
python - <<'PY'
from pathlib import Path
root = Path('outputs/spoofing_metrics/risanamento_top3_multilevel_msci')
for name in [
    'client_metric_time_series.parquet',
    'execution_metrics.parquet',
    'candidate_fake_orders.parquet',
    'direct_cancellations.parquet',
    'client_mcps_scores.parquet',
    'summary_report.md',
    'spoofing_metric_dashboard.html',
]:
    path = root / name
    print(name, path.exists(), path.stat().st_size if path.exists() else None)
PY
```

Expected: all files exist and have non-zero size.

---

## Task 13: Run multidepth grid on Risanamento

**Objective:** Produce the paper's `MCPS_n` multiscale alert profile.

**Command:**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/run_multilevel_spoofing_grid.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --quote-panel outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG/lob_event_state_panel.parquet \
  --output-dir outputs/spoofing_metrics/risanamento_multilevel_grid \
  --depth-grid 1,2,3,5,10 \
  --kappa 1.0 \
  --lambda 1.0 \
  --window-seconds 1.0 \
  --gamma-grid 0.25,0.5,0.75,1.0 \
  --make-dashboard
```

**Verify outputs:**

```text
outputs/spoofing_metrics/risanamento_multilevel_grid/
  metadata.json
  summary_report.md
  combined_client_mcps_scores.parquet
  topn_1/
  topn_2/
  topn_3/
  topn_5/
  topn_10/
```

**Inspect top clients:**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python - <<'PY'
import polars as pl
scores = pl.read_parquet('outputs/spoofing_metrics/risanamento_multilevel_grid/combined_client_mcps_scores.parquet')
print(scores.sort(['MCPS', 'max_MSCI'], descending=[True, True]).head(20))
PY
```

Expected: printed table with top clients across depths and gamma values.

---

## Task 14: Update documentation / skill reference

**Objective:** Prevent future work from accidentally using the old paper version again.

**Files:**

- Modify if present/relevant: `docs/lob_implementation_status.md`
- Modify if present/relevant: `.hermes/plans/2026-06-19_114738-client-spoofing-metrics-exploration.md` only if we want to mark it superseded.
- Patch existing skill reference loaded earlier:
  - `scientific-analysis-pipelines` linked reference `references/lob-client-spoofing-metrics-exploration.md`

**Content to add:**

```markdown
## Current spoofing model version

The active manuscript model is the multilevel top-n DWI/MSCI/MCPS formulation:
- top-n agent-specific depth vectors;
- per-level relative depth;
- normalized depth kernel with kappa and lambda;
- DWI state score;
- side collapse after small executions;
- MSCI event score;
- MCPS client-level repetition score.

The older single `imbalance` / `SCI` implementation is superseded except as a compatibility alias and diagnostic baseline.
```

**Verification:**

Use `read_file` or `skill_view` to confirm the reference says the old SCI-only version is superseded.

---

## Task 15: Full validation pass

**Objective:** Verify software behavior and scientific behavior before reporting completion.

**Commands:**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q
```

Expected:

```text
all tests passed
```

Compile changed scripts/modules:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python -m py_compile \
  src/spoofing_detection/lob/spoofing_metrics.py \
  src/spoofing_detection/lob/spoofing_metric_plots.py \
  scripts/compute_spoofing_metrics.py \
  scripts/plot_spoofing_metrics.py \
  scripts/run_multilevel_spoofing_grid.py
```

Expected: no output and exit code 0.

Run a smoke real-data job before the full job:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/compute_spoofing_metrics.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --quote-panel outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG/lob_event_state_panel.parquet \
  --output-dir outputs/spoofing_metrics/smoke_risanamento_multilevel_msci \
  --top-n 3 \
  --kappa 1.0 \
  --lambda 1.0 \
  --window-seconds 1.0 \
  --gamma-grid 0.25,0.5 \
  --max-rows 5000
```

Expected: all expected parquet and report outputs exist.

Scientific sanity checks:

1. `DWI` must be in `[-1, 1]` whenever finite.
2. `collapse_bid`, `collapse_ask`, `collapse_opposite_side`, `collapse_same_side` must be in `[0, 1]` whenever finite.
3. `MSCI` must be non-negative.
4. `MCPS` must be in `[0, 1]`.
5. At least some rows should have finite `DWI`, `SCI`, and `MSCI` on real data.
6. Top MCPS clients should have enough executions to be meaningful; do not over-interpret clients with tiny `N_i`.

Add a small inspection script or one-off command:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python - <<'PY'
import polars as pl
root = 'outputs/spoofing_metrics/risanamento_top3_multilevel_msci'
execs = pl.read_parquet(f'{root}/execution_metrics.parquet')
scores = pl.read_parquet(f'{root}/client_mcps_scores.parquet')
print(execs.select(
    pl.col('DWI_pre_window').is_not_null().sum().alias('finite_DWI_pre'),
    pl.col('SCI').is_not_null().sum().alias('finite_SCI'),
    pl.col('MSCI').is_not_null().sum().alias('finite_MSCI'),
    pl.col('MSCI').min().alias('min_MSCI'),
    pl.col('MSCI').max().alias('max_MSCI'),
))
print(scores.select(
    pl.col('MCPS').min().alias('min_MCPS'),
    pl.col('MCPS').max().alias('max_MCPS'),
))
PY
```

Expected: finite counts are non-zero; min/max satisfy bounds.

---

## Files Likely to Change

Core code:

- `src/spoofing_detection/lob/spoofing_metrics.py`
- `src/spoofing_detection/lob/spoofing_metric_plots.py`

Scripts:

- `scripts/compute_spoofing_metrics.py`
- `scripts/plot_spoofing_metrics.py`
- `scripts/run_multilevel_spoofing_grid.py` new

Tests:

- `tests/lob/test_spoofing_metrics.py`
- `tests/lob/test_spoofing_metric_plots.py`
- `tests/lob/test_spoofing_metric_report.py`
- `tests/lob/test_spoofing_grid_runner.py` optional/new

Documentation / procedural memory:

- `docs/lob_implementation_status.md` if present/relevant
- `scientific-analysis-pipelines` skill reference `references/lob-client-spoofing-metrics-exploration.md`

Generated outputs:

- `outputs/spoofing_metrics/risanamento_top3_multilevel_msci/`
- `outputs/spoofing_metrics/risanamento_multilevel_grid/`

---

## Risks, Tradeoffs, and Open Questions

### Risk 1: The old `imbalance` denominator was different

Old implementation divided by raw side fractions. New DWI divides by weighted top-$n$ side liquidity. This will change all values and rankings.

Mitigation:

- Keep old `imbalance` alias only after changing it to equal `DWI`.
- Use report wording to say the old SCI-only outputs are superseded.

### Risk 2: Choice of `lambda` is not calibrated

The new paper introduces `lambda` but does not give a default.

Mitigation:

- Use `lambda_=1.0` as an exploratory default.
- Expose CLI option.
- Include sensitivity grid later if results depend strongly on it.

### Risk 3: MCPS can be unstable for low-execution clients

A client with one execution can have MCPS = 1.

Mitigation:

- Always report `executions` and `finite_msci_executions`.
- In reports/dashboard, sort by MCPS but include execution count.
- Later add `--min-executions` for display only, not for raw output.

### Risk 4: Side collapse can happen without direct cancellation

The paper says update through executions, cancellations, and amendments. The MSCI side-collapse score should capture any disappearance of weighted liquidity, not only direct cancels. Matched fake-order cancellation remains a forensic explanation, not a required condition.

Mitigation:

- Do not multiply MSCI by `has_matched_fake_cancel_window`.
- Keep matched cancel columns as audit evidence.

### Risk 5: Dense grid runs may be slow

Running top-n grid `{1,2,3,5,10}` repeats reconstruction five times.

Mitigation:

- Implement correctness first.
- Use `--max-rows` smoke runs.
- Optimize only if runtime becomes a real bottleneck.

---

## Completion Criteria

The update is complete when:

1. Unit tests pass for:
   - shifted depth distance;
   - normalized kernel weights;
   - DWI state rows;
   - side collapse;
   - MSCI;
   - MCPS aggregation;
   - updated reports and dashboard wording.
2. Full test suite passes.
3. `py_compile` passes for changed modules/scripts.
4. A Risanamento single-depth run produces:
   - `client_metric_time_series.parquet`
   - `execution_metrics.parquet`
   - `candidate_fake_orders.parquet`
   - `direct_cancellations.parquet`
   - `client_mcps_scores.parquet`
   - `summary_report.md`
   - `spoofing_metric_dashboard.html`
5. A Risanamento multidepth grid run produces:
   - per-depth outputs;
   - `combined_client_mcps_scores.parquet`;
   - a summary explaining the multiscale interpretation.
6. Reports use simple language and clearly state that MSCI/MCPS are surveillance cues, not proof of intent.
7. Old SCI-only terminology is marked as superseded in the relevant docs/skill reference.
