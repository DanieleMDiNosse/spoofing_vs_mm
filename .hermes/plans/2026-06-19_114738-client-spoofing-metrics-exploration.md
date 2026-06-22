# Client-Level Spoofing Metrics Exploration Implementation Plan

> Superseded schema note, 2026-06-20: the active implementation has moved from this first-pass single-imbalance SCI/CPS
> exploration to the clean multilevel DWI/MSCI/MCPS schema in
> `.hermes/plans/2026-06-19_163043-update-spoofing-pipeline-multilevel-msci-mcps.md`. New outputs should use `DWI`,
> `L_bid_topN`, `L_ask_topN`, `MSCI`, `MCPS`, and `candidate_deceptive_*` names directly; do not reintroduce legacy
> `imbalance`, `weighted_*_fraction_topN`, or `candidate_fake_*` compatibility aliases.

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Build a small, exploratory, client-level pipeline that validates the client/proprietary-account assumption, computes top-N client spoofing-metric time series around limit-order executions, and produces inspectable outputs/HTML diagnostics for one instrument first.

**Architecture:** Keep this as an exploratory layer on top of the existing LOB reconstruction, not a production detector. Add a small client-identity audit, a new pure metrics module, one compute CLI, and one visualization/reporting path. Reuse the existing normalized event semantics and visible-order state logic, but compute client top-N exposures explicitly because the current `agent_event_state_panel.parquet` only contains event-agent aggregate totals and not client top-N fractions by level.

**Tech Stack:** Python 3, Polars, Plotly, pytest, existing `spoofing_detection.lob` reconstruction modules.

---

## Current context / assumptions

- Active workspace: `/home/danielemdn/Documents/repositories/spoofing_detection`.
- Existing relevant code:
  - `src/spoofing_detection/lob/normalize.py` normalizes raw event rows and currently preserves `ORDER_TRADINGCAPACITY (*)`, but not the tooltip column.
  - `src/spoofing_detection/lob/panel.py` holds the visible-book state machine, active order model, market top-N summaries, and agent aggregate rows.
  - `src/spoofing_detection/lob/models.py` defines `ActiveOrder` with `firm_id` and `client_original_id`.
  - `scripts/plot_best_quotes.py` and `src/spoofing_detection/lob/quote_diagnostics.py` now produce event-time quote diagnostics and are useful style references for HTML + metadata output.
- Raw data columns inspected:
  - All three raw parquet files have `ORDER_TRADINGCAPACITY (*)` and `NMSC_ORIGINALCLIENTIDSHORTCODE`.
  - Ferrari and Risanamento have `ORDER_TRADINGCAPACITY (*) (Tooltip)`.
  - Nexi does not expose the tooltip column in the raw schema inspected, so the audit must validate the numeric code everywhere and validate the tooltip text only where available.
- Existing fast post-fix reconstruction outputs for Risanamento have:
  - `lob_event_state_panel.parquet`: non-empty and has market top-N levels.
  - `agent_event_state_panel.parquet`: non-empty but only event-agent aggregate totals, not all clients at all event times.
  - `active_order_snapshots.parquet`: empty because the fast verification run used snapshot mode `none`.
- User-provided metric choices for this exploratory phase:
  - Correct client id field: `NMSC_ORIGINALCLIENTIDSHORTCODE`.
  - Missing client id means proprietary order, to be verified against trading capacity.
  - For now compute metrics only for non-missing client ids, not brokers/firms.
  - `V_a` and `V_b`: visible quantity held by client `i` within market top-N levels, preferably as fractions of total visible market quantity in those top-N levels.
  - Distance `delta`: ticks from same-side best quote.
  - Infer tick size from the smallest price change in best bid / best ask.
  - Use `kappa = 1` for now.
  - Small bona fide execution: execution/fill of a small resting limit order from the same client, on the opposite side from candidate fake volume. For now no market-order version.
  - Compute metrics around every execution for each client, to inspect metric behavior before designing a strict detector.
  - Candidate fake-side criteria for now: same client, opposite side from the small execution.
  - `t_e^-` and `t_e^+`: fixed clock windows, default `1` second.
  - For now consider only direct cancellations of non-bona-fide orders.
- Repository instruction overrides the generic plan-skill commit guidance: do **not** commit unless the user explicitly asks.

---

## Proposed operational definitions for the first implementation

These are intentionally simple and exploratory.

### Identity

- Keep only rows with `client_original_id` / `NMSC_ORIGINALCLIENTIDSHORTCODE` non-null for metric computation.
- Do not compute firm-level spoofing metrics in this phase.
- Add a separate audit validating the proprietary-order interpretation of missing client ids:
  - for every raw row with missing `NMSC_ORIGINALCLIENTIDSHORTCODE`, require `ORDER_TRADINGCAPACITY (*) == 1`;
  - when `ORDER_TRADINGCAPACITY (*) (Tooltip)` exists, require the tooltip text to contain `Dealing_on_own_account`.

### Market top-N levels

For each state and side `s in {bid, ask}`:

- Market level price/qty comes from the current visible market book.
- Top-N levels are same as existing market ranking:
  - bid levels sorted descending by price;
  - ask levels sorted ascending by price.

### Client top-N quantities and fractions

For each client `i`, side `s`, and top-N level `l`:

```text
market_qty_{s,l}(t) = total visible market quantity at side s, level l
client_qty_{i,s,l}(t) = visible quantity of client i at the same side/price level
level_fraction_{i,s,l}(t) = client_qty_{i,s,l}(t) / market_qty_{s,l}(t)
```

Also compute side aggregates:

```text
client_qty_topN_{i,s}(t) = sum_l client_qty_{i,s,l}(t)
market_qty_topN_s(t) = sum_l market_qty_{s,l}(t)
raw_fraction_{i,s}(t) = client_qty_topN_{i,s}(t) / market_qty_topN_s(t)
```

Use zero when the market side has no top-N quantity; do not divide by zero.

### Tick size and distance

Infer one tick size per input file/instrument:

```text
tick_size = min positive difference among distinct non-null post_best_bid and post_best_ask values
```

For side-level distance:

```text
delta_bid_l = (best_bid - bid_level_price_l) / tick_size
delta_ask_l = (ask_level_price_l - best_ask) / tick_size
```

- L1 distance should be zero.
- Clamp tiny negative floating errors to zero if `abs(delta) < 1e-9`.
- If `tick_size` is absent or non-positive, fail clearly.

### Exploratory distance-weighted imbalance

Because the user wants fractions rather than raw volumes, define side weighted fractions as:

```text
w(delta) = 1 - exp(-kappa * delta)
weighted_fraction_{i,s}(t) = sum_l [client_qty_{i,s,l}(t) / market_qty_topN_s(t)] * w(delta_{s,l})
```

Then:

```text
imb_i(t) = [weighted_fraction_{i,ask}(t) - weighted_fraction_{i,bid}(t)]
           / [raw_fraction_{i,ask}(t) + raw_fraction_{i,bid}(t)]
```

If the denominator is zero, set `imb_i(t)` to null and carry explicit denominator fields. Do not impute silently.

Also output all intermediate components so we can change the formula later without recomputing the whole state machine:

- `raw_bid_fraction_topN`
- `raw_ask_fraction_topN`
- `weighted_bid_fraction_topN`
- `weighted_ask_fraction_topN`
- `client_bid_qty_topN`
- `client_ask_qty_topN`
- `market_bid_qty_topN`
- `market_ask_qty_topN`
- per-level client/market/fraction/delta columns for levels 1..N when feasible.

### Executions considered

For initial execution-level metrics, include fill events where:

- `event_class == "fill"`;
- `client_original_id` is non-null;
- event/order type is a limit/resting-visible type, not market/stop/midpoint;
- the order existed in active visible state before the fill, so this is interpreted as a passive resting limit order execution;
- event timestamp can be parsed.

Do not filter by “smallness” yet. Instead compute diagnostic smallness fields:

```text
fill_qty
same_level_market_visible_qty_pre
same_level_client_visible_qty_pre
fill_qty / same_level_market_visible_qty_pre
fill_qty / same_level_client_visible_qty_pre
```

This keeps the first run exploratory and lets us inspect the distribution before choosing a threshold.

### Execution window and SCI

Default `window_seconds = 1.0`.

For each execution event `e`:

```text
pre_target_ts = event_ts - 1 second
post_target_ts = event_ts + 1 second
```

Use piecewise-constant state lookup:

- `imb_pre_window`: latest metric state for the same client/partition at or before `pre_target_ts`.
- `imb_post_window`: latest metric state for the same client/partition at or before `post_target_ts`.
- `SCI = abs(imb_pre_window - imb_post_window)` when both are finite.

Also store immediate pre/post event-state metrics around `sort_index` if cheap, but do not use them as the primary SCI definition in this phase.

### Direct cancellation feature

For each execution event, define fake side as opposite side from the executed small limit order.

Within `(event_ts, event_ts + window_seconds]`, find direct cancel events satisfying:

- same partition;
- same client id;
- side equals fake side;
- `event_class == "cancel"`;
- the canceled order was active visible before the cancel.

Output:

- `direct_opposite_cancel_count_window`
- `direct_opposite_cancel_visible_qty_window`
- `direct_opposite_cancel_order_ids_window` as a `;`-joined debug string for now
- `has_direct_opposite_cancel_window`

Do not yet turn this into a binary spoofing label.

---

## Step-by-step plan

### Task 1: Add trading-capacity tooltip normalization

**Objective:** Preserve the trading-capacity code and tooltip label in normalized events so the client/proprietary audit can be performed reproducibly.

**Files:**
- Modify: `src/spoofing_detection/lob/normalize.py`
- Test: `tests/lob/test_spoofing_client_identity.py`

**Step 1: Write failing test**

Create `tests/lob/test_spoofing_client_identity.py` with a first test:

```python
from __future__ import annotations

from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.normalize import normalize_event


def minimal_raw_event(**overrides):
    row = {
        "TRADEDATE": "2024-01-02",
        "MIC": "XMIL",
        "MARKETCODE": "MTA",
        "SYMBOLINDEX": 123,
        "EMM (*)": 1,
        "ORDEREVENTTYPE (*)": 1,
        "ORDERID": "O1",
        "ORDERPRIORITY": "1",
        "ORDERSIDE (*)": 1,
        "ORDERPX": 100.0,
        "ORDERQTY": 10.0,
        "DISPLAYEDQTY": 10.0,
        "LEAVESQTY": 10.0,
        "ORDERTYPE (*)": 2,
        "TIMEINFORCE (*)": 0,
        "FIRMID": "F1",
        "NMSC_ORIGINALCLIENTIDSHORTCODE": None,
        "ORDER_TRADINGCAPACITY (*)": 1,
        "ORDER_TRADINGCAPACITY (*) (Tooltip)": "1 : Dealing_on_own_account",
    }
    row.update(overrides)
    return row


def test_normalize_event_preserves_order_trading_capacity_code_and_tooltip():
    event = normalize_event(minimal_raw_event(), sort_index=1, config=LOBConfig())

    assert event["order_trading_capacity_code"] == 1
    assert event["order_trading_capacity_label"] == "1 : Dealing_on_own_account"
    assert event["ORDER_TRADINGCAPACITY (*)"] == 1
    assert event["ORDER_TRADINGCAPACITY (*) (Tooltip)"] == "1 : Dealing_on_own_account"
```

**Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_client_identity.py::test_normalize_event_preserves_order_trading_capacity_code_and_tooltip
```

Expected: FAIL because the new normalized keys are absent.

**Step 3: Minimal implementation**

In `normalize_event()` add before the return dict:

```python
trading_capacity_code = normalize_enum_code(
    get_first(row, "ORDER_TRADINGCAPACITY (*)", "ORDER_TRADINGCAPACITY")
)
trading_capacity_label = get_first(row, "ORDER_TRADINGCAPACITY (*) (Tooltip)")
```

Then add to the returned dict:

```python
"order_trading_capacity_code": trading_capacity_code,
"order_trading_capacity_label": trading_capacity_label,
"ORDER_TRADINGCAPACITY (*)": trading_capacity_code,
"ORDER_TRADINGCAPACITY (*) (Tooltip)": trading_capacity_label,
```

Keep the existing `ORDER_TRADINGCAPACITY (*)` key behavior compatible; if downstream code expects the raw code, this still returns the normalized integer code.

**Step 4: Verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_client_identity.py::test_normalize_event_preserves_order_trading_capacity_code_and_tooltip
```

Expected: PASS.

---

### Task 2: Add client-id/trading-capacity audit helper

**Objective:** Verify the user’s claim that missing client ids correspond to proprietary/dealing-on-own-account orders before excluding them from client-level metrics.

**Files:**
- Create: `src/spoofing_detection/lob/client_identity_audit.py`
- Modify: `tests/lob/test_spoofing_client_identity.py`
- Create: `scripts/audit_client_identity_capacity.py`

**Step 1: Add failing tests**

Append tests:

```python
import polars as pl

from spoofing_detection.lob.client_identity_audit import audit_missing_client_trading_capacity


def test_audit_missing_client_trading_capacity_accepts_own_account_rows():
    df = pl.DataFrame(
        {
            "NMSC_ORIGINALCLIENTIDSHORTCODE": [None, "C1"],
            "ORDER_TRADINGCAPACITY (*)": [1, 3],
            "ORDER_TRADINGCAPACITY (*) (Tooltip)": ["1 : Dealing_on_own_account", "3 : Any_other_capacity"],
        }
    )

    audit = audit_missing_client_trading_capacity(df)

    assert audit["missing_client_rows"] == 1
    assert audit["missing_client_bad_capacity_rows"] == 0
    assert audit["missing_client_bad_tooltip_rows"] == 0
    assert audit["claim_holds"] is True


def test_audit_missing_client_trading_capacity_reports_bad_capacity():
    df = pl.DataFrame(
        {
            "NMSC_ORIGINALCLIENTIDSHORTCODE": [None],
            "ORDER_TRADINGCAPACITY (*)": [3],
            "ORDER_TRADINGCAPACITY (*) (Tooltip)": ["3 : Any_other_capacity"],
        }
    )

    audit = audit_missing_client_trading_capacity(df)

    assert audit["missing_client_rows"] == 1
    assert audit["missing_client_bad_capacity_rows"] == 1
    assert audit["claim_holds"] is False
```

**Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_client_identity.py
```

Expected: FAIL because the module does not exist.

**Step 3: Implement helper**

Create `src/spoofing_detection/lob/client_identity_audit.py`:

```python
from __future__ import annotations

from typing import Any

import polars as pl

CLIENT_ID_COL = "NMSC_ORIGINALCLIENTIDSHORTCODE"
CAPACITY_CODE_COL = "ORDER_TRADINGCAPACITY (*)"
CAPACITY_TOOLTIP_COL = "ORDER_TRADINGCAPACITY (*) (Tooltip)"
OWN_ACCOUNT_CODE = 1
OWN_ACCOUNT_TOKEN = "dealing_on_own_account"


def _missing_expr(column: str) -> pl.Expr:
    return (
        pl.col(column).is_null()
        | (pl.col(column).cast(pl.Utf8).str.strip_chars().str.to_lowercase().is_in(["", "nan", "none", "null"]))
    )


def audit_missing_client_trading_capacity(df: pl.DataFrame) -> dict[str, Any]:
    if CLIENT_ID_COL not in df.columns:
        raise ValueError(f"missing required column: {CLIENT_ID_COL}")
    if CAPACITY_CODE_COL not in df.columns:
        raise ValueError(f"missing required column: {CAPACITY_CODE_COL}")

    has_tooltip = CAPACITY_TOOLTIP_COL in df.columns
    missing_client = _missing_expr(CLIENT_ID_COL)
    bad_capacity = missing_client & (pl.col(CAPACITY_CODE_COL) != OWN_ACCOUNT_CODE)

    if has_tooltip:
        normalized_tooltip = pl.col(CAPACITY_TOOLTIP_COL).cast(pl.Utf8).str.to_lowercase()
        bad_tooltip = missing_client & ~normalized_tooltip.str.contains(OWN_ACCOUNT_TOKEN, literal=True)
    else:
        bad_tooltip = pl.lit(False)

    summary = df.select(
        pl.len().alias("rows"),
        missing_client.sum().alias("missing_client_rows"),
        bad_capacity.sum().alias("missing_client_bad_capacity_rows"),
        bad_tooltip.sum().alias("missing_client_bad_tooltip_rows"),
    ).to_dicts()[0]
    summary["tooltip_available"] = has_tooltip
    summary["claim_holds"] = (
        summary["missing_client_bad_capacity_rows"] == 0
        and summary["missing_client_bad_tooltip_rows"] == 0
    )
    return summary
```

**Step 4: Add CLI**

Create `scripts/audit_client_identity_capacity.py`:

```python
#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import polars as pl

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from spoofing_detection.lob.client_identity_audit import audit_missing_client_trading_capacity


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Audit missing client ids against order trading capacity.")
    parser.add_argument("inputs", nargs="+", type=Path, help="Raw parquet files to audit")
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    results = {}
    for path in args.inputs:
        df = pl.read_parquet(path)
        results[str(path)] = audit_missing_client_trading_capacity(df)
    print(json.dumps(results, indent=2, sort_keys=True))
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(results, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
```

**Step 5: Verify**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_client_identity.py
PYTHONDONTWRITEBYTECODE=1 conda run -n main python -m py_compile scripts/audit_client_identity_capacity.py src/spoofing_detection/lob/client_identity_audit.py
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/audit_client_identity_capacity.py data/*.parquet --output-json outputs/spoofing_metrics/client_identity_capacity_audit.json
```

Expected:

- tests pass;
- CLI writes JSON;
- if the claim fails for any file, stop the implementation before metric computation and inspect violations.

---

### Task 3: Add tick-size inference tests and helper

**Objective:** Infer an exploratory tick size from best bid/ask changes and fail clearly when impossible.

**Files:**
- Create: `src/spoofing_detection/lob/spoofing_metrics.py`
- Create: `tests/lob/test_spoofing_metrics.py`

**Step 1: Write failing tests**

Create `tests/lob/test_spoofing_metrics.py`:

```python
from __future__ import annotations

import math

import polars as pl
import pytest

from spoofing_detection.lob.spoofing_metrics import infer_tick_size_from_best_quotes


def test_infer_tick_size_from_best_bid_and_ask_changes():
    panel = pl.DataFrame(
        {
            "post_best_bid": [100.00, 100.01, 100.03, 100.03],
            "post_best_ask": [100.05, 100.06, 100.08, 100.09],
        }
    )

    assert infer_tick_size_from_best_quotes(panel) == pytest.approx(0.01)


def test_infer_tick_size_rejects_flat_quotes():
    panel = pl.DataFrame(
        {
            "post_best_bid": [100.00, 100.00],
            "post_best_ask": [100.05, 100.05],
        }
    )

    with pytest.raises(ValueError, match="tick size"):
        infer_tick_size_from_best_quotes(panel)
```

**Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_infer_tick_size_from_best_bid_and_ask_changes tests/lob/test_spoofing_metrics.py::test_infer_tick_size_rejects_flat_quotes
```

Expected: FAIL because module/function does not exist.

**Step 3: Implement helper**

In `src/spoofing_detection/lob/spoofing_metrics.py`:

```python
from __future__ import annotations

import math

import polars as pl

BEST_QUOTE_COLUMNS = ("post_best_bid", "post_best_ask")


def infer_tick_size_from_best_quotes(panel: pl.DataFrame) -> float:
    missing = [column for column in BEST_QUOTE_COLUMNS if column not in panel.columns]
    if missing:
        raise ValueError(f"cannot infer tick size; missing columns: {', '.join(missing)}")

    prices: list[float] = []
    for column in BEST_QUOTE_COLUMNS:
        prices.extend(
            value for value in panel.get_column(column).drop_nulls().to_list()
            if value is not None and math.isfinite(float(value))
        )
    unique = sorted(set(float(value) for value in prices))
    diffs = [b - a for a, b in zip(unique, unique[1:]) if b > a]
    positive = [diff for diff in diffs if diff > 0]
    if not positive:
        raise ValueError("cannot infer tick size from best quotes; no positive price changes")
    return min(positive)
```

**Step 4: Verify GREEN**

Run the two tests again. Expected: PASS.

---

### Task 4: Add top-N client exposure computation

**Objective:** Compute per-client top-N visible quantity, fraction, distance-weighted fractions, and exploratory imbalance from active visible orders and market top-N levels.

**Files:**
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py`
- Modify: `tests/lob/test_spoofing_metrics.py`

**Step 1: Add failing test for fractions and deltas**

Append:

```python
from spoofing_detection.lob.models import ActiveOrder
from spoofing_detection.lob.spoofing_metrics import compute_client_top_n_exposures


def order(order_id, side, price, qty, client):
    return ActiveOrder(
        order_id=order_id,
        side=side,
        price=price,
        leaves_qty=qty,
        displayed_qty=qty,
        order_qty=qty,
        order_priority=order_id,
        order_type_code=2,
        order_type_label="limit",
        time_in_force_code=0,
        firm_id="F1",
        client_original_id=client,
        first_seen_sort_index=1,
        last_update_sort_index=1,
        last_event_class="new_order",
    )


def test_compute_client_top_n_exposures_uses_client_fractions_and_tick_distances():
    active = {
        "B1": order("B1", "bid", 100.0, 10.0, "C1"),
        "B2": order("B2", "bid", 99.9, 20.0, "C1"),
        "B3": order("B3", "bid", 99.9, 30.0, "C2"),
        "A1": order("A1", "ask", 100.2, 5.0, "C1"),
        "A2": order("A2", "ask", 100.3, 15.0, "C2"),
        "P1": order("P1", "ask", 100.4, 100.0, None),
    }

    rows = compute_client_top_n_exposures(
        active,
        top_n=2,
        tick_size=0.1,
        kappa=1.0,
        partition_id="P",
        sort_index=10,
        event_ts=None,
    )
    by_client = {row["client_id"]: row for row in rows}

    c1 = by_client["C1"]
    assert c1["client_bid_qty_topN"] == 30.0
    assert c1["market_bid_qty_topN"] == 60.0
    assert c1["raw_bid_fraction_topN"] == pytest.approx(0.5)
    assert c1["client_ask_qty_topN"] == 5.0
    assert c1["market_ask_qty_topN"] == 20.0
    assert c1["raw_ask_fraction_topN"] == pytest.approx(0.25)
    assert c1["bid_level_1_delta_ticks"] == pytest.approx(0.0)
    assert c1["bid_level_2_delta_ticks"] == pytest.approx(1.0)
    assert c1["imbalance"] is not None
    assert "P1" not in by_client
```

**Step 2: Verify RED**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_compute_client_top_n_exposures_uses_client_fractions_and_tick_distances
```

Expected: FAIL because `compute_client_top_n_exposures` does not exist.

**Step 3: Implement minimal exposure helper**

Add helpers in `spoofing_metrics.py`:

```python
from collections import defaultdict
from collections.abc import Mapping
from typing import Any

from spoofing_detection.lob.models import ActiveOrder


def _visible_qty(order: ActiveOrder) -> float:
    if order.leaves_qty <= 0 or order.displayed_qty <= 0:
        return 0.0
    return float(order.displayed_qty)


def _market_levels(active_orders: Mapping[str, ActiveOrder], *, side: str, top_n: int):
    by_price: dict[float, float] = defaultdict(float)
    for order in active_orders.values():
        qty = _visible_qty(order)
        if qty > 0 and order.side == side:
            by_price[order.price] += qty
    prices = sorted(by_price, reverse=(side == "bid"))[:top_n]
    return [(price, by_price[price]) for price in prices]


def _distance_ticks(side: str, price: float, best_price: float, tick_size: float) -> float:
    raw = (best_price - price) / tick_size if side == "bid" else (price - best_price) / tick_size
    return 0.0 if abs(raw) < 1e-9 else max(raw, 0.0)


def compute_client_top_n_exposures(
    active_orders: Mapping[str, ActiveOrder],
    *,
    top_n: int,
    tick_size: float,
    kappa: float,
    partition_id: str | None,
    sort_index: int,
    event_ts,
) -> list[dict[str, Any]]:
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    if tick_size <= 0:
        raise ValueError("tick_size must be positive")

    levels = {
        "bid": _market_levels(active_orders, side="bid", top_n=top_n),
        "ask": _market_levels(active_orders, side="ask", top_n=top_n),
    }
    best = {side: side_levels[0][0] if side_levels else None for side, side_levels in levels.items()}
    market_total = {side: sum(qty for _, qty in side_levels) for side, side_levels in levels.items()}

    client_level_qty: dict[tuple[str, str, int], float] = defaultdict(float)
    active_client_ids: set[str] = set()
    price_to_rank = {
        side: {price: rank for rank, (price, _) in enumerate(side_levels, start=1)}
        for side, side_levels in levels.items()
    }
    for order in active_orders.values():
        client_id = order.client_original_id
        qty = _visible_qty(order)
        if client_id is None or qty <= 0 or order.side not in {"bid", "ask"}:
            continue
        rank = price_to_rank[order.side].get(order.price)
        if rank is None:
            continue
        active_client_ids.add(client_id)
        client_level_qty[(client_id, order.side, rank)] += qty

    rows: list[dict[str, Any]] = []
    for client_id in sorted(active_client_ids):
        row: dict[str, Any] = {
            "partition_id": partition_id,
            "sort_index": sort_index,
            "event_ts": event_ts,
            "client_id": client_id,
            "top_n": top_n,
            "tick_size": tick_size,
            "kappa": kappa,
        }
        raw_fraction: dict[str, float] = {}
        weighted_fraction: dict[str, float] = {}
        for side in ("bid", "ask"):
            client_side_qty = 0.0
            weighted_side = 0.0
            for rank, (price, market_qty) in enumerate(levels[side], start=1):
                client_qty = client_level_qty[(client_id, side, rank)]
                client_side_qty += client_qty
                delta = _distance_ticks(side, price, best[side], tick_size) if best[side] is not None else None
                weight = 1.0 - math.exp(-kappa * delta) if delta is not None else 0.0
                denominator = market_total[side]
                contribution = (client_qty / denominator) * weight if denominator > 0 else 0.0
                weighted_side += contribution
                row[f"{side}_level_{rank}_price"] = price
                row[f"{side}_level_{rank}_market_visible_qty"] = market_qty
                row[f"{side}_level_{rank}_client_visible_qty"] = client_qty
                row[f"{side}_level_{rank}_client_fraction"] = client_qty / market_qty if market_qty > 0 else 0.0
                row[f"{side}_level_{rank}_delta_ticks"] = delta
            raw = client_side_qty / market_total[side] if market_total[side] > 0 else 0.0
            raw_fraction[side] = raw
            weighted_fraction[side] = weighted_side
            row[f"client_{side}_qty_topN"] = client_side_qty
            row[f"market_{side}_qty_topN"] = market_total[side]
            row[f"raw_{side}_fraction_topN"] = raw
            row[f"weighted_{side}_fraction_topN"] = weighted_side
        denom = raw_fraction["ask"] + raw_fraction["bid"]
        row["imbalance_denominator"] = denom
        row["imbalance"] = (
            (weighted_fraction["ask"] - weighted_fraction["bid"]) / denom
            if denom > 0
            else None
        )
        rows.append(row)
    return rows
```

**Step 4: Verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_compute_client_top_n_exposures_uses_client_fractions_and_tick_distances
```

Expected: PASS.

---

### Task 5: Add timestamp parsing helper

**Objective:** Produce a single event timestamp column suitable for one-second clock windows.

**Files:**
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py`
- Modify: `tests/lob/test_spoofing_metrics.py`

**Step 1: Add failing tests**

Add:

```python
from datetime import datetime

from spoofing_detection.lob.spoofing_metrics import choose_event_timestamp


def test_choose_event_timestamp_prefers_trade_time_then_book_fields():
    event = {
        "TRADETIME": datetime(2024, 1, 2, 9, 30, 1),
        "BOOKOUTTIME": "2024-01-02 09:30:02",
        "BOOKIN": "2024-01-02 09:30:03",
        "SEQUENCETIME": "2024-01-02 09:30:04",
    }

    assert choose_event_timestamp(event) == datetime(2024, 1, 2, 9, 30, 1)


def test_choose_event_timestamp_falls_back_to_bookout():
    event = {
        "TRADETIME": None,
        "BOOKOUTTIME": "2024-01-02 09:30:02",
        "BOOKIN": "2024-01-02 09:30:03",
        "SEQUENCETIME": "2024-01-02 09:30:04",
    }

    assert choose_event_timestamp(event) == datetime(2024, 1, 2, 9, 30, 2)
```

**Step 2: Verify RED**

Run the new tests. Expected: FAIL.

**Step 3: Implement helper**

Use only standard library parsing first; keep it permissive:

```python
from datetime import datetime


def _parse_ts(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    # Pandas-like strings often work with fromisoformat after replacing Z.
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def choose_event_timestamp(event: dict[str, Any]) -> datetime | None:
    for key in ("TRADETIME", "BOOKOUTTIME", "BOOKIN", "SEQUENCETIME"):
        parsed = _parse_ts(event.get(key))
        if parsed is not None:
            return parsed
    return None
```

**Step 4: Verify GREEN**

Run all `tests/lob/test_spoofing_metrics.py` so far.

---

### Task 6: Stream client metric time series from raw events

**Objective:** Build a first exploratory time series of client top-N imbalance states without materializing every active-order snapshot.

**Files:**
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py`
- Modify: `tests/lob/test_spoofing_metrics.py`

**Implementation note:** For this exploratory phase, it is acceptable to reuse internal reconstruction helpers from `panel.py` (`sort_events`, `_apply_event`, `_partition_id`) to avoid duplicating active-order state logic. If this becomes stable, later refactor those helpers into public APIs.

**Step 1: Add failing synthetic stream test**

Add a small raw-event fixture similar to existing LOB tests:

```python
from spoofing_detection.lob.spoofing_metrics import compute_client_metric_time_series


def raw_event(seq, event_type, order_id, side, price, qty, displayed, client, *, trade_time=None):
    return {
        "TRADEDATE": "2024-01-02",
        "MIC": "XMIL",
        "MARKETCODE": "MTA",
        "SYMBOLINDEX": 123,
        "EMM (*)": 1,
        "SEQUENCETIME": f"2024-01-02 09:30:{seq:02d}",
        "BOOKOUTTIME": f"2024-01-02 09:30:{seq:02d}",
        "TRADETIME": trade_time,
        "HDR_APPLKEYSEQUENCENUMBER": seq,
        "HDR_HWMSEQUENCENUMBER": seq,
        "HDR_OFFSETID": seq,
        "ROW_NUMBER": seq,
        "ORDEREVENTTYPE (*)": event_type,
        "ORDERID": order_id,
        "ORDERPRIORITY": str(seq),
        "ORDERSIDE (*)": side,
        "ORDERPX": price,
        "ORDERQTY": qty,
        "DISPLAYEDQTY": displayed,
        "LEAVESQTY": qty,
        "ORDERTYPE (*)": 2,
        "TIMEINFORCE (*)": 0,
        "FIRMID": "F1",
        "NMSC_ORIGINALCLIENTIDSHORTCODE": client,
        "ORDER_TRADINGCAPACITY (*)": 3,
    }


def test_compute_client_metric_time_series_emits_client_only_top_n_states():
    df = pl.DataFrame(
        [
            raw_event(1, 1, "B1", 1, 100.0, 10, 10, "C1"),
            raw_event(2, 1, "A1", 2, 101.0, 10, 10, "C2"),
            raw_event(3, 1, "B2", 1, 99.9, 20, 20, "C1"),
            raw_event(4, 1, "A2", 2, 101.1, 20, 20, None),
        ]
    )

    states = compute_client_metric_time_series(df, top_n=2, tick_size=0.1, kappa=1.0)

    assert set(states["client_id"].drop_nulls().to_list()) == {"C1", "C2"}
    c1_latest = states.filter(pl.col("client_id") == "C1").tail(1).to_dicts()[0]
    assert c1_latest["client_bid_qty_topN"] == 30.0
    assert c1_latest["client_ask_qty_topN"] == 0.0
```

**Step 2: Verify RED**

Run the test. Expected: FAIL.

**Step 3: Implement streaming function**

Sketch:

```python
from spoofing_detection.lob.config import LOBConfig
from spoofing_detection.lob.normalize import normalize_event
from spoofing_detection.lob.panel import _apply_event, _partition_id, sort_events


def compute_client_metric_time_series(raw_events: pl.DataFrame, *, top_n: int, tick_size: float, kappa: float) -> pl.DataFrame:
    config = LOBConfig(top_n=max(top_n, 1), snapshot_mode="none")
    sorted_events = sort_events(raw_events).with_row_index("__row_nr")
    active_orders: dict[str, ActiveOrder] = {}
    pending_aggressive_residuals = {}
    non_resting_order_ids: set[str] = set()
    rows: list[dict[str, Any]] = []

    for sort_index, row in enumerate(sorted_events.iter_rows(named=True), start=1):
        event = normalize_event(row, sort_index=sort_index, config=config)
        event_ts = choose_event_timestamp(event)
        _apply_event(
            active_orders,
            event,
            pending_aggressive_residuals=pending_aggressive_residuals,
            non_resting_order_ids=non_resting_order_ids,
        )
        rows.extend(
            compute_client_top_n_exposures(
                active_orders,
                top_n=top_n,
                tick_size=tick_size,
                kappa=kappa,
                partition_id=_partition_id(event),
                sort_index=sort_index,
                event_ts=event_ts,
            )
        )
    return pl.DataFrame(rows) if rows else pl.DataFrame()
```

**Step 4: Verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_compute_client_metric_time_series_emits_client_only_top_n_states
```

Expected: PASS.

---

### Task 7: Extract passive limit execution events and direct cancel windows

**Objective:** Produce one row per eligible client limit execution with fake-side direct-cancellation diagnostics.

**Files:**
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py`
- Modify: `tests/lob/test_spoofing_metrics.py`

**Step 1: Add failing test**

Synthetic sequence:

1. Client C1 posts large bid at L2.
2. Client C1 posts small ask at L1.
3. The small ask receives a fill.
4. C1 directly cancels the large bid within one second.

Expected execution row:

- `client_id == "C1"`
- `execution_side == "ask"`
- `fake_side == "bid"`
- `has_direct_opposite_cancel_window is True`
- `direct_opposite_cancel_count_window == 1`

Test name:

```python
def test_execution_metrics_detect_same_client_opposite_side_direct_cancel_window():
    ...
```

**Step 2: Verify RED**

Run only that test. Expected: FAIL.

**Step 3: Implement functions**

Add:

```python
@dataclass(frozen=True)
class ExploratoryMetricsConfig:
    top_n: int = 3
    kappa: float = 1.0
    window_seconds: float = 1.0
```

Add a streaming collector that returns:

```python
@dataclass
class ExploratoryMetricsResult:
    state_time_series: pl.DataFrame
    execution_metrics: pl.DataFrame
    direct_cancellations: pl.DataFrame
    rejected_executions: pl.DataFrame
```

Simplest algorithm:

- During the same active-order stream:
  - before applying each event, if event is a fill and qualifies as passive client limit execution, record execution candidate using pre-state fields;
  - if event is a cancel and active order exists before applying event, record direct cancellation candidate with pre-cancel visible quantity and client id;
  - after applying event, emit state exposure rows.
- After stream, call `attach_window_metrics(...)`:
  - use Polars joins/filtering to attach direct cancellations in `(event_ts, event_ts + window]` for same partition/client/fake_side;
  - use `join_asof` on state time series to attach `imb_pre_window` and `imb_post_window`.

**Step 4: Verify GREEN**

Run:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_metrics.py::test_execution_metrics_detect_same_client_opposite_side_direct_cancel_window
```

Expected: PASS.

---

### Task 8: Add SCI window lookup tests

**Objective:** Verify clock-time one-second SCI computation independent of candidate-cancel detection.

**Files:**
- Modify: `tests/lob/test_spoofing_metrics.py`
- Modify: `src/spoofing_detection/lob/spoofing_metrics.py`

**Step 1: Add failing test**

Create simple state rows with timestamps and one execution at `09:30:10`:

- C1 imbalance at `09:30:09` = `-0.8`
- C1 imbalance at `09:30:11` = `-0.1`
- SCI should be `0.7`.

Test helper:

```python
from spoofing_detection.lob.spoofing_metrics import attach_sci_window_metrics


def test_attach_sci_window_metrics_uses_fixed_clock_window():
    states = pl.DataFrame(...)
    executions = pl.DataFrame(...)
    out = attach_sci_window_metrics(executions, states, window_seconds=1.0)
    assert out.item(0, "SCI") == pytest.approx(0.7)
```

**Step 2: Verify RED**

Run test. Expected: FAIL.

**Step 3: Implement `attach_sci_window_metrics`**

Use `join_asof` sorted by `partition_id`, `client_id`, timestamp:

- Add `pre_target_ts` and `post_target_ts` columns to executions.
- Asof join state rows at or before target times.
- Compute `SCI = abs(imb_post_window - imb_pre_window)`.

Careful Polars requirement: both frames sorted by join keys and timestamp.

**Step 4: Verify GREEN**

Run metric tests. Expected: PASS.

---

### Task 9: Add compute CLI and metadata/report output

**Objective:** Run the exploratory metric pipeline on one raw parquet file and save traceable outputs.

**Files:**
- Create: `scripts/compute_spoofing_metrics.py`
- Modify: `tests/lob/test_spoofing_metrics.py` only if parsing helpers need adjustment.

**CLI contract:**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/compute_spoofing_metrics.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --quote-panel outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG/lob_event_state_panel.parquet \
  --output-dir outputs/spoofing_metrics/risanamento_exploratory_top3_k1_window1s \
  --top-n 3 \
  --kappa 1.0 \
  --window-seconds 1.0
```

**Outputs:**

- `client_metric_time_series.parquet`
- `execution_metrics.parquet`
- `direct_cancellations.parquet`
- `rejected_executions.parquet`
- `metadata.json`
- `summary_report.md`

**Metadata fields:**

```json
{
  "input": "...",
  "quote_panel": "...",
  "top_n": 3,
  "kappa": 1.0,
  "window_seconds": 1.0,
  "tick_size": 0.0001,
  "identity": "NMSC_ORIGINALCLIENTIDSHORTCODE",
  "client_only": true,
  "market_orders_included": false,
  "direct_cancellations_only": true,
  "row_counts": {
    "state_time_series": 0,
    "execution_metrics": 0,
    "direct_cancellations": 0,
    "rejected_executions": 0
  }
}
```

**Summary report should include:**

- client identity/proprietary audit result path/status;
- tick size inferred;
- number of clients with top-N exposure;
- number of eligible limit executions;
- number/fraction with direct opposite-side cancellation within window;
- SCI finite coverage;
- top 20 executions by SCI with fields:
  - `sort_index`
  - `event_ts`
  - `client_id`
  - `execution_side`
  - `fake_side`
  - `fill_qty`
  - `smallness_fraction_market_level`
  - `SCI`
  - `has_direct_opposite_cancel_window`
  - `direct_opposite_cancel_visible_qty_window`

**Validation:**

Run first on a small row cap if implemented:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/compute_spoofing_metrics.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --quote-panel outputs/lob_reconstruction/20260618_105229_all_files_validation_post_warning_fix/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG/lob_event_state_panel.parquet \
  --output-dir outputs/spoofing_metrics/smoke_risanamento_top3_k1_window1s \
  --top-n 3 \
  --kappa 1.0 \
  --window-seconds 1.0 \
  --max-rows 5000
```

Then full Risanamento if smoke passes.

---

### Task 10: Add exploratory HTML dashboard

**Objective:** Produce a small visual diagnostic dashboard for the execution-level metrics so we can inspect behavior before designing thresholds.

**Files:**
- Create: `scripts/plot_spoofing_metrics.py`
- Create or extend: `src/spoofing_detection/lob/spoofing_metric_plots.py`
- Test: `tests/lob/test_spoofing_metric_plots.py`

**Dashboard inputs:**

- `execution_metrics.parquet`
- optional `client_metric_time_series.parquet`

**Initial dashboard layout:**

1. Scatter over event time:
   - x: execution `event_ts` or `sort_index`
   - y: `SCI`
   - color: `has_direct_opposite_cancel_window`
   - hover: client id, side, fill qty, smallness fraction, direct cancel qty.
2. Histogram/distribution of `SCI`.
3. Top clients summary:
   - number of executions;
   - finite SCI count;
   - mean/median/max SCI;
   - share with direct opposite-side cancel.
4. Optional selected-client time series:
   - `imbalance` over event time for `--client-id`.

**Test first:**

Create a tiny execution metrics dataframe and assert the HTML contains:

- title;
- `SCI`;
- `direct opposite-side cancel`;
- `Top clients`.

**Run example:**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/plot_spoofing_metrics.py \
  --execution-metrics outputs/spoofing_metrics/risanamento_exploratory_top3_k1_window1s/execution_metrics.parquet \
  --state-time-series outputs/spoofing_metrics/risanamento_exploratory_top3_k1_window1s/client_metric_time_series.parquet \
  --output-html outputs/spoofing_metrics/risanamento_exploratory_top3_k1_window1s/spoofing_metric_dashboard.html \
  --title "Risanamento exploratory client spoofing metrics — top 3, kappa 1, 1s window"
```

---

### Task 11: End-to-end verification

**Objective:** Verify all code and generated exploratory artifacts before reporting any metric values.

**Commands:**

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q tests/lob/test_spoofing_client_identity.py tests/lob/test_spoofing_metrics.py tests/lob/test_spoofing_metric_plots.py
PYTHONDONTWRITEBYTECODE=1 conda run -n main pytest -q
PYTHONDONTWRITEBYTECODE=1 conda run -n main python -m py_compile \
  scripts/audit_client_identity_capacity.py \
  scripts/compute_spoofing_metrics.py \
  scripts/plot_spoofing_metrics.py \
  src/spoofing_detection/lob/client_identity_audit.py \
  src/spoofing_detection/lob/spoofing_metrics.py \
  src/spoofing_detection/lob/spoofing_metric_plots.py
```

**Artifact checks:**

Use a read-only Python one-liner after generation:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python -c "from pathlib import Path; import polars as pl, json; root=Path('outputs/spoofing_metrics/risanamento_exploratory_top3_k1_window1s'); print(json.loads((root/'metadata.json').read_text())); print(pl.read_parquet(root/'execution_metrics.parquet').shape); print(pl.read_parquet(root/'client_metric_time_series.parquet').shape)"
```

Expected:

- no rows in metric outputs with null `client_id`;
- no market-order executions included;
- finite tick size;
- finite SCI coverage reported explicitly;
- output HTML exists and is non-empty;
- no claim that high SCI proves spoofing.

---

## Files likely to change

Create:

- `src/spoofing_detection/lob/client_identity_audit.py`
- `src/spoofing_detection/lob/spoofing_metrics.py`
- `src/spoofing_detection/lob/spoofing_metric_plots.py`
- `scripts/audit_client_identity_capacity.py`
- `scripts/compute_spoofing_metrics.py`
- `scripts/plot_spoofing_metrics.py`
- `tests/lob/test_spoofing_client_identity.py`
- `tests/lob/test_spoofing_metrics.py`
- `tests/lob/test_spoofing_metric_plots.py`

Modify:

- `src/spoofing_detection/lob/normalize.py`

Generated outputs:

- `outputs/spoofing_metrics/client_identity_capacity_audit.json`
- `outputs/spoofing_metrics/smoke_risanamento_top3_k1_window1s/*`
- `outputs/spoofing_metrics/risanamento_exploratory_top3_k1_window1s/*`

Do not edit `paper/spoofing.tex` in this implementation pass unless the user asks; this phase is empirical plumbing and diagnostics.

---

## Risks, tradeoffs, and open questions

1. **Tooltip missing for Nexi.** The audit must validate numeric trading-capacity code for all files and tooltip text only where the tooltip column exists.
2. **Timestamp parsing.** The raw files have multiple candidate time columns (`TRADETIME`, `BOOKOUTTIME`, `BOOKIN`, `SEQUENCETIME`). The first version should report timestamp coverage and rejected rows. If parsing coverage is low, stop before SCI interpretation.
3. **Tick inference.** Smallest best-quote difference is simple but can be sensitive to floating representation and tick-regime changes. This is acceptable for exploration, but metadata must record the inferred tick size.
4. **Use of private reconstruction helpers.** Reusing `_apply_event` is pragmatic for a first exploratory module, but if metrics become central, refactor state application into a public API.
5. **Market best quote changes by other agents.** The streaming state computes exposures after every market event for active top-N clients, which is scientifically safer than only updating a client on that client’s own events.
6. **Metric formula is exploratory.** The fraction-based version of `V_a`, `V_b` is a modeling choice that differs from the manuscript’s raw-volume equation. Keep raw quantities and fractions in output so we can compare alternatives later.
7. **L1 weight is zero with `1-exp(-kappa*delta)`.** This follows the current formula but means top-of-book quantity affects the denominator, not the weighted numerator. Inspect whether this behavior is desirable.
8. **Smallness is not a filter yet.** First output should include smallness fractions and let us inspect distributions before thresholding.
9. **Direct cancellations only.** This intentionally excludes modify-to-zero, move-away, and other risk-termination mechanisms. The report must say so.
10. **No labels.** The output is a surveillance/exploration score, not a proof of spoofing or intent.

---

## Recommended first execution target

Use Risanamento first because we already generated and inspected quote diagnostics for it:

```text
data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet
```

Run with:

```text
top_n = 3
kappa = 1.0
window_seconds = 1.0
identity = client_original only
market_orders = excluded
candidate cancellation = direct cancel only
```

Only after inspecting the Risanamento dashboard should we scale to Ferrari and Nexi.
