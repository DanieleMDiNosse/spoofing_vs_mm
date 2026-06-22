# Crossed/Locked LOB Diagnostic

This diagnostic investigates why the current first-version LOB reconstruction produces many locked/crossed book states on the full RISANAMENTO parquet run.

It is a debugging report, not a scientific market claim.

## Inputs inspected

- Full reconstruction output:
  - `outputs/lob_reconstruction/20260617_101549_full_risanamento/lob_event_state_panel.parquet`
  - `outputs/lob_reconstruction/20260617_101549_full_risanamento/normalized_events.parquet`
- Raw input:
  - `data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet`
- Enriched diagnostic output:
  - `outputs/lob_reconstruction/20260617_101549_full_risanamento_crossed_diagnostic/crossed_diagnostic.md`
  - `outputs/lob_reconstruction/20260617_101549_full_risanamento_crossed_diagnostic/crossed_summary.json`
  - `outputs/lob_reconstruction/20260617_101549_full_risanamento_crossed_diagnostic/enriched_panel_context.parquet`

The generated `outputs/` paths are local, git-ignored artifacts.

## Symptom

The reconstructed book often has:

```text
best bid >= best ask
```

Counts in the current full RISANAMENTO reconstruction:

| Diagnostic | Count |
|---|---:|
| Rows | 143,018 |
| Post-event crossed rows, `best_bid > best_ask` | 52,429 |
| Post-event locked rows, `best_bid == best_ask` | 9,571 |
| Newly crossed rows | 1,048 |
| Newly locked rows | 4,427 |
| Newly crossed-or-locked from a previously clean book | 4,975 |
| Crossed episodes | 1,048 |

This does not mean the market was truly crossed that often. It means the reconstruction semantics are still too naive.

## Root-cause summary

The main problem is that the current state machine treats too many event rows as if they immediately become resting visible book liquidity.

In reality, the feed contains multi-row order lifecycles. A marketable incoming order can appear first as a `New` row with price and positive quantity, then immediately appear as one or more `Fill` rows. The current code inserts the `New` row into the resting active book before processing the fills. This creates artificial locked/crossed states.

There is a second related problem: stop-limit orders are currently treated as visible resting orders, but stop-limit orders should generally be conditional until triggered. Including them as visible depth can produce long artificial crossed episodes.

## Evidence 1: most clean-to-bad transitions are New rows that immediately fill

Among the 4,975 rows that turn a clean book into a locked/crossed book:

| Event class | Count |
|---|---:|
| `new_order` | 4,432 |
| `move_dark_to_cob` | 295 |
| `modify_order` | 203 |
| `session_reload` | 45 |

For the 4,432 `new_order` transitions:

| Follow-up condition | Count |
|---|---:|
| Same `ORDERID` has a fill within 2 rows | 4,259 |
| Same `ORDERID` has a fill within 20 rows | 4,299 |
| Same `ORDERID` has a fill within 100 rows | 4,340 |

For the 4,259 same-order fills within 2 rows:

| Fill flags | Count |
|---|---:|
| `PASSIVEORDER=N`, `AGGRESSIVEORDER=Y` | 4,248 |
| Other | 11 |

Also, 4,248 of the 4,259 immediate fills have the same `SEQUENCETIME` as the preceding `New` row.

Interpretation:

- The problematic `New` row is usually not passive resting liquidity.
- It is usually the accepted incoming side of an aggressive order that immediately trades.
- The passive/aggressive flags appear on the subsequent fill rows, not on the preceding `New` row.
- Therefore checking only `PASSIVEORDER` / `AGGRESSIVEORDER` on the `New` row is not enough.

Concrete example from the first locked transition:

| sort_index | event | order | side | price | leaves | passive | aggressive | best bid after | best ask after |
|---:|---|---|---|---:|---:|---|---|---:|---:|
| 161 | `New` | `939609509` | bid | 0.0291 | 20,000 | N | N | 0.0291 | 0.0291 |
| 162 | `Fill` | `939609509` | bid | 0.0291 | 6,000 | N | Y | 0.0291 | 0.0291 |
| 163 | `Fill` | `872500645` | ask | 0.0291 | 0 | Y | N | 0.0291 | 0.0292 |
| 165 | `Cancel` | `939609509` | bid | 0.0291 | 0 | N | N | 0.0290 | 0.0292 |

The economic trade is not a stable locked book. It is an aggressive bid consuming a passive ask. The reconstruction temporarily inserted the incoming bid as resting liquidity before the passive ask was removed.

## Evidence 2: fill-pair ordering creates transient artificial states

For the same execution in the example above:

- aggressive fill row comes first;
- passive fill row comes next;
- both rows share the same `EXECUTIONID = 257` and `TRADEUNIQUEIDENTIFIER = 1UHCMGOUT`;
- the aggressive row and passive row have the same trade price and trade time.

Current row-by-row processing updates the aggressive side before updating/removing the passive side. That can create temporary locked/crossed states inside what should probably be treated as one atomic execution group.

This suggests a future fix should consider execution grouping by fields such as:

- `EXECUTIONID`
- `TRADEUNIQUEIDENTIFIER`
- `TRADETIME`
- `SEQUENCETIME`
- passive/aggressive flags

But this should be designed carefully and tested with synthetic paired-fill cases before changing production logic.

## Evidence 3: stop-limit orders create long artificial crossed episodes

Some of the longest crossed episodes start when stop-limit orders are inserted as if they were visible resting orders.

Example around sort index 6,875:

| sort_index | event | order type | side | price | pre best bid | pre best ask | post best bid | post best ask |
|---:|---|---|---|---:|---:|---:|---:|---:|
| 6,875 | `New` | `stop_limit_or_stop_limit_on_quote` | ask | 0.0312 | 0.0329 | 0.0330 | 0.0329 | 0.0312 |
| 6,877 | `New` | `stop_limit_or_stop_limit_on_quote` | ask | 0.0305 | 0.0329 | 0.0312 | 0.0329 | 0.0305 |
| 6,878 | `New` | `stop_limit_or_stop_limit_on_quote` | ask | 0.0299 | 0.0329 | 0.0305 | 0.0329 | 0.0299 |

Those stop-limit rows appear to be conditional orders, not normal visible resting ask depth. Treating them as visible produces a crossed book for many rows.

What-if test:

| Reconstruction variant | Crossed rows | Locked rows |
|---|---:|---:|
| Current baseline | 52,429 | 9,571 |
| Exclude stop-limit orders from visible resting depth | 12,034 | 14,873 |
| Diagnostic skip of all marketable/crossing visible updates | 0 | 0 |

The stop-limit exclusion is not a complete fix, but it reduces crossed rows by about 77% in this file. That is strong evidence that stop-limit handling is one major source of persistent crossed states.

The diagnostic skip of all marketable/crossing updates is intentionally too crude; it proves the mechanism but is not a production fix because it creates many unseen-order side effects and would likely understate residual resting liquidity.

## Hypotheses assessed

### Side mapping is reversed

Unlikely.

Evidence against this hypothesis:

- Paired fill rows show the expected passive/aggressive opposite sides.
- Example: aggressive bid fill and passive ask fill share the same execution and price.
- Reversing side mapping would not explain the immediate New -> aggressive Fill lifecycle pattern.

### Unknown enum mappings cause the issue

Unlikely for the audited files.

Evidence:

- Schema/enum audit found no unknown enum codes under the accepted provisional mappings across 868,149 rows.

### The data are simply crossed all day

Not a safe conclusion.

Some auction/order-collection states may legitimately allow crossing interest, but the first locked/crossed examples in the continuous-looking stream are clearly caused by row-level processing of aggressive order lifecycles. Therefore the diagnostic evidence points to reconstruction semantics first, not market reality.

### Main cause: row-level mutation is too naive

Supported.

Evidence:

- Clean-to-bad transitions are mostly `New` rows.
- Most of those `New` rows are immediately followed by same-order aggressive fills.
- Fill-pair ordering updates aggressive side before passive side.
- Stop-limit rows are treated as visible when they should probably be conditional.

## What this means for the implementation

The current first-version reconstruction is useful as an audit skeleton, but it should not yet be used for spoofing features.

The state machine needs to distinguish:

1. passive visible resting order additions;
2. marketable incoming order acknowledgements;
3. aggressive fill rows;
4. passive fill rows;
5. residual quantity after an aggressive order finishes matching;
6. conditional stop/stop-limit orders before trigger;
7. dark-to-lit transformations.

The current code mostly sees only:

```text
price + side + positive LEAVESQTY + positive DISPLAYEDQTY => active visible order
```

That rule is too broad.

## Recommended next fix direction

Do not jump directly to SCI/CPS features.

Next implementation step should be TDD around marketable order lifecycle semantics:

1. Add synthetic tests for a marketable limit order:
   - resting ask exists at 10.00;
   - incoming bid limit at 10.00 or 10.05 arrives;
   - New row appears before Fill rows;
   - aggressive fill row and passive fill row share execution fields;
   - reconstructed book should not show stable locked/crossed depth between these rows.

2. Add synthetic tests for stop-limit orders:
   - stop-limit order should not enter visible resting depth on `New` / `Reload` unless a documented trigger/transformation row makes it visible.

3. Add tests for residual aggressive orders:
   - if an aggressive limit order partially fills and has residual quantity, add only the final residual to the resting book after the relevant passive fills have been applied.

4. Add a diagnostic execution-group stage:
   - group fill rows by `EXECUTIONID`, `TRADEUNIQUEIDENTIFIER`, `TRADETIME`, and possibly `SEQUENCETIME`;
   - process passive-side removals/reductions and aggressive residuals in an atomic order.

5. Rerun the full RISANAMENTO reconstruction and require locked/crossed counts to be explained by documented auction/phase states, not by row-order artifacts.

## Implemented fix checkpoint

The working-tree implementation now addresses the diagnosed artifact by:

1. excluding stop-limit orders from visible resting depth before trigger;
2. skipping marketable New/Reload/Modify rows instead of inserting them into active depth;
3. deferring positive-quantity fill residuals when they are aggressive or still marketable against the current opposite side;
4. flushing deferred residuals only when they can rest without locking/crossing the visible book.

The full RISANAMENTO rerun at
`outputs/lob_reconstruction/20260617_124135_full_risanamento_lifecycle_fix_v2/`
produced zero locked/crossed pre/post rows on 143,018 events. The fix remains conservative: it prioritizes a valid visible continuous-book state over reconstructing exact queue position for every marketable lifecycle.

## Current conclusion

The problem is not simply “aggressive rows have `AGGRESSIVEORDER=Y` and should be skipped.”

The more precise problem is:

```text
The feed represents one economic event with several rows. The current reconstruction mutates the book after each individual row. That inserts marketable incoming orders and conditional stop-limit orders into visible resting depth too early or when they should not be visible at all.
```

That is why the reconstructed book becomes locked/crossed.
