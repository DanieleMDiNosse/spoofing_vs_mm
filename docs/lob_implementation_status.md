# LOB Reconstruction Implementation Status

This document records the implementation status after the schema/enum audit and the first full-file reconstruction run. It is an engineering validation checkpoint, not a scientific spoofing result.

## Current artifacts

Code and docs:

- Plan: `docs/lob_reconstruction_plan.md`
- Audit module: `src/spoofing_detection/lob/audit.py`
- Audit CLI: `scripts/audit_lob_schema.py`
- Reconstruction CLI: `scripts/reconstruct_lob.py`
- Main reconstruction implementation: `src/spoofing_detection/lob/panel.py`
- Normalization and enum mapping: `src/spoofing_detection/lob/normalize.py`, `src/spoofing_detection/lob/enums.py`

Generated validation artifacts:

- All-file schema/enum audit:
  - `outputs/lob_reconstruction/20260617_101511_schema_audit/schema_audit.md`
  - `outputs/lob_reconstruction/20260617_101511_schema_audit/schema_audit.json`
- Full reconstruction of the RISANAMENTO parquet:
  - `outputs/lob_reconstruction/20260617_101549_full_risanamento/validation_report.md`
  - `outputs/lob_reconstruction/20260617_101549_full_risanamento/metadata.json`
  - `outputs/lob_reconstruction/20260617_101549_full_risanamento/lob_event_state_panel.parquet`
  - `outputs/lob_reconstruction/20260617_101549_full_risanamento/normalized_events.parquet`
  - `outputs/lob_reconstruction/20260617_101549_full_risanamento/agent_event_state_panel.parquet`
  - `outputs/lob_reconstruction/20260617_101549_full_risanamento/active_order_snapshots.parquet`
  - `outputs/lob_reconstruction/20260617_101549_full_risanamento/price_level_depth_snapshots.parquet`

The `outputs/` directory is ignored by git, so the report paths above are local run artifacts.

## User-confirmed domain semantics now encoded in the plan

- `PASSIVEORDER`: passive liquidity, i.e. a limit order that does not cross the spread and is not filled immediately.
- `AGGRESSIVEORDER`: a market order or a limit order that crosses the spread and consumes resting liquidity.
- The current data do not contain trade-bust events. Trade-bust rollback is therefore not a current implementation target; if future inputs contain trade-bust-like records, they should be flagged and kept non-mutating until documented.

## Schema/enum audit findings

Command run:

```text
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/audit_lob_schema.py data --output-dir outputs/lob_reconstruction/20260617_101511_schema_audit
```

Observed audit facts:

- Files audited: 3 parquet files.
- Total rows audited: 868,149.
- Required reconstruction columns present in all audited files: yes.
- Full ordering-key duplicate count: 0.
- Unknown enum codes under the accepted provisional mappings: none.
- `FIRMID` missing count: 0.
- `NMSC_ORIGINALCLIENTIDSHORTCODE` missing count: 634,273.
- `PASSIVEORDER` observed values: `N`, `Y`.
- `AGGRESSIVEORDER` observed values: `N`, `Y`.
- First event code by partition: all 132 audited partitions start with `11 : GTC_GTD_Reload`.
- First event code by partition/order:
  - `1 : New`: 385,436 orders
  - `11 : GTC_GTD_Reload`: 8,588 orders

Interpretation:

- The available files are structurally suitable for the current reconstruction implementation.
- The reload-as-opening-seed assumption is supported at partition starts in the audited data.
- Client-original identity is sparse; client-level panels must keep missingness flags and should not drop rows with missing `NMSC_ORIGINALCLIENTIDSHORTCODE`.
- The provisional enum mapping is sufficient for these three files, but official dictionaries would still be preferable if available.

## First full-file reconstruction findings

Command run:

```text
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/reconstruct_lob.py \
  data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  outputs/lob_reconstruction/20260617_101549_full_risanamento \
  --snapshot-mode end_of_partition
```

Observed run facts:

- Input rows: 143,018.
- Normalized rows: 143,018.
- LOB event-state panel rows: 143,018.
- Agent event-state rows: 286,036, i.e. two rows per event for `firm` and `client_original`.
- Partitions processed: 130.
- Active orders at partition ends: 7,342.
- Active-order snapshot rows: 7,342.
- Price-level depth snapshot rows: 5,519.
- Top-N: 10.
- Agent dimensions: `firm`, `client_original`.

Event class counts in the full RISANAMENTO run:

| Event class | Rows |
|---|---:|
| `new_order` | 52,849 |
| `cancel` | 35,075 |
| `fill` | 34,338 |
| `modify_order` | 12,202 |
| `session_reload` | 7,390 |
| `move_dark_to_cob` | 509 |
| `special_validity_event` | 402 |
| `iceberg_refill` | 209 |
| `trigger` | 44 |

Issue counts in the original full-file validation report:

| Issue | Rows |
|---|---:|
| `missing_client_original_id` | 31,762 |
| `non_resting_unpriced_event` | 8,805 |
| `missing_price_for_potential_resting_event` | 53 |
| `full_fill_for_unseen_order` | 2 |

Book-state diagnostic flags in the original full run:

| Flag | Rows |
|---|---:|
| `book_crossed_pre_flag` | 52,384 |
| `book_crossed_post_flag` | 52,429 |
| `book_locked_pre_flag` | 9,571 |
| `book_locked_post_flag` | 9,571 |

The two `full_fill_for_unseen_order` rows are full-fill iceberg rows for `SYMBOLINDEX = 3308928` on `2024-06-12`, with `ORDERID` values `27632160174` and `29376990638`.

After the marketable-order lifecycle fix, the full RISANAMENTO rerun at
`outputs/lob_reconstruction/20260617_124135_full_risanamento_lifecycle_fix_v2/`
produced:

| Flag | Rows |
|---|---:|
| `book_crossed_pre_flag` | 0 |
| `book_crossed_post_flag` | 0 |
| `book_locked_pre_flag` | 0 |
| `book_locked_post_flag` | 0 |

Its validation issue counts were:

| Issue | Rows |
|---|---:|
| `marketable_order_not_resting` | 8,715 |
| `missing_client_original_id` | 31,762 |
| `non_resting_unpriced_event` | 8,805 |
| `missing_price_for_potential_resting_event` | 53 |
| `modify_for_unseen_order` | 29 |

## Plan-stage status

| Plan stage | Status | Notes |
|---|---|---|
| Stage 0: approval and decisions | Done | Top-N 10, all events, as-is parquet values, separate firm/client dimensions, reload seed, enum mapping accepted. Passive/aggressive semantics now clarified by user. |
| Stage 1: repository/package setup | Mostly done | Package exists under `src/spoofing_detection/lob`. There is no separate `schema.py`; required schema is currently in `audit.py` and normalization code. |
| Stage 2: schema inspection and enum audit | Implemented in this increment | `audit.py` and `scripts/audit_lob_schema.py` now emit `schema_audit.md/json`. Audit was run over all three parquet files. |
| Stage 3: synthetic tests first | Partially done | Existing tests cover enum mappings, reload/new/modify/fill/cancel, iceberg displayed quantity, market events, missing client IDs, output artifacts, partition isolation, max-row sorting, marketable New/Fill lifecycle handling, and stop-limit non-visibility before trigger. Missing: chunked-vs-full, deterministic hash, and deeper invariant tests. |
| Stage 4: normalized event parser | Done for first version | `normalize.py` emits core event fields, enum labels/classes, issue flags, and separate firm/client identifiers. |
| Stage 5: active order state machine | Implemented, but inline | Mutation logic exists in `panel.py`, not separate `state.py`. It now skips marketable New/Reload/Modify rows, defers marketable/aggressive fill residuals until they can rest without locking/crossing the book, and excludes pre-trigger stop-limit orders from visible depth. |
| Stage 6: price-level depth aggregation | Implemented, but inline | `book_summary` and snapshot output derive depth from active orders. There is no separate `book.py` or formal invariant report. |
| Stage 7: top-N LOB event panel | Implemented | `lob_event_state_panel.parquet` includes pre/post top-10 book state and issue flags. |
| Stage 8: agent-level active liquidity panel | Partially done | Firm/client pre/post bid/ask visible and leaves quantities, same/opposite-side quantities, and price/bps distance are present. Distance-bucket liquidity columns are not implemented. |
| Stage 9: CLI and reproducible outputs | Mostly done | CLI writes Parquet outputs, metadata, validation report, source hash, config, and row counts. Package versions and command provenance are not yet recorded. |
| Stage 10: real-file smoke/invariant reports | Partially done | One full parquet was reconstructed. Validation report is still minimal and lacks formal aggregation/reconstruction-quality diagnostics. |
| Stage 11: scalability and chunked processing | Not done | Current implementation reads full files into memory and processes sorted rows in a Python event loop. |
| Stage 12: future model-feature extraction | Not done | SCI/CPS/imbalance feature modules are not implemented. |
| Future exact queue reconstruction | Not done | Queue/FIFO logic remains intentionally deferred. |

## Main unresolved scientific/engineering risks

1. **Marketable lifecycle fix needs broader validation.**
   The follow-up diagnostic in `docs/lob_crossed_book_diagnostic.md` found that the original row-by-row mutation rule was too broad. The implemented fix eliminates locked/crossed rows in the full RISANAMENTO rerun, but it has not yet been validated across all three parquet files or against independent venue documentation for every special order type.

2. **Passive/aggressive semantics are encoded conservatively but remain partial.**
   The state machine now defers marketable fill residuals even when `AGGRESSIVEORDER` is missing, because real rows show many marketable fills with `PASSIVEORDER=N`, `AGGRESSIVEORDER=N`. This is conservative for visible depth, but queue/execution ordering remains approximate.

3. **Validation report is too shallow.**
   The report should add reconstruction-quality classifications, invariant summaries, counts of unseen-order references by event type/order type/partition, and aggregation checks.

4. **Client-original missingness is large.**
   This is expected from the data but means client-level spoofing features must either carry missingness explicitly or restrict to rows/agents with reliable client identity. Firm-level analysis remains complete for `FIRMID` in the audited files.

5. **No chunked/full equivalence test.**
   The current implementation is acceptable for correctness-first full-file runs of the inspected file sizes, but not yet proven as a scalable production pipeline.

## Current spoofing model version

The active exploratory spoofing pipeline follows the manuscript's multilevel top-n DWI/MSCI/MCPS formulation, not the older
single-imbalance SCI/CPS draft. The implemented schema is intentionally clean: legacy `imbalance`, `weighted_*_fraction_topN`,
and `candidate_fake_*` aliases are not emitted. Downstream analysis should use the current DWI/MSCI/MCPS and
`candidate_deceptive_*` column names directly.

Current model objects:

- top-n agent-specific visible-depth vectors by client;
- per-level relative client depth, `client_visible_qty_at_level / market_visible_qty_at_level`;
- shifted same-side tick distance, so level 1 has positive distance;
- normalized depth kernel with `kappa` and `lambda_`;
- state-level `L_bid_topN`, `L_ask_topN`, and `DWI`;
- event-level side collapses after passive small executions;
- event-level `MSCI`, high only when DWI changes and the opposite-side profile collapses more than the same side;
- client-level `MCPS`, the fraction of executions whose MSCI exceeds a chosen `gamma` threshold.
- candidate deceptive orders are restricted to a configurable pre-execution age window, currently 600 seconds by default,
  so long-lived resting liquidity is not attributed to a later spoofing episode.

The current event-level spoofing focus is the matched `candidate_deceptive_*` cancellation field: the cancelled order
must be one of the pre-existing opposite-side candidate deceptive orders. Broad opposite-side cancellations are not used
for the main spoofing-event interpretation.

These scores are surveillance cues, not labels or proof of manipulative intent. High-MSCI or high-MCPS cases require
episode-level review.

## Recommended next implementation step

For reconstruction quality, still prioritize:

1. Rerun reconstruction on all three parquet files and summarize locked/crossed counts by file/partition.
2. Add reconstruction-quality classifications to the validation report rather than relying on ad hoc notebook/script diagnostics.
3. Add chunked-vs-full and deterministic-output regression tests.
4. Audit the `marketable_order_not_resting` and `modify_for_unseen_order` rows by event type/order type/partition.

For spoofing analysis, use the current clean multilevel DWI/MSCI/MCPS pipeline and avoid reviving the superseded SCI-only
or `fake`-named schema except when reading historical outputs.
