# LOB Reconstruction Implementation Plan

> Scope guard: this document is planning only. It does not implement the pipeline.

**Goal:** design a reproducible Python pipeline that reconstructs an event-sourced, order-level visible limit order book and emits a queue-compatible price-level and agent-level event panel for the spoofing-surveillance model in `paper/spoofing.tex` / `paper/spoofing.pdf`.

**Architecture:** process each instrument/session independently in deterministic matching-engine order. Maintain an order-level active state keyed by `ORDERID`; derive price-level depth, top-N LOB variables, and event-agent liquidity from that active order state. Exact FIFO queue reconstruction is deferred, but all order-priority and sequencing fields must be preserved so a later queue module can be added without redesigning the pipeline.

**Tech stack:** Python, Polars for input/schema/sorting/writing, a transparent Python event loop for the correctness-critical state machine, PyArrow/Parquet outputs, pytest for synthetic and regression tests.

---

## 1. Evidence inspected

Inputs inspected for this plan:

- `docs/keep_cols_data_dictionary.md`
- `docs/tracciato_database_tabella_ordini.docx`
- `paper/spoofing.tex`
- `paper/spoofing.pdf` via `pdftotext`
- `notebooks/sample.csv`, used only for schema sanity checks
- lightweight diagnostics on the available `data/*.parquet` files to assess feasibility and ordering, not to infer substantive empirical conclusions

Observed schema facts:

- `notebooks/sample.csv` has 191 columns.
- The DOCX has one schema table with 153 rows including the header.
- The available parquet files have 191-192 columns; the substantive queue/reconstruction fields requested by the user are present in all three parquet files.
- Tooltip columns are not perfectly identical across parquet files, so implementation should not rely on tooltip columns for core logic.

Lightweight feasibility diagnostics from the parquet files:

- The full candidate ordering key using partition keys plus `SEQUENCETIME`, `HDR_APPLKEYSEQUENCENUMBER`, `HDR_HWMSEQUENCENUMBER`, `HDR_OFFSETID`, and `ROW_NUMBER` had zero duplicate keys in the three inspected parquet files.
- The first event type per observed instrument/session partition was `11 : GTC_GTD_Reload` in all inspected partitions.
- The first observed event per `ORDERID` was always either `1 : New` or `11 : GTC_GTD_Reload` in the three inspected files.
- Fill rows can have positive `LEAVESQTY` even though `ORDERSTATUS` is `N`; therefore fill continuation/removal must be driven primarily by `LEAVESQTY`, not by `ORDERSTATUS` alone.
- Cancel rows had `LEAVESQTY == 0` in the inspected files.

These diagnostics support feasibility, but they do not replace formal validation tests.

---

## 1A. Decisions after first review

User decisions recorded before implementation:

1. **Agent identity:** use both `FIRMID` and `NMSC_ORIGINALCLIENTIDSHORTCODE` as separate agent dimensions. Do **not** replace them with the composite `(FIRMID, NMSC_ORIGINALCLIENTIDSHORTCODE)`, and do not use only `NMSC_ORIGINALCLIENTIDSHORTCODE`. The first implementation should emit and aggregate active liquidity separately at firm level and at original-client-short-code level.
2. **Opening seed:** treat `11 : GTC_GTD_Reload` as the reloading of existing active GTC/GTD orders into the opening active book. The DOCX does not explicitly define `GTC_GTD_Reload`, but the observed parquet tooltip maps event code `11` to `GTC_GTD_Reload` and `ACKTYPE` code `18` to `Reload_Ack`. The implementation should seed active state from these rows and validate that later modifies/fills/cancels do not reference unseen orders.
3. **Event inclusion:** use all events. Keep every input event in the normalized event table and emit every order event in the LOB state panel. The visible LOB state still mutates only by the visible resting component implied by each event: iceberg reserve is not reconstructed, but iceberg displayed quantity/refills are processed; unpriced market events are retained but do not create resting visible depth; dark/midpoint/special events are retained and flagged unless they explicitly move visible quantity into the central order book.
4. **Top-N depth:** use top 10 bid and ask levels by default.
5. **Scaling:** use the values exactly as they are in the parquet files for `ORDERPX`, `ORDERQTY`, `DISPLAYEDQTY`, `LEAVESQTY`, `LASTSHARES`, and `LASTTRADEDPX`. The implementation must not rescale these fields.
6. **Enum mappings:** use the provisional mapping from observed parquet tooltip values, fail loudly on unknown numeric codes, and keep the enum-audit step in the pipeline. Official dictionaries can replace the provisional mappings later if provided.

---

## 2. Assumptions

1. **Visible-book target.** First implementation reconstructs the visible lit order book. Hidden volume, undisclosed iceberg reserve, and dark-pool state are not reconstructed unless visible quantities are explicitly reported.
2. **Order-level state is authoritative.** The internal state is order-level, keyed by `ORDERID` within an instrument/session partition. Price-level book depth is derived from the active order state, never maintained as the only source of truth.
3. **Displayed quantity drives visible depth.** Price-level depth and top-N book levels use `DISPLAYEDQTY` as the visible quantity. `LEAVESQTY` is still retained for active order state and later sensitivity checks.
4. **No exact queue reconstruction yet.** The first version does not simulate FIFO/pro-rata matching. It preserves `ORDERPRIORITY` and sequencing fields so exact queue logic can be added later.
5. **No look-ahead in event features.** Pre-event features use only active state before applying the current event. Post-event features use state immediately after applying the current event. Later SCI/CPS windows must be computed from the already reconstructed panel with explicit event-time windows.
6. **Use parquet price/quantity values as-is.** The DOCX says prices/quantities may need Price/Index Level Decimals and Quantity Decimals in raw feeds, but the decision for this project is to use the values exactly as stored in the parquet files and not rescale them.
7. **Enum labels from tooltip columns are observed ETL labels, not authoritative dictionaries.** Tooltip columns are useful for auditing but should not be required by core logic. Numeric code columns must drive the state machine once audited.
8. **GTC/GTD reloads seed opening state.** `11 : GTC_GTD_Reload` is treated as reloading existing active GTC/GTD orders into the opening active book. Validation must still check for later unseen-order references.

---

## 3. Reconstruction feasibility assessment

### 3.1 What appears feasible

A visible order-level LOB reconstruction appears feasible for each observed instrument/session if the following hold:

- the event feed contains all order events for that instrument/session from the session seed onward;
- `11 : GTC_GTD_Reload` fully seeds active carry-over orders before intraday events;
- `ORDERID` is stable over the order lifecycle within `(TRADEDATE, SYMBOLINDEX, EMM (*))`;
- `LEAVESQTY`, `DISPLAYEDQTY`, `ORDERPX`, and `ORDERSIDE (*)` are post-event order-state fields for active events;
- special events such as `Move_Dark_to_COB`, `Refill`, `VFA_VFC`, and `Trigger` can be classified consistently after enum audit.

Under those assumptions, the first implementation can reconstruct:

- active order state;
- visible depth by price and side;
- best bid, best ask, mid, spread;
- top-N visible levels before and after each event;
- event-agent active bid/ask liquidity before and after each event;
- event order distance from the same-side best quote;
- sufficient pre/post state for later imbalance, SCI, and CPS computation.

### 3.2 What is not yet exact

The first implementation should not claim exact reconstruction for:

- FIFO or pro-rata queue position;
- hidden iceberg reserve or dark-order state;
- off-book or wholesale states not represented in central visible book events;
- trade-bust rollback unless trade-bust events are explicitly documented and tested;
- full-market context if files contain only selected instruments rather than all instruments;
- exact opening state if reload validation finds later references to unseen orders.

### 3.3 Exact versus partial reconstruction rule

For each partition, classify reconstruction quality as one of:

- `exact_visible_seeded`: first event stream includes complete GTC/GTD reload or begins before any active order exists, and all later order IDs start with New/Reload;
- `partial_visible_seeded`: first event stream starts with some active reloads but validation finds modifies/fills/cancels for unseen orders;
- `midstream_partial`: first events include modifications/fills/cancels for unseen orders, so only post-observation state can be reconstructed;
- `invalid_ordering`: deterministic ordering cannot be established.

The pipeline should write these quality flags to run metadata and attach per-row issue flags where relevant.

---

## 4. Canonical partition keys

### Recommended processing partition

Process event streams independently by:

```text
TRADEDATE
MIC
MARKETCODE
SYMBOLINDEX
EMM (*)
```

Retain and validate these metadata columns in every row:

```text
ISIN
INSTRUMENTID
INSTRUMENTGROUPCODE
OFFICIALSEGMENT
TRADINGCURRENCY
```

### Rationale

- The DOCX states that `ORDERID` is unique per instrument and EMM.
- The DOCX states that `EXECUTIONID` is unique per instrument and per day.
- `SYMBOLINDEX` is the exchange instrument identifier and is unique for the triplet `MIC`, `ISIN`, and currency.
- `TRADEDATE` is required because order IDs and execution IDs are not guaranteed to be unique across days.
- `ISIN` alone is too weak because the same ISIN can appear under different market mechanisms, MICs, currencies, or symbol indexes.
- `MARKETCODE` is retained in the partition key for safety even if it is redundant with `MIC` in the current files.

Validation should assert that within each canonical partition, `ISIN`, `INSTRUMENTID`, `TRADINGCURRENCY`, and other instrument metadata are constant unless explicitly documented.

---

## 5. Deterministic event ordering

### Candidate fields

Native or ETL fields available for ordering:

- `SEQUENCETIME`: unique sequence time set by the Matching Engine for synchronization across Kafka topics.
- `BOOKIN`: Matching Engine input time.
- `BOOKOUTTIME`: Matching Engine output time.
- `TRADETIME`: trade time; DOCX says it equals Matching Engine input time when the aggressor enters the Matching Engine.
- `HDR_APPLKEYSEQUENCENUMBER`: technical sequence field.
- `HDR_HWMSEQUENCENUMBER`: technical sequence field.
- `HDR_OFFSETID`: technical offset field.
- `ROW_NUMBER`: ETL-derived row counter, not native exchange semantics.
- `EVENTID` and `ORDERID`: identity fields, useful only as last-resort deterministic tie-breakers if required.

### Proposed sort key

Within each canonical partition, sort by:

```python
sort_key = [
    "SEQUENCETIME",
    "HDR_APPLKEYSEQUENCENUMBER",
    "HDR_HWMSEQUENCENUMBER",
    "HDR_OFFSETID",
    "BOOKIN",
    "BOOKOUTTIME",
    "TRADETIME",
    "ROW_NUMBER",
]
```

If a field is null, sort nulls last within that field. `ROW_NUMBER` is the final intended deterministic tie-breaker because it is dataset-specific, not exchange-native.

### Why `SEQUENCETIME` is primary

The DOCX describes `SEQUENCETIME` as a unique Matching Engine sequence time used for synchronization across Kafka topics. It is therefore the closest available field to matching-engine event order. `BOOKIN` and `BOOKOUTTIME` are still retained because they are useful for latency diagnostics and for auditing whether `SEQUENCETIME` order is consistent with matching-engine input/output order.

### Ordering validation tests

For every input file and partition:

1. Parse time columns to a consistent nanosecond representation.
2. Assert that the full ordering key has no duplicates. If duplicates exist, append `EVENTID`, `ORDERID`, and finally original file row position, and flag `ordering_tie_resolved`.
3. Assert repeated sorting gives byte-identical ordering hashes.
4. Assert each `ORDERID` lifecycle starts with `New` or `GTC_GTD_Reload`; otherwise flag `unseen_prior_order`.
5. Compare ordering by `SEQUENCETIME` versus `BOOKIN` where both are non-null; report inversions rather than silently correcting.
6. Test that chunked processing within a partition produces the same row order and output hashes as full-file processing.

---

## 6. Field mapping

### 6.1 Core reconstruction fields

| Normalized concept | Dataset column | Source / type | Native, renamed, derived, or ambiguous | Use |
|---|---|---|---|---|
| Trading date | `TRADEDATE` | `tradedate` | native, upper-case | Session partition. |
| ISIN | `ISIN` | DOCX `codisin` | renamed | Instrument metadata and validation. |
| Market identifier | `MIC`, `MARKETCODE` | source fields | native | Partition and market metadata. |
| Exchange market mechanism | `EMM (*)` | `emm` | coded enum | Partition and book context. Enum labels need audit. |
| Symbol index | `SYMBOLINDEX` | `symbolindex` | native | Primary instrument key. |
| Instrument ID | `INSTRUMENTID` | `enr_instrumentid` | renamed prefix removed | Instrument metadata and validation. |
| Sequence time | `SEQUENCETIME` | `sequencetime` | native | Primary event ordering. |
| ME input time | `BOOKIN` | `bookin` | native | Ordering audit and latency. |
| ME output time | `BOOKOUTTIME` | `bookouttime` | native | Ordering audit and latency. |
| Trade time | `TRADETIME` | `tradetime` | native | Fill timing and SCI anchor. |
| Technical sequence | `HDR_APPLKEYSEQUENCENUMBER` | same lower-case | native technical | Tie-breaker, queue compatibility. |
| HWM sequence | `HDR_HWMSEQUENCENUMBER` | same lower-case | native technical | Tie-breaker, queue compatibility. |
| Offset | `HDR_OFFSETID` | same lower-case | native technical | Tie-breaker, queue compatibility. |
| Row number | `ROW_NUMBER` | not in DOCX | ETL-derived | Last deterministic tie-breaker. |
| Event id | `EVENTID` | `eventid` | native | Event identity and debugging. |
| Event type | `ORDEREVENTTYPE (*)` | `ordereventtype` | coded enum | State-machine classification. Numeric mapping needs audit. |
| Order id | `ORDERID` | `orderid` | native | Active-order key. |
| Order priority | `ORDERPRIORITY` | `orderpriority` | native | Queue-compatible metadata. |
| Client order id | `CLIENTORDERID` | `clientorderid` | native | Lifecycle/debug reconciliation. |
| Original client order id | `ORIGCLIENTORDERID` | `origclientorderid` | native | Cancel/replace reconciliation. |
| Execution id | `EXECUTIONID` | `executionid` | native | Fill identity; future trade pairing. |
| Trade unique id | `TRADEUNIQUEIDENTIFIER` | `tradeuniqueidentifier` | native | Trade identity and future pairing. |
| Side | `ORDERSIDE (*)` | `orderside` | coded enum | Bid/ask side. Mapping must be audited. |
| Order price | `ORDERPX` | `orderpx` | native | Price-level state. Use parquet value as-is. |
| Total order quantity | `ORDERQTY` | `orderqty` | native | Original/current total order quantity. Use parquet value as-is. |
| Displayed quantity | `DISPLAYEDQTY` | `displayedqty` | native | Visible depth. |
| Remaining quantity | `LEAVESQTY` | `leavesqty` | native | Active state after event. |
| Last traded quantity | `LASTSHARES` | `lastshares` | native | Fill size. |
| Last traded price | `LASTTRADEDPX` | `lasttradedpx` | native | Fill price. |
| Order type | `ORDERTYPE (*)` | `ordertype` | coded enum | Resting/market/iceberg/stop classification. |
| Time in force | `TIMEINFORCE (*)` | `timeinforce` | coded enum | Persistence and validity controls. |
| Kill reason | `KILLREASON (*)` | `killreason` | coded enum | Cancellation reason. |
| Order status | `ORDERSTATUS` | likely decoded from `orderqualifiers` | derived and ambiguous | Cross-check only until semantics validated. Do not let it override positive `LEAVESQTY` on fills. |
| Passive role | `PASSIVEORDER` | likely decoded from `tradequalifier` or `orderqualifiers` | derived and ambiguous | Fill role / aggressor-passive logic after audit. |
| Aggressive role | `AGGRESSIVEORDER` | likely decoded from `tradequalifier` or `orderqualifiers` | derived and ambiguous | Fill role / marketable order logic after audit. |

### 6.2 Agent and regulatory fields

| Agent concept | Dataset column | Source / type | Recommended use |
|---|---|---|---|
| Member firm | `FIRMID` | native | Required firm-level agent dimension. |
| Original client short code | `NMSC_ORIGINALCLIENTIDSHORTCODE` | native | Required original-client-level agent dimension, subject to null handling. |
| Event client short code | `MSC_EVENTCLIENTIDSHORTCODE` | native | Event-level alternative/fallback; can change over order life. |
| Original execution-within-firm short code | `NMSC_ORIGINALEXECWFIRMSHORTCODE` | native | Trader/algorithm responsible for execution; useful alternative agent definition. |
| Event execution-within-firm short code | `MSC_EVENTEXECWFIRMSHORTCODE` | native | Event-level execution actor alternative. |
| Original investment-decision short code | `NMSC_ORIGINALINVESTDECISWFIRMSHORTCODE` | native | Investment-decision actor alternative. |
| Original non-executing broker short code | `NMSC_ORIGINALNONEXECBROKERSHORTCODE` | native | Broker role/control. |
| Account type | `ACCOUNTTYPEINTERNAL (*)` | coded enum | Client/house/liquidity-provider control. |
| Liquidity provider role | `LPROLE (*)` | coded enum | Market-maker / liquidity-provider control. |
| Trading capacity | `ORDER_TRADINGCAPACITY (*)` | renamed from `nmof_tradingcapacity` | Own account / matched principal / other-capacity control. |
| DEA flag | `DEAINDICATOR` | decoded from `mifidindicators` | Regulatory control. |
| Investment algo flag | `INVESTMENTALGOINDICATOR` | decoded from `mifidindicators` | Algorithmic decision control. |
| Execution algo flag | `EXECUTIONALGOINDICATOR` | decoded from `mifidindicators` | Algorithmic execution control. |
| French AMP LP flag | `FRMARAMPLP` | decoded from `mifidindicators` | Liquidity-provision control. |

### 6.3 Native, renamed, derived, ambiguous fields

Native fields to preserve directly:

```text
TRADEDATE, MIC, MARKETCODE, SYMBOLINDEX, ORDERID, ORDERPRIORITY,
ORDERPX, ORDERQTY, DISPLAYEDQTY, LEAVESQTY, LASTSHARES,
LASTTRADEDPX, CLIENTORDERID, ORIGCLIENTORDERID, EXECUTIONID,
TRADEUNIQUEIDENTIFIER, BOOKIN, BOOKOUTTIME, TRADETIME, SEQUENCETIME,
NMSC_ORIGINALCLIENTIDSHORTCODE, FIRMID
```

Renamed fields to audit:

```text
ISIN <- codisin
INSTRUMENTID <- enr_instrumentid
ORDER_TRADINGCAPACITY (*) <- nmof_tradingcapacity
```

Derived bit-field columns to audit before using substantively:

```text
ORDERSTATUS
PASSIVEORDER
AGGRESSIVEORDER
DEAINDICATOR
INVESTMENTALGOINDICATOR
EXECUTIONALGOINDICATOR
FRMARAMPLP
UNCROSSINGTRADE
DARKINDICATOR
SWEEPORDERINDICATOR
```

Ambiguous or incomplete enum dictionaries:

```text
EMM (*)
ORDEREVENTTYPE (*)
ORDERSIDE (*)
ORDERTYPE (*)
TIMEINFORCE (*)
KILLREASON (*)
ACKPHASE (*)
ACKTYPE (*)
EXECUTIONPHASE (*)
TRADETYPE (*)
ACCOUNTTYPEINTERNAL (*)
LPROLE (*)
ORDER_TRADINGCAPACITY (*)
```

Observed tooltip labels may be used for exploratory audits, but final logic must be explicit about numeric codes and must fail loudly on unknown codes.

---

## 7. Normalized event semantics

### 7.1 Event classification outputs

The normalizer should add these event fields:

```text
event_type_code
event_type_label_observed
event_class
is_book_affecting
is_resting_candidate
is_execution
is_partial_fill
is_full_fill
is_cancel
is_replace_or_modify
is_new_order
is_reload
is_reject
is_special_event
unknown_enum_flag
```

### 7.2 Proposed first-pass event classes

| Observed code/label | Proposed class | First-version state mutation | Ambiguities / notes |
|---|---|---|---|
| `1 : New` | `new_order` | If priced, visible, and `LEAVESQTY > 0`, add/update active order keyed by `ORDERID`. Market orders with null price are not added as resting orders. | Need side mapping and price/qty scaling audit. |
| `2 : Modify` | `modify_order` | Update active order fields: price, side if changed, leaves, displayed, priority, order type, time-in-force, agent metadata. If missing prior order and active/priced, initialize with issue flag. | Need confirm cancel-replace semantics and whether `ORDERID` remains stable. |
| `3 : Fill` | `fill` | Use `LEAVESQTY` as post-fill remaining quantity. If `LEAVESQTY > 0`, keep/update active order; if `LEAVESQTY == 0`, remove active order. Use `LASTSHARES` and `LASTTRADEDPX` for fill metadata. | Observed fills can have positive `LEAVESQTY` while `ORDERSTATUS=N`, so `ORDERSTATUS` is not sufficient. Need passive/aggressive derivation audit. |
| `4 : Cancel` | `cancel` | Remove active order. If cancel has positive leaves in future data, treat as reduce/update only after enum audit; flag unexpected positive leaves. | Observed cancels had zero leaves. Need cancellation reason mapping. |
| `6 : Trigger` | `trigger` | Do not add unless subsequent audit proves it creates a visible active order. Emit event and issue flag if it appears to require state mutation. | Stop-trigger semantics unclear from available docs. |
| `7 : Refill` | `iceberg_refill` | Update `DISPLAYEDQTY`, `LEAVESQTY`, priority if applicable; keep order active if visible. | Need confirm iceberg displayed/refill semantics. |
| `9 : VFA_VFC` | `special_validity_event` | Keep row, flag special. If active/priced with positive leaves, update state conservatively only after audit. | Meaning not fully documented in available files. |
| `11 : GTC_GTD_Reload` | `session_reload` | Seed active state at session start if priced and `LEAVESQTY > 0`. Treat as reloading existing active GTC/GTD orders into the opening active book. | User decision: this event seeds opening active orders; validate no later unseen-order references. |
| `23 : Move_Dark_to_COB` | `move_dark_to_cob` | Treat as add/update to visible book only if `DARKINDICATOR`/visible fields imply it is now in the central order book. Preserve dark flag. | Dark-to-lit semantics require audit. |
| Reject events | `reject` | Do not add to active state. Keep event row for audit. | Rejects may appear in `ACKTYPE` rather than `ORDEREVENTTYPE`. |
| Trade bust / trade cancellation | `trade_bust` | First version should classify and flag, not attempt rollback unless documented. | DOCX says `EXECUTIONID` is reused for trade cancellation, but observed event codes did not expose trade busts. |

### 7.3 `ORDERSTATUS`, `LEAVESQTY`, `LASTSHARES`, passive/aggressive usage

- `LEAVESQTY` is the primary post-event active quantity for fills and active updates.
- `DISPLAYEDQTY` is the primary visible-book quantity.
- `ORDERSTATUS` is a consistency check, not the only source of truth. In observed data, fills can have `LEAVESQTY > 0` while `ORDERSTATUS=N`.
- `LASTSHARES` and `LASTTRADEDPX` are execution fields and should be non-null for fill events. Missing values should raise issue flags.
- `PASSIVEORDER` and `AGGRESSIVEORDER` should not be used as definitive role labels until the ETL source bit field is confirmed. The DOCX describes both trade-level and order-level aggressive/passive bits.
- For market orders with null `ORDERPX`, the event can still be an execution but should not create a resting visible order unless a later market-to-limit transformation event provides a price.

---

## 8. In-memory reconstruction state

### 8.1 Active order state

Maintain:

```python
active_orders: dict[PartitionKey, dict[ORDERID, ActiveOrder]]
```

Each `ActiveOrder` must retain at least:

```text
partition keys
ORDERID
ORDERPRIORITY
side_code
side_label
price
leaves_qty
displayed_qty
original_order_qty
order_type_code
time_in_force_code
client_order_id
orig_client_order_id
FIRMID
NMSC_ORIGINALCLIENTIDSHORTCODE
MSC_EVENTCLIENTIDSHORTCODE
NMSC_ORIGINALEXECWFIRMSHORTCODE
MSC_EVENTEXECWFIRMSHORTCODE
NMSC_ORIGINALINVESTDECISWFIRMSHORTCODE
NMSC_ORIGINALNONEXECBROKERSHORTCODE
account_type_code
lp_role_code
trading_capacity_code
DEAINDICATOR
INVESTMENTALGOINDICATOR
EXECUTIONALGOINDICATOR
FRMARAMPLP
first_seen_time
last_update_time
last_update_sort_key
last_event_type
current_active_status
issue_flags
```

### 8.2 Derived price-level state

Derive price-level visible depth from `active_orders` after each mutation:

```text
price_level_depth[side][price] = sum(displayed_qty for active orders at side/price)
price_level_order_count[side][price] = count(active orders at side/price)
```

Implementation can initially recompute the affected side/price levels for correctness. After correctness is validated, maintain incremental aggregates with tests proving equality to full aggregation.

### 8.3 Top-N state

For configurable `N` (default proposal: 10), derive:

```text
bid_level_1_price, bid_level_1_visible_qty, ..., bid_level_N_price, bid_level_N_visible_qty
ask_level_1_price, ask_level_1_visible_qty, ..., ask_level_N_price, ask_level_N_visible_qty
```

Bids sort by descending price. Asks sort by ascending price. Empty levels are null price and zero/null quantity by documented convention.

### 8.4 Agent-level active liquidity state

For the event agent, derive before and after each event:

```text
agent_active_bid_visible_qty
agent_active_ask_visible_qty
agent_active_bid_leaves_qty
agent_active_ask_leaves_qty
agent_active_bid_order_count
agent_active_ask_order_count
agent_active_same_side_visible_qty
agent_active_opposite_side_visible_qty
agent_active_same_side_order_count
agent_active_opposite_side_order_count
```

Maintain both visible and leaves quantities. Use visible quantities for spoofing model default; use leaves quantities for diagnostics/sensitivity.

### 8.5 Distance and bucket state

For each active order of the event agent, compute distance to the same-side best quote using the relevant pre or post book state:

- bid order distance in price units: `best_bid - order_price`;
- ask order distance in price units: `order_price - best_ask`.

Distances should be non-negative for non-crossed books. If negative, flag `distance_negative_crossed_or_stale_book`.

Also compute optional basis-point distance:

```text
bid_distance_bps = 10000 * (best_bid - price) / mid
ask_distance_bps = 10000 * (price - best_ask) / mid
```

Tick distance requires a tick-size source or an explicit inference rule and should be deferred unless approved.

Proposed initial distance buckets:

```text
at_touch: distance == 0
near: 0 < distance_bps <= 5
medium: 5 < distance_bps <= 25
far: distance_bps > 25
unknown: missing best quote, missing price, or crossed/locked issue
```

Bucket thresholds are placeholders for engineering diagnostics only; model thresholds must be calibrated later.

---

## 9. Agent identity design

### Required first-version agent dimensions

The first implementation should use two separate agent dimensions:

```text
firm: FIRMID
client_original: NMSC_ORIGINALCLIENTIDSHORTCODE
```

Do **not** construct `(FIRMID, NMSC_ORIGINALCLIENTIDSHORTCODE)` as the agent key for the first implementation. The user decision is to use both `FIRMID` and `NMSC_ORIGINALCLIENTIDSHORTCODE` as separate identities, not a composite key and not client-only.

Implementation implication:

- store both identifiers on every normalized event and every active order;
- compute active liquidity at firm level;
- compute active liquidity at original-client-short-code level;
- keep rows with missing `NMSC_ORIGINALCLIENTIDSHORTCODE` in the event stream and flag the missing client-level identity rather than dropping them;
- allow downstream spoofing features to be computed separately by firm or by original client short code.

### Identity fields retained for later sensitivity analysis

Always retain these identity fields in normalized event output:

```text
FIRMID
NMSC_ORIGINALCLIENTIDSHORTCODE
MSC_EVENTCLIENTIDSHORTCODE
NMSC_ORIGINALEXECWFIRMSHORTCODE
MSC_EVENTEXECWFIRMSHORTCODE
NMSC_ORIGINALINVESTDECISWFIRMSHORTCODE
NMSC_ORIGINALNONEXECBROKERSHORTCODE
```

Use `FIRMID` and `NMSC_ORIGINALCLIENTIDSHORTCODE` as the two required first-version agent dimensions. Preserve the other identity fields so later analyses can compare event-client, execution-within-firm, investment-decision, and non-executing-broker identities.

### Agent-state output convention

For long-format agent-state output, use:

```text
agent_dimension  # "firm" or "client_original"
agent_id
agent_id_missing_flag
agent_id_source
```

For wide event-panel convenience, emit both firm-level and client-level aggregates with explicit prefixes, for example:

```text
pre_firm_active_bid_visible_qty
pre_client_original_active_bid_visible_qty
post_firm_active_bid_visible_qty
post_client_original_active_bid_visible_qty
```

---

## 10. Output schemas

### 10.1 Normalized event table

One row per input event after cleaning and normalization.

Suggested path pattern:

```text
outputs/lob_reconstruction/{run_id}/normalized_events.parquet
```

Core columns:

```text
run_id
source_file
partition_id
TRADEDATE
MIC
MARKETCODE
SYMBOLINDEX
EMM (*)
ISIN
INSTRUMENTID
TRADINGCURRENCY
sort_index
SEQUENCETIME
BOOKIN
BOOKOUTTIME
TRADETIME
HDR_APPLKEYSEQUENCENUMBER
HDR_HWMSEQUENCENUMBER
HDR_OFFSETID
ROW_NUMBER
EVENTID
ORDERID
ORDERPRIORITY
CLIENTORDERID
ORIGCLIENTORDERID
EXECUTIONID
TRADEUNIQUEIDENTIFIER
event_type_code
event_type_label_observed
event_class
side_code
side_label
ORDERPX
ORDERQTY
DISPLAYEDQTY
LEAVESQTY
LASTSHARES
LASTTRADEDPX
ORDERTYPE (*)
TIMEINFORCE (*)
KILLREASON (*)
ORDERSTATUS
PASSIVEORDER
AGGRESSIVEORDER
UNCROSSINGTRADE
DARKINDICATOR
SWEEPORDERINDICATOR
QUOTEINDICATOR
firm_id
client_original_id
client_original_id_missing_flag
FIRMID
NMSC_ORIGINALCLIENTIDSHORTCODE
MSC_EVENTCLIENTIDSHORTCODE
NMSC_ORIGINALEXECWFIRMSHORTCODE
MSC_EVENTEXECWFIRMSHORTCODE
NMSC_ORIGINALINVESTDECISWFIRMSHORTCODE
NMSC_ORIGINALNONEXECBROKERSHORTCODE
ACCOUNTTYPEINTERNAL (*)
LPROLE (*)
ORDER_TRADINGCAPACITY (*)
DEAINDICATOR
INVESTMENTALGOINDICATOR
EXECUTIONALGOINDICATOR
FRMARAMPLP
unknown_enum_flag
normalization_issue_flags
```

### 10.2 Optional active order state snapshots for debugging

Not every run needs full snapshots because they can be large. Provide a debug option:

```text
outputs/lob_reconstruction/{run_id}/active_order_snapshots.parquet
```

Granularity options:

- `none`: default for large runs;
- `every_event_for_sample`: only for small synthetic/sample tests;
- `issue_rows_only`: snapshot around invariant violations;
- `end_of_partition`: final active book per partition.

Columns:

```text
run_id
partition_id
snapshot_sort_index
snapshot_reason
ORDERID
side_code
price
leaves_qty
displayed_qty
original_order_qty
order_type_code
time_in_force_code
ORDERPRIORITY
firm_id
client_original_id
client_original_id_missing_flag
FIRMID
NMSC_ORIGINALCLIENTIDSHORTCODE
first_seen_time
last_update_time
last_event_type
issue_flags
```

### 10.3 LOB event-state panel

One row per relevant order-book event, with pre/post state and top-N levels.

Suggested path:

```text
outputs/lob_reconstruction/{run_id}/lob_event_state_panel.parquet
```

Columns include all normalized event identifiers plus:

```text
pre_best_bid
pre_best_ask
pre_mid
pre_spread
pre_bid_visible_qty_total
pre_ask_visible_qty_total
pre_bid_order_count_total
pre_ask_order_count_total
post_best_bid
post_best_ask
post_mid
post_spread
post_bid_visible_qty_total
post_ask_visible_qty_total
post_bid_order_count_total
post_ask_order_count_total
pre_bid_level_{k}_price
pre_bid_level_{k}_visible_qty
pre_bid_level_{k}_order_count
pre_ask_level_{k}_price
pre_ask_level_{k}_visible_qty
pre_ask_level_{k}_order_count
post_bid_level_{k}_price
post_bid_level_{k}_visible_qty
post_bid_level_{k}_order_count
post_ask_level_{k}_price
post_ask_level_{k}_visible_qty
post_ask_level_{k}_order_count
book_locked_pre_flag
book_crossed_pre_flag
book_locked_post_flag
book_crossed_post_flag
lob_issue_flags
```

where `k = 1..N`.

### 10.4 Agent-state panel

For the first version, include firm-level and original-client-level event-agent state directly in the LOB event-state panel. Also allow a separate long panel for debugging or aggregation. In long format, emit one row per `(event, agent_dimension)` for `firm` and `client_original`:

```text
outputs/lob_reconstruction/{run_id}/agent_event_state_panel.parquet
```

Columns:

```text
run_id
partition_id
sort_index
agent_dimension
agent_id
agent_id_source
agent_id_missing_flag
event_ORDERID
event_side_code
event_class
pre_agent_bid_visible_qty
pre_agent_ask_visible_qty
pre_agent_bid_leaves_qty
pre_agent_ask_leaves_qty
pre_agent_bid_order_count
pre_agent_ask_order_count
post_agent_bid_visible_qty
post_agent_ask_visible_qty
post_agent_bid_leaves_qty
post_agent_ask_leaves_qty
post_agent_bid_order_count
post_agent_ask_order_count
pre_agent_same_side_visible_qty
pre_agent_opposite_side_visible_qty
post_agent_same_side_visible_qty
post_agent_opposite_side_visible_qty
pre_event_order_same_side_distance_price
pre_event_order_same_side_distance_bps
post_event_order_same_side_distance_price
post_event_order_same_side_distance_bps
pre_agent_distance_bucket_{bucket}_{side}_visible_qty
post_agent_distance_bucket_{bucket}_{side}_visible_qty
agent_issue_flags
```

### 10.5 Later model-feature panel

Do not implement in the first LOB reconstruction step, but reserve schema names:

```text
outputs/lob_reconstruction/{run_id}/spoofing_feature_panel.parquet
outputs/lob_reconstruction/{run_id}/sci_events.parquet
outputs/lob_reconstruction/{run_id}/cps_by_agent_session.parquet
```

Future columns:

```text
imbalance_pre
imbalance_post
kappa
agent_bid_distance_weighted_volume_pre
agent_ask_distance_weighted_volume_pre
agent_bid_distance_weighted_volume_post
agent_ask_distance_weighted_volume_post
execution_event_id
pre_fill_imbalance
post_fill_window_imbalance
SCI
SCI_threshold_gamma
CPS
post_fill_cancel_detected
post_fill_cancel_latency_ns
spoofing_candidate_flag
```

---

## 11. First-version LOB features

The first implementation should emit, at minimum:

```text
pre_best_bid
pre_best_ask
pre_mid
pre_spread
post_best_bid
post_best_ask
post_mid
post_spread
pre/post top-N bid price and visible volume levels
pre/post top-N ask price and visible volume levels
event_side
event_price
event_order_qty
event_displayed_qty
event_leaves_qty
event_last_shares
event_last_traded_price
event_firm_id
event_client_original_id
pre_firm_active_bid_visible_qty
pre_firm_active_ask_visible_qty
post_firm_active_bid_visible_qty
post_firm_active_ask_visible_qty
pre_client_original_active_bid_visible_qty
pre_client_original_active_ask_visible_qty
post_client_original_active_bid_visible_qty
post_client_original_active_ask_visible_qty
pre_firm_active_same_side_visible_qty
pre_firm_active_opposite_side_visible_qty
post_firm_active_same_side_visible_qty
post_firm_active_opposite_side_visible_qty
pre_client_original_active_same_side_visible_qty
pre_client_original_active_opposite_side_visible_qty
post_client_original_active_same_side_visible_qty
post_client_original_active_opposite_side_visible_qty
pre_event_order_same_side_distance_price
pre_event_order_same_side_distance_bps
post_event_order_same_side_distance_price
post_event_order_same_side_distance_bps
optional distance-bucket visible quantities
```

For events that cannot have a meaningful same-side distance, such as unpriced market orders, distance fields should be null and an issue/status flag should explain why.

---

## 12. Future spoofing features supported by the panel

### 12.1 Normalized distance-weighted imbalance

The paper defines, for agent `i`, ask volume `V_a`, bid volume `V_b`, and same-side distances `delta_a`, `delta_b`:

```text
imb_i_tau = [V_a * (1 - exp(-kappa * delta_a)) - V_b * (1 - exp(-kappa * delta_b))] / (V_a + V_b)
```

The reconstructed panel supports this by providing active visible volumes by side and distance buckets/active orders for both required agent dimensions: `firm` (`FIRMID`) and `client_original` (`NMSC_ORIGINALCLIENTIDSHORTCODE`). The first implementation should not calibrate `kappa`; it should only preserve the order-level and agent-level state needed for later calibration.

### 12.2 Pre-fill and post-fill imbalance

For execution events, pre-fill imbalance is computed from the event-agent pre-state. Post-fill imbalance should be computed at `t_e + Delta_tau` using a future event-time window, not by peeking into raw future events during reconstruction. The LOB event-state panel provides deterministic event indices and timestamps for this later window join.

### 12.3 SCI

The paper defines:

```text
SCI_i_e = |imb_i(t_e^-) - imb_i(t_e^+)|
```

The reconstructed panel supports SCI because each event row has pre-state, post-state, event class, execution markers, and active agent liquidity. SCI should be computed only after the LOB panel is built, using explicit windows and avoiding train/test leakage.

### 12.4 CPS

The paper defines:

```text
CPS_i = (1 / N) * sum_e I(SCI_i_e > gamma)
```

The panel supports CPS by grouping execution events separately by `(FIRMID, TRADEDATE, SYMBOLINDEX, EMM (*))` and by `(NMSC_ORIGINALCLIENTIDSHORTCODE, TRADEDATE, SYMBOLINDEX, EMM (*))` after SCI is computed. The threshold `gamma` must be calibrated or set in a documented exploratory way.

### 12.5 Spoofing-like versus market-making behavior

The panel preserves controls needed to separate spoofing-like behavior from market making:

- liquidity-provider/account-role fields: `ACCOUNTTYPEINTERNAL (*)`, `LPROLE (*)`, `FRMARAMPLP`;
- symmetric bid/ask active liquidity before/after events;
- top-of-book versus deep-book placement through distance fields;
- cancellation latency after opposite-side fills;
- repeated SCI threshold crossings by agent/session;
- passive/aggressive flags once audited.

---

## 13. Validation strategy and invariants

### 13.1 Deterministic sorting tests

- Sorting the same input twice produces identical event order hashes.
- Full ordering keys are unique within each canonical partition, or deterministic tie-break issue flags are emitted.
- `ROW_NUMBER` is used only as the final tie-breaker.
- Sorting is stable across full-file and chunked processing.

### 13.2 State-machine invariants

- No active order has negative `leaves_qty` or negative `displayed_qty`.
- No active order has null side.
- No active visible resting order has null price, except documented special order types excluded from visible book.
- For non-iceberg visible orders, `displayed_qty <= leaves_qty` unless documented otherwise.
- For every fill with a known prior active order, post-event `leaves_qty` equals the event `LEAVESQTY`.
- For every fill, `LASTSHARES` is positive and non-null; if not, flag it.
- Full fills remove orders from active state.
- Partial fills keep orders active with updated `LEAVESQTY` and visible quantity.
- Cancels remove active orders or reduce them only under a documented cancel-reduce event type.
- Modifies update active state without creating duplicate active `ORDERID` records.
- Unknown event types do not silently mutate active state.

### 13.3 Book aggregation invariants

- For every event, price-level visible volume equals aggregation of active orders by side and price.
- Best bid equals max active bid price; best ask equals min active ask price.
- `pre_mid = (pre_best_bid + pre_best_ask) / 2` when both sides exist.
- `pre_spread = pre_best_ask - pre_best_bid` when both sides exist.
- Best bid should not exceed best ask except in explicitly flagged locked/crossed cases.
- Top-N levels are sorted correctly: bids descending, asks ascending.
- Top-N level quantities equal price-level aggregates.

### 13.4 Agent-state invariants

- Agent bid/ask active visible quantities equal aggregation of active orders for that agent and side.
- Event-agent same-side and opposite-side quantities are consistent with event side.
- Distance values are non-negative in non-crossed books.
- Missing agent identifiers produce flags, not dropped rows.

### 13.5 Reproducibility tests

- Repeated processing of the same file produces identical output hashes and metadata.
- Chunked processing produces identical output to full-file processing.
- Processing partitions independently and concatenating outputs equals processing the full sorted file.
- Output metadata records source file path, source file size, source file hash, code version, config, top-N, agent dimensions, and timestamp.

### 13.6 Synthetic hand-computable unit tests

Create synthetic event sequences before processing real files:

1. `test_new_orders_create_best_quotes`: add bid and ask; verify best bid/ask/mid/spread.
2. `test_modify_reprices_order`: modify bid price and qty; verify old price level removed and new level added.
3. `test_partial_fill_reduces_leaves`: fill with positive `LEAVESQTY`; verify order remains active.
4. `test_full_fill_removes_order`: fill with zero `LEAVESQTY`; verify order removed.
5. `test_cancel_removes_order`: cancel active order; verify depth decreases.
6. `test_market_order_null_price_not_resting`: market order with null price does not enter active book.
7. `test_iceberg_displayed_quantity_drives_visible_depth`: leaves greater than displayed; visible depth uses displayed qty.
8. `test_gtc_reload_seeds_opening_book`: reload events initialize active state before new intraday events.
9. `test_same_timestamp_tiebreak_is_deterministic`: duplicate timestamps sorted by technical fields.
10. `test_chunked_equals_full`: split a synthetic stream into chunks and compare output hashes.
11. `test_unknown_enum_flags_and_does_not_mutate`: unknown event type is emitted with issue flag and no silent state mutation.
12. `test_crossed_book_flag`: synthetic crossed state triggers flag without crashing.

### 13.7 Sample-file smoke tests

Use `notebooks/sample.csv` only for schema/sanity smoke tests:

- all required columns can be normalized or missing columns are reported clearly;
- parsing and sorting complete without crashing;
- state machine emits output on the sample;
- invariant reports are generated.

Do not report empirical spoofing conclusions from the sample.

---

## 14. Staged implementation checklist

### Stage 0: Approval and decisions

- Resolve open questions listed at the end of this document.
- Freeze first-run config: `top_n=10`, `agent_dimensions=["firm", "client_original"]`, all-event inclusion, output directory.

### Stage 1: Repository/package setup

Proposed files:

```text
src/spoofing_detection/lob/__init__.py
src/spoofing_detection/lob/schema.py
src/spoofing_detection/lob/enums.py
src/spoofing_detection/lob/config.py
tests/lob/
```

Tasks:

- Define required columns and optional columns in `schema.py`.
- Define enum mappings in `enums.py`, initially from audited observed codes with explicit unknown handling.
- Define a small dataclass/config object for top-N, required agent dimensions, input/output paths, and all-event inclusion flags.

### Stage 2: Schema inspection and enum audit

Proposed files:

```text
src/spoofing_detection/lob/audit.py
scripts/audit_lob_schema.py
```

Tasks:

- Read parquet/CSV schema with Polars.
- Compare required columns to actual columns.
- Print observed enum values and tooltip labels if present.
- Emit Markdown audit report under `outputs/lob_reconstruction/{run_id}/schema_audit.md`.
- Do not mutate book state in this stage.

### Stage 3: Synthetic tests first

Proposed files:

```text
tests/lob/test_synthetic_state_machine.py
tests/lob/conftest.py
```

Tasks:

- Write hand-computable synthetic event streams.
- Define expected active states and top-of-book states after each event.
- Run tests and verify they fail before state-machine implementation.

### Stage 4: Normalized event parser

Proposed files:

```text
src/spoofing_detection/lob/normalize.py
tests/lob/test_normalize.py
```

Tasks:

- Normalize column names and dtypes.
- Parse timestamps to nanosecond precision.
- Normalize side labels and event classes from numeric codes.
- Resolve both required agent identifiers: `firm_id = FIRMID` and `client_original_id = NMSC_ORIGINALCLIENTIDSHORTCODE`.
- Emit issue flags for unknown enums, missing price, missing agent, and inconsistent status/quantity fields.

### Stage 5: Active order state machine

Proposed files:

```text
src/spoofing_detection/lob/state.py
tests/lob/test_state_machine.py
```

Tasks:

- Implement `ActiveOrder` state object.
- Implement event mutation functions for New, Reload, Modify, Fill, Cancel, Refill, and special/unknown events.
- Keep mutation logic explicit and readable; do not over-vectorize.
- Pass all synthetic state-machine tests.

### Stage 6: Price-level depth aggregation

Proposed files:

```text
src/spoofing_detection/lob/book.py
tests/lob/test_book_aggregation.py
```

Tasks:

- Derive bid/ask price-level depth from active orders.
- Implement best bid/ask/mid/spread extraction.
- Implement full-aggregation check against any incremental aggregate.
- Test locked/crossed book flags.

### Stage 7: Top-N LOB event panel

Proposed files:

```text
src/spoofing_detection/lob/panel.py
tests/lob/test_lob_panel.py
```

Tasks:

- Emit one row per relevant event with raw normalized fields, pre-state, mutation, and post-state.
- Add top-N bid/ask levels before and after each event.
- Include issue flags and reconstruction quality metadata.

### Stage 8: Agent-level active liquidity panel

Proposed files:

```text
src/spoofing_detection/lob/agent.py
tests/lob/test_agent_state.py
```

Tasks:

- Compute event-agent pre/post bid/ask active visible and leaves quantities.
- Compute same-side/opposite-side quantities.
- Compute event-order distance from same-side best quote.
- Add optional distance-bucket aggregations.

### Stage 9: CLI runner and reproducible outputs

Proposed files:

```text
scripts/reconstruct_lob.py
src/spoofing_detection/lob/io.py
```

Tasks:

- Add CLI to run one input file or a directory of parquet files.
- Write Parquet outputs and Markdown/JSON metadata.
- Record source hashes, config, package versions, and output row counts.

### Stage 10: Real-file smoke tests and invariant reports

Proposed files:

```text
tests/lob/test_real_schema_smoke.py
src/spoofing_detection/lob/validate.py
```

Tasks:

- Run schema smoke tests on `notebooks/sample.csv`.
- Run small real parquet partitions in dry-run mode.
- Generate invariant reports without making empirical spoofing claims.

### Stage 11: Scalability and chunked processing

Tasks:

- Process each canonical partition independently.
- Add chunked streaming mode that carries active state across chunks within the same partition.
- Test chunked output equals full-file output.
- Only after correctness tests pass, optimize aggregation hot spots.

### Stage 12: Future model-feature extraction

Proposed files later:

```text
src/spoofing_detection/features/imbalance.py
src/spoofing_detection/features/sci.py
src/spoofing_detection/features/cps.py
```

Tasks later, not in first LOB implementation:

- Calibrate or configure `kappa`.
- Compute normalized distance-weighted imbalance.
- Compute SCI around fill events using explicit windows.
- Aggregate CPS by agent/session.
- Validate no look-ahead leakage.

---

## 15. Future exact queue-reconstruction extension

The first implementation must preserve queue compatibility by keeping:

```text
ORDERID
ORDERPRIORITY
ORDERPX
ORDERQTY
DISPLAYEDQTY
LEAVESQTY
SEQUENCETIME
HDR_APPLKEYSEQUENCENUMBER
HDR_HWMSEQUENCENUMBER
HDR_OFFSETID
ROW_NUMBER
EXECUTIONID
TRADETIME
PASSIVEORDER
AGGRESSIVEORDER
```

Future queue module design:

1. For each `(partition, side, price)`, maintain an ordered queue of active `ORDERID`s sorted by `ORDERPRIORITY` and tie-broken by event ordering.
2. On New/Reload, insert order at its reported priority.
3. On Modify, update quantity/price/priority. If price or priority changes, remove and reinsert in the appropriate queue.
4. On Fill, reduce the specific `ORDERID` reported by the row, not necessarily the front of the inferred queue until exact matching semantics are audited.
5. Compare observed fill order against inferred queue front to validate queue reconstruction.
6. Use `PASSIVEORDER`/`AGGRESSIVEORDER` after audit to distinguish passive resting fills from aggressive incoming orders.
7. Add queue-position features only after queue validation passes on synthetic and real diagnostic cases.

Queue-specific validation later:

- `ORDERPRIORITY` ordering is monotonic within price levels unless documented priority resets occur.
- Passive fills occur at or near the queue front under FIFO assumptions.
- Modifications that change price or increase quantity update priority according to venue rules.
- Queue-derived price-level depth equals active-order aggregation.

---

## 16. Implementation constraints

- Correctness over speed.
- Use a transparent Python event loop for the first state-machine implementation.
- Use Polars/Pandas for reading, normalization, sorting, grouping diagnostics, and Parquet writing.
- Do not over-vectorize the state machine until synthetic and real invariant tests pass.
- Write tests before scaling.
- Save intermediate artifacts in inspectable formats: Parquet for large tables, Markdown/JSON for reports and metadata.
- Preserve raw normalized event fields in outputs so downstream features can be audited.
- Do not silently drop unknown event types, missing agent IDs, null prices, locked/crossed states, or unseen prior orders; flag them.
- Keep modules separated: schema/enums, normalization, state mutation, book aggregation, agent aggregation, validation, CLI.

---

## 17. Decisions resolved and remaining open questions

### Resolved before first implementation

1. **Agent identity:** use both `FIRMID` and `NMSC_ORIGINALCLIENTIDSHORTCODE` as separate agent dimensions; do not use the composite `(FIRMID, NMSC_ORIGINALCLIENTIDSHORTCODE)` as the first-version agent key.
2. **GTC/GTD reload semantics:** treat `11 : GTC_GTD_Reload` as the reloading of existing active GTC/GTD orders into the opening active book, while validating for unseen later orders.
3. **Top-N depth:** use 10 levels on each side.
4. **Event inclusion:** use all events: keep all order events in normalized output and emit all order events in the LOB event-state panel; visible depth mutates only when an event has a visible resting-book component.
5. **Scaling:** use parquet price/quantity values exactly as stored; do not rescale.
6. **Enum mappings:** use the accepted provisional mappings from observed tooltip values and fail loudly on unknown numeric codes.

### Still open or requiring validation

1. **Official enum dictionaries:** observed tooltip mappings are sufficient for provisional implementation, but official dictionaries should replace them if available.
2. **Derived flags:** confirm which native bit fields produced `ORDERSTATUS`, `PASSIVEORDER`, `AGGRESSIVEORDER`, `DEAINDICATOR`, `INVESTMENTALGOINDICATOR`, and `EXECUTIONALGOINDICATOR`.
3. **ORDERSTATUS on fills:** until confirmed, use `LEAVESQTY` rather than `ORDERSTATUS` as the primary active-state field for fill events.
4. **Tick-size source:** first implementation should use price-unit and basis-point distances; exact tick distances need an official tick-size source or approved inference rule.
5. **Trade bust handling:** classify and flag if encountered; do not rollback book state until trade-bust semantics are documented.
6. **Session boundaries:** first implementation partitions by `TRADEDATE` plus instrument/market keys; market-phase/session markers should be used later if available.

---

## 18. Approval gate

Do not implement the pipeline until the user explicitly approves implementation. The main design decisions above are resolved. The first implementation should proceed with these validation defaults:

1. accepted provisional enum mappings from observed tooltip values;
2. `LEAVESQTY` as primary post-fill active quantity pending `ORDERSTATUS` audit;
3. price-unit/basis-point distances for the first version, with tick-distance deferred;
4. trade-bust rows, if encountered, flagged and non-mutating until documented.
