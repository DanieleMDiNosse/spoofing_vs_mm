/plan

We need to implement the spoofing-surveillance model described in `spoofing.tex` / `spoofing.pdf`.

The first technical step is to reconstruct an event-sourced limit order book state from the order-event data. This is correctness-critical, so do not jump directly to implementation. First produce a detailed plan, identify ambiguities, and define validation tests.

Scope of this task: planning only. Do not implement the pipeline yet unless I explicitly approve the plan.

Input documents and data:

* `docs/keep_cols_data_dictionary.md`
* `docs/tracciato_database_tabella_ordini.docx`
* `spoofing.tex` / `spoofing.pdf`
* `sample.csv`, only as a schema and sanity-check example. Do not infer empirical conclusions from the sample.

Main objective:
Design a reproducible Python pipeline to reconstruct a price-level and agent-level LOB event panel. The first version should reconstruct active order state, top-N visible book depth, and agent-level active liquidity. Do not attempt exact queue reconstruction in the first implementation.

Important architectural requirement:
Even though exact queue reconstruction is deferred, the pipeline must remain queue-compatible. Preserve all fields needed for a later queue-level module, especially:

* `ORDERID`
* `ORDERPRIORITY`
* `ORDERPX`
* `ORDERQTY`
* `DISPLAYEDQTY`
* `LEAVESQTY`
* `SEQUENCETIME`
* `HDR_APPLKEYSEQUENCENUMBER`
* `HDR_HWMSEQUENCENUMBER`
* `HDR_OFFSETID`
* `ROW_NUMBER`
* `EXECUTIONID`
* `TRADETIME`
* `PASSIVEORDER`
* `AGGRESSIVEORDER`

Do not collapse the book directly into price-level aggregates without keeping an order-level active state. The internal state should be order-level, keyed by `ORDERID`; price-level depth and top-N book variables should be derived by aggregation.

Target output:
One row per relevant order-book event, sorted in deterministic matching-engine order, containing:

1. raw normalized event fields;
2. event classification;
3. pre-event LOB state;
4. post-event LOB state;
5. pre/post agent-level active liquidity state;
6. top-N visible depth by side;
7. fields needed later to compute the spoofing-model variables.

Model-driven requirements:
The spoofing model requires, at minimum:

* best bid, best ask, mid-price, and spread before and after each event;
* active visible depth by side and price;
* top-N book levels before and after each event;
* agent-level active bid and ask volume;
* distance of each active agent order from the same-side best quote;
* active agent volume by side and distance bucket;
* event states around executions so that SCI can later be computed without look-ahead leakage;
* fill/cancel sequences that allow us to identify posturing, opposite-side execution, and post-execution cancellation.

For the first implementation, exact FIFO queue reconstruction is not required. However, the plan should explain how a future queue module could be added using `ORDERPRIORITY` and the order-level state.

Planning instructions:

1. Inspect the documents and sample schema.

   * Map every field needed for reconstruction and for the spoofing model.
   * Explicitly list fields that are native, renamed, derived, or ambiguous.
   * Do not silently invent enum meanings. If an enum is not fully documented in the available files, flag it as an open question.

2. Assess reconstruction feasibility.

   * Verify whether the available data is sufficient for exact LOB reconstruction.
   * Check whether we have complete event coverage from the start of each trading day/session for each instrument.
   * If not, distinguish exact reconstruction from partial reconstruction.
   * State what can still be reconstructed reliably from the available data.

3. Define canonical partition keys.

   * At minimum consider `TRADEDATE` and `ISIN`.
   * Also evaluate whether fields such as `SYMBOLINDEX`, `EMM`, `MIC`, and `MARKETCODE` are needed if available.
   * Define the unit over which the event stream must be processed independently.

4. Define deterministic event ordering.

   * Candidate ordering fields include `SEQUENCETIME`, `HDR_APPLKEYSEQUENCENUMBER`, `HDR_HWMSEQUENCENUMBER`, `HDR_OFFSETID`, `BOOKIN`, `BOOKOUTTIME`, `TRADETIME`, and `ROW_NUMBER`.
   * Prefer matching-engine ordering fields.
   * Define the final sorting key and tie-breakers.
   * Explain how to test that ordering is stable and deterministic.

5. Define normalized event semantics.

   * Define how to classify:

     * new orders;
     * modifications / cancel-replace events;
     * partial fills;
     * full fills;
     * cancellations / kills;
     * rejects;
     * trade busts or trade cancellations, if present;
     * market orders with null price;
     * iceberg orders and displayed quantity.
   * Explain how `ORDERSTATUS`, `LEAVESQTY`, `LASTSHARES`, `LASTTRADEDPX`, `PASSIVEORDER`, and `AGGRESSIVEORDER` should be used.
   * Explicitly flag unclear cases.

6. Define the in-memory reconstruction state.

   * Maintain active orders keyed by `ORDERID`.
   * Each active order should retain at least:

     * side;
     * price;
     * leaves quantity;
     * displayed quantity;
     * original quantity;
     * order type;
     * time in force;
     * agent identifiers;
     * order priority;
     * last update event time;
     * current active/inactive status.
   * Derive price-level book depth from active orders.
   * Derive top-N bid/ask levels from the price-level book.
   * Derive agent-level active bid/ask volume from active orders.
   * Preserve enough metadata for future queue reconstruction, but do not implement exact queue logic yet.

7. Define agent identity alternatives.

   * The model should support multiple possible definitions of “agent”, for example:

     * `FIRMID`;
     * `NMSC_ORIGINALCLIENTIDSHORTCODE`;
     * `MSC_EVENTCLIENTIDSHORTCODE`;
     * `NMSC_ORIGINALEXECWFIRMSHORTCODE`;
     * `MSC_EVENTEXECWFIRMSHORTCODE`;
     * `NMSC_ORIGINALINVESTDECISWFIRMSHORTCODE`;
     * combinations of these fields.
   * The plan should recommend a default agent key and explain trade-offs.

8. Define output schemas.
   Create schemas for:

   * normalized event table;
   * active order state snapshots, if needed for debugging;
   * LOB event-state panel;
   * agent-state panel;
   * later model-feature panel for imbalance, SCI, and CPS.

9. Define first-version LOB features.
   The first implementation should produce:

   * `pre_best_bid`, `pre_best_ask`, `pre_mid`, `pre_spread`;
   * `post_best_bid`, `post_best_ask`, `post_mid`, `post_spread`;
   * top-N price and volume levels on bid and ask;
   * event side, price, quantity, displayed quantity, leaves quantity;
   * event agent identifier;
   * agent active bid volume before/after;
   * agent active ask volume before/after;
   * agent active same-side and opposite-side volumes;
   * distance of event order from same-side best quote;
   * optional distance buckets for active agent liquidity.

10. Define future spoofing features, but do not implement yet.
    Explain how the reconstructed panel will later support:

* normalized distance-weighted imbalance;
* pre-fill and post-fill imbalance;
* SCI at execution events;
* CPS aggregated by agent and session;
* post-fill cancellation detection;
* separation of spoofing-like behavior from market-making behavior.

11. Define validation tests and invariants.
    Include at least:

* deterministic sorting;
* no negative active quantity;
* fills reduce active quantity consistently;
* cancels remove or reduce active orders consistently;
* active book volume equals aggregation of active orders;
* best bid is not above best ask except in explicitly documented locked/crossed cases;
* active state is consistent with `ORDERSTATUS` and `LEAVESQTY`;
* repeated processing gives identical output;
* chunked processing gives the same result as full-file processing;
* synthetic hand-computable event sequences pass exactly;
* sample-file smoke tests run successfully.

12. Define staged implementation plan.
    Propose stages:

* schema inspection and field normalization;
* enum audit;
* single-ISIN/single-day prototype;
* synthetic unit tests;
* active order state machine;
* price-level depth aggregation;
* top-N LOB panel;
* agent-level active liquidity panel;
* model-feature extraction;
* scalability and chunked processing;
* optional future exact queue reconstruction.

13. Define implementation constraints.

* Use correctness over speed.
* Prefer a transparent event loop for the first implementation.
* Use Polars/Pandas for reading, normalization, sorting, and writing.
* Do not over-vectorize the state machine before the logic is validated.
* Write clear tests before scaling.
* Save intermediate artifacts in inspectable formats, preferably Parquet for large tables and Markdown for documentation.
* Keep code modular enough that queue reconstruction can be added later.

Deliverable:
Create or update `docs/lob_reconstruction_plan.md` containing:

* assumptions;
* reconstruction feasibility assessment;
* field mapping;
* event ordering;
* event semantics;
* state-machine design;
* agent identity design;
* output schemas;
* validation strategy;
* staged implementation checklist;
* future queue-reconstruction extension;
* open questions before implementation.

At the end, print a concise terminal summary with:

* whether exact reconstruction seems feasible;
* the proposed first implementation scope;
* the main risks;
* the open questions that require my decision before coding.
