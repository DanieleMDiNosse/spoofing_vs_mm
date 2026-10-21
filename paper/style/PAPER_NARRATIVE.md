# PAPER_NARRATIVE.md

## Purpose

This file governs the scientific narrative of the spoofing-detection paper.

It does not define the numerical values of thresholds, field mappings, timing rules, or implementation details. Those belong in `METHOD_SPEC.md`.

It should be read together with `STYLE_PROFILE.md`.

---

## Central Narrative Problem

Start from the observable economic sequence and the ambiguity that makes it difficult to interpret.

The records may show:

1. displayed liquidity;
2. an execution on the opposite side;
3. subsequent withdrawal of displayed liquidity.

That sequence is compatible with more than one explanation.

The manuscript must therefore distinguish clearly between:

- the hypothesized strategy;
- the observable sequence in the records;
- the model's operational reconstruction of that sequence;
- the evidence used to prioritize or compare candidate cases.

Do not write as if the records directly reveal manipulation or intent.

---

## Explanatory Order

Explain each part of the model by the ambiguity it resolves.

Before introducing a model component, make clear whether it is intended to:

- describe the state of the order book;
- reconstruct an economically meaningful sequence;
- impose a screening or compatibility condition;
- quantify the strength of a pattern;
- prioritize episodes for review;
- compare candidate behavior with an alternative population or benchmark.

Do not introduce all acronyms or technical objects at once.

The reader should understand the need for each object before encountering its formal definition.

---

## Hypothetical Sequences

Use simple, explicitly hypothetical sequences when they help the reader follow:

- participants;
- book sides;
- timing;
- order placement;
- execution;
- subsequent cancellation or withdrawal.

A hypothetical example must illustrate the verified model.

It must not:

- introduce a new assumption;
- imply an empirical frequency not established by the data;
- masquerade as a real observed episode unless it is explicitly identified as one.

---

## Messages, Executions, and Episodes

Explain why observations must be grouped before introducing the grouping terminology.

Preserve the distinction between:

- individual messages or book events;
- executions;
- reconstructed episodes.

Once fill aggregation has been defined, use **execution** in ordinary prose when the technical distinction is not important.

Reserve **execution cluster** for passages where that distinction matters scientifically or operationally.

Do not allow implementation-level grouping terminology to dominate the narrative.

---

## Observable Compatibility vs Manipulation

Keep the distinction between compatibility with the hypothesized sequence and evidence of manipulation explicit.

The detector is descriptive unless a stronger claim is separately justified.

Do not turn:

- a compatible sequence;
- a high score;
- a screening condition;
- a descriptive association;

into a causal model, proof of intent, or confirmed spoofing event.

State the general scope limitation clearly in the introduction and recall it in the conclusions.

The abstract may state the limitation briefly because it must stand alone.

Do not repeat a generic disclaimer in every model paragraph, figure caption, or result.

Use local qualifications only when they constrain the interpretation of a specific result.

---

## Passive and Aggressive Roles

Treat the passive/aggressive execution-role decoding used in this project as confirmed by CONSOB, as reported by the author.

Define the economic meaning of the roles once in the data or methodology discussion.

Do not repeatedly describe this mapping as uncertified or unverified.

This confirmation concerns execution roles only.

It does not, by itself, validate separate assumptions about clock origin, event timing, or attribution.

Those assumptions must be documented independently in `METHOD_SPEC.md`.

---

## Scientific Questions Before Empirical Blocks

Organize the empirical discussion around scientific questions and findings, not around:

- script order;
- pipeline stages;
- output files;
- table-generation order.

Before presenting a comparison, explain what that comparison can establish.

Then present the evidence.

Then state what follows from it.

---

## Comparing Explanations

When alternative explanations are relevant, present them explicitly.

For each explanation:

1. state the mechanism or interpretation;
2. state what observable pattern it predicts;
3. identify the common criterion used to compare it with alternatives;
4. evaluate the evidence against that criterion;
5. state what remains unresolved.

Do not present one explanation in mechanistic terms and another only as a residual category.

---

## Result Narrative

For each empirical block, use the following progression when appropriate:

**scientific question -> comparison -> finding -> interpretation -> limitation -> implication**

State the finding before pointing to the figure or table.

Describe variation across instruments, samples, or settings when it matters to the scientific conclusion.

If differences may arise from alternative definitions, aggregation levels, or measurement conditions, discuss those possibilities before attributing the discrepancy to behavior.

---

## Conclusion Narrative

The conclusion should synthesize the evidence by scientific question or mechanism.

Do not repeat the order of figures or subsections.

Move from:

1. what the observable sequence captures;
2. what the model successfully distinguishes or prioritizes;
3. what the empirical comparisons show;
4. what remains descriptive rather than causal;
5. what additional evidence would be required for stronger claims.

The final discussion should preserve the distinction between pattern compatibility and proof of manipulation or intent.

## Coverage, market scale, and profile dynamics

Keep the author's objective of distinguishing spoofing-like behavior from ordinary liquidity management. The multiscale comparison supports that objective by separating three questions, not by replacing it with a cancellation-ranking exercise:

1. Is the initial display followed by an opposite-side execution and assigned removal in the observed records?
2. How substantial is that removal relative to the participant's display and to available market depth?
3. Does the relative liquidity profile show a rapid positive transition, and when?

Motivate each comparison before its formula. Use market-normalized WMSCI as the sole withdrawal ranking in the active manuscript. Explain its display/execution, removal/execution and removal/market components in reader order; the physical removal share remains a diagnostic, not a second ranking score. Explain horizon and age treatment as exploratory observation choices whose consequences can be inspected, not externally validated behavioral constants. Preserve data-informed selection provenance privately without implying that the settings were prespecified.

Separate retaining a candidate from giving it review priority. Order every candidate by descending market-normalized WMSCI within instrument, with deterministic identity-based ties and no score-percentile cut. Finite zero scores stay at the end; unsupported scores remain explicit unranked records. Show J alongside each execution as a diagnostic, never as a second filter, ordering key, tie-breaker or composite score. A low rank, missing profile, or zero rapid increment must not disappear rhetorically into a claim that the physical sequence was not observed. Conversely, a positive profile increment alone does not prove cancellation of the initial display. A zero rapid-increment diagnostic does not imply an unchanged profile. Removing score cuts does not remove structural candidature, timing, lifecycle or observation-support rules.

Use diagnostic disagreements constructively: locate which assumption or observable creates the disagreement, develop the strongest supported interpretation, then name the event-level check that could distinguish replacement, fills, rank changes, and changing market-depth shares. Do not treat another detector's alert as a more authoritative manipulation label.

Keep price evidence attached to its explicitly reconciled population. The rank-only presentation requires actual price diagnostics for every primary raw candidate, including zero-score rows, with no WMSCI, J or price-sign selection. Do not relabel the former selected-list prices as full-population results. Historical selected lists, their price results and the hard-age/H1 profile cohort remain archived as provenance, not active rank-only evidence. Present corroborating and noncorroborating price observations together as differing evidential strength, not as either proof of the proposed mechanism or a failure that erases useful behavioral evidence. All signs and missing outcomes retain the full instrument-level denominator; whole-population price summaries do not establish ranking performance.

## Participant-level grouping

For participant-level results, distinguish two overlapping perspectives: identified clients using the existing canonical client key, and intermediaries using the original firm identifier of every execution, including both client-identified and firm-only activity. Firm fallback is an attribution level, not the entire intermediary population. Compare each group's candidate count with all its observed executions within instrument. Do not sum the two perspectives, add execution WMSCI values into a participant score, or sum overlapping display quantities into unique physical volume. Keep stable report-local client/intermediary aliases, all zero/missing execution diagnostics and complete CSV rows for zero-candidate groups. Explain recurrence as repeated observed activity, not independent evidence or manipulative intent. Verify all child fills of an execution belong to the attributed intermediary before using its anchor firm.

## Main age specification and supplementary alternatives

Use exponential age weighting without an age cutoff as the primary specification; the current half-life and main horizon are recorded as author decisions in `METHOD_SPEC.md`. Explain its purpose positively: retain the observable physical sequence while assigning greater importance to recent display. Present the hard age limit and all-age unit weighting in the appendix, with longer-half-life sensitivity there as well. Main tables, rankings, diagnostic scores, and candidate counts must use the chosen primary cell, not silently relabel a parameter-grid union or unit-weight ranking.

Do not mention or cite internal-detector alerts, their coverage, or externally flagged cohorts in the manuscript or displayed captions. Preserve their private selection provenance in the method specification and review notes. Describe the parameter choice and illustrative cases honestly as exploratory on the inspected samples; do not imply independent validation, prespecification, or an optimal decay estimate. The invariance of raw membership across age kernels does not identify an optimal half-life.

Retain economic interpretation of the primary market ranking. If an execution's position changes relative to unit weights, explain its actual score and physical quantities rather than repeating an earlier selected-list description. Keep numerical score support distinct from raw membership; positive-score support is not the new ranking denominator because zero scores are retained. The author's choice of one ordering and accompanying diagnostics simplifies the argument; it is not proof of superior correlation, classification accuracy or external-alert prioritization. Retain missing/zero/negative price outcomes and separate complete reversion from observed-subset averages. Do not infer a within-role benefit from pooled composition differences, a causal effect from sampled market quotes, or improved review yield without independent outcomes. No self-score formulas, comparison plots, selected-list enrichment claims, conditional-J stages or historical self-selected trajectories belong in the active paper; preserve them in immutable historical outputs instead.
