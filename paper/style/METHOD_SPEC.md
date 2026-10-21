# METHOD_SPEC.md

## Status

**Status: incomplete specification template.**

Empty entries mean **not yet documented**, not **not applicable**.

Until a section has been populated and verified against the analysis version actually reported in the manuscript, this file does **not** supersede existing methodological sources, validated analysis bundles, code snapshots, author decisions, or project documentation.

An agent must never infer a missing rule from an empty field, and must never fill a gap silently from the current codebase when the manuscript may rely on a different validated analysis version.

## Purpose

This file is the designated location for formal methodological specifications that should not be embedded in `STYLE_PROFILE.md` or `PAPER_NARRATIVE.md`.

A specification becomes authoritative only when its entry is populated and explicitly marked with its source, applicable version or policy, and verification status.

Populate it only from verified project sources: code, data documentation, validated analysis notes or bundles, or confirmed methodological decisions.

Do not infer or invent missing values.

The manuscript may explain verified specifications in reader-friendly language, but its scientific content must remain consistent with the methodological source applicable to the reported analysis.

## Required metadata for every populated specification

For every completed definition, rule, threshold, mapping, or convention, record:

- **Source:** exact file, document, analysis bundle, code location, or author decision from which the specification is taken;
- **Applicable version / policy:** the analysis version, dataset version, bundle, commit, date, or policy to which the specification applies;
- **Status:** one of `verified`, `operational assumption`, `author decision`, or `unresolved`;
- **Verification note:** how the entry was checked, when useful.

`verified` means that the entry has been checked against the source applicable to the analysis reported in the manuscript. It does not mean merely that the rule exists in the current implementation.

---

## 1. Observational Units

Document the exact observational units used at each stage.

### Raw event / message
- Definition:
- Source table or file:
- Required identifiers:
- Timestamp field:
- Side convention:
- Agent/client/broker identifier fields:

### Execution
- Definition:
- Rule for aggregating fills, if applicable:
- Execution identifier:
- Passive/aggressive role fields:

### Episode
- Definition:
- Start condition:
- End condition:
- Grouping key:
- Whether episodes may overlap:
- Treatment of multiple executions:

---

## 2. Passive/Aggressive Execution Roles

Project narrative assumption:

- The passive/aggressive decoding used in this project is treated as confirmed by CONSOB, as reported by the author.

Record here the verified technical mapping.

- Passive-side source field(s):
- Aggressive-side source field(s):
- Buy/sell convention:
- Any venue-specific encoding:
- Validation source or note:

This confirmation concerns execution roles only. Keep separate any assumptions about clock origin, event timing, or attribution.

---

## 3. Clock and Timing Conventions

Document all scientifically relevant timing assumptions.

- Primary event timestamp:
- Timestamp resolution:
- Clock origin:
- Synchronization assumptions:
- Whether order, trade, and cancellation timestamps share the same clock:
- Handling of equal timestamps:
- Ordering rule for simultaneous events:
- Time-zone handling, if relevant:

### Analysis windows

For each window used in the method, record:

- Name:
- Source:
- Applicable version / policy:
- Status:
- Scientific purpose:
- Start relative to anchor event:
- End relative to anchor event:
- Inclusive/exclusive boundaries:
- Units:
- Sensitivity/robustness variants:

Do not place numeric timing values in `PAPER_NARRATIVE.md`.

---

## 4. Book-State Construction

Document how the relevant state of the limit order book is reconstructed.

- Book depth used:
- Price levels included:
- Treatment of hidden/iceberg liquidity, if observable:
- Treatment of modifications:
- Treatment of partial fills:
- Treatment of cancellations:
- Treatment of invalid/corrupt events:
- State-reset rules:
- Any queue-position assumptions:

If exact queue reconstruction is not part of the verified method, state that explicitly rather than implying it.

---

## 5. Candidate Sequence Reconstruction

Document the exact operational definition of a candidate sequence.

For every condition, specify:

- condition name;
- economic rationale;
- formal rule;
- required fields;
- threshold or window;
- denominator, where applicable;
- whether the condition is mandatory or descriptive;
- treatment of missing data.

### Sequence components

Record the verified rules for:

1. displayed liquidity;
2. opposite-side execution;
3. subsequent withdrawal/cancellation;
4. any additional screening conditions;
5. any exclusion conditions.

---

## 6. Grouping and Fill Aggregation

Document why and how multiple records are grouped.

- Fill aggregation rule:
- Maximum time gap, if any:
- Required shared identifiers:
- Price condition:
- Side condition:
- Agent/client condition:
- Treatment of partial executions:
- Treatment of cross-venue records, if applicable:

State explicitly when the manuscript may use the ordinary term **execution** and when the technical term **execution cluster** is required.

---

## 7. Primary Metrics and Scores

For every metric used in the paper, record the authoritative definition.

### Metric template

**Name:**
- Source:
- Applicable version / policy:
- Status:
- Symbol:
- Scientific purpose:
- Formal definition:
- Unit/range:
- Numerator:
- Denominator:
- Conditioning set:
- Required data:
- Missing-value treatment:
- Interpretation of high values:
- Interpretation of low values:
- Boundary cases:
- Whether calibrated:
- Whether descriptive, predictive, or screening-only:

Create a separate subsection for each metric.

---

## 8. Screening Conditions

Record every screening condition separately from continuous scores.

For each condition:

- Name:
- Source:
- Applicable version / policy:
- Status:
- Formal condition:
- Rationale:
- Threshold:
- Whether threshold is calibrated or heuristic:
- Population to which it applies:
- Interaction with other conditions:
- Missing-data rule:

If a strict gate exists, distinguish clearly between:

- the continuous/descriptive metrics;
- the gate conditions;
- the scientific role of each.

Do not let implementation order determine narrative importance.

---

## 9. Comparison Populations and Controls

Document every comparison population used in the results.

For each population:

- Name:
- Source:
- Applicable version / policy:
- Status:
- Scientific question:
- Inclusion rule:
- Exclusion rule:
- Matching/conditioning variables:
- Time period:
- Instrument coverage:
- Agent coverage:
- Denominator:
- Known coverage differences:

Explain in the manuscript what each comparison can establish before interpreting its values.

---

## 10. Thresholds and Calibration

Maintain one authoritative table of thresholds.

| Parameter / threshold | Value | Unit | Scientific role | Source | Applicable version / policy | Status | Robustness variants |
|---|---:|---|---|---|---|---|---|
| TODO | TODO | TODO | TODO | TODO | TODO | unresolved | TODO |

Do not copy threshold values manually into multiple specification files.

If a value changes, update the authoritative source first.

---

## 11. Missing Data and NaN Handling

Document all rules that can affect sample composition or metric values.

- Required fields:
- Drop rules:
- Imputation rules:
- NaN propagation:
- Zero-denominator handling:
- Empty-window handling:
- Instruments/days/agents excluded:
- Whether exclusions differ across metrics:

Any rule that changes the effective population is scientifically relevant and must be reported in the paper or reproducibility material.

---

## 12. Attribution and Agent Identity

Document how actions are attributed to participants.

- Client identifier:
- Broker identifier:
- Agent identifier used in the analysis:
- Whether identifiers are stable across days:
- Whether identifiers are stable across instruments:
- Treatment of missing identifiers:
- Treatment of broker/client hierarchy:
- Any known ambiguity:

Do not infer intent from identity persistence or sequence compatibility alone.

---

## 13. Denominators

List every denominator used in rates, fractions, normalized scores, or comparisons.

For each denominator:

- Metric/result:
- Population:
- Time scope:
- Instrument scope:
- Agent scope:
- Exclusions:
- Reason for choice:

A denominator change is a methodological change, not a cosmetic one.

---

## 14. Robustness and Sensitivity

Record all verified robustness checks.

- Alternative time windows:
- Alternative thresholds:
- Alternative grouping rules:
- Alternative comparison populations:
- Instrument-level checks:
- Agent-level checks:
- Subperiod checks:
- Coverage restrictions:

For each check, state which scientific assumption it probes.

---

## 15. Scope of Inference

The method identifies sequences or observations that are compatible with the hypothesized pattern according to the verified operational rules.

Unless separately established, it does not by itself provide:

- proof of manipulation;
- proof of intent;
- a causal estimate;
- a complete behavioral model.

Record here any additional evidence or assumptions required for stronger claims.

---

## 16. Reproducibility Mapping

Keep implementation details here or in repository documentation when they are necessary for reproducibility.

### Data sources
- Input dataset(s):
- Version/date:
- Relevant tables:
- Relevant fields:

### Code
- Main pipeline:
- Metric implementation:
- Episode reconstruction:
- Figure generation:
- Table generation:

### Output mapping
- Scientific object:
- Internal column/file:
- Manuscript location:

Do not use internal names as substitutes for scientific explanations in the main text.

---

## 17. Verification Checklist

Before a manuscript revision is considered scientifically consistent, verify:

- [ ] observational units match the implementation snapshot applicable to the reported analysis;
- [ ] timing conventions match the implementation snapshot applicable to the reported analysis;
- [ ] passive/aggressive mapping matches the confirmed project convention;
- [ ] grouping rules match the implementation snapshot applicable to the reported analysis;
- [ ] all thresholds match the authoritative source;
- [ ] all denominators are explicit and correct;
- [ ] missing-data rules match the effective sample;
- [ ] metric definitions match the code snapshot applicable to the reported analysis;
- [ ] comparison populations match the reported results;
- [ ] no descriptive score is described as a causal or behavioral law;
- [ ] no candidate sequence is described as confirmed spoofing without independent justification.

## 18. Verified multiscale application in spoofing_simple

This populated entry records the verified native multiscale computation. Its former dual-score and percentile presentations are superseded by section21 for the active manuscript; frozen producers still retain both columns and old priority flags for compatibility. The unpopulated template above remains incomplete; it must not be used to infer rules. Sources: `configs/withdrawal_multiscale_v1.json`, `docs/withdrawal_multiscale_v1.md`, frozen V4 identity in `.hermes/checkpoints/withdrawal_multiscale_20261013_103507/`, and the completed prepare/analysis/report artifacts listed below. Status: verified against the reported artifacts, except assumptions and author decisions explicitly distinguished here.

- Main sources: `outputs/withdrawal_multiscale_20261013_v4/RISANAMENTO`, the corresponding `NEXI` root, and `outputs/withdrawal_multiscale_20261013_v4_ferrari_extended/FERRARI`. Resource budgets differ for Ferrari, not scientific settings.
- Units and identity: distinct executions keyed by instrument, partition, participant, and execution identifier. Twenty parameter cells reuse each execution. Counts: 48,481 executions, 65,471 source fills, 969,620 cells, 6,924 distinct candidate-union executions. Verified from authenticated campaign summary and native population tables. Union is not the longest-horizon cell.
- Primary presentation decision (author decision, 2026-10-14): all observable ages, exponential half-life b=30 s, H=10 s. This selects the existing exp_h30/H10 cell of the verified V4 campaign, not a new detector run or config mutation. Native primary counts are 3,526 Ferrari, 2,393 Nexi, 995 Risanamento: 6,914, distinct from the grid union 6,924. Verified independently in `paper/reviews/20261014_152806_kernel30_simple/primary_native_evidence.json`. Hard age<=30 s, all-age unit weights, and longer half-lives remain supplementary. Former selected-list prices are preserved in sections19and20; section21 requires actual whole-candidate prices for the rank-only presentation. Hard-cap/H1 prices remain historical provenance only.
- Initial display: same recorded participant, opposite execution side, active immediately before first fill, within ten occupied MARKET price levels. P is raw initial visible shares. Q is original executed shares. D is all-participant opposite-side top-ten depth from that exact pre-first-fill snapshot, including the participant itself. D is distinct from the two-sided calibration denominator.
- Age: first-fill SEQUENCETIME minus lifecycle first-observed SEQUENCETIME, in integer ns. Modify does not reset lifecycle age. Reload origins denote observed lifecycle origin, not a certified first economic submission. Policies: age <=30 s with unit weights; all observable ages with unit weights; all ages with shared exponential half-lives 30/120/300 s. These choices are author-approved exploratory settings, not estimated optimal parameters.
- Withdrawal: explicit Cancel of an initial order from the same lifecycle, canonical source index strictly after last fill, delay from last-fill TRADETIME in [0,H), H=1/2/5/10 s. Eligibility and exclusive assignment are recomputed per setting. Latest last source index wins, then earliest first source index and stable execution ID. Cumulative attributed removal per initial lifecycle is capped at its pre-fill quantity; Modify is not Cancel. U is gross initial-order removal, not net displayed liquidity loss.
- Shared age weights: age frozen before first fill; g(a)=exp(-ln(2)*a/b), same g multiplies P and the corresponding attributed U. Q, D, depth kernel and MSCI are not discounted. Self W=log1p(P_g/Q)*log1p(U_g/Q)*(U_g/P_g). Market W uses U_g/D instead of U_g/P_g; market=self*(P_g/D). Scores are dependent normalizations, not probabilities. P_g=0 gives self W=0 and undefined removal share. Missing/zero D gives unsupported market score, without denominator epsilon. Score or weight underflow never removes a raw candidate.
- Profile: unchanged empirical H1 side/rank depth weights, unchanged MSCI=SCI/2+C_opposite-C_same. Fixed last retained actor baseline strictly before first fill. Exact post-last-fill state at offset zero; supported canonical forward carry at later targets. Closed [0,H] grid in SEQUENCETIME, 10 ms spacing, positive MSCI increment at 50 ms lag. Complete finite J includes zero; incomplete profiles remain missing. Valid profile support is represented source coverage, not independent feed completeness.
- Candidate: P_raw>0, U_raw>0, Q>0 per setting. No W percentile, J positivity/completeness, U/Q threshold or price condition in raw membership.
- Native stored priorities: separate within-instrument and setting top 10% finite positive self/market W candidate lists, then conditional top 10% complete finite J within each list. These flags are historical views, not the active rank-only selection or price population (section21). Ceil counts; deterministic ties by execution ID, then partition and participant. Candidates outside every priority flag remain in population. Common-horizon comparisons use the union of candidates in at least one H and valid profiles at every H, not withdrawal positivity at every H.
- Role interpretation: passive is a resting non-crossing limit order and aggressive is a market/crossing limit order. Status: author-provided CONSOB confirmation. Historical metadata saying the mapping is unverified is stale and must not override this confirmation. This does not confirm a common clock origin.
- Clock: ns-integer arithmetic and provisional common origin of sequence/trade clocks. Status: operational assumption, not precision or synchronization certification. TRADETIME's observed millisecond grid is not effective ns precision. Cancellation and profile horizons have equal durations but distinct clock anchors.
- Historical reference prices (not active primary tables): H1, age<=30 s, self priority only. Exact full-key parity: first lists 192/127/10 and conditional J lists 20/13/1 (Ferrari/Nexi/Risanamento), J equality, W agreement within 1e-12, raw quantity and assigned cancellation parity. Complete REV requires every assigned positive-quantity Cancel supported at two seconds. Both positive FPM and complete REV: 3/20, 3/13, 1/1. Source and reconciliation evidence: `paper/reviews/20261014_091222_multiscale_simple/baseline_*` and frozen `outputs/new_kernel_h1_top10_study` / `outputs/new_kernel_h1_top10_paper`. These preserved outputs are compatibility references, not newly computed prices across the parameter grid. Their strict floating-point signs are not transferred into the active numerical-zero convention.
- Private selection provenance (author decision, 2026-10-14): choose the shortest TESTED half-life giving full raw coverage of timed internal-detector reference windows at H10; all three tested exponential half-lives give the same 34/34 union-window coverage (36/36 timed source records). Therefore 30 s is shortest in the tested grid, not a globally identified minimum or an empirically fitted optimum. Raw membership and assignment do not depend on these exponential weights; coverage cannot identify their half-life. H10 is the first fully covering TESTED horizon. Source: authenticated `outputs/consob_horizon_window_coverage_20261014_145940`, baseline decision receipt in this revision's review directory. The choices are exploratory on inspected data, not prespecified or held-out validation. Per author request these alert comparisons are not included or cited in the manuscript; private provenance remains intact.
- Diagnostic selection: post-hoc examples originally found through externally flagged participant/window disagreements, not confirmatory case sampling. Case R is original client 17295's Risanamento execution; case N is original client 31425's Nexi execution. Case labels in reader assets do not replace native identities. R's alert is date only, precluding exact timed alert recall. In the paper, describe the cases as post-hoc diagnostic illustrations, without external-alert attribution or claims of independent case selection. Do not describe the parameter selection as independent of inspected data. Case N at exp_h30/H10 is market-W and conditional-market-J priority but not self-W priority; this differs from its unweighted ranking.
- Historical dual-score presentation: `scripts/generate_withdrawal_multiscale_manuscript.py`, former `paper/generated/withdrawal_multiscale_20261014_primary_prices_v1` package, now preserved unchanged. Authenticated source markers/artifacts, table/figure CSV companions, source hashes and versions. Fresh output roots required; earlier presentation packages, including V4 and kernel30 V1/V2, remain preserved. Calibration figures stay on the unchanged authenticated H1 calibration. The generator performs no new replay or statistical inference. Case plots use identical J limits with margin below zero; appendix tables remain with their source-order context. Historical rank comparisons use actual stored exp_h30 flags and disclose differing positive-score support in Risanamento (984 self /978 market at H10, top99/98), not the all-age unit-weight support or an invented matched-budget reranking. Section20 records the former market-only percentile presentation; section21 defines the active complete ranking.
- Validation scope: all parameter-cell ledger and priority flags independently checked in the campaign. Profile sampler checks every computed profile; separate final independent numerical recheck samples 32 executions per instrument. Do not claim every profile independently rederived. No manipulation labels, independently identified market makers, held-out classification evaluation, false-positive rate, causal price effect, or optimal age/horizon calibration are supplied.

## 19. Historical dual-ranking primary price diagnostics

Status: verified completed retained-cache computation and downstream numerical-sign artifacts, preserved as source evidence. Its906-execution dual-ranking presentation and section20's691-execution market-only presentation are historical; section21 defines the active complete ranking and whole-candidate price scope. Sources: `scripts/study_primary_kernel_prices.py`, `scripts/refine_primary_price_signs.py`, completed `outputs/primary_kernel30_h10_prices_20261014_213056` and its separate `_numeric_v2` root, and independent endpoint/code review in `paper/reviews/20261014_213056_primary_prices/price_code_review.md`. No raw replay or detector/config change. Historical price parity is compatibility evidence, not relabeling selected-list outcomes as whole-candidate evidence.

- Population: union of native exp_h30/H10 self/market W and conditional-J flags, frozen before prices in `expected_price_memberships.parquet`: 906 full-key executions (480 Ferrari, 308 Nexi, 118 Risanamento). Twelve overlapping list summaries. Self W/J sizes 353/36, 240/24, 99/10; market 353/36, 240/24, 98/10. J stays nested in its own W list. All signs and missing observations retain the original list denominator.
- Endpoint policy: market midprices sampled at participant-retained states, not the participant's own bid/ask or necessarily fresh global quotes. Placement is the greatest original candidate posting source index, strictly before first fill; all candidate placement observations must be supported. Pre-state is latest source predecessor, without skipping an invalid midpoint. Post-cancellation state maximizes integer `(replay_time_ns,sort_index)` with causal source index and clock <= removal+2s; a same-participant source state at/beyond target is required. No maximum age/freshness is imposed. This coverage criterion does not certify a continuous external feed. Independent review rederived all 906 execution and 2,482 cancellation endpoints with zero disagreements; hashes in that review bind its exact scope.
- REV: cancellation anchored H_R=2s, using explicit physical cancellations assigned within primary H=10s. Weights are `attributed_cancel_visible_qty`, not discounted age quantities. Complete REV requires a nonempty positive-quantity cancellation set and finite observations for every member. Available-only REV and observed weight fraction remain separate private diagnostics. Five Risanamento execution REVs are incomplete (ten cancellation observations missing); their complete values remain null. FPM/ADV missing on 13 union executions due to unobserved placement: 3 Ferrari, 1 Nexi, 9 Risanamento. Q and VWAP are original prepared execution values.
- Numerical equality: raw FPM/REV/ADV and all identities, flags, assigned quantities and endpoint provenance remain unchanged. The separate numerical stage appends endpoint-scale tolerance/sign columns. FPM/ADV tolerance is 8 times the larger endpoint binary64 ULP. REV uses the raw-quantity-weighted mean of cancellation endpoint bounds plus 8 aggregate-REV ULPs. Only sign/median accounting normalizes values within the bound to zero; nonfinite remains missing. No price gate, economic movement floor, ratio epsilon or assignment change. All twelve full summaries/medians agree at factors 4/8/16/32; strict counts are retained and joint-positive counts are unchanged under strict signs.
- Observed joint-positive FPM/complete-REV counts W→J: self Ferrari 33/353→6/36, Nexi 23/240→5/24, Risanamento 4/99→1/10; market Ferrari 36/353→1/36, Nexi 34/240→4/24, Risanamento 3/98→1/10. Self enrichment is descriptive in all three samples; market enrichment is not uniform. Retained role/identity summaries reconcile to each list and are exported without original actor identifiers. Pooled differences do not identify within-role improvement.
- Diagnostic cases: R and N have FPM zero, complete REV approximately 0.0000296361/0.0008, ADV -0.0001/-0.001 in source price units. Neither supplies both positive price features. Preserve these noncorroborating components alongside their order-level withdrawal and profile evidence.

## 20. Historical market-only percentile presentation

Status: verified historical publication, recorded in `paper/reviews/20261015_085912_market_only`. The sole market normalization remains the author's choice, not a finding of superior correlations, alert prioritization or classification accuracy. The percentile views and691-execution price presentation below are superseded by the rank-only contract in section21. Sections18and19 continue to preserve the frozen native computation and former dual-ranking price evidence.

- Active score: market W=log1p(P_g/Q)*log1p(U_g/Q)*(U_g/D), called WMSCI in the paper. Q>0 is its execution domain. Missing/zero D gives undefined; with positive D and zero P_g or U_g the value is zero. No epsilon. Physical removal share U/P remains a diagnostic, not an alternative WMSCI ranking. Remove self-score formula, self/market identity/comparison, overlap figure and self-specific empirical cohorts from the active source; preserve all historical artifacts and source bundles.
- Primary setting unchanged: exp_h30,H10, all observable ages, same raw6914 candidates and same full execution keys/assignments. Original market flags `priority_top_w_market` and `priority_top_j_given_market`, no reranking or price filtering. W counts Ferrari353,Nexi240,Risanamento98; conditionalJ36,24,10. Total distinct market W population691; conditional lists nested. Score support and upward rounding remain explicit; Risanamento finite positive market support978/995.
- Prices reuse ONLY the exact native market subset of the independently verified numeric_v2 ledger,691 executions and2098physical cancellation rows, not the former906-execution union. No price recomputation or modified endpoints is claimed. The six market W/J groups retain their original prices, quantities, numerical-zero convention and all-sign/missing partitions. Thirteen FPM/ADV missing (3Ferrari,1Nexi,9Risanamento); five completeREV missing, allRisanamento. REV remains physical-quantity weighted with H_R2s after each assigned cancellation, separate from H10.
- Joint-price observations remain Ferrari36/353→1/36,Nexi34/240→4/24,Risanamento3/98→1/10. No uniform enrichment claim. Ferrari conditional-market ADV positive13/36, not the former self22/36. CasesR/N remain FPMzero,positivecompleteREV,negativeADV; R enters marketW atH2 andmarketJ atH5,N entersbothatH10. No conversion of partial/missing/negative results into positive corroboration.
- Presentation: new `scripts/generate_market_only_manuscript.py`, fresh `paper/generated/withdrawal_market_20261015_v1` with market-only tables/CSV and authenticated reuse of unchanged candidate-count and physical-withdrawal/profile figures. Old renderer/tests/publications and all native/price producers are immutable. Remove the self-selected H1/age30 profile figure only from the active include graph, not the shared historical file or other manuscripts. Public source/captions still omit external-alert comparisons; private provenance in section18 remains unchanged.

## 21. Active complete WMSCI ranking with J alongside

Status: author-approved rank-only decision,2026-10-15; exact native population and ranking support verified in `paper/reviews/20261015_101002_rank_only/expected_candidate_memberships.parquet` and independently frozen expected ranks. Actual whole-candidate price artifacts and primitive/endpoint parity are verified in the V2 bundle below; independent reviews and final publication evidence are recorded in that review directory at their respective scopes. Sections18–20 are historical provenance where they describe selected lists.

- Scientific scope unchanged: exp_h30,H10, all observable ages; original structural rule P_raw>0,U_raw>0,Q>0, same original identities, cancellation assignments, market-depth denominator, score/profile formulas, integer-ns clocks and empirical H1 depth weights. No native detector/config mutation. Primarypopulation6914=3526Ferrari+2393Nexi+995Risanamento, not6897positive scores,691selected-list rows or6924grid union.
- Active review rule: descending finite market WMSCI within instrument and parameter setting, including zero. Stable ties use execution_cluster_id,partition_id,actor_key and NEVER J or prices. Nonfinite scores, if any, stay in the output with nullrank and explicit unsupported status, not an imputed zero or exclusion. No percentile budgets, conditional-J stage or score/price thresholds. The analyst's stopping depth is an operational workload decision, not a behavioral boundary.
- J: accompany every execution with its original complete finite value (including zero) or explicit missing/invalid-support status; no filtering, ordering or tie-breaking. Primarydata contain6914complete finite J values; zero counts Ferrari4,Nexi5,Risanamento47. All6914primary W values are finite;17Risanamento values are exactly zero and remain at the end of their instrument's ranking. This corrects any inference that the earlier978positive-score support meant17undefined scores.
- Price scope: recompute actual diagnostics for ALL6914rawcandidates, including17Wzero rows, from the authenticated retained-state caches and unchanged assignments. Do not relabel former906or691row ledgers as full-population results. Full-key/quantity/role checks against the frozen population, endpoint parity on all906common historical rows and2482common cancellations, completeREV with HR2s after each physical cancellation, original VWAP and no invalid-predecessor fallback are required before publication. Raw values and8ULPendpoint-based sign/median accounting remain separate; factors0/4/8/16/32 sensitivity and allsign/missing partitions use full instrument denominators.
- Current artifacts: `scripts/study_ranked_candidate_prices.py`, `outputs/ranked_candidate_prices_20261015_101002_v2`; `scripts/generate_rank_only_manuscript.py`, fresh `paper/generated/withdrawal_rank_20261015_v2`. The unsuffixed price V1 bundle remains historical. Parent review corrected its overly restrictive D-support predicate to the declared P/U/Q structural domain, observed failing then passing regressions, and executed a fresh V2 run. All6914raw price/endpoints/sign rows and12426physical cancellations match V1 exactly by full keys; no observed numerical result changed. Public fullranking uses report-local aliases, scores,J/status and price diagnostics; private fullkeys retain traceability. No self-score column, oldpriority flags or externally seeded coverage claim in active reader assets. The two approved quantity/profile figures are reused only with exact hash and native-companion checks. Historical scripts/tests/packages remain immutable.
- Full-population price findings: three instrument summaries,12role/identity strata and15factor-sensitivity rows. Jointly positive FPM/completeREV397/3526Ferrari,308/2393Nexi,44/995Risanamento. Missing FPM/ADV14/9/72andmissingcompleteREV6/3/16, respectively. CompleteREV physical cancellations6452/4686/1288byinstrument. Factors4/8/16/32giveidenticalsign/median summaries; strict signs remain separate. Every candidate stays in its denominator. These are descriptive whole-population outcomes, not evidence that high ranks discriminate manipulation.
- Inference: a complete ranking does not establish review yield, a false-positive rate, independent corroboration or intent. Whole-population price counts describe available evidence, not superiority of high ranks. Low/zero J can coexist with physical cancellation; large J can precede it. Keep R/N observed FPMzero,positivecompleteREV,negativeADV unless an actual numerical discrepancy is found. Private exploratory settings/case provenance and clock limitations remain unchanged in section18.

## 22. Compact figures and participant-level summaries

Status: author-approved presentation and aggregation extension,2026-10-15; real native aggregation exercised with `scripts/summarize_ranked_participants.py`. Release evidence belongs to `paper/reviews/20261015_152614_participant_results`. Section21's computational specification and primary6914candidates are unchanged. Its two-figure presentation is superseded: one common-axis three-instrument candidate-count plot replaces the three panels; the diagnostic-case figure is omitted while the case observations and table remain. New presentation consumer: `scripts/generate_participant_manuscript.py`; fresh package `paper/generated/withdrawal_rank_20261015_participants_v3`. Earlier packages remain immutable.

- Denominators: all48481primary native executions, each represented once by instrument/partition/actor/execution key. Identified-client view includes only canonical `client_original` attribution; intermediary view groups ALL executions by original `event_firm_id`, including original-client and firmfallback activity. The two views overlap and are not additive. Missing firms, if any, remain an explicit bucket; missing client codes never create a synthetic client. All65471child fill members were checked against execution anchor firms, with no mismatches or multifirm executions. Mixed-firm execution clusters must stop attribution rather than be assigned to their first firm.
- Identity and export: namespaced native client/firm keys remain private in a separate mapping; stable reportlocal C/F aliases span instruments. Existing client consolidation is unchanged. Public tables are instrument-specific; no cross-instrument score, WMSCI sum, or potentially overlapping P/U volume sum. Group candidatefraction is candidateexecutions/allobservedgroupexecutions, not a manipulation rate. Partition count is not relabeled as trading-day count.
- Observed clients: Ferrari7/323candidate/allclients,355/4344candidate/allexecutions; Nexi9/587,496/5902; Risanamento17/1041,403/20722. Largest client candidatecounts293/355,385/496,264/403; repeated-candidate client counts5,7,9. Observed intermediaries20/49,19/48,11/35, covering3526/12997,2393/10512,995/24972candidate/allexecutions. Largest intermediary candidatecounts702/3526,385/2393,316/995; repeated-candidate intermediarycounts18,17,8. Client/firm unions1815/57are distinct from perinstrument entityrows2083. These are observed concentration/recurrence, not independent-event counts or coordinated-intent evidence.
- Complete CSV tables retain zero-candidate groups and8332role/identity strata. The appendix prints all33client/instrument and50intermediary/instrument groups with candidates, in alias order; W/J candidate medians are diagnostics, not participant priorities. Prices remain the exact completed6914recordV2ledger with rawvalues and8ULPsigns; every sign/missing/joint denominator is the complete group's candidatepopulation. The original full executionranking and all old scientific companions remain unchanged in the new package. No rawreplay or detector/config changes.
