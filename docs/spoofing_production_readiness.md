# Spoofing Detection Production-Readiness Workflow

## Purpose

This workflow promotes event-level spoofing-like detections into analyst-reviewable actor-session alerts.

The detector does not infer legal intent. It produces surveillance cues for human review.

## Layers

1. Event-level DWI/SCI/resting-profile MSCI and withdrawal-profile metrics.
2. Analyst annotations.
3. Actor-session repeated-pattern features.
4. Legitimate market-maker baseline features.
5. Negative controls and placebo checks.
6. Threshold calibration by analyst workload and false-positive pressure.
7. Actor-session alert objects.

## Recommended order of use

```bash
# 1. Build/refresh event dashboard outputs.
# 2. Bootstrap annotation CSV.
# 3. Analysts edit annotation CSV.
# 4. Compute actor-session features.
# 5. Build negative-control report.
# 6. Build calibration report.
# 7. Run production-readiness pipeline.
# 8. Regenerate dashboard with annotation and alert files.
```

## Scientific interpretation

### Canonical actor identity

The operational detector resolves one namespace-aware identity per event using the following hierarchy:

1. use `NMSC_ORIGINALCLIENTIDSHORTCODE` when it is present, producing `actor_key = client_original:<id>`;
2. otherwise fall back to `FIRMID`, producing `actor_key = firm:<id>`;
3. if neither field is available, leave the event unattributed rather than inventing or imputing an identity.

The accompanying fields `actor_id`, `identity_level`, `identity_source`, and `identity_fallback_flag` preserve how the key was obtained. Client and firm identities are never pooled or joined across levels, even if their unnamespaced values happen to match. A firm fallback is a coarser member-firm aggregate that can contain activity from multiple underlying clients; it must not be interpreted or labelled as client-level attribution. Actor-state time series remain actor-specific; execution-derived clusters, candidate links, scores, rankings, and alerts additionally preserve the passive/aggressive anchor stratum.

### Passive and aggressive execution anchors

The passive and aggressive branches are separate analytical populations. A fill is assigned only when exactly one of `PASSIVEORDER` or `AGGRESSIVEORDER` is `Y`; rows with both or neither flag set are not assigned to either branch. The aggressive branch additionally requires positive finite trade evidence from `LASTSHARES` and `LASTTRADEDPX`; `event_price` is not accepted as an aggressive execution price. The configured anchor set and the anchor modes actually observed in output are recorded separately, so an empty observed set is not silently reinterpreted as passive evidence.

These role flags depend on ETL decoding from venue bit fields. Before empirical conclusions are drawn from the aggressive extension, the ETL source and bit positions must be verified as described in `docs/keep_cols_data_dictionary.md`. Until that check and a dedicated empirical rerun are complete, the aggressive branch is an experimental surveillance extension, not a validated main-result population.

`MSCI_resting_profile = SCI / 2 + C_opposite - C_same` is the common resting-profile contrast. `withdrawal_profile_scale_event` is a separate post-execution withdrawal-evidence scale. The branch-specific products `WMSCI_passive` and `WMSCI_aggressive` are populated only on their applicable anchor; the other branch is null, not zero. The v2 schema retains `MSCI`, `WMSCI_event`, `fill_qty`, and `withdrawal_to_fill_ratio` only as documented compatibility aliases for the canonical fields.

Actor-session alerts are ranked by observed withdrawal-profile scale and resting-profile MSCI, without combining them into an uncalibrated scalar. Threshold calibration is stratified by `execution_anchor_mode`: passive and aggressive rows must not be pooled. The checked-in passive cutoff is retained, while the aggressive cutoff is explicitly null. Consequently, threshold-dependent aggressive alerts are inapplicable until dedicated calibration evidence exists; the pipeline must not borrow or invent a threshold.

An alert should be considered stronger when:

- suspicious episodes repeat for the same canonical actor/session and execution anchor;
- resting-profile MCPS remains high across depth choices within the same execution anchor;
- the pattern is robust over kappa/lambda values;
- opposite-side collapse is stronger than same-side collapse;
- negative controls score lower than real events;
- the behavior is unusual relative to the same actor's own baseline and an appropriately stratified peer group.

## Non-goals

- The LLM does not decide manipulation.
- No individual event-level metric is a legal conclusion.
- Thresholds are anchor-specific and are not final until calibrated against labels and analyst workload.
