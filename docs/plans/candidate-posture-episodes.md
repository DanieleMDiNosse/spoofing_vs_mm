# Candidate-posture episodes implementation plan

> **For Hermes:** Use subagent-driven-development for disjoint implementation tasks, followed by specification and regression review.

**Goal:** Jointly evaluate clusters sharing candidate posture without losing passive/aggressive provenance, actor identity, or trading-day boundaries.

**Architecture:** Preserve execution-cluster diagnostics and raw membership. Add a primary candidate-posture episode table plus explicit cluster membership, unique withdrawal accounting, branch summary, and actor-day summary. Existing cluster strict flags are diagnostic only, not independent episode counts. Historical paper artifacts remain historical and are not overwritten.

**Tech stack:** Existing Python 3.11 environment, Polars, pytest. No new dependencies.

## Decisions and scope

- Pre-change checkpoint: abb581cb5fc6e756323d026615243d10fd46a3fc. No push. Chat attachment left untracked.
- Both passive and aggressive executions remain enabled. Passive clusters identify one resting order; aggressive clusters identify one incoming order and may sweep several counterparty prices. Never mix branches inside an execution cluster.
- Episodes are connected components of clusters sharing an opposite-side candidate order lifecycle (order ID plus first-seen source index), within partition, canonical actor, execution side, and recorded event date. This is an explicit overlap-posture definition, not independent evidence or a causal estimator. Transitive overlap is retained and duration/order/cluster counts exposed.
- Mixed-anchor episodes are jointly evaluated, with separately summed passive/aggressive execution quantities. Branch summaries report episode participation: mixed episodes must not be summed as independent episodes. Physical withdrawal is counted once globally, not once per branch.
- Use total execution quantity of every member, including members with no individually assigned cancellation. Sum only unique canonically assigned physical cancellations. Do not duplicate candidate quantities across cluster snapshots; use the maximum observed pre-execution quantity per candidate lifecycle, sum over lifecycles, and label this descriptive evolving-posture scale.
- Favorable movement starts after actual candidate placement. Use the latest placement among the cluster's fixed candidate set (complete-posture baseline), strictly before its first fill; exact placement state only, no later-state imputation. Reload-only first observations are unknown placement, so FPM unavailable. Episode FPM is execution-quantity-weighted across cluster FPM with complete coverage required; preserve individual diagnostics. Reversion is weighted over unique assigned cancellations with complete coverage required at episode level.
- Cluster smallness uses full cluster execution quantity. Passive queue diagnostic divides by first-fill pre-queue quantity (may exceed one with replenishment). Episode withdrawal smallness uses sum of all cluster quantities.
- No cluster/episode/response joins cross partition or recorded day. Actor-day aggregation includes partition. Multi-day inference is not implemented or claimed.
- Existing ETL economic role definitions remain; verification is explicitly unverified until upstream mapping evidence is supplied. No fabricated bit positions or vendor provenance.

## Tasks and verification

1. Cluster smallness and calendar boundary (owned execution_clusters.py and focused tests): RED for multifill numerator and midnight crossing; GREEN both anchors, canonical/legacy aliases; pytest focused then full after integration.
2. Pure episode builder (new candidate_episodes.py and test_candidate_episodes.py): typed empty outputs, stable deterministic IDs, actor/partition/lifecycle boundaries, transitive overlap, mixed anchors, quantity conservation, unique cancellations, complete outcome coverage. RED before implementation.
3. Core timing (spoofing_metrics.py and new regression tests): exact latest placement baseline; reload unknown; covered-only SCI target; same-day response bounds; direct cancellation source-order guard. Preserve FPM when SCI horizon censored. RED/GREEN focused.
4. Integration: add episode tables to result and both standard/grid writers, semantic metadata and hashes, branch and actor-day summaries. Invalidate caches via semantic version and output requirements. Keep legacy cluster artifacts explicitly diagnostic.
5. Downstream actor-day features and alerts: preserve partition when supplied; legacy no-partition direct calls remain labelled legacy. Episode-aware counts prevent cluster repetition being mistaken for episode repetition. New dedicated episode reports are authoritative; old dashboards/paper not silently relabelled.
6. ETL status note + current-method documentation; expose verification status in run metadata. Upstream evidence required for certification.
7. Execute CLI smoke with realistic raw synthetic actor/day/anchor fixture and a bounded real-data subset if prerequisites available, writing new output directory only. Check artifacts, conservation, branch/day splits and hashes.
8. Scientific/specification review, regression review, fix concrete findings, full canonical pytest, git diff --check. Report changed files, actual tests, new artifacts, and historical outputs not regenerated.
