# Empirical controls v2 — implementation checkpoint

## Scope and provenance
- Plan: `docs/plans/2026-09-16-empirical-controls-v2-light.md`.
- User now explicitly authorizes implementation using `resumable-work-checkpoints`; this supersedes the plan's historical documentation-only restriction for tasks 1–6. Software/synthetic tests are part of implementation. Task 7 real-data pilot and scale remain unauthorized. No raw-data analysis, real preflight, commits, push, paper/dashboard changes, or LATEST promotion.
- Workspace: `/home/danielemdn/Documents/repositories/spoofing_detection`; branch `main`; initial HEAD `7e0b822cb44f8bbd4bb9691fd650ff3ab727e110`.
- Started: 2026-09-16T15:37:16+02:00.
- Environment verified: `/home/danielemdn/miniconda3/envs/main/bin/python`, Polars 1.41.2, pytest 9.0.2, NumPy 2.3.5. No dependencies installed.
- Dirty baseline: existing edits in paper/generated and paper/spoofing_new.tex; scripts build_spoofing_event_review_dashboard, build_spoofing_negative_control_report, generate_consob_detected_events_report, generate_spoofing_paper_tables; lob candidate_episodes, negative_controls, panel, spoofing_metrics; matching existing tests. Existing untracked v1 config/runner, replay_observation.py, withdrawal_risk.py and tests; plan documents, reports/consob and desktop attachment. Preserve all.

## Implementation decisions
- Isolate v2 internals in new explicitly versioned modules to avoid collision with dirty v1 implementations. Existing entry points will receive small dispatch integrations only; legacy semantics remain unchanged. This is a file-ownership adjustment to the plan, not a scientific design change.
- Public table vocabulary uses `side` (existing risk schema); report groups refer to it as posture side. Use naive canonical timestamps as existing replay does, labelled source_clock_timezone_unknown unless certified otherwise.
- Pure accounting accepts compact intervals, explicit eligible cancellation events, canonical execution clusters and coverage epochs. It never replays the detector.

## Units
| Unit | State | Evidence |
|---|---|---|
| 1: v2 config, certification and gates | implemented_unverified | 15 focused tests pass; independent review pending |
| 2: compact canonical-hook risk observer | implemented_unverified | focused tests pass including corrected point semantics; final review pending |
| 3–4: observed accounting and fixed shifts | implemented_unverified | point-only integration tested; independent re-review interrupted by quota |
| 5: balance, checkpoints, bundle and report | in_progress | balance and partition disk-cache implemented; full runner/report not implemented |
| 6: scientific/spec + quality review and regressions | in_progress | first independent spec review identified two corrected P0s; final review and broad suite pending |
| 7: empirical pilot and scale | blocked | separate authorization required |

## Jobs / ownership
- No empirical jobs launched.
- Controller owns progress document, new v2 runner/config/storage and entry-point dispatch.
- Worker `sa-0-b6f95fd7` returned: point-only groups preserved, epoch identity required, zero-duration coverage accepted without positive domain. Ownership released.
- Workers `sa-0-d7aec97b` and `sa-0-811fdbde` failed on HTTP429 quota. No live workers remain; ownership released. Reviewer produced no verdict. Implementation worker left only test_empirical_controls_v2_analysis.py, no implementation. Parent explicitly took over after reconciling files and task state.
- Earlier workers completed and released ownership. Controller owns observer, IO and new partition cache module/tests.
- Do not edit these until completion/handoff. No collection-sensitive full-suite tests during parallel edits.

## Controller checkpoint: IO/config
- New `src/spoofing_detection/lob/empirical_controls_v2_io.py`, `tests/lob/test_empirical_controls_v2_io.py`, `configs/spoofing_empirical_controls_v2_light.json`.
- Existing runner: minimal v2 preflight dispatch and explicit required `--config`; run integration still pending, do not run v2 empirically.
- RED: 7 expected missing-feature failures; GREEN 7. Certification RED: 1 expected missing gate failure; GREEN 14. CLI dispatch RED: rejected v2 version; GREEN 15 total.
- Verified command: `/home/danielemdn/miniconda3/envs/main/bin/python -m pytest tests/lob/test_empirical_controls_v2_io.py -q` → 15 passed.
- No real inputs read or hashed. All certification sources above are temporary synthetic empty schema fixtures, not a certificate for real data.
- New `empirical_controls_v2_budget.py` and test: independent Linux process-group watchdog, sampled RSS cap and timeout. RED 4 expected missing supervisor failures → GREEN 4 passed using only tiny synthetic Python subprocesses. Sampling can overshoot between observations; this is not an OS hard memory reservation. No protection disabled.

## Next action
Controller resumed storage review: checkpoint completion previously collected every chunk/table into RAM despite bounded writing. Added `scan_checkpoint` with hash/count/schema/table-graph and duplicate-path validation; completion now validates lazily, eager compatibility reader remains explicit. Two regression tests RED (materialization and missing API), then IO + budget suite GREEN: 21 passed in 0.55s. `git diff --check` passed. Tests use only temporary synthetic inputs. Runner/report integration and independent review remain pending.

Resume reconciliation found no live previous subagents or empirical/test processes; HEAD unchanged. The focused four-module suite returned 10 failed, 30 passed: accounting balance schema/overlap validation and observer original-index mapping, market lookahead, and cancellation coverage ID are unfinished. Prior workers' files are not verified handoffs.

Both resumed implementers returned (iteration-limited summaries); ownership released. Parent reran focused tests: 55 passed. Accounting balance schema, analytic ages and overlap rejection implemented; observer indices, market pre-mutation expiry and cancellation epoch fixed. Independent read-only scientific review dispatched as `sa-0-ec4cdc99`.

Parent found two scientifically wrong observer tests despite GREEN: exact age cutoff and same-timestamp new/cancel were excluding real eligible points. Corrected expectations (RED: 2 failed, 13 passed), then snapshot inclusive-age cancellation eligibility from canonical pre-event state separately from duration expiry. Added exact90 versus90+1 microsecond after another same-time event triggers expiry. Final focused command `/home/danielemdn/miniconda3/envs/main/bin/pytest tests/lob/test_withdrawal_risk_v2.py tests/lob/test_empirical_controls_v2.py tests/lob/test_empirical_controls_v2_io.py tests/lob/test_empirical_controls_v2_budget.py tests/lob/test_replay_observation.py -q --tb=short` → 57 passed in 0.68s; diff check passed. Core remains implemented_unverified pending independent review. Investigate downstream point-only groups and zero-duration coverage before integration: observer now correctly preserves those points but accounting may discard them.

Independent scientific review returned NOT READY on the two point bugs; its inspected version preceded the controller fix. Re-review current version required, not an approval. New internal `empirical_controls_v2_partition.py` integrates complete one-partition canonical replay with bounded disk chunks, constant-space local-to-original index mapping and validated resume without replay. Count/partition/date/offset contract failures precede writes. Interrupted chunks are preserved, never appended; retry starts the partition clean. Tests compare every stored risk table to direct canonical-hook output, index23 on cancellation, rejection of changed cache identity, and interrupted flush retry. RED 5 missing-module failures; GREEN command `pytest tests/lob/test_empirical_controls_v2_partition.py tests/lob/test_empirical_controls_v2_io.py tests/lob/test_withdrawal_risk_v2.py -q --tb=short` in project environment: 39 passed in0.29s. This is an internal primitive, not the full runner or a certification issuer.

Parent full regression after point-only accounting handoff: `export PATH=/home/danielemdn/miniconda3/envs/main/bin:$PATH; pytest -q --tb=short` → 778 passed, 1 skipped in4.22s; `git diff --check` passed. Small subsequent test-only type-cleanup in partition test rechecked:5 passed. This passing suite is not completion of missing runner/report/bundle.

Quota-resume reconciliation: HEAD unchanged, no live workers. Parent completed `empirical_controls_v2_analysis.py`: verified disk-backed risk inputs, per-actor lazy filtering and bounded result flush, no-fill actor retention, orphan cluster partition/date rejection, schema-stable empties and atomic verified resume. Cache identity binds caller label plus risk manifest bytes, cluster IPC bytes and stage/accounting source hashes. Output `summary_contributions` is NOT a final pooled summary; later aggregate must retain original weights. Partition coverage/market frames and one actor's inputs/results still materialize; outer resource supervision remains required. Test fixture clusters are explicitly unit-level schedule inputs, NOT a detector end-to-end certification. RED4 missing-module failures (after correcting shell PATH), GREEN6 including empty schemas and changed-schedule rejection,1.32s. No packages installed.

Latest parent full-suite check after the new disk-analysis stage: `pytest -q --tb=short` in project environment → 784 passed,1 skipped in5.87s, runtime evidence scope=full/status=passed. Bounded scientific re-review retry dispatched as `sa-0-808019bd` (read-only,4-call scope); result pending, do not assume quota is reset.

Bounded scientific review `sa-0-808019bd` returned PASS on exact-age/same-time points, observed/shift domains and point-only anomalies, with37 focused tests. It does NOT approve full aggregation or runner. Quality/resource read-only review dispatched as `sa-0-86c450db`; handoff pending.

Parent aggregation audit then found a further frozen-contract defect: shift summary selected common groups but reused offset-specific weights instead of zero weights. Added changing-risk-overlap/two-actor regression (RED1: observed weight3 expected4), corrected summary to use shift_support.zero_weight_seconds for every shift; observed contrasts and per-offset individual contrasts remain unchanged. Full `pytest -q --tb=short` →785 passed,1 skipped in6.18s. New fix needs review; previous bounded PASS does not cover it.

Quality review returned changes required: duplicate expiry heap entries, repeatedly rebuilt schedule starts, prefix market scans, eager whole-partition market collection. Parent fixed heap with one pending expiry per lifecycle (RED465 pushes for30 lifecycles; GREEN bounded unique30); observer/partition/replay focused32 passed. Actor analysis now lazily filters market inputs by actor risk epoch/time bounds, not whole-partition collection (RED test reproduced unrelated market rows; GREEN pending accounting worker handoff). One large actor can still span most of a partition, so RSS supervision remains mandatory.

Worker `sa-0-0a8df3e8` was explicitly interrupted to freeze file ownership for mandatory verification; terminal interrupted handoff received. No live worker retains files. Parent inspected completed edits: WindowIndex caches schedule starts, shifted zero reuses index, market intersections bisect lower/upper bounds and avoid prefix slicing. Instrumented tests check exact examined indices among10,000 source rows rather than flaky timing. Latest actor-market filter also verified after worker stopped.

Parent verification: focused accounting/analysis/observer/partition suite47 passed in1.70s, then full `pytest -q --tb=short` →789 passed,1 skipped in6.09s. These results include heap deduplication, lazy actor market selection, retained-window bisection, bounded market intersections, and fixed-zero shift weights. Quality fixes implemented/tested; independent final re-review of fixes still outstanding. Runtime cost on real data remains unmeasured and no empirical artifact was produced.

Next: independently re-review quality fixes and fixed-weight aggregate; then implement complete sequential runner/bundle/report and raw-detector synthetic end-to-end acceptance. Full runner still needs certified lazy partition extraction, sequential resource-supervised execution, disk-backed cross-partition aggregation, persisted accounting reconciliation, manifest/report and genuine canonical-detector raw-smoke acceptance. Integrate sequential partition runner + disk-backed aggregate/report only on the repaired contracts. Independent scientific/spec review precedes quality review, then permitted regression suite. No empirical runs, real preflight, commits, or promotions authorized. No Kanban task is attached to this interactive session (HERMES_KANBAN_TASK unset).
