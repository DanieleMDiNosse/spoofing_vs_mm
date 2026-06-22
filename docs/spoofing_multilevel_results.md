# Spoofing metric implementation and results

This note summarizes the current state of the spoofing-metric implementation and the empirical results obtained on the Risanamento sample.

## What was implemented

The model now follows the multilevel DWI/MSCI/MCPS formulation used in the manuscript.

The implemented quantities are:

- `DWI`: the client's multilevel distance-weighted imbalance over the top `n` price levels.
- `SCI`: the absolute change in DWI from just before a small passive execution to after the post-execution window.
- `collapse_bid` and `collapse_ask`: how much weighted liquidity disappears on each side after the execution.
- `MSCI`: the event-level spoofing conditionality score. It is high only when the DWI changes sharply and the opposite side collapses more than the same side.
- `MCPS`: the client-level repetition score, i.e. the fraction of eligible executions whose MSCI exceeds a threshold `gamma`.

The implementation uses the clean current naming scheme:

- `DWI`
- `MSCI`
- `MCPS`
- `candidate_deceptive_*`
- `matched_deceptive_*`

Old transitional names such as `imbalance`, `weighted_*_fraction_topN`, and `candidate_fake_*` are not used in the new outputs.

## Timing correction

A key issue was fixed: before the correction, candidate deceptive volume could include opposite-side liquidity that had been resting for a very long time before the execution. This was not consistent with the intended spoofing dynamics.

The model now imposes a pre-execution timing window:

- default window: 600 seconds, i.e. 10 minutes;
- an opposite-side order is counted as a candidate deceptive order only if it was first observed within 10 minutes before the small execution;
- old standing liquidity is not attributed to the spoofing episode;
- the analysis now focuses only on matched deceptive-order cancellations: the post-execution cancellation must remove one of the time-filtered candidate deceptive orders.

This change was added both to the code and to the manuscript.

## Dashboard correction

The dashboard was also simplified.

Previously, event-level points separated several cancellation diagnostics:

- red: direct cancellation of a candidate deceptive order;
- orange: cancellation of another opposite-side order;
- blue: no opposite-side cancellation.

This was misleading because orange and blue points were not spoofing-like executions. They were weaker diagnostics or non-matched executions, and they are no longer considered for the main spoofing-event analysis.

The dashboard now shows only spoofing-like executions in the event-level scatter plots:

- red points: executions where the client directly cancels one of the pre-existing opposite-side candidate deceptive orders after the execution;
- other executions are omitted from these scatter plots.

The dashboard text now states that these red points are surveillance cues, not proof of intent.

## Main empirical run

The main updated run is:

`outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/`

Configuration:

- instrument/sample: Risanamento;
- top-n depth: 3;
- post-execution cancellation window: 1 second;
- candidate deceptive order age window: 600 seconds;
- gamma grid: 0.25, 0.5, 0.75, 1.0.

Generated files include:

- `client_metric_time_series.parquet`
- `execution_metrics.parquet`
- `candidate_deceptive_orders.parquet`
- `rejected_executions.parquet`
- `client_mcps_scores.parquet`
- `metadata.json`
- `summary_report.md`
- `spoofing_metric_dashboard.html`

## Row counts from the updated run

For the updated 10-minute timing-window run:

| Output | Rows |
|---|---:|
| client metric time series | 1,114,955 |
| execution metrics | 15,341 |
| candidate deceptive orders | 3,365 |
| rejected executions | 18,997 |
| client MCPS scores | 14,844 |

Important event counts:

| Quantity | Count |
|---|---:|
| executions with at least one candidate deceptive order | 2,072 |
| executions with a matched deceptive-order cancellation | 32 |

The 32 matched deceptive-order cancellations are the executions currently highlighted as spoofing-like in the dashboard.

## Effect of the timing correction

The previous top-n 3 run without the 10-minute timing restriction was:

`outputs/spoofing_metrics/risanamento_top3_multilevel_msci/`

Comparison:

| Quantity | Before timing restriction | After 10-minute timing restriction |
|---|---:|---:|
| execution metrics | 15,341 | 15,341 |
| candidate deceptive orders | 6,320 | 3,365 |
| executions with candidate deceptive orders | 3,912 | 2,072 |
| matched deceptive-order cancellation executions | 46 | 32 |

The timing restriction therefore removed many old opposite-side orders from the candidate deceptive profile.

This is the intended behavior: the number of candidate deceptive orders fell from 6,320 to 3,365 because orders older than 10 minutes are no longer attributed to later executions.

## Timing sanity check

For the updated run, the candidate deceptive order ages are:

| Quantity | Value |
|---|---:|
| minimum age | 0.000052 seconds |
| maximum age | 599.837086 seconds |
| mean age | 169.635024 seconds |

This confirms that the 10-minute constraint is active: no candidate deceptive order in the updated output is older than 600 seconds.

For comparison, in the older unrestricted run the maximum candidate age was about 51,892.5 seconds, which is more than 14 hours. That was the behavior we wanted to remove.

## MSCI range

For the updated run:

| Quantity | Value |
|---|---:|
| minimum MSCI | 0.0 |
| maximum MSCI | 1.770448 |

The maximum MSCI did not change relative to the previous unrestricted top-n 3 run. The timing correction mainly changed which opposite-side orders are attributed as candidate deceptive orders and which cancellations count as matched deceptive cancellations.

## Interpretation

The updated pipeline is now more consistent with the intended spoofing timeline:

1. a candidate deceptive profile must be posted shortly before the small execution;
2. it must be visible on the opposite side before the execution;
3. it must be cancelled after the execution to be highlighted as spoofing-like.

The dashboard now focuses only on the strongest event-level cue: matched cancellation of a pre-existing opposite-side candidate deceptive order.

These results should still be interpreted as exploratory surveillance signals, not as proof of manipulative intent. The next scientific step is calibration and robustness analysis: thresholds, window lengths, depth choices, and client-level false-positive behavior still need to be studied before using the scores as detection claims.

## Validation performed

The implementation was checked with:

- targeted spoofing metric tests;
- full test suite;
- Python bytecode compilation of the changed modules/scripts;
- a smoke run;
- a full Risanamento top-n 3 run with the 10-minute timing window;
- regeneration of the updated dashboard.

Observed test result after the latest dashboard correction:

- full suite: 55 passed.

## Main files changed

Code:

- `src/spoofing_detection/lob/spoofing_metrics.py`
- `src/spoofing_detection/lob/spoofing_metric_plots.py`
- `scripts/compute_spoofing_metrics.py`
- `scripts/run_multilevel_spoofing_grid.py`

Tests:

- `tests/lob/test_spoofing_metrics.py`
- `tests/lob/test_spoofing_metric_plots.py`
- `tests/lob/test_spoofing_metric_report.py`

Manuscript:

- `paper/spoofing.tex`

Documentation:

- `docs/lob_implementation_status.md`
- `docs/spoofing_multilevel_results.md`
