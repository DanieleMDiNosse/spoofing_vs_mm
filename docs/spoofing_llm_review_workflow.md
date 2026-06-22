# Spoofing Event LLM Review Workflow

## Purpose

The LLM review explains candidate spoofing-like events selected by the DWI/MSCI/MCPS model. It does not decide legal intent and it is not evidence by itself. It is a structured aid for human surveillance review.

## Generate one event dossier

Choose an event id from the review output:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python - <<'PY'
import polars as pl
print(pl.read_parquet('outputs/spoofing_event_review/risanamento_top3_timing_window/matched_spoofing_events.parquet').select('review_event_id').head())
PY
```

Build the dossier:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/build_spoofing_event_dossier.py \
  --review-dir outputs/spoofing_event_review/risanamento_top3_timing_window \
  --event-id <EVENT_ID> \
  --parameter-grid-root outputs/spoofing_metrics/kappa_lambda_sensitivity_top3
```

This writes:

```text
outputs/spoofing_event_review/risanamento_top3_timing_window/llm_reviews/<EVENT_ID>/dossier.md
outputs/spoofing_event_review/risanamento_top3_timing_window/llm_reviews/<EVENT_ID>/dossier.json
```

## Analyze one event with Ollama/gemma4-hermes:latest

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/analyze_spoofing_event_with_llm.py \
  --dossier outputs/spoofing_event_review/risanamento_top3_timing_window/llm_reviews/<EVENT_ID>/dossier.md \
  --prompt prompts/spoofing_surveillance_analyst.md \
  --output-dir outputs/spoofing_event_review/risanamento_top3_timing_window/llm_reviews/<EVENT_ID> \
  --model gemma4-hermes:latest \
  --timeout-seconds 180
```

This writes:

```text
prompt.md
response.md
metadata.json
```

inside the event's `llm_reviews/<EVENT_ID>/` directory.

## Batch analyze events

Dry run first:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/batch_analyze_spoofing_events_with_llm.py \
  --review-dir outputs/spoofing_event_review/risanamento_top3_timing_window \
  --parameter-grid-root outputs/spoofing_metrics/kappa_lambda_sensitivity_top3 \
  --model gemma4-hermes:latest \
  --limit 3 \
  --dry-run
```

Then run for real, optionally with `--limit 1` for a smoke test.

## Regenerate dashboard with saved reviews

After generating reviews, regenerate the dashboard:

```bash
PYTHONDONTWRITEBYTECODE=1 conda run -n main python scripts/build_spoofing_event_review_dashboard.py \
  --input data/ExportGridData_2026-05-20_090247703_RISANAMENTO_01062024_30112024_FG.parquet \
  --execution-metrics outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/execution_metrics.parquet \
  --candidate-deceptive-orders outputs/spoofing_metrics/risanamento_top3_multilevel_msci_timing_window/candidate_deceptive_orders.parquet \
  --parameter-grid-root outputs/spoofing_metrics/kappa_lambda_sensitivity_top3 \
  --output-dir outputs/spoofing_event_review/risanamento_top3_timing_window \
  --top-n 10 \
  --pre-window-seconds 30 \
  --post-window-seconds 5
```

The dashboard will show the saved LLM review for events with `response.md`; otherwise it shows a message explaining how to generate one.

## Caveats

- The local LLM output is explanatory, not confirmatory evidence.
- The model and LLM must not claim manipulative intent.
- Review `dossier.md`, `prompt.md`, `response.md`, and `metadata.json` together.
- The static dashboard does not execute Ollama. This is intentional for reproducibility and safety.
