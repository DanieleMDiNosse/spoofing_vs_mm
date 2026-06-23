# Spoofing Detection Production-Readiness Workflow

## Purpose

This workflow promotes event-level spoofing-like detections into analyst-reviewable client-session alerts.

The detector does not infer legal intent. It produces surveillance cues for human review.

## Layers

1. Event-level DWI/SCI/MSCI/MCPS metrics.
2. Analyst annotations.
3. Client-session repeated-pattern features.
4. Legitimate market-maker baseline features.
5. Negative controls and placebo checks.
6. Threshold calibration by analyst workload and false-positive pressure.
7. Client-session alert objects.

## Recommended order of use

```bash
# 1. Build/refresh event dashboard outputs.
# 2. Bootstrap annotation CSV.
# 3. Analysts edit annotation CSV.
# 4. Compute client-session features.
# 5. Build negative-control report.
# 6. Build calibration report.
# 7. Run production-readiness pipeline.
# 8. Regenerate dashboard with annotation and alert files.
```

## Scientific interpretation

A production alert should be considered stronger when:

- suspicious episodes repeat for the same client/session;
- MCPS remains high across depth choices;
- the pattern is robust over kappa/lambda values;
- opposite-side collapse is stronger than same-side collapse;
- negative controls score lower than real events;
- the behavior is unusual relative to the client's own baseline and peer market makers.

## Non-goals

- The LLM does not decide manipulation.
- The event-level score alone is not a legal conclusion.
- Thresholds are not final until calibrated against labels and analyst workload.
