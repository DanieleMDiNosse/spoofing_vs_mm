# Paper writing instructions

When editing the manuscript, read and apply the documents in `paper/style/`:

- `paper/style/STYLE_PROFILE.md` governs exposition, readability, paragraph structure, mathematical explanation, and the mandatory writing/revision process.
- `paper/style/PAPER_NARRATIVE.md` governs narrative choices specific to the spoofing-detection paper, including the interpretation of observable sequences, executions/episodes, passive/aggressive roles, and the distinction between compatibility and manipulation or intent.
- `paper/style/METHOD_SPEC.md` is a methodological specification only for entries that are populated, versioned, and marked as verified for the analysis reported in the manuscript. Empty or unresolved entries are not evidence that a rule is absent.

Scientific correctness and scientifically necessary precision take precedence over stylistic convenience. If a writing instruction conflicts with a verified methodological source, a validated analysis version, or the results actually reported, flag the conflict instead of resolving it through stylistic rewriting.

Do not infer missing methodological rules from an incomplete `METHOD_SPEC.md`, and do not silently substitute the current codebase for the validated analysis version used by the paper.
