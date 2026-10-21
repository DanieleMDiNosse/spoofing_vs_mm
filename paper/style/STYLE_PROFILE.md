# STYLE_PROFILE.md

## Purpose

This file governs exposition only. It does not define scientific claims, notation, model assumptions, data construction, thresholds, or implementation details.

The reference paper is `paper/reference-papers/0809.0822v1.pdf`, *How markets slowly digest changes in supply and demand*. Use it as a model for the progression of an argument: motivate the problem, build intuition, introduce formal objects only when they become necessary, and return from mathematics to interpretation. Do not imitate its sentence length, terminology, rhetoric, or density.

## Governing Objective

Make the scientific argument easy to understand on the first careful reading.

Scientific correctness and scientifically necessary precision are inviolable constraints. Within those constraints, minimize cognitive load and make the reasoning explicit.

If a necessary detail makes the main argument difficult to follow, reorganize or relocate the detail rather than omit, weaken, or blur it.

A passage is not successful merely because every sentence is technically correct. If the reader must reconstruct the rationale unaided, revise the order, the connections, or both.

---

## Core Cognitive Rules

### 1. One inferential move at a time

A sentence should normally perform one main reasoning step.

Avoid combining several of the following in one sentence unless the relationship is genuinely simple:

- motivation;
- definition;
- assumption;
- mechanism;
- empirical finding;
- qualification;
- implication.

If a sentence requires the reader to unpack several logical steps, split it.

Sentence length itself is not the target. Inferential density is.

### 2. Minimize working-memory load

Do not ask the reader to hold several unresolved concepts at once.

Resolve one conceptual dependency before introducing another.

Avoid:

- several unfamiliar acronyms in the same passage;
- definitions whose purpose becomes clear only later;
- long chains of pronouns with ambiguous referents;
- distant qualifications that force reinterpretation of an earlier claim;
- dense noun phrases that compress a relationship into a label.

Repeat a key noun when repetition is clearer than a pronoun.

### 3. Motivation before important paper-specific definitions

Before introducing an important paper-specific object, establish why the reader needs it.

Use this progression when appropriate:

**problem -> need -> object -> formal definition -> interpretation**

Do not introduce terminology merely because it exists in the implementation.

Standard terminology need not be re-motivated when its role is already obvious to the intended reader.

### 4. Intuition before formalization

Before an important equation or construction, explain in ordinary scientific language:

- what question it answers;
- what relationship it formalizes;
- why the relationship matters.

When useful, begin with a stripped-down example, limiting case, or hypothetical sequence.

Then introduce the general notation.

### 5. Do not make the reader reconstruct the argument

State important logical connections explicitly.

Transitions must explain why the next step follows from the current one, not merely announce another topic.

Prefer:

> A cancellation after an execution is not informative by itself because several mechanisms can generate it. We therefore examine the sequence of events that precedes and follows the execution.

over prose that leaves the reader to infer why the second sentence follows from the first.

---

## Paragraphs and Section Narrative

Each paragraph should have one main rhetorical function, such as:

- motivating a problem;
- explaining an intuition;
- defining an object;
- developing a mechanism;
- comparing explanations;
- presenting evidence;
- interpreting a result;
- stating a limitation.

Open with the paragraph's substantive point or local question.

A conceptual paragraph will often follow:

**point -> explanation -> evidence or formalization -> implication**

Start a new paragraph when the rhetorical task changes.

Do not force the main finding, an extended caveat, and a competing interpretation into the same paragraph unless they are tightly connected.

### Section continuity

Begin a major section by identifying the scientific question it addresses and by connecting that question to the preceding argument.

A roadmap can be useful, but a list of forthcoming subsections is not a substitute for motivation.

Cross-references may defer detail, but they must not replace the explanation needed to understand the current passage.

### Selective conceptual reminders

Briefly recall an earlier concept only when it is directly needed for the next step and intervening material has moved the reader's attention elsewhere.

Use the reminder as a bridge to new reasoning, not as a retrospective summary.

Do not re-explain a concept developed in the preceding one or two paragraphs. Build directly on it.

Recall only the part that matters now, normally in a short clause or sentence, and immediately state its new consequence, application, or contrast.

A section or equation reference may locate the original discussion, but it should not replace the conceptual link needed here.

### Competing explanations

When comparing alternative explanations:

1. introduce the alternatives in a stable order;
2. state what each explanation predicts;
3. evaluate them against a common empirical or logical criterion;
4. state what the comparison can and cannot establish.

Do not shift criteria across alternatives without explaining why.

---

## Scientific Tone

Write for a researcher who knows the general field but has not seen this paper, model, dataset, codebase, or notation.

Use direct declarative prose with visible subjects, actions, and claims.

Prefer concrete verbs such as:

- measures;
- compares;
- identifies;
- increases;
- restricts;
- predicts;
- suggests;
- rules out;
- remains consistent with.

Avoid inflated academic filler such as:

- "in order to operationalize the aforementioned distinction";
- "it is worth noting that";
- "with regard to";
- "in the context of the present analysis";
- "the results appear to indicate that" when "the results suggest that" is sufficient.

Do not mistake abstraction for rigor.

Use first-person plural selectively for authorial actions: defining an object, stating an assumption, comparing alternatives, presenting evidence, or drawing an inference.

Use impersonal phrasing for established background and general properties.

Use causal verbs only for an established or explicitly hypothesized mechanism. Describe observational relationships as associations.

Keep qualifications close to the claims they restrict.

---

## Mathematics

Every central equation must be motivated and interpreted.

Before the equation:

1. explain why it is needed;
2. state what it will measure, impose, or predict.

Define unfamiliar symbols immediately before or after the equation, whichever ordering makes the expression easier to read.

Explain symbols in the order that matters to the argument. Explain what important terms do, not only what they denote.

Distinguish clearly among:

- definitions;
- assumptions;
- identities;
- approximations;
- model predictions;
- empirical estimates.

Do not present a descriptive score, selection rule, or diagnostic quantity as though it were a behavioral law.

Use special cases, boundary values, and limiting cases when they expose the mechanism.

After a derivation or central definition, return explicitly to the motivating scientific question.

---

## Results, Figures, and Tables

State the empirical finding before directing the reader to a figure or table.

Prefer:

> Cancellation intensity is substantially higher in the candidate episodes than in the comparison sample (Fig. 4).

over:

> Figure 4 reports cancellation intensity for the two samples.

Whenever the evidence permits, report together:

- direction;
- magnitude;
- relevant population;
- comparison;
- scale;
- scope;
- uncertainty or important limitation.

Describe figures by their scientific implication rather than by walking mechanically through panels or graphical elements.

Emphasize:

- the pattern that matters;
- the range over which it appears;
- deviations from the expected pattern;
- variation across samples, settings, or scales;
- differences in definitions, aggregation levels, or measurement conditions when comparing with previous findings.

Compare theory and evidence through common quantities whenever possible.

Place robustness statements near the main finding and caveats immediately afterward.

End a results discussion with the answer the evidence provides to the motivating question, including a limitation or unresolved issue when that limitation determines the answer.

A descriptive result does not automatically establish a mechanism, causality, or intent.

### Captions

Write captions as compact, self-contained records of what is shown.

Include, when relevant:

- observational unit;
- conditioning or normalization;
- visual encoding;
- fitted relation;
- uncertainty display;
- sample or population scope.

---

## Scientific Detail vs Implementation Detail

The main text must make the scientific argument understandable and must expose the assumptions, definitions, selection choices, and limitations that materially affect interpretation.

Complete reproducibility detail may be placed in appendices, supplementary material, or repository documentation when doing so improves readability, provided that the main text clearly signals the relevant assumption or methodological choice and points the reader to the full specification.

Keep scientifically necessary details explicit at the point where they affect interpretation, including:

- definitions;
- assumptions;
- observational units;
- selection conditions;
- thresholds;
- timing conventions;
- denominators;
- scientifically relevant data limitations.

Move software-specific material elsewhere unless it changes scientific interpretation:

- function names;
- configuration keys;
- paths;
- output-column names;
- internal status labels;
- pipeline stages;
- provenance bookkeeping.

Explain substantive meaning first. Give implementation mappings only where reproducibility requires them.

---

## Mandatory Writing Process

Do not write a difficult subsection directly from the specification.

### Pass 1 — Argument skeleton

Write only the logical argument.

Use roughly one sentence per intended paragraph.

Do not include equations, citations, acronyms, implementation details, or stylistic decoration.

The skeleton must make clear:

- what question is being addressed;
- why the question is difficult;
- what idea resolves or reduces the difficulty;
- how that idea becomes measurable or testable;
- what the evidence can and cannot establish.

If the skeleton is not clear, do not proceed to prose.

### Pass 2 — Reader state

For each paragraph, identify internally:

- what the reader already knows;
- what new idea is introduced;
- why the reader needs it now;
- what the reader should understand afterward.

Do not write these notes into the manuscript.

If a paragraph requires knowledge that has not yet been established, reorder the exposition.

### Pass 3 — Draft

Write the subsection from the argument skeleton.

Prefer explanation over compression.

Do not introduce several technical objects merely because they belong to the same computational procedure.

### Pass 4 — Cognitive-load edit

Rewrite any sentence that:

- contains more than one important inferential step;
- introduces several unfamiliar concepts at once;
- hides a logical relationship inside a noun phrase;
- requires rereading to identify the subject, action, or main claim;
- places an important qualification so far from its claim that the reader must reinterpret the sentence;
- requires unnecessary notation or terminology to remain in working memory.

Rewrite any paragraph whose rationale is implicit rather than stated.

### Pass 5 — Cold read

Read the subsection continuously as if the reader had not seen the code or previous drafts.

Ask:

- Is the reason for each step clear before the step is introduced?
- Can each paragraph be understood without decoding several new concepts at once?
- Does each equation have a visible purpose?
- Does each result answer a previously stated question?
- Is the argument understandable without access to the repository?
- Are conceptual reminders useful bridges rather than repetitions?

If individual sentences are correct but the overall logic is hard to follow, revise the order and explanatory connections. Do not merely polish wording.

---

## Seminar Test

Imagine explaining the subsection to a knowledgeable colleague at a blackboard.

The written version should normally follow the same conceptual order:

**problem -> intuition -> example when useful -> formalization -> evidence -> implication**

The manuscript should remain scientific and precise, but it should not be unnecessarily harder to understand than a careful oral explanation of the same idea.

If the spoken explanation would be substantially clearer than the written one, rewrite the prose.

---

## Conclusions

Move from specific findings to broader conceptual implications.

Organize the synthesis by mechanism, analytical level, or scientific question rather than by the order of figures, tables, or subsections.

Do not merely repeat the results section.

State clearly what has been established, what remains descriptive or suggestive, and what remains unresolved.

---

## Final Acceptance Test

Before considering a subsection finished, verify:

1. Can a field researcher understand the central argument on one careful reading?
2. Does every important paper-specific definition arrive after its purpose is clear?
3. Does each sentence perform a manageable amount of reasoning?
4. Does each paragraph have one main function?
5. Are logical connections stated rather than left for the reader to reconstruct?
6. Are competing explanations compared in a stable and explicit way?
7. Are equations motivated and interpreted?
8. Are findings stated before figures and tables are invoked?
9. Can the paper be understood without knowing the codebase or internal vocabulary?
10. Have scientifically necessary qualifications and reproducibility details been preserved?
11. Are conceptual reminders selective rather than repetitive?
12. Is any passage formally correct but unnecessarily difficult to decode?

If the answer to the last question is yes, rewrite it.

---

## Scope Separation

Keep this file generic.

Project-specific narrative choices, scientific distinctions, terminology, confirmed assumptions, and paper-specific warnings belong in:

`PAPER_NARRATIVE.md`

Formal definitions, thresholds, timing rules, data-field mappings, denominators, and reproducibility specifications belong in:

`METHOD_SPEC.md`

Do not turn `STYLE_PROFILE.md` into a specification of the current project.

---

## Integration Contract for Paper Agents

When editing the paper, apply the instruction files in the following roles:

- `STYLE_PROFILE.md` governs exposition and reader-facing clarity.
- `PAPER_NARRATIVE.md` governs project-specific narrative choices and the interpretation of the spoofing-detection argument.
- `METHOD_SPEC.md` governs only those methodological entries that are populated, versioned, and verified for the analysis being reported.

If a stylistic instruction conflicts with a scientific claim, a verified methodological specification, or the analysis actually reported, do not resolve the conflict by rewriting the science. Flag the inconsistency explicitly.

If `METHOD_SPEC.md` is incomplete or an entry is marked `unresolved`, consult the applicable validated source rather than inferring the rule from surrounding prose or the current implementation.
