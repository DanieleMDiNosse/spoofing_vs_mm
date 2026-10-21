# Style Profile

This profile captures exposition only. It does not import subject matter, terminology, notation, citations, or scientific claims from the reference material.

The reference is `paper/reference-papers/0809.0822v1.pdf`, *How markets slowly digest changes in supply and demand*. Its useful model is the progression of an argument, not sentence length or textual density. In particular, the opening moves from a familiar economic problem to the difficulty that motivates the analysis; Section 3 develops competing explanations through verbal reasoning and examples before assessing their implications. Adapt these explanatory habits to this paper rather than copying the reference's scale, rhetoric, or theoretical claims.

## Governing Principle: Guide the Reader

- Write a scientific argument, not a technical report, an implementation specification, or an audit log. The reader must understand the question, the proposed model, and the meaning of the results without consulting the code.
- Assume a researcher who knows the general field but has not seen this model, dataset, or notation. Explain paper-specific concepts when they become necessary; do not introduce them merely because they exist in the implementation.
- Build the narrative around a connected progression: what phenomenon needs explanation; why existing observations are insufficient; what idea motivates the model; how the model makes that idea measurable; what the evidence shows; what remains unresolved. This is a logical progression, not a mandatory list of headings or a template to repeat in every paragraph.
- Make each new definition answer a question the preceding text has made understandable. Explain the need for a quantity before giving it a name, acronym, or formula. A catalogue of correctly defined quantities is not an explanation of the model.
- Prefer adding a needed explanatory sentence to compressing several conceptual steps. Clarity does not mean merely shortening the text, removing mathematics, or replacing precise terms with vague ones.
- Keep a coherent line of reasoning across paragraphs and sections. Transitions should explain why the next step is needed, not merely announce that another topic follows.
- When stylistic imitation conflicts with comprehension, choose comprehension. Read this profile together with `paper/AGENTS.md`; neither long sentences nor dense paragraphs are goals.

## Narrative of the Present Paper

- Start from the observable economic sequence and the ambiguity that makes it difficult to interpret: displayed liquidity, an execution on the opposite side, and subsequent withdrawal can be consistent with more than one explanation. Distinguish the hypothesized strategy from what the records actually reveal.
- Explain how each part of the model addresses that ambiguity before presenting its formal definition. Make clear which quantities describe the book, which reconstruct a sequence, which impose screening conditions, and which prioritize review. Do not introduce all their acronyms at once.
- Use a simple, explicitly hypothetical sequence when it helps the reader follow the participants, book sides, and order of events. It must illustrate the verified model, not add an assumption or masquerade as an empirical example.
- Explain the reason for grouping observations before introducing the grouping terminology. Preserve the distinctions between messages, executions, and episodes; once fill aggregation has been defined, use “execution” in ordinary prose and reserve “execution cluster” for passages where that technical distinction matters.
- Organize empirical discussion around scientific questions and findings, not the order in which scripts run or output files are produced. Explain what a comparison can establish before interpreting its values.
- Keep the distinction between compatibility with a hypothesized sequence and evidence of manipulation explicit. The narrative must not turn a descriptive detector into a causal model or a test of intent.

## Scientific Detail Versus Implementation Detail

- Keep the scientific method reproducible: definitions, assumptions, selection conditions, thresholds, timing conventions, observational units, denominators, and limitations must remain explicit. Explain them where the reader needs them rather than repeating the entire specification in the abstract, introduction, and results.
- Give the abstract and introduction the question, central idea, principal evidence, and scope. They should not be compressed inventories of all operational rules. A high-level description is acceptable; a purportedly complete definition must include every required condition and agree with the formal method.
- Use scientific names in the main exposition, not function names, configuration keys, output-column labels, paths, bundle versions, or internal status labels. For example, explain a comparison period in words rather than making a label such as `no_own_observed` carry its meaning.
- When a source-field distinction affects interpretation, explain its substantive meaning first. Give the exact field mapping once in the data or reproducibility material. Do not conceal scientifically relevant clock or attribution assumptions by treating them as mere software details.
- Place file mappings, software invocation, provenance records, and repetitive bookkeeping in appropriate tables, appendices, supplementary material, or repository documentation. Keep any detail essential to evaluating a main claim accessible at the point of that claim.
- State the general scope limitation (compatible sequences are not confirmed spoofing or proof of intent) in the introduction and recall it in the conclusions. The abstract may state its scope briefly because it must stand alone. Do not repeat this general disclaimer throughout the model, instrument-level results, captions, or descriptions of passive/aggressive roles.
- Retain local qualifications only when they explain a specific mathematical or evidential restriction, such as an uncalibrated score, a descriptive covariance, unequal sample coverage, or a missing counterfactual. Express these restrictions precisely rather than appending generic warnings about manipulation to every result.
- Treat the passive/aggressive decoding used in this project as confirmed by CONSOB, as reported by the author. Define the role meanings once in the data section; do not call that mapping uncertified or unverified. This confirmation concerns execution roles, not the separate clock-origin assumptions.

## Sentence Construction

- Prefer direct declarative sentences with a visible subject, action, and principal claim. Use a longer sentence only when its reasoning remains linear and easy to understand on first reading; length is not a marker of scientific maturity.
- Unfold multi-step reasoning across sentences. Vary sentence length naturally, without alternating density and brevity as a stylistic exercise.
- Make logical relations explicit near the beginning of the sentence with constructions such as cause, contrast, condition, or consequence. Avoid leaving the relation between adjacent claims implicit.
- Use first-person plural selectively for authorial actions: defining an object, stating an assumption, comparing alternatives, presenting evidence, or drawing an inference. Use impersonal phrasing for established background and general properties.
- Introduce an unfamiliar concept through a direct definition or an appositive explanation at first use. Follow the formal name with an ordinary-language restatement when the concept is central.
- Use rhetorical questions sparingly to isolate a genuine scientific problem. Answer the question immediately through an explanation, model, or empirical test rather than using it decoratively.
- Keep qualifications close to the claim they restrict. Use parentheses or dashes for secondary clarification, but place essential assumptions and limitations in the main sentence.
- Prefer concrete causal verbs and explicit referents. Repeat the key noun when needed instead of relying on ambiguous pronouns across a dense argument.
- Use causal verbs only for an established or explicitly hypothesized mechanism; describe observational associations as associations.
- Avoid stacked noun modifiers and chains of hyphenated qualifiers. Explain relationships with verbs and clauses instead of compressing them into labels such as “age-conditioned candidate-order reconstruction”. Retain established technical terminology when it is necessary and already explained.

## Paragraph Organization

- Give each paragraph one clear function: motivate a problem, define an object, develop a mechanism, compare explanations, present evidence, interpret a result, or state a limitation.
- Open with the paragraph's substantive point or local question. Develop it through explanation and evidence, then close with its implication or with a transition to the next step.
- Build conceptual passages from familiar intuition to the unresolved difficulty and only then to the proposed formal account. This progression makes the motivation for the mathematics visible.
- Organize comparisons of competing explanations explicitly. Introduce the alternatives in a stable order, state what each predicts, and evaluate them against a common empirical or logical criterion.
- Begin major sections by identifying the scientific question they address and connecting it to the preceding argument. Add a roadmap only when it helps; a list of forthcoming subsections is not a substitute for motivation. Cross-references may defer detail, but not the explanation needed to understand the current passage.
- Give each paragraph enough space to develop its point without requiring the reader to unpack several new concepts at once. Start a new paragraph when the rhetorical task changes. In particular, separate the main result from an extended caveat or alternative interpretation.
- Let conclusions move from specific findings to broader conceptual implications. Group the synthesis by analytical level or mechanism rather than repeating the order of figures and subsections.

## Selective Conceptual Reminders

- Briefly recall an earlier concept or argument when it is directly needed to understand the next step and intervening material has moved the reader's attention elsewhere. Use the reminder as a bridge to new reasoning, not as a retrospective summary.
- Do not re-explain a concept developed in the preceding one or two paragraphs. Build directly on it. Distance alone does not justify a reminder: the earlier idea must have a specific role in the forthcoming argument.
- Recall only the part that matters now, normally in a short clause or sentence, and immediately explain its new consequence, application, or contrast. Do not repeat the full definition, derivation, or list of conditions.
- Prefer a substantive connection over stock phrases such as “as discussed above”. A section or equation reference may locate the original explanation, but should not replace the brief conceptual link needed here.
- Check whether removing the reminder would make the next inference harder to follow. If it would not, omit it. Preserve the original scope and evidential status: recalling a hypothesis does not make it an established result.

## Integration of Intuition and Mathematics

- Motivate every important equation before displaying it. State the question the equation answers, the relationship it formalizes, or the inconsistency it is intended to resolve.
- Introduce the simplest informative case first. Use a verbal scenario, stripped-down model, or limiting example to make the mechanism intuitive before presenting a more general formulation.
- Define symbols immediately before or after the equation in the order in which they matter to the argument. Explain what each term does, not merely what it denotes.
- Treat a derivation as a sequence of explained scientific steps. Where a model yields predictions, connect assumptions, formal relations, and testable implications. Where a formula instead defines a descriptive quantity or selection rule, explain its construction and role without presenting it as a derived behavioral law.
- Interpret each central equation in prose. State the sign, scale, monotonicity, asymptotic behavior, or constraint that carries the scientific meaning.
- Use special cases and boundary values to expose the mechanism. Explain what changes when a correlation vanishes, a parameter reaches a threshold, a horizon becomes large, or an effect is absent.
- Distinguish definitions, assumptions, identities, approximations, predictions, and empirical estimates. Do not allow algebraic continuity to blur differences in their evidential status.
- Return from the mathematics to the motivating question. Explicitly state whether the formal result supports the proposed mechanism, rules out an alternative, or identifies a condition that must hold.

## Presentation of Results

- State the empirical finding before directing the reader to a figure or table. The visual should serve as evidence for a claim already expressed in words.
- Report direction, magnitude, scale, and scope together when the evidence permits. Distinguish a robust qualitative pattern from the exact functional form or numerical estimate used to summarize it.
- Describe figures by their scientific implication rather than by walking through panels or graphical elements. Emphasize the pattern that matters, the range over which it appears, and any departure from the expected behavior.
- Compare theory and evidence directly through common quantities. Present the predicted relationship, the observed estimate, the degree of agreement, and the discrepancy that remains.
- Discuss variation across samples, settings, or scales as part of the result. When previous findings differ, first ask whether the studies use different definitions, aggregation levels, or measurement conditions.
- Place robustness statements near the main finding and caveats immediately afterward. Identify assumptions that may weaken the interpretation, omitted dependencies, measurement limits, and ranges with large uncertainty.
- Use calibrated language. Reserve strong conclusions for direct tests under explicit assumptions; use qualified wording for suggestive patterns, model-dependent inversions, and extrapolations.
- Write captions as compact, self-contained records of what is shown: the observational unit, conditioning or normalization, visual encodings, fitted relation, uncertainty display, and sample scope when relevant.
- End a results discussion with the answer the evidence provides to its motivating question, including an unresolved issue when that is the justified conclusion. Explain a mechanism only when supported; a descriptive result need not establish one. Avoid ending with a bare figure reference or an isolated numerical estimate.

## Acceptance Test Before Revising the Manuscript

- Read the opening sentences of consecutive paragraphs. Do they advance an intelligible argument, rather than list components of a detector?
- At each new concept, can the reader explain why it is needed before encountering its notation or implementation?
- After each central equation, can the reader say what it measures or imposes, why it has that form, and how it contributes to the question?
- For each result, can the reader identify the finding, the relevant population and comparison, and what follows from it without reconstructing the claim from a table?
- Can the argument be understood without access to the repository or familiarity with its internal vocabulary?
- Have scientific distinctions and necessary qualifications been preserved without making every sentence carry the full methodological specification?
- Read the prose continuously, not just sentence by sentence. If all sentences are locally correct but the reader must assemble the rationale unaided, revise the order and explanatory connections, not only the wording.
- Check conceptual reminders in context: do they reconnect a genuinely earlier idea to a new step, without repeating explanations from the preceding one or two paragraphs?
