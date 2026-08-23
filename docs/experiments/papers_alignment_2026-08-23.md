# Perquire × `franklinbaldo/papers` experimental alignment — 2026-08-23

Status: **RESEARCH MAP / NO TARGET-SCORED CHANGE**

This note checks whether the active Perquire experimental programme still matches the broader research programme in `franklinbaldo/papers`, especially the newer machine-interaction family. It is a scope-control document, not a new empirical result and not an amendment to the frozen Gate-B primary question.

## 1. Active Perquire claim remains correctly narrow

Perquire causal-feedback v2 asks whether target-relevant scalar feedback causes better downstream search trajectories than target-irrelevant feedback under an otherwise shared proposer mechanism.

That remains the right first gate. The newer papers strengthen the case for **not** collapsing this mechanism claim into stronger claims about teaching, progressive decodability, semantic recovery, discovery, or agency.

Therefore this review does **not** add arms to Gate B and does not alter its primary estimand or kill criterion.

## 2. `generative_machine_teaching.md`

Relevant paper claim: procedural order is not exchangeable. Under matched content, ordered demonstrations should be compared with shuffled/flat exposure; if shuffled exposure performs equally, a claim that sequential structure itself teaches the protocol weakens.

### Already covered

- Gate B uses a chronological history rather than a bag of independent scores.
- Checkpoints are prefixes of one nested trajectory, so later steps inherit earlier state rather than restarting at each budget.
- `true_feedback`, `decoy_feedback`, and `null_feedback` isolate whether correct-target scalar information matters before asking how that information is organized.

### Not yet covered — deliberately downstream

Gate B does not test whether **chronological order / candidate-feedback sequence structure** is itself necessary once target-relevant feedback has been shown useful.

If Gate B is positive, preregister a matched-content follow-up with the same candidate-feedback rows but a prospectively fixed history-order intervention, such as:

- ordinary chronological true-feedback history;
- the same history rows under deterministic shuffled presentation order;
- optionally a separate feedback-binding permutation that preserves the score multiset while destroying candidate↔score association.

This follow-up must not rescue a negative Gate B result. Its question is second-order: *how is useful feedback organized and consumed?*

## 3. `pedagogical_signal_extraction.md` / Structured Irregularity

Relevant paper claims:

- retrospective fit is not enough; learned structure should support future prediction, transfer, perturbation and intervention;
- temporary opacity should be distinguished from noise by whether later structure makes earlier evidence causally useful;
- progressive decodability is a trajectory property, not a final-score property.

### Already covered

- Gate B keeps the full per-step trajectory and forbids a success claim based only on the final maximum.
- It reports AUC, improvement frequency, mean/median candidate score and best-so-far curves.
- `true` vs `decoy` vs `null` is already an intervention on the information channel, not merely a correlational trajectory analysis.

### Not yet covered — downstream diagnostics

A positive Gate B would still not establish progressive decodability in the stronger paper sense. That would require prospective history interventions such as masking/permuting earlier feedback and measuring whether later candidate quality degrades, plus transfer to new targets or evaluation conditions.

Treat these as mechanism diagnostics after Gate B, not as post-hoc reinterpretations of Gate B.

## 4. `forbidden_relay.md`

Relevant experimental discipline:

- distinguish the optimized receiver/channel from independent-observer decodability;
- test held-out targets and model transfer;
- use perturbations and receiver/model swaps to distinguish robust representation from private co-adaptation;
- record complete model/memory identity and prevent side channels;
- group inference by target rather than treating repeated episodes as independent targets.

### Already covered

- Perquire treats target as the primary inferential unit.
- The hidden source does not reach the proposer.
- generation and embedding model/provider routing are frozen and recorded before Gate B;
- cache/replay and transport accounting are explicit;
- semantic recovery is explicitly deferred to an evaluator that never feeds search.

### Required strengthening of the semantic gate

Issue #67 should not stop at “another embedding score”. A semantic-recovery claim should prospectively require at least:

1. a held-out evaluator that never supplies optimization feedback;
2. **unseen target cases** not used to shape the v2 prompt/design;
3. at least one representation/model-transfer condition (for example held-out embedder or independently justified semantic judge);
4. perturbation robustness sufficient to show that the result is not a brittle lexical/embedding-space exploit;
5. an independent-observer style analysis when practical: can an evaluator not co-adapted with the search recover the intended semantic distinctions?

A result that exists only in the optimized embedding space remains an embedding-space search result, not demonstrated semantic recovery.

## 5. `machine_discovery.md`

Relevant discipline: a strong scientific claim is not just an emitted result. Correctness/certification, novelty, provenance, public acceptance and downstream usefulness are distinct properties. Verification independence is graded, and common-mode dependencies should be represented rather than hidden.

### Already covered

Perquire now records software environment, model/routing identity, raw trajectories, failure evidence and preregistration boundaries. Negative evidence is preserved rather than rerun away.

### Still missing for a broad published claim

A successful Gate B run is a **candidate empirical result**, not by itself evidence that Perquire is generally established. Before a broad external claim, require a separate confirmatory/replication stage with:

- frozen prior result and analysis plan;
- a fresh target corpus or other declared external-validity dimension;
- independent or heterogeneous evaluation where feasible;
- provenance sufficient to reconstruct model/provider/software/artifact lineage;
- explicit statement of common-mode dependencies that remain (OpenRouter, model families, embedding providers, etc.).

This is especially important because the active 24-case English corpus is intentionally narrow.

## 6. `embedding_seeded_tournament.md` and `empirical_evaluation.md`

The recent ESHTR edits reinforce a general lesson relevant to Perquire: a calibration source can itself have ground-truth error or domain-dependent reliability. Calibration does not become truth merely because it is frozen.

Perquire's fixed paraphrase calibration should therefore be described as **construct/calibration probes**, not a semantic gold standard. Its purpose is to characterize the measurement surface (including domain heterogeneity), not certify that a given cosine corresponds to a universal semantic distance.

This is already consistent with the v2 construct-validity boundary; future reports should preserve that language.

## 7. `machine_interaction_program.md` and `informational_time.md`

The machine-interaction map explicitly warns against collapsing causal depth, hop count, symbolic description length and recoverable informational depth.

Perquire's step count is an **evaluation/generation budget**, not a claim about informational time or causal depth. The current v2 wording should stay that way. A B=16 trajectory demonstrates sixteen proposer opportunities, not sixteen equal units of causal work.

If future work claims that feedback history itself forms a learned representation, causal contribution should be tested by intervention on history, not inferred from the number of steps alone.

## 8. Alignment verdict

| Paper-derived requirement | Perquire state | Action |
|---|---|---|
| isolate useful target feedback causally | **covered** | Gate B true/decoy/null |
| matched-content order/history ablation | **not yet, correctly downstream** | preregister only after positive Gate B |
| trajectory rather than final-max evidence | **covered** | keep AUC/distributions/improvement metrics |
| independent semantic evaluator | **partially covered** | strengthen #67 with held-out targets, transfer, perturbation |
| independent-observer / private-code distinction | **partially covered** | add to semantic follow-up where practical |
| provider/model/provenance identity | **covered strongly** | keep explicit routing freeze and artifact lineage |
| external replication before broad generalization | **missing** | create post-Gate-B replication gate |
| calibration is not ground truth | **conceptually covered** | make explicit in future reports |
| step budget ≠ causal/informational depth | **covered by scope** | do not relabel scaling as informational time |

## 9. Program order after this review

```text
target-free substrate qualification
        ↓
explicit substrate freeze
        ↓
Gate B: true vs decoy vs null
        ↓
if negative: stop / weaken scalar-feedback mechanism
        ↓ if positive
matched-content history/order mechanism ablation
        ↓
held-out semantic transfer + perturbation / independent observer
        ↓
external replication / provenance-aware confirmation
        ↓
only then broaden from “search signal in one embedding space”
to a stronger semantic-inversion claim
```

This ordering intentionally treats the papers as sources of **new falsifiers and scope boundaries**, not as reasons to add complexity before the simpler causal hypothesis survives.