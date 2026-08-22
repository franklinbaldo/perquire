# Research contract: semantic inversion

## Current program status

The original **v1 adaptive-scaling hypothesis is historical, not the active primary claim**. Scaling v1 recovered all 1080 preregistered cells but only 78 were valid (~7.22%), far below its frozen 95% operational gate. V1 therefore produced strong operational/methodological evidence but no clean comparative adaptive-scaling estimate.

The active preregistered scientific gate is now the **causal-feedback v2 mechanism experiment** in `docs/experiments/causal_feedback_v2_preregistration.md`: holding proposer mechanism and exogenous resources fixed, does target-relevant scalar feedback improve downstream search trajectories relative to prospectively paired decoy feedback and null feedback?

The target-free OpenRouter reliability observatory and `docs/experiments/openrouter_reliability_freeze_v2.md` qualify/freeze the generation substrate only. They must not use Perquire target scores to choose the substrate. A dynamic observatory winner is not itself the scientific substrate.

A positive causal-feedback result establishes only useful search information in the optimized embedding space. The stronger semantic-recovery claim still requires a separately frozen held-out evaluator.

## Problem

Perquire receives a target vector `v` produced by a known embedding function `E` and searches natural-language candidates `x` using only evaluations of a similarity function such as:

`score(x) = cosine(E(x), v)`

The target text that produced `v`, when one exists, is hidden from the search procedure.

The objective is **not** to recover the unique original text. Embedding maps are many-to-one for practical purposes, so the target is a useful **semantic preimage**: text whose representation is close to `v` and whose meaning preserves independently measured properties of the hidden source.

## Assumptions

- The embedding model is known, or the search uses a separately justified compatible/calibrated space.
- Raw coordinates are model-relative; dimension equality does not imply semantic interoperability.
- A high cosine score for a natural-language probe is not a yes/no answer from the vector.
- Similarity is an optimization signal, not ground truth about semantic recovery.

These assumptions align Perquire with the representation-geometry cautions developed in `franklinbaldo/papers/semantic_atlas.md`: native embedding coordinates are not universal and stronger cross-model claims require calibration and held-out validation.

## Historical v1 hypothesis: adaptive scaling

As the evaluation budget increases, adaptive candidate generation conditioned on previous similarity feedback should convert its accumulated search history into additional semantic-preimage quality more effectively than simpler equal-budget baselines.

The primary empirical object for v1 was the **quality × budget scaling curve**, not whether adaptive won at one selected budget. The preregistered scaling range was evaluation budgets **2, 4, 8, 16, and 32** on the same frozen cases and comparison methods.

V1 failed its operational-reliability gate and must not be rescued by survivor analysis, model switching, deleted cases/budgets, or rerun-until-success. Its surviving cells are descriptive only.

### Historical predictions

If the adaptive mechanism were useful under the v1 complete-policy comparison, aggregate results would show one or more of the following related signatures across the preregistered range:

- stronger improvement in semantic-preimage quality as budget grows than `independent_best_of_n` and `mutation_hill_climber`;
- positive marginal value from additional budget after simpler baselines begin to saturate;
- a crossover followed by a sustained adaptive advantage at larger budgets;
- favorable quality per separately reported resource cost, rather than an apparent gain caused by hidden extra LLM work.

These signatures remain part of the historical v1 contract only. They are not the active causal-identification criterion for v2.

## Active v2 mechanism hypothesis

The active question is narrower and causal:

> Does feedback that contains information about the **correct target** improve future candidate trajectories relative to target-irrelevant feedback, when proposer code/prompt structure, generation opportunities, model configuration, history format/access, parser, and target-evaluation budget are otherwise held fixed?

The frozen arms are:

1. `true_feedback`: candidate feedback is cosine against the correct target;
2. `decoy_feedback`: candidate feedback is cosine against a prospectively paired wrong target;
3. `null_feedback`: candidate feedback is constant `0.0`.

All outcomes are always scored against the correct target. Candidate texts may diverge after treatment; that divergence is a causal mediator, not an exogenous design difference.

Evidence weighs against the mechanism if true feedback does not outperform both decoy and null feedback in target-paired trajectory measures, or if an apparent final-maximum advantage disappears in trajectory-sensitive measures.

The exact active design, staged sample, failure policy, inferential unit, prohibited post-outcome changes, and kill criterion are frozen in `docs/experiments/causal_feedback_v2_preregistration.md`.

## Required v1 baselines

The v1 complete-policy comparison used:

1. `independent_best_of_n`: candidate generation without target-score feedback; choose the best scored candidate.
2. `mutation_hill_climber`: mutate the current best candidate and retain improvements.
3. `adaptive_perquire`: the original feedback-conditioned strategy.

They remain historical frozen comparison points. They are **not** substitutes for the v2 causal ablation, because the three policies differ in more than scalar-feedback information.

## Budget and resource fairness

For the active v2 causal experiment, all arms receive the same number of logical generation opportunities and the same correct-target evaluation budget. LLM calls, generated candidate counts, transport attempts, wall-clock time, token/cost data when available, and model/provider configuration remain separately observable.

Checkpoint budgets are prefixes of one nested trajectory, not independently generated runs with budget-dependent policies.

Cache/replay may avoid purchasing an already observed provider response, but replayed observations are not fresh samples. Gate-B generation uses `fresh`; logical experimental calls and real transport attempts remain distinguishable.

## Generation substrate v1

Before observing any v1 2→32 scaling curve, a target-free reliability probe selected `google/gemini-2.5-flash-lite` through OpenRouter after eligible candidates completed 12/12 calls. The later v1 scaling run falsified the sufficiency of that qualification criterion.

For scaling v1:

- generation model was frozen to `google/gemini-2.5-flash-lite`;
- temperature was `0.7` and bounded retries `max_retries=2`;
- exhausted generation invalidated the experimental cell;
- invalid cells remained counted and were not silently rerun or switched;
- exact replay was not a fresh stochastic sample;
- fewer than 95% valid cells blocked a clean v1 claim.

The recovered result was ~7.22% valid, so the clean claim is ineligible.

## Generation substrate v2

V2 separates **target-free substrate qualification** from the scientific mechanism experiment.

- The scheduled observatory may dynamically discover free models for qualification only.
- Prospective eligibility is governed by `docs/experiments/openrouter_reliability_freeze_v2.md`.
- Evidence is candidate-specific and begins only after its frozen prospective timestamp.
- A specific generation model, upstream provider path, fallback policy, temperature, max tokens, retries, pacing/concurrency, environment, cache semantics, and validity threshold must be recorded before Gate B.
- The scientific run cannot dynamically rediscover or switch substrate.
- OpenRouter provider routing must be explicitly constrained; a model slug alone is not treated as inference-function identity.

If no substrate satisfies the target-free rule, Gate B does not run.

## Evaluation

The optimizer's target cosine is necessary for the mechanism test but not sufficient for semantic recovery.

Active v2 reports, by target and checkpoint:

- all candidate correct-target scores;
- best-so-far curve;
- mean and median candidate score;
- per-step improvement over prior best;
- AUC of best-so-far over steps;
- improvement frequency;
- paired `true - decoy` and `true - null` effects;
- logical LLM calls and transport attempts;
- failure/invalid-output rate;
- per-domain and target-level raw results.

Targets are the primary inferential units. Provider replicates are repeated stochastic trajectories within target and may not be naively counted as independent targets.

No success claim may rely only on a final maximum, because maxima can reward candidate-score variance.

A later semantic-recovery experiment must use an evaluator not identical to the optimization signal, such as a held-out embedding space, independent semantic judge, retrieval-neighborhood measure, or human evaluation under a separately frozen protocol.

## Benchmark integrity

- Hidden source text is unavailable to the proposer during a run.
- Benchmark definitions are versioned separately from result records.
- Raw per-step observations are retained so trajectories are reconstructible.
- Arm order is deterministically varied prospectively rather than fixed to provider time.
- Model, upstream routing constraints, environment, prompt/strategy version, cache semantics, and accounting are persisted.
- Failure preserves completed observations while keeping the failed trajectory invalid for the primary endpoint.
- Method changes prompted by observed outcomes require a new prospective experiment; they do not rewrite already observed results.
- Contrastive probes and other algorithmic additions remain out of v2 unless prospectively registered as a later experiment; they cannot rescue a negative causal-feedback result.

## Cross-project boundary

- `perquire` owns executable experiments, benchmark data contracts, algorithms, and results.
- `franklinbaldo/papers` owns broader theoretical claims and may cite Perquire results after they exist; Perquire must not turn paper hypotheses into empirical claims automatically.
- `franklinbaldo/okf-parser` is used as a validator/query surface for the small experiment-memory bundle. It is not part of the search algorithm and benchmark execution must remain possible without using OKF as an algorithmic dependency.
