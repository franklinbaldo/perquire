# Research contract: semantic inversion

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

## Primary hypothesis: adaptive scaling

As the evaluation budget increases, adaptive candidate generation conditioned on previous similarity feedback should convert its accumulated search history into additional semantic-preimage quality more effectively than simpler equal-budget baselines.

The primary empirical object is therefore the **quality × budget scaling curve**, not whether adaptive wins at one selected budget. A low-budget loss is compatible with the hypothesis if adaptive subsequently improves faster and develops a sustained advantage; conversely, a one-budget win is not sufficient evidence for superior scaling.

For benchmark v1, the preregistered scaling range is evaluation budgets **2, 4, 8, 16, and 32** on the same frozen cases and comparison methods. Extending the range later is allowed as a new experiment, but must not retroactively rescue the v1 scaling claim.

### Predictions

If the adaptive mechanism is useful, aggregate results should show one or more of the following related signatures across the preregistered range:

- stronger improvement in semantic-preimage quality as budget grows than `independent_best_of_n` and `mutation_hill_climber`;
- positive marginal value from additional budget after simpler baselines begin to saturate;
- a crossover followed by a sustained adaptive advantage at larger budgets;
- favorable quality per separately reported resource cost, rather than an apparent gain caused by hidden extra LLM work.

These are scaling signatures, not independent opportunities to declare success. The complete curves and uncertainty across frozen cases must be reported even when they do not support the adaptive mechanism.

## Null / falsification conditions

Within the preregistered 2–32 range, evidence weighs against the adaptive-scaling hypothesis if adaptive:

- remains statistically indistinguishable from or below both simpler baselines without a stronger improving trend;
- improves with budget but at the same or a weaker rate than the baselines;
- saturates earlier than the baselines;
- shows only an isolated crossover/win that is not sustained across larger budgets; or
- obtains higher target similarity only through materially greater separately measured resource use, without a corresponding quality-per-cost advantage.

Failure to find the regime within 2–32 does **not** prove that no larger finite budget could ever help. It falsifies the preregistered v1 claim that useful adaptive scaling is observable in this operational range. Any later larger-budget hypothesis must be registered prospectively and reported as a new experiment.

A negative result is an accepted project result and should trigger simplification or a new explicitly preregistered hypothesis rather than benchmark changes designed to preserve the method.

## Secondary mechanism hypothesis

The scaling advantage, if observed, is hypothesized to come from useful information in accumulated similarity feedback rather than merely from repeated generation. After the primary scaling benchmark is stable, this mechanism should be tested by an ablation that removes or destroys useful feedback information while preserving comparable evaluation budget and generation opportunity.

If the ablated strategy scales comparably to the intact adaptive strategy, that is evidence against the proposed feedback mechanism even if both outperform a baseline.

## Required baselines

1. `independent_best_of_n`: candidate generation without target-score feedback; choose the best scored candidate.
2. `mutation_hill_climber`: mutate the current best candidate and retain improvements.
3. `adaptive_perquire`: the current feedback-conditioned strategy.

Additional methods may be added, but these three remain frozen comparison points once benchmark v1 is released.

## Budget fairness

Methods are compared under the same maximum number of target similarity evaluations at each preregistered budget. LLM calls, generated candidate counts, transport attempts, wall-clock time, token/cost data when available, and model/provider configuration are recorded separately so apparent gains cannot be hidden in unequal resource use.

Cache/replay may avoid purchasing an already observed provider response, but replayed observations are not fresh samples. Logical experimental calls and real transport attempts must remain distinguishable.

## Generation substrate v1

Before observing any 2→32 scaling curve, a target-free reliability probe selected the generation substrate by a criterion independent of adaptive quality. Both eligible candidates completed 12/12 logical calls; the prospectively frozen lower-price tie-break selected `google/gemini-2.5-flash-lite` through OpenRouter. The decision artifact is documented in `docs/openrouter_reliability_probe_v1_result.md`.

For scaling v1:

- generation model is frozen to `google/gemini-2.5-flash-lite` across all methods, budgets and replicates;
- temperature remains `0.7` and the existing bounded retry policy remains `max_retries=2`;
- a required generation that exhausts retries invalidates the experimental cell;
- an invalid cell remains counted as invalid and may not be erased by silent model switching or rerun-until-success;
- exact replay may preserve already observed provider responses but is not a fresh stochastic sample;
- if fewer than 95% of preregistered cells are valid, v1 fails its operational-reliability gate and no clean scaling claim is made from the surviving subset.

Changing the generation model or failure policy after observing scaling results defines a new experiment version rather than repairing v1 retrospectively.

## Evaluation

The optimizer's target cosine is necessary but not sufficient. Benchmark reports must distinguish:

- best target cosine by evaluation step;
- final target cosine;
- full per-step trajectory sufficient to reconstruct quality × budget curves;
- aggregate central tendency and uncertainty across frozen cases at each budget;
- marginal quality gain when budget increases, including each budget doubling;
- area under the quality × budget curve (or an explicitly versioned equivalent summary);
- semantic similarity between recovered and hidden source text using an evaluator not identical to the optimization signal when practical;
- retrieval-neighborhood overlap where a reference corpus exists;
- evaluation count, logical LLM-call count, and real transport-attempt count;
- token/cost accounting when available;
- failure/invalid-output rate;
- per-domain and aggregate results.

No single aggregate replaces the raw curves. Means alone are insufficient to establish scaling when heterogeneous cases can hide saturation or isolated wins.

## Benchmark integrity

- Hidden source text is unavailable to the search strategy during a run.
- The v1 scaling budgets are frozen at `2, 4, 8, 16, 32` before observing the scaling sweep.
- The same frozen benchmark cases and required baselines are used at every budget in the scaling comparison.
- Dataset split, seed, embedding model, provider/model identifiers, prompt/strategy version, and budget are persisted.
- Raw per-step observations are retained so convergence curves can be reconstructed.
- Benchmark definitions are versioned separately from result records.
- Algorithm PRs must evaluate against a benchmark definition committed before the algorithm change when possible.
- Method changes prompted by observed benchmark outcomes require a new prospective experiment; they do not rewrite the interpretation of already observed results.

## Cross-project boundary

- `perquire` owns executable experiments, benchmark data contracts, algorithms, and results.
- `franklinbaldo/papers` owns broader theoretical claims and may cite Perquire results after they exist; Perquire must not turn paper hypotheses into empirical claims automatically.
- `franklinbaldo/okf-parser` is used as a validator/query surface for the small experiment-memory bundle. It is not part of the search algorithm and benchmark execution must remain possible without using OKF as an algorithmic dependency.
