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

## Primary hypothesis

Under an equal evaluation budget, adaptive candidate generation conditioned on previous similarity feedback reaches a better semantic preimage than independent candidate sampling.

## Null / falsification condition

The adaptive mechanism is not justified if a simpler equal-budget baseline such as independent best-of-N sampling or a mutation hill-climber matches or exceeds it within the uncertainty of the benchmark.

A negative result is an accepted project result and should trigger simplification rather than benchmark changes designed to preserve the method.

## Required baselines

1. `independent_best_of_n`: candidate generation without target-score feedback; choose the best scored candidate.
2. `mutation_hill_climber`: mutate the current best candidate and retain improvements.
3. `adaptive_perquire`: the current feedback-conditioned strategy.

Additional methods may be added, but these three remain frozen comparison points once benchmark v1 is released.

## Budget fairness

Methods are compared under the same maximum number of target similarity evaluations. LLM calls, generated candidate counts, wall-clock time, and model/provider configuration are recorded separately so apparent gains cannot be hidden in unequal resource use.

## Evaluation

The optimizer's target cosine is necessary but not sufficient. Benchmark reports must distinguish:

- best target cosine by evaluation step;
- final target cosine;
- semantic similarity between recovered and hidden source text using an evaluator not identical to the optimization signal when practical;
- retrieval-neighborhood overlap where a reference corpus exists;
- evaluation count and LLM-call count;
- failure/invalid-output rate;
- per-domain and aggregate results.

## Benchmark integrity

- Hidden source text is unavailable to the search strategy during a run.
- Dataset split, seed, embedding model, provider/model identifiers, prompt/strategy version, and budget are persisted.
- Raw per-step observations are retained so convergence curves can be reconstructed.
- Benchmark definitions are versioned separately from result records.
- Algorithm PRs must evaluate against a benchmark definition committed before the algorithm change when possible.

## Cross-project boundary

- `perquire` owns executable experiments, benchmark data contracts, algorithms, and results.
- `franklinbaldo/papers` owns broader theoretical claims and may cite Perquire results after they exist; Perquire must not turn paper hypotheses into empirical claims automatically.
- `franklinbaldo/okf-parser` is used as a validator/query surface for the small experiment-memory bundle. It is not part of the search algorithm and benchmark execution must remain possible without using OKF as an algorithmic dependency.
