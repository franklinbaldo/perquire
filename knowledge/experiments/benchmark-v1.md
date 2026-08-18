---
type: Experiment Protocol
title: Semantic inversion benchmark v1
description: Preregistered equal-budget comparison of adaptive Perquire, independent best-of-N, and mutation hill-climbing on hidden source texts.
status: planned
experiment_id: semantic-inversion-benchmark-v1
claim_id: adaptive-feedback-v1
budget_unit: target_similarity_evaluation
required_methods:
  - adaptive_perquire
  - independent_best_of_n
  - mutation_hill_climber
---

# Semantic inversion benchmark v1

## Input contract

Each case contains source text, a target embedding produced by the frozen benchmark embedding model, a domain label, and stable case identity. Search methods receive the target embedding and permitted public case metadata, but never the source text.

## Comparison contract

All required methods receive the same maximum number of target-similarity evaluations per case. Provider/model identities, seeds, prompts or strategy versions, candidate counts, and LLM calls are recorded.

## Outputs

Every evaluated candidate is recorded with method, case, step, candidate text, target cosine, and resource counters. Summary reports are derived from raw observations rather than replacing them.

## Primary comparison

Compare best target cosine as a function of evaluation budget. Report paired case-level differences between `adaptive_perquire` and each simpler baseline.

## Independent checks

Where practical, evaluate recovered text using a semantic evaluator distinct from the target embedding model and compute reference-corpus neighborhood overlap.

## Decision rule

If adaptive Perquire does not demonstrate a stable advantage over the simpler baselines, do not add algorithmic complexity to preserve the adaptive design. Prefer the simpler method or reformulate the research question.
