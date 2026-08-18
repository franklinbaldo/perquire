---
type: Research Claim
title: Adaptive similarity feedback improves semantic inversion
description: Equal-budget adaptive search should outperform independent candidate sampling when seeking a semantic preimage in a known embedding space.
status: proposed
claim_id: adaptive-feedback-v1
falsifier: Equal-budget independent best-of-N or mutation hill-climbing matches or exceeds adaptive Perquire within benchmark uncertainty.
source_theory:
  - https://github.com/franklinbaldo/papers/blob/main/semantic_atlas.md
---

# Adaptive similarity feedback

Perquire tests whether scalar similarity feedback contains enough useful local information for an LLM-guided search process to navigate toward a hidden target vector more efficiently or accurately than simpler candidate generation.

This claim is deliberately narrower than "decoding embeddings". It assumes a known embedding space and does not imply recovery of the original source text.
