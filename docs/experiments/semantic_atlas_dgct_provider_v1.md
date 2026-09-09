# Semantic Atlas DGCT provider-backed v1 — preregistration

Status: **PREREGISTERED / NOT YET LIVE-RUN**

This is an execution-substrate replication of the Dynamic Gauge Compression Test defined in `franklinbaldo/papers`. Perquire supplies provider routing, caching, retry/pacing and auditable workflow artifacts; the scientific gauge implementation remains pinned to the `papers` commit recorded in `semantic_atlas_dgct_provider_v1.json`.

## Question

Does one externally frozen dynamic gauge make independently calibrated model transition fields descriptively simpler and/or more mutually concordant when both observers see the exact same semantic trajectories?

The primary decomposition is

`F_M(q) = a_M V(q) + r_M(q)`

with one constant fitted scalar `a_M` per observer and field. The gauge is useful only if held-out residuals are materially simpler than raw dynamics and the effect is not reproduced equally by matched controls.

## Frozen execution substrate

All live parameters are in `semantic_atlas_dgct_provider_v1.json` and are committed before the first live outcome. The provider-backed variant uses OpenRouter through Perquire's existing provider classes. The reference and transfer observers use different embedding models; generation happens once, and both observers embed the exact same cumulative text states.

The provider route is recorded from Perquire metadata. This pilot does not claim provider-route invariance.

## Leakage controls

- Corpus selection is frozen by `corpus_source_commit` plus SHA-256 path ordering.
- The Semantic Atlas implementation is frozen by `papers_harness_commit`.
- Field family, seed, scale and strength are frozen before any live trajectory is observed.
- Train/test split is by whole trajectory, never adjacent step.
- The same generated text states are used for both embedding observers.
- The gauge is never fit from transition-field geometry; only the scalar amplitude is fit on training trajectories.
- Any post-outcome change to model, field, split, chunking, amplitude basis or complexity metric requires a new experiment version.

## Field arms

1. `zero` — exact no-gauge baseline (`r = F`).
2. `radial_gradient` — non-vortical structured control.
3. `vortex_smooth` — DQRF candidate.
4. `random_divergence_free` — complexity-matched rotational control.

No self-similar or Navier–Stokes-inspired field is tested here.

## Primary evidence vector

For each observer and held-out field residual:

- test residual energy and energy ratio versus raw dynamics;
- entropy effective rank and ratio versus raw dynamics;
- sample-complexity curve under the frozen affine ridge predictor;
- local sensitivity;
- cross-observer residual row cosine, normalized RMSE and pairwise-distance correlation.

There is no primary weighted scalar score.

## Decision boundary

A small R² or next-step gain is insufficient. Evidence for a shared dynamic gauge requires a coherent reduction in descriptive complexity and/or better cross-observer residual concordance on held-out trajectories. Evidence for vortex specificity additionally requires `vortex_smooth` to outperform the non-vortical and random divergence-free controls.

A negative result terminates this provider-backed v1; parameters are not tuned to rescue it.
