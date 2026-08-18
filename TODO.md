# Perquire roadmap

Perquire is an experimental system for approximate semantic inversion in a known embedding space.

The project has one priority: determine whether adaptive similarity feedback helps recover a useful textual semantic preimage of a hidden target embedding better than simpler equal-budget baselines.

## Current gate

1. Keep the supported runtime and public interfaces green.
2. Define the claim and its falsifiers precisely.
3. Build a reproducible benchmark with hidden source text.
4. Compare adaptive Perquire against equal-budget baselines.
5. Change the search algorithm only after the benchmark establishes where the current method wins or loses.

## Deliberately deferred

Until the benchmark produces evidence, do not spend project effort on:

- new web UI features;
- provider proliferation or migrations that are not needed by the benchmark;
- observability dashboards;
- database/VSS refactors;
- line-count reduction projects;
- new strategy registries or plugin abstractions;
- claims of generic embedding decoding, interpretability, or recovery of original text.

Existing working infrastructure may remain. Deferred means "not the current bottleneck", not "must be deleted".

## Minimal PR stack

### 1. Truthful, executable base

Merged in #19: runtime/CLI repair, supported Python CI, and truthful public documentation.

### 2. Research contract + minimal package + OKF memory

Define semantic inversion, assumptions, metrics, baselines, and falsification criteria. Keep claims and experiment protocols as a small OKF bundle validated by `okf-parser`. Make provider, persistence, web, VSS, and analysis dependencies opt-in rather than part of the import-time core.

Exit gate: another agent can reconstruct what is being tested without reading implementation history, and `pip install perquire` has a genuinely small dependency/import surface.

### 3. Benchmark + equal-budget baselines

Create the hidden-text benchmark and run current adaptive Perquire against independent best-of-N sampling and a simple mutation hill-climber under the same evaluation budget.

Exit gate: machine-readable results show whether adaptive feedback adds value and where.

### 4. Contrastive probing, conditionally

Only if the benchmark identifies signal worth improving, replace the misleading yes/no-question interpretation with explicit semantic probes and evaluate contrastive probe sets against the frozen benchmark.

Exit gate: contrastive probing beats or characterizes the current method on predeclared metrics. If it does not, keep the simpler method.

## Success is allowed to be negative

A benchmark showing that best-of-N or a trivial hill-climber matches Perquire is a successful experiment. It falsifies the need for the current adaptive machinery and tells us how to simplify the project.
