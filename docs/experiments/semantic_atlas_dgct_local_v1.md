# Semantic Atlas DGCT v1 — Perquire local-model execution carrier

Status: **EXECUTION-ONLY / SCIENTIFIC PARAMETERS INHERITED FROM PAPERS**

This workflow does not define a new DGCT experiment. It executes the already frozen `franklinbaldo/papers` DGCT v1 at commit `dabfd5102199212896a68b8b6f6b12297e1a5963` using Perquire's GitHub Actions execution substrate.

The provider-backed Perquire attempt (`semantic-atlas-dgct-provider-v1`) was infrastructure-inconclusive: the OpenRouter key returned HTTP 403 `Key limit exceeded (total limit)` before the first embedding batch completed and before any DGCT transition field or residual result was produced. No scientific parameter is changed here to rescue that run.

## Scientific source of truth

The following remain owned and frozen by `papers`:

- corpus and source commit;
- observer identities and pinned revisions;
- generator identity and pinned revision;
- SRF dimension;
- generation seeds/chunking;
- whole-trajectory train/test split;
- field families, seeds, scales and strengths;
- amplitude model;
- residual-complexity metrics and claim boundary.

The Perquire workflow merely checks out the exact frozen `papers` commit and invokes its existing command:

```text
python scripts/run_dynamic_gauge_compression.py
```

No environment secret is required. Public Hugging Face models are downloaded by their revisions already frozen in the `papers` manifests. Hugging Face and pip/uv caches are execution optimizations only and do not alter model identity.

## Execution boundary

A successful workflow means that the frozen model-backed DGCT was executed reproducibly. Scientific interpretation begins only after the generated `artifacts/dynamic_gauge_compression_v1.json` is inspected against the preregistered controls.
