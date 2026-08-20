# OpenRouter reliability probe v1 — decision

The target-free reliability probe completed in GitHub Actions run `32341899151` before any 2→32 scaling curve was observed.

The frozen selection rule was: an eligible candidate must return non-empty text for all 12 logical calls under the existing bounded retry policy; if both eligible candidates pass, choose the lower frozen OpenRouter list-price candidate. The previously failing free model is a control and cannot win.

## Observed result

| model | role | logical successes | logical failures | transport attempts |
|---|---|---:|---:|---:|
| `openai/gpt-oss-20b:free` | control | 0/12 | 12 | 36 |
| `google/gemini-2.5-flash-lite` | eligible | 12/12 | 0 | 12 |
| `google/gemini-3.1-flash-lite` | eligible | 12/12 | 0 | 12 |

Both eligible candidates satisfied the reliability gate. Applying the preregistered tie-break mechanically selects **`google/gemini-2.5-flash-lite`** for scaling v1 because its frozen list price is lower.

The control's complete failure is operational evidence about the old generation substrate, not evidence for or against adaptive scaling. No target embeddings, target similarities, method scores, or scaling results were consulted by the probe.

## Frozen v1 generation substrate

- provider: OpenRouter
- model: `google/gemini-2.5-flash-lite`
- generation temperature: `0.7`
- bounded retries: `max_retries=2`
- benchmark generation model must remain uniform across methods, budgets and replicates
- a logical generation that exhausts retries invalidates that experimental cell
- invalid cells remain reported; no silent model switching or rerun-until-success may erase them
- fewer than 95% valid preregistered cells blocks a clean v1 scaling claim

Artifact: GitHub Actions run `32341899151`, artifact `openrouter-reliability-probe-v1` (artifact id `9397100892`).

The probe run overlapped in wall-clock time with an earlier probe attempt. That overlap clearly affected the already-ineligible free control through upstream rate limiting. Both paid eligible candidates nevertheless completed 12/12 with exactly one transport attempt per logical call. Because the decision between those candidates depends only on their perfect-pass status and the prospectively frozen price tie-break, the observed concurrency does not distinguish between them or alter the selected model. Any future reliability probe is a new version and does not retroactively rewrite this v1 decision.
