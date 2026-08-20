# OpenRouter generation reliability probe v1

This probe resolves #45 **before any 2→32 scaling curve is observed**. It does not embed or score candidate text and therefore cannot select a model by whether it helps `adaptive_perquire`.

## Frozen candidate set

The current default `openai/gpt-oss-20b:free` is retained as a control because it produced repeated `EmptyProviderResponse` failures in #39 and #44. That prior evidence makes it ineligible to become the v1 frozen model even if this small probe happens to complete.

Two low-cost GA OpenRouter candidates are tested prospectively:

- `google/gemini-2.5-flash-lite` — OpenRouter list price observed 2026-08-20: $0.10/M input, $0.40/M output.
- `google/gemini-3.1-flash-lite` — OpenRouter list price observed 2026-08-20: $0.25/M input, $1.50/M output.

The candidate set and tie-break are frozen before the live probe runs. Adding another model after seeing these results is a new probe version.

## Probe workload

Each model receives 12 fresh logical generation calls: four target-free prompt shapes repeated three times. The shapes mirror the benchmark's generation boundary (independent description, semantic mutation, feedback-shaped refinement, concise synthesis) without using benchmark source texts, embeddings, target similarities, or method scores.

Configuration is uniform: temperature 0.7, max output 64 tokens, existing bounded retry policy (`max_retries=2`), cache off. The artifact records logical successes/failures, transport attempts and elapsed time per model.

## Selection rule

A candidate is eligible only if all 12 logical calls return non-empty text after the bounded retry policy. If both eligible candidates pass, freeze the lower OpenRouter list-price candidate (`google/gemini-2.5-flash-lite`). If exactly one passes, freeze that one. If neither passes, #45 remains unresolved; do not choose by scaling quality or add models retroactively to rescue v1.

The free control cannot win this selection because its repeated failures are already observed evidence from the pre-probe live gates.

## Failure policy for scaling v1

The frozen model must be used uniformly across every method/budget/replicate. A required logical generation that exhausts the bounded retries makes that experimental cell **invalid**; the system may preserve successful prior calls by exact replay identity, but must not silently switch model, count a replay as a new stochastic sample, or rerun the failed cell until success and then erase the failure.

The primary scaling artifact must report invalid-cell rate. If fewer than 95% of preregistered cells are valid, the v1 substrate fails its operational reliability gate and no clean scaling claim should be made from the surviving subset.

This policy is about the generation substrate, not evidence for or against adaptive scaling.