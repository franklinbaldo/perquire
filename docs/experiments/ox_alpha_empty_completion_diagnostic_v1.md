# Ox Alpha empty-completion diagnostic v1

This is a **target-free apparatus diagnostic** for issue #79. It does not read or score any Perquire target and none of its calls count toward the reliability freeze in #65.

## Frozen question

The current qualification contract (`temperature=0.7`, `max_tokens=64`, no explicit reasoning setting) repeatedly reaches the exact `stealth/ox-alpha` route but returns no visible completion. The diagnostic asks whether this is consistent with output/reasoning budget exhaustion or whether the route remains unusable under a small preregistered matrix.

## Frozen matrix

The machine-readable authority is `benchmarks/ox_alpha_diagnostic_v1.json`. Four cells are tested, two calls each, using one existing target-free observatory prompt:

1. current contract: 64 output tokens, provider-default reasoning;
2. 256 output tokens, provider-default reasoning;
3. 256 output tokens, `reasoning.effort=low`;
4. 1024 output tokens, `reasoning.effort=low`.

The matrix is intentionally small. No cell may be added after observing a result in this diagnostic version.

OpenRouter documents reasoning tokens as output tokens and exposes normalized reasoning controls through the `reasoning` object. The purpose of the two low-effort cells is therefore diagnostic, not optimization: they distinguish a budget allocation explanation from a route/provider failure without introducing a broad tuning search.

## Recorded evidence

For every call the artifact records transport success separately from visible-text success, response model when exposed, finish reason, visible-content length, reasoning length when exposed, usage including reasoning-token metadata when available, elapsed time, and exact error classification. It never records credentials.

## Frozen decision boundary

A cell is operational only if both calls have transport success and non-empty visible text.

- If at least one cell is operational, select the operational cell with the smallest `max_tokens`, breaking ties by preregistered matrix order. That result does **not** amend #65 by itself: a separate prospective change must freeze the chosen qualification configuration and move `EVIDENCE_AFTER` to after that change lands. All historical windows remain diagnostic only.
- If no cell is operational, record `no_eligible_ox_alpha_substrate`. Do not rescue-switch models or add more cells in this diagnostic version.

No Gate B target score is authorized by this diagnostic.

## Collection

`.github/workflows/ox-alpha-diagnostic-v1.yml` collects the evidence on a recurring 30-minute schedule. Each run is an independent window over the same frozen matrix, uploaded under its own timestamped artifact.

Recurring collection means the diagnostic accumulates many windows rather than one. The frozen matrix and the frozen decision boundary above are unchanged by this: no cell may be added, and no window may be selected after inspection to stand in for the diagnostic result. Analysis across accumulated windows is therefore a repeated-observation reading, not a single preregistered trial, and any conclusion drawn from it must say which windows it used and why before reporting a cell as operational.
