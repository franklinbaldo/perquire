# OpenRouter v2 generation-substrate freeze rule

Status: **PROSPECTIVE / TARGET-FREE**

Rule version: `openrouter-reliability-freeze-v2-minimax-m3`

Required generation model: **`minimax/minimax-m3:free`**

Prospective MiniMax M3 evidence begins: **2026-08-30T18:00:00Z**

This contract governs only when the fixed MiniMax M3 OpenRouter path may be converted into the frozen generation substrate for the causal-feedback v2 experiment. It does not use Perquire target embeddings, similarities, adaptive scores, or benchmark outcomes.

## 1. Why the evidence boundary restarted

The v2 qualification boundary has now been reset twice.

The original qualification began at `2026-08-22T12:30:00Z` and dynamically selected among free OpenRouter models. On 2026-08-24, #74/#75 fixed `stealth/ox-alpha` as the prospective substrate and moved the boundary to `2026-08-24T14:00:00Z`.

That substrate was withdrawn by the provider. As of 2026-08-30, `stealth/ox-alpha` is absent from `GET /api/v1/models` (396 models listed, nothing in the `stealth/` namespace), and its canonical endpoints record returns the model card with `"endpoints": []` — zero serving providers. This also explains the empty completions recorded in #79: the route was being retired, not exhausting an output or reasoning budget.

A withdrawn model cannot be qualified, so the substrate is replaced rather than re-observed. The replacement is not a rescue after inspecting outcomes: Gate B has never run, no Perquire target score exists under any version of this rule, and the withdrawal is a public provider fact independent of any Perquire result.

`minimax/minimax-m3:free` was selected by applying the eligibility predicate and ranking already implemented in `benchmarks/probe_openrouter_reliability_v2.py` to the live catalogue, not by hand. Of 21 free text models, 10 passed the endpoint-health thresholds (route mean uptime >= 99.5%, every operational endpoint >= 95%), and the existing quality ordering — `intelligence_index`, then `agentic_index`, then `coding_index` — ranked `minimax/minimax-m3:free` first among them at 45.4/36.1/58.6 across 11 operational endpoints. `z-ai/glm-5.2:free` scores higher on intelligence (52.6) but fails the health gate, with a worst endpoint at 66.7% uptime.

Windows observed under either earlier boundary remain valid target-free engineering evidence about the models they actually observed. They cannot establish longitudinal reliability for MiniMax M3, because they observed a different inference function. This boundary was fixed before any eligible MiniMax M3 window existed under this rule.

## 2. Prospective evidence boundary

Only observatory windows whose `observed_at_utc` is **at or after `2026-08-30T18:00:00Z`** and whose exact `selected_model` is **`minimax/minimax-m3:free`** count toward substrate eligibility.

Windows before that timestamp and windows selecting any other model remain descriptive evidence but cannot satisfy this stopping rule.

## 3. Fixed-candidate evidence

The observatory no longer chooses the experimental generation model from a dynamic candidate pool. It observes the exact prospective substrate fixed by this rule version.

Each window must first verify target-free public facts about `minimax/minimax-m3:free`: zero prompt/completion price, text output, and the existing OpenRouter endpoint-health thresholds. It then requires 2/2 target-free account probes before making the ten longitudinal observation calls.

For MiniMax M3 aggregate:

- selected windows;
- first/last observation timestamp and temporal span;
- observation logical calls;
- observation successes/failures;
- observation success rate;
- completely clean observation windows;
- clean-window fraction;
- transport attempts and transport/logical ratio;
- maximum failures in any one window;
- qualification successes/attempts;
- public route-health metadata observed in those windows.

## 4. Minimum longitudinal coverage

MiniMax M3 cannot be frozen until all of these are true:

1. at least **48 selected windows**;
2. those selected windows span at least **24 hours** from first to last observation;
3. at least **480 observation calls** (48 × 10 under the current probe);
4. at least **95% of selected windows are completely clean**;
5. aggregate observation-call success rate is at least **99.5%**;
6. aggregate `transport_attempts / logical_calls <= 1.01` for observation calls;
7. no selected window contains more than **1 failed observation call**;
8. no two consecutive selected windows contain an observation failure.

These thresholds are operational gates, not estimates that requests are IID. Temporal spread and window-level conditions are included because provider failures can be bursty and correlated.

If MiniMax M3 does not satisfy the rule, **there is no eligible substrate** and Gate B must not run. The failure does not authorize selecting another model after inspecting target results.

## 5. Model selection is no longer an outcome

The experimental generation model is fixed prospectively as `minimax/minimax-m3:free`; reliability qualification may accept or reject it, but may not replace it with another model. Provider withdrawal of the model itself is the sole exception, and it forces a new rule version and a new evidence boundary rather than a substitution inside this one. This removes model-selection freedom from the qualification phase and aligns the scientific substrate with the prospective OpenRouter default used by Perquire.

## 6. Freeze record

Reliability eligibility alone does not start the experiment. A separate versioned freeze record must state, before Gate B target scoring:

- exact generation model slug (`minimax/minimax-m3:free`);
- exact provider routing order/allow-list;
- `allow_fallbacks=false`;
- temperature;
- max tokens;
- retry policy;
- requests/minute and concurrency;
- Python/LiteLLM/uv lock identity;
- cache mode (`fresh` for experimental generation);
- observed remote identity limitations;
- exact optimization embedding model and its routing/drift policy;
- Gate-B minimum-validity threshold.

Any change to these values after target scores are observed creates a new experiment version.

## 7. Provider-routing requirement

A model slug alone is insufficient identity. Before Gate B, the exact OpenRouter upstream provider path used by MiniMax M3 must be identified and constrained, with fallback outside that constraint disabled. The same principle applies to the optimization embedding model if it has multiple upstream providers.

If the required provider path cannot be identified and constrained with enough stability to satisfy the experiment's identity requirement, MiniMax M3 is ineligible even if its aggregate success rate is high.

## 8. Relationship to the observatory

The scheduled observatory remains entirely target-free. It now tests one prospectively chosen generation substrate rather than dynamically picking the experimental model each window.

Once a substrate is frozen for Gate B, the causal experiment itself must **not** run discovery and must **not** switch model/provider on failure.

## 9. Failure and no-result outcomes

Valid outcomes include:

- `eligible`: MiniMax M3 satisfies the prospective reliability rule and may proceed to routing/configuration freeze review;
- `insufficient_coverage`: not enough qualifying MiniMax M3 windows/time yet;
- `reliability_failure`: enough coverage exists but reliability thresholds fail;
- `identity_failure`: MiniMax M3 is reliable but its upstream route cannot be constrained/identified sufficiently;
- `no_eligible_substrate`: MiniMax M3 cannot be frozen under the contract.

Only `eligible` followed by an explicit freeze record opens Gate B.
