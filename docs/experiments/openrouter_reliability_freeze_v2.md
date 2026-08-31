# OpenRouter v2 generation-substrate freeze rule

Status: **PROSPECTIVE / TARGET-FREE**

Rule version: `openrouter-reliability-freeze-v2-inkling`

Required generation model: **`thinkingmachines/inkling:free`**

Prospective Inkling evidence begins: **2026-08-31T02:00:00Z**

This contract governs only when the fixed Inkling OpenRouter path may be converted into the frozen generation substrate for the causal-feedback v2 experiment. It does not use Perquire target embeddings, similarities, adaptive scores, or benchmark outcomes.

## 1. Why the evidence boundary restarted

The v2 qualification boundary has now been reset three times. The first two resets are recorded here because the third only makes sense against them.

1. Qualification began at `2026-08-22T12:30:00Z`, dynamically selecting among free OpenRouter models.
2. On 2026-08-24, #74/#75 fixed `stealth/ox-alpha` as the prospective substrate and moved the boundary to `2026-08-24T14:00:00Z`.
3. On 2026-08-30, #83 replaced the withdrawn Ox Alpha route with `minimax/minimax-m3:free` and moved the boundary to `2026-08-30T18:00:00Z`.

### Why Ox Alpha was replaced

OpenRouter withdrew it. As of 2026-08-30 `stealth/ox-alpha` is absent from `GET /api/v1/models`, and its canonical endpoints record returns the model card with `"endpoints": []`. That also explains the empty completions in #79: the route was being retired, not exhausting an output or reasoning budget.

### Why MiniMax M3 was replaced

It never produced a single valid window. The two scheduled observatory windows after #83 merged (2026-08-30T22:02Z and 2026-08-31T00:29Z) both recorded zero qualification calls, zero observation calls and `selected_model: null`: the candidate list was empty because the route failed the endpoint-health gate.

The cause is margin, not a defect in the gate. `minimax/minimax-m3:free` serves from around ten endpoints, and its route mean uptime sits on the threshold rather than above it — observed at 99.5076% and then 99.5560% against a 99.5% requirement, a margin between 0.008 and 0.056 percentage points, with a single weak provider (GMICloud, 95.89% over 30 minutes) accounting for the whole shortfall. Excluding that endpoint the mean would be 99.87%. A substrate this close to the line crosses it between windows, so it alternates between eligible and ineligible and cannot accumulate consecutive clean windows.

#83 selected it by applying the frozen ranking to a single instantaneous health snapshot. That was the error: the ranking is correct, but passing the gate at one instant is not evidence of staying above it.

### Why Inkling

`thinkingmachines/inkling:free` was the runner-up under the same frozen ordering, and it is the model this rule now fixes. It serves from two endpoints, both at 100.00% uptime — a margin of 0.5 percentage points rather than 0.008. It costs quality: `intelligence_index` 42.3 against 45.4, with `agentic_index` 34.1 and `coding_index` 52.1.

That trade is deliberate. A substrate that oscillates across the eligibility threshold produces no evidence at all, so its nominally higher quality is unrealisable. The choice is still not a hand-pick: Inkling is what the existing ranking returns once the top entry is disqualified by its own observed windows.

### Why none of this is a rescue

Gate B has never run. No Perquire target score exists under any version of this rule. Every replacement so far was forced by a public provider fact or by target-free observatory evidence, and each boundary was fixed before any eligible window existed under the new rule version.

Windows observed under any earlier boundary remain valid target-free engineering evidence about the models they actually observed. They cannot establish longitudinal reliability for Inkling, because they observed a different inference function.

## 2. Prospective evidence boundary

Only observatory windows whose `observed_at_utc` is **at or after `2026-08-31T02:00:00Z`** and whose exact `selected_model` is **`thinkingmachines/inkling:free`** count toward substrate eligibility.

Windows before that timestamp and windows selecting any other model remain descriptive evidence but cannot satisfy this stopping rule.

## 3. Fixed-candidate evidence

The observatory no longer chooses the experimental generation model from a dynamic candidate pool. It observes the exact prospective substrate fixed by this rule version.

Each window must first verify target-free public facts about `thinkingmachines/inkling:free`: zero prompt/completion price, text output, and the existing OpenRouter endpoint-health thresholds. It then requires 2/2 target-free account probes before making the ten longitudinal observation calls.

For Inkling aggregate:

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

Inkling cannot be frozen until all of these are true:

1. at least **48 selected windows**;
2. those selected windows span at least **24 hours** from first to last observation;
3. at least **480 observation calls** (48 × 10 under the current probe);
4. at least **95% of selected windows are completely clean**;
5. aggregate observation-call success rate is at least **99.5%**;
6. aggregate `transport_attempts / logical_calls <= 1.01` for observation calls;
7. no selected window contains more than **1 failed observation call**;
8. no two consecutive selected windows contain an observation failure.

These thresholds are operational gates, not estimates that requests are IID. Temporal spread and window-level conditions are included because provider failures can be bursty and correlated.

If Inkling does not satisfy the rule, **there is no eligible substrate** and Gate B must not run. The failure does not authorize selecting another model after inspecting target results.

## 5. Model selection is no longer an outcome

The experimental generation model is fixed prospectively as `thinkingmachines/inkling:free`; reliability qualification may accept or reject it, but may not replace it with another model. Two things override this, both target-free and both forcing a new rule version with a new evidence boundary rather than a substitution inside this one: provider withdrawal of the model, and observed failure of the model to hold the availability/health gate across windows. Neither may be invoked from target results. This removes model-selection freedom from the qualification phase and aligns the scientific substrate with the prospective OpenRouter default used by Perquire.

## 6. Freeze record

Reliability eligibility alone does not start the experiment. A separate versioned freeze record must state, before Gate B target scoring:

- exact generation model slug (`thinkingmachines/inkling:free`);
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

A model slug alone is insufficient identity. Before Gate B, the exact OpenRouter upstream provider path used by Inkling must be identified and constrained, with fallback outside that constraint disabled. The same principle applies to the optimization embedding model if it has multiple upstream providers.

If the required provider path cannot be identified and constrained with enough stability to satisfy the experiment's identity requirement, Inkling is ineligible even if its aggregate success rate is high.

## 8. Relationship to the observatory

The scheduled observatory remains entirely target-free. It now tests one prospectively chosen generation substrate rather than dynamically picking the experimental model each window.

Once a substrate is frozen for Gate B, the causal experiment itself must **not** run discovery and must **not** switch model/provider on failure.

## 9. Failure and no-result outcomes

Valid outcomes include:

- `eligible`: Inkling satisfies the prospective reliability rule and may proceed to routing/configuration freeze review;
- `insufficient_coverage`: not enough qualifying Inkling windows/time yet;
- `reliability_failure`: enough coverage exists but reliability thresholds fail;
- `identity_failure`: Inkling is reliable but its upstream route cannot be constrained/identified sufficiently;
- `no_eligible_substrate`: Inkling cannot be frozen under the contract.

Only `eligible` followed by an explicit freeze record opens Gate B.
