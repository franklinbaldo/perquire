# OpenRouter v2 generation-substrate freeze rule

Status: **PROSPECTIVE / TARGET-FREE**

Rule version: `openrouter-reliability-freeze-v2-ox-alpha`

Required generation model: **`stealth/ox-alpha`**

Prospective Ox Alpha evidence begins: **2026-08-24T14:00:00Z**

This contract governs only when the fixed Ox Alpha OpenRouter path may be converted into the frozen generation substrate for the causal-feedback v2 experiment. It does not use Perquire target embeddings, similarities, adaptive scores, or benchmark outcomes.

## 1. Why the evidence boundary restarted

The original v2 qualification began at `2026-08-22T12:30:00Z` and dynamically selected among free OpenRouter models. On 2026-08-24, #74/#75 established `stealth/ox-alpha` as the prospective OpenRouter generation default for Perquire.

Those older windows remain valid target-free engineering evidence about the models actually selected, but they cannot establish longitudinal reliability for Ox Alpha when they observed a different inference function. The new boundary was fixed before eligible Ox Alpha windows existed under this rule.

This is not a rescue after seeing Perquire target outcomes: Gate B target scoring remains prohibited and no target result participates in this change.

## 2. Prospective evidence boundary

Only observatory windows whose `observed_at_utc` is **at or after `2026-08-24T14:00:00Z`** and whose exact `selected_model` is **`stealth/ox-alpha`** count toward substrate eligibility.

Windows before that timestamp and windows selecting any other model remain descriptive evidence but cannot satisfy this stopping rule.

## 3. Fixed-candidate evidence

The observatory no longer chooses the experimental generation model from a dynamic candidate pool. It observes the exact prospective substrate fixed by #74/#75.

Each window must first verify target-free public facts about `stealth/ox-alpha`: zero prompt/completion price, text output, and the existing OpenRouter endpoint-health thresholds. It then requires 2/2 target-free account probes before making the ten longitudinal observation calls.

For Ox Alpha aggregate:

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

Ox Alpha cannot be frozen until all of these are true:

1. at least **48 selected windows**;
2. those selected windows span at least **24 hours** from first to last observation;
3. at least **480 observation calls** (48 × 10 under the current probe);
4. at least **95% of selected windows are completely clean**;
5. aggregate observation-call success rate is at least **99.5%**;
6. aggregate `transport_attempts / logical_calls <= 1.01` for observation calls;
7. no selected window contains more than **1 failed observation call**;
8. no two consecutive selected windows contain an observation failure.

These thresholds are operational gates, not estimates that requests are IID. Temporal spread and window-level conditions are included because provider failures can be bursty and correlated.

If Ox Alpha does not satisfy the rule, **there is no eligible substrate** and Gate B must not run. The failure does not authorize selecting another model after inspecting target results.

## 5. Model selection is no longer an outcome

The experimental generation model is fixed prospectively as `stealth/ox-alpha`; reliability qualification may accept or reject it, but may not replace it with another model. This removes model-selection freedom from the qualification phase and aligns the scientific substrate with the prospective OpenRouter default used by Perquire.

## 6. Freeze record

Reliability eligibility alone does not start the experiment. A separate versioned freeze record must state, before Gate B target scoring:

- exact generation model slug (`stealth/ox-alpha`);
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

A model slug alone is insufficient identity. Before Gate B, the exact OpenRouter upstream provider path used by Ox Alpha must be identified and constrained, with fallback outside that constraint disabled. The same principle applies to the optimization embedding model if it has multiple upstream providers.

If the required provider path cannot be identified and constrained with enough stability to satisfy the experiment's identity requirement, Ox Alpha is ineligible even if its aggregate success rate is high.

## 8. Relationship to the observatory

The scheduled observatory remains entirely target-free. It now tests one prospectively chosen generation substrate rather than dynamically picking the experimental model each window.

Once a substrate is frozen for Gate B, the causal experiment itself must **not** run discovery and must **not** switch model/provider on failure.

## 9. Failure and no-result outcomes

Valid outcomes include:

- `eligible`: Ox Alpha satisfies the prospective reliability rule and may proceed to routing/configuration freeze review;
- `insufficient_coverage`: not enough qualifying Ox Alpha windows/time yet;
- `reliability_failure`: enough coverage exists but reliability thresholds fail;
- `identity_failure`: Ox Alpha is reliable but its upstream route cannot be constrained/identified sufficiently;
- `no_eligible_substrate`: Ox Alpha cannot be frozen under the contract.

Only `eligible` followed by an explicit freeze record opens Gate B.
