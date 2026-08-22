# OpenRouter v2 generation-substrate freeze rule

Status: **PROSPECTIVE / TARGET-FREE**

This contract governs only when the dynamic OpenRouter reliability observatory may be converted into one frozen generation substrate for the causal-feedback v2 experiment. It does not use Perquire target embeddings, similarities, adaptive scores, or benchmark outcomes.

## 1. Why a stopping rule is required

A dynamic selector that keeps sampling until an attractive model appears is not a scientific substrate. The observatory is allowed to discover and qualify candidates, but the causal experiment needs one prospectively frozen model/routing/configuration.

The earlier 12/12 v1 qualification was falsified by the scaling run. Likewise, one 10/10 or 20/20 window proves that the path can work, not that it is reliable enough for a long experiment.

## 2. Prospective evidence boundary

Only observatory windows whose `observed_at_utc` is **at or after the merge commit/time of this contract** count toward substrate eligibility.

Earlier target-free windows remain descriptive evidence and engineering history but cannot satisfy this stopping rule. This prevents choosing the rule after inspecting which candidate happened to look best.

## 3. Candidate-specific evidence

Evidence is never pooled across different selected models as if they were one inference function.

For each exact selected model slug, aggregate only windows in which that slug was selected after qualification. Report separately:

- selected windows;
- first/last observation timestamp and temporal span;
- observation logical calls;
- observation successes/failures;
- observation success rate;
- completely clean observation windows;
- clean-window fraction;
- transport attempts and transport/logical ratio;
- maximum failures in any one window;
- qualification successes/attempts for windows selecting that model;
- public route-health metadata observed in those windows.

## 4. Minimum longitudinal coverage

A candidate cannot be frozen until all of these are true:

1. at least **48 selected windows** for that same model;
2. those selected windows span at least **24 hours** from first to last observation;
3. at least **480 observation calls** for that model (48 × 10 under the current probe);
4. at least **95% of selected windows are completely clean** (all 10 observation calls succeed);
5. aggregate observation-call success rate is at least **99.5%**;
6. aggregate `transport_attempts / logical_calls <= 1.01` for observation calls;
7. no selected window contains more than **1 failed observation call**;
8. no two consecutive selected windows for that candidate contain an observation failure.

These thresholds are operational gates, not estimates that requests are IID. Temporal spread and window-level conditions are included specifically because provider failures can be bursty and correlated.

If no candidate satisfies the rule, **there is no eligible substrate** and Gate B must not run.

## 5. Deterministic selection among eligible candidates

If more than one exact model slug is eligible, select mechanically using target-free data only:

1. highest OpenRouter Artificial Analysis intelligence index recorded at the candidate's most recent eligible window;
2. then agentic index;
3. then coding index;
4. then higher observed call success rate;
5. then higher clean-window fraction;
6. then lower transport/logical ratio;
7. then lexical model slug as final deterministic tie-break.

No Perquire target score may participate.

## 6. Freeze record

Eligibility alone does not start the experiment. A separate versioned freeze record must state, before Gate B target scoring:

- exact generation model slug;
- exact provider routing order/allow-list;
- `allow_fallbacks` policy;
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

OpenRouter defaults to provider-level routing/failover when no provider override is supplied. For the scientific freeze, a model slug alone is insufficient identity.

The frozen generation request must therefore use an explicit OpenRouter provider constraint and disable fallback outside that constraint. The same principle applies to the optimization embedding model if it has multiple upstream providers.

If the required provider path cannot be identified and constrained with enough stability to satisfy the experiment's identity requirement, the candidate is ineligible even if its aggregate success rate is high.

## 8. Relationship to the dynamic observatory

The scheduled observatory may continue to discover current free models and dynamically select a healthy candidate per window. That dynamic behavior is only for target-free qualification.

Once a substrate is frozen for Gate B, the causal experiment itself must **not** run discovery and must **not** switch to another model/provider on failure.

## 9. Failure and no-result outcomes

Valid outcomes of this qualification phase include:

- `eligible`: one or more candidates satisfy the prospective rule and one is mechanically selected;
- `insufficient_coverage`: no candidate has yet accumulated enough prospective windows/time;
- `reliability_failure`: candidates have enough coverage but fail the thresholds;
- `identity_failure`: a candidate is reliable but cannot be constrained/identified sufficiently for the scientific run;
- `no_eligible_substrate`: no candidate can be frozen under the contract.

Only `eligible` followed by an explicit freeze record opens Gate B.