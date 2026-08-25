# Causal feedback v2 — quantitative decision-rule addendum

Status: **PROSPECTIVE / FROZEN BEFORE TARGET SCORES**

Version: `causal-feedback-v2-decision-rule-1`

This addendum resolves the degrees of freedom identified by `causal_feedback_v2_decision_rule_gate.md`. It must not be changed after any Gate B target score exists without creating a new experiment version.

## Inferential unit and estimands

The target case is the inferential unit. Provider replicates are averaged within each target and arm before contrasts are computed, using the existing `target_level_effects()` procedure.

At B=16 and, only if escalated, B=32, the four decision cells are:

- `true_feedback - decoy_feedback` on best true-target cosine;
- `true_feedback - null_feedback` on best true-target cosine;
- `true_feedback - decoy_feedback` on AUC of best-so-far true-target cosine over steps;
- `true_feedback - null_feedback` on that AUC.

Best-so-far alone cannot open the next gate: both best and trajectory-sensitive AUC must satisfy the rule against both controls.

## Quantitative rule

A checkpoint is **support** only if all four cells have at least 20 valid paired targets, strictly positive target-level median effect, and at least 2/3 of target effects strictly positive.

A checkpoint is **against** if any one of the four cells has at least 20 valid paired targets, median effect <= 0, and positive fraction <= 1/2.

All other configurations are **ambiguous**. Exact zero is not positive. Missing cells or fewer than 20 paired targets cannot produce support.

The 20/24 target floor is an inferential completeness guard in addition to the separately frozen trajectory-validity threshold. It is not a substitute for that operational gate.

## B=16 -> B=32 escalation

Run B=16 first. Only `support` at B=16 authorizes B=32. `against` or `ambiguous` stops Gate B without increasing budget.

A B=16 support is provisional (`pending_b32`). The final Gate B conclusion is the mechanically evaluated B=32 decision. Thus a B=32 ambiguous or against result is not pooled with B=16 to rescue support.

## Uncertainty and descriptive summaries

Report the existing target-level means, medians, positive fractions and raw target effects for all four cells. Bootstrap intervals, sign-test p-values, plots, or other uncertainty summaries may be reported descriptively after the frozen decision is computed, but they do not participate in this version's decision rule.

This intentionally avoids choosing an effect-size threshold in cosine units without an independently justified semantic scale. The rule instead requires directional consistency across targets, both controls, and both endpoint/trajectory evidence.

## Deterministic implementation

`benchmarks/causal_decision_rule_v2.py` is the executable authority. `decide(target_level_effects)` consumes only the already target-aggregated effects and returns B=16 decision, escalation status, and B=32 final decision when available.

Ties, missingness and invalidity are handled exactly as above; no manual adjudication is permitted. Any change to constants, estimands, comparisons, metric set, escalation, or outcome mapping after target scoring creates a new experiment version.
