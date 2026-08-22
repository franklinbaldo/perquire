# Causal feedback v2 — decision-rule gate

Status: **PROSPECTIVE / NO TARGET SCORES MAY PRECEDE THIS GATE**

This document closes an interpretation gap in the causal-feedback v2 preregistration without choosing a substantive effect threshold after the fact.

## Why this gate exists

`causal_feedback_v2_preregistration.md` correctly freezes the causal arms, inferential unit, trajectory metrics, operational validity boundary, and kill criterion. Its current decision language still contains two degrees of freedom that are not mechanically decidable before outcomes are seen:

- `true_feedback` must be "meaningfully better" than decoy/null feedback;
- Gate C may proceed if Gate B "supports" target-relevant feedback.

Those phrases are scientifically sensible summaries, but they do not yet specify a prospective decision function. With only 24 target-level effects, deciding what counts as meaningful after seeing their distribution would reintroduce researcher discretion at the exact point v2 is intended to protect.

## Gate

**No Gate B target score may be generated until a versioned decision-rule addendum is committed and reviewed.**

The addendum must be frozen before the first target-scored provider call and must state, at minimum:

1. the exact target-level estimands used to compare `true_feedback` with both `decoy_feedback` and `null_feedback`;
2. how replicate-level observations, if any, are reduced to one inferential contribution per target;
3. the quantitative rule that distinguishes support, ambiguity, and evidence against the mechanism;
4. the exact role of best-so-far versus trajectory-sensitive evidence such as AUC/improvement frequency, so a lucky maximum cannot independently open Gate C;
5. the uncertainty summary to report and whether it participates in the decision or is descriptive only;
6. the Gate-C escalation rule at B=16;
7. the treatment of ties, missing/invalid trajectories, and any minimum number of valid targets beyond the separately frozen operational-validity threshold;
8. the analysis code/version or deterministic procedure that implements the rule.

The rule may be frequentist, Bayesian, equivalence/ROPE-based, sign/rank-based, or another defensible target-level procedure. This gate deliberately does **not** choose among those scientific options.

## Falsification discipline

After any Gate B target score exists:

- changing the decision threshold, estimand, uncertainty criterion, or Gate-C escalation rule creates a new experiment version;
- exploratory summaries may be added, but they cannot redefine the preregistered Gate B conclusion;
- a negative or ambiguous Gate B result cannot be converted into support by selecting a more favorable metric post hoc.

## Relationship to the substrate freeze

The OpenRouter reliability freeze answers whether a sufficiently stable, identifiable generation substrate exists. This decision-rule gate answers how the target-level causal evidence will be interpreted once that substrate exists.

Both gates must be satisfied before Gate B:

```text
target-free substrate eligibility
  -> explicit model/provider/config freeze
  -> quantitative causal decision-rule freeze
  -> first Gate B target score
```

The held-out semantic evaluator remains a later claim boundary and must not participate in the Gate B mechanism decision.
