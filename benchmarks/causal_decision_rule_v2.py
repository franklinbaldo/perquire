"""Prospective deterministic decision rule for causal-feedback v2 Gate B."""

from __future__ import annotations

from typing import Any

DECISION_RULE_VERSION = "causal-feedback-v2-decision-rule-1"
PRIMARY_CHECKPOINT = 16
CONFIRMATORY_CHECKPOINT = 32
MIN_VALID_TARGETS = 20
MIN_POSITIVE_FRACTION = 2 / 3
COMPARISONS = ("true_minus_decoy", "true_minus_null")
METRICS = ("best", "auc")


def _summary(effects: dict[str, Any], checkpoint: int, comparison: str, metric: str) -> dict[str, Any]:
    try:
        return effects[str(checkpoint)][f"{comparison}_{metric}"]
    except KeyError as exc:
        raise ValueError(
            f"missing {comparison}_{metric} at checkpoint {checkpoint}"
        ) from exc


def _passes(summary: dict[str, Any]) -> bool:
    return (
        int(summary.get("n_targets", 0)) >= MIN_VALID_TARGETS
        and summary.get("median") is not None
        and float(summary["median"]) > 0.0
        and summary.get("positive_fraction") is not None
        and float(summary["positive_fraction"]) >= MIN_POSITIVE_FRACTION
    )


def _against(summary: dict[str, Any]) -> bool:
    return (
        int(summary.get("n_targets", 0)) >= MIN_VALID_TARGETS
        and summary.get("median") is not None
        and float(summary["median"]) <= 0.0
        and summary.get("positive_fraction") is not None
        and float(summary["positive_fraction"]) <= 0.5
    )


def evaluate_checkpoint(effects: dict[str, Any], checkpoint: int) -> dict[str, Any]:
    cells = {
        f"{comparison}_{metric}": _summary(effects, checkpoint, comparison, metric)
        for comparison in COMPARISONS
        for metric in METRICS
    }
    if all(_passes(summary) for summary in cells.values()):
        decision = "support"
    elif any(_against(summary) for summary in cells.values()):
        decision = "against"
    else:
        decision = "ambiguous"
    return {
        "checkpoint": checkpoint,
        "decision": decision,
        "criteria": {
            "minimum_valid_targets": MIN_VALID_TARGETS,
            "minimum_positive_fraction": MIN_POSITIVE_FRACTION,
            "median_must_be_strictly_positive": True,
            "both_controls_and_both_metrics_required": True,
        },
    }


def decide(target_level_effects: dict[str, Any]) -> dict[str, Any]:
    """Return the preregistered Gate-B decision without inspecting raw records."""
    at_16 = evaluate_checkpoint(target_level_effects, PRIMARY_CHECKPOINT)
    result: dict[str, Any] = {
        "decision_rule_version": DECISION_RULE_VERSION,
        "b16": at_16,
        "escalate_to_b32": at_16["decision"] == "support",
        "final_decision": at_16["decision"] if at_16["decision"] != "support" else "pending_b32",
    }
    if at_16["decision"] == "support" and str(CONFIRMATORY_CHECKPOINT) in target_level_effects:
        at_32 = evaluate_checkpoint(target_level_effects, CONFIRMATORY_CHECKPOINT)
        result["b32"] = at_32
        result["final_decision"] = at_32["decision"]
    return result
