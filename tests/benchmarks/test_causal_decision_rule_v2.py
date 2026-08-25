from benchmarks.causal_decision_rule_v2 import decide


def _summary(median: float, positive_fraction: float, n_targets: int = 24):
    return {
        "n_targets": n_targets,
        "mean": median,
        "median": median,
        "positive_fraction": positive_fraction,
        "target_effects": [],
    }


def _checkpoint(median: float, positive_fraction: float, n_targets: int = 24):
    return {
        f"true_minus_{control}_{metric}": _summary(median, positive_fraction, n_targets)
        for control in ("decoy", "null")
        for metric in ("best", "auc")
    }


def test_support_at_16_requires_confirmation_at_32():
    result = decide({"16": _checkpoint(0.02, 0.75)})
    assert result["b16"]["decision"] == "support"
    assert result["escalate_to_b32"] is True
    assert result["final_decision"] == "pending_b32"


def test_support_must_replicate_at_32():
    result = decide({"16": _checkpoint(0.02, 0.75), "32": _checkpoint(0.03, 0.8)})
    assert result["final_decision"] == "support"


def test_one_clear_negative_cell_is_against():
    checkpoint = _checkpoint(0.02, 0.75)
    checkpoint["true_minus_null_auc"] = _summary(0.0, 0.5)
    result = decide({"16": checkpoint})
    assert result["final_decision"] == "against"
    assert result["escalate_to_b32"] is False


def test_mixed_direction_is_ambiguous():
    checkpoint = _checkpoint(0.02, 0.75)
    checkpoint["true_minus_null_auc"] = _summary(0.001, 0.6)
    result = decide({"16": checkpoint})
    assert result["final_decision"] == "ambiguous"


def test_too_few_valid_targets_cannot_support():
    result = decide({"16": _checkpoint(0.02, 0.9, n_targets=19)})
    assert result["final_decision"] == "ambiguous"
