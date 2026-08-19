from perquire.core.probes import evaluate_contrastive_set


def test_contrastive_evidence_ranks_by_similarity() -> None:
    evidence = evaluate_contrastive_set(
        ["short", "a much longer semantic candidate", "middle sized"],
        score=lambda text: len(text) / 100,
    )

    assert evidence.best.text == "a much longer semantic candidate"
    assert evidence.ranking[0] == evidence.best
    assert evidence.margin() > 0


def test_contrastive_evidence_rejects_empty_set() -> None:
    try:
        evaluate_contrastive_set([], score=lambda _text: 0.0)
    except ValueError as exc:
        assert "At least one probe" in str(exc)
    else:
        raise AssertionError("empty contrastive set should fail")
