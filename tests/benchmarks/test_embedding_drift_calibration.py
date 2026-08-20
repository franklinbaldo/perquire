from pathlib import Path

from benchmarks.check_embedding_drift_and_calibrate import load_paraphrases
from benchmarks.run_semantic_inversion import load_cases


def test_fixed_calibration_paraphrases_cover_every_v1_case_once():
    cases = load_cases(Path("benchmarks/cases_v1.jsonl"))
    paraphrases = load_paraphrases(Path("benchmarks/calibration_paraphrases_v1.jsonl"))

    assert set(paraphrases) == {case.case_id for case in cases}
    assert len(paraphrases) == 24
    assert all(paraphrase.strip() for paraphrase in paraphrases.values())
