"""The equal-budget claim is only auditable if non-evaluation resources are recorded.

The research contract fixes the budget in target-similarity evaluations and requires
LLM-call counts to be reported separately, so a win bought with extra generation
compute cannot be presented as an adaptivity win.
"""

import numpy as np
import pytest

from benchmarks.run_semantic_inversion import run_case, trace_to_dict
from benchmarks.semantic_inversion import BenchmarkCase

BUDGET = 6


class StubEmbedder:
    def embed_text(self, text):
        vector = np.zeros(4)
        vector[len(text) % 4] = 1.0
        return type("Response", (), {"embedding": vector})()


class StubLLM:
    def __init__(self):
        self.prompts = []

    def generate_response(self, prompt):
        self.prompts.append(prompt)
        count = next((int(w) for w in prompt.split() if w.isdigit()), 1)
        content = "\n".join(f"candidate {index}" for index in range(count))
        return type("Response", (), {"content": content})()

    def generate_questions(self, current_description, target_similarity, phase, previous_questions):
        self.prompts.append(f"{phase}:{current_description}")
        return [f"{phase} candidate {len(previous_questions)}"]


@pytest.fixture
def executed_case():
    case = BenchmarkCase(case_id="c1", domain="test", source_text="a hidden source sentence")
    llm = StubLLM()
    traces, meter = run_case(case, llm=llm, embedder=StubEmbedder(), budget=BUDGET)
    return case, llm, traces, meter


def test_every_method_spends_the_same_evaluation_budget(executed_case):
    _, _, traces, _ = executed_case
    assert [len(trace.observations) for trace in traces] == [BUDGET] * 3


def test_llm_calls_are_recorded_per_method(executed_case):
    _, _, _, meter = executed_case
    assert meter.by_method["independent_best_of_n"]["llm_calls"] == 1
    assert meter.by_method["mutation_hill_climber"]["llm_calls"] == BUDGET - 1
    assert meter.by_method["adaptive_perquire"]["llm_calls"] == BUDGET


def test_generation_asymmetry_is_visible_in_the_record(executed_case):
    case, _, traces, meter = executed_case
    records = {
        trace.method: trace_to_dict(case, trace, meter.by_method.get(trace.method))
        for trace in traces
    }
    assert {record["evaluations"] for record in records.values()} == {BUDGET}
    assert records["adaptive_perquire"]["llm_calls"] > records["independent_best_of_n"]["llm_calls"]
    for record in records.values():
        assert record["empty_generations"] == 0


def test_hidden_source_text_never_reaches_a_proposer(executed_case):
    case, llm, _, _ = executed_case
    assert llm.prompts
    assert all(case.source_text not in prompt for prompt in llm.prompts)


def test_candidate_parsing_keeps_leading_digits():
    """A candidate that legitimately starts with a digit must survive parsing.

    Stripping enumeration with a character class also eats real leading digits,
    which silently changes the text that gets embedded and scored.
    """
    from benchmarks.run_semantic_inversion import parse_candidates

    assert parse_candidates("- 3D printing techniques", 1) == ["3D printing techniques"]
    assert parse_candidates("1980s synthpop revival", 1) == ["1980s synthpop revival"]
    assert parse_candidates("2. 1980s synthpop revival", 1) == ["1980s synthpop revival"]
