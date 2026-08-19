"""The equal-budget claim is only auditable if all resources are recorded."""

import numpy as np
import pytest

from benchmarks.run_semantic_inversion import run_case, trace_to_dict
from benchmarks.semantic_inversion import BenchmarkCase

BUDGET = 6


class StubEmbedder:
    def __init__(self):
        self.transport_attempts = 0

    def embed_text(self, text):
        self.transport_attempts += 1
        vector = np.zeros(4)
        vector[len(text) % 4] = 1.0
        return type("Response", (), {"embedding": vector})()


class StubLLM:
    def __init__(self):
        self.prompts = []
        self.request_ids = []
        self.transport_attempts = 0

    def generate_response(self, prompt, **kwargs):
        self.prompts.append(prompt)
        self.request_ids.append(kwargs.get("cache_request_id"))
        self.transport_attempts += 1
        count = next((int(w) for w in prompt.split() if w.isdigit()), 1)
        content = "\n".join(f"candidate {index}" for index in range(count))
        return type("Response", (), {"content": content})()

    def generate_questions(
        self, current_description, target_similarity, phase, previous_questions, **kwargs
    ):
        self.prompts.append(f"{phase}:{current_description}")
        self.request_ids.append(kwargs.get("cache_request_id"))
        self.transport_attempts += 1
        return [f"{phase} candidate {len(previous_questions)}"]


@pytest.fixture
def executed_case():
    case = BenchmarkCase(case_id="c1", domain="test", source_text="a hidden source sentence")
    llm = StubLLM()
    embedder = StubEmbedder()
    traces, meter, target_attempts = run_case(
        case, llm=llm, embedder=embedder, budget=BUDGET
    )
    return case, llm, embedder, traces, meter, target_attempts


def test_every_method_spends_the_same_evaluation_budget(executed_case):
    _, _, _, traces, _, _ = executed_case
    assert [len(trace.observations) for trace in traces] == [BUDGET] * 3


def test_llm_calls_are_recorded_per_method(executed_case):
    _, _, _, _, meter, _ = executed_case
    assert meter.by_method["independent_best_of_n"]["llm_calls"] == 1
    assert meter.by_method["mutation_hill_climber"]["llm_calls"] == BUDGET
    assert meter.by_method["adaptive_perquire"]["llm_calls"] == BUDGET


def test_transport_attempts_are_separate_from_logical_calls(executed_case):
    _, _, _, _, meter, target_attempts = executed_case
    assert target_attempts == 1
    assert meter.by_method["independent_best_of_n"]["llm_transport_attempts"] == 1
    assert meter.by_method["mutation_hill_climber"]["llm_transport_attempts"] == BUDGET
    assert meter.by_method["adaptive_perquire"]["llm_transport_attempts"] == BUDGET
    assert {
        meter.by_method[method]["embedding_transport_attempts"]
        for method in meter.by_method
    } == {BUDGET}


def test_hill_climber_pays_for_its_own_seed(executed_case):
    _, llm, _, traces, meter, _ = executed_case
    hill = next(trace for trace in traces if trace.method == "mutation_hill_climber")
    independent = next(trace for trace in traces if trace.method == "independent_best_of_n")
    assert hill.observations[0].candidate == "candidate 0"
    assert meter.by_method["mutation_hill_climber"]["generated_candidates"] == BUDGET
    assert f"c1:mutation_hill_climber:seed" in llm.request_ids
    assert hill.observations[0] is not independent.observations[0]


def test_generation_asymmetry_is_visible_in_the_record(executed_case):
    case, _, _, traces, meter, _ = executed_case
    records = {
        trace.method: trace_to_dict(case, trace, meter.by_method.get(trace.method))
        for trace in traces
    }
    assert {record["evaluations"] for record in records.values()} == {BUDGET}
    assert records["adaptive_perquire"]["llm_calls"] > records["independent_best_of_n"]["llm_calls"]
    assert records["mutation_hill_climber"]["llm_calls"] == records["adaptive_perquire"]["llm_calls"]
    for record in records.values():
        assert record["empty_generations"] == 0
        assert "llm_transport_attempts" in record
        assert "embedding_transport_attempts" in record


def test_hidden_source_text_never_reaches_a_proposer(executed_case):
    case, llm, _, _, _, _ = executed_case
    assert llm.prompts
    assert all(case.source_text not in prompt for prompt in llm.prompts)


def test_every_llm_request_has_a_stable_case_scoped_replay_identity(executed_case):
    case, llm, _, _, _, _ = executed_case
    assert llm.request_ids
    assert all(request_id and request_id.startswith(f"{case.case_id}:") for request_id in llm.request_ids)
    assert len(llm.request_ids) == len(set(llm.request_ids))
    assert all(case.source_text not in request_id for request_id in llm.request_ids)


def test_candidate_parsing_keeps_leading_digits():
    from benchmarks.run_semantic_inversion import parse_candidates

    assert parse_candidates("- 3D printing techniques", 1) == ["3D printing techniques"]
    assert parse_candidates("1980s synthpop revival", 1) == ["1980s synthpop revival"]
    assert parse_candidates("2. 1980s synthpop revival", 1) == ["1980s synthpop revival"]


def test_independent_generation_fills_partial_model_output_to_budget():
    class PartialLLM(StubLLM):
        def generate_response(self, prompt, **kwargs):
            self.prompts.append(prompt)
            self.request_ids.append(kwargs.get("cache_request_id"))
            self.transport_attempts += 1
            return type("Response", (), {"content": f"candidate {self.transport_attempts}"})()

    case = BenchmarkCase(case_id="partial", domain="test", source_text="hidden")
    llm = PartialLLM()
    traces, meter, _ = run_case(case, llm=llm, embedder=StubEmbedder(), budget=2)
    independent = traces[0]
    assert len(independent.observations) == 2
    assert meter.by_method["independent_best_of_n"]["llm_calls"] == 2
