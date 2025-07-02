"""Massive question generation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Dict
import json


@dataclass
class DimensionalQuestion:
    """Question representing one pole of a semantic dimension."""

    text: str
    dimension: str
    pole: str
    category: str


class MassiveQuestionGenerator:
    """Generate a simple question bank.

    This placeholder uses predefined examples instead of an LLM. In a full
    implementation, questions would be generated with a language model and
    embedded for similarity calculations.
    """

    def generate_question_bank(self) -> List[DimensionalQuestion]:
        questions = [
            DimensionalQuestion(
                text="This represents something abstract",
                dimension="abstraction_level",
                pole="positive",
                category="ontological",
            ),
            DimensionalQuestion(
                text="This represents something concrete",
                dimension="abstraction_level",
                pole="negative",
                category="ontological",
            ),
        ]
        return questions

    def save_question_bank(self, questions: List[DimensionalQuestion], path: str) -> None:
        data = [q.__dict__ for q in questions]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
