"""Utilities for building a simple Wikipedia knowledge base."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List
import json
import numpy as np


@dataclass
class WikipediaConcept:
    """Representation of a Wikipedia concept."""

    title: str
    category: str
    url: str
    embedding: List[float] | None = None


class WikipediaDatasetBuilder:
    """Create a small concept dataset for experimentation.

    This is a very minimal placeholder implementation. A real implementation
    would fetch data from Wikipedia APIs and curate a balanced concept list.
    """

    def build_dataset(self) -> List[WikipediaConcept]:
        concepts = [
            WikipediaConcept(
                title="Coffee shop",
                category="places",
                url="https://en.wikipedia.org/wiki/Coffee_shop",
            ),
            WikipediaConcept(
                title="Chair",
                category="physical_objects",
                url="https://en.wikipedia.org/wiki/Chair",
            ),
        ]
        return concepts

    def save_dataset(self, concepts: List[WikipediaConcept], path: str) -> None:
        data = [concept.__dict__ for concept in concepts]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


class BatchEmbeddingGenerator:
    """Generate simple random embeddings for a list of concepts."""

    def __init__(self, dimensions: int = 8):
        self.dimensions = dimensions

    def generate(self, concepts: List[WikipediaConcept]) -> None:
        for concept in concepts:
            concept.embedding = np.random.rand(self.dimensions).astype(float).tolist()
