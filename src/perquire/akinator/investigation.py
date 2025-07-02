"""Akinator-style investigation engine components."""

from __future__ import annotations

from typing import List
import numpy as np


class OptimizedBootstrapInvestigator:
    """Fast bootstrap using vectorized similarity."""

    def __init__(self, knowledge_embeddings: np.ndarray):
        self.knowledge_embeddings = knowledge_embeddings

    def fast_bootstrap(self, target_embedding: np.ndarray, top_k: int = 10) -> List[int]:
        sims = self._cosine_sim(target_embedding, self.knowledge_embeddings)
        top_indices = np.argsort(-sims)[:top_k]
        return top_indices.tolist()

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(b_norm, a_norm)


class DimensionalAnalyzer:
    """Evaluate questions across semantic dimensions."""

    def __init__(self, question_embeddings: np.ndarray):
        self.question_embeddings = question_embeddings

    def evaluate_all_dimensions(self, target_embedding: np.ndarray) -> np.ndarray:
        sims = self._cosine_sim(target_embedding, self.question_embeddings)
        return sims

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a_norm = a / (np.linalg.norm(a) + 1e-8)
        b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
        return np.dot(b_norm, a_norm)
