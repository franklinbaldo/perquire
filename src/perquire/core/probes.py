"""Semantic probe primitives.

A probe score is evidence about proximity in an embedding space, not a yes/no
answer from the target vector. Contrastive sets make that distinction explicit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence


@dataclass(frozen=True)
class ScoredProbe:
    text: str
    similarity: float


@dataclass(frozen=True)
class ContrastiveEvidence:
    probes: tuple[ScoredProbe, ...]

    @property
    def best(self) -> ScoredProbe:
        if not self.probes:
            raise ValueError("Contrastive evidence requires at least one probe")
        return max(self.probes, key=lambda probe: probe.similarity)

    @property
    def ranking(self) -> tuple[ScoredProbe, ...]:
        return tuple(sorted(self.probes, key=lambda probe: probe.similarity, reverse=True))

    def margin(self) -> float:
        ranked = self.ranking
        if len(ranked) < 2:
            return 0.0
        return ranked[0].similarity - ranked[1].similarity


def evaluate_contrastive_set(
    probes: Sequence[str], score: Callable[[str], float]
) -> ContrastiveEvidence:
    """Evaluate semantically contrasting candidate texts against one target."""
    if not probes:
        raise ValueError("At least one probe is required")
    return ContrastiveEvidence(
        probes=tuple(ScoredProbe(text=probe, similarity=score(probe)) for probe in probes)
    )
