"""
Investigation result classes and metrics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any
import json
from pathlib import Path

from ..exceptions import ValidationError


@dataclass
class QuestionResult:
    """Represents the result of a single question during investigation."""
    question: str
    similarity_score: float
    phase: str  # 'exploration', 'refinement', 'convergence'
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QuestionAnswer:
    """Backwards compatible question representation used in tests."""

    question: str
    similarity: float
    phase: str
    iteration: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def similarity_score(self) -> float:
        """Expose similarity as similarity_score for compatibility."""
        return self.similarity


@dataclass
class InvestigationResult:
    """
    Contains the complete result of an embedding investigation.
    
    This class holds all information about the investigation process,
    including the final description, confidence metrics, and investigation history.
    """
    
    # Core results
    description: str
    final_similarity: float
    iterations: int
    
    # Investigation metadata
    investigation_id: str = field(default_factory=lambda: f"inv_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None
    phase_reached: str = "exploration"  # exploration, refinement, convergence
    
    # Question history
    question_history: List[QuestionAnswer] = field(default_factory=list)
    
    # Convergence metrics
    convergence_achieved: bool = False
    convergence_reason: str = ""
    similarity_threshold: float = 0.90
    
    # Performance metrics
    total_duration_seconds: Optional[float] = None
    questions_per_phase: Dict[str, int] = field(default_factory=dict)
    average_similarity_improvement: Optional[float] = None
    
    # Configuration used
    strategy_name: str = "default"
    model_config: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate and compute derived metrics after initialization."""
        if self.final_similarity < 0 or self.final_similarity > 1:
            raise ValidationError(f"Similarity score must be between 0 and 1, got {self.final_similarity}")
        
        if self.iterations < 0:
            raise ValidationError(f"Iterations must be non-negative, got {self.iterations}")
        
        # Compute duration if end_time is set
        if self.end_time:
            self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
        
        # Compute questions per phase
        if self.question_history:
            phase_counts = {}
            for question in self.question_history:
                phase_counts[question.phase] = phase_counts.get(question.phase, 0) + 1
            self.questions_per_phase = phase_counts
    
    @property
    def confidence_percentage(self) -> float:
        """Return final similarity as a percentage."""
        return self.final_similarity * 100
    
    @property
    def investigation_summary(self) -> str:
        """Generate a human-readable summary of the investigation."""
        return (
            f"Investigation {self.investigation_id}: "
            f"'{self.description}' "
            f"(confidence: {self.confidence_percentage:.1f}%, "
            f"iterations: {self.iterations}, "
            f"phase: {self.phase_reached})"
        )
    
    def add_question_result(self, question: str, similarity: float, phase: str, metadata: Dict[str, Any] = None):
        """Add a question result to the investigation history."""
        result = QuestionResult(
            question=question,
            similarity_score=similarity,
            phase=phase,
            metadata=metadata or {}
        )
        self.question_history.append(result)

    def add_question_answer(self, qa: QuestionAnswer) -> None:
        """Add a QuestionAnswer object to the history (legacy support)."""
        if "iteration" not in qa.metadata:
            qa.metadata["iteration"] = qa.iteration
        self.question_history.append(qa)
    
    def get_similarity_progression(self) -> List[float]:
        """Get the progression of similarity scores throughout the investigation."""
        return [q.similarity_score for q in self.question_history]
    
    def get_best_questions(self, top_k: int = 5) -> List[QuestionAnswer]:
        """Get the top-k questions with highest similarity scores."""
        return sorted(self.question_history, key=lambda x: x.similarity_score, reverse=True)[:top_k]
    
    def calculate_improvement_rate(self) -> float:
        """Calculate the average improvement rate between questions."""
        if len(self.question_history) < 2:
            return 0.0
        
        improvements = []
        for i in range(1, len(self.question_history)):
            current = self.question_history[i].similarity_score
            previous = self.question_history[i-1].similarity_score
            improvements.append(current - previous)
        
        self.average_similarity_improvement = sum(improvements) / len(improvements)
        return self.average_similarity_improvement
    
    def mark_convergence(self, reason: str):
        """Mark the investigation as converged with a reason."""
        self.convergence_achieved = True
        self.convergence_reason = reason
        self.end_time = datetime.now()
        if self.total_duration_seconds is None and self.end_time:
            self.total_duration_seconds = (self.end_time - self.start_time).total_seconds()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the result to a dictionary for serialization."""
        return {
            "investigation_id": self.investigation_id,
            "description": self.description,
            "final_similarity": self.final_similarity,
            "iterations": self.iterations,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "phase_reached": self.phase_reached,
            "convergence_achieved": self.convergence_achieved,
            "convergence_reason": self.convergence_reason,
            "similarity_threshold": self.similarity_threshold,
            "total_duration_seconds": self.total_duration_seconds,
            "questions_per_phase": self.questions_per_phase,
            "average_similarity_improvement": self.average_similarity_improvement,
            "strategy_name": self.strategy_name,
            "model_config": self.model_config,
            "question_history": [
                {
                    "question": q.question,
                    "similarity_score": q.similarity_score,
                    "phase": q.phase,
                    "iteration": getattr(q, "iteration", 0),
                    "timestamp": q.timestamp.isoformat(),
                    "metadata": q.metadata,
                }
                for q in self.question_history
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvestigationResult":
        """Create an InvestigationResult from a dictionary."""
        # Parse question history
        question_history: List[QuestionAnswer] = []
        for q_data in data.get("question_history", []):
            ts_raw = q_data.get("timestamp")
            ts = datetime.fromisoformat(ts_raw) if isinstance(ts_raw, str) else ts_raw
            meta_raw = q_data.get("metadata", {})
            meta = json.loads(meta_raw) if isinstance(meta_raw, str) else meta_raw
            question_history.append(QuestionAnswer(
                question=q_data["question"],
                similarity=q_data.get("similarity", q_data.get("similarity_score", 0.0)),
                phase=q_data["phase"],
                iteration=q_data.get("iteration", meta.get("iteration", 0)),
                timestamp=ts,
                metadata=meta
            ))
        
        return cls(
            investigation_id=data["investigation_id"],
            description=data["description"],
            final_similarity=data["final_similarity"],
            iterations=data["iterations"],
            start_time=datetime.fromisoformat(data["start_time"]) if isinstance(data.get("start_time"), str) else data.get("start_time"),
            end_time=datetime.fromisoformat(data["end_time"]) if isinstance(data.get("end_time"), str) else data.get("end_time"),
            phase_reached=data.get("phase_reached", "exploration"),
            question_history=question_history,
            convergence_achieved=data.get("convergence_achieved", False),
            convergence_reason=data.get("convergence_reason", ""),
            similarity_threshold=data.get("similarity_threshold", 0.90),
            total_duration_seconds=data.get("total_duration_seconds"),
            questions_per_phase=data.get("questions_per_phase", {}),
            average_similarity_improvement=data.get("average_similarity_improvement"),
            strategy_name=data.get("strategy_name", "default"),
            model_config=json.loads(data["model_config"]) if isinstance(data.get("model_config"), str) else data.get("model_config", {})
        )
    
    def save_to_file(self, file_path: str):
        """Save the investigation result to a JSON file."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, file_path: str) -> "InvestigationResult":
        """Load an investigation result from a JSON file."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def export_to_csv(self, file_path: str):
        """Export question history to CSV for analysis."""
        import csv
        
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w', newline='') as csvfile:
            fieldnames = ['question', 'similarity_score', 'phase', 'timestamp']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for question in self.question_history:
                writer.writerow({
                    'question': question.question,
                    'similarity_score': question.similarity_score,
                    'phase': question.phase,
                    'timestamp': question.timestamp.isoformat()
                })