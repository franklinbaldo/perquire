"""
Questioning strategy classes and templates.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import random

from ..exceptions import ValidationError, QuestionGenerationError


class QuestioningStrategyRegistry:
    """Registry for managing available questioning strategies."""

    def __init__(self) -> None:
        self._strategies: Dict[str, "QuestioningStrategy"] = {}
        self._default: Optional[str] = None

    def register(self, name: str, strategy: "QuestioningStrategy", set_as_default: bool = False) -> None:
        """Register a strategy with the given name."""
        self._strategies[name] = strategy
        if set_as_default or self._default is None:
            self._default = name

    def get(self, name: Optional[str] = None) -> "QuestioningStrategy":
        """Retrieve a strategy by name or return the default."""
        if name is None:
            if self._default is None:
                raise KeyError("No strategy registered")
            name = self._default
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' not found")
        return self._strategies[name]

    def list_names(self) -> List[str]:
        """Return a list of registered strategy names."""
        return list(self._strategies.keys())

    def set_default(self, name: str) -> None:
        if name not in self._strategies:
            raise KeyError(f"Strategy '{name}' not found")
        self._default = name


class InvestigationPhase(Enum):
    """Investigation phases enum."""
    EXPLORATION = "exploration"
    REFINEMENT = "refinement"
    CONVERGENCE = "convergence"


@dataclass
class QuestionTemplate:
    """Represents a question template with metadata."""
    template: str
    phase: InvestigationPhase
    priority: int = 1  # Higher numbers = higher priority
    tags: List[str] = field(default_factory=list)
    variables: List[str] = field(default_factory=list)  # Variables that can be filled in template


class QuestioningStrategy:
    """
    Defines how Perquire approaches different types of content investigation.
    
    This class encapsulates the logic for generating questions during different
    phases of the investigation process.
    """
    
    def __init__(
        self,
        name: str = "default",
        exploration_templates: Optional[List[str]] = None,
        refinement_templates: Optional[List[str]] = None,
        convergence_templates: Optional[List[str]] = None,
        refinement_threshold: float = 0.7,
        convergence_threshold: float = 0.90,
        max_iterations: int = 25,
        exploration_depth: int = 5,
        min_improvement: float = 0.001,
        convergence_window: int = 3,
        custom_generators: Optional[Dict[str, Callable]] = None
    ):
        """
        Initialize a questioning strategy.
        
        Args:
            name: Strategy name for identification
            exploration_templates: List of broad exploration questions
            refinement_templates: List of focused refinement questions  
            convergence_templates: List of specific convergence questions
            refinement_threshold: Similarity threshold to enter refinement phase
            convergence_threshold: Similarity threshold to enter convergence phase
            max_iterations: Maximum number of questions to ask
            exploration_depth: Number of questions in exploration phase
            min_improvement: Minimum similarity improvement to continue
            convergence_window: Number of iterations to check for convergence
            custom_generators: Custom question generation functions
        """
        self.name = name
        self.refinement_threshold = refinement_threshold
        self.convergence_threshold = convergence_threshold
        self.max_iterations = max_iterations
        self.exploration_depth = exploration_depth
        self.min_improvement = min_improvement
        self.convergence_window = convergence_window
        self.custom_generators = custom_generators or {}
        
        # Initialize question templates
        self.templates = {
            InvestigationPhase.EXPLORATION: self._create_exploration_templates(exploration_templates),
            InvestigationPhase.REFINEMENT: self._create_refinement_templates(refinement_templates),
            InvestigationPhase.CONVERGENCE: self._create_convergence_templates(convergence_templates)
        }
        
        # Validate configuration
        self._validate_configuration()
    
    def _validate_configuration(self):
        """Validate strategy configuration."""
        if not 0 < self.refinement_threshold < 1:
            raise ValidationError(f"Refinement threshold must be between 0 and 1, got {self.refinement_threshold}")
        
        if not 0 < self.convergence_threshold < 1:
            raise ValidationError(f"Convergence threshold must be between 0 and 1, got {self.convergence_threshold}")
        
        if self.refinement_threshold >= self.convergence_threshold:
            raise ValidationError("Refinement threshold must be less than convergence threshold")
        
        if self.max_iterations <= 0:
            raise ValidationError(f"Max iterations must be positive, got {self.max_iterations}")
        
        if self.exploration_depth <= 0:
            raise ValidationError(f"Exploration depth must be positive, got {self.exploration_depth}")
    
    def _create_exploration_templates(self, custom_templates: Optional[List[str]]) -> List[QuestionTemplate]:
        """Create exploration phase question templates."""
        default_templates = [
            "Does this relate to human emotions or feelings?",
            "Is this about concrete objects or abstract concepts?",
            "Does this involve people, places, or things?",
            "Is this related to art, science, or everyday life?",
            "Does this express positive or negative sentiment?",
            "Is this about past, present, or future?",
            "Does this involve action or description?",
            "Is this formal or informal in nature?",
            "Does this relate to personal or universal experiences?",
            "Is this about physical or mental phenomena?"
        ]
        
        templates = custom_templates if custom_templates else default_templates
        return [
            QuestionTemplate(
                template=template,
                phase=InvestigationPhase.EXPLORATION,
                priority=1,
                tags=["general", "broad"]
            )
            for template in templates
        ]
    
    def _create_refinement_templates(self, custom_templates: Optional[List[str]]) -> List[QuestionTemplate]:
        """Create refinement phase question templates."""
        default_templates = [
            "What specific emotion or feeling does this convey?",
            "What is the main subject or topic?",
            "What is the tone or mood?",
            "What level of intensity or strength does this have?",
            "What specific context or setting is this about?",
            "What relationships or connections are involved?",
            "What specific qualities or characteristics are present?",
            "What specific actions or processes are described?",
            "What specific time period or duration is involved?",
            "What specific perspective or viewpoint is taken?"
        ]
        
        templates = custom_templates if custom_templates else default_templates
        return [
            QuestionTemplate(
                template=template,
                phase=InvestigationPhase.REFINEMENT,
                priority=2,
                tags=["focused", "specific"]
            )
            for template in templates
        ]
    
    def _create_convergence_templates(self, custom_templates: Optional[List[str]]) -> List[QuestionTemplate]:
        """Create convergence phase question templates."""
        default_templates = [
            "What are the subtle nuances or details?",
            "What is the exact meaning or interpretation?",
            "What specific words or phrases capture this best?",
            "What fine distinctions or subtleties are present?",
            "What precise emotional undertones exist?",
            "What specific cultural or contextual elements are involved?",
            "What exact imagery or sensory details are present?",
            "What specific stylistic or linguistic features are notable?",
            "What precise relationships or dynamics are at play?",
            "What exact implications or connotations are present?"
        ]
        
        templates = custom_templates if custom_templates else default_templates
        return [
            QuestionTemplate(
                template=template,
                phase=InvestigationPhase.CONVERGENCE,
                priority=3,
                tags=["precise", "nuanced"]
            )
            for template in templates
        ]
    
    def determine_phase(self, current_similarity: float, iteration: int) -> InvestigationPhase:
        """
        Determine the current investigation phase based on similarity and iteration.
        
        Args:
            current_similarity: Current highest similarity score
            iteration: Current iteration number
            
        Returns:
            The appropriate investigation phase
        """
        # Force exploration phase for first few iterations
        if iteration < self.exploration_depth:
            return InvestigationPhase.EXPLORATION
        
        # Move to convergence if similarity is high enough
        if current_similarity >= self.convergence_threshold:
            return InvestigationPhase.CONVERGENCE
        
        # Move to refinement if similarity meets threshold
        if current_similarity >= self.refinement_threshold:
            return InvestigationPhase.REFINEMENT
        
        # Stay in exploration if similarity is still low
        return InvestigationPhase.EXPLORATION
    
    def generate_question(
        self, 
        phase: InvestigationPhase, 
        context: Optional[Dict[str, Any]] = None,
        used_questions: Optional[List[str]] = None
    ) -> str:
        """
        Generate a question for the specified phase.
        
        Args:
            phase: Investigation phase
            context: Additional context for question generation
            used_questions: List of already used questions to avoid repetition
            
        Returns:
            Generated question string
            
        Raises:
            QuestionGenerationError: If no suitable question can be generated
        """
        used_questions = used_questions or []
        context = context or {}
        
        # Try custom generator first if available
        if phase.value in self.custom_generators:
            try:
                question = self.custom_generators[phase.value](context)
                if question and question not in used_questions:
                    return question
            except Exception as e:
                # Fall back to template-based generation if custom generator fails
                pass
        
        # Get available templates for the phase
        available_templates = [
            template for template in self.templates[phase]
            if template.template not in used_questions
        ]
        
        if not available_templates:
            raise QuestionGenerationError(f"No available templates for phase {phase.value}")
        
        # Select template based on priority and randomness
        weights = [template.priority for template in available_templates]
        selected_template = random.choices(available_templates, weights=weights)[0]
        
        # Apply template variables if context provided
        question = self._apply_template_variables(selected_template.template, context)
        
        return question
    
    def _apply_template_variables(self, template: str, context: Dict[str, Any]) -> str:
        """Apply context variables to question template."""
        # Simple template variable substitution
        # This could be expanded with more sophisticated templating
        result = template
        for key, value in context.items():
            placeholder = f"{{{key}}}"
            if placeholder in result:
                result = result.replace(placeholder, str(value))
        
        return result
    
    def should_continue(
        self, 
        similarity_scores: List[float], 
        iteration: int, 
        current_phase: InvestigationPhase
    ) -> tuple[bool, str]:
        """
        Determine if investigation should continue.
        
        Args:
            similarity_scores: List of similarity scores from investigation
            iteration: Current iteration number
            current_phase: Current investigation phase
            
        Returns:
            Tuple of (should_continue, reason)
        """
        # Check max iterations
        if iteration >= self.max_iterations:
            return False, f"Reached maximum iterations ({self.max_iterations})"
        
        # Need at least some scores to evaluate
        if len(similarity_scores) < 2:
            return True, "Need more data points"
        
        current_similarity = max(similarity_scores)
        
        # Check if we've reached convergence threshold
        if current_similarity >= self.convergence_threshold:
            # Check if improvement has plateaued
            if len(similarity_scores) >= self.convergence_window:
                recent_scores = similarity_scores[-self.convergence_window:]
                improvements = [
                    recent_scores[i] - recent_scores[i-1] 
                    for i in range(1, len(recent_scores))
                ]
                avg_improvement = sum(improvements) / len(improvements)
                
                if avg_improvement < self.min_improvement:
                    return False, f"Convergence achieved (similarity: {current_similarity:.3f}, avg improvement: {avg_improvement:.3f})"
        
        # Check for stagnation
        if len(similarity_scores) >= self.convergence_window * 2:
            recent_scores = similarity_scores[-self.convergence_window:]
            if max(recent_scores) - min(recent_scores) < self.min_improvement:
                return False, f"No significant improvement in last {self.convergence_window} iterations"
        
        return True, "Investigation continuing"
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """Get information about this strategy."""
        return {
            "name": self.name,
            "refinement_threshold": self.refinement_threshold,
            "convergence_threshold": self.convergence_threshold,
            "max_iterations": self.max_iterations,
            "exploration_depth": self.exploration_depth,
            "min_improvement": self.min_improvement,
            "convergence_window": self.convergence_window,
            "template_counts": {
                phase.value: len(templates) 
                for phase, templates in self.templates.items()
            },
            "has_custom_generators": len(self.custom_generators) > 0
        }


# Predefined strategies for common use cases
def create_artistic_strategy() -> QuestioningStrategy:
    """Create a strategy optimized for investigating artistic content."""
    return QuestioningStrategy(
        name="artistic",
        exploration_templates=[
            "Does this relate to visual, auditory, or literary art?",
            "Is this about the creation process or the appreciation of art?",
            "Does this involve specific artistic techniques or general aesthetic concepts?",
            "Is this about traditional or contemporary art forms?",
            "Does this express emotional or intellectual artistic content?"
        ],
        refinement_templates=[
            "What specific artistic medium or technique is involved?",
            "What emotional impact does this artistic work convey?",
            "What historical or cultural context is this art from?",
            "What specific aesthetic qualities are being expressed?",
            "What artistic movement or style does this represent?"
        ],
        convergence_threshold=0.92,
        refinement_threshold=0.75
    )


def create_scientific_strategy() -> QuestioningStrategy:
    """Create a strategy optimized for investigating scientific content."""
    return QuestioningStrategy(
        name="scientific",
        exploration_templates=[
            "Does this relate to physical sciences, life sciences, or formal sciences?",
            "Is this about theoretical concepts or practical applications?",
            "Does this involve research methods or scientific findings?",
            "Is this about established knowledge or cutting-edge research?",
            "Does this relate to natural phenomena or human-made systems?"
        ],
        refinement_templates=[
            "What specific scientific field or discipline is this about?",
            "What specific phenomena or processes are being described?",
            "What level of scientific complexity is involved?",
            "What specific methodologies or approaches are used?",
            "What specific applications or implications are discussed?"
        ],
        convergence_threshold=0.88,
        refinement_threshold=0.72
    )


def create_emotional_strategy() -> QuestioningStrategy:
    """Create a strategy optimized for investigating emotional content."""
    return QuestioningStrategy(
        name="emotional",
        exploration_templates=[
            "Does this express positive or negative emotions?",
            "Is this about personal or universal emotional experiences?",
            "Does this relate to current feelings or memories?",
            "Is this about emotional reactions or emotional states?",
            "Does this involve interpersonal or intrapersonal emotions?"
        ],
        refinement_templates=[
            "What specific emotion or combination of emotions is present?",
            "What intensity level does this emotion have?",
            "What specific triggers or causes are behind this emotion?",
            "What specific context or situation evokes this emotion?",
            "What specific expression or manifestation does this emotion take?"
        ],
        convergence_threshold=0.93,
        refinement_threshold=0.78,
        min_improvement=0.0005  # More sensitive to subtle emotional nuances
    )


# Global registry with some predefined strategies
strategy_registry = QuestioningStrategyRegistry()
strategy_registry.register("default", QuestioningStrategy(), set_as_default=True)
strategy_registry.register("artistic", create_artistic_strategy())
strategy_registry.register("scientific", create_scientific_strategy())
strategy_registry.register("emotional", create_emotional_strategy())
