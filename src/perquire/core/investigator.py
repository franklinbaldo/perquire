"""
Core PerquireInvestigator implementation.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np
import uuid

from .result import InvestigationResult, QuestionResult
from .strategy import QuestioningStrategy, InvestigationPhase
from ..llm.base import BaseLLMProvider, provider_registry as llm_registry
from ..embeddings.base import BaseEmbeddingProvider, embedding_registry
from ..embeddings.utils import cosine_similarity
from ..convergence.algorithms import ConvergenceDetector, ConvergenceReason
from ..database.base import BaseDatabaseProvider
from ..exceptions import (
    InvestigationError, EmbeddingError, LLMProviderError, 
    ConvergenceError, ValidationError
)

logger = logging.getLogger(__name__)


class PerquireInvestigator:
    """
    Main investigation engine for Perquire.
    
    This class orchestrates the investigation process by coordinating
    LLM providers, embedding models, questioning strategies, and convergence detection
    to systematically decode unknown embeddings through iterative questioning.
    """
    
    def __init__(
        self,
        llm_provider: Optional[Union[str, BaseLLMProvider]] = None,
        embedding_provider: Optional[Union[str, BaseEmbeddingProvider]] = None,
        questioning_strategy: Optional[QuestioningStrategy] = None,
        database_provider: Optional[BaseDatabaseProvider] = None,
        convergence_detector: Optional[ConvergenceDetector] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize PerquireInvestigator.
        
        Args:
            llm_provider: LLM provider instance or name
            embedding_provider: Embedding provider instance or name
            questioning_strategy: Strategy for generating questions
            database_provider: Database for storing results
            convergence_detector: Detector for convergence analysis
            config: Additional configuration options
        """
        self.config = config or {}
        
        # Initialize providers
        self._setup_llm_provider(llm_provider)
        self._setup_embedding_provider(embedding_provider)
        self._setup_questioning_strategy(questioning_strategy)
        self._setup_database_provider(database_provider)
        self._setup_convergence_detector(convergence_detector)
        
        # Investigation state
        self._current_investigation: Optional[InvestigationResult] = None
        self._investigation_cache: Dict[str, Any] = {}
        
        logger.info("Initialized PerquireInvestigator")
    
    def _setup_llm_provider(self, provider: Optional[Union[str, BaseLLMProvider]]):
        """Setup LLM provider."""
        if isinstance(provider, str):
            self.llm_provider = llm_registry.get_provider(provider)
        elif isinstance(provider, BaseLLMProvider):
            self.llm_provider = provider
        elif provider is None:
            # Try to get default provider
            try:
                self.llm_provider = llm_registry.get_provider()
            except Exception:
                raise InvestigationError("No LLM provider available. Please configure a provider.")
        else:
            raise InvestigationError("Invalid LLM provider type")
    
    def _setup_embedding_provider(self, provider: Optional[Union[str, BaseEmbeddingProvider]]):
        """Setup embedding provider."""
        if isinstance(provider, str):
            self.embedding_provider = embedding_registry.get_provider(provider)
        elif isinstance(provider, BaseEmbeddingProvider):
            self.embedding_provider = provider
        elif provider is None:
            # Try to get default provider
            try:
                self.embedding_provider = embedding_registry.get_provider()
            except Exception:
                raise InvestigationError("No embedding provider available. Please configure a provider.")
        else:
            raise InvestigationError("Invalid embedding provider type")
    
    def _setup_questioning_strategy(self, strategy: Optional[QuestioningStrategy]):
        """Setup questioning strategy."""
        if strategy is None:
            self.questioning_strategy = QuestioningStrategy()
        else:
            self.questioning_strategy = strategy
    
    def _setup_database_provider(self, provider: Optional[BaseDatabaseProvider]):
        """Setup database provider."""
        self.database_provider = provider
    
    def _setup_convergence_detector(self, detector: Optional[ConvergenceDetector]):
        """Setup convergence detector."""
        if detector is None:
            self.convergence_detector = ConvergenceDetector(
                similarity_threshold=self.questioning_strategy.convergence_threshold,
                min_improvement=self.questioning_strategy.min_improvement,
                convergence_window=self.questioning_strategy.convergence_window,
                max_iterations=self.questioning_strategy.max_iterations
            )
        else:
            self.convergence_detector = detector
    
    def investigate(
        self,
        target_embedding: np.ndarray,
        custom_strategy: Optional[QuestioningStrategy] = None,
        save_to_database: bool = True,
        verbose: bool = False
    ) -> InvestigationResult:
        """
        Investigate an unknown embedding through systematic questioning.
        
        Args:
            target_embedding: The embedding to investigate
            custom_strategy: Optional custom questioning strategy
            save_to_database: Whether to save results to database
            verbose: Whether to print progress information
            
        Returns:
            InvestigationResult with investigation findings
            
        Raises:
            InvestigationError: If investigation fails
        """
        try:
            # Validate inputs
            self._validate_investigation_inputs(target_embedding)
            
            # Use custom strategy if provided
            strategy = custom_strategy or self.questioning_strategy
            
            # Initialize investigation result
            investigation_id = f"inv_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:8]}"
            result = InvestigationResult(
                investigation_id=investigation_id,
                description="",  # Will be filled during investigation
                final_similarity=0.0,
                iterations=0,
                start_time=datetime.now(),
                strategy_name=strategy.name,
                model_config=self._get_model_config()
            )
            
            self._current_investigation = result
            
            if verbose:
                print(f"🔍 Starting investigation {investigation_id}")
                print(f"📊 Target embedding dimensions: {len(target_embedding)}")
                print(f"🎯 Strategy: {strategy.name}")
                print(f"📈 Convergence threshold: {strategy.convergence_threshold}")
            
            # Main investigation loop
            similarity_scores = []
            questions_asked = []
            current_phase = InvestigationPhase.EXPLORATION
            best_description = ""
            
            for iteration in range(strategy.max_iterations):
                try:
                    # Determine current phase
                    current_similarity = max(similarity_scores) if similarity_scores else 0.0
                    current_phase = strategy.determine_phase(current_similarity, iteration)
                    
                    if verbose:
                        print(f"\n🔄 Iteration {iteration + 1}/{strategy.max_iterations}")
                        print(f"📍 Phase: {current_phase.value}")
                        print(f"📊 Current best similarity: {current_similarity:.3f}")
                    
                    # Generate question
                    question = self._generate_question(
                        phase=current_phase,
                        strategy=strategy,
                        previous_questions=questions_asked,
                        current_description=best_description,
                        current_similarity=current_similarity
                    )
                    
                    if verbose:
                        print(f"❓ Question: {question}")
                    
                    # Calculate similarity for question
                    question_similarity = self._calculate_question_similarity(
                        question, target_embedding
                    )
                    
                    if verbose:
                        print(f"📈 Similarity: {question_similarity:.3f}")
                    
                    # Update tracking
                    similarity_scores.append(question_similarity)
                    questions_asked.append(question)
                    
                    # Add question to result
                    result.add_question_result(
                        question=question,
                        similarity=question_similarity,
                        phase=current_phase.value,
                        metadata={
                            "iteration": iteration + 1,
                            "phase": current_phase.value,
                            "generated_by": "llm" if hasattr(self.llm_provider, 'generate_questions') else "template"
                        }
                    )
                    
                    # Update best description if this is the best similarity so far
                    if question_similarity == max(similarity_scores):
                        best_description = question
                    
                    # Check convergence
                    convergence_result = self.convergence_detector.should_continue(
                        similarity_scores=similarity_scores,
                        current_iteration=iteration + 1,
                        phase=current_phase.value
                    )
                    
                    if verbose:
                        print(f"🎯 Convergence check: {convergence_result.converged} ({convergence_result.reason.value})")
                    
                    if convergence_result.converged:
                        result.mark_convergence(convergence_result.reason.value)
                        result.phase_reached = current_phase.value
                        break
                    
                except Exception as e:
                    logger.warning(f"Error in iteration {iteration + 1}: {str(e)}")
                    if verbose:
                        print(f"⚠️ Error in iteration: {str(e)}")
                    continue
            
            # Finalize investigation
            result.iterations = len(similarity_scores)
            result.final_similarity = max(similarity_scores) if similarity_scores else 0.0
            result.phase_reached = current_phase.value
            
            # Generate final description
            final_description = self._synthesize_description(
                questions_and_scores=[
                    {"question": q, "similarity": s} 
                    for q, s in zip(questions_asked, similarity_scores)
                ],
                final_similarity=result.final_similarity
            )
            
            result.description = final_description
            result.end_time = datetime.now()
            
            if verbose:
                print(f"\n✅ Investigation completed!")
                print(f"📝 Final description: {final_description}")
                print(f"📊 Final similarity: {result.final_similarity:.3f}")
                print(f"🔄 Total iterations: {result.iterations}")
                print(f"⏱️ Duration: {result.total_duration_seconds:.2f}s")
            
            # Save to database if requested and available
            if save_to_database and self.database_provider:
                try:
                    self.database_provider.save_investigation(result.to_dict())
                    if verbose:
                        print(f"💾 Saved to database: {result.investigation_id}")
                except Exception as e:
                    logger.warning(f"Failed to save investigation to database: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Investigation failed: {str(e)}")
            raise InvestigationError(f"Investigation failed: {str(e)}")
        
        finally:
            self._current_investigation = None
    
    def _validate_investigation_inputs(self, target_embedding: np.ndarray):
        """Validate investigation inputs."""
        if not isinstance(target_embedding, np.ndarray):
            raise ValidationError("Target embedding must be a numpy array")
        
        if len(target_embedding.shape) != 1:
            raise ValidationError("Target embedding must be 1-dimensional")
        
        if not np.isfinite(target_embedding).all():
            raise ValidationError("Target embedding contains invalid values (NaN or inf)")
        
        if np.linalg.norm(target_embedding) == 0:
            raise ValidationError("Target embedding is a zero vector")
        
        # Check provider health
        if not self.llm_provider.is_available():
            raise InvestigationError("LLM provider is not available")
        
        if not self.embedding_provider.is_available():
            raise InvestigationError("Embedding provider is not available")
    
    def _generate_question(
        self,
        phase: InvestigationPhase,
        strategy: QuestioningStrategy,
        previous_questions: List[str],
        current_description: str,
        current_similarity: float
    ) -> str:
        """Generate a question for the current investigation phase."""
        try:
            # Try LLM-based question generation first
            if hasattr(self.llm_provider, 'generate_questions'):
                try:
                    questions = self.llm_provider.generate_questions(
                        current_description=current_description,
                        target_similarity=current_similarity,
                        phase=phase.value,
                        previous_questions=previous_questions[-10:]  # Last 10 questions
                    )
                    
                    if questions:
                        # Return first new question that wasn't asked before
                        for question in questions:
                            if question not in previous_questions:
                                return question
                
                except Exception as e:
                    logger.warning(f"LLM question generation failed: {str(e)}")
            
            # Fallback to strategy-based question generation
            context = {
                "current_description": current_description,
                "current_similarity": current_similarity,
                "iteration": len(previous_questions) + 1,
                "phase": phase.value
            }
            
            question = strategy.generate_question(
                phase=phase,
                context=context,
                used_questions=previous_questions
            )
            
            return question
            
        except Exception as e:
            raise InvestigationError(f"Failed to generate question: {str(e)}")
    
    def _calculate_question_similarity(self, question: str, target_embedding: np.ndarray) -> float:
        """Calculate similarity between question and target embedding."""
        try:
            # Generate embedding for question
            question_result = self.embedding_provider.embed_text(question)
            question_embedding = question_result.embedding
            
            # Calculate cosine similarity
            similarity = cosine_similarity(target_embedding, question_embedding)
            
            # Ensure similarity is in valid range
            similarity = max(0.0, min(1.0, similarity))
            
            return similarity
            
        except Exception as e:
            raise EmbeddingError(f"Failed to calculate question similarity: {str(e)}")
    
    def _synthesize_description(
        self,
        questions_and_scores: List[Dict[str, Any]],
        final_similarity: float
    ) -> str:
        """Synthesize final description from investigation results."""
        try:
            # Try LLM-based synthesis first
            if hasattr(self.llm_provider, 'synthesize_description'):
                try:
                    description = self.llm_provider.synthesize_description(
                        questions_and_scores=questions_and_scores,
                        final_similarity=final_similarity
                    )
                    
                    if description and description.strip():
                        return description.strip()
                
                except Exception as e:
                    logger.warning(f"LLM description synthesis failed: {str(e)}")
            
            # Fallback to simple best question approach
            if questions_and_scores:
                # Sort by similarity and take the best questions
                sorted_questions = sorted(
                    questions_and_scores, 
                    key=lambda x: x["similarity"], 
                    reverse=True
                )
                
                best_questions = sorted_questions[:3]
                
                if len(best_questions) == 1:
                    return best_questions[0]["question"]
                else:
                    # Combine top questions
                    descriptions = [q["question"] for q in best_questions]
                    return f"Likely represents: {' and '.join(descriptions[:2])}"
            
            return "Unable to determine description from investigation"
            
        except Exception as e:
            logger.warning(f"Description synthesis failed: {str(e)}")
            return "Error during description synthesis"
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get current model configuration."""
        return {
            "llm_provider": {
                "provider": getattr(self.llm_provider, 'provider', 'unknown'),
                "model_info": self.llm_provider.get_model_info()
            },
            "embedding_provider": {
                "provider": getattr(self.embedding_provider, 'provider', 'unknown'),
                "model_info": self.embedding_provider.get_model_info()
            },
            "strategy": self.questioning_strategy.get_strategy_info()
        }
    
    def investigate_batch(
        self,
        embeddings: List[np.ndarray],
        custom_strategy: Optional[QuestioningStrategy] = None,
        parallel: bool = False,
        save_to_database: bool = True,
        verbose: bool = False
    ) -> List[InvestigationResult]:
        """
        Investigate multiple embeddings.
        
        Args:
            embeddings: List of embeddings to investigate
            custom_strategy: Optional custom questioning strategy
            parallel: Whether to use parallel processing (not implemented yet)
            save_to_database: Whether to save results to database
            verbose: Whether to print progress information
            
        Returns:
            List of InvestigationResult objects
        """
        results = []
        
        for i, embedding in enumerate(embeddings):
            if verbose:
                print(f"\n🔍 Investigating embedding {i + 1}/{len(embeddings)}")
            
            try:
                result = self.investigate(
                    target_embedding=embedding,
                    custom_strategy=custom_strategy,
                    save_to_database=save_to_database,
                    verbose=verbose
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to investigate embedding {i}: {str(e)}")
                if verbose:
                    print(f"❌ Failed to investigate embedding {i}: {str(e)}")
        
        return results
    
    def get_investigation_history(
        self,
        limit: int = 10,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Get investigation history from database.
        
        Args:
            limit: Maximum number of results
            filters: Optional filters to apply
            
        Returns:
            List of investigation summaries
        """
        if not self.database_provider:
            raise InvestigationError("No database provider configured")
        
        return self.database_provider.list_investigations(
            limit=limit,
            filters=filters
        )
    
    def load_investigation(self, investigation_id: str) -> Optional[InvestigationResult]:
        """
        Load a previous investigation from database.
        
        Args:
            investigation_id: Investigation ID to load
            
        Returns:
            InvestigationResult if found, None otherwise
        """
        if not self.database_provider:
            raise InvestigationError("No database provider configured")
        
        data = self.database_provider.load_investigation(investigation_id)
        
        if data:
            return InvestigationResult.from_dict(data)
        
        return None
    
    def get_investigation_stats(self) -> Dict[str, Any]:
        """Get investigation statistics from database."""
        if not self.database_provider:
            raise InvestigationError("No database provider configured")
        
        return self.database_provider.get_investigation_stats()
    
    def health_check(self) -> Dict[str, Any]:
        """Perform comprehensive health check."""
        health = {
            "overall_status": "healthy",
            "components": {}
        }
        
        # Check LLM provider
        llm_health = self.llm_provider.health_check()
        health["components"]["llm_provider"] = llm_health
        
        # Check embedding provider
        embedding_health = self.embedding_provider.health_check()
        health["components"]["embedding_provider"] = embedding_health
        
        # Check database provider
        if self.database_provider:
            db_health = self.database_provider.health_check()
            health["components"]["database_provider"] = db_health
        else:
            health["components"]["database_provider"] = {
                "status": "not_configured",
                "reason": "No database provider configured"
            }
        
        # Check overall status
        component_statuses = [
            comp["status"] for comp in health["components"].values()
        ]
        
        if any(status == "unhealthy" for status in component_statuses):
            health["overall_status"] = "unhealthy"
        elif any(status == "not_configured" for status in component_statuses):
            health["overall_status"] = "degraded"
        
        return health