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

# For hashing cache keys
import hashlib
import json # For hashing complex dicts/lists
from typing import Union, List, Dict, Any # Ensure Union, List, Dict, Any are available if not already


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

    def _generate_hash_for_cache(self, data: Union[str, Dict, List, np.ndarray]) -> str:
        """Generate SHA256 hash for cache keys."""
        if isinstance(data, np.ndarray):
            # Using tobytes() for a consistent representation of the array's data
            return hashlib.sha256(data.tobytes()).hexdigest()
        if isinstance(data, str):
            return hashlib.sha256(data.encode('utf-8')).hexdigest()
        # For dicts/lists, ensure consistent serialization (sort keys, no extra whitespace)
        return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    
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
        """Generate a question for the current investigation phase, with DB caching for LLM calls."""

        # Prepare data for hashing LLM input
        llm_input_data_for_cache = {
            "type": "question_generation",
            "current_description": current_description,
            "target_similarity": round(current_similarity, 5), # Round float for stable hash
            "phase": phase.value,
            "previous_questions_summary": previous_questions[-5:],
        }
        input_hash = self._generate_hash_for_cache(llm_input_data_for_cache)

        # 1. Check LLM question generation cache
        if self.database_provider:
            try:
                cached_questions = self.database_provider.get_cached_llm_question_gen(input_hash)
                if cached_questions and isinstance(cached_questions, list) and len(cached_questions) > 0:
                    for q_from_cache in cached_questions:
                        if q_from_cache not in previous_questions:
                            logger.debug(f"LLM question_gen cache hit for input hash {input_hash}, using cached question: {q_from_cache[:30]}...")
                            return q_from_cache
                    logger.debug(f"LLM question_gen cache hit for input hash {input_hash}, but all were repeats.")
                elif cached_questions is not None:
                     logger.debug(f"LLM question_gen cache hit for input hash {input_hash}, but content was empty/invalid. Regenerating.")
            except Exception as db_err:
                logger.warning(f"Error checking LLM question_gen cache: {db_err}")

        # 2. If not in cache or all cached questions were repeats, generate new questions
        generated_question_list: Optional[List[str]] = None
        final_question_to_use: Optional[str] = None

        try:
            if hasattr(self.llm_provider, 'generate_questions'):
                try:
                    generated_question_list = self.llm_provider.generate_questions(
                        current_description=current_description,
                        target_similarity=current_similarity,
                        phase=phase.value,
                        previous_questions=previous_questions[-10:]
                    )
                    
                    if generated_question_list:
                        if self.database_provider:
                            try: # Cache up to 5 generated questions
                                self.database_provider.set_cached_llm_question_gen(input_hash, generated_question_list[:5])
                            except Exception as db_err:
                                logger.warning(f"Failed to cache LLM generated questions: {db_err}")

                        for q_new in generated_question_list:
                            if q_new not in previous_questions:
                                final_question_to_use = q_new
                                break
                        if not final_question_to_use:
                            logger.debug("LLM generated questions, but all were repeats of previous_questions.")
                    else:
                        logger.debug("LLM provider generate_questions returned empty list or None.")
                except Exception as e_llm_gen:
                    logger.warning(f"LLM question generation failed: {e_llm_gen}. Falling back to strategy if no question yet.")
            
            if not final_question_to_use: # Fallback to strategy
                logger.debug("Falling back to strategy-based question generation.")
                strategy_context = {
                    "current_description": current_description,
                    "current_similarity": current_similarity,
                    "iteration": len(previous_questions) + 1,
                    "phase": phase.value
                }
                strat_question = strategy.generate_question(
                    phase=phase,
                    context=strategy_context,
                    used_questions=previous_questions
                )
                final_question_to_use = strat_question

            if not final_question_to_use or not final_question_to_use.strip():
                logger.error("Failed to generate any valid question. Using a generic fallback.")
                return f"What are the general properties or category of the item represented by the embedding? (iteration {len(previous_questions)+1})"

            return final_question_to_use.strip()

        except Exception as e: # Catch-all for unexpected errors during generation logic
            logger.error(f"Critical error in _generate_question: {e}")
            # Provide a very generic fallback to prevent crash
            return f"Describe any known aspect of the embedding. (iteration {len(previous_questions)+1}, error recovery)"

    def _calculate_question_similarity(self, question: str, target_embedding: np.ndarray) -> float:
        """
        Calculate similarity between question and target embedding, using DB cache for
        both question embeddings and similarity scores.
        """
        question_text_hash = self._generate_hash_for_cache(question)
        # Assuming target_embedding is a numpy array. Hash its bytes.
        target_embedding_hash = self._generate_hash_for_cache(target_embedding)

        # 1. Check for cached similarity score
        if self.database_provider:
            try:
                cached_score = self.database_provider.get_cached_similarity(question_text_hash, target_embedding_hash)
                if cached_score is not None:
                    logger.debug(f"Similarity cache hit for question '{question[:30]}...'")
                    return float(cached_score)
            except Exception as db_err: # Catch specific DB errors if possible
                logger.warning(f"Error checking similarity cache: {db_err}")

        # 2. Get or generate question embedding (with DB caching)
        question_embedding: Optional[np.ndarray] = None

        if self.database_provider:
            try:
                cached_q_emb = self.database_provider.get_cached_embedding(question_text_hash)
                if cached_q_emb is not None:
                    question_embedding = cached_q_emb
                    logger.debug(f"Question embedding DB cache hit for '{question[:30]}...'")
            except Exception as db_err:
                logger.warning(f"Error checking question embedding DB cache: {db_err}")

        if question_embedding is None: # Not in DB cache, try provider (which has its own LRU)
            try:
                emb_result = self.embedding_provider.embed_text(question) # This uses LRU
                question_embedding = emb_result.embedding

                if self.database_provider and question_embedding is not None: # Check if embedding was successful
                    try:
                        # Calculate norm for storage, ensure it's float
                        emb_norm = float(np.linalg.norm(question_embedding))
                        if not np.isfinite(emb_norm): emb_norm = 0.0 # Handle potential NaN/inf from norm

                        self.database_provider.set_cached_embedding(
                            text_content=question,
                            embedding=question_embedding,
                            model_name=emb_result.model or self.embedding_provider.get_model_info().get('model', 'unknown'),
                            provider_name=self.embedding_provider.get_model_info().get('provider', 'unknown'),
                            dimensions=emb_result.dimensions or len(question_embedding),
                            embedding_norm=emb_norm,
                            metadata={'source': 'investigator_cache_fill', 'original_text_hash': question_text_hash}
                        )
                        logger.debug(f"Question embedding for '{question[:30]}...' cached to DB.")
                    except Exception as db_err:
                        logger.warning(f"Failed to cache question embedding to DB: {db_err}")
            except Exception as e:
                logger.error(f"Failed to generate embedding for question '{question}': {e}")
                raise EmbeddingError(f"Failed to generate embedding for question similarity calculation: {str(e)}")

        if question_embedding is None:
            raise EmbeddingError(f"Could not obtain embedding for question: {question}")

        # 3. Calculate cosine similarity
        try:
            similarity = cosine_similarity(target_embedding, question_embedding)
            similarity = max(0.0, min(1.0, similarity)) # Ensure valid range
        except Exception as e:
            logger.error(f"Error computing cosine_similarity for '{question}': {e}")
            raise EmbeddingError(f"Failed to compute cosine similarity: {str(e)}")

        # 4. Cache the newly calculated similarity score in DB
        if self.database_provider:
            try:
                self.database_provider.set_cached_similarity(question_text_hash, target_embedding_hash, similarity)
                logger.debug(f"Similarity score for '{question[:30]}...' cached to DB.")
            except Exception as db_err:
                logger.warning(f"Failed to cache similarity score to DB: {db_err}")

        return similarity
    
    def _synthesize_description(
        self,
        questions_and_scores: List[Dict[str, Any]],
        final_similarity: float
    ) -> str:
        """Synthesize final description from investigation results, with DB caching for LLM calls."""

        # Prepare data for hashing LLM input
        stable_q_and_s = sorted(questions_and_scores, key=lambda x: x.get("similarity", 0.0), reverse=True)[:10]
        stable_q_and_s_for_hash = sorted(stable_q_and_s, key=lambda x: x.get("question", ""))

        llm_input_data_for_cache = {
            "type": "synthesis",
            "questions_and_scores_summary": [{"q": item["question"], "s": round(item["similarity"], 5)} for item in stable_q_and_s_for_hash],
            "final_similarity": round(final_similarity, 5)
        }
        input_hash = self._generate_hash_for_cache(llm_input_data_for_cache)

        # 1. Check LLM synthesis cache
        if self.database_provider:
            try:
                cached_description = self.database_provider.get_cached_llm_synthesis(input_hash)
                if cached_description is not None:
                    logger.debug(f"LLM synthesis cache hit for input hash {input_hash}.")
                    return cached_description
            except Exception as db_err:
                logger.warning(f"Error checking LLM synthesis cache: {db_err}")

        # 2. If not in cache, generate new description
        description_to_cache: Optional[str] = None
        try:
            if hasattr(self.llm_provider, 'synthesize_description'):
                try:
                    llm_description = self.llm_provider.synthesize_description(
                        questions_and_scores=questions_and_scores,
                        final_similarity=final_similarity
                    )
                    if llm_description and llm_description.strip():
                        description_to_cache = llm_description.strip()
                except Exception as e_llm_synth:
                    logger.warning(f"LLM description synthesis failed: {e_llm_synth}. Falling back.")
            
            if not description_to_cache: # Fallback
                if questions_and_scores:
                    sorted_by_sim = sorted(questions_and_scores, key=lambda x: x.get("similarity", 0.0), reverse=True)
                    best_questions = sorted_by_sim[:3]
                    if len(best_questions) == 1:
                        description_to_cache = best_questions[0]["question"]
                    elif best_questions:
                        descriptions_parts = [q["question"] for q in best_questions]
                        description_to_cache = f"Likely represents concepts such as: '{descriptions_parts[0]}'"
                        if len(descriptions_parts) > 1: description_to_cache += f" and '{descriptions_parts[1]}'."
                        else: description_to_cache += "."
                    else:
                        description_to_cache = "Unable to determine description (no data)."
                else:
                    description_to_cache = "Unable to determine description (no data)."

            if self.database_provider and description_to_cache:
                try:
                    self.database_provider.set_cached_llm_synthesis(input_hash, description_to_cache)
                except Exception as db_err:
                    logger.warning(f"Failed to cache synthesized description: {db_err}")
            
            return description_to_cache if description_to_cache else "Error in description synthesis process."

        except Exception as e:
            logger.error(f"Critical error in _synthesize_description: {e}")
            return "Error during description synthesis process."
    
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