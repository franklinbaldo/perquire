"""
Ensemble investigation using multiple models and strategies.
"""

import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import statistics

from .investigator import PerquireInvestigator
from .result import InvestigationResult
from .strategy import QuestioningStrategy
from ..llm.base import BaseLLMProvider
from ..embeddings.base import BaseEmbeddingProvider
from ..database.base import BaseDatabaseProvider
from ..exceptions import InvestigationError

logger = logging.getLogger(__name__)


class EnsembleInvestigator:
    """
    Ensemble investigation using multiple investigators with different configurations.
    
    This class runs multiple investigations in parallel or sequence and combines
    their results to produce more robust and accurate descriptions.
    """
    
    def __init__(
        self,
        investigators: Optional[List[PerquireInvestigator]] = None,
        llm_providers: Optional[List[Union[str, BaseLLMProvider]]] = None,
        embedding_providers: Optional[List[Union[str, BaseEmbeddingProvider]]] = None,
        strategies: Optional[List[QuestioningStrategy]] = None,
        database_provider: Optional[BaseDatabaseProvider] = None,
        voting_method: str = "weighted_similarity",
        min_agreement: float = 0.6
    ):
        """
        Initialize EnsembleInvestigator.
        
        Args:
            investigators: Pre-configured investigators to use
            llm_providers: List of LLM providers for creating investigators
            embedding_providers: List of embedding providers
            strategies: List of questioning strategies
            database_provider: Shared database provider
            voting_method: Method for combining results ("weighted_similarity", "majority", "average")
            min_agreement: Minimum agreement threshold for ensemble confidence
        """
        self.database_provider = database_provider
        self.voting_method = voting_method
        self.min_agreement = min_agreement
        
        # Initialize investigators
        if investigators:
            self.investigators = investigators
        else:
            self.investigators = self._create_investigators(
                llm_providers, embedding_providers, strategies
            )
        
        if not self.investigators:
            raise InvestigationError("No investigators available for ensemble")
        
        logger.info(f"Initialized EnsembleInvestigator with {len(self.investigators)} investigators")
    
    def _create_investigators(
        self,
        llm_providers: Optional[List[Union[str, BaseLLMProvider]]],
        embedding_providers: Optional[List[Union[str, BaseEmbeddingProvider]]],
        strategies: Optional[List[QuestioningStrategy]]
    ) -> List[PerquireInvestigator]:
        """Create investigators from provider lists."""
        investigators = []
        
        # Default configurations if none provided
        if not llm_providers:
            llm_providers = ["gemini"]  # Default to Gemini
        
        if not embedding_providers:
            embedding_providers = ["gemini"]  # Default to Gemini
        
        if not strategies:
            strategies = [
                QuestioningStrategy(name="default"),
                QuestioningStrategy(name="artistic", convergence_threshold=0.92),
                QuestioningStrategy(name="scientific", convergence_threshold=0.88)
            ]
        
        # Create investigators for each combination
        for i, (llm_prov, embed_prov, strategy) in enumerate(
            zip(llm_providers, embedding_providers, strategies)
        ):
            try:
                investigator = PerquireInvestigator(
                    llm_provider=llm_prov,
                    embedding_provider=embed_prov,
                    questioning_strategy=strategy,
                    database_provider=self.database_provider
                )
                investigators.append(investigator)
                
            except Exception as e:
                logger.warning(f"Failed to create investigator {i}: {str(e)}")
        
        return investigators
    
    def investigate(
        self,
        target_embedding: np.ndarray,
        parallel: bool = True,
        save_individual_results: bool = False,
        save_ensemble_result: bool = True,
        verbose: bool = False
    ) -> InvestigationResult:
        """
        Investigate embedding using ensemble of investigators.
        
        Args:
            target_embedding: The embedding to investigate
            parallel: Whether to run investigations in parallel
            save_individual_results: Whether to save individual results to database
            save_ensemble_result: Whether to save ensemble result to database
            verbose: Whether to print progress information
            
        Returns:
            Ensemble InvestigationResult
        """
        if verbose:
            print(f"🔍 Starting ensemble investigation with {len(self.investigators)} investigators")
        
        # Run individual investigations
        individual_results = []
        
        if parallel:
            individual_results = self._investigate_parallel(
                target_embedding, save_individual_results, verbose
            )
        else:
            individual_results = self._investigate_sequential(
                target_embedding, save_individual_results, verbose
            )
        
        if not individual_results:
            raise InvestigationError("No successful individual investigations")
        
        # Combine results
        ensemble_result = self._combine_results(individual_results, target_embedding)
        
        if verbose:
            print(f"✅ Ensemble investigation completed")
            print(f"📝 Ensemble description: {ensemble_result.description}")
            print(f"📊 Ensemble similarity: {ensemble_result.final_similarity:.3f}")
            print(f"🤝 Agreement score: {ensemble_result.metadata.get('agreement_score', 'N/A')}")
        
        # Save ensemble result
        if save_ensemble_result and self.database_provider:
            try:
                self.database_provider.save_investigation(ensemble_result.to_dict())
                if verbose:
                    print(f"💾 Saved ensemble result: {ensemble_result.investigation_id}")
            except Exception as e:
                logger.warning(f"Failed to save ensemble result: {str(e)}")
        
        return ensemble_result
    
    def _investigate_parallel(
        self,
        target_embedding: np.ndarray,
        save_results: bool,
        verbose: bool
    ) -> List[InvestigationResult]:
        """Run investigations in parallel."""
        results = []
        
        with ThreadPoolExecutor(max_workers=min(4, len(self.investigators))) as executor:
            # Submit all investigations
            future_to_investigator = {
                executor.submit(
                    investigator.investigate,
                    target_embedding,
                    save_to_database=save_results,
                    verbose=False  # Disable individual verbose to avoid output collision
                ): i for i, investigator in enumerate(self.investigators)
            }
            
            # Collect results
            for future in as_completed(future_to_investigator):
                investigator_idx = future_to_investigator[future]
                try:
                    result = future.result()
                    results.append(result)
                    
                    if verbose:
                        print(f"✅ Investigator {investigator_idx + 1} completed: {result.final_similarity:.3f}")
                
                except Exception as e:
                    logger.warning(f"Investigator {investigator_idx} failed: {str(e)}")
                    if verbose:
                        print(f"❌ Investigator {investigator_idx + 1} failed: {str(e)}")
        
        return results
    
    def _investigate_sequential(
        self,
        target_embedding: np.ndarray,
        save_results: bool,
        verbose: bool
    ) -> List[InvestigationResult]:
        """Run investigations sequentially."""
        results = []
        
        for i, investigator in enumerate(self.investigators):
            try:
                if verbose:
                    print(f"🔄 Running investigator {i + 1}/{len(self.investigators)}")
                
                result = investigator.investigate(
                    target_embedding,
                    save_to_database=save_results,
                    verbose=verbose
                )
                results.append(result)
                
                if verbose:
                    print(f"✅ Investigator {i + 1} completed: {result.final_similarity:.3f}")
            
            except Exception as e:
                logger.warning(f"Investigator {i} failed: {str(e)}")
                if verbose:
                    print(f"❌ Investigator {i + 1} failed: {str(e)}")
        
        return results
    
    def _combine_results(
        self,
        individual_results: List[InvestigationResult],
        target_embedding: np.ndarray
    ) -> InvestigationResult:
        """Combine individual investigation results into ensemble result."""
        if not individual_results:
            raise InvestigationError("No results to combine")
        
        # Create ensemble result
        ensemble_result = InvestigationResult(
            investigation_id=f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="",  # Will be determined by voting
            final_similarity=0.0,  # Will be calculated
            iterations=0,  # Will be averaged
            start_time=min(r.start_time for r in individual_results),
            end_time=max(r.end_time for r in individual_results if r.end_time),
            strategy_name="ensemble",
            metadata={
                "ensemble_method": self.voting_method,
                "individual_count": len(individual_results),
                "individual_results": [r.investigation_id for r in individual_results]
            }
        )
        
        # Combine based on voting method
        if self.voting_method == "weighted_similarity":
            ensemble_result = self._weighted_similarity_voting(individual_results, ensemble_result)
        elif self.voting_method == "majority":
            ensemble_result = self._majority_voting(individual_results, ensemble_result)
        elif self.voting_method == "average":
            ensemble_result = self._average_voting(individual_results, ensemble_result)
        else:
            raise InvestigationError(f"Unknown voting method: {self.voting_method}")
        
        # Calculate ensemble metrics
        similarities = [r.final_similarity for r in individual_results]
        iterations = [r.iterations for r in individual_results]
        
        ensemble_result.final_similarity = statistics.mean(similarities)
        ensemble_result.iterations = int(statistics.mean(iterations))
        
        # Calculate agreement score
        agreement_score = self._calculate_agreement(individual_results)
        ensemble_result.metadata["agreement_score"] = agreement_score
        ensemble_result.metadata["similarity_std"] = statistics.stdev(similarities) if len(similarities) > 1 else 0.0
        ensemble_result.metadata["similarity_range"] = max(similarities) - min(similarities)
        
        return ensemble_result
    
    def _weighted_similarity_voting(
        self,
        results: List[InvestigationResult],
        ensemble_result: InvestigationResult
    ) -> InvestigationResult:
        """Combine results using weighted voting based on similarity scores."""
        # Weight each result by its similarity score
        total_weight = sum(r.final_similarity for r in results)
        
        if total_weight == 0:
            # Fallback to equal weights
            weights = [1.0 / len(results)] * len(results)
        else:
            weights = [r.final_similarity / total_weight for r in results]
        
        # Select description from highest-weighted result
        best_idx = max(range(len(results)), key=lambda i: weights[i])
        ensemble_result.description = results[best_idx].description
        
        # Store weights in metadata
        ensemble_result.metadata["weights"] = weights
        ensemble_result.metadata["best_result_idx"] = best_idx
        
        return ensemble_result
    
    def _majority_voting(
        self,
        results: List[InvestigationResult],
        ensemble_result: InvestigationResult
    ) -> InvestigationResult:
        """Combine results using majority voting on description keywords."""
        # Extract keywords from descriptions
        all_keywords = []
        for result in results:
            # Simple keyword extraction (could be improved with NLP)
            words = result.description.lower().split()
            keywords = [w for w in words if len(w) > 3]  # Filter short words
            all_keywords.extend(keywords)
        
        # Count keyword frequency
        keyword_counts = {}
        for keyword in all_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        # Select most frequent keywords
        if keyword_counts:
            sorted_keywords = sorted(keyword_counts.items(), key=lambda x: x[1], reverse=True)
            top_keywords = [kw for kw, count in sorted_keywords[:5]]  # Top 5 keywords
            ensemble_result.description = f"Majority consensus: {' '.join(top_keywords)}"
        else:
            # Fallback to best similarity result
            best_result = max(results, key=lambda r: r.final_similarity)
            ensemble_result.description = best_result.description
        
        ensemble_result.metadata["keyword_counts"] = keyword_counts
        
        return ensemble_result
    
    def _average_voting(
        self,
        results: List[InvestigationResult],
        ensemble_result: InvestigationResult
    ) -> InvestigationResult:
        """Combine results using simple averaging (selects median description)."""
        # Sort by similarity and select median
        sorted_results = sorted(results, key=lambda r: r.final_similarity)
        median_idx = len(sorted_results) // 2
        median_result = sorted_results[median_idx]
        
        ensemble_result.description = median_result.description
        ensemble_result.metadata["median_result_idx"] = median_idx
        
        return ensemble_result
    
    def _calculate_agreement(self, results: List[InvestigationResult]) -> float:
        """Calculate agreement score between individual results."""
        if len(results) < 2:
            return 1.0
        
        # Calculate similarity variance as disagreement measure
        similarities = [r.final_similarity for r in results]
        similarity_variance = statistics.variance(similarities)
        
        # Convert variance to agreement score (0-1)
        # Lower variance = higher agreement
        max_possible_variance = 0.25  # Assumed maximum variance
        agreement = max(0.0, 1.0 - (similarity_variance / max_possible_variance))
        
        return min(1.0, agreement)
    
    def investigate_batch(
        self,
        embeddings: List[np.ndarray],
        parallel: bool = True,
        save_results: bool = True,
        verbose: bool = False
    ) -> List[InvestigationResult]:
        """
        Investigate multiple embeddings using ensemble approach.
        
        Args:
            embeddings: List of embeddings to investigate
            parallel: Whether to use parallel processing
            save_results: Whether to save results to database
            verbose: Whether to print progress information
            
        Returns:
            List of ensemble InvestigationResult objects
        """
        results = []
        
        for i, embedding in enumerate(embeddings):
            if verbose:
                print(f"\n🔍 Ensemble investigating embedding {i + 1}/{len(embeddings)}")
            
            try:
                result = self.investigate(
                    target_embedding=embedding,
                    parallel=parallel,
                    save_ensemble_result=save_results,
                    verbose=verbose
                )
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to ensemble investigate embedding {i}: {str(e)}")
                if verbose:
                    print(f"❌ Failed to investigate embedding {i}: {str(e)}")
        
        return results
    
    def get_ensemble_stats(self) -> Dict[str, Any]:
        """Get statistics about ensemble configuration."""
        return {
            "num_investigators": len(self.investigators),
            "voting_method": self.voting_method,
            "min_agreement": self.min_agreement,
            "investigator_configs": [
                {
                    "llm_provider": inv.llm_provider.get_model_info().get("provider", "unknown"),
                    "embedding_provider": inv.embedding_provider.get_model_info().get("provider", "unknown"),
                    "strategy": inv.questioning_strategy.name
                }
                for inv in self.investigators
            ]
        }
    
    def health_check(self) -> Dict[str, Any]:
        """Perform health check on all investigators."""
        health = {
            "overall_status": "healthy",
            "investigators": []
        }
        
        unhealthy_count = 0
        
        for i, investigator in enumerate(self.investigators):
            inv_health = investigator.health_check()
            health["investigators"].append({
                "investigator_id": i,
                "status": inv_health["overall_status"],
                "details": inv_health
            })
            
            if inv_health["overall_status"] == "unhealthy":
                unhealthy_count += 1
        
        # Determine overall health
        if unhealthy_count == len(self.investigators):
            health["overall_status"] = "unhealthy"
        elif unhealthy_count > 0:
            health["overall_status"] = "degraded"
        
        health["healthy_count"] = len(self.investigators) - unhealthy_count
        health["total_count"] = len(self.investigators)
        
        return health