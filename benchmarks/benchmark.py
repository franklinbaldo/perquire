#!/usr/bin/env python3
"""
Perquire Benchmark Script
Validates investigation results against known ground truth scenarios.
"""

import argparse
import asyncio
import json
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

# Provider imports
from src.perquire.llm.base import provider_registry, BaseLLMProvider
from src.perquire.embeddings.base import embedding_registry, BaseEmbeddingProvider

# Import all specific provider modules to ensure they register themselves
from src.perquire.llm import anthropic_provider, gemini_provider, ollama_provider, openai_provider
from src.perquire.embeddings import gemini_embeddings, openai_embeddings

from src.perquire.core.investigator import PerquireInvestigator
from src.perquire.embeddings.utils import calculate_similarity


@dataclass
class BenchmarkScenario:
    """Defines a benchmark test scenario with ground truth."""
    name: str
    user_text: str  # Text representing the user's initial query/concept
    ground_truth_text: str  # Text representing the ideal target concept
    expected_similarity_threshold: float  # Minimum expected similarity between final embedding and ground_truth_text embedding
    max_iterations: int = 10
    description: str = ""


class PerquireBenchmark:
    """Benchmark suite for validating Perquire investigation results."""
    
    def __init__(self,
                 llm_provider: BaseLLMProvider,
                 embedding_provider: BaseEmbeddingProvider,
                 cli_max_iterations: int):
        self.llm_provider = llm_provider
        self.embedding_provider = embedding_provider
        self.cli_max_iterations = cli_max_iterations
        self.scenarios: List[BenchmarkScenario] = []
        self.results: List[Dict[str, Any]] = []
        
    def add_scenario(self, scenario: BenchmarkScenario):
        """Add a benchmark scenario."""
        self.scenarios.append(scenario)
        
    def create_default_scenarios(self):
        """Create default benchmark scenarios with text prompts."""
        # Scenario 1: High similarity texts
        self.add_scenario(BenchmarkScenario(
            name="High Similarity Texts",
            user_text="A very fast red car",
            ground_truth_text="A quick crimson automobile",
            expected_similarity_threshold=0.7, # Indicative, may need tuning per model
            max_iterations=5,
            description="Test with texts that are semantically very close."
        ))
        
        # Scenario 2: Moderate similarity texts
        self.add_scenario(BenchmarkScenario(
            name="Moderate Similarity Texts",
            user_text="A happy dog playing in the park",
            ground_truth_text="A joyful canine frolicking outdoors",
            expected_similarity_threshold=0.65, # Indicative
            max_iterations=10,
            description="Test with texts that are related but less directly synonymous."
        ))
        
        # Scenario 3: Contextual understanding / Factual
        self.add_scenario(BenchmarkScenario(
            name="Contextual Understanding - Factual",
            user_text="What is the tallest mountain in the world?",
            ground_truth_text="Mount Everest is Earth's highest mountain above sea level, located in the Mahalangur Himal sub-range of the Himalayas.",
            expected_similarity_threshold=0.5, # Indicative, as this is more about information content
            max_iterations=10, # May need more iterations to refine towards factual answer
            description="Test ability to converge towards a factual concept."
        ))

    async def run_scenario(self, scenario: BenchmarkScenario) -> Dict[str, Any]:
        """Run a single benchmark scenario."""
        print(f"\n🔍 Running scenario: {scenario.name}")
        print(f"   Description: {scenario.description}")
        
        start_time = time.time()

        try:
            # Generate embeddings using the selected provider
            user_embedding_result = self.embedding_provider.embed_text(scenario.user_text)
            user_embedding = user_embedding_result.embedding

            ground_truth_embedding_result = self.embedding_provider.embed_text(scenario.ground_truth_text)
            ground_truth_embedding = ground_truth_embedding_result.embedding

            # Calculate initial similarity
            # Note: calculate_similarity is imported from src.perquire.embeddings.utils and expects np.ndarray
            initial_similarity = calculate_similarity(user_embedding, ground_truth_embedding)

            # Create investigator with the selected providers
            investigator = PerquireInvestigator(
                llm_provider=self.llm_provider,
                embedding_provider=self.embedding_provider
            )
            
            # Determine max_iterations: CLI override or scenario default
            # The PerquireInvestigator's investigate method itself has a max_iterations parameter.
            # The scenario.max_iterations from BenchmarkScenario is more of a descriptor for the scenario's intent.
            # We will use self.cli_max_iterations for the actual investigation run.
            # The scenario.max_iterations will be reported in results for context.

            effective_max_iterations = self.cli_max_iterations

            # Run investigation
            # The 'user_embedding' for the investigator is the 'target_embedding' it tries to understand.
            # The investigator's own 'similarity_threshold' is for its internal convergence logic.
            # We'll use a reasonably high one to ensure it tries to improve.
            investigation_result = await asyncio.to_thread(
                investigator.investigate,
                target_embedding=user_embedding,
                max_iterations=effective_max_iterations, # Use CLI provided iterations
                # similarity_threshold=0.95 # This is an internal Perquire threshold, not the benchmark's pass/fail one.
                                           # Let's rely on max_iterations for now, or Perquire's default.
                                           # Or, make it configurable if needed. For now, remove to use default.
            )
            
            # Embed the final synthesized description from the investigator
            if investigation_result.description:
                final_description_embedding_result = self.embedding_provider.embed_text(investigation_result.description)
                final_description_embedding = final_description_embedding_result.embedding

                # Calculate final similarity: compare investigator's final description (re-embedded) to the ground truth text (embedded)
                final_similarity_to_ground_truth = calculate_similarity(
                    final_description_embedding,
                    ground_truth_embedding
                )
            else:
                print(f"   ⚠️ Warning: Investigation for '{scenario.name}' produced an empty description.")
                final_similarity_to_ground_truth = 0.0

            # Calculate improvement based on this final_similarity_to_ground_truth
            improvement = final_similarity_to_ground_truth - initial_similarity
            
            # Determine if scenario passed
            passed = final_similarity_to_ground_truth >= scenario.expected_similarity_threshold
            
            scenario_result = {
                "scenario_name": scenario.name,
                "benchmark_scenario_description": scenario.description, # Original scenario description
                "description_generated_by_investigator": investigation_result.description, # Perquire's output
                "initial_similarity_to_ground_truth": float(initial_similarity) if initial_similarity is not None else 0.0,
                "final_similarity_to_ground_truth": float(final_similarity_to_ground_truth),
                "improvement": float(improvement),
                "expected_threshold": scenario.expected_similarity_threshold,
                "passed": passed,
                "iterations_used_by_investigator": investigation_result.iterations,
                "scenario_max_iterations_setting": scenario.max_iterations,
                "investigator_max_iterations_used": effective_max_iterations,
                "converged_by_investigator": investigation_result.convergence_achieved, # Use .convergence_achieved
                "convergence_reason": investigation_result.convergence_reason, # Directly use the string value
                "duration_seconds": time.time() - start_time,
                "perquire_internal_final_similarity": investigation_result.final_similarity, # Investigator's internal similarity (best question to input)
                "questions_asked_count": len(investigation_result.question_history), # Use .question_history
                "llm_provider": self.llm_provider.get_model_info().get('provider_name', str(self.llm_provider.__class__.__name__)),
                "embedding_provider": self.embedding_provider.get_model_info().get('provider_name', str(self.embedding_provider.__class__.__name__)),
                "llm_model": investigation_result.model_config.get('llm_provider', {}).get('model_info', {}).get('model'),
                "embedding_model": investigation_result.model_config.get('embedding_provider', {}).get('model_info', {}).get('model'),
            }
            
            # Print results
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {status}")
            print(f"   Initial similarity (User Text Emb vs Ground Truth Text Emb): {scenario_result['initial_similarity_to_ground_truth']:.3f}")
            print(f"   Final similarity (Generated Description Emb vs Ground Truth Text Emb): {scenario_result['final_similarity_to_ground_truth']:.3f}")
            print(f"   Improvement: {scenario_result['improvement']:+.3f}")
            print(f"   Expected threshold: {scenario.expected_similarity_threshold:.3f}")
            print(f"   Iterations used by investigator: {scenario_result['iterations_used_by_investigator']}/{effective_max_iterations}")
            print(f"   Converged by investigator: {scenario_result['converged_by_investigator']} (Reason: {scenario_result['convergence_reason']})")
            print(f"   Duration: {scenario_result['duration_seconds']:.2f}s")
            
            return scenario_result
            
        except Exception as e:
            error_result = {
                "scenario_name": scenario.name,
                "benchmark_scenario_description": scenario.description,
                "error": str(e),
                "passed": False,
                "duration_seconds": time.time() - start_time
            }
            print(f"   ❌ ERROR: {e}")
            return error_result
    
    async def run_all_scenarios(self) -> Dict[str, Any]:
        """Run all benchmark scenarios."""
        if not self.scenarios:
            self.create_default_scenarios()
        
        print("🚀 Starting Perquire Benchmark Suite")
        print(f"   Running {len(self.scenarios)} scenarios...")
        
        overall_start = time.time()
        self.results = []
        
        for scenario in self.scenarios:
            result = await self.run_scenario(scenario)
            self.results.append(result)
        
        # Calculate summary statistics
        passed_scenarios = [r for r in self.results if r.get("passed", False)]
        failed_scenarios = [r for r in self.results if not r.get("passed", False)]
        
        if self.results:
            avg_improvement = sum(r.get("improvement", 0) for r in self.results) / len(self.results)
            avg_duration = sum(r.get("duration_seconds", 0) for r in self.results) / len(self.results)
        else:
            avg_improvement = 0
            avg_duration = 0
        
        summary = {
            "total_scenarios": len(self.scenarios),
            "passed_scenarios": len(passed_scenarios),
            "failed_scenarios": len(failed_scenarios),
            "pass_rate": len(passed_scenarios) / len(self.scenarios) if self.scenarios else 0,
            "average_improvement": avg_improvement,
            "average_duration": avg_duration,
            "total_duration": time.time() - overall_start,
            "results": self.results
        }
        
        # Print summary
        print(f"\n📊 Benchmark Summary")
        print(f"   Total scenarios: {summary['total_scenarios']}")
        print(f"   Passed: {summary['passed_scenarios']} ✅")
        print(f"   Failed: {summary['failed_scenarios']} ❌")
        print(f"   Pass rate: {summary['pass_rate']:.1%}")
        print(f"   Average improvement: {summary['average_improvement']:+.3f}")
        print(f"   Average duration: {summary['average_duration']:.2f}s")
        print(f"   Total duration: {summary['total_duration']:.2f}s")
        
        return summary
    
    def save_results(self, filepath: str = "benchmark_results.json"):
        """Save benchmark results to file."""
        if not self.results:
            print("No results to save.")
            return
        
        summary = {
            "total_scenarios": len(self.scenarios),
            "passed_scenarios": len([r for r in self.results if r.get("passed", False)]),
            "failed_scenarios": len([r for r in self.results if not r.get("passed", False)]),
            "results": self.results
        }
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"💾 Results saved to {filepath}")


async def main():
    """Run the benchmark suite."""
    parser = argparse.ArgumentParser(description="Perquire Benchmark Suite")
    parser.add_argument(
        "--llm_provider",
        type=str,
        default="gemini",
        choices=["anthropic", "gemini", "ollama", "openai"],
        help="LLM provider to use for the benchmark."
    )
    parser.add_argument(
        "--embedding_provider",
        type=str,
        default="gemini",
        choices=["gemini", "openai"],
        help="Embedding provider to use for the benchmark."
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Max iterations for the investigator per scenario."
    )
    args = parser.parse_args()

    # Get provider instances
    try:
        llm_provider_instance = provider_registry.get_provider(args.llm_provider)
        embedding_provider_instance = embedding_registry.get_provider(args.embedding_provider)
    except Exception as e:
        print(f"Error initializing providers: {e}")
        return

    benchmark = PerquireBenchmark(
        llm_provider=llm_provider_instance,
        embedding_provider=embedding_provider_instance,
        cli_max_iterations=args.iterations
    )
    
    summary = await benchmark.run_all_scenarios()
    
    # Save results
    benchmark.save_results()
    
    # Return summary for potential programmatic use
    return summary


if __name__ == "__main__":
    asyncio.run(main())