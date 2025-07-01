#!/usr/bin/env python3
"""
Perquire Benchmark Script
Validates investigation results against known ground truth scenarios.
"""

import asyncio
import json
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from pathlib import Path

from src.perquire.core.investigator import PerquireInvestigator
from src.perquire.embeddings.utils import calculate_similarity


@dataclass
class BenchmarkScenario:
    """Defines a benchmark test scenario with ground truth."""
    name: str
    user_embedding: List[float]
    ground_truth_embedding: List[float]  # Known optimal target
    expected_similarity_threshold: float  # Minimum expected similarity
    max_iterations: int = 10
    description: str = ""


class PerquireBenchmark:
    """Benchmark suite for validating Perquire investigation results."""
    
    def __init__(self):
        self.scenarios: List[BenchmarkScenario] = []
        self.results: List[Dict[str, Any]] = []
        
    def add_scenario(self, scenario: BenchmarkScenario):
        """Add a benchmark scenario."""
        self.scenarios.append(scenario)
        
    def create_default_scenarios(self):
        """Create default benchmark scenarios with synthetic data."""
        # Scenario 1: Similar embeddings (should converge quickly)
        self.add_scenario(BenchmarkScenario(
            name="High Similarity Convergence",
            user_embedding=[0.8, 0.6, 0.4, 0.2] + [0.0] * 1532,  # Padded to 1536
            ground_truth_embedding=[0.9, 0.7, 0.5, 0.3] + [0.1] * 1532,
            expected_similarity_threshold=0.85,
            max_iterations=5,
            description="Test convergence when starting point is already similar to target"
        ))
        
        # Scenario 2: Distant embeddings (should improve significantly)
        self.add_scenario(BenchmarkScenario(
            name="Distant Embedding Improvement",
            user_embedding=[1.0, 0.0, -1.0, 0.5] + [0.0] * 1532,
            ground_truth_embedding=[-0.5, 1.0, 0.2, -0.8] + [0.5] * 1532,
            expected_similarity_threshold=0.75,
            max_iterations=10,
            description="Test improvement when starting from distant embeddings"
        ))
        
        # Scenario 3: Moderate case
        self.add_scenario(BenchmarkScenario(
            name="Moderate Similarity Case",
            user_embedding=[0.5, -0.3, 0.8, -0.2] + [0.2] * 1532,
            ground_truth_embedding=[0.3, -0.1, 0.6, 0.1] + [0.3] * 1532,
            expected_similarity_threshold=0.80,
            max_iterations=8,
            description="Test typical case with moderate starting similarity"
        ))
    
    async def run_scenario(self, scenario: BenchmarkScenario) -> Dict[str, Any]:
        """Run a single benchmark scenario."""
        print(f"\n🔍 Running scenario: {scenario.name}")
        print(f"   Description: {scenario.description}")
        
        start_time = time.time()
        
        # Calculate initial similarity
        initial_similarity = calculate_similarity(
            scenario.user_embedding, 
            scenario.ground_truth_embedding
        )
        
        # Create investigator
        investigator = PerquireInvestigator()
        
        # Run investigation
        try:
            result = await investigator.investigate(
                user_embedding=scenario.user_embedding,
                max_iterations=scenario.max_iterations,
                similarity_threshold=0.95  # High threshold to test improvement
            )
            
            # Calculate final similarity to ground truth
            final_similarity = calculate_similarity(
                result.final_embedding,
                scenario.ground_truth_embedding
            )
            
            # Calculate improvement
            improvement = final_similarity - initial_similarity
            
            # Determine if scenario passed
            passed = final_similarity >= scenario.expected_similarity_threshold
            
            scenario_result = {
                "scenario_name": scenario.name,
                "description": scenario.description,
                "initial_similarity": initial_similarity,
                "final_similarity": final_similarity,
                "improvement": improvement,
                "expected_threshold": scenario.expected_similarity_threshold,
                "passed": passed,
                "iterations_used": result.iteration_count,
                "max_iterations": scenario.max_iterations,
                "converged": result.converged,
                "duration_seconds": time.time() - start_time,
                "perquire_final_similarity": result.final_similarity,
                "questions_asked": len(result.question_history)
            }
            
            # Print results
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"   {status}")
            print(f"   Initial similarity to ground truth: {initial_similarity:.3f}")
            print(f"   Final similarity to ground truth: {final_similarity:.3f}")
            print(f"   Improvement: {improvement:+.3f}")
            print(f"   Expected threshold: {scenario.expected_similarity_threshold:.3f}")
            print(f"   Iterations: {result.iteration_count}/{scenario.max_iterations}")
            print(f"   Converged: {result.converged}")
            print(f"   Duration: {scenario_result['duration_seconds']:.2f}s")
            
            return scenario_result
            
        except Exception as e:
            error_result = {
                "scenario_name": scenario.name,
                "description": scenario.description,
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
    benchmark = PerquireBenchmark()
    
    # Run all scenarios
    summary = await benchmark.run_all_scenarios()
    
    # Save results
    benchmark.save_results()
    
    # Return summary for potential programmatic use
    return summary


if __name__ == "__main__":
    asyncio.run(main())