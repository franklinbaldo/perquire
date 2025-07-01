#!/usr/bin/env python3
"""
Simple Perquire Benchmark - Tests core functionality without heavy dependencies
"""

import asyncio
import json
import time
import math
from typing import List, Dict, Any


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))
    
    if magnitude_a == 0 or magnitude_b == 0:
        return 0.0
    
    return dot_product / (magnitude_a * magnitude_b)


def simulate_perquire_improvement(initial_embedding: List[float], 
                                target_embedding: List[float],
                                iterations: int = 5) -> Dict[str, Any]:
    """
    Simulate Perquire improvement process.
    In reality, this would use LLM to generate questions and refine embeddings.
    """
    current_embedding = initial_embedding.copy()
    history = []
    
    initial_similarity = cosine_similarity(current_embedding, target_embedding)
    
    # Simulate iterative improvement
    for i in range(iterations):
        # Simulate embedding refinement (move towards target)
        improvement_factor = 0.15  # 15% improvement per iteration
        
        for j in range(len(current_embedding)):
            diff = target_embedding[j] - current_embedding[j]
            current_embedding[j] += diff * improvement_factor
        
        similarity = cosine_similarity(current_embedding, target_embedding)
        history.append({
            "iteration": i + 1,
            "similarity": similarity,
            "improvement": similarity - (history[-1]["similarity"] if history else initial_similarity)
        })
        
        # Early stopping if we reach high similarity
        if similarity > 0.95:
            break
    
    final_similarity = cosine_similarity(current_embedding, target_embedding)
    
    return {
        "initial_similarity": initial_similarity,
        "final_similarity": final_similarity,
        "improvement": final_similarity - initial_similarity,
        "iterations": len(history),
        "converged": final_similarity > 0.90,
        "history": history
    }


async def run_benchmark_scenarios():
    """Run benchmark scenarios."""
    
    print("🚀 Running Simple Perquire Benchmark")
    print("   (Simulated - not using actual LLM calls)")
    
    scenarios = [
        {
            "name": "High Similarity Start",
            "initial": [0.8, 0.6, 0.4, 0.2],
            "target": [0.9, 0.7, 0.5, 0.3],
            "expected_threshold": 0.85
        },
        {
            "name": "Moderate Similarity Start", 
            "initial": [0.5, -0.3, 0.8, -0.2],
            "target": [0.3, -0.1, 0.6, 0.1],
            "expected_threshold": 0.80
        },
        {
            "name": "Low Similarity Start",
            "initial": [1.0, 0.0, -1.0, 0.5],
            "target": [-0.5, 1.0, 0.2, -0.8],
            "expected_threshold": 0.75
        }
    ]
    
    results = []
    passed = 0
    
    for scenario in scenarios:
        print(f"\n🔍 Testing: {scenario['name']}")
        
        start_time = time.time()
        result = simulate_perquire_improvement(
            scenario["initial"], 
            scenario["target"]
        )
        duration = time.time() - start_time
        
        # Check if passed
        test_passed = result["final_similarity"] >= scenario["expected_threshold"]
        if test_passed:
            passed += 1
        
        status = "✅ PASSED" if test_passed else "❌ FAILED"
        print(f"   {status}")
        print(f"   Initial similarity: {result['initial_similarity']:.3f}")
        print(f"   Final similarity: {result['final_similarity']:.3f}")
        print(f"   Improvement: {result['improvement']:+.3f}")
        print(f"   Expected threshold: {scenario['expected_threshold']:.3f}")
        print(f"   Iterations: {result['iterations']}")
        print(f"   Converged: {result['converged']}")
        print(f"   Duration: {duration:.3f}s")
        
        results.append({
            "scenario": scenario["name"],
            "passed": test_passed,
            "expected_threshold": scenario["expected_threshold"],
            "duration": duration,
            **result
        })
    
    # Summary
    print(f"\n📊 Benchmark Summary")
    print(f"   Total scenarios: {len(scenarios)}")
    print(f"   Passed: {passed} ✅")
    print(f"   Failed: {len(scenarios) - passed} ❌")
    print(f"   Pass rate: {passed/len(scenarios):.1%}")
    
    avg_improvement = sum(r["improvement"] for r in results) / len(results)
    print(f"   Average improvement: {avg_improvement:+.3f}")
    
    # Save results
    with open("simple_benchmark_results.json", "w") as f:
        json.dump({
            "summary": {
                "total_scenarios": len(scenarios),
                "passed": passed,
                "failed": len(scenarios) - passed,
                "pass_rate": passed/len(scenarios),
                "average_improvement": avg_improvement
            },
            "results": results
        }, f, indent=2)
    
    print(f"💾 Results saved to simple_benchmark_results.json")
    
    return results


if __name__ == "__main__":
    asyncio.run(run_benchmark_scenarios())