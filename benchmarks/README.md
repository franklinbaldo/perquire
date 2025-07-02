# Benchmarks

Performance benchmarks and evaluation scripts for Perquire.

## Files

- **`benchmark.py`** - Comprehensive benchmark suite
- **`simple_benchmark.py`** - Basic performance testing
- **`simple_benchmark_results.json`** - Latest benchmark results

## Running Benchmarks

```bash
cd benchmarks

# Simple benchmark
uv run python simple_benchmark.py

# Full benchmark suite
uv run python benchmark.py --provider gemini --iterations 10
```

## Current Performance

Based on live testing:
- **Investigation Speed**: 6.76 seconds average
- **Convergence**: 5 iterations typical
- **Memory Usage**: ~100MB with knowledge base
- **API Efficiency**: Batch processing supported

## Benchmark Categories

1. **Speed Tests**: Investigation time vs embedding complexity
2. **Accuracy Tests**: LLM evaluation scores vs ground truth
3. **Convergence Tests**: Iterations needed vs similarity threshold
4. **Memory Tests**: Resource usage vs knowledge base size
5. **Scalability Tests**: Batch processing performance