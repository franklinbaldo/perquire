# Perquire

_From Latin "perquirere" - to investigate thoroughly, to question deeply_

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Status](https://img.shields.io/badge/status-experimental-orange.svg)

**Perquire** is a revolutionary AI system that reverses the traditional embedding search process. Instead of finding embeddings that match a known query, Perquire investigates mysterious embeddings through systematic questioning, gradually uncovering what they represent.

## The Core Idea

Think of Perquire as a digital detective. Imagine you found a footprint in the sand but don't know who made it. Traditional systems would require you to already know whose footprint you're looking for. Perquire, however, asks strategic questions about the footprint itself - its size, depth, pattern - until it can tell you not just who made it, but their story.

In technical terms, Perquire receives an embedding vector of unknown origin and uses a Large Language Model to generate hypotheses through iterative questioning. Each question is designed to move closer to the semantic space represented by the target embedding, using cosine similarity as a guide. When the system reaches statistical convergence, it synthesizes its discoveries into a human-readable description.

## Why Perquire Matters

Traditional embedding systems work like asking "do you have any books about dogs?" in a library. Perquire works like being handed a mysterious book and figuring out it's about "the loyalty and courage of rescue dogs in wartime" through careful investigation.

This approach opens fascinating possibilities for content discovery, sentiment analysis, and understanding the latent representations that AI systems create internally. Perquire can help us decode the "thoughts" that exist within high-dimensional vector spaces.

## Quick Start

Install Perquire and watch it decode a mysterious embedding in just a few lines of code:

```python
from perquire import PerquireInvestigator
from sentence_transformers import SentenceTransformer

# Initialize the system
investigator = PerquireInvestigator(
    embedding_model=SentenceTransformer('all-MiniLM-L6-v2'),
    llm_provider='openai',  # or 'anthropic', 'cohere'
)

# Load a mysterious embedding (this could come from anywhere)
mysterious_text = "The melancholic beauty of abandoned places"
target_embedding = investigator.embedding_model.encode(mysterious_text)

# Let Perquire investigate
result = investigator.investigate(target_embedding)

print(f"Discovery: {result.description}")
print(f"Confidence: {result.final_similarity:.1%}")
print(f"Questions asked: {result.iterations}")
```

## Installation

Perquire requires Python 3.8 or higher and works with various LLM providers and embedding models.

```bash
# Install from PyPI
pip install perquire

# Or install from source
git clone https://github.com/franklinbaldo/perquire.git
cd perquire
pip install -e .
```

Configure your preferred LLM provider by setting environment variables:

```bash
# For OpenAI
export OPENAI_API_KEY="your-key-here"

# For Anthropic
export ANTHROPIC_API_KEY="your-key-here"

# For local models
export PERQUIRE_LOCAL_MODEL_PATH="/path/to/your/model"
```

## How It Works

Understanding Perquire requires grasping three key concepts that work together like instruments in an orchestra.

### The Investigation Process

Perquire operates through adaptive questioning phases, much like how a skilled interviewer adjusts their approach based on the responses they receive.

**Exploration Phase**: The system begins with broad, categorical questions designed to map the general semantic territory. Questions like "Does this relate to human emotions?" or "Is this about concrete objects or abstract concepts?" help establish the fundamental nature of the target embedding.

**Refinement Phase**: As similarity scores improve, Perquire narrows its focus. If the exploration revealed emotional content, refinement questions might explore "Are these positive or negative emotions?" or "Do they relate to personal relationships or broader social dynamics?"

**Convergence Phase**: In the final phase, Perquire asks highly specific questions to capture nuanced details. The system recognizes when additional questions yield diminishing returns and prepares to synthesize its findings.

### Semantic Similarity Guidance

The magic happens through cosine similarity calculations between the target embedding and the embeddings of Perquire's questions and hypotheses. Think of this as a "hot and cold" game played in multidimensional space.

When Perquire asks "Is this about sadness?" and generates an embedding for that question, the cosine similarity with the target embedding acts like a temperature reading. High similarity means "warmer" - the question is pointing in the right semantic direction. Low similarity means "colder" - time to explore different conceptual territories.

### Convergence Detection

Perquire doesn't just stop arbitrarily. The system employs statistical methods to detect when it has extracted maximum information from the embedding. It monitors the rate of improvement in similarity scores and recognizes when additional questions are unlikely to yield significant gains.

This convergence detection prevents both premature stopping (missing important details) and infinite loops (asking questions forever without meaningful progress).

## Advanced Usage

### Custom Questioning Strategies

You can customize how Perquire approaches different types of embeddings by providing strategy templates:

```python
from perquire import PerquireInvestigator, QuestioningStrategy

# Define a strategy for investigating artistic content
artistic_strategy = QuestioningStrategy(
    exploration_templates=[
        "Does this relate to visual, auditory, or literary art?",
        "Is this about the creation process or the appreciation of art?",
        "Does this involve specific artistic techniques or general aesthetic concepts?"
    ],
    refinement_patterns=[
        "emotional_impact",
        "historical_context",
        "technical_execution"
    ],
    convergence_threshold=0.92
)

investigator = PerquireInvestigator(
    embedding_model=your_model,
    questioning_strategy=artistic_strategy
)
```

### Ensemble Investigation

For complex embeddings, you can use multiple embedding models to get different perspectives on the same target:

```python
from perquire import EnsembleInvestigator

ensemble = EnsembleInvestigator([
    'sentence-transformers/all-MiniLM-L6-v2',
    'sentence-transformers/all-mpnet-base-v2',
    'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
])

# The ensemble investigates from multiple semantic perspectives
result = ensemble.investigate(target_embedding)
```

### Real-time Investigation Monitoring

Watch Perquire work in real-time by enabling verbose mode:

```python
investigator = PerquireInvestigator(verbose=True)

# This will print each question, similarity score, and reasoning
result = investigator.investigate(target_embedding)
```

## Use Cases and Applications

### Content Discovery in Vector Databases

Many organizations have large collections of embeddings representing documents, images, or user preferences, but limited metadata about what each embedding actually represents. Perquire can systematically investigate these embeddings to generate descriptive metadata, making vast vector databases searchable and understandable.

### Sentiment and Opinion Analysis

Traditional sentiment analysis tells you if text is positive or negative. Perquire can investigate embeddings to uncover nuanced emotional content like "nostalgic longing with hints of regret" or "confident optimism tempered by practical concerns."

### AI Model Interpretability

When working with large language models or recommendation systems, Perquire can help decode the internal representations these models create. By investigating the embeddings that represent user profiles, content items, or intermediate model states, researchers can gain insights into how AI systems "think" about complex concepts.

### Creative Writing and Brainstorming

Writers and creators can use Perquire to explore the semantic neighborhood around concepts they're working with. By investigating embeddings of their initial ideas, they might discover unexpected connections and themes to explore in their work.

## Configuration and Customization

Perquire offers extensive configuration options to adapt to different use cases and requirements.

### Embedding Model Selection

Choose embedding models based on your domain and requirements:

```python
# For general text (fast)
investigator = PerquireInvestigator(
    embedding_model='sentence-transformers/all-MiniLM-L6-v2'
)

# For high-quality semantic understanding
investigator = PerquireInvestigator(
    embedding_model='sentence-transformers/all-mpnet-base-v2'
)

# For multilingual content
investigator = PerquireInvestigator(
    embedding_model='sentence-transformers/paraphrase-multilingual-mpnet-base-v2'
)
```

### LLM Provider Configuration

Perquire supports multiple LLM providers with different strengths:

```python
# OpenAI GPT models (excellent general reasoning)
investigator = PerquireInvestigator(
    llm_provider='openai',
    llm_model='gpt-4',
    llm_config={'temperature': 0.7, 'max_tokens': 150}
)

# Anthropic Claude (strong analytical capabilities)
investigator = PerquireInvestigator(
    llm_provider='anthropic',
    llm_model='claude-3-sonnet',
    llm_config={'temperature': 0.5}
)

# Local models via Ollama (privacy-focused)
investigator = PerquireInvestigator(
    llm_provider='ollama',
    llm_model='llama2:13b'
)
```

### Investigation Parameters

Fine-tune the investigation process for your specific needs:

```python
investigator = PerquireInvestigator(
    # Convergence settings
    similarity_threshold=0.90,          # Stop when similarity reaches 90%
    max_iterations=25,                  # Maximum questions to ask
    convergence_window=3,               # Look at last 3 iterations for convergence
    min_improvement=0.001,              # Minimum improvement to continue

    # Question generation settings
    exploration_depth=5,                # Questions in exploration phase
    refinement_focus=0.7,               # Similarity threshold to enter refinement

    # Output settings
    verbose=True,                       # Show investigation progress
    save_history=True,                  # Keep detailed question history
    generate_confidence_score=True      # Include confidence metrics
)
```

## Performance Considerations

### Computational Efficiency

Perquire's performance depends on several factors. The choice of embedding model significantly impacts both speed and quality. Smaller models like MiniLM process quickly but may miss nuanced semantic relationships. Larger models like MPNet provide richer understanding but require more computational resources.

LLM calls represent the primary latency bottleneck. Each question requires a round-trip to your chosen language model. To optimize performance, consider using local models for development and testing, then switching to cloud-based models for production accuracy.

### Caching and Optimization

Implement embedding caching to avoid recomputing embeddings for repeated questions:

```python
from perquire import PerquireInvestigator
from functools import lru_cache

class CachedInvestigator(PerquireInvestigator):
    @lru_cache(maxsize=1000)
    def _embed_text(self, text):
        return super()._embed_text(text)

investigator = CachedInvestigator(embedding_model=your_model)
```

### Memory Management

For large-scale investigations, monitor memory usage, especially when using ensemble methods or keeping detailed investigation histories:

```python
# Configure memory-efficient investigation
investigator = PerquireInvestigator(
    save_history=False,          # Don't keep detailed question history
    batch_size=1,                # Process one investigation at a time
    embedding_cache_size=100     # Limit embedding cache size
)
```

## API Reference

### Core Classes

**PerquireInvestigator**: The main investigation engine that orchestrates the questioning process and manages similarity calculations.

**InvestigationResult**: Contains the investigation outcome, including the discovered description, confidence metrics, and optional investigation history.

**QuestioningStrategy**: Defines how Perquire approaches different types of content, including question templates and convergence criteria.

### Key Methods

```python
investigate(target_embedding, custom_strategy=None)
```

Investigates a target embedding and returns an InvestigationResult object containing the discovered description and metadata.

```python
investigate_batch(embeddings_list, parallel=True)
```

Efficiently investigates multiple embeddings, optionally using parallel processing for faster throughput.

```python
explain_investigation(result)
```

Provides detailed explanation of how Perquire reached its conclusions, useful for understanding and debugging the investigation process.

## Contributing

Perquire thrives on community contributions. Whether you're improving questioning strategies, adding support for new embedding models, or enhancing the core investigation algorithms, your contributions help make AI more interpretable and accessible.

### Development Setup

Setting up a development environment ensures you can test your changes effectively:

```bash
# Clone and set up development environment
git clone https://github.com/yourusername/perquire.git
cd perquire

# Create virtual environment
python -m venv perquire-dev
source perquire-dev/bin/activate  # On Windows: perquire-dev\Scripts\activate

# Install development dependencies
pip install -e ".[dev]"

# Run tests to verify setup
pytest tests/
```

### Areas for Contribution

**Questioning Strategy Development**: Create specialized questioning strategies for specific domains like scientific literature, creative writing, or technical documentation.

**Embedding Model Integration**: Add support for new embedding models or improve integration with existing ones.

**Performance Optimization**: Implement caching strategies, parallel processing improvements, or more efficient similarity calculations.

**Evaluation Metrics**: Develop better methods for evaluating investigation quality and convergence detection.

### Code Style and Testing

Perquire follows established Python conventions and maintains high test coverage:

```bash
# Format code
black perquire/
isort perquire/

# Run linting
flake8 perquire/

# Run full test suite
pytest tests/ --cov=perquire
```

## Research and Academic Use

Perquire opens new avenues for research in natural language processing, machine learning interpretability, and cognitive science. The system provides a novel framework for studying how semantic representations can be decoded through systematic inquiry.

### Citation

If you use Perquire in academic research, please cite:

```bibtex
@software{perquire2024,
  title={Perquire: Investigating Unknown Embeddings Through Systematic Questioning},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/perquire}
}
```

### Research Applications

**Cognitive Modeling**: Investigate how human-like questioning strategies compare to algorithmic approaches in semantic discovery.

**Embedding Analysis**: Study the structure and interpretability of different embedding spaces through systematic investigation.

**AI Alignment**: Explore how well AI systems can communicate their internal representations to humans through natural language.

## License

Perquire is released under the MIT License, encouraging both academic research and commercial applications while maintaining open collaboration.

## Support and Community

Join the Perquire community to share discoveries, get help, and contribute to the project's development:

- **Documentation**: [https://perquire.readthedocs.io](https://perquire.readthedocs.io)
- **Issues and Bugs**: [GitHub Issues](https://github.com/yourusername/perquire/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/perquire/discussions)
- **Discord**: [Perquire Community](https://discord.gg/perquire)

For commercial support and consulting, contact us at support@perquire.dev.

## Repository Structure

```
perquire/
├── src/perquire/           # Core library code
├── tests/                  # Test suite
├── examples/               # Example scripts and live E2E tests
├── benchmarks/             # Performance benchmarks
├── tools/                  # Development and maintenance tools
├── docs/                   # Documentation
│   ├── reports/           # Test reports and analysis
│   └── post-v1-backlog/   # Future development plans
└── README.md              # This file
```

### Quick Start Directories

- **`examples/`** - Try the live E2E test to see Perquire in action
- **`docs/reports/`** - View detailed test results and performance analysis
- **`benchmarks/`** - Run performance tests and comparisons
- **`src/perquire/`** - Explore the core investigation engine

---

_Perquire transforms the question from "what embeddings match this query?" to "what query would create this embedding?" - and in that inversion lies a new way of understanding the semantic spaces that AI creates._
