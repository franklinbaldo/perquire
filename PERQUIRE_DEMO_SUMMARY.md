# Perquire - Demonstration Summary

## Repository
- **URL**: https://github.com/franklinbaldo/perquire
- **Author**: Franklin Baldo
- **Status**: Experimental (v0.2.0)
- **License**: MIT
- **Local Path**: `/home/frank/workspace/perquire/`

## What is Perquire?

**Perquire** (from Latin *perquirere* - "to investigate thoroughly") is a revolutionary AI system that **reverses the traditional embedding search process**.

### The Core Innovation

- **Traditional Search**: "Find embeddings that match this query"
- **Perquire**: "What query would create this embedding?"

Think of it as a digital detective that receives an embedding vector of unknown origin and systematically questions it until uncovering what it represents.

## How It Works

### Three-Phase Investigation Process

1. **🌍 Exploration Phase**
   - Broad, categorical questions
   - Maps general semantic territory
   - Example: "Does this relate to emotions or objects?"

2. **🎯 Refinement Phase**
   - Narrower, focused questions
   - Zeroes in on specifics
   - Example: "Are these positive or negative emotions?"

3. **✨ Convergence Phase**
   - Highly specific questions
   - Statistical convergence detection
   - Synthesizes final description

### Similarity Guidance

- Uses **cosine similarity** as a "hot and cold" game
- Each question gets a similarity score to the target
- Higher similarity = warmer, getting closer
- Lower similarity = colder, explore different territory

### Example Investigation

| Phase | Question | Similarity | Status |
|-------|----------|------------|--------|
| 🌍 Exploration | "Does this relate to human emotions?" | 0.45 | Getting warmer... |
| 🌍 Exploration | "Is this about positive or negative feelings?" | 0.62 | Progress! |
| 🎯 Refinement | "Does this involve memory?" | 0.78 | Closing in... |
| 🎯 Refinement | "Is there longing or wistfulness?" | 0.89 | Almost there! |
| ✨ Convergence | "Does this capture nostalgia?" | 0.94 | Converged! ✓ |

**Discovery**: "The bittersweet feeling of nostalgia when looking through old photo albums"

## Architecture

```
┌─────────────────────────────────┐
│  1. Unknown Embedding          │  ← Input: mystery vector
│     (vector of unknown origin) │
└────────────┬───────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  2. Question Generator (LLM)   │  ← Generate strategic questions
│     • Exploration (broad)      │
│     • Refinement (narrow)      │
│     • Convergence detection    │
└────────────┬───────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  3. Similarity Calculator      │  ← "Hot and cold" guidance
│     • Cosine similarity        │
│     • Score each question      │
│     • Guide next questions     │
└────────────┬───────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  4. Synthesis                  │  ← Output: human description
│     • Human-readable text      │
│     • Confidence metrics       │
│     • Investigation history    │
└─────────────────────────────────┘
```

## Key Features

### 🤖 Pydantic AI Integration (New in v0.2.0)
- Type-safe LLM interactions
- 50% code reduction vs manual implementations
- Full IDE autocomplete and type checking
- Structured, validated outputs

### 🔄 Multi-Model Support
- **Gemini** (Google)
- **OpenAI** (GPT-4, GPT-3.5)
- **Anthropic** (Claude)
- **Ollama** (local models)

### 📊 Smart Convergence
- Automatic detection when investigation complete
- Statistical methods to avoid infinite loops
- Monitors rate of improvement

### 🎯 Adaptive Questioning
- Adjusts strategy based on similarity scores
- Custom questioning strategies per domain
- Ensemble methods for complex embeddings

### 🌐 Web Interface
- FastAPI server for batch processing
- File upload (.json, .npy, .txt, .csv)
- Status page with investigation history
- REST API

### 📈 Progress Tracking
- Rich terminal output
- Real-time similarity scores
- Investigation phase indicators

## Use Cases

### 1. Content Discovery in Vector Databases
**Problem**: Large collections of embeddings with limited metadata
**Solution**: Systematically investigate to generate descriptive metadata
**Example**: Making vast document collections searchable

### 2. Advanced Sentiment Analysis
**Problem**: Basic positive/negative isn't nuanced enough
**Solution**: Uncover complex emotional content
**Example**: "Nostalgic longing with hints of regret"

### 3. AI Model Interpretability
**Problem**: Black-box neural networks are hard to understand
**Solution**: Decode internal representations into natural language
**Example**: Understanding what hidden layers "think"

### 4. Creative Writing & Brainstorming
**Problem**: Need inspiration and unexpected connections
**Solution**: Explore semantic neighborhoods around concepts
**Example**: Finding themes for creative projects

## Quick Start

### Installation

```bash
git clone https://github.com/franklinbaldo/perquire.git
cd perquire

# Fix dependency issue first
# Edit pyproject.toml: change requires-python = ">=3.8" to ">=3.11"

uv sync  # Install dependencies
```

### Basic Usage

```python
from perquire import PerquireInvestigator
from sentence_transformers import SentenceTransformer

# Initialize with Gemini (via Pydantic AI)
investigator = PerquireInvestigator(
    embedding_model=SentenceTransformer('all-MiniLM-L6-v2'),
    llm_provider='gemini'  # or 'openai', 'anthropic', 'ollama'
)

# Create a mysterious embedding
mysterious_text = "The melancholic beauty of abandoned places"
target_embedding = investigator.embedding_model.encode(mysterious_text)

# Let Perquire investigate!
result = investigator.investigate(target_embedding)

print(f"Discovery: {result.description}")
print(f"Confidence: {result.final_similarity:.1%}")
print(f"Questions asked: {result.iterations}")
```

### CLI Commands

```bash
# Start web server
perquire serve --host 0.0.0.0 --port 8080

# Investigate from file
perquire investigate embedding.npy --output result.json

# Batch processing
perquire batch embeddings/ --format csv
```

### Web Interface

```bash
pip install "perquire[web]"
perquire serve --reload  # Development mode
```

Access at `http://127.0.0.1:8000`

## Live Testing

The repository includes comprehensive end-to-end tests:

```bash
cd perquire/examples

# Set API key
export GOOGLE_API_KEY="your-key-here"

# Interactive test
uv run --env-file ../.env python live_e2e_test.py

# Automated test
bash run_live_e2e.sh
```

Recent test results:
- **4.0/5 EXCELLENT** subjective rating
- **6.76 seconds** investigation time
- **5-iteration convergence**

## Project Structure

```
perquire/
├── src/perquire/
│   ├── core/                 # Core investigation engine
│   │   ├── investigator.py   # Main PerquireInvestigator class
│   │   ├── strategy.py       # Questioning strategies
│   │   ├── result.py         # Investigation results
│   │   └── ensemble.py       # Multi-model ensemble
│   ├── llm/                  # LLM provider integrations
│   │   ├── pydantic_ai_provider.py  # New Pydantic AI provider
│   │   ├── gemini_provider.py
│   │   ├── openai_provider.py
│   │   └── anthropic_provider.py
│   ├── convergence/          # Convergence detection algorithms
│   ├── database/             # DuckDB for persistence
│   ├── akinator/             # Akinator-style questioning
│   └── cli/                  # CLI interface
├── examples/
│   ├── live_e2e_test.py      # Live end-to-end test
│   ├── pydantic_ai_integration_demo.py
│   └── run_live_e2e.sh
├── tests/                    # Test suite
├── benchmarks/               # Performance tests
├── docs/                     # Documentation
│   ├── reports/              # Test reports
│   └── PYDANTIC_AI_FIX.md   # Pydantic AI migration guide
└── README.md
```

## Key Dependencies

- `pydantic-ai>=0.0.14` - Type-safe LLM interactions
- `google-generativeai>=0.8.5` - Gemini API
- `anthropic>=0.56.0` - Claude API
- `llama-index-*` - Various LLM integrations
- `scikit-learn>=1.7.0` - Similarity calculations
- `duckdb>=1.3.0` - Vector database
- `fastapi>=0.111.0` - Web interface
- `rich>=14.0.0` - Terminal UI

## Known Issues

1. **Python Version Mismatch**: `pyproject.toml` specifies `requires-python = ">=3.8"` but `numpy>=2.3.1` requires `>=3.11`.
   - **Fix**: Change to `requires-python = ">=3.11"`

2. **Package Mode**: Project needs `tool.uv.package = true` or `build-system` for entry points

## Demonstration Script

Run the demonstration:

```bash
cd /home/frank/workspace
python3 perquire-demo-auto.py
```

This shows:
- Conceptual explanation
- Investigation process example
- Architecture diagram
- Real-world use cases
- Key features
- Quick start code
- CLI commands

## Why Perquire Matters

Traditional embedding search requires knowing what you're looking for. Perquire enables:

1. **Discovery without prior knowledge** - Investigate unknown embeddings
2. **Semantic archaeology** - Uncover what embeddings represent
3. **AI transparency** - Make black-box models more interpretable
4. **Content understanding** - Generate metadata for vector databases

### The Paradigm Shift

```
Traditional: Query → Embedding → Search → Match
Perquire:    Embedding → Questions → Similarity → Discovery
```

## Academic & Research Applications

- **Cognitive Modeling**: Study human-like vs algorithmic questioning
- **Embedding Analysis**: Understand structure of embedding spaces
- **AI Alignment**: Improve AI-human communication
- **NLP Research**: Novel framework for semantic representation study

## Citation

```bibtex
@software{perquire2024,
  title={Perquire: Investigating Unknown Embeddings Through Systematic Questioning},
  author={Franklin Baldo},
  year={2024},
  url={https://github.com/franklinbaldo/perquire}
}
```

## Summary

**Perquire reverses the question**: Instead of "what embeddings match this query?", it asks "what query would create this embedding?"

This inversion opens new possibilities for:
- Content discovery in large vector databases
- Nuanced sentiment analysis
- AI interpretability and transparency
- Creative exploration and brainstorming

By treating embeddings as mysteries to be solved rather than targets to be matched, Perquire provides a powerful new tool for understanding the semantic spaces that AI systems create.

---

*Demo generated on 2025-11-04*
*Perquire v0.2.0 - Franklin Baldo*
