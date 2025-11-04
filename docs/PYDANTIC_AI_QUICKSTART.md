# Pydantic AI Provider - Quickstart Guide

Get started with PERQUIRE's modernized, type-safe LLM provider in 5 minutes.

## Why Pydantic AI Provider?

- ✅ **50% less code** - One provider for all models
- ✅ **Type-safe** - Automatic validation via Pydantic
- ✅ **Better DX** - Full IDE autocomplete
- ✅ **Multi-model** - Gemini, OpenAI, Anthropic, Ollama
- ✅ **Backward compatible** - Drop-in replacement

## Installation

```bash
# Install PERQUIRE with Pydantic AI support
pip install perquire

# The package already includes pydantic-ai>=0.0.14
```

## 5-Minute Tutorial

### Step 1: Set API Keys

```bash
# For Gemini
export GOOGLE_API_KEY="your-gemini-key"

# For OpenAI
export OPENAI_API_KEY="your-openai-key"

# For Anthropic
export ANTHROPIC_API_KEY="your-anthropic-key"

# Ollama (local) - no API key needed
```

### Step 2: Create a Provider

```python
from perquire.llm.pydantic_ai_provider import create_pydantic_gemini_provider

# Create provider with sensible defaults
provider = create_pydantic_gemini_provider(
    model="gemini-1.5-flash",  # Fast and cost-effective
    temperature=0.7
)
```

### Step 3: Use with PerquireInvestigator

```python
from perquire import PerquireInvestigator

# Create investigator with Pydantic AI provider
investigator = PerquireInvestigator(llm_provider=provider)

# Investigate an embedding
result = investigator.investigate(your_embedding)
print(result.description)
```

### Step 4: (Optional) Access Structured Outputs

```python
# Generate questions (returns List[str] for backward compatibility)
questions = provider.generate_questions(
    current_description="emotions",
    target_similarity=0.65,
    phase="exploration",
    previous_questions=[]
)

# Bonus: Access structured metadata
batch = provider.get_last_question_batch()
if batch:
    print(f"Strategy: {batch.strategy}")
    print(f"Similarity: {batch.current_similarity}")
    for q in batch.questions:
        print(f"  • {q.question}")
        print(f"    Expected gain: {q.expected_similarity_gain:.3f}")
        print(f"    Rationale: {q.rationale}")
```

## Complete Example

```python
from perquire import PerquireInvestigator
from perquire.llm.pydantic_ai_provider import create_pydantic_gemini_provider
from sentence_transformers import SentenceTransformer

# 1. Create embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Create Pydantic AI provider
llm_provider = create_pydantic_gemini_provider(
    model="gemini-1.5-flash",
    temperature=0.7,
    max_tokens=1000
)

# 3. Create investigator
investigator = PerquireInvestigator(
    embedding_provider=embedding_model,
    llm_provider=llm_provider
)

# 4. Investigate a mysterious embedding
mysterious_text = "The melancholic beauty of abandoned places"
target_embedding = embedding_model.encode(mysterious_text)

result = investigator.investigate(target_embedding)

# 5. See results
print(f"📝 Description: {result.description}")
print(f"🎯 Confidence: {result.final_similarity:.1%}")
print(f"❓ Questions asked: {result.iterations}")
```

## Provider Factory Functions

### Gemini
```python
from perquire.llm.pydantic_ai_provider import create_pydantic_gemini_provider

provider = create_pydantic_gemini_provider(
    model="gemini-1.5-flash",  # or "gemini-1.5-pro"
    api_key="your-key",  # optional, uses GOOGLE_API_KEY env var
    temperature=0.7,
    max_tokens=1000
)
```

### OpenAI
```python
from perquire.llm.pydantic_ai_provider import create_pydantic_openai_provider

provider = create_pydantic_openai_provider(
    model="gpt-4",  # or "gpt-4-turbo", "gpt-3.5-turbo"
    api_key="your-key",  # optional, uses OPENAI_API_KEY env var
    temperature=0.7,
    max_tokens=1000
)
```

### Anthropic
```python
from perquire.llm.pydantic_ai_provider import create_pydantic_anthropic_provider

provider = create_pydantic_anthropic_provider(
    model="claude-3-5-sonnet-20241022",  # or other Claude models
    api_key="your-key",  # optional, uses ANTHROPIC_API_KEY env var
    temperature=0.7,
    max_tokens=1000
)
```

### Ollama (Local)
```python
from perquire.llm.pydantic_ai_provider import create_pydantic_ollama_provider

provider = create_pydantic_ollama_provider(
    model="llama2",  # or any Ollama model you have installed
    temperature=0.7,
    max_tokens=1000
)
```

## Using with Registry

```python
from perquire.llm.base import provider_registry
from perquire.llm.pydantic_ai_provider import (
    create_pydantic_gemini_provider,
    create_pydantic_openai_provider
)

# Register multiple providers
provider_registry.register_provider(
    "pydantic-gemini",
    create_pydantic_gemini_provider()
)

provider_registry.register_provider(
    "pydantic-openai",
    create_pydantic_openai_provider()
)

# Use by name
investigator = PerquireInvestigator(llm_provider="pydantic-gemini")
```

## Advanced: Direct Provider Creation

For more control, create the provider directly:

```python
from perquire.llm.pydantic_ai_provider import PydanticAIProvider

provider = PydanticAIProvider({
    'model': 'google-gla:gemini-1.5-flash',  # Note: prefix required
    'api_key': 'your-key',
    'temperature': 0.7,
    'max_tokens': 1000
})
```

**Model prefixes:**
- Gemini: `google-gla:gemini-1.5-flash`
- OpenAI: `openai:gpt-4`
- Anthropic: `anthropic:claude-3-5-sonnet-20241022`
- Ollama: `ollama:llama2`

## Type-Safe Outputs

The provider uses Pydantic models internally:

```python
from perquire.llm.models import InvestigationContext

# Create type-safe context
context = InvestigationContext(
    current_description="emotions and experiences",
    current_similarity=0.65,
    phase="refinement",  # Type-checked!
    previous_questions=["Is this positive?", "About relationships?"],
    iteration=3
)

# This would raise a validation error:
# context = InvestigationContext(
#     phase="invalid_phase"  # ❌ Not in Literal["exploration", "refinement", "convergence"]
# )
```

## Error Handling

```python
from perquire.exceptions import LLMProviderError, ConfigurationError

try:
    provider = create_pydantic_gemini_provider(
        model="gemini-1.5-flash"
    )
    questions = provider.generate_questions(...)

except ConfigurationError as e:
    print(f"Configuration error: {e}")
    # Handle missing API key, invalid model, etc.

except LLMProviderError as e:
    print(f"LLM provider error: {e}")
    # Handle generation failures, rate limits, etc.
```

## Configuration Tips

### For Development
```python
# Use faster, cheaper models
provider = create_pydantic_gemini_provider(
    model="gemini-1.5-flash",
    temperature=0.7,
    max_tokens=500  # Faster responses
)
```

### For Production
```python
# Use more capable models
provider = create_pydantic_openai_provider(
    model="gpt-4",
    temperature=0.5,  # More consistent
    max_tokens=1500  # More detailed
)
```

### For Privacy
```python
# Use local Ollama
provider = create_pydantic_ollama_provider(
    model="llama2",
    temperature=0.7
)
```

## Next Steps

- ✅ Run the integration demo: `python examples/pydantic_ai_integration_demo.py`
- 📖 Read the migration guide: [`docs/PYDANTIC_AI_MIGRATION.md`](PYDANTIC_AI_MIGRATION.md)
- 🔧 See implementation details: [`docs/PYDANTIC_AI_FIX.md`](PYDANTIC_AI_FIX.md)
- ⚠️  Check deprecation notice: [`docs/DEPRECATION_NOTICE.md`](DEPRECATION_NOTICE.md)

## Troubleshooting

### "No module named 'pydantic_ai'"
```bash
# Reinstall with latest dependencies
pip install -e .
```

### "API key not found"
```bash
# Set environment variable
export GOOGLE_API_KEY="your-key"

# Or pass directly
provider = create_pydantic_gemini_provider(api_key="your-key")
```

### "Model not found"
```python
# Check model name format
# Gemini: "gemini-1.5-flash" (NOT "google-gla:gemini-1.5-flash")
# OpenAI: "gpt-4" (NOT "openai:gpt-4")
# Factory functions handle prefixes automatically
```

### Deprecation Warnings
```python
# If you see deprecation warnings, you're using old providers
# Replace:
from perquire.llm.gemini_provider import GeminiProvider

# With:
from perquire.llm.pydantic_ai_provider import create_pydantic_gemini_provider
```

## FAQ

**Q: Is this backward compatible?**
A: Yes! All return types match the old providers exactly.

**Q: Can I switch between providers easily?**
A: Yes! All factory functions have the same API.

**Q: Do I need to change my existing code?**
A: Not immediately. Old providers work but are deprecated.

**Q: What's the performance impact?**
A: Minimal (<10ms overhead). Pydantic validation is fast.

**Q: Can I use this in production?**
A: Yes! It's fully tested and backward compatible.

## Support

- 🐛 Report issues: https://github.com/franklinbaldo/perquire/issues
- 💬 Discussions: https://github.com/franklinbaldo/perquire/discussions
- 📧 Email: support@perquire.dev
