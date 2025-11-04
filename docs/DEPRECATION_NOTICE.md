# Deprecation Notice

## Deprecated LLM Providers

The following LLM provider classes are **deprecated** as of version 0.2.0 and will be **removed in version 1.0.0**:

### Deprecated Classes

- ❌ `GeminiProvider` (src/perquire/llm/gemini_provider.py)
- ❌ `OpenAIProvider` (src/perquire/llm/openai_provider.py)
- ❌ `AnthropicProvider` (src/perquire/llm/anthropic_provider.py)
- ❌ `OllamaProvider` (src/perquire/llm/ollama_provider.py)

### Replacement

Use **`PydanticAIProvider`** instead:

```python
# OLD (Deprecated)
from perquire.llm.gemini_provider import GeminiProvider
provider = GeminiProvider(config={'model': 'gemini-pro'})

# NEW (Recommended)
from perquire.llm.pydantic_ai_provider import create_pydantic_gemini_provider
provider = create_pydantic_gemini_provider(model='gemini-1.5-flash')
```

## Why the Change?

The new `PydanticAIProvider` offers:

1. **50% less code** - Single provider replaces four manual implementations
2. **Type safety** - Automatic validation via Pydantic models
3. **Better DX** - Full IDE autocomplete and type checking
4. **Unified interface** - Same API for all models (Gemini, OpenAI, Anthropic, Ollama)
5. **Structured outputs** - Optional access to validated question metadata

## Migration Guide

### For Gemini Users

**Before:**
```python
from perquire.llm.gemini_provider import GeminiProvider

provider = GeminiProvider({
    'api_key': 'your-key',
    'model': 'gemini-pro',
    'temperature': 0.7
})
```

**After:**
```python
from perquire.llm.pydantic_ai_provider import create_pydantic_gemini_provider

provider = create_pydantic_gemini_provider(
    model='gemini-1.5-flash',  # or 'gemini-1.5-pro'
    api_key='your-key',
    temperature=0.7
)
```

### For OpenAI Users

**Before:**
```python
from perquire.llm.openai_provider import OpenAIProvider

provider = OpenAIProvider({
    'api_key': 'your-key',
    'model': 'gpt-4',
    'temperature': 0.7
})
```

**After:**
```python
from perquire.llm.pydantic_ai_provider import create_pydantic_openai_provider

provider = create_pydantic_openai_provider(
    model='gpt-4',
    api_key='your-key',
    temperature=0.7
)
```

### For Anthropic Users

**Before:**
```python
from perquire.llm.anthropic_provider import AnthropicProvider

provider = AnthropicProvider({
    'api_key': 'your-key',
    'model': 'claude-3-opus',
    'temperature': 0.7
})
```

**After:**
```python
from perquire.llm.pydantic_ai_provider import create_pydantic_anthropic_provider

provider = create_pydantic_anthropic_provider(
    model='claude-3-5-sonnet-20241022',
    api_key='your-key',
    temperature=0.7
)
```

### For Ollama Users

**Before:**
```python
from perquire.llm.ollama_provider import OllamaProvider

provider = OllamaProvider({
    'model': 'llama2',
    'temperature': 0.7
})
```

**After:**
```python
from perquire.llm.pydantic_ai_provider import create_pydantic_ollama_provider

provider = create_pydantic_ollama_provider(
    model='llama2',
    temperature=0.7
)
```

## With PerquireInvestigator

**Before:**
```python
from perquire import PerquireInvestigator
from perquire.llm.gemini_provider import GeminiProvider

provider = GeminiProvider({'model': 'gemini-pro'})
investigator = PerquireInvestigator(llm_provider=provider)
```

**After:**
```python
from perquire import PerquireInvestigator
from perquire.llm.pydantic_ai_provider import create_pydantic_gemini_provider

provider = create_pydantic_gemini_provider(model='gemini-1.5-flash')
investigator = PerquireInvestigator(llm_provider=provider)
```

## With Registry

**Before:**
```python
from perquire.llm.base import provider_registry
from perquire.llm.gemini_provider import GeminiProvider

provider = GeminiProvider({'model': 'gemini-pro'})
provider_registry.register_provider('gemini', provider)
```

**After:**
```python
from perquire.llm.base import provider_registry
from perquire.llm.pydantic_ai_provider import create_pydantic_gemini_provider

provider = create_pydantic_gemini_provider(model='gemini-1.5-flash')
provider_registry.register_provider('pydantic-gemini', provider)
```

## Breaking Changes

### API Changes

| Old API | New API |
|---------|---------|
| `GeminiProvider(config: Dict)` | `create_pydantic_gemini_provider(**kwargs)` |
| `OpenAIProvider(config: Dict)` | `create_pydantic_openai_provider(**kwargs)` |
| `AnthropicProvider(config: Dict)` | `create_pydantic_anthropic_provider(**kwargs)` |
| `OllamaProvider(config: Dict)` | `create_pydantic_ollama_provider(**kwargs)` |

### Return Types

All methods maintain the same return types (backward compatible):
- `generate_questions()` → `List[str]` ✅
- `synthesize_description()` → `str` ✅
- `generate_response()` → `LLMResponse` ✅

**Bonus:** You can now access structured outputs:
```python
questions = provider.generate_questions(...)  # Returns List[str]

# Optional: Get structured data
batch = provider.get_last_question_batch()  # Returns QuestionBatch
if batch:
    for q in batch.questions:
        print(f"{q.question} (expected gain: {q.expected_similarity_gain})")
```

## Timeline

- **v0.2.0** (Current): Deprecated providers still work, deprecation warnings added
- **v0.3.0** (Q1 2025): Deprecated providers removed from documentation
- **v1.0.0** (Q2 2025): Deprecated providers completely removed

## Need Help?

- See full migration guide: [`docs/PYDANTIC_AI_MIGRATION.md`](PYDANTIC_AI_MIGRATION.md)
- See implementation details: [`docs/PYDANTIC_AI_FIX.md`](PYDANTIC_AI_FIX.md)
- Run demo: `python examples/pydantic_ai_integration_demo.py`
- Open an issue: https://github.com/franklinbaldo/perquire/issues

## FAQ

### Will my code break immediately?

No. The old providers will continue to work until version 1.0.0. You'll see deprecation warnings, but functionality is unchanged.

### Do I need to update my code now?

Not immediately, but we recommend migrating when convenient. The new provider offers significant improvements.

### What if I encounter issues?

Open an issue on GitHub. We'll help with migration and fix any bugs in the new provider.

### Can I use both old and new providers?

Yes, during the deprecation period you can use both. They're fully compatible via the registry.

### Will provider_registry work with both?

Yes, the new `PydanticAIProvider` properly inherits from `BaseLLMProvider` and works seamlessly with the registry.
