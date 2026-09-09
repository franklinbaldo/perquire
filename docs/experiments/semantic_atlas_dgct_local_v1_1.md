# Semantic Atlas DGCT local v1.1

Perquire is only the execution carrier for the corrected, prospectively frozen DGCT v1.1 owned by `franklinbaldo/papers`.

Frozen papers commit: `0f4e8f6b117a8f461b0253f2b072269968dac8a1`.

The v1.1 amendment changes only observer visibility after v1 was invalidated by MiniLM truncating away the changing continuation. The generator, field families and parameters, trajectory split, amplitude model and complexity metrics are unchanged.

The workflow reuses the existing Hugging Face model cache and executes:

```bash
python scripts/run_dynamic_gauge_compression_v1_1.py
```

The scientific artifact is `artifacts/dynamic_gauge_compression_v1_1.json`. The runner includes a hard observability gate; a transfer observer with degenerate dynamics causes failure rather than a green but uninterpretable result.
