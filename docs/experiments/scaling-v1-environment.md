# Scaling v1 resolved software environment

This manifest records the software environment actually observed in GitHub Actions run `32356968748`, job `96388139312` (`shard (1, 2)`), before any v1 resume attempt.

It is evidence recovered from the original run log, not a reconstruction from today's dependency resolver.

## Execution identity

- GitHub Actions run: `32356968748`
- job: `96388139312` (`shard (1, 2)`)
- checked-out merge commit: `f5f54908081e5caf19600daa7ed0539b373509d3`
- experiment head SHA: `482a86ff08856c2588d18ae44ee44c816b799532`
- base SHA: `a37be9d7e2983a67eefeb56471d07096fafdb4b1`
- runner OS: Ubuntu 24.04.4
- runner image: `ubuntu-24.04`, image version `20260810.271.1`
- Python: CPython 3.12.13
- pip: 26.2.1
- perquire: 0.2.0

## Experiment-critical packages

These packages directly affect provider/error translation, numerical scoring, or the OpenRouter transport boundary and therefore must not drift silently inside scaling v1:

- `litellm==1.97.0`
- `numpy==2.5.2`
- `openai==2.54.0`
- `httpx==0.28.1`
- `httpcore==1.0.9`
- `pydantic==2.13.4`
- `pydantic-core==2.46.4`
- `pydantic-settings==2.15.0`
- `aiohttp==3.14.3`
- `tiktoken==0.14.0`
- `tokenizers==0.23.1`

The project dependency for OpenRouter is pinned to `litellm==1.97.0` specifically because LiteLLM maps provider HTTP responses into Perquire-visible exceptions. Changing it during a resumed frozen experiment could silently change retry and invalid-cell behavior.

## Full installed set reported by pip

The original successful install reported:

`MarkupSafe==3.0.3 aiohappyeyeballs==2.7.1 aiohttp==3.14.3 aiosignal==1.4.0 annotated-doc==0.0.5 annotated-types==0.8.0 anyio==4.14.2 attrs==26.1.0 certifi==2026.7.22 charset_normalizer==3.5.1 click==8.4.2 distro==1.9.0 fastuuid==0.14.0 filelock==3.32.3 frozenlist==1.8.0 fsspec==2026.7.0 h11==0.16.0 hf-xet==1.6.0 httpcore==1.0.9 httpx==0.28.1 huggingface-hub==1.28.0 idna==3.19 importlib-metadata==8.9.0 Jinja2==3.1.6 jiter==0.16.0 jsonschema==4.26.0 jsonschema-specifications==2025.9.1 litellm==1.97.0 markdown-it-py==4.2.0 mdurl==0.1.2 multidict==6.7.1 numpy==2.5.2 openai==2.54.0 packaging==26.3 perquire==0.2.0 propcache==0.5.2 pydantic==2.13.4 pydantic-core==2.46.4 pydantic-settings==2.15.0 pygments==2.21.0 python-dotenv==1.2.3 PyYAML==6.0.3 referencing==0.37.0 regex==2026.7.19 requests==2.34.2 rich==15.0.0 rpds-py==2026.6.3 shellingham==1.5.4 sniffio==1.3.1 tiktoken==0.14.0 tokenizers==0.23.1 tqdm==4.70.0 typer==0.27.1 typing-inspection==0.4.4 typing_extensions==4.16.0 urllib3==2.7.0 yarl==1.24.5 zipp==4.1.0`

## Freeze rule

Any resume or extension that claims continuity with scaling v1 must:

1. use CPython 3.12.x and record the exact patch version;
2. install from committed `uv.lock` with `uv sync --frozen`;
3. preserve `litellm==1.97.0` unless the run is explicitly classified as a new protocol/version;
4. record git SHA, Python version, Perquire version, LiteLLM version and NumPy version in the produced artifact;
5. treat any environment drift as metadata-visible and never silently combine drifted cells into the canonical v1 artifact.

The lock introduced after the first partial v1 run cannot retroactively prove the whole original environment. This manifest is the provenance bridge: the original run log is the source of truth for already-produced cells, while the lock constrains future cells.