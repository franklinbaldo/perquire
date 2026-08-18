# Perquire

_From Latin **perquirere** — to investigate thoroughly, to question deeply._

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)
![Status](https://img.shields.io/badge/status-experimental-orange.svg)

Perquire is an experimental system for investigating an embedding of unknown meaning through iterative questioning. It combines an LLM, an embedding provider and similarity feedback to generate and refine hypotheses about what a target vector may represent.

The repository currently exposes the same investigation model through several concrete interfaces:

- a Python library;
- the `perquire` CLI for single and batch investigation, status and export;
- a local FastAPI/Jinja web interface launched with `perquire serve`;
- DuckDB-backed investigation history/status when a database is configured;
- provider-backed end-to-end tests for the live investigation path.

Perquire is research/developer tooling. Claims produced by an investigation are model outputs to inspect, not ground truth about an embedding.

## Requirements and installation

The current package contract requires **Python 3.12 or newer**.

```bash
# From PyPI
pip install perquire

# Or from source
git clone https://github.com/franklinbaldo/perquire.git
cd perquire
pip install -e .
```

Provider requirements depend on the LLM and embedding backends you choose. The CLI can show the providers available in the current environment:

```bash
perquire providers
```

## CLI

Investigate one embedding file:

```bash
perquire investigate embedding.npy
```

Process a directory of embeddings:

```bash
perquire batch ./embeddings --format npy
```

Inspect a DuckDB investigation database:

```bash
perquire status --database perquire.db
```

Export stored investigations:

```bash
perquire export --database perquire.db --output investigations.json
```

Run `perquire --help` or a command's `--help` for the current options.

## Local web interface

Perquire includes a human-facing FastAPI/Jinja interface for manual investigation, embedding-file upload, batch work, results and status/history.

Launch it with:

```bash
perquire serve
```

The default bind address is `http://127.0.0.1:8000`. Host, port, database path and development reload can be changed explicitly:

```bash
perquire serve \
  --host 127.0.0.1 \
  --port 8080 \
  --database perquire.db
```

### Deployment status

The web interface is currently documented and tested as a **local application surface**. This repository does not currently provide or claim a supported public deployment of that interface. Binding to another host address is a runtime option, not evidence of a maintained hosted service.

That distinction is deliberate: the repository has deterministic rendered-template smoke coverage and a provider-backed E2E path, while public hosting remains a separate product/deployment decision.

## How the investigation works

At a high level, an investigation follows this loop:

```text
unknown embedding
  → generate a question or hypothesis
  → embed that candidate
  → compare it with the target
  → use similarity feedback to refine the next step
  → synthesize a result when the strategy stops
```

The exact questioning and convergence behavior depends on the configured providers and strategy. Similarity is guidance inside the investigation process; it should not be read as an independent factual confidence measure about the real-world meaning of the target vector.

## Persistence and reuse

When a DuckDB database is configured, Perquire can persist investigation results and expose them again through CLI/web status and history surfaces. The CLI can also export stored investigations to JSON, CSV or text, which makes the result set available for independent inspection or downstream analysis.

This is separate from the live provider path: a stored/exported investigation records what Perquire produced; it does not turn the generated interpretation into verified ground truth.

## Development and verification

Install the repository in editable mode and run the test suite:

```bash
pip install -e .
pytest
```

The repository also contains CI coverage for the rendered Jinja surface and a live Gemini-backed end-to-end workflow on trusted runs. Provider-backed tests require their credentials through the GitHub Actions secret/environment boundary; secret values must never be committed or printed.

Useful project areas include:

```text
src/perquire/core/       investigation engine and result model
src/perquire/cli/        CLI surface
src/perquire/web/        FastAPI/Jinja human-facing surface
src/perquire/database/   persistence providers
src/perquire/llm/        LLM provider integration
src/perquire/embeddings/ embedding provider integration
tests/                   deterministic and end-to-end verification
benchmarks/              performance experiments
```

## Project status

Perquire is experimental software. The most reliable description of a capability is the current source, package contract and passing tests/workflows. Historical demo and upgrade notes in the repository are useful development records, but they should not be treated as stronger authority than the current implementation.

## License

Perquire is released under the MIT License. See [LICENSE](LICENSE).
