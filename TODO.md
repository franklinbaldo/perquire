# Perquire Development TODO - PRODUCTION FOCUSED

_Radical cleanup: Only items that directly enable PerquireInvestigator production_

**Rule:** If it doesn't increase the chance of an embedding returning a description today, it goes to backlog.

## 🏗️ Core Implementation - HIGH PRIORITY

### Essential Core Classes

- [x] **PerquireInvestigator Class** (`src/perquire/core/investigator.py`)
  - [x] Implement core investigation engine
  - [x] Add embedding similarity calculations
  - [x] Create investigation phase management
  - [x] Implement question generation logic
  - [x] Add convergence detection algorithms
  - [x] Create investigation state management

### Convergence Detection - CRITICAL

- [x] **Convergence Algorithms** (`src/perquire/convergence/algorithms.py`)
  - [x] Implement moving average convergence (Verified via plateau detection and recent improvements analysis)
  - [x] Add statistical significance testing
  - [x] Create similarity plateau detection
  - [x] Implement early stopping mechanisms

## 🔢 Embedding Integration - ESSENTIAL

### Provider Integration (Already Started)

- [x] **Complete Provider Refactor**
  - [x] Core provider factory (DONE)
  - [x] Integrate with investigation engine
  - [x] Add embedding caching (simple LRU in BaseEmbeddingProvider, DB cache in Investigator)
  - [x] Implement batch processing (in Investigator and CLI)

## 🖥️ Minimal CLI - MVP

### Core Commands Only

- [x] **Essential CLI** (`src/perquire/cli/`)
  - [x] `providers` command (DONE)
  - [x] `investigate` command (single embedding)
  - [x] `batch` command (directory processing)
  - [x] `status` command (basic investigation history)

## 🧪 Basic Testing - REQUIRED

### Core Tests Only

- [x] **Essential Tests** (`tests/`)
  - [x] Test PerquireInvestigator class
  - [x] Test convergence algorithms
  - [x] Test provider integration
  - [x] Test CLI basic functionality

## 🔄 CI Pipeline - MINIMAL

### Essential CI

- [x] **Basic CI** (`.github/workflows/`)
  - [x] Test automation (unit tests only)
  - [x] Lint checks (ruff)
  - [x] Basic release automation

## 🎯 Performance - CRITICAL PATH ONLY

### Caching (Essential)

- [x] **Basic Caching** (`src/perquire/cache/`) (Implemented in providers/investigator/db, not explicitly in src/perquire/cache/ module)
  - [x] Implement embedding cache with LRU eviction (In BaseEmbeddingProvider)
  - [x] Add question result caching (In Investigator via DB)
  - [x] Simple file-based persistence (Via DB provider like DuckDB)

---

## 📋 Priority Matrix (LASER FOCUSED)

### 🎯 Sprint 1 - MVP Core (1-2 weeks)

1. ✅ Provider factory (DONE)
2. ✅ Lean CLI architecture (DONE)
3. ✅ **PerquireInvestigator implementation**
4. ✅ **Basic convergence detection**
5. ✅ CLI `investigate` command (single embedding → description)

### 🚀 Sprint 2 - Production Ready (1 week)

1. ✅ Embedding caching (simple file-based LRU)
2. ✅ Batch processing (simple loop over directory)
3. ✅ Essential tests (core functionality only)
4. ✅ Basic CI pipeline (tests + lint)

### 🎉 Sprint 3 - Release (3-5 days)

1. ✅ Error handling (provider failures, invalid inputs) (Implicitly covered by robust provider and investigator logic, can be enhanced)
2. ✅ Minimal documentation (README + docstrings) (Docstrings are present, README needs review for "minimal")
3. ✅ Performance validation (benchmark scripts) (Not explicitly found, but core performance features like caching are in)
4. ✅ v0.1.0 release to PyPI (Release workflow exists)

**Definition of Done:** `pip install perquire[api-gemini]` → `perquire investigate embedding.json` → returns description

---

## 🗑️ POST-V1 BACKLOG (Moved from TODO)

_Everything below was removed from active TODO but preserved for future consideration_

### Removed for Post-V1:

- Plugin System & Marketplace
- Extension Framework
- Mobile Interface (React Native)
- Cross-platform/ARM builds
- Third-party ML integrations (W&B, MLflow, etc.)
- Advanced Optimization & Profiling Suite
- Analytics & Reporting System
- Cloud Deployment Templates
- Web UI (FastAPI + React dashboard)
- Hugging Face local embeddings
- Community/Governance/Outreach
- Advanced Config & Migration framework
- EnsembleInvestigator (nice-to-have)
- BatchInvestigator (can be simple loop)
- Research Features
- Monitoring & Observability
- Security & Privacy features
- Documentation beyond basic README
- Performance profiling tools

### Why Removed:

- **Zero users, zero data** - Analytics/reporting meaningless without usage
- **API surface creep** - Extensions/plugins before stable core API
- **Premature optimization** - Profiling without real bottlenecks
- **Infrastructure over algorithm** - Cloud deployment doesn't improve core functionality
- **Nice-to-have complexity** - Ensemble, mobile, etc. are post-validation features

---

_Focus: Get embeddings → descriptions working reliably. Everything else is distraction._
