# Perquire Development TODO - PRODUCTION FOCUSED

*Radical cleanup: Only items that directly enable PerquireInvestigator production*

**Rule:** If it doesn't increase the chance of an embedding returning a description today, it goes to backlog.

## 🏗️ Core Implementation - HIGH PRIORITY

### Essential Core Classes
- [ ] **PerquireInvestigator Class** (`src/perquire/core/investigator.py`)
  - [ ] Implement core investigation engine 
  - [ ] Add embedding similarity calculations
  - [ ] Create investigation phase management
  - [ ] Implement question generation logic
  - [ ] Add convergence detection algorithms
  - [ ] Create investigation state management

### Convergence Detection - CRITICAL
- [ ] **Convergence Algorithms** (`src/perquire/convergence/algorithms.py`)
  - [ ] Implement moving average convergence
  - [ ] Add statistical significance testing  
  - [ ] Create similarity plateau detection
  - [ ] Implement early stopping mechanisms

## 🔢 Embedding Integration - ESSENTIAL

### Provider Integration (Already Started)
- [ ] **Complete Provider Refactor** 
  - [x] Core provider factory (DONE)
  - [ ] Integrate with investigation engine
  - [ ] Add embedding caching (simple LRU)
  - [ ] Implement batch processing

## 🖥️ Minimal CLI - MVP

### Core Commands Only
- [ ] **Essential CLI** (`src/perquire/cli/`)
  - [x] `providers` command (DONE)
  - [ ] `investigate` command (single embedding)
  - [ ] `batch` command (directory processing)  
  - [ ] `status` command (basic investigation history)

## 🧪 Basic Testing - REQUIRED

### Core Tests Only
- [ ] **Essential Tests** (`tests/`)
  - [ ] Test PerquireInvestigator class
  - [ ] Test convergence algorithms
  - [ ] Test provider integration
  - [ ] Test CLI basic functionality

## 🔄 CI Pipeline - MINIMAL

### Essential CI
- [ ] **Basic CI** (`.github/workflows/`)
  - [ ] Test automation (unit tests only)
  - [ ] Lint checks (ruff)
  - [ ] Basic release automation

## 🎯 Performance - CRITICAL PATH ONLY

### Caching (Essential)
- [ ] **Basic Caching** (`src/perquire/cache/`)
  - [ ] Implement embedding cache with LRU eviction
  - [ ] Add question result caching
  - [ ] Simple file-based persistence

---

## 📋 Priority Matrix (LASER FOCUSED)

### 🎯 Sprint 1 - MVP Core (1-2 weeks)
1. ✅ Provider factory (DONE)
2. ✅ Lean CLI architecture (DONE)  
3. **PerquireInvestigator implementation** ← NEXT
4. **Basic convergence detection** ← NEXT
5. CLI `investigate` command (single embedding → description)

### 🚀 Sprint 2 - Production Ready (1 week)  
1. Embedding caching (simple file-based LRU)
2. Batch processing (simple loop over directory)
3. Essential tests (core functionality only)
4. Basic CI pipeline (tests + lint)

### 🎉 Sprint 3 - Release (3-5 days)
1. Error handling (provider failures, invalid inputs)
2. Minimal documentation (README + docstrings)
3. Performance validation (benchmark scripts)
4. v0.1.0 release to PyPI

**Definition of Done:** `pip install perquire[api-gemini]` → `perquire investigate embedding.json` → returns description

---

## 🗑️ POST-V1 BACKLOG (Moved from TODO)

*Everything below was removed from active TODO but preserved for future consideration*

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

*Focus: Get embeddings → descriptions working reliably. Everything else is distraction.*