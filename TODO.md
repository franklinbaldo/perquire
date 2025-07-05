# Perquire Development TODO

This TODO list outlines planned tasks for the Perquire project, focusing on enhancements, new features, and maintenance. Tasks are prioritized from P0 (Critical) to P3 (Low).

## P0: Critical / Highest Priority

These tasks provide immediate high value, address core functionality, or are prerequisites for other important tasks.

- [ ] **Implement CLI Demo (`perquire demo --text "..."`)**
    - Create a simple CLI command that takes text, generates an embedding (e.g., using `all-MiniLM-L6-v2` or a configured default), and runs the investigation loop.
    - Utilize an in-memory DuckDB for this demo to minimize setup for new users.
    - *Related to: docs/plan.md suggestion #1*

- [ ] **Implement Pluggable Interrogator Interface**
    - Refactor the question generation logic in `PerquireInvestigator` (currently in `_generate_question`) to use a strategy pattern.
    - Define an `InterrogatorStrategy` interface.
    - Allow users to pass their own interrogator strategy instance to `PerquireInvestigator`.
    - *Related to: docs/plan.md suggestion #3*

- [ ] **Housekeeping: Review `DONE.md` & Archive Old Docs**
    - Review the existing `DONE.md` and the previous `TODO.md` (the one this file replaces).
    - Archive or clearly mark them as outdated to avoid confusion.
    - Ensure this `TODO.md` becomes the single source of truth for ongoing tasks.

- [ ] **Review and Clarify Licensing**
    - Examine question templates (e.g., in `QuestioningStrategy`) and any other assets potentially derived from LLM outputs.
    - Ensure the project's overall licensing (MIT) is consistent and clear, addressing any sub-licensing needs for generated content.
    - *Related to: docs/plan.md suggestion #9*

- [ ] **Add Test Coverage for `src/perquire/database/duckdb_provider.py`**
    - Write dedicated unit tests for the caching logic (get/set for embeddings, similarities, LLM generations) in `duckdb_provider.py`.

## P1: High Priority

These tasks offer significant improvements or address important aspects of the project.

- [ ] **Develop Public Benchmark Suite**
    - Select 1-2 suitable open datasets (e.g., STS-Benchmark, MS MARCO dev).
    - Create scripts/notebooks in `benchmarks/` to run Perquire on these datasets.
    - Define and measure metrics like inversion accuracy, question count, and resource usage.
    - Output results in a clear, shareable format (tables/charts).
    - *Related to: docs/plan.md suggestion #2*

- [ ] **Implement Privacy-Risk Scoring Feature**
    - Add functionality to compare Perquire's output description against ground-truth text (if available for an embedding).
    - Use metrics like BLEU, ROUGE, or embedding similarity for this comparison.
    - Provide an option to flag or score "leak risk" based on the comparison.
    - *Related to: docs/plan.md suggestion #4*

- [ ] **Clarify & Complete Embedding Provider Implementations**
    - Review `src/perquire/embeddings/gemini_embeddings.py` and `src/perquire/embeddings/openai_embeddings.py`. Determine their exact role (e.g., generating embeddings via these services vs. LLM tasks).
    - Ensure they are fully implemented and tested if they are intended as primary embedding providers alongside sentence-transformers.
    - Consider adding other relevant embedding provider integrations (e.g., Cohere, direct Hugging Face Hub models).

- [ ] **Verify & Test `EnsembleInvestigator`**
    - Review the implementation of `EnsembleInvestigator` in `src/perquire/core/ensemble.py`.
    - Ensure it is fully functional, with comprehensive tests.
    - *Mentioned in README.md and POST-V1 BACKLOG*

- [ ] **Set Up Documentation Generation & Hosting**
    - Configure Sphinx with autodoc (or similar tool) to generate documentation from docstrings.
    - Publish the documentation to ReadTheDocs or GitHub Pages.
    - Ensure the link `https://perquire.readthedocs.io` (from `README.md`) points to the live documentation.

- [ ] **Enhance CLI Functionality**
    - Review existing CLI commands (`providers`, `investigate`, `batch`, `status`) for usability, features, and consistency.
    - Add a `--json` output option for all relevant commands to support scriptability.

## P2: Medium Priority

Valuable additions that can follow P0 and P1 tasks.

- [ ] **Create REST Microservice for Perquire**
    - Develop a simple FastAPI application.
    - Expose an `/invert` endpoint: input an embedding (or text to be embedded), returns investigation result.
    - Consider a `/probe` endpoint for asking a single question against an embedding or similar focused interactions.
    - *Related to: docs/plan.md suggestion #5*

- [ ] **Flesh Out Benchmarking Scripts in `benchmarks/`**
    - Improve `benchmarks/benchmark.py` and `benchmarks/simple_benchmark.py`.
    - Ensure these scripts are robust and usable for the Public Benchmark Suite task (P1).

- [ ] **Systematic Error Handling and Resilience Pass**
    - Conduct a thorough review of the codebase to identify areas where error handling can be improved.
    - Focus on API calls, file I/O, unexpected data formats, and external service interactions.
    - Ensure graceful failure modes and informative error messages.

- [ ] **Determine Fate of `src/perquire/cache/` Directory**
    - The directory `src/perquire/cache/` currently only contains `__init__.py`.
    - Confirm if the current database-driven caching in `PerquireInvestigator` and LRU caching in providers is sufficient.
    - If `src/perquire/cache/` is not needed for distinct functionality, remove it.

- [ ] **Create Public Roadmap (`ROADMAP.md`)**
    - Develop a `ROADMAP.md` file.
    - Outline planned features, milestones (e.g., v0.2, v0.3, v1.0), and general project direction.
    - *Related to: docs/plan.md suggestion #10*

## P3: Low Priority

Nice-to-haves or more advanced features for later stages.

- [ ] **Add Interactive Visualizations (Plotly)**
    - For Jupyter notebook environments, enhance or create interactive visualizations.
    - Examples: Plot similarity scores over iterations, display cosine similarity heatmaps.
    - *Related to: docs/plan.md suggestion #7*

- [ ] **Review and Refine Configuration Handling**
    - The `docs/plan.md` mentioned a potential `configs/` folder (not observed in current listings). If it exists and is problematic, address it.
    - Generally, review how configuration is loaded and managed. Ensure defaults are sensible and override mechanisms (env vars, constructor args) are clear and well-documented.
    - *Related to: docs/plan.md suggestion #6*

- [ ] **Incrementally Improve MyPy Strictness**
    - Work towards stricter MyPy compliance.
    - Add type stubs or refine type hints for dependencies to reduce `--ignore-missing-imports`.
    - Resolve type redefinitions flagged by `--allow-redefinition`.

- [ ] **Investigate Model Drift Alerts (Advanced Feature)**
    - Explore feasibility of a system to monitor production vectors and detect distribution drift compared to the embedding model's baseline.
    - This is a longer-term research/advanced feature.
    - *Related to: docs/plan.md suggestion #8*

- [ ] **Review Need for Separate `BatchInvestigator` Class**
    - `PerquireInvestigator` has an `investigate_batch` method.
    - Evaluate if a separate `BatchInvestigator` class (mentioned in POST-V1 BACKLOG) offers significant advantages or if the existing method is sufficient.

---
_This TODO list is a living document and will be updated as the project progresses._
