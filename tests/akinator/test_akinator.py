import numpy as np
from perquire.akinator import (
    WikipediaDatasetBuilder,
    BatchEmbeddingGenerator,
    MassiveQuestionGenerator,
    OptimizedBootstrapInvestigator,
    DimensionalAnalyzer,
)


def test_dataset_builder_creates_concepts():
    builder = WikipediaDatasetBuilder()
    concepts = builder.build_dataset()
    assert len(concepts) >= 2
    assert concepts[0].title


def test_embedding_generator_adds_embeddings():
    builder = WikipediaDatasetBuilder()
    concepts = builder.build_dataset()
    generator = BatchEmbeddingGenerator(dimensions=4)
    generator.generate(concepts)
    assert all(c.embedding is not None for c in concepts)
    assert len(concepts[0].embedding) == 4


def test_question_generator_produces_questions():
    qgen = MassiveQuestionGenerator()
    questions = qgen.generate_question_bank()
    assert len(questions) >= 2
    assert questions[0].text


def test_bootstrap_and_dimensional_analysis():
    rng = np.random.default_rng(0)
    knowledge = rng.random((5, 4))
    target = rng.random(4)
    boot = OptimizedBootstrapInvestigator(knowledge)
    top = boot.fast_bootstrap(target, top_k=3)
    assert len(top) == 3
    analyzer = DimensionalAnalyzer(knowledge)
    sims = analyzer.evaluate_all_dimensions(target)
    assert sims.shape[0] == 5

