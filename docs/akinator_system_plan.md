# Akinator-Style Investigation System Implementation Plan

**Status**: Planning Phase  
**Target**: Enhanced questioning strategy with Wikipedia bootstrap and dimensional analysis  
**Expected Impact**: 10x improvement in investigation accuracy and speed

## Executive Summary

This plan outlines the implementation of an intelligent "Akinator-style" investigation system that bootstraps understanding using Wikipedia knowledge and systematically explores dimensional axes through individual pole testing. The system will replace the current abstract questioning approach with a data-driven, hierarchical investigation strategy.

## Current Problem Analysis

### Issues with Current System
1. **Abstract Questions**: "What kind of abstract concepts does this relate to?" - too vague
2. **Low Discriminative Power**: Questions don't effectively narrow semantic space
3. **Poor Convergence**: Investigations plateau at low similarity scores (0.038 in live test)
4. **No Knowledge Bootstrap**: Starts from zero context instead of leveraging existing knowledge

### Target Improvements
- **Concrete Binary Questions**: "Is this typically found indoors?" vs "Is this typically found outdoors?"
- **Knowledge-Driven Bootstrap**: Start with closest Wikipedia concepts for context
- **Dimensional Analysis**: Systematic exploration of semantic dimensions
- **Individual Pole Testing**: Test each option separately, not compound questions

## Implementation Plan

### Phase 1: Wikipedia Knowledge Base Construction (Week 1)

#### 1.1 Dataset Collection
```python
# Target: 5,000-10,000 Wikipedia concepts across major categories
categories = {
    "physical_objects": ["Chair", "Table", "Car", ...],  # 500 concepts
    "places": ["Restaurant", "Coffee_shop", ...],        # 300 concepts  
    "activities": ["Cooking", "Reading", ...],           # 400 concepts
    "concepts": ["Love", "Happiness", ...],              # 300 concepts
    "science": ["Physics", "Chemistry", ...],            # 500 concepts
    "arts": ["Music", "Painting", ...],                  # 200 concepts
}
```

**Deliverables**:
- `WikipediaDatasetBuilder` class
- Curated dataset: `wikipedia_knowledge_base.json`
- Category-balanced concept coverage
- URL and metadata for each concept

#### 1.2 Batch Embedding Generation
```python
# Use batch APIs for efficiency
- Wikipedia concepts: ~5,000 embeddings
- Processing time: ~30 minutes with batch API
- Storage: Compressed JSON format (~50MB)
```

**Deliverables**:
- `BatchEmbeddingGenerator` class
- Optimized batch processing pipeline
- Compressed knowledge base: `perquire_knowledge_base.json.gz`

### Phase 2: Massive Question Generation (Week 1)

#### 2.1 LLM-Generated Question Bank
```python
# Single LLM call generates 200 question pairs
prompt_strategy = {
    "input": "20 diverse Wikipedia concepts",
    "output": "200 dimensional question pairs", 
    "format": "JSON with positive/negative poles",
    "total_target": "1,000 question pairs"
}
```

**Question Categories**:
- **Ontological**: Abstract vs Concrete, Living vs Non-living
- **Spatial**: Indoor vs Outdoor, Large vs Small, Near vs Far  
- **Temporal**: Brief vs Permanent, Historical vs Modern
- **Social**: Social vs Solitary, Public vs Private
- **Functional**: Practical vs Aesthetic, Simple vs Complex
- **Emotional**: Pleasant vs Unpleasant, Calming vs Energizing

**Deliverables**:
- `MassiveQuestionGenerator` class
- Question bank: `dimensional_question_bank.json`
- Categorized and tagged questions
- Batch embedding of all question pairs

#### 2.2 Question Validation
- Test question discriminative power on sample embeddings
- Remove low-quality or redundant questions
- Ensure balanced coverage across dimensions

### Phase 3: Bootstrap Investigation Engine (Week 2)

#### 3.1 Fast Bootstrap System
```python
class OptimizedBootstrapInvestigator:
    def fast_bootstrap(self, target_embedding):
        # Vectorized similarity computation
        # Return top-10 Wikipedia matches in <100ms
        
    def evaluate_dimensional_axis(self, axis):
        # Test positive pole: similarity_pos
        # Test negative pole: similarity_neg  
        # Position = (similarity_pos - similarity_neg)
        # Confidence = abs(position)
```

**Key Features**:
- **Vectorized Operations**: Numpy-based batch similarity computation
- **Pre-computed Embeddings**: No real-time embedding generation during investigation
- **Fast Lookup**: <100ms bootstrap, <50ms per dimensional test

#### 3.2 Dimensional Analysis Engine
```python
class DimensionalAnalyzer:
    def evaluate_all_dimensions(self, target_embedding):
        # Test all question pairs against target
        # Rank by discriminative power
        # Return dimensional profile
        
    def adaptive_refinement(self, unclear_dimensions):
        # Generate targeted questions for low-confidence dimensions
        # Use Wikipedia context for specific refinement
```

**Algorithm**:
1. **Bootstrap**: Find 10 closest Wikipedia concepts (context)
2. **Dimensional Testing**: Evaluate 50-100 most relevant question pairs  
3. **Ranking**: Sort dimensions by discriminative power
4. **Refinement**: Generate targeted questions for unclear dimensions
5. **Synthesis**: Combine dimensional profile into final description

### Phase 4: Integration and Optimization (Week 2)

#### 4.1 CLI Integration
```python
# Enhanced investigate command
perquire investigate embedding.npy --strategy akinator --verbose

# Output format:
# 🔍 Wikipedia Bootstrap: Coffee shop (0.89), Café (0.88), Restaurant (0.76)
# 📏 Dimensional Analysis:
#   atmosphere_warmth: +0.82 (warm, cozy)
#   social_context: +0.67 (social gathering place)  
#   commercial_type: +0.45 (commercial establishment)
# 🎯 Final: A warm, cozy social establishment like a coffee shop
```

#### 4.2 Performance Optimization
- **Memory Management**: Lazy loading of knowledge base
- **Caching**: LRU cache for computed similarities
- **Parallelization**: Batch processing for multiple investigations
- **Storage**: Efficient compressed format for knowledge base

### Phase 5: Validation and Testing (Week 3)

#### 5.1 Comparative Testing
```python
# Test against current system
test_cases = [
    "A cozy coffee shop on a rainy evening",
    "Machine learning optimization algorithms", 
    "The feeling of nostalgia",
    "A jazz musician in a dimly lit club"
]

metrics = {
    "accuracy": "LLM evaluation score (1-5)",
    "speed": "Investigation time (seconds)",
    "convergence": "Final similarity score", 
    "iterations": "Questions needed"
}
```

**Success Criteria**:
- **Accuracy**: >4.5/5 average LLM evaluation
- **Speed**: <5 seconds total investigation time
- **Convergence**: >0.7 final similarity scores
- **Efficiency**: <10 questions to convergence

#### 5.2 Live E2E Testing
- Update existing live E2E test to use Akinator system
- Compare side-by-side with current abstract questioning
- Document performance improvements

## Implementation Timeline

### Week 1: Knowledge Base Construction
- **Days 1-3**: Wikipedia dataset collection and curation
- **Days 4-5**: Batch embedding generation and storage
- **Days 6-7**: Question bank generation and validation

### Week 2: Investigation Engine
- **Days 1-3**: Bootstrap system implementation
- **Days 4-5**: Dimensional analysis engine
- **Days 6-7**: CLI integration and basic testing

### Week 3: Validation and Optimization
- **Days 1-3**: Comparative testing and benchmarking
- **Days 4-5**: Performance optimization
- **Days 6-7**: Documentation and final validation

## Technical Architecture

### Core Components

```
AkinatorInvestigator
├── WikipediaKnowledgeBase
│   ├── load_knowledge_base()
│   └── fast_bootstrap()
├── DimensionalAnalyzer  
│   ├── evaluate_all_dimensions()
│   └── adaptive_refinement()
├── MassiveQuestionGenerator
│   ├── generate_question_bank()
│   └── parse_dimensional_axes()
└── BatchEmbeddingGenerator
    ├── batch_embed_wikipedia()
    └── batch_embed_questions()
```

### Data Flow

```
Target Embedding
    ↓
Wikipedia Bootstrap (context)
    ↓
Dimensional Analysis (systematic testing)
    ↓  
Adaptive Refinement (targeted questions)
    ↓
Synthesis (final description)
```

### Storage Format

```json
{
  "metadata": {
    "created_at": "2025-01-01T00:00:00Z",
    "wikipedia_concepts": 5000,
    "dimensional_questions": 1000,
    "embedding_dimensions": 768
  },
  "wikipedia_concepts": [
    {
      "title": "Coffee shop",
      "category": "places", 
      "url": "https://en.wikipedia.org/wiki/Coffee_shop",
      "embedding": [0.1, 0.2, ...],
      "embedding_dim": 768
    }
  ],
  "dimensional_questions": [
    {
      "text": "This represents something abstract",
      "dimension": "abstraction_level",
      "pole": "positive",
      "category": "ontological", 
      "embedding": [0.3, 0.4, ...]
    }
  ]
}
```

## Expected Outcomes

### Performance Improvements
- **10x faster bootstrap**: Wikipedia context in <100ms vs multi-iteration discovery
- **3x higher accuracy**: Concrete questions vs abstract exploration  
- **5x better convergence**: Systematic dimensional analysis vs random questioning
- **2x fewer iterations**: Targeted refinement vs broad exploration

### User Experience Improvements
- **Transparent process**: Clear dimensional analysis output
- **Faster results**: Sub-5 second investigations
- **Higher confidence**: Better similarity scores and descriptions
- **Reproducible**: Consistent results across runs

### System Capabilities
- **Scalable knowledge**: Easy to add new Wikipedia concepts
- **Extensible questions**: LLM can generate domain-specific question sets
- **Multi-domain support**: Works across text, image, audio embeddings
- **Adaptive learning**: System improves with usage data

## Risk Mitigation

### Technical Risks
- **Knowledge Base Size**: Start with curated 5K concepts, expand gradually
- **Question Quality**: Validate with manual review and discriminative power testing
- **Memory Usage**: Implement lazy loading and compression
- **API Rate Limits**: Use batch APIs and caching extensively

### Performance Risks  
- **Bootstrap Speed**: Pre-compute all embeddings for instant lookup
- **Question Redundancy**: Remove low-quality questions through validation
- **Convergence Failure**: Maintain fallback to current system
- **Integration Issues**: Implement as optional strategy initially

## Success Metrics

### Quantitative Metrics
- **Investigation Speed**: <5 seconds (vs current ~7 seconds)
- **Accuracy Score**: >4.5/5 (vs current 4.0/5)
- **Final Similarity**: >0.7 (vs current 0.038)
- **Question Efficiency**: <10 questions (vs current 5 but low quality)

### Qualitative Metrics
- **Description Quality**: More specific and accurate descriptions
- **Process Transparency**: Users understand how conclusions were reached
- **System Confidence**: Higher similarity scores indicate better understanding
- **Reproducibility**: Consistent results across multiple runs

## Future Enhancements

### Phase 2 Extensions
- **Domain-Specific Knowledge**: Medical, legal, technical concept bases
- **Multi-Modal Support**: Image and audio Wikipedia content
- **Dynamic Question Generation**: Real-time question creation based on results
- **Active Learning**: System learns from user feedback to improve questions

### Advanced Features
- **Hierarchical Refinement**: Drill deeper into specific knowledge areas
- **Ensemble Investigation**: Multiple dimensional analysis approaches
- **Confidence Calibration**: Better uncertainty quantification
- **Interactive Debugging**: Users can explore dimensional analysis interactively

## Conclusion

This Akinator-style investigation system represents a fundamental improvement to Perquire's questioning strategy. By bootstrapping with Wikipedia knowledge and systematically testing dimensional axes, we expect to achieve significantly better accuracy, speed, and user satisfaction while maintaining the core "digital detective" philosophy that makes Perquire unique.

The implementation plan balances ambitious improvements with pragmatic engineering, ensuring deliverable milestones and measurable progress toward the goal of production-ready, intelligent embedding investigation.