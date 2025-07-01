"""
Database models for Perquire.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Any, List, Optional
import numpy as np
import json


@dataclass
class InvestigationRecord:
    """Database record for investigation results."""
    
    investigation_id: str
    description: str
    final_similarity: float
    iterations: int
    start_time: datetime
    end_time: Optional[datetime] = None
    phase_reached: str = "exploration"
    convergence_achieved: bool = False
    convergence_reason: str = ""
    similarity_threshold: float = 0.90
    total_duration_seconds: Optional[float] = None
    strategy_name: str = "default"
    model_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "investigation_id": self.investigation_id,
            "description": self.description,
            "final_similarity": self.final_similarity,
            "iterations": self.iterations,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "phase_reached": self.phase_reached,
            "convergence_achieved": self.convergence_achieved,
            "convergence_reason": self.convergence_reason,
            "similarity_threshold": self.similarity_threshold,
            "total_duration_seconds": self.total_duration_seconds,
            "strategy_name": self.strategy_name,
            "model_config": json.dumps(self.model_config),
            "metadata": json.dumps(self.metadata),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InvestigationRecord":
        """Create from dictionary from database."""
        return cls(
            investigation_id=data["investigation_id"],
            description=data["description"],
            final_similarity=data["final_similarity"],
            iterations=data["iterations"],
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]) if data.get("end_time") else None,
            phase_reached=data.get("phase_reached", "exploration"),
            convergence_achieved=data.get("convergence_achieved", False),
            convergence_reason=data.get("convergence_reason", ""),
            similarity_threshold=data.get("similarity_threshold", 0.90),
            total_duration_seconds=data.get("total_duration_seconds"),
            strategy_name=data.get("strategy_name", "default"),
            model_config=json.loads(data.get("model_config", "{}")),
            metadata=json.loads(data.get("metadata", "{}")),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"])
        )


@dataclass
class EmbeddingRecord:
    """Database record for embeddings."""
    
    embedding_id: str
    text: str
    embedding_vector: np.ndarray
    dimensions: int
    model_name: str
    provider: str
    text_hash: str  # Hash of text for deduplication
    embedding_norm: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "embedding_id": self.embedding_id,
            "text": self.text,
            "embedding_vector": self.embedding_vector.tolist(),  # Store as list for DuckDB FLOAT array
            "dimensions": self.dimensions,
            "model_name": self.model_name,
            "provider": self.provider,
            "text_hash": self.text_hash,
            "embedding_norm": self.embedding_norm,
            "metadata": json.dumps(self.metadata),
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "EmbeddingRecord":
        """Create from dictionary from database."""
        # Reconstruct numpy array from list
        if isinstance(data["embedding_vector"], list):
            embedding_vector = np.array(data["embedding_vector"], dtype=np.float64)
        else:
            # Fallback for old blob format
            embedding_vector = np.frombuffer(data["embedding_vector"], dtype=np.float64)
        
        return cls(
            embedding_id=data["embedding_id"],
            text=data["text"],
            embedding_vector=embedding_vector,
            dimensions=data["dimensions"],
            model_name=data["model_name"],
            provider=data["provider"],
            text_hash=data["text_hash"],
            embedding_norm=data["embedding_norm"],
            metadata=json.loads(data.get("metadata", "{}")),
            created_at=datetime.fromisoformat(data["created_at"])
        )


@dataclass
class QuestionRecord:
    """Database record for investigation questions."""
    
    question_id: str
    investigation_id: str
    question: str
    similarity_score: float
    phase: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for database storage."""
        return {
            "question_id": self.question_id,
            "investigation_id": self.investigation_id,
            "question": self.question,
            "similarity_score": self.similarity_score,
            "phase": self.phase,
            "timestamp": self.timestamp.isoformat(),
            "metadata": json.dumps(self.metadata),
            "created_at": self.created_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "QuestionRecord":
        """Create from dictionary from database."""
        return cls(
            question_id=data["question_id"],
            investigation_id=data["investigation_id"],
            question=data["question"],
            similarity_score=data["similarity_score"],
            phase=data["phase"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            metadata=json.loads(data.get("metadata", "{}")),
            created_at=datetime.fromisoformat(data["created_at"])
        )


# SQL Schema definitions
INVESTIGATION_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS investigations (
    investigation_id VARCHAR PRIMARY KEY,
    description TEXT NOT NULL,
    final_similarity DOUBLE NOT NULL,
    iterations INTEGER NOT NULL,
    start_time TIMESTAMP NOT NULL,
    end_time TIMESTAMP,
    phase_reached VARCHAR DEFAULT 'exploration',
    convergence_achieved BOOLEAN DEFAULT FALSE,
    convergence_reason TEXT DEFAULT '',
    similarity_threshold DOUBLE DEFAULT 0.90,
    total_duration_seconds DOUBLE,
    strategy_name VARCHAR DEFAULT 'default',
    model_config TEXT DEFAULT '{}',
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

EMBEDDING_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    embedding_id VARCHAR PRIMARY KEY,
    text TEXT NOT NULL,
    embedding_vector FLOAT[768] NOT NULL,  -- Using DuckDB FLOAT array for VSS
    dimensions INTEGER NOT NULL,
    model_name VARCHAR NOT NULL,
    provider VARCHAR NOT NULL,
    text_hash VARCHAR NOT NULL,
    embedding_norm DOUBLE NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# VSS Index for fast vector similarity search
VSS_INDEX_SCHEMA = """
CREATE INDEX IF NOT EXISTS embeddings_vss_idx 
ON embeddings 
USING HNSW (embedding_vector) 
WITH (metric = 'cosine');
"""

QUESTION_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    question_id VARCHAR PRIMARY KEY,
    investigation_id VARCHAR NOT NULL,
    question TEXT NOT NULL,
    similarity_score DOUBLE NOT NULL,
    phase VARCHAR NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    metadata TEXT DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id)
);
"""

# Indexes for performance
INVESTIGATION_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_investigations_created_at ON investigations(created_at);",
    "CREATE INDEX IF NOT EXISTS idx_investigations_final_similarity ON investigations(final_similarity);",
    "CREATE INDEX IF NOT EXISTS idx_investigations_strategy ON investigations(strategy_name);",
    "CREATE INDEX IF NOT EXISTS idx_investigations_convergence ON investigations(convergence_achieved);"
]

EMBEDDING_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_embeddings_text_hash ON embeddings(text_hash);",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_model ON embeddings(model_name);",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_provider ON embeddings(provider);",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_dimensions ON embeddings(dimensions);",
    "CREATE INDEX IF NOT EXISTS idx_embeddings_created_at ON embeddings(created_at);"
]

QUESTION_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_questions_investigation_id ON questions(investigation_id);",
    "CREATE INDEX IF NOT EXISTS idx_questions_similarity_score ON questions(similarity_score);",
    "CREATE INDEX IF NOT EXISTS idx_questions_phase ON questions(phase);",
    "CREATE INDEX IF NOT EXISTS idx_questions_timestamp ON questions(timestamp);"
]

ALL_SCHEMAS = [
    INVESTIGATION_TABLE_SCHEMA,
    EMBEDDING_TABLE_SCHEMA, # This serves as persistent embedding cache
    QUESTION_TABLE_SCHEMA,
    # New Cache Table Schemas added below
]

# New Cache Table Schemas
SIMILARITY_CACHE_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS similarity_cache (
    query_hash VARCHAR NOT NULL,
    target_embedding_hash VARCHAR NOT NULL,
    similarity_score DOUBLE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (query_hash, target_embedding_hash)
);
"""

LLM_QUESTION_GEN_CACHE_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_question_gen_cache (
    input_hash VARCHAR PRIMARY KEY,
    generated_questions_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

LLM_SYNTHESIS_CACHE_TABLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_synthesis_cache (
    input_hash VARCHAR PRIMARY KEY,
    synthesized_description TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

# Add new schemas to ALL_SCHEMAS
ALL_SCHEMAS.extend([
    SIMILARITY_CACHE_TABLE_SCHEMA,
    LLM_QUESTION_GEN_CACHE_TABLE_SCHEMA,
    LLM_SYNTHESIS_CACHE_TABLE_SCHEMA
])


# Indexes for new cache tables
SIMILARITY_CACHE_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_similarity_cache_query_hash ON similarity_cache(query_hash);",
    "CREATE INDEX IF NOT EXISTS idx_similarity_cache_target_hash ON similarity_cache(target_embedding_hash);"
]
LLM_QUESTION_GEN_CACHE_INDEXES = [ # Renamed for consistency
    "CREATE INDEX IF NOT EXISTS idx_llm_question_gen_cache_input_hash ON llm_question_gen_cache(input_hash);"
]
LLM_SYNTHESIS_CACHE_INDEXES = [ # Renamed for consistency
    "CREATE INDEX IF NOT EXISTS idx_llm_synthesis_cache_input_hash ON llm_synthesis_cache(input_hash);"
]


# Include VSS index and new cache table indexes in ALL_INDEXES
ALL_INDEXES = (
    INVESTIGATION_INDEXES +
    EMBEDDING_INDEXES +
    QUESTION_INDEXES +
    [VSS_INDEX_SCHEMA] +
    SIMILARITY_CACHE_INDEXES +
    LLM_QUESTION_GEN_CACHE_INDEXES +
    LLM_SYNTHESIS_CACHE_INDEXES
)