"""
DuckDB database provider implementation with VSS vector search.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
import uuid

import duckdb
import numpy as np
import pandas as pd

from .base import BaseDatabaseProvider, DatabaseConfig, DatabaseError
from .models import (
    InvestigationRecord, EmbeddingRecord, QuestionRecord,
    ALL_SCHEMAS, ALL_INDEXES
)
from ..embeddings.utils import cosine_similarity

logger = logging.getLogger(__name__)


class DuckDBProvider(BaseDatabaseProvider):
    """
    DuckDB database provider implementation with VSS vector search.
    
    This provider uses DuckDB with the VSS (Vector Similarity Search) extension
    for fast, analytical queries and optimized vector operations with HNSW indexing.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize DuckDB provider.
        
        Args:
            config: Database configuration
        """
        super().__init__(config)
        self._connection = None
        
    def validate_config(self) -> None:
        """Validate DuckDB configuration."""
        if not self.config.connection_string:
            raise DatabaseError("Connection string is required for DuckDB")
        
        # Check if directory exists for file-based databases
        if self.config.connection_string != ":memory:":
            db_path = Path(self.config.connection_string)
            db_path.parent.mkdir(parents=True, exist_ok=True)
    
    def connect(self) -> None:
        """Establish DuckDB connection with VSS extension."""
        try:
            self._connection = duckdb.connect(self.config.connection_string)
            
            # Install and load VSS extension for vector similarity search
            self._connection.execute("INSTALL vss;")
            self._connection.execute("LOAD vss;")
            
            # Set DuckDB configuration for optimal performance
            self._connection.execute("SET enable_progress_bar = false;")
            self._connection.execute("SET threads = 4;")
            self._connection.execute("SET memory_limit = '2GB';")
            
            if self.config.create_tables:
                self.create_tables()
            
            logger.info(f"Connected to DuckDB database with VSS extension: {self.config.connection_string}")
            
        except Exception as e:
            raise DatabaseError(f"Failed to connect to DuckDB: {str(e)}")
    
    def disconnect(self) -> None:
        """Close DuckDB connection."""
        if self._connection:
            try:
                self._connection.close()
                self._connection = None
                logger.info("Disconnected from DuckDB database")
            except Exception as e:
                logger.warning(f"Error disconnecting from DuckDB: {str(e)}")
    
    def create_tables(self) -> None:
        """Create necessary database tables and VSS indexes."""
        try:
            # Create tables
            for schema in ALL_SCHEMAS:
                self._connection.execute(schema)
            
            # Create indexes (including VSS index)
            for index in ALL_INDEXES:
                try:
                    self._connection.execute(index)
                except Exception as e:
                    # VSS index creation might fail if table is empty
                    if "HNSW" in index:
                        logger.warning(f"VSS index creation deferred (will be created when embeddings are added): {str(e)}")
                    else:
                        raise
            
            logger.info("Created DuckDB tables and indexes")
            
        except Exception as e:
            raise DatabaseError(f"Failed to create DuckDB tables: {str(e)}")
    
    def is_connected(self) -> bool:
        """Check if DuckDB is connected."""
        try:
            if self._connection is None:
                return False
            
            # Try a simple query
            self._connection.execute("SELECT 1").fetchone()
            return True
            
        except Exception:
            return False
    
    def save_embedding(
        self, 
        text: str, 
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> str:
        """Save embedding to DuckDB with VSS support."""
        try:
            # Generate text hash for deduplication
            text_hash = hashlib.sha256(text.encode()).hexdigest()
            
            # Check if embedding already exists
            existing = self._connection.execute(
                "SELECT embedding_id FROM embeddings WHERE text_hash = ?",
                [text_hash]
            ).fetchone()
            
            if existing:
                return existing[0]
            
            # Ensure embedding has correct dimensions (768 for Gemini)
            if len(embedding) != 768:
                # Pad or truncate to 768 dimensions
                if len(embedding) < 768:
                    padded_embedding = np.zeros(768)
                    padded_embedding[:len(embedding)] = embedding
                    embedding = padded_embedding
                else:
                    embedding = embedding[:768]
                
                logger.warning(f"Adjusted embedding dimensions to 768 from {len(embedding)}")
            
            # Create embedding record
            record = EmbeddingRecord(
                embedding_id=str(uuid.uuid4()),
                text=text,
                embedding_vector=embedding.astype(np.float32),  # Use float32 for efficiency
                dimensions=len(embedding),
                model_name=metadata.get("model", "unknown"),
                provider=metadata.get("provider", "unknown"),
                text_hash=text_hash,
                embedding_norm=float(np.linalg.norm(embedding)),
                metadata=metadata
            )
            
            # Convert to dict for insertion
            data = record.to_dict()
            
            # Insert embedding
            columns = ", ".join(data.keys())
            placeholders = ", ".join("?" * len(data))
            query = f"INSERT INTO embeddings ({columns}) VALUES ({placeholders})"
            
            self._connection.execute(query, list(data.values()))
            
            # Try to create VSS index if it doesn't exist yet
            try:
                self._connection.execute("""
                    CREATE INDEX IF NOT EXISTS embeddings_vss_idx 
                    ON embeddings 
                    USING HNSW (embedding_vector) 
                    WITH (metric = 'cosine');
                """)
            except Exception as idx_e:
                logger.debug(f"VSS index creation skipped: {str(idx_e)}")
            
            logger.debug(f"Saved embedding: {record.embedding_id}")
            return record.embedding_id
            
        except Exception as e:
            raise DatabaseError(f"Failed to save embedding: {str(e)}")
    
    def search_embeddings(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings using DuckDB VSS extension."""
        try:
            # Ensure query embedding has correct dimensions
            if len(query_embedding) != 768:
                if len(query_embedding) < 768:
                    padded_embedding = np.zeros(768)
                    padded_embedding[:len(query_embedding)] = query_embedding
                    query_embedding = padded_embedding
                else:
                    query_embedding = query_embedding[:768]
            
            # Convert query embedding to list for DuckDB
            query_vector = query_embedding.astype(np.float32).tolist()
            
            # Use VSS extension for fast cosine similarity search
            query = """
            SELECT embedding_id, text, model_name, provider, embedding_norm, 
                   metadata, created_at,
                   array_cosine_similarity(embedding_vector, ?::FLOAT[768]) as similarity_score
            FROM embeddings
            WHERE dimensions = 768
            AND array_cosine_similarity(embedding_vector, ?::FLOAT[768]) >= ?
            ORDER BY similarity_score DESC
            LIMIT ?
            """
            
            results = self._connection.execute(
                query, 
                [query_vector, query_vector, similarity_threshold, limit]
            ).fetchall()
            
            columns = [desc[0] for desc in self._connection.description]
            similar_embeddings = [dict(zip(columns, row)) for row in results]
            
            return similar_embeddings
            
        except Exception as e:
            # Fallback to manual similarity calculation if VSS fails
            logger.warning(f"VSS search failed, falling back to manual calculation: {str(e)}")
            return self._search_embeddings_fallback(query_embedding, limit, similarity_threshold)
    
    def _search_embeddings_fallback(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Fallback embedding search using manual cosine similarity calculation."""
        try:
            # Get all embeddings
            query = """
            SELECT embedding_id, text, embedding_vector, model_name, provider, 
                   embedding_norm, metadata, created_at
            FROM embeddings
            WHERE dimensions = ?
            """
            
            results = self._connection.execute(query, [len(query_embedding)]).fetchall()
            columns = [desc[0] for desc in self._connection.description]
            
            # Calculate similarities manually
            similar_embeddings = []
            
            for row in results:
                data = dict(zip(columns, row))
                
                # Reconstruct embedding vector from list
                if isinstance(data["embedding_vector"], list):
                    stored_embedding = np.array(data["embedding_vector"], dtype=np.float32)
                else:
                    # Fallback for old blob format
                    stored_embedding = np.frombuffer(data["embedding_vector"], dtype=np.float32)
                
                # Calculate similarity
                similarity = cosine_similarity(query_embedding, stored_embedding)
                
                if similarity >= similarity_threshold:
                    data["similarity_score"] = similarity
                    data.pop("embedding_vector")  # Remove vector data for response
                    similar_embeddings.append(data)
            
            # Sort by similarity and limit results
            similar_embeddings.sort(key=lambda x: x["similarity_score"], reverse=True)
            return similar_embeddings[:limit]
            
        except Exception as e:
            raise DatabaseError(f"Failed to search embeddings: {str(e)}")
    
    def get_embedding_statistics(self) -> Dict[str, Any]:
        """Get statistics about embeddings in the database."""
        try:
            stats = {}
            
            # Total embeddings
            stats["total_embeddings"] = self._connection.execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()[0]
            
            # Provider distribution
            provider_stats = self._connection.execute("""
                SELECT provider, COUNT(*) as count 
                FROM embeddings 
                GROUP BY provider
            """).fetchall()
            stats["provider_distribution"] = dict(provider_stats)
            
            # Model distribution
            model_stats = self._connection.execute("""
                SELECT model_name, COUNT(*) as count 
                FROM embeddings 
                GROUP BY model_name
            """).fetchall()
            stats["model_distribution"] = dict(model_stats)
            
            # Average embedding norm
            avg_norm = self._connection.execute(
                "SELECT AVG(embedding_norm) FROM embeddings"
            ).fetchone()[0]
            stats["average_embedding_norm"] = float(avg_norm) if avg_norm else 0.0
            
            return stats
            
        except Exception as e:
            raise DatabaseError(f"Failed to get embedding statistics: {str(e)}")
    
    def create_vss_index_if_needed(self) -> bool:
        """Create VSS index if embeddings table has data."""
        try:
            # Check if we have embeddings
            count = self._connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
            
            if count > 0:
                # Try to create VSS index
                self._connection.execute("""
                    CREATE INDEX IF NOT EXISTS embeddings_vss_idx 
                    ON embeddings 
                    USING HNSW (embedding_vector) 
                    WITH (metric = 'cosine');
                """)
                logger.info("Created VSS index for embeddings")
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Failed to create VSS index: {str(e)}")
            return False
    
    def save_investigation(self, investigation_data: Dict[str, Any]) -> str:
        """Save investigation result to DuckDB."""
        try:
            # Create investigation record
            record = InvestigationRecord(
                investigation_id=investigation_data.get("investigation_id", str(uuid.uuid4())),
                description=investigation_data["description"],
                final_similarity=investigation_data["final_similarity"],
                iterations=investigation_data["iterations"],
                start_time=datetime.fromisoformat(investigation_data["start_time"]),
                end_time=datetime.fromisoformat(investigation_data["end_time"]) if investigation_data.get("end_time") else None,
                phase_reached=investigation_data.get("phase_reached", "exploration"),
                convergence_achieved=investigation_data.get("convergence_achieved", False),
                convergence_reason=investigation_data.get("convergence_reason", ""),
                similarity_threshold=investigation_data.get("similarity_threshold", 0.90),
                total_duration_seconds=investigation_data.get("total_duration_seconds"),
                strategy_name=investigation_data.get("strategy_name", "default"),
                model_config=investigation_data.get("model_config", {}),
                metadata=investigation_data.get("metadata", {})
            )
            
            # Convert to dict for insertion
            data = record.to_dict()
            
            # Insert investigation
            columns = ", ".join(data.keys())
            placeholders = ", ".join("?" * len(data))
            query = f"INSERT OR REPLACE INTO investigations ({columns}) VALUES ({placeholders})"
            
            self._connection.execute(query, list(data.values()))
            
            # Save question history if provided
            if "question_history" in investigation_data:
                for question_data in investigation_data["question_history"]:
                    self.save_question_result(
                        investigation_id=record.investigation_id,
                        question=question_data["question"],
                        similarity_score=question_data["similarity_score"],
                        phase=question_data["phase"],
                        metadata=question_data.get("metadata", {})
                    )
            
            logger.info(f"Saved investigation: {record.investigation_id}")
            return record.investigation_id
            
        except Exception as e:
            raise DatabaseError(f"Failed to save investigation: {str(e)}")
    
    def load_investigation(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """Load investigation result from DuckDB."""
        try:
            # Load investigation
            query = "SELECT * FROM investigations WHERE investigation_id = ?"
            result = self._connection.execute(query, [investigation_id]).fetchone()
            
            if not result:
                return None
            
            # Convert to dict
            columns = [desc[0] for desc in self._connection.description]
            investigation_data = dict(zip(columns, result))
            
            # Load question history
            questions = self.get_investigation_questions(investigation_id)
            investigation_data["question_history"] = questions
            
            return investigation_data
            
        except Exception as e:
            raise DatabaseError(f"Failed to load investigation: {str(e)}")
    
    def list_investigations(
        self, 
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """List investigations with optional filtering."""
        try:
            query = "SELECT * FROM investigations"
            params = []
            
            if filters:
                conditions = []
                for key, value in filters.items():
                    if key == "min_similarity":
                        conditions.append("final_similarity >= ?")
                        params.append(value)
                    elif key == "strategy":
                        conditions.append("strategy_name = ?")
                        params.append(value)
                    elif key == "convergence":
                        conditions.append("convergence_achieved = ?")
                        params.append(value)
                    elif key == "date_from":
                        conditions.append("created_at >= ?")
                        params.append(value)
                    elif key == "date_to":
                        conditions.append("created_at <= ?")
                        params.append(value)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
            
            query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
            params.extend([limit, offset])
            
            results = self._connection.execute(query, params).fetchall()
            columns = [desc[0] for desc in self._connection.description]
            
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            raise DatabaseError(f"Failed to list investigations: {str(e)}")
    
    def delete_investigation(self, investigation_id: str) -> bool:
        """Delete investigation from DuckDB."""
        try:
            # Delete questions first (foreign key constraint)
            self._connection.execute(
                "DELETE FROM questions WHERE investigation_id = ?",
                [investigation_id]
            )
            
            # Delete investigation
            result = self._connection.execute(
                "DELETE FROM investigations WHERE investigation_id = ?",
                [investigation_id]
            )
            
            deleted = self._connection.execute("SELECT changes()").fetchone()[0] > 0
            
            if deleted:
                logger.info(f"Deleted investigation: {investigation_id}")
            
            return deleted
            
        except Exception as e:
            raise DatabaseError(f"Failed to delete investigation: {str(e)}")
    
    def load_embedding(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """Load embedding from DuckDB."""
        try:
            query = "SELECT * FROM embeddings WHERE embedding_id = ?"
            result = self._connection.execute(query, [embedding_id]).fetchone()
            
            if not result:
                return None
            
            columns = [desc[0] for desc in self._connection.description]
            return dict(zip(columns, result))
            
        except Exception as e:
            raise DatabaseError(f"Failed to load embedding: {str(e)}")
    
    def save_question_result(
        self,
        investigation_id: str,
        question: str,
        similarity_score: float,
        phase: str,
        metadata: Dict[str, Any]
    ) -> str:
        """Save question result to DuckDB."""
        try:
            record = QuestionRecord(
                question_id=str(uuid.uuid4()),
                investigation_id=investigation_id,
                question=question,
                similarity_score=similarity_score,
                phase=phase,
                timestamp=datetime.now(),
                metadata=metadata
            )
            
            data = record.to_dict()
            
            columns = ", ".join(data.keys())
            placeholders = ", ".join("?" * len(data))
            query = f"INSERT INTO questions ({columns}) VALUES ({placeholders})"
            
            self._connection.execute(query, list(data.values()))
            
            return record.question_id
            
        except Exception as e:
            raise DatabaseError(f"Failed to save question result: {str(e)}")
    
    def get_investigation_questions(self, investigation_id: str) -> List[Dict[str, Any]]:
        """Get all questions for an investigation."""
        try:
            query = """
            SELECT * FROM questions 
            WHERE investigation_id = ? 
            ORDER BY timestamp ASC
            """
            
            results = self._connection.execute(query, [investigation_id]).fetchall()
            columns = [desc[0] for desc in self._connection.description]
            
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            raise DatabaseError(f"Failed to get investigation questions: {str(e)}")
    
    def get_investigation_stats(self) -> Dict[str, Any]:
        """Get investigation statistics."""
        try:
            stats = {}
            
            # Total investigations
            stats["total_investigations"] = self._connection.execute(
                "SELECT COUNT(*) FROM investigations"
            ).fetchone()[0]
            
            # Converged investigations
            stats["converged_investigations"] = self._connection.execute(
                "SELECT COUNT(*) FROM investigations WHERE convergence_achieved = true"
            ).fetchone()[0]
            
            # Average similarity
            avg_similarity = self._connection.execute(
                "SELECT AVG(final_similarity) FROM investigations"
            ).fetchone()[0]
            stats["average_similarity"] = float(avg_similarity) if avg_similarity else 0.0
            
            # Average iterations
            avg_iterations = self._connection.execute(
                "SELECT AVG(iterations) FROM investigations"
            ).fetchone()[0]
            stats["average_iterations"] = float(avg_iterations) if avg_iterations else 0.0
            
            # Strategy distribution
            strategy_stats = self._connection.execute("""
                SELECT strategy_name, COUNT(*) as count 
                FROM investigations 
                GROUP BY strategy_name
            """).fetchall()
            stats["strategy_distribution"] = dict(strategy_stats)
            
            # Total embeddings
            stats["total_embeddings"] = self._connection.execute(
                "SELECT COUNT(*) FROM embeddings"
            ).fetchone()[0]
            
            # Total questions
            stats["total_questions"] = self._connection.execute(
                "SELECT COUNT(*) FROM questions"
            ).fetchone()[0]
            
            return stats
            
        except Exception as e:
            raise DatabaseError(f"Failed to get investigation stats: {str(e)}")
    
    def get_top_questions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get top-performing questions across all investigations."""
        try:
            query = """
            SELECT question, phase, AVG(similarity_score) as avg_similarity,
                   COUNT(*) as usage_count, MAX(similarity_score) as max_similarity
            FROM questions 
            GROUP BY question, phase
            HAVING COUNT(*) >= 2
            ORDER BY avg_similarity DESC, usage_count DESC
            LIMIT ?
            """
            
            results = self._connection.execute(query, [limit]).fetchall()
            columns = [desc[0] for desc in self._connection.description]
            
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            raise DatabaseError(f"Failed to get top questions: {str(e)}")
    
    def optimize_database(self) -> None:
        """Optimize DuckDB performance."""
        try:
            # Analyze tables for query optimization
            self._connection.execute("ANALYZE;")
            
            # Checkpoint if it's a persistent database
            if self.config.connection_string != ":memory:":
                self._connection.execute("CHECKPOINT;")
            
            # Try to create VSS index if needed
            self.create_vss_index_if_needed()
            
            logger.info("Optimized DuckDB database")
            
        except Exception as e:
            raise DatabaseError(f"Failed to optimize database: {str(e)}")
    
    def backup_database(self, backup_path: str) -> None:
        """Create DuckDB backup."""
        try:
            if self.config.connection_string == ":memory:":
                raise DatabaseError("Cannot backup in-memory database")
            
            backup_path = Path(backup_path)
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Export to backup file
            self._connection.execute(f"EXPORT DATABASE '{backup_path}';")
            
            logger.info(f"Created database backup: {backup_path}")
            
        except Exception as e:
            raise DatabaseError(f"Failed to backup database: {str(e)}")
    
    def get_database_info(self) -> Dict[str, Any]:
        """Get DuckDB information and statistics."""
        try:
            info = {
                "provider": "duckdb",
                "connection_string": self.config.connection_string,
                "version": duckdb.__version__,
                "connected": self.is_connected(),
                "vss_enabled": True
            }
            
            if self.is_connected():
                # Table sizes
                tables = ["investigations", "embeddings", "questions"]
                for table in tables:
                    try:
                        count = self._connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        info[f"{table}_count"] = count
                    except:
                        info[f"{table}_count"] = 0
                
                # Database size (for file-based databases)
                if self.config.connection_string != ":memory:":
                    try:
                        db_path = Path(self.config.connection_string)
                        if db_path.exists():
                            info["database_size_bytes"] = db_path.stat().st_size
                    except:
                        pass
                
                # VSS extension info
                try:
                    vss_info = self._connection.execute("PRAGMA show_all_tables;").fetchall()
                    info["vss_tables"] = len([t for t in vss_info if "vss" in str(t).lower()])
                except:
                    pass
            
            return info
            
        except Exception as e:
            raise DatabaseError(f"Failed to get database info: {str(e)}")
    
    def execute_raw_query(self, query: str, params: Optional[List] = None) -> List[Dict[str, Any]]:
        """Execute raw SQL query (for advanced usage)."""
        try:
            if params:
                results = self._connection.execute(query, params).fetchall()
            else:
                results = self._connection.execute(query).fetchall()
            
            columns = [desc[0] for desc in self._connection.description]
            return [dict(zip(columns, row)) for row in results]
            
        except Exception as e:
            raise DatabaseError(f"Failed to execute query: {str(e)}")

    # --- Caching Specific Methods ---

    def _generate_hash(self, data: Union[str, Dict, List]) -> str:
        """Generate SHA256 hash for given data."""
        if isinstance(data, str):
            return hashlib.sha256(data.encode('utf-8')).hexdigest()
        else: # For dicts, lists (e.g. LLM inputs)
            return hashlib.sha256(json.dumps(data, sort_keys=True).encode('utf-8')).hexdigest()

    # Embedding Cache (uses existing 'embeddings' table)
    def get_cached_embedding(self, text_content: str) -> Optional[np.ndarray]:
        """Retrieve a cached embedding by its text content."""
        text_hash = self._generate_hash(text_content)
        try:
            result = self._connection.execute(
                "SELECT embedding_vector FROM embeddings WHERE text_hash = ?",
                [text_hash]
            ).fetchone()
            if result and result[0] is not None:
                # DuckDB stores FLOAT[768] as list of floats
                return np.array(result[0], dtype=np.float32)
            return None
        except Exception as e:
            logger.error(f"Error getting cached embedding for hash {text_hash}: {e}")
            return None # On error, treat as cache miss

    def set_cached_embedding(self, text_content: str, embedding: np.ndarray,
                             model_name: str, provider_name: str,
                             dimensions: int, embedding_norm: float,
                             metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Saves an embedding to the cache (existing 'embeddings' table).
        This is essentially a wrapper around save_embedding, ensuring parameters match.
        """
        full_metadata = metadata or {}
        full_metadata.update({"model": model_name, "provider": provider_name})

        # save_embedding already handles hashing and deduplication
        return self.save_embedding(text=text_content, embedding=embedding, metadata=full_metadata)

    # Similarity Score Cache
    def get_cached_similarity(self, question_text: str, target_embedding_identifier: str) -> Optional[float]:
        """Retrieve a cached similarity score."""
        query_hash = self._generate_hash(question_text)
        # target_embedding_identifier could be a hash of the target embedding array or a unique ID
        target_hash = self._generate_hash(target_embedding_identifier)
        try:
            result = self._connection.execute(
                "SELECT similarity_score FROM similarity_cache WHERE query_hash = ? AND target_embedding_hash = ?",
                [query_hash, target_hash]
            ).fetchone()
            return float(result[0]) if result else None
        except Exception as e:
            logger.error(f"Error getting cached similarity: {e}")
            return None

    def set_cached_similarity(self, question_text: str, target_embedding_identifier: str, score: float):
        """Cache a similarity score."""
        query_hash = self._generate_hash(question_text)
        target_hash = self._generate_hash(target_embedding_identifier)
        try:
            self._connection.execute(
                "INSERT OR REPLACE INTO similarity_cache (query_hash, target_embedding_hash, similarity_score, created_at) VALUES (?, ?, ?, ?)",
                [query_hash, target_hash, score, datetime.now().isoformat()]
            )
        except Exception as e:
            logger.error(f"Error setting cached similarity: {e}")

    # LLM Question Generation Cache
    def get_cached_llm_question_gen(self, input_data: Dict[str, Any]) -> Optional[List[str]]:
        """Retrieve cached generated questions for LLM input."""
        input_hash = self._generate_hash(input_data)
        try:
            result = self._connection.execute(
                "SELECT generated_questions_json FROM llm_question_gen_cache WHERE input_hash = ?",
                [input_hash]
            ).fetchone()
            return json.loads(result[0]) if result else None
        except Exception as e:
            logger.error(f"Error getting cached LLM question gen: {e}")
            return None

    def set_cached_llm_question_gen(self, input_data: Dict[str, Any], questions: List[str]):
        """Cache generated questions from LLM."""
        input_hash = self._generate_hash(input_data)
        questions_json = json.dumps(questions)
        try:
            self._connection.execute(
                "INSERT OR REPLACE INTO llm_question_gen_cache (input_hash, generated_questions_json, created_at) VALUES (?, ?, ?)",
                [input_hash, questions_json, datetime.now().isoformat()]
            )
        except Exception as e:
            logger.error(f"Error setting cached LLM question gen: {e}")

    # LLM Synthesis Cache
    def get_cached_llm_synthesis(self, input_data: Dict[str, Any]) -> Optional[str]:
        """Retrieve cached synthesized description from LLM."""
        input_hash = self._generate_hash(input_data)
        try:
            result = self._connection.execute(
                "SELECT synthesized_description FROM llm_synthesis_cache WHERE input_hash = ?",
                [input_hash]
            ).fetchone()
            return result[0] if result else None
        except Exception as e:
            logger.error(f"Error getting cached LLM synthesis: {e}")
            return None

    def set_cached_llm_synthesis(self, input_data: Dict[str, Any], description: str):
        """Cache synthesized description from LLM."""
        input_hash = self._generate_hash(input_data)
        try:
            self._connection.execute(
                "INSERT OR REPLACE INTO llm_synthesis_cache (input_hash, synthesized_description, created_at) VALUES (?, ?, ?)",
                [input_hash, description, datetime.now().isoformat()]
            )
        except Exception as e:
            logger.error(f"Error setting cached LLM synthesis: {e}")