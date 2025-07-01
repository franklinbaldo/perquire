"""
Base database provider interface.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
from datetime import datetime
import numpy as np

from ..exceptions import PerquireException


class DatabaseError(PerquireException):
    """Raised when database operations fail."""
    pass


@dataclass
class DatabaseConfig:
    """Database configuration."""
    connection_string: str
    create_tables: bool = True
    auto_vacuum: bool = True
    checkpoint_interval: int = 1000
    metadata: Dict[str, Any] = None


class BaseDatabaseProvider(ABC):
    """
    Abstract base class for all database providers.
    
    This class defines the interface that all database providers must implement
    to work with Perquire's data persistence system.
    """
    
    def __init__(self, config: DatabaseConfig):
        """
        Initialize the database provider with configuration.
        
        Args:
            config: Database configuration object
        """
        self.config = config
        self._connection = None
        self.validate_config()
    
    @abstractmethod
    def validate_config(self) -> None:
        """
        Validate the database configuration.
        
        Raises:
            DatabaseError: If configuration is invalid
        """
        pass
    
    @abstractmethod
    def connect(self) -> None:
        """
        Establish database connection.
        
        Raises:
            DatabaseError: If connection fails
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> None:
        """Close database connection."""
        pass
    
    @abstractmethod
    def create_tables(self) -> None:
        """
        Create necessary database tables.
        
        Raises:
            DatabaseError: If table creation fails
        """
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """
        Check if database is connected.
        
        Returns:
            True if connected, False otherwise
        """
        pass
    
    # Investigation operations
    @abstractmethod
    def save_investigation(self, investigation_data: Dict[str, Any]) -> str:
        """
        Save investigation result to database.
        
        Args:
            investigation_data: Investigation data dictionary
            
        Returns:
            Investigation ID
            
        Raises:
            DatabaseError: If save operation fails
        """
        pass
    
    @abstractmethod
    def load_investigation(self, investigation_id: str) -> Optional[Dict[str, Any]]:
        """
        Load investigation result from database.
        
        Args:
            investigation_id: Investigation ID
            
        Returns:
            Investigation data dictionary or None if not found
            
        Raises:
            DatabaseError: If load operation fails
        """
        pass
    
    @abstractmethod
    def list_investigations(
        self, 
        limit: int = 100,
        offset: int = 0,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        List investigations with optional filtering.
        
        Args:
            limit: Maximum number of results
            offset: Number of results to skip
            filters: Optional filters to apply
            
        Returns:
            List of investigation summaries
            
        Raises:
            DatabaseError: If list operation fails
        """
        pass
    
    @abstractmethod
    def delete_investigation(self, investigation_id: str) -> bool:
        """
        Delete investigation from database.
        
        Args:
            investigation_id: Investigation ID
            
        Returns:
            True if deleted, False if not found
            
        Raises:
            DatabaseError: If delete operation fails
        """
        pass
    
    # Embedding operations
    @abstractmethod
    def save_embedding(
        self, 
        text: str, 
        embedding: np.ndarray,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Save embedding to database.
        
        Args:
            text: Original text
            embedding: Embedding vector
            metadata: Embedding metadata
            
        Returns:
            Embedding ID
            
        Raises:
            DatabaseError: If save operation fails
        """
        pass
    
    @abstractmethod
    def load_embedding(self, embedding_id: str) -> Optional[Dict[str, Any]]:
        """
        Load embedding from database.
        
        Args:
            embedding_id: Embedding ID
            
        Returns:
            Embedding data dictionary or None if not found
            
        Raises:
            DatabaseError: If load operation fails
        """
        pass
    
    @abstractmethod
    def search_embeddings(
        self,
        query_embedding: np.ndarray,
        limit: int = 10,
        similarity_threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings.
        
        Args:
            query_embedding: Query embedding vector
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of similar embeddings with similarity scores
            
        Raises:
            DatabaseError: If search operation fails
        """
        pass
    
    # Question operations
    @abstractmethod
    def save_question_result(
        self,
        investigation_id: str,
        question: str,
        similarity_score: float,
        phase: str,
        metadata: Dict[str, Any]
    ) -> str:
        """
        Save question result to database.
        
        Args:
            investigation_id: Parent investigation ID
            question: Question text
            similarity_score: Similarity score achieved
            phase: Investigation phase
            metadata: Additional metadata
            
        Returns:
            Question record ID
            
        Raises:
            DatabaseError: If save operation fails
        """
        pass
    
    @abstractmethod
    def get_investigation_questions(self, investigation_id: str) -> List[Dict[str, Any]]:
        """
        Get all questions for an investigation.
        
        Args:
            investigation_id: Investigation ID
            
        Returns:
            List of question records
            
        Raises:
            DatabaseError: If query operation fails
        """
        pass
    
    # Analytics operations
    @abstractmethod
    def get_investigation_stats(self) -> Dict[str, Any]:
        """
        Get investigation statistics.
        
        Returns:
            Dictionary with statistics
            
        Raises:
            DatabaseError: If query operation fails
        """
        pass
    
    @abstractmethod
    def get_top_questions(self, limit: int = 20) -> List[Dict[str, Any]]:
        """
        Get top-performing questions across all investigations.
        
        Args:
            limit: Maximum number of results
            
        Returns:
            List of top questions with performance metrics
            
        Raises:
            DatabaseError: If query operation fails
        """
        pass
    
    # Utility operations
    @abstractmethod
    def optimize_database(self) -> None:
        """
        Optimize database performance.
        
        Raises:
            DatabaseError: If optimization fails
        """
        pass
    
    @abstractmethod
    def backup_database(self, backup_path: str) -> None:
        """
        Create database backup.
        
        Args:
            backup_path: Path for backup file
            
        Raises:
            DatabaseError: If backup operation fails
        """
        pass
    
    @abstractmethod
    def get_database_info(self) -> Dict[str, Any]:
        """
        Get database information and statistics.
        
        Returns:
            Database information dictionary
        """
        pass
    
    def __enter__(self):
        """Context manager entry."""
        self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.disconnect()
    
    def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the database.
        
        Returns:
            Dictionary with health check results
        """
        try:
            if not self.is_connected():
                return {
                    "status": "unhealthy",
                    "reason": "Not connected to database"
                }
            
            # Try a simple query
            stats = self.get_database_info()
            
            return {
                "status": "healthy",
                "database_info": stats,
                "connected": True
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "reason": str(e),
                "connected": False
            }