"""
Embedding utility functions.
"""

import numpy as np
from typing import List, Union, Tuple, Optional
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging

from ..exceptions import EmbeddingError

logger = logging.getLogger(__name__)


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score between -1 and 1
        
    Raises:
        EmbeddingError: If embeddings have different dimensions or are invalid
    """
    try:
        # Validate inputs
        if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
            raise EmbeddingError("Embeddings must be numpy arrays")
        
        if embedding1.shape != embedding2.shape:
            raise EmbeddingError(f"Embedding dimensions don't match: {embedding1.shape} vs {embedding2.shape}")
        
        # Check for zero vectors
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Calculate cosine similarity
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
        
    except Exception as e:
        raise EmbeddingError(f"Failed to calculate cosine similarity: {str(e)}")


def batch_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Calculate cosine similarity between batches of embeddings.
    
    Args:
        embeddings1: First batch of embeddings (n1 x dimensions)
        embeddings2: Second batch of embeddings (n2 x dimensions)
        
    Returns:
        Similarity matrix (n1 x n2)
    """
    try:
        return sklearn_cosine_similarity(embeddings1, embeddings2)
    except Exception as e:
        raise EmbeddingError(f"Failed to calculate batch cosine similarity: {str(e)}")


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    """
    Normalize an embedding to unit length.
    
    Args:
        embedding: Embedding vector to normalize
        
    Returns:
        Normalized embedding vector
    """
    try:
        norm = np.linalg.norm(embedding)
        if norm == 0:
            return embedding
        return embedding / norm
    except Exception as e:
        raise EmbeddingError(f"Failed to normalize embedding: {str(e)}")


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Euclidean distance
    """
    try:
        return float(np.linalg.norm(embedding1 - embedding2))
    except Exception as e:
        raise EmbeddingError(f"Failed to calculate Euclidean distance: {str(e)}")


def manhattan_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """
    Calculate Manhattan (L1) distance between two embeddings.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Manhattan distance
    """
    try:
        return float(np.sum(np.abs(embedding1 - embedding2)))
    except Exception as e:
        raise EmbeddingError(f"Failed to calculate Manhattan distance: {str(e)}")


def find_nearest_embeddings(
    target_embedding: np.ndarray,
    embedding_database: np.ndarray,
    top_k: int = 5,
    metric: str = "cosine"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find the k nearest embeddings to a target embedding.
    
    Args:
        target_embedding: Target embedding vector
        embedding_database: Database of embeddings to search (n x dimensions)
        top_k: Number of nearest neighbors to return
        metric: Distance metric ("cosine", "euclidean", "manhattan")
        
    Returns:
        Tuple of (indices, similarities/distances)
    """
    try:
        if metric == "cosine":
            # Calculate cosine similarities
            similarities = batch_cosine_similarity(
                target_embedding.reshape(1, -1), 
                embedding_database
            )[0]
            # Get top k (highest similarities)
            indices = np.argsort(similarities)[::-1][:top_k]
            scores = similarities[indices]
            
        elif metric == "euclidean":
            # Calculate Euclidean distances
            distances = np.array([
                euclidean_distance(target_embedding, emb)
                for emb in embedding_database
            ])
            # Get top k (lowest distances)
            indices = np.argsort(distances)[:top_k]
            scores = distances[indices]
            
        elif metric == "manhattan":
            # Calculate Manhattan distances
            distances = np.array([
                manhattan_distance(target_embedding, emb)
                for emb in embedding_database
            ])
            # Get top k (lowest distances)
            indices = np.argsort(distances)[:top_k]
            scores = distances[indices]
            
        else:
            raise EmbeddingError(f"Unknown metric: {metric}")
        
        return indices, scores
        
    except Exception as e:
        raise EmbeddingError(f"Failed to find nearest embeddings: {str(e)}")


def reduce_embedding_dimensions(
    embeddings: np.ndarray,
    target_dimensions: int,
    method: str = "pca"
) -> np.ndarray:
    """
    Reduce embedding dimensions using PCA or t-SNE.
    
    Args:
        embeddings: Input embeddings (n x original_dimensions)
        target_dimensions: Target number of dimensions
        method: Reduction method ("pca" or "tsne")
        
    Returns:
        Reduced embeddings (n x target_dimensions)
    """
    try:
        if method == "pca":
            reducer = PCA(n_components=target_dimensions, random_state=42)
            return reducer.fit_transform(embeddings)
            
        elif method == "tsne":
            # t-SNE is computationally expensive, so limit input dimensions
            if embeddings.shape[1] > 50:
                # First reduce to 50 dimensions with PCA
                pca = PCA(n_components=50, random_state=42)
                embeddings = pca.fit_transform(embeddings)
            
            reducer = TSNE(
                n_components=target_dimensions,
                random_state=42,
                perplexity=min(30, embeddings.shape[0] - 1)
            )
            return reducer.fit_transform(embeddings)
            
        else:
            raise EmbeddingError(f"Unknown reduction method: {method}")
            
    except Exception as e:
        raise EmbeddingError(f"Failed to reduce embedding dimensions: {str(e)}")


def calculate_embedding_stats(embeddings: np.ndarray) -> dict:
    """
    Calculate statistics for a collection of embeddings.
    
    Args:
        embeddings: Collection of embeddings (n x dimensions)
        
    Returns:
        Dictionary with statistics
    """
    try:
        stats = {
            "count": embeddings.shape[0],
            "dimensions": embeddings.shape[1],
            "mean_norm": float(np.mean(np.linalg.norm(embeddings, axis=1))),
            "std_norm": float(np.std(np.linalg.norm(embeddings, axis=1))),
            "min_norm": float(np.min(np.linalg.norm(embeddings, axis=1))),
            "max_norm": float(np.max(np.linalg.norm(embeddings, axis=1))),
            "mean_values": embeddings.mean(axis=0).tolist(),
            "std_values": embeddings.std(axis=0).tolist(),
        }
        
        # Calculate pairwise similarities for sample
        sample_size = min(100, embeddings.shape[0])
        if sample_size > 1:
            sample_indices = np.random.choice(embeddings.shape[0], sample_size, replace=False)
            sample_embeddings = embeddings[sample_indices]
            
            similarities = batch_cosine_similarity(sample_embeddings, sample_embeddings)
            # Remove diagonal (self-similarities)
            mask = ~np.eye(similarities.shape[0], dtype=bool)
            off_diagonal = similarities[mask]
            
            stats.update({
                "mean_pairwise_similarity": float(np.mean(off_diagonal)),
                "std_pairwise_similarity": float(np.std(off_diagonal)),
                "min_pairwise_similarity": float(np.min(off_diagonal)),
                "max_pairwise_similarity": float(np.max(off_diagonal))
            })
        
        return stats
        
    except Exception as e:
        raise EmbeddingError(f"Failed to calculate embedding statistics: {str(e)}")


def validate_embedding_compatibility(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    tolerance: float = 1e-6
) -> bool:
    """
    Check if two embeddings are compatible for comparison operations.
    
    Args:
        embedding1: First embedding
        embedding2: Second embedding
        tolerance: Numerical tolerance for validation
        
    Returns:
        True if embeddings are compatible
    """
    try:
        # Check dimensions
        if embedding1.shape != embedding2.shape:
            return False
        
        # Check for valid values (no NaN or inf)
        if not (np.isfinite(embedding1).all() and np.isfinite(embedding2).all()):
            return False
        
        # Check for zero vectors
        if np.linalg.norm(embedding1) < tolerance or np.linalg.norm(embedding2) < tolerance:
            logger.warning("One or both embeddings are near-zero vectors")
            return False
        
        return True
        
    except Exception:
        return False


def create_embedding_index(embeddings: np.ndarray) -> dict:
    """
    Create an index for fast embedding similarity search.
    
    Args:
        embeddings: Collection of embeddings to index
        
    Returns:
        Dictionary containing the index and metadata
    """
    try:
        # Normalize embeddings for faster cosine similarity
        normalized_embeddings = np.array([
            normalize_embedding(emb) for emb in embeddings
        ])
        
        index = {
            "embeddings": normalized_embeddings,
            "original_embeddings": embeddings,
            "count": embeddings.shape[0],
            "dimensions": embeddings.shape[1],
            "stats": calculate_embedding_stats(embeddings),
            "created_at": np.datetime64('now')
        }
        
        return index
        
    except Exception as e:
        raise EmbeddingError(f"Failed to create embedding index: {str(e)}")


def search_embedding_index(
    index: dict,
    query_embedding: np.ndarray,
    top_k: int = 10
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Search an embedding index for similar embeddings.
    
    Args:
        index: Embedding index created by create_embedding_index
        query_embedding: Query embedding
        top_k: Number of results to return
        
    Returns:
        Tuple of (indices, similarities)
    """
    try:
        normalized_query = normalize_embedding(query_embedding)
        
        similarities = batch_cosine_similarity(
            normalized_query.reshape(1, -1),
            index["embeddings"]
        )[0]
        
        indices = np.argsort(similarities)[::-1][:top_k]
        scores = similarities[indices]
        
        return indices, scores
        
    except Exception as e:
        raise EmbeddingError(f"Failed to search embedding index: {str(e)}")