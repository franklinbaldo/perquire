"""NumPy-first utilities for embedding vectors."""

from __future__ import annotations

import numpy as np

from ..exceptions import EmbeddingError


def cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
        raise EmbeddingError("Embeddings must be numpy arrays")
    if embedding1.shape != embedding2.shape:
        raise EmbeddingError(
            f"Embedding dimensions don't match: {embedding1.shape} vs {embedding2.shape}"
        )
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(embedding1, embedding2) / (norm1 * norm2))


def batch_cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    left = np.asarray(embeddings1, dtype=float)
    right = np.asarray(embeddings2, dtype=float)
    if left.ndim != 2 or right.ndim != 2:
        raise EmbeddingError("Batch embeddings must be 2-dimensional")
    if left.shape[1] != right.shape[1]:
        raise EmbeddingError(
            f"Embedding dimensions don't match: {left.shape[1]} vs {right.shape[1]}"
        )
    left_norms = np.linalg.norm(left, axis=1, keepdims=True)
    right_norms = np.linalg.norm(right, axis=1, keepdims=True)
    left_safe = np.divide(left, left_norms, out=np.zeros_like(left), where=left_norms != 0)
    right_safe = np.divide(right, right_norms, out=np.zeros_like(right), where=right_norms != 0)
    return left_safe @ right_safe.T


def normalize_embedding(embedding: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(embedding)
    return embedding if norm == 0 else embedding / norm


def euclidean_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    return float(np.linalg.norm(embedding1 - embedding2))


def manhattan_distance(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    return float(np.sum(np.abs(embedding1 - embedding2)))


def find_nearest_embeddings(
    target_embedding: np.ndarray,
    embedding_database: np.ndarray,
    top_k: int = 5,
    metric: str = "cosine",
) -> tuple[np.ndarray, np.ndarray]:
    if metric == "cosine":
        scores = batch_cosine_similarity(target_embedding.reshape(1, -1), embedding_database)[0]
        indices = np.argsort(scores)[::-1][:top_k]
        return indices, scores[indices]
    if metric == "euclidean":
        scores = np.linalg.norm(embedding_database - target_embedding, axis=1)
        indices = np.argsort(scores)[:top_k]
        return indices, scores[indices]
    if metric == "manhattan":
        scores = np.sum(np.abs(embedding_database - target_embedding), axis=1)
        indices = np.argsort(scores)[:top_k]
        return indices, scores[indices]
    raise EmbeddingError(f"Unknown metric: {metric}")


def reduce_embedding_dimensions(
    embeddings: np.ndarray,
    target_dimensions: int,
    method: str = "pca",
) -> np.ndarray:
    """Optional visualization helper; scikit-learn is loaded only on demand."""
    try:
        from sklearn.decomposition import PCA
        from sklearn.manifold import TSNE
    except ImportError as exc:
        raise EmbeddingError(
            "Dimension reduction requires the optional 'analysis' dependencies"
        ) from exc

    if method == "pca":
        return PCA(n_components=target_dimensions, random_state=42).fit_transform(embeddings)
    if method == "tsne":
        working = embeddings
        if working.shape[1] > 50:
            working = PCA(n_components=50, random_state=42).fit_transform(working)
        return TSNE(
            n_components=target_dimensions,
            random_state=42,
            perplexity=min(30, working.shape[0] - 1),
        ).fit_transform(working)
    raise EmbeddingError(f"Unknown reduction method: {method}")


def calculate_embedding_stats(embeddings: np.ndarray) -> dict:
    norms = np.linalg.norm(embeddings, axis=1)
    stats = {
        "count": embeddings.shape[0],
        "dimensions": embeddings.shape[1],
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "min_norm": float(np.min(norms)),
        "max_norm": float(np.max(norms)),
        "mean_values": embeddings.mean(axis=0).tolist(),
        "std_values": embeddings.std(axis=0).tolist(),
    }
    sample_size = min(100, embeddings.shape[0])
    if sample_size > 1:
        sample = embeddings[:sample_size]
        similarities = batch_cosine_similarity(sample, sample)
        off_diagonal = similarities[~np.eye(sample_size, dtype=bool)]
        stats.update(
            {
                "mean_pairwise_similarity": float(np.mean(off_diagonal)),
                "std_pairwise_similarity": float(np.std(off_diagonal)),
                "min_pairwise_similarity": float(np.min(off_diagonal)),
                "max_pairwise_similarity": float(np.max(off_diagonal)),
            }
        )
    return stats


def validate_embedding_compatibility(
    embedding1: np.ndarray,
    embedding2: np.ndarray,
    tolerance: float = 1e-6,
) -> bool:
    if embedding1.shape != embedding2.shape:
        return False
    if not (np.isfinite(embedding1).all() and np.isfinite(embedding2).all()):
        return False
    return bool(
        np.linalg.norm(embedding1) >= tolerance and np.linalg.norm(embedding2) >= tolerance
    )


def create_embedding_index(embeddings: np.ndarray) -> dict:
    normalized = np.array([normalize_embedding(item) for item in embeddings])
    return {
        "embeddings": normalized,
        "original_embeddings": embeddings,
        "count": embeddings.shape[0],
        "dimensions": embeddings.shape[1],
        "stats": calculate_embedding_stats(embeddings),
        "created_at": np.datetime64("now"),
    }


def search_embedding_index(
    index: dict,
    query_embedding: np.ndarray,
    top_k: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    normalized_query = normalize_embedding(query_embedding)
    similarities = batch_cosine_similarity(
        normalized_query.reshape(1, -1), index["embeddings"]
    )[0]
    indices = np.argsort(similarities)[::-1][:top_k]
    return indices, similarities[indices]
