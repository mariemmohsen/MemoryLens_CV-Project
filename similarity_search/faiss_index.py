"""
Similarity Search for MemoryLens.

Builds a FAISS index over CLIP image embeddings and supports nearest-neighbor
search, either by a filename already in the index or by a raw query vector.
Embeddings are expected to be L2-normalized (as produced by
embeddings.clip_embedder.ClipEmbedder), so inner product search is
equivalent to cosine similarity.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

logger = logging.getLogger(__name__)


class SimilaritySearchIndex:
    """FAISS-backed nearest-neighbor search over image embeddings."""

    def __init__(self) -> None:
        self.index: faiss.Index | None = None
        self.filenames: List[str] = []

    def build(self, embeddings: np.ndarray, filenames: List[str]) -> None:
        """Build a flat inner-product FAISS index from embeddings.

        Args:
            embeddings: (N, D) L2-normalized float32 array.
            filenames: Image filename for each row, same order as `embeddings`.
        """
        if embeddings.shape[0] != len(filenames):
            raise ValueError(
                f"embeddings has {embeddings.shape[0]} rows but got "
                f"{len(filenames)} filenames"
            )

        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings.astype("float32"))
        self.filenames = list(filenames)
        logger.info(
            "Built FAISS index | vectors=%d | dim=%d", self.index.ntotal, dimension
        )

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
        """Find the k nearest neighbors of a query embedding.

        Args:
            query_embedding: (D,) or (1, D) L2-normalized float32 vector.
            k: Number of neighbors to return.

        Returns:
            List of (filename, similarity_score) tuples, most similar first.
        """
        if self.index is None:
            raise RuntimeError("Index is empty. Call build() or load() first.")

        query = np.asarray(query_embedding, dtype="float32").reshape(1, -1)
        scores, indices = self.index.search(query, k)

        return [
            (self.filenames[idx], float(score))
            for score, idx in zip(scores[0], indices[0])
            if idx != -1
        ]

    def search_by_filename(self, filename: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find the k nearest neighbors of an image already in the index.

        The query image itself is excluded from the results.

        Args:
            filename: Filename of an image already added via build().
            k: Number of neighbors to return (excluding the query image).
        """
        if self.index is None:
            raise RuntimeError("Index is empty. Call build() or load() first.")
        if filename not in self.filenames:
            raise ValueError(f"{filename} is not in the index")

        query_idx = self.filenames.index(filename)
        query_embedding = self.index.reconstruct(query_idx)
        results = self.search(query_embedding, k=k + 1)
        return [(name, score) for name, score in results if name != filename][:k]

    def save(self, index_path: Path, filenames_path: Path) -> None:
        """Persist the index and its filename mapping to disk."""
        if self.index is None:
            raise RuntimeError("Index is empty. Call build() first.")

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        with open(filenames_path, "w") as f:
            json.dump(self.filenames, f, indent=2)
        logger.info("Saved FAISS index to %s", index_path)

    @classmethod
    def load(cls, index_path: Path, filenames_path: Path) -> "SimilaritySearchIndex":
        """Load a previously saved index and its filename mapping."""
        instance = cls()
        instance.index = faiss.read_index(str(index_path))
        with open(filenames_path) as f:
            instance.filenames = json.load(f)
        logger.info("Loaded FAISS index from %s | vectors=%d", index_path, instance.index.ntotal)
        return instance
