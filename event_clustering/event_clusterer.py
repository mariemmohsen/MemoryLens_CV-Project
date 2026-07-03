"""
Event Clustering for MemoryLens.

Groups images into events automatically over embeddings that combine CLIP
image features with CLIP text features of each image's caption. Two
algorithms are supported (select via `algorithm`):

- "agglomerative" (default): cosine-distance-threshold Agglomerative
  Clustering. This is the recommended choice for this dataset - see below.
- "hdbscan": the density-based algorithm from the original project spec.

Why the default isn't HDBSCAN: HDBSCAN is density-based, and with this
dataset's mix of very large, near-duplicate photo pools (e.g. 50 generic
airport photos) and small, visually-diverse event pools (e.g. 5 photos from
one birthday party), the small pools never reach the local density HDBSCAN
requires and always get marked as noise, regardless of parameters. A flat
cosine-distance threshold does not have that bias, and blending in caption
text (e.g. "cake", "candles", "christmas tree") gives real events a
distinguishing signal that raw pixels alone don't carry.
"""

import logging
from collections import Counter
from typing import List, Optional

import numpy as np
from sklearn.cluster import AgglomerativeClustering

logger = logging.getLogger(__name__)

NOISE_LABEL = -1  # label for images too small/loose a group to count as an event


class EventClusterer:
    """Clusters image (+ caption) embeddings into events."""

    def __init__(
        self,
        distance_threshold: float = 0.3,
        min_cluster_size: int = 3,
        linkage: str = "average",
        algorithm: str = "agglomerative",
    ) -> None:
        """
        Args:
            distance_threshold: (agglomerative) Cosine distance above which
                clusters are not merged. Lower = more, tighter clusters.
            min_cluster_size: Smallest group of photos treated as its own
                event; anything smaller is labeled noise (-1).
            linkage: (agglomerative) Linkage strategy.
            algorithm: "agglomerative" or "hdbscan".
        """
        if algorithm not in ("agglomerative", "hdbscan"):
            raise ValueError(f"Unknown clustering algorithm: {algorithm}")
        self.distance_threshold = distance_threshold
        self.min_cluster_size = min_cluster_size
        self.linkage = linkage
        self.algorithm = algorithm

    def fit_predict(self, embeddings: np.ndarray) -> List[int]:
        """Cluster embeddings and return one cluster label per row.

        Args:
            embeddings: (N, D) array of embeddings (ideally L2-normalized
                image+caption CLIP embeddings).

        Returns:
            List of N cluster labels; -1 marks images that don't belong to
            any group of at least `min_cluster_size`.
        """
        if self.algorithm == "hdbscan":
            labels = self._fit_hdbscan(embeddings)
        else:
            labels = self._fit_agglomerative(embeddings)

        n_clusters = len(set(labels) - {NOISE_LABEL})
        n_noise = labels.count(NOISE_LABEL)
        logger.info(
            "%s clustering found %d clusters | %d/%d images unclustered (noise)",
            self.algorithm.capitalize(), n_clusters, n_noise, len(labels),
        )
        return labels

    def _fit_agglomerative(self, embeddings: np.ndarray) -> List[int]:
        """Agglomerative clustering; small clusters are relabeled as noise."""
        clusterer = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=self.distance_threshold,
            metric="cosine",
            linkage=self.linkage,
        )
        raw_labels = clusterer.fit_predict(embeddings)

        sizes = Counter(raw_labels)
        return [
            int(label) if sizes[label] >= self.min_cluster_size else NOISE_LABEL
            for label in raw_labels
        ]

    def _fit_hdbscan(self, embeddings: np.ndarray) -> List[int]:
        """HDBSCAN clustering (marks its own noise as -1)."""
        import hdbscan  # optional dependency, only needed for this algorithm

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            metric="euclidean",  # ~cosine on L2-normalized embeddings
        )
        return [int(label) for label in clusterer.fit_predict(embeddings)]
