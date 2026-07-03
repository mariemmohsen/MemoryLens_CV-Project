"""
Timeline Reconstruction for MemoryLens.

Orders event clusters chronologically. Each image's timestamp comes from its
EXIF capture time if present, otherwise its file creation time (see
utils.image_utils.get_capture_datetime). If every cluster ends up with an
indistinguishable timestamp - e.g. a bulk-downloaded dataset with no EXIF
data, all extracted within the same second - clusters are instead chained by
embedding similarity, so the timeline still reads as a plausible sequence of
events instead of an arbitrary one.
"""

import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.image_utils import get_capture_datetime  # noqa: E402

logger = logging.getLogger(__name__)


def _cluster_timestamp(images: List[str], raw_images_dir: Path) -> Tuple:
    """Earliest available timestamp among a cluster's images."""
    timestamps = [get_capture_datetime(raw_images_dir / name) for name in images]
    return min(timestamps, key=lambda t: t[0])


def _similarity_chain_order(centroids: np.ndarray) -> List[int]:
    """Order cluster indices by nearest-neighbor chaining over centroids.

    Starts from cluster 0 and repeatedly jumps to the most similar
    not-yet-visited cluster, so consecutive events in the timeline are
    visually/semantically related - a reasonable substitute for real
    chronology when no timestamp signal exists.
    """
    n = len(centroids)
    visited = [False] * n
    order = [0]
    visited[0] = True
    for _ in range(n - 1):
        sims = centroids @ centroids[order[-1]]
        sims[visited] = -np.inf
        next_idx = int(np.argmax(sims))
        order.append(next_idx)
        visited[next_idx] = True
    return order


def _cluster_centroids(clusters: List[Dict], embeddings: np.ndarray, filenames: List[str]) -> np.ndarray:
    """Mean (L2-normalized) embedding of each cluster's images."""
    name_to_row = {name: i for i, name in enumerate(filenames)}
    centroids = np.array([
        embeddings[[name_to_row[name] for name in c["images"] if name in name_to_row]].mean(axis=0)
        for c in clusters
    ])
    return centroids / np.linalg.norm(centroids, axis=1, keepdims=True)


def build_timeline(
    clusters: List[Dict],
    raw_images_dir: Path,
    embeddings: Optional[np.ndarray] = None,
    filenames: Optional[List[str]] = None,
    timestamp_spread_seconds: float = 1.0,
) -> List[Dict]:
    """Order clusters into a timeline.

    Args:
        clusters: Cluster dicts from event clustering (cluster_id, title,
            images, ...), with the noise cluster (-1) already excluded.
        raw_images_dir: Folder containing the dataset images (for EXIF/file time).
        embeddings: (N, D) embeddings aligned with `filenames`, used for the
            similarity-chain fallback.
        filenames: Filenames aligned with `embeddings`.
        timestamp_spread_seconds: If the earliest and latest cluster
            timestamps differ by less than this, timestamps are treated as
            uninformative and the similarity-chain fallback is used instead.

    Returns:
        `clusters`, each with an added "timestamp" (ISO string) and
        "timestamp_source" ("exif" | "file_time" | "cluster_similarity"),
        ordered as the timeline.
    """
    timestamped = []
    for cluster in clusters:
        dt, source = _cluster_timestamp(cluster["images"], raw_images_dir)
        timestamped.append({**cluster, "timestamp": dt, "timestamp_source": source})

    all_dts = [c["timestamp"] for c in timestamped]
    spread = (max(all_dts) - min(all_dts)).total_seconds() if len(all_dts) > 1 else 0

    if spread >= timestamp_spread_seconds:
        timestamped.sort(key=lambda c: c["timestamp"])
        logger.info("Ordered timeline by EXIF/file timestamps (spread=%.0fs)", spread)
    elif embeddings is not None and filenames is not None:
        logger.info(
            "Timestamps are indistinguishable (spread=%.2fs); falling back to cluster similarity",
            spread,
        )
        centroids = _cluster_centroids(timestamped, embeddings, filenames)
        order = _similarity_chain_order(centroids)
        timestamped = [timestamped[i] for i in order]
        for c in timestamped:
            c["timestamp_source"] = "cluster_similarity"
    else:
        logger.warning("Timestamps are indistinguishable and no embeddings were given; keeping input order")

    for c in timestamped:
        c["timestamp"] = c["timestamp"].isoformat()

    return timestamped
