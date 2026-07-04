"""
Clustering evaluation metrics.

Two families are reported:

Intrinsic (no labels needed) - measure geometric quality of the partition:
  - Silhouette  (cosine): [-1, 1], higher is better.
  - Davies-Bouldin:       >= 0, lower is better.
  - Calinski-Harabasz:    higher is better.

Extrinsic (vs pseudo-ground-truth) - measure agreement with true classes:
  - Adjusted Rand Index (ARI):        [-0.5, 1], chance-corrected.
  - Normalized Mutual Information (NMI): [0, 1].
  - Homogeneity / Completeness / V-measure: [0, 1].
  - Purity: [0, 1], fraction correctly assigned if each cluster took its
    majority class.

Extrinsic metrics are computed only over images that have a ground-truth
label. Noise points (cluster -1) are treated as one additional predicted
cluster, which is the honest full-dataset accounting; `coverage` separately
reports the fraction of images that were actually clustered (not noise).
"""

from typing import Dict, List, Optional

import numpy as np
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    completeness_score,
    davies_bouldin_score,
    homogeneity_score,
    normalized_mutual_info_score,
    silhouette_score,
    v_measure_score,
)

NOISE_LABEL = -1


def purity_score(pred_labels: List[int], true_labels: List[str]) -> float:
    """Fraction of items that would be correct if each cluster took its
    majority true class."""
    pred = np.asarray(pred_labels)
    true = np.asarray(true_labels, dtype=object)
    total = len(true)
    if total == 0:
        return float("nan")
    correct = 0
    for cluster in set(pred):
        mask = pred == cluster
        classes, counts = np.unique(true[mask], return_counts=True)
        correct += counts.max()
    return correct / total


def intrinsic_metrics(embeddings: np.ndarray, labels: List[int]) -> Dict[str, float]:
    """Geometric cluster-quality metrics over non-noise points.

    Args:
        embeddings: (N, D) array the labels were computed on.
        labels: N cluster labels; -1 is treated as noise and excluded.

    Returns:
        Dict of metric name -> value (NaN when undefined, e.g. <2 clusters).
    """
    labels_arr = np.asarray(labels)
    keep = labels_arr != NOISE_LABEL
    x = embeddings[keep]
    y = labels_arr[keep]

    result = {
        "silhouette": float("nan"),
        "davies_bouldin": float("nan"),
        "calinski_harabasz": float("nan"),
        "n_clusters": int(len(set(y.tolist()))),
        "n_evaluated": int(keep.sum()),
    }
    if result["n_clusters"] >= 2 and result["n_clusters"] < len(x):
        result["silhouette"] = float(silhouette_score(x, y, metric="cosine"))
        result["davies_bouldin"] = float(davies_bouldin_score(x, y))
        result["calinski_harabasz"] = float(calinski_harabasz_score(x, y))
    return result


def extrinsic_metrics(
    pred_labels: List[int],
    true_labels: List[Optional[str]],
) -> Dict[str, float]:
    """Agreement metrics between predicted clusters and ground-truth classes.

    Only items with a non-None ground-truth label are scored.

    Args:
        pred_labels: Predicted cluster label per item (aligned with true_labels).
        true_labels: Ground-truth class per item; None means unlabeled (skipped).

    Returns:
        Dict of metric name -> value.
    """
    pred = [p for p, t in zip(pred_labels, true_labels) if t is not None]
    true = [t for t in true_labels if t is not None]

    if not true:
        return {k: float("nan") for k in
                ("ari", "nmi", "homogeneity", "completeness", "v_measure", "purity",
                 "n_labeled", "n_true_classes")}

    return {
        "ari": float(adjusted_rand_score(true, pred)),
        "nmi": float(normalized_mutual_info_score(true, pred)),
        "homogeneity": float(homogeneity_score(true, pred)),
        "completeness": float(completeness_score(true, pred)),
        "v_measure": float(v_measure_score(true, pred)),
        "purity": float(purity_score(pred, true)),
        "n_labeled": int(len(true)),
        "n_true_classes": int(len(set(true))),
    }


def coverage(labels: List[int]) -> float:
    """Fraction of items assigned to a real cluster (i.e. not noise)."""
    labels_arr = np.asarray(labels)
    if len(labels_arr) == 0:
        return float("nan")
    return float((labels_arr != NOISE_LABEL).mean())
