"""
Evaluates the current production outputs against pseudo-ground-truth and saves
a metrics report to evaluation/results/metrics.json.

Reports:
  1. Event clustering - intrinsic + extrinsic metrics on the saved clusters.json.
  2. Scene recognition - Places365 top-1 accuracy on the labeled scene photos.

Usage:
    python -m evaluation.run_evaluation
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    EMBEDDINGS_OUTPUT,
    EMBEDDINGS_FILENAMES_OUTPUT,
    CAPTION_EMBEDDINGS_OUTPUT,
    CAPTIONS_OUTPUT,
    CLUSTERS_OUTPUT,
    SCENE_PREDICTIONS_OUTPUT,
    CLIP_MODEL_NAME,
    EVALUATION_RESULTS_DIR,
)
from event_clustering.embedding_fusion import (  # noqa: E402
    load_pipeline_arrays,
    caption_text_embeddings,
    fuse,
)
from event_clustering.event_clusterer import NOISE_LABEL  # noqa: E402
from evaluation.ground_truth import load_ground_truth, ground_truth_label, SCENE_CLASSES  # noqa: E402
from evaluation import metrics as M  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _predicted_labels(filenames, clusters_path: Path):
    """Per-filename cluster id from clusters.json (NOISE_LABEL if absent)."""
    with open(clusters_path) as f:
        clusters = json.load(f)
    label_of = {}
    for cluster in clusters:
        for name in cluster["images"]:
            label_of[name] = cluster["cluster_id"]
    return [label_of.get(name, NOISE_LABEL) for name in filenames]


def _scene_accuracy(scene_predictions_path: Path) -> dict:
    """Places365 top-1 accuracy on images whose true scene class is known."""
    if not scene_predictions_path.exists():
        return {}
    with open(scene_predictions_path) as f:
        predictions = json.load(f)

    correct = total = 0
    for item in predictions:
        truth = ground_truth_label(item["image"])
        if truth not in SCENE_CLASSES:  # only the 4 scene stock-photo classes
            continue
        total += 1
        if item["scene"].lower() == truth.lower():
            correct += 1
    return {
        "scene_top1_accuracy": (correct / total) if total else float("nan"),
        "scene_images_evaluated": total,
    }


def run(
    embeddings_path: Path = EMBEDDINGS_OUTPUT,
    filenames_path: Path = EMBEDDINGS_FILENAMES_OUTPUT,
    captions_path: Path = CAPTIONS_OUTPUT,
    clusters_path: Path = CLUSTERS_OUTPUT,
    scene_predictions_path: Path = SCENE_PREDICTIONS_OUTPUT,
    output_dir: Path = EVALUATION_RESULTS_DIR,
) -> dict:
    """Compute and save evaluation metrics for the current pipeline outputs."""
    if not clusters_path.exists():
        logger.error("No clusters.json - run `python -m event_clustering.run_clustering` first.")
        return {}

    image_embeddings, filenames, captions_by_filename = load_pipeline_arrays(
        embeddings_path, filenames_path, captions_path
    )
    text_embeddings = caption_text_embeddings(
        filenames, captions_by_filename, CLIP_MODEL_NAME, CAPTION_EMBEDDINGS_OUTPUT
    )
    fused = fuse(image_embeddings, text_embeddings)

    predicted = _predicted_labels(filenames, clusters_path)
    truth = load_ground_truth(filenames)

    report = {
        "dataset": {
            "n_images": len(filenames),
            "n_labeled": sum(t is not None for t in truth),
            "n_true_classes": len({t for t in truth if t is not None}),
        },
        "clustering_intrinsic": M.intrinsic_metrics(fused, predicted),
        "clustering_extrinsic": M.extrinsic_metrics(predicted, truth),
        "clustering_coverage": M.coverage(predicted),
        "scene_recognition": _scene_accuracy(scene_predictions_path),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(report, f, indent=2)

    _print_report(report)
    logger.info("Saved metrics to %s", output_dir / "metrics.json")
    return report


def _print_report(report: dict) -> None:
    d = report["dataset"]
    intr = report["clustering_intrinsic"]
    extr = report["clustering_extrinsic"]
    scene = report["scene_recognition"]

    print("\n" + "=" * 58)
    print("  EVENT CLUSTERING EVALUATION")
    print("=" * 58)
    print(f"  Images: {d['n_images']}  |  labeled: {d['n_labeled']}  |  "
          f"true classes: {d['n_true_classes']}")
    print(f"  Clusters found: {intr['n_clusters']}  |  "
          f"coverage (non-noise): {report['clustering_coverage']:.1%}")
    print("-" * 58)
    print("  Intrinsic (geometry)")
    print(f"    Silhouette (cosine)   {intr['silhouette']:.3f}   (higher better)")
    print(f"    Davies-Bouldin        {intr['davies_bouldin']:.3f}   (lower better)")
    print(f"    Calinski-Harabasz     {intr['calinski_harabasz']:.1f}")
    print("-" * 58)
    print("  Extrinsic (vs ground truth)")
    print(f"    Adjusted Rand (ARI)   {extr['ari']:.3f}")
    print(f"    NMI                   {extr['nmi']:.3f}")
    print(f"    Homogeneity           {extr['homogeneity']:.3f}")
    print(f"    Completeness          {extr['completeness']:.3f}")
    print(f"    V-measure             {extr['v_measure']:.3f}")
    print(f"    Purity                {extr['purity']:.3f}")
    if scene:
        print("-" * 58)
        print("  Scene recognition (Places365)")
        print(f"    Top-1 accuracy        {scene['scene_top1_accuracy']:.1%}"
              f"   (n={scene['scene_images_evaluated']})")
    print("=" * 58 + "\n")


if __name__ == "__main__":
    run()
