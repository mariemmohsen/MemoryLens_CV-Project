"""
Ablation study for event clustering.

Answers, with numbers, the design questions a reviewer would ask:

  1. Does adding caption text to image embeddings actually help?
     -> sweeps the distance threshold for three modalities:
        image-only, caption-only, and fused (image + caption).
  2. What threshold is best, and how sensitive are results to it?
     -> the threshold sweep + an auto-selected threshold (max silhouette).
  3. Agglomerative vs HDBSCAN (the spec's original choice)?
     -> both run on the fused embeddings and compared head to head.

Outputs (evaluation/results/):
  - ablation_sweep.csv / .json  - every (modality, threshold) row
  - algorithm_comparison.json   - Agglomerative vs HDBSCAN
  - ablation_ari.png, ablation_nmi.png, ablation_silhouette.png,
    ablation_n_clusters.png     - threshold-sweep plots

Usage:
    python -m evaluation.ablation
"""

import csv
import json
import logging
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless: write PNGs, never open a window
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    EMBEDDINGS_OUTPUT,
    EMBEDDINGS_FILENAMES_OUTPUT,
    CAPTION_EMBEDDINGS_OUTPUT,
    CAPTIONS_OUTPUT,
    CLIP_MODEL_NAME,
    EVENT_CLUSTERING_MIN_CLUSTER_SIZE,
    EVENT_CLUSTERING_THRESHOLD_SWEEP,
    EVALUATION_RESULTS_DIR,
)
from event_clustering.embedding_fusion import (  # noqa: E402
    load_pipeline_arrays,
    caption_text_embeddings,
    build_modalities,
)
from event_clustering.event_clusterer import (  # noqa: E402
    EventClusterer,
    select_threshold_by_silhouette,
)
from evaluation.ground_truth import load_ground_truth, EVENT_CLASSES  # noqa: E402
from evaluation import metrics as M  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

MODALITY_STYLE = {  # consistent colors/labels across plots
    "image": ("#94a3b8", "Image only"),
    "caption": ("#f59e0b", "Caption only"),
    "fused": ("#6366f1", "Fused (image + caption)"),
}


def _event_only(truth):
    """Truth vector with only the fine-grained event classes kept (others -> None),
    so extrinsic metrics measure how well the hard event photos separate."""
    return [t if t in EVENT_CLASSES else None for t in truth]


def _evaluate(embeddings, labels, truth) -> dict:
    """One row of metrics for a given clustering.

    Reports both full-dataset extrinsic metrics (dominated by the large scene
    pools) and event-subset metrics (the hard minority the fusion is meant to
    help), so the two effects can be told apart.
    """
    intr = M.intrinsic_metrics(embeddings, labels)
    extr = M.extrinsic_metrics(labels, truth)
    extr_events = M.extrinsic_metrics(labels, _event_only(truth))
    return {
        "n_clusters": intr["n_clusters"],
        "coverage": M.coverage(labels),
        "silhouette": intr["silhouette"],
        "davies_bouldin": intr["davies_bouldin"],
        "ari": extr["ari"],
        "nmi": extr["nmi"],
        "v_measure": extr["v_measure"],
        "purity": extr["purity"],
        "ari_events": extr_events["ari"],
        "nmi_events": extr_events["nmi"],
        "purity_events": extr_events["purity"],
    }


def _sweep(modalities, truth, thresholds, min_cluster_size) -> list:
    """Cluster every (modality, threshold) pair and collect metrics."""
    rows = []
    for modality, embeddings in modalities.items():
        for threshold in thresholds:
            labels = EventClusterer(
                distance_threshold=threshold, min_cluster_size=min_cluster_size,
                algorithm="agglomerative",
            ).fit_predict(embeddings)
            rows.append({"modality": modality, "threshold": threshold,
                         **_evaluate(embeddings, labels, truth)})
    return rows


def _plot_metric(rows, metric, ylabel, out_path, thresholds) -> None:
    """One line per modality: metric vs threshold."""
    plt.figure(figsize=(7, 4.5))
    for modality, (color, label) in MODALITY_STYLE.items():
        xs = [r["threshold"] for r in rows if r["modality"] == modality]
        ys = [r[metric] for r in rows if r["modality"] == modality]
        plt.plot(xs, ys, marker="o", ms=4, color=color, label=label)
    plt.xlabel("Agglomerative distance threshold (cosine)")
    plt.ylabel(ylabel)
    plt.title(ylabel + " vs clustering threshold")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


def _algorithm_comparison(fused, truth, threshold, min_cluster_size) -> dict:
    """Agglomerative vs HDBSCAN on the fused embeddings."""
    comparison = {}
    for algorithm in ("agglomerative", "hdbscan"):
        try:
            labels = EventClusterer(
                distance_threshold=threshold, min_cluster_size=min_cluster_size,
                algorithm=algorithm,
            ).fit_predict(fused)
            comparison[algorithm] = _evaluate(fused, labels, truth)
        except Exception as exc:  # hdbscan may be missing
            logger.warning("Skipping %s: %s", algorithm, exc)
            comparison[algorithm] = {"error": str(exc)}
    return comparison


def run(output_dir: Path = EVALUATION_RESULTS_DIR) -> dict:
    """Run the full ablation study and save tables + plots."""
    image_embeddings, filenames, captions_by_filename = load_pipeline_arrays(
        EMBEDDINGS_OUTPUT, EMBEDDINGS_FILENAMES_OUTPUT, CAPTIONS_OUTPUT
    )
    text_embeddings = caption_text_embeddings(
        filenames, captions_by_filename, CLIP_MODEL_NAME, CAPTION_EMBEDDINGS_OUTPUT
    )
    modalities = build_modalities(image_embeddings, text_embeddings)
    truth = load_ground_truth(filenames)
    thresholds = EVENT_CLUSTERING_THRESHOLD_SWEEP

    logger.info("Sweeping %d thresholds x %d modalities...", len(thresholds), len(modalities))
    rows = _sweep(modalities, truth, thresholds, EVENT_CLUSTERING_MIN_CLUSTER_SIZE)

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "ablation_sweep.json", "w") as f:
        json.dump(rows, f, indent=2)
    with open(output_dir / "ablation_sweep.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    _plot_metric(rows, "ari", "Adjusted Rand Index (ARI) - full dataset", output_dir / "ablation_ari.png", thresholds)
    _plot_metric(rows, "ari_events", "ARI - event subset (hard cases)", output_dir / "ablation_ari_events.png", thresholds)
    _plot_metric(rows, "nmi", "Normalized Mutual Information", output_dir / "ablation_nmi.png", thresholds)
    _plot_metric(rows, "silhouette", "Silhouette (cosine)", output_dir / "ablation_silhouette.png", thresholds)
    _plot_metric(rows, "n_clusters", "Number of clusters", output_dir / "ablation_n_clusters.png", thresholds)

    # Best fused threshold, by extrinsic (ARI) and intrinsic (auto-selected).
    fused_rows = [r for r in rows if r["modality"] == "fused"]
    best_by_ari = max(fused_rows, key=lambda r: (r["ari"] if r["ari"] == r["ari"] else -1))
    auto_threshold, _ = select_threshold_by_silhouette(modalities["fused"], thresholds)

    algo = _algorithm_comparison(
        modalities["fused"], truth, best_by_ari["threshold"], EVENT_CLUSTERING_MIN_CLUSTER_SIZE
    )
    with open(output_dir / "algorithm_comparison.json", "w") as f:
        json.dump(algo, f, indent=2)

    summary = {
        "best_fused_threshold_by_ari": best_by_ari["threshold"],
        "best_fused_ari": best_by_ari["ari"],
        "auto_selected_threshold_by_silhouette": auto_threshold,
        "algorithm_comparison": algo,
    }
    with open(output_dir / "ablation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    _print_summary(rows, best_by_ari, auto_threshold, algo)
    logger.info("Saved ablation tables and plots to %s", output_dir)
    return summary


def _print_summary(rows, best_by_ari, auto_threshold, algo) -> None:
    def _best(mrows, key):
        return max(mrows, key=lambda r: (r[key] if r[key] == r[key] else -1))

    print("\n" + "=" * 66)
    print("  ABLATION: does caption text help clustering?")
    print("=" * 66)
    print("  Full dataset (dominated by the 200 near-identical scene photos):")
    print(f"  {'modality':<12}{'best ARI':>10}{'best NMI':>10}{'@threshold':>12}")
    for modality in ("image", "caption", "fused"):
        mrows = [r for r in rows if r["modality"] == modality]
        best = _best(mrows, "ari")
        print(f"  {modality:<12}{best['ari']:>10.3f}{best['nmi']:>10.3f}{best['threshold']:>12.3f}")
    print()
    print("  Event subset only (the hard minority fusion is meant to help):")
    print(f"  {'modality':<12}{'best ARI':>10}{'best NMI':>10}{'@threshold':>12}")
    for modality in ("image", "caption", "fused"):
        mrows = [r for r in rows if r["modality"] == modality]
        best = _best(mrows, "ari_events")
        print(f"  {modality:<12}{best['ari_events']:>10.3f}{best['nmi_events']:>10.3f}{best['threshold']:>12.3f}")
    print("-" * 66)
    print(f"  Best fused threshold by ARI:        {best_by_ari['threshold']:.3f} "
          f"(ARI={best_by_ari['ari']:.3f})")
    print(f"  Auto-selected threshold (silhouette): {auto_threshold:.3f}")
    print("-" * 66)
    print("  Agglomerative vs HDBSCAN (fused embeddings)")
    for algorithm, m in algo.items():
        if "error" in m:
            print(f"    {algorithm:<14} unavailable ({m['error']})")
        else:
            print(f"    {algorithm:<14} ARI={m['ari']:.3f}  NMI={m['nmi']:.3f}  "
                  f"clusters={m['n_clusters']}  coverage={m['coverage']:.0%}")
    print("=" * 66 + "\n")


if __name__ == "__main__":
    run()
