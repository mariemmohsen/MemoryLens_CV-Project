"""
Runs Timeline Reconstruction over event_clustering/clusters.json and saves
the ordered result to timeline/timeline.json.

Usage:
    python -m timeline.run_timeline
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    RAW_IMAGES_DIR,
    CLUSTERS_OUTPUT,
    EMBEDDINGS_OUTPUT,
    EMBEDDINGS_FILENAMES_OUTPUT,
    TIMELINE_OUTPUT,
    TIMELINE_TIMESTAMP_SPREAD_SECONDS,
)
from event_clustering.event_clusterer import NOISE_LABEL  # noqa: E402
from timeline.timeline_builder import build_timeline  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run(
    clusters_path: Path = CLUSTERS_OUTPUT,
    raw_images_dir: Path = RAW_IMAGES_DIR,
    embeddings_path: Path = EMBEDDINGS_OUTPUT,
    filenames_path: Path = EMBEDDINGS_FILENAMES_OUTPUT,
    output_path: Path = TIMELINE_OUTPUT,
) -> None:
    """Order event clusters into a timeline and save results as JSON.

    Args:
        clusters_path: Path to clusters.json (from event clustering).
        raw_images_dir: Folder containing the dataset images.
        embeddings_path: Path to embeddings.npy, used for the similarity-chain fallback.
        filenames_path: Path to image_filenames.json, aligned with embeddings.npy.
        output_path: Where to write timeline.json.
    """
    if not clusters_path.exists():
        logger.error("No clusters found. Run `python -m event_clustering.run_clustering` first.")
        return

    with open(clusters_path) as f:
        clusters = json.load(f)
    events = [c for c in clusters if c["cluster_id"] != NOISE_LABEL]
    if not events:
        logger.error("No non-noise clusters to build a timeline from.")
        return

    embeddings, filenames = None, None
    if embeddings_path.exists() and filenames_path.exists():
        embeddings = np.load(embeddings_path)
        with open(filenames_path) as f:
            filenames = json.load(f)

    timeline = build_timeline(
        events, raw_images_dir, embeddings=embeddings, filenames=filenames,
        timestamp_spread_seconds=TIMELINE_TIMESTAMP_SPREAD_SECONDS,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(timeline, f, indent=2)

    logger.info("Timeline (%d events):", len(timeline))
    for event in timeline:
        logger.info(
            "  %s  [%s photos, source=%s]", event["title"], event["size"], event["timestamp_source"]
        )
    logger.info("Saved timeline to %s", output_path)


if __name__ == "__main__":
    run()
