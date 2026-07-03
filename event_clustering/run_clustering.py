"""
Runs Event Clustering over embeddings/embeddings.npy (combined with CLIP text
embeddings of each image's caption) and saves the results to
event_clustering/clusters.json.

Each cluster is titled with its majority scene label (from
scene_recognition/scene_predictions.json) when that's informative, otherwise
with the most common distinctive word across its captions (from
captioning/captions.json) - this is what lets a "birthday party" or
"christmas" event get a meaningful title instead of just "Other". Images
that don't fit any group of at least `min_cluster_size` are grouped under
cluster_id -1 / title "Other".

Usage:
    python -m event_clustering.run_clustering
"""

import json
import logging
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    EMBEDDINGS_OUTPUT,
    EMBEDDINGS_FILENAMES_OUTPUT,
    SCENE_PREDICTIONS_OUTPUT,
    CAPTIONS_OUTPUT,
    CLUSTERS_OUTPUT,
    CLIP_MODEL_NAME,
    EVENT_CLUSTERING_ALGORITHM,
    EVENT_CLUSTERING_DISTANCE_THRESHOLD,
    EVENT_CLUSTERING_MIN_CLUSTER_SIZE,
)
from embeddings.clip_embedder import ClipEmbedder  # noqa: E402
from event_clustering.event_clusterer import EventClusterer, NOISE_LABEL  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Generic words filtered out when picking a caption-based cluster title.
STOPWORDS = {
    "a", "an", "the", "of", "with", "and", "on", "in", "at", "is", "are", "to",
    "for", "up", "down", "into", "over", "near", "next", "some", "two", "three",
    "group", "person", "people", "man", "woman", "standing", "sitting", "large",
    "small", "photo", "picture", "image", "background", "front", "side", "top",
}


def _load_by_filename(path: Path, value_key: str) -> Dict[str, str]:
    """Load a results JSON file (list of dicts with an "image" key) into a
    {filename: value} lookup. Returns {} if the file doesn't exist yet."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {item["image"]: item[value_key] for item in data}


def _caption_words(caption: str) -> set:
    """Non-generic words in a caption, deduplicated (one image = one vote per word)."""
    return {w for w in re.findall(r"[a-z]+", caption.lower()) if w not in STOPWORDS}


def _dataset_word_document_frequency(captions_by_filename: Dict[str, str]) -> Counter:
    """Count how many images (dataset-wide) each word appears in."""
    doc_freq = Counter()
    for caption in captions_by_filename.values():
        doc_freq.update(_caption_words(caption))
    return doc_freq


def _caption_keyword_title(
    images: List[str],
    captions_by_filename: Dict[str, str],
    dataset_word_doc_freq: Counter,
) -> str:
    """Fall back title: the caption word most distinctive to this cluster.

    Scored as (how many of this cluster's images mention the word) divided by
    (how many images dataset-wide mention it), so a word shared by every
    cluster (e.g. "family") loses to one that's concentrated here (e.g.
    "cake", "christmas"). Requires the word to appear in >=2 images of the
    cluster so a single caption's wording can't decide the title.
    """
    local_doc_freq = Counter()
    for name in images:
        local_doc_freq.update(_caption_words(captions_by_filename.get(name, "")))

    candidates = {w: c for w, c in local_doc_freq.items() if c >= 2}
    if not candidates:
        return None

    best_word = max(candidates, key=lambda w: candidates[w] / dataset_word_doc_freq[w])
    return best_word.title()


def _title_for_cluster(
    cluster_id: int,
    images: List[str],
    scenes_by_filename: Dict[str, str],
    captions_by_filename: Dict[str, str],
    dataset_word_doc_freq: Counter,
) -> str:
    """Title a cluster: majority scene if informative, else a caption keyword."""
    if cluster_id == NOISE_LABEL:
        return "Other"

    scene_counts = Counter(
        scenes_by_filename[name] for name in images if name in scenes_by_filename
    )
    if scene_counts:
        top_scene, top_count = scene_counts.most_common(1)[0]
        if top_scene != "Other" and top_count / len(images) >= 0.5:
            return top_scene

    return (
        _caption_keyword_title(images, captions_by_filename, dataset_word_doc_freq)
        or f"Cluster {cluster_id}"
    )


def run(
    embeddings_path: Path = EMBEDDINGS_OUTPUT,
    filenames_path: Path = EMBEDDINGS_FILENAMES_OUTPUT,
    scene_predictions_path: Path = SCENE_PREDICTIONS_OUTPUT,
    captions_path: Path = CAPTIONS_OUTPUT,
    output_path: Path = CLUSTERS_OUTPUT,
    distance_threshold: float = EVENT_CLUSTERING_DISTANCE_THRESHOLD,
    min_cluster_size: int = EVENT_CLUSTERING_MIN_CLUSTER_SIZE,
) -> None:
    """Cluster every embedded image into events and save results as JSON.

    Args:
        embeddings_path: Path to embeddings.npy (from the embeddings step).
        filenames_path: Path to image_filenames.json (from the embeddings step).
        scene_predictions_path: Path to scene_predictions.json, used to title clusters.
        captions_path: Path to captions.json, used both as a clustering signal
            and as a titling fallback.
        output_path: Where to write clusters.json.
        distance_threshold: Cosine distance above which clusters aren't merged.
        min_cluster_size: Smallest group of photos treated as its own event.
    """
    if not embeddings_path.exists() or not filenames_path.exists():
        logger.error(
            "Embeddings not found. Run `python -m embeddings.generate_embeddings` first."
        )
        return

    image_embeddings = np.load(embeddings_path)
    with open(filenames_path) as f:
        filenames = json.load(f)
    scenes_by_filename = _load_by_filename(scene_predictions_path, "scene")
    captions_by_filename = _load_by_filename(captions_path, "caption")

    if captions_by_filename:
        logger.info("Embedding captions to enrich clustering signal...")
        embedder = ClipEmbedder(model_name=CLIP_MODEL_NAME)
        captions = [captions_by_filename.get(name, "") for name in filenames]
        text_embeddings = embedder.embed_texts(captions)
        combined = image_embeddings + text_embeddings
        embeddings = combined / np.linalg.norm(combined, axis=1, keepdims=True)
    else:
        logger.warning("No captions found; clustering on image embeddings only.")
        embeddings = image_embeddings

    logger.info("Clustering %d images into events...", len(filenames))
    clusterer = EventClusterer(
        distance_threshold=distance_threshold,
        min_cluster_size=min_cluster_size,
        algorithm=EVENT_CLUSTERING_ALGORITHM,
    )
    labels = clusterer.fit_predict(embeddings)

    images_by_cluster: Dict[int, List[str]] = defaultdict(list)
    for filename, label in zip(filenames, labels):
        images_by_cluster[label].append(filename)

    dataset_word_doc_freq = _dataset_word_document_frequency(captions_by_filename)

    clusters = []
    for cluster_id, images in sorted(images_by_cluster.items(), key=lambda kv: (kv[0] == NOISE_LABEL, -len(kv[1]))):
        title = _title_for_cluster(
            cluster_id, images, scenes_by_filename, captions_by_filename, dataset_word_doc_freq
        )
        clusters.append({
            "cluster_id": cluster_id,
            "title": title,
            "size": len(images),
            "images": images,
        })
        logger.info("Cluster %s: %s (%d images)", cluster_id, title, len(images))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(clusters, f, indent=2)

    logger.info("Saved %d clusters to %s", len(clusters), output_path)


if __name__ == "__main__":
    run()
