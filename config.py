"""
Central configuration for MemoryLens.

Keeping paths and constants in one place makes it easy to reuse them
across modules (embeddings, scene recognition, captioning, clustering, etc.)
without hardcoding strings everywhere.
"""

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Project root & data folders
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent


def _load_env_file(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Real environment variables win over .env values. Quotes around values
    are stripped, blank lines and # comments are ignored. This keeps secrets
    like GROQ_API_KEY out of the codebase without requiring python-dotenv.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("'\""))


_load_env_file(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_IMAGES_DIR = DATA_DIR / "raw_images"

MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Pipeline step folders
# ---------------------------------------------------------------------------
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
SCENE_RECOGNITION_DIR = PROJECT_ROOT / "scene_recognition"
CAPTIONING_DIR = PROJECT_ROOT / "captioning"
SIMILARITY_SEARCH_DIR = PROJECT_ROOT / "similarity_search"
EVENT_CLUSTERING_DIR = PROJECT_ROOT / "event_clustering"
TIMELINE_DIR = PROJECT_ROOT / "timeline"
STORY_GENERATION_DIR = PROJECT_ROOT / "story_generation"
EVALUATION_DIR = PROJECT_ROOT / "evaluation"
EVALUATION_RESULTS_DIR = EVALUATION_DIR / "results"

# ---------------------------------------------------------------------------
# Embeddings (CLIP) settings
# ---------------------------------------------------------------------------
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
EMBEDDINGS_OUTPUT = EMBEDDINGS_DIR / "embeddings.npy"
EMBEDDINGS_FILENAMES_OUTPUT = EMBEDDINGS_DIR / "image_filenames.json"
CAPTION_EMBEDDINGS_OUTPUT = EMBEDDINGS_DIR / "caption_embeddings.npy"  # cached CLIP text embeddings
EMBEDDING_BATCH_SIZE = 32

# ---------------------------------------------------------------------------
# Scene Recognition (Places365) settings
# ---------------------------------------------------------------------------
PLACES365_WEIGHTS_PATH = MODELS_DIR / "resnet18_places365.pth.tar"
PLACES365_CATEGORIES_PATH = MODELS_DIR / "categories_places365.txt"

PLACES365_WEIGHTS_URL = "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar"
PLACES365_CATEGORIES_URL = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"

SCENE_PREDICTIONS_OUTPUT = SCENE_RECOGNITION_DIR / "scene_predictions.json"
SCENE_TOP_K = 3  # how many raw Places365 predictions to keep per image

# ---------------------------------------------------------------------------
# Image Captioning (BLIP) settings
# ---------------------------------------------------------------------------
BLIP_MODEL_NAME = "Salesforce/blip-image-captioning-base"
CAPTIONS_OUTPUT = CAPTIONING_DIR / "captions.json"
CAPTION_MAX_NEW_TOKENS = 30

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---------------------------------------------------------------------------
# Similarity Search (FAISS) settings
# ---------------------------------------------------------------------------
FAISS_INDEX_OUTPUT = SIMILARITY_SEARCH_DIR / "faiss_index.bin"
FAISS_INDEX_FILENAMES_OUTPUT = SIMILARITY_SEARCH_DIR / "indexed_filenames.json"
SIMILARITY_SEARCH_TOP_K = 5  # how many nearest neighbors to return per query

# ---------------------------------------------------------------------------
# Event Clustering settings
#
# Agglomerative Clustering (cosine distance) over combined image+caption CLIP
# embeddings - HDBSCAN was tried first but its density-based approach always
# treated the small, visually-diverse event groups as noise next to this
# dataset's much larger, near-duplicate scene photo pools. See
# event_clustering/event_clusterer.py for the full rationale.
# ---------------------------------------------------------------------------
CLUSTERS_OUTPUT = EVENT_CLUSTERING_DIR / "clusters.json"
EVENT_CLUSTERING_ALGORITHM = "agglomerative"  # or "hdbscan" (the spec's original choice)
EVENT_CLUSTERING_DISTANCE_THRESHOLD = 0.3  # cosine distance; lower = more, tighter clusters
EVENT_CLUSTERING_MIN_CLUSTER_SIZE = 3  # smallest group of photos considered its own event

# When True, run_clustering ignores the fixed threshold above and instead
# auto-selects one by maximizing the mean silhouette score over a sweep
# (addresses the "fixed threshold may not generalize" critique).
EVENT_CLUSTERING_AUTO_THRESHOLD = False
EVENT_CLUSTERING_THRESHOLD_SWEEP = [round(0.15 + 0.025 * i, 3) for i in range(17)]  # 0.15 .. 0.55

# ---------------------------------------------------------------------------
# Timeline Reconstruction settings
#
# Priority per image: EXIF capture time > file creation time > (if every
# cluster's timestamp is indistinguishable) cluster-similarity chaining.
# ---------------------------------------------------------------------------
TIMELINE_OUTPUT = TIMELINE_DIR / "timeline.json"
TIMELINE_TIMESTAMP_SPREAD_SECONDS = 300  # below this, timestamps are treated as uninformative
# (5 minutes: catches bulk-downloaded/extracted datasets with no real EXIF,
# while still trusting real photo timestamps that are usually hours/days apart)

# ---------------------------------------------------------------------------
# Story Generation settings (Groq API running a Llama model)
#
# Requires a GROQ_API_KEY environment variable - get one at
# https://console.groq.com/keys. Never hardcode the key in this file.
# ---------------------------------------------------------------------------
GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
STORY_GENERATION_OUTPUT = STORY_GENERATION_DIR / "stories.json"
STORY_CAPTIONS_PER_EVENT = 5  # how many sample captions to feed into the prompt

# Ensure required directories exist
for directory in (
    RAW_IMAGES_DIR, MODELS_DIR, EMBEDDINGS_DIR, SCENE_RECOGNITION_DIR,
    CAPTIONING_DIR, SIMILARITY_SEARCH_DIR, EVENT_CLUSTERING_DIR,
    TIMELINE_DIR, STORY_GENERATION_DIR, EVALUATION_DIR, EVALUATION_RESULTS_DIR,
):
    directory.mkdir(parents=True, exist_ok=True)
