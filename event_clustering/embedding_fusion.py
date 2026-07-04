"""
Shared embedding construction for event clustering and its evaluation.

Both the production clustering step and the evaluation/ablation experiments
must build image, caption, and fused embeddings the exact same way, so that
logic lives here instead of being duplicated. CLIP text embeddings of the
captions are cached to disk (they're deterministic and slow to recompute),
which makes the threshold/modality ablations fast and reproducible.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def caption_text_embeddings(
    filenames: List[str],
    captions_by_filename: Dict[str, str],
    clip_model_name: str,
    cache_path: Optional[Path] = None,
) -> np.ndarray:
    """CLIP text embedding of each image's caption, in `filenames` order.

    Uses a cached `.npy` when its row count matches (so repeated experiments
    don't reload CLIP); otherwise computes the embeddings and writes the cache.

    Args:
        filenames: Image filenames defining row order.
        captions_by_filename: {filename: caption}.
        clip_model_name: HuggingFace CLIP model id.
        cache_path: Optional path to read/write the cached embeddings.

    Returns:
        (N, D) L2-normalized float32 array aligned with `filenames`.
    """
    if cache_path is not None and cache_path.exists():
        cached = np.load(cache_path)
        if cached.shape[0] == len(filenames):
            logger.info("Loaded cached caption embeddings from %s", cache_path)
            return cached
        logger.info("Caption-embedding cache is stale (%d vs %d rows); recomputing",
                    cached.shape[0], len(filenames))

    from embeddings.clip_embedder import ClipEmbedder  # local import: heavy dependency

    logger.info("Embedding %d captions with CLIP...", len(filenames))
    embedder = ClipEmbedder(model_name=clip_model_name)
    captions = [captions_by_filename.get(name, "") for name in filenames]
    text_embeddings = embedder.embed_texts(captions)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, text_embeddings)
        logger.info("Cached caption embeddings to %s", cache_path)
    return text_embeddings


def fuse(image_embeddings: np.ndarray, text_embeddings: np.ndarray) -> np.ndarray:
    """Combine image + caption embeddings into one L2-normalized space."""
    return _l2_normalize(image_embeddings + text_embeddings)


def build_modalities(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
) -> Dict[str, np.ndarray]:
    """The three embedding variants compared in the ablation study.

    Returns:
        {"image": ..., "caption": ..., "fused": ...}, each L2-normalized.
    """
    return {
        "image": _l2_normalize(image_embeddings),
        "caption": _l2_normalize(text_embeddings),
        "fused": fuse(image_embeddings, text_embeddings),
    }


def load_pipeline_arrays(
    embeddings_path: Path,
    filenames_path: Path,
    captions_path: Path,
):
    """Load image embeddings, filenames, and the caption lookup off disk.

    Returns:
        (image_embeddings, filenames, captions_by_filename).
    """
    image_embeddings = np.load(embeddings_path)
    with open(filenames_path) as f:
        filenames = json.load(f)
    captions_by_filename = {}
    if captions_path.exists():
        with open(captions_path) as f:
            captions_by_filename = {item["image"]: item["caption"] for item in json.load(f)}
    return image_embeddings, filenames, captions_by_filename
