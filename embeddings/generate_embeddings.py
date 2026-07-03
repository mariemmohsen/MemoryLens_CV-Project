"""
Runs CLIP embedding generation over every image in data/raw_images and saves
the results to embeddings/embeddings.npy (+ embeddings/image_filenames.json).

Usage:
    python -m embeddings.generate_embeddings
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    RAW_IMAGES_DIR,
    CLIP_MODEL_NAME,
    EMBEDDINGS_OUTPUT,
    EMBEDDINGS_FILENAMES_OUTPUT,
    EMBEDDING_BATCH_SIZE,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from embeddings.clip_embedder import ClipEmbedder  # noqa: E402
from utils.image_utils import list_images  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run(
    raw_images_dir: Path = RAW_IMAGES_DIR,
    output_path: Path = EMBEDDINGS_OUTPUT,
    filenames_output_path: Path = EMBEDDINGS_FILENAMES_OUTPUT,
) -> None:
    """Generate a CLIP embedding for every image in `raw_images_dir`.

    Args:
        raw_images_dir: Folder containing the dataset images.
        output_path: Where to write embeddings.npy.
        filenames_output_path: Where to write the filename-to-row mapping,
            since embeddings.npy alone has no image labels.
    """
    image_paths = list_images(raw_images_dir, SUPPORTED_IMAGE_EXTENSIONS)
    if not image_paths:
        logger.error("No images to process in %s. Add photos there and re-run.", raw_images_dir)
        return

    logger.info("Loading CLIP model...")
    embedder = ClipEmbedder(model_name=CLIP_MODEL_NAME)

    logger.info("Generating embeddings for %d images...", len(image_paths))
    embeddings, filenames = embedder.embed_all(image_paths, batch_size=EMBEDDING_BATCH_SIZE)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, embeddings)
    with open(filenames_output_path, "w") as f:
        json.dump(filenames, f, indent=2)

    logger.info(
        "Saved %d embeddings (dim=%d) to %s", embeddings.shape[0], embeddings.shape[1], output_path
    )


if __name__ == "__main__":
    run()
