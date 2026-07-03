"""
Builds a FAISS index over embeddings/embeddings.npy and saves it to
similarity_search/faiss_index.bin (+ similarity_search/indexed_filenames.json).

Also runs a demo nearest-neighbor search on the first image to confirm the
index works end-to-end.

Usage:
    python -m similarity_search.build_index
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
    FAISS_INDEX_OUTPUT,
    FAISS_INDEX_FILENAMES_OUTPUT,
    SIMILARITY_SEARCH_TOP_K,
)
from similarity_search.faiss_index import SimilaritySearchIndex  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run(
    embeddings_path: Path = EMBEDDINGS_OUTPUT,
    filenames_path: Path = EMBEDDINGS_FILENAMES_OUTPUT,
    index_output_path: Path = FAISS_INDEX_OUTPUT,
    index_filenames_output_path: Path = FAISS_INDEX_FILENAMES_OUTPUT,
) -> None:
    """Build a FAISS index from saved CLIP embeddings and persist it to disk.

    Args:
        embeddings_path: Path to embeddings.npy (from the embeddings step).
        filenames_path: Path to image_filenames.json (from the embeddings step).
        index_output_path: Where to write the FAISS index.
        index_filenames_output_path: Where to write the index's filename mapping.
    """
    if not embeddings_path.exists() or not filenames_path.exists():
        logger.error(
            "Embeddings not found. Run `python -m embeddings.generate_embeddings` first."
        )
        return

    embeddings = np.load(embeddings_path)
    with open(filenames_path) as f:
        filenames = json.load(f)

    logger.info("Building FAISS index for %d embeddings...", embeddings.shape[0])
    search_index = SimilaritySearchIndex()
    search_index.build(embeddings, filenames)
    search_index.save(index_output_path, index_filenames_output_path)

    # Demo: nearest neighbors of the first indexed image.
    query_filename = filenames[0]
    neighbors = search_index.search_by_filename(query_filename, k=SIMILARITY_SEARCH_TOP_K)
    logger.info("Nearest neighbors of %s:", query_filename)
    for rank, (name, score) in enumerate(neighbors, start=1):
        logger.info("  %d. %s (similarity=%.4f)", rank, name, score)


if __name__ == "__main__":
    run()
