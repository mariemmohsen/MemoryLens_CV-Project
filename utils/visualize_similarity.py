"""
Builds a side-by-side montage of a query image and its FAISS nearest
neighbors, so similarity search results can be checked visually.

Usage:
    python -m utils.visualize_similarity <query_filename> [--k 5]
"""

import argparse
import logging
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    RAW_IMAGES_DIR,
    FAISS_INDEX_OUTPUT,
    FAISS_INDEX_FILENAMES_OUTPUT,
    SIMILARITY_SEARCH_TOP_K,
    PROJECT_ROOT,
)
from similarity_search.faiss_index import SimilaritySearchIndex  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "similarity"
THUMB_SIZE = (220, 220)
BANNER_HEIGHT = 24
BANNER_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)
BORDER_COLOR = (255, 0, 0)


def _thumbnail_with_label(image_path: Path, label: str, border: bool = False) -> Image.Image:
    """Return a fixed-size thumbnail with a text label banner underneath."""
    image = Image.open(image_path).convert("RGB")
    image.thumbnail(THUMB_SIZE)

    canvas = Image.new("RGB", THUMB_SIZE, (30, 30, 30))
    offset = ((THUMB_SIZE[0] - image.width) // 2, (THUMB_SIZE[1] - image.height) // 2)
    canvas.paste(image, offset)

    if border:
        draw = ImageDraw.Draw(canvas)
        draw.rectangle([0, 0, THUMB_SIZE[0] - 1, THUMB_SIZE[1] - 1], outline=BORDER_COLOR, width=4)

    banner = Image.new("RGB", (THUMB_SIZE[0], BANNER_HEIGHT), BANNER_COLOR)
    ImageDraw.Draw(banner).text((4, 4), label, fill=TEXT_COLOR)

    tile = Image.new("RGB", (THUMB_SIZE[0], THUMB_SIZE[1] + BANNER_HEIGHT))
    tile.paste(canvas, (0, 0))
    tile.paste(banner, (0, THUMB_SIZE[1]))
    return tile


def run(query_filename: str, k: int = SIMILARITY_SEARCH_TOP_K) -> None:
    """Build and save a query + nearest-neighbors montage.

    Args:
        query_filename: Filename of an image already in the FAISS index.
        k: Number of neighbors to show.
    """
    if not FAISS_INDEX_OUTPUT.exists():
        logger.error("No FAISS index found. Run `python -m similarity_search.build_index` first.")
        return

    search_index = SimilaritySearchIndex.load(FAISS_INDEX_OUTPUT, FAISS_INDEX_FILENAMES_OUTPUT)
    neighbors = search_index.search_by_filename(query_filename, k=k)

    tiles = [_thumbnail_with_label(RAW_IMAGES_DIR / query_filename, "QUERY", border=True)]
    for name, score in neighbors:
        tiles.append(_thumbnail_with_label(RAW_IMAGES_DIR / name, f"{score:.3f}"))

    montage = Image.new("RGB", (sum(t.width for t in tiles), tiles[0].height), (255, 255, 255))
    x = 0
    for tile in tiles:
        montage.paste(tile, (x, 0))
        x += tile.width

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / f"{Path(query_filename).stem}_neighbors.jpg"
    montage.save(out_path)
    logger.info("Saved %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("query_filename", help="Filename of the query image (must be in the index)")
    parser.add_argument("--k", type=int, default=SIMILARITY_SEARCH_TOP_K)
    args = parser.parse_args()
    run(args.query_filename, k=args.k)
