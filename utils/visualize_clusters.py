"""
Builds one thumbnail-grid image per event cluster, so clustering results can
be checked visually instead of reading clusters.json.

Usage:
    python -m utils.visualize_clusters [--samples-per-cluster 6] [--include-noise]
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import RAW_IMAGES_DIR, CLUSTERS_OUTPUT, PROJECT_ROOT  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "clusters"
THUMB_SIZE = (180, 180)
HEADER_HEIGHT = 36
HEADER_COLOR = (20, 20, 20)
TEXT_COLOR = (255, 255, 255)


def _thumbnail(image_path: Path) -> Image.Image:
    """Return a fixed-size, centered thumbnail."""
    image = Image.open(image_path).convert("RGB")
    image.thumbnail(THUMB_SIZE)
    canvas = Image.new("RGB", THUMB_SIZE, (30, 30, 30))
    offset = ((THUMB_SIZE[0] - image.width) // 2, (THUMB_SIZE[1] - image.height) // 2)
    canvas.paste(image, offset)
    return canvas


def build_cluster_grid(cluster: dict, samples_per_cluster: int) -> Image.Image:
    """Build a header + row-of-thumbnails image for one cluster."""
    sample_images = cluster["images"][:samples_per_cluster]
    tiles = [_thumbnail(RAW_IMAGES_DIR / name) for name in sample_images]

    grid_width = THUMB_SIZE[0] * len(tiles) if tiles else THUMB_SIZE[0]
    header = Image.new("RGB", (grid_width, HEADER_HEIGHT), HEADER_COLOR)
    ImageDraw.Draw(header).text(
        (8, 8),
        f"Cluster {cluster['cluster_id']}: {cluster['title']} ({cluster['size']} images)",
        fill=TEXT_COLOR,
    )

    combined = Image.new("RGB", (grid_width, HEADER_HEIGHT + THUMB_SIZE[1]), (255, 255, 255))
    combined.paste(header, (0, 0))
    x = 0
    for tile in tiles:
        combined.paste(tile, (x, HEADER_HEIGHT))
        x += tile.width
    return combined


def run(samples_per_cluster: int = 6, include_noise: bool = False) -> None:
    """Save one grid image per cluster (skipping the noise cluster by default)."""
    if not CLUSTERS_OUTPUT.exists():
        logger.error("No clusters found. Run `python -m event_clustering.run_clustering` first.")
        return

    with open(CLUSTERS_OUTPUT) as f:
        clusters = json.load(f)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for cluster in clusters:
        if cluster["cluster_id"] == -1 and not include_noise:
            continue
        grid = build_cluster_grid(cluster, samples_per_cluster)
        out_path = OUTPUT_DIR / f"cluster_{cluster['cluster_id']}_{cluster['title']}.jpg"
        grid.save(out_path)
        logger.info("Saved %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-cluster", type=int, default=6)
    parser.add_argument("--include-noise", action="store_true")
    args = parser.parse_args()
    run(samples_per_cluster=args.samples_per_cluster, include_noise=args.include_noise)
