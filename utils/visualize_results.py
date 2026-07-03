"""
Draws scene-recognition and captioning results onto sample images so the
pipeline output can be checked visually instead of reading raw JSON.

Usage:
    python -m utils.visualize_results --count 6
"""

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    RAW_IMAGES_DIR,
    SCENE_PREDICTIONS_OUTPUT,
    CAPTIONS_OUTPUT,
    PROJECT_ROOT,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ANNOTATED_DIR = PROJECT_ROOT / "outputs" / "annotated"
BANNER_HEIGHT = 70
BANNER_COLOR = (0, 0, 0)
TEXT_COLOR = (255, 255, 255)


def _load_by_filename(path: Path) -> dict:
    """Load a results JSON file (list of dicts with an "image" key) into a
    {filename: entry} lookup. Returns {} if the file doesn't exist yet."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {item["image"]: item for item in data}


def annotate_image(image_path: Path, scene: str, caption: str) -> Image.Image:
    """Return a copy of `image_path` with a text banner showing scene + caption."""
    image = Image.open(image_path).convert("RGB")
    banner = Image.new("RGB", (image.width, BANNER_HEIGHT), BANNER_COLOR)
    draw = ImageDraw.Draw(banner)
    draw.text((10, 8), f"Scene: {scene}", fill=TEXT_COLOR)
    draw.text((10, 34), f"Caption: {caption}", fill=TEXT_COLOR)

    combined = Image.new("RGB", (image.width, image.height + BANNER_HEIGHT))
    combined.paste(image, (0, 0))
    combined.paste(banner, (0, image.height))
    return combined


def run(count: int = 6) -> None:
    """Annotate `count` sample images with their scene + caption and save them
    to outputs/annotated/."""
    scenes = _load_by_filename(SCENE_PREDICTIONS_OUTPUT)
    captions = _load_by_filename(CAPTIONS_OUTPUT)

    if not scenes and not captions:
        logger.error(
            "No results found. Run scene_recognition and/or captioning first."
        )
        return

    filenames = sorted(set(scenes) | set(captions))[:count]
    ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)

    for filename in filenames:
        image_path = RAW_IMAGES_DIR / filename
        if not image_path.exists():
            logger.warning("Image not found, skipping: %s", image_path)
            continue

        scene = scenes.get(filename, {}).get("scene", "?")
        caption = captions.get(filename, {}).get("caption", "?")
        annotated = annotate_image(image_path, scene, caption)

        out_path = ANNOTATED_DIR / filename
        annotated.save(out_path)
        logger.info("Saved %s", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=6, help="Number of sample images to annotate")
    args = parser.parse_args()
    run(count=args.count)
