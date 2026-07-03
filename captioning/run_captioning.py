"""
Runs Image Captioning over every image in data/raw_images and saves the
results to captioning/captions.json.

Usage:
    python -m captioning.run_captioning
"""

import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    RAW_IMAGES_DIR,
    BLIP_MODEL_NAME,
    CAPTIONS_OUTPUT,
    CAPTION_MAX_NEW_TOKENS,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from captioning.captioner import ImageCaptioner  # noqa: E402
from utils.image_utils import list_images  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run(
    raw_images_dir: Path = RAW_IMAGES_DIR,
    output_path: Path = CAPTIONS_OUTPUT,
    skip_existing: bool = False,
) -> None:
    """Generate a caption for every image in `raw_images_dir` and save results as JSON.

    Args:
        raw_images_dir: Folder containing the dataset images.
        output_path: Where to write captions.json.
        skip_existing: If True, keep captions already in `output_path` and only
            caption new images (BLIP is the slowest pipeline stage, so this
            makes re-processing after adding photos much faster).
    """
    image_paths = list_images(raw_images_dir, SUPPORTED_IMAGE_EXTENSIONS)
    if not image_paths:
        logger.error("No images to process in %s. Add photos there and re-run.", raw_images_dir)
        return

    existing = {}
    if skip_existing and output_path.exists():
        current_names = {p.name for p in image_paths}
        with open(output_path) as f:
            existing = {
                item["image"]: item["caption"]
                for item in json.load(f)
                if item["image"] in current_names  # drop captions of removed images
            }
        image_paths = [p for p in image_paths if p.name not in existing]
        logger.info("Keeping %d existing captions; %d images to caption", len(existing), len(image_paths))

    results = [{"image": name, "caption": caption} for name, caption in existing.items()]
    if image_paths:
        logger.info("Loading BLIP model...")
        captioner = ImageCaptioner(model_name=BLIP_MODEL_NAME)

        logger.info("Captioning %d images...", len(image_paths))
        for i, path in enumerate(image_paths, start=1):
            caption = captioner.caption(path, max_new_tokens=CAPTION_MAX_NEW_TOKENS)
            results.append({"image": path.name, "caption": caption})
            logger.info("[%d/%d] %s -> %s", i, len(image_paths), path.name, caption)

    results.sort(key=lambda item: item["image"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved %d captions to %s", len(results), output_path)


if __name__ == "__main__":
    run()
