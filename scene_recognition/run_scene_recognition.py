"""
Runs Scene Recognition over every image in data/raw_images and saves the
results to scene_recognition/scene_predictions.json.

Usage:
    python -m scene_recognition.run_scene_recognition
"""

import json
import logging
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    RAW_IMAGES_DIR,
    PLACES365_WEIGHTS_PATH,
    PLACES365_CATEGORIES_PATH,
    SCENE_PREDICTIONS_OUTPUT,
    SCENE_TOP_K,
    SUPPORTED_IMAGE_EXTENSIONS,
)
from scene_recognition.scene_recognizer import SceneRecognizer  # noqa: E402
from utils.image_utils import list_images  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def run(raw_images_dir: Path = RAW_IMAGES_DIR, output_path: Path = SCENE_PREDICTIONS_OUTPUT) -> None:
    """Recognize the scene of every image in `raw_images_dir` and save results as JSON.

    Args:
        raw_images_dir: Folder containing the dataset images.
        output_path: Where to write scene_predictions.json.
    """
    image_paths = list_images(raw_images_dir, SUPPORTED_IMAGE_EXTENSIONS)
    if not image_paths:
        logger.error("No images to process in %s. Add photos there and re-run.", raw_images_dir)
        return

    logger.info("Loading Places365 model...")
    recognizer = SceneRecognizer(
        weights_path=PLACES365_WEIGHTS_PATH,
        categories_path=PLACES365_CATEGORIES_PATH,
    )

    logger.info("Recognizing scenes for %d images...", len(image_paths))
    results = []
    for i, path in enumerate(image_paths, start=1):
        prediction = recognizer.predict(path, top_k=SCENE_TOP_K)
        results.append(prediction)
        logger.info("[%d/%d] %s -> %s (%.1f%%)", i, len(image_paths), path.name,
                    prediction["scene"], prediction["confidence"] * 100)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info("Saved %d predictions to %s", len(results), output_path)


if __name__ == "__main__":
    run()
