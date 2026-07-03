"""
Downloads the pretrained Places365 ResNet-18 weights and category list.

Run this once before using SceneRecognizer:
    python -m utils.download_places365
"""

import logging
import urllib.request
from pathlib import Path

import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    PLACES365_WEIGHTS_PATH,
    PLACES365_WEIGHTS_URL,
    PLACES365_CATEGORIES_PATH,
    PLACES365_CATEGORIES_URL,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def download_file(url: str, destination: Path) -> None:
    """Download a file to `destination` if it doesn't already exist."""
    if destination.exists():
        logger.info("Already exists, skipping: %s", destination)
        return

    logger.info("Downloading %s -> %s", url, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, destination)
    logger.info("Done: %s", destination)


def main() -> None:
    """Download both the Places365 weights and the category labels."""
    download_file(PLACES365_CATEGORIES_URL, PLACES365_CATEGORIES_PATH)
    download_file(PLACES365_WEIGHTS_URL, PLACES365_WEIGHTS_PATH)
    logger.info("Places365 resources are ready.")


if __name__ == "__main__":
    main()
