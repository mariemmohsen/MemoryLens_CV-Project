"""
Small shared helpers for working with image files.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

from PIL import Image, ExifTags

logger = logging.getLogger(__name__)

_EXIF_DATETIME_TAGS = {"DateTimeOriginal", "DateTime", "DateTimeDigitized"}
_EXIF_DATETIME_FORMAT = "%Y:%m:%d %H:%M:%S"


def get_capture_datetime(image_path: Path) -> Tuple[datetime, str]:
    """Best-available timestamp for an image: EXIF capture time if present,
    otherwise the file's creation time.

    Args:
        image_path: Path to the image file.

    Returns:
        (datetime, source) where source is "exif" or "file_time".
    """
    try:
        exif = Image.open(image_path).getexif()
        tag_names = {ExifTags.TAGS.get(tag_id): tag_id for tag_id in exif}
        for tag_name in _EXIF_DATETIME_TAGS:
            tag_id = tag_names.get(tag_name)
            if tag_id is not None and exif[tag_id]:
                return datetime.strptime(exif[tag_id], _EXIF_DATETIME_FORMAT), "exif"
    except Exception as exc:
        logger.debug("Could not read EXIF from %s: %s", image_path.name, exc)

    return datetime.fromtimestamp(image_path.stat().st_ctime), "file_time"


def list_images(folder: Path, extensions: set) -> List[Path]:
    """Return every image file in `folder` whose extension is in `extensions`.

    Args:
        folder: Directory to scan.
        extensions: Set of lowercase extensions to accept, e.g. {".jpg", ".png"}.

    Returns:
        Sorted list of image file paths. Empty list if the folder doesn't
        exist or contains no matching images.
    """
    if not folder.exists():
        logger.warning("Image folder does not exist: %s", folder)
        return []

    images = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    if not images:
        logger.warning("No images found in %s", folder)

    return images
