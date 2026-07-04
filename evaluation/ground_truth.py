"""
Ground-truth (pseudo-label) extraction for evaluation.

The demo dataset encodes each photo's true category in its filename:
  - Scene stock photos:  "000_airport_airport_unsplash_<hash>.jpg" -> "airport"
    (categories: airport, restaurant, museum, beach; 50 photos each)
  - Curated events:      "birthday_party_03.jpg" -> "birthday_party"
    (10 events, ~5 photos each)
  - Filler images:       "sample_012.jpg", "hero.jpg" -> None (no defined class)

These are *weak* labels derived from how the dataset was assembled, not
hand-verified annotations, so they're best described as pseudo-ground-truth.
They are still a valid, honest basis for extrinsic clustering metrics because
each labeled group is a genuinely distinct scene or event.
"""

import re
from typing import List, Optional

SCENE_CLASSES = ("airport", "restaurant", "museum", "beach")
EVENT_CLASSES = (
    "birthday_party", "family_beach_day", "family_christmas", "family_dinner",
    "family_gathering", "family_picnic", "family_vacation", "graduation_party",
    "kids_playing", "wedding_celebration",
)

_SCENE_RE = re.compile(r"^\d+_(" + "|".join(SCENE_CLASSES) + r")_\1_unsplash_")
_EVENT_RE = re.compile(r"^(" + "|".join(EVENT_CLASSES) + r")_\d+")


def ground_truth_label(filename: str) -> Optional[str]:
    """Return the pseudo-ground-truth class for a filename, or None if unlabeled."""
    scene = _SCENE_RE.match(filename)
    if scene:
        return scene.group(1)
    event = _EVENT_RE.match(filename)
    if event:
        return event.group(1)
    return None


def load_ground_truth(filenames: List[str]) -> List[Optional[str]]:
    """Ground-truth label per filename (None for unlabeled filler images)."""
    return [ground_truth_label(name) for name in filenames]
