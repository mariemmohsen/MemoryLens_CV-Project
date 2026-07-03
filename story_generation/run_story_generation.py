"""
Runs Story Generation over timeline/timeline.json (or event_clustering/clusters.json
if no timeline exists yet) and saves the results to story_generation/stories.json.

Requires a GROQ_API_KEY environment variable - get one at
https://console.groq.com/keys and set it before running, e.g.:
    PowerShell:  $env:GROQ_API_KEY = "..."
    bash:        export GROQ_API_KEY="..."

Usage:
    python -m story_generation.run_story_generation
"""

import json
import logging
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import (  # noqa: E402
    TIMELINE_OUTPUT,
    CLUSTERS_OUTPUT,
    SCENE_PREDICTIONS_OUTPUT,
    CAPTIONS_OUTPUT,
    STORY_GENERATION_OUTPUT,
    GROQ_MODEL_NAME,
    STORY_CAPTIONS_PER_EVENT,
)
from event_clustering.event_clusterer import NOISE_LABEL  # noqa: E402
from story_generation.story_generator import StoryGenerator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _load_by_filename(path: Path, value_key: str) -> Dict[str, str]:
    """Load a results JSON file (list of dicts with an "image" key) into a
    {filename: value} lookup. Returns {} if the file doesn't exist yet."""
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)
    return {item["image"]: item[value_key] for item in data}


def _dominant_scene(images: List[str], scenes_by_filename: Dict[str, str]) -> str:
    """Most common non-"Other" scene label among a cluster's images."""
    counts = Counter(
        scenes_by_filename[name] for name in images if name in scenes_by_filename
    )
    return counts.most_common(1)[0][0] if counts else ""


def run(
    timeline_path: Path = TIMELINE_OUTPUT,
    clusters_path: Path = CLUSTERS_OUTPUT,
    scene_predictions_path: Path = SCENE_PREDICTIONS_OUTPUT,
    captions_path: Path = CAPTIONS_OUTPUT,
    output_path: Path = STORY_GENERATION_OUTPUT,
    model_name: str = GROQ_MODEL_NAME,
) -> None:
    """Generate a title + short story for every event and save results as JSON.

    Args:
        timeline_path: Path to timeline.json (preferred, keeps events ordered).
        clusters_path: Path to clusters.json, used if no timeline exists yet.
        scene_predictions_path: Path to scene_predictions.json.
        captions_path: Path to captions.json.
        output_path: Where to write stories.json.
        model_name: Groq model id to use.
    """
    if timeline_path.exists():
        with open(timeline_path) as f:
            events = json.load(f)
    elif clusters_path.exists():
        with open(clusters_path) as f:
            clusters = json.load(f)
        events = [c for c in clusters if c["cluster_id"] != NOISE_LABEL]
    else:
        logger.error(
            "No clusters/timeline found. Run event_clustering (and optionally timeline) first."
        )
        return

    if not events:
        logger.error("No events to generate stories for.")
        return

    scenes_by_filename = _load_by_filename(scene_predictions_path, "scene")
    captions_by_filename = _load_by_filename(captions_path, "caption")

    generator = StoryGenerator(model_name=model_name)

    stories = []
    for event in events:
        images = event["images"]
        sample_captions = [
            captions_by_filename[name]
            for name in images[:STORY_CAPTIONS_PER_EVENT]
            if name in captions_by_filename
        ]
        scene = _dominant_scene(images, scenes_by_filename)

        result = generator.generate(event["title"], scene, sample_captions)
        stories.append({
            "cluster_id": event["cluster_id"],
            "cluster_title": event["title"],
            "title": result["title"],
            "story": result["story"],
            "size": event["size"],
        })
        logger.info('%s -> "%s": %s', event["title"], result["title"], result["story"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(stories, f, indent=2)

    logger.info("Saved %d stories to %s", len(stories), output_path)


if __name__ == "__main__":
    run()
