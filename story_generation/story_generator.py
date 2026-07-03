"""
Story Generation for MemoryLens.

For each event, generates a short title and a warm narrative paragraph using
a Llama model hosted on Groq (fast, cheap Llama inference). Requires a
GROQ_API_KEY environment variable - get one at
https://console.groq.com/keys. The key is never hardcoded or logged.
"""

import json
import logging
import os
from typing import Dict, List, Optional

from groq import Groq

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are helping build a photo-memory app. Given a description of one "
    "event (its dominant scene, and captions of a few of its photos), write "
    "a short, warm, second-person recap of that event, as if reminding the "
    "user what they did.\n"
    'Respond with strict JSON only: {"title": "...", "story": "..."}\n'
    "title: 2-4 words, e.g. \"Beach Sunset\".\n"
    "story: 1-2 sentences, second person (\"You spent...\")."
)


class StoryGenerator:
    """Generates an event title + short story from cluster info via Groq/Llama."""

    def __init__(self, model_name: str, api_key: Optional[str] = None) -> None:
        """
        Args:
            model_name: Groq model id, e.g. "llama-3.3-70b-versatile".
            api_key: Groq API key. Defaults to the GROQ_API_KEY environment variable.
        """
        api_key = api_key or os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY is not set. Get a key at https://console.groq.com/keys "
                "and set it as an environment variable before running this step."
            )
        self.client = Groq(api_key=api_key)
        self.model_name = model_name

    def generate(self, cluster_title: str, scene: str, captions: List[str]) -> Dict[str, str]:
        """Generate a title + story for one event.

        Args:
            cluster_title: The cluster's auto-generated title (e.g. "Cake", "Beach").
            scene: Dominant scene label for the event, if known (e.g. "Restaurant").
            captions: Captions of a few representative photos from the event.

        Returns:
            {"title": ..., "story": ...}
        """
        user_prompt = (
            f"Event label: {cluster_title}\n"
            f"Dominant scene: {scene or 'unknown'}\n"
            "Sample photo captions:\n" + "\n".join(f"- {c}" for c in captions)
        )
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.7,
        )
        result = json.loads(response.choices[0].message.content)
        return {"title": result["title"].strip(), "story": result["story"].strip()}
