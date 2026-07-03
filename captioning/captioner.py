"""
Image Captioning for MemoryLens.

Generates a natural-language caption for every image using a pretrained
BLIP image captioning model. No training is performed here - we only load
pretrained weights and run inference.
"""

import logging
from pathlib import Path
from typing import List, Optional

import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

logger = logging.getLogger(__name__)


class ImageCaptioner:
    """Generates captions for images using a pretrained BLIP model."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        """
        Args:
            model_name: HuggingFace model id, e.g. "Salesforce/blip-image-captioning-base".
            device: "cuda" or "cpu". Auto-detected if not given.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading BLIP model %s...", model_name)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(model_name).to(self.device)
        self.model.eval()
        logger.info("ImageCaptioner ready | device=%s", self.device)

    @torch.no_grad()
    def caption(self, image_path: Path, max_new_tokens: int = 30) -> str:
        """Generate a caption for a single image.

        Args:
            image_path: Path to the image file.
            max_new_tokens: Maximum caption length in tokens.

        Returns:
            The generated caption text.
        """
        image = Image.open(image_path).convert("RGB")
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        # repetition_penalty + no_repeat_ngram_size stop the decoder from
        # occasionally looping on a single token (e.g. "con con con con...").
        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            repetition_penalty=1.5,
            no_repeat_ngram_size=3,
        )
        return self.processor.decode(output_ids[0], skip_special_tokens=True).strip()

    def caption_batch(self, image_paths: List[Path], max_new_tokens: int = 30) -> List[dict]:
        """Caption many images, skipping any that fail to load.

        Args:
            image_paths: List of image file paths.
            max_new_tokens: Maximum caption length in tokens.

        Returns:
            List of {"image": filename, "caption": text} dicts, one per
            successfully processed image.
        """
        results = []
        for path in image_paths:
            try:
                caption = self.caption(path, max_new_tokens=max_new_tokens)
                results.append({"image": path.name, "caption": caption})
            except Exception as exc:
                logger.warning("Skipping %s due to error: %s", path.name, exc)
        return results
