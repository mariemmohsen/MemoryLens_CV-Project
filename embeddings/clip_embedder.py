"""
CLIP Embeddings for MemoryLens.

Generates one semantic embedding per image using a pretrained CLIP model.
No training is performed here - we only load pretrained weights and run
inference. Embeddings are L2-normalized so cosine similarity can later be
computed with a plain dot product (used by FAISS similarity search).
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

logger = logging.getLogger(__name__)


class ClipEmbedder:
    """Generates CLIP image embeddings."""

    def __init__(self, model_name: str, device: Optional[str] = None) -> None:
        """
        Args:
            model_name: HuggingFace model id, e.g. "openai/clip-vit-base-patch32".
            device: "cuda" or "cpu". Auto-detected if not given.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Loading CLIP model %s...", model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        self.embedding_dim = self.model.config.projection_dim
        logger.info(
            "ClipEmbedder ready | device=%s | embedding_dim=%d",
            self.device, self.embedding_dim,
        )

    @torch.no_grad()
    def embed_batch(self, image_paths: List[Path]) -> Tuple[np.ndarray, List[str]]:
        """Embed a batch of images, skipping any that fail to load.

        Args:
            image_paths: Batch of image file paths.

        Returns:
            embeddings: (B, D) L2-normalized float32 array.
            filenames: Names of the successfully embedded images, same order
                as `embeddings`.
        """
        images = []
        filenames = []
        for path in image_paths:
            try:
                images.append(Image.open(path).convert("RGB"))
                filenames.append(path.name)
            except Exception as exc:
                logger.warning("Skipping %s due to error: %s", path.name, exc)

        if not images:
            return np.empty((0, self.embedding_dim), dtype="float32"), []

        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        outputs = self.model.get_image_features(**inputs)
        # Newer transformers versions wrap the projected embedding in
        # BaseModelOutputWithPooling (as `.pooler_output`); older versions
        # return the tensor directly.
        image_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
        return image_features.cpu().numpy().astype("float32"), filenames

    def embed_all(self, image_paths: List[Path], batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
        """Embed every image, processing in batches for efficiency.

        Args:
            image_paths: List of image file paths.
            batch_size: Number of images per forward pass.

        Returns:
            embeddings: (N, D) L2-normalized float32 array.
            filenames: Names of the successfully embedded images, same order
                as `embeddings` (rows for skipped/unreadable images are omitted).
        """
        all_embeddings = []
        all_filenames = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[start:start + batch_size]
            embeddings, filenames = self.embed_batch(batch_paths)
            if embeddings.shape[0] == 0:
                continue
            all_embeddings.append(embeddings)
            all_filenames.extend(filenames)
            logger.info("Embedded %d/%d images", len(all_filenames), len(image_paths))

        if not all_embeddings:
            return np.empty((0, self.embedding_dim), dtype="float32"), []

        return np.concatenate(all_embeddings, axis=0), all_filenames

    @torch.no_grad()
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed a list of texts (e.g. captions) into the same CLIP space as images.

        Args:
            texts: List of text strings.

        Returns:
            (N, D) L2-normalized float32 array, one row per text.
        """
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        outputs = self.model.get_text_features(**inputs)
        text_features = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        return text_features.cpu().numpy().astype("float32")
