"""
Scene Recognition for MemoryLens.

Recognizes the scene/environment of a photo (e.g. Beach, Restaurant, Museum,
Street...) using a pretrained Places365 ResNet-18 model. No training is
performed here - we only load pretrained weights and run inference.

Places365 predicts 365 fine-grained categories (e.g. "boardwalk",
"restaurant_kitchen"). We map those down to the simple scene labels
MemoryLens cares about (Beach, Restaurant, Hotel, Museum, Airport, Park,
Street, Classroom, ...). Anything unmapped is labeled "Other".
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

logger = logging.getLogger(__name__)

# Fine-grained Places365 label -> simple MemoryLens scene category.
# Labels come from categories_places365.txt (with the leading "/a/" etc. stripped).
SCENE_CATEGORY_MAP: Dict[str, str] = {
    # Beach
    "beach": "Beach",
    "beach_house": "Beach",
    "coast": "Beach",
    "boardwalk": "Beach",
    "ocean": "Beach",
    "lagoon": "Beach",
    # Restaurant
    "restaurant": "Restaurant",
    "restaurant_kitchen": "Restaurant",
    "restaurant_patio": "Restaurant",
    "diner/outdoor": "Restaurant",
    "cafeteria": "Restaurant",
    "food_court": "Restaurant",
    "bistro/indoor": "Restaurant",
    "bar": "Restaurant",
    "pub/indoor": "Restaurant",
    "coffee_shop": "Restaurant",
    "bakery/shop": "Restaurant",
    # Hotel
    "hotel/outdoor": "Hotel",
    "hotel_room": "Hotel",
    "resort": "Hotel",
    "motel": "Hotel",
    "lobby": "Hotel",
    # Museum
    "museum/indoor": "Museum",
    "museum/outdoor": "Museum",
    "art_gallery": "Museum",
    "art_studio": "Museum",
    "artists_loft": "Museum",
    # Airport
    "airport_terminal": "Airport",
    "airfield": "Airport",
    "runway": "Airport",
    "control_tower/outdoor": "Airport",
    "airplane_cabin": "Airport",
    # Park
    "park": "Park",
    "amusement_park": "Park",
    "botanical_garden": "Park",
    "picnic_area": "Park",
    "playground": "Park",
    "formal_garden": "Park",
    "japanese_garden": "Park",
    "forest_path": "Park",
    # Street
    "street": "Street",
    "alley": "Street",
    "crosswalk": "Street",
    "downtown": "Street",
    "plaza": "Street",
    "courtyard": "Street",
    # Classroom
    "classroom": "Classroom",
    "lecture_room": "Classroom",
    "kindergarden_classroom": "Classroom",
    "conference_room": "Classroom",
}

# Standard ImageNet-style preprocessing used by the official Places365 models.
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class SceneRecognizer:
    """Predicts the scene category of images using pretrained Places365."""

    def __init__(
        self,
        weights_path: Path,
        categories_path: Path,
        device: Optional[str] = None,
    ) -> None:
        """
        Args:
            weights_path: Path to the resnet18_places365.pth.tar checkpoint.
            categories_path: Path to categories_places365.txt.
            device: "cuda" or "cpu". Auto-detected if not given.
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.categories = self._load_categories(categories_path)
        self.model = self._load_model(weights_path)
        self.model.eval()
        logger.info(
            "SceneRecognizer ready | device=%s | categories=%d",
            self.device, len(self.categories),
        )

    @staticmethod
    def _load_categories(categories_path: Path) -> List[str]:
        """Read the 365 Places category names from the categories file.

        Each line looks like: "/a/airport_terminal 0"
        We keep just the label part (e.g. "airport_terminal").
        """
        if not categories_path.exists():
            raise FileNotFoundError(
                f"Categories file not found at {categories_path}.\n"
                f"Download it with: utils/download_places365.py"
            )

        categories = []
        with open(categories_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path_part = line.split(" ")[0]  # e.g. "/a/airport_terminal"
                label = path_part[3:]  # drop the leading "/x/"
                categories.append(label)
        return categories

    def _load_model(self, weights_path: Path) -> nn.Module:
        """Build a ResNet-18 and load the pretrained Places365 weights into it."""
        if not weights_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at {weights_path}.\n"
                f"Download them with: utils/download_places365.py"
            )

        model = models.resnet18(num_classes=365)
        checkpoint = torch.load(weights_path, map_location=self.device)

        # The official checkpoint stores weights under "state_dict" with a
        # "module." prefix (from training with nn.DataParallel). Strip it.
        state_dict = {
            key.replace("module.", ""): value
            for key, value in checkpoint["state_dict"].items()
        }
        model.load_state_dict(state_dict)
        return model.to(self.device)

    @staticmethod
    def _map_to_simple_category(raw_label: str) -> str:
        """Map a fine-grained Places365 label to a simple MemoryLens scene."""
        return SCENE_CATEGORY_MAP.get(raw_label, "Other")

    @torch.no_grad()
    def predict(self, image_path: Path, top_k: int = 3) -> Dict:
        """Predict the scene of a single image.

        Args:
            image_path: Path to the image file.
            top_k: How many raw Places365 predictions to include.

        Returns:
            Dict with the simplified scene, the raw Places365 label,
            confidence, and the top_k raw predictions.
        """
        image = Image.open(image_path).convert("RGB")
        input_tensor = IMAGE_TRANSFORM(image).unsqueeze(0).to(self.device)

        logits = self.model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        top_probs, top_idxs = probabilities.topk(top_k)

        top_predictions = [
            {"label": self.categories[idx], "confidence": round(prob.item(), 4)}
            for prob, idx in zip(top_probs, top_idxs)
        ]

        best_label = top_predictions[0]["label"]
        best_confidence = top_predictions[0]["confidence"]

        return {
            "image": image_path.name,
            "scene": self._map_to_simple_category(best_label),
            "raw_label": best_label,
            "confidence": best_confidence,
            "top_predictions": top_predictions,
        }

    def predict_batch(self, image_paths: List[Path], top_k: int = 3) -> List[Dict]:
        """Predict scenes for many images, skipping any that fail to load.

        Args:
            image_paths: List of image file paths.
            top_k: How many raw Places365 predictions to keep per image.

        Returns:
            List of prediction dicts (see `predict`), one per successfully
            processed image.
        """
        results = []
        for path in image_paths:
            try:
                results.append(self.predict(path, top_k=top_k))
            except Exception as exc:
                logger.warning("Skipping %s due to error: %s", path.name, exc)
        return results
