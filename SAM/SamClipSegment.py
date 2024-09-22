import os
from typing import Optional
import urllib
from functools import lru_cache
from typing import Any, Callable, Dict, List, Tuple
import clip
import cv2
import numpy as np
from random import randint
import PIL
import torch
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPProcessor, CLIPModel
import torchvision.transforms as transforms

CHECKPOINT_PATH = os.path.join(os.path.expanduser("~"), ".cache", "SAM")
CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
CHECKPOINT_URL = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
MODEL_TYPE = "default"
MAX_WIDTH = MAX_HEIGHT = 256  # Set according to your input size (256x256)
TOP_K_OBJ = 100


device = torch.device("cpu")
print(f"Used Device {device})")

@lru_cache
def load_sam_mask_generator():
    print("Loading SAM Mask Generator...")

    # Create the checkpoint directory if it does not exist
    if not os.path.exists(CHECKPOINT_PATH):
        os.makedirs(CHECKPOINT_PATH)

    checkpoint = os.path.join(CHECKPOINT_PATH, CHECKPOINT_NAME)
    
    # Download the checkpoint if it does not exist
    if not os.path.exists(checkpoint):
        print(f"Downloading checkpoint from {CHECKPOINT_URL}...")
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint)
        print(f"Downloaded checkpoint successfully to {checkpoint}")

    sam_model = sam_model_registry["vit_h"](checkpoint=checkpoint).to(device)
    mask_generator = SamAutomaticMaskGenerator(sam_model)
    print(f"Model downloaded successfully")

    return mask_generator

@lru_cache
def load_clip(name: str = "ViT-B/32") -> Tuple[torch.nn.Module, Callable[[PIL.Image.Image], torch.Tensor]]:
    model, preprocess = clip.load(name, device=device)
    return model.to(device), preprocess

def adjust_image_size(image: np.ndarray) -> np.ndarray:
    """Resize image to fit within the max width and height while maintaining aspect ratio."""
    height, width = image.shape[:2]
    if height > width:
        if height > MAX_HEIGHT:
            height, width = MAX_HEIGHT, int(MAX_HEIGHT / height * width)
    else:
        if width > MAX_WIDTH:
            height, width = int(MAX_WIDTH / width * height), MAX_WIDTH

    # Ensure image is a NumPy array for OpenCV functions
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()

    image = cv2.resize(image, (width, height))  # Resize with OpenCV
    return image

@torch.no_grad()
def get_score(crop: PIL.Image.Image, texts: List[str]) -> torch.Tensor:
    model, preprocess = load_clip()
    preprocessed = preprocess(crop).unsqueeze(0).to(device).float()  # Ensure float32
    tokens = clip.tokenize(texts).to(device)
    logits_per_image, _ = model(preprocessed, tokens)
    similarity = logits_per_image.softmax(-1).cpu()
    return similarity[0, 0]

def crop_image(image: np.ndarray, mask: Dict[str, Any]) -> Optional[PIL.Image.Image]:
    # Extract the bounding box and segmentation
    x, y, w, h = mask["bbox"]
    print(f"Cropping with bbox: x={x}, y={y}, w={w}, h={h}")

    # Apply the mask to the image
    masked = image * np.expand_dims(mask["segmentation"], -1)
    crop = masked[y: y + h, x: x + w]
    
    if crop.size == 0:
        print(f"Invalid crop: Crop size is 0, bbox might be out of bounds.")
        return None

    print(f"Crop shape before padding: {crop.shape}")

    # Add padding if necessary
    top, bottom, left, right = (0, 0, 0, 0)
    if h > w:
        left = (h - w) // 2
        right = (h - w) // 2
    else:
        top = (w - h) // 2
        bottom = (w - h) // 2

    # Ensure non-negative padding values
    crop = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    print(f"Crop shape after padding: {crop.shape}")

    # Convert to PIL image
    crop = PIL.Image.fromarray(crop)
    return crop

def get_texts(query: str) -> List[str]:
    return [f"a picture of {query}", "a picture of background"]


def filter_masks(image: np.ndarray, masks: List[Dict[str, Any]], predicted_iou_threshold: float, stability_score_threshold: float, query: str, clip_threshold: float) -> List[Dict[str, Any]]:
    filtered_masks: List[Dict[str, Any]] = []

    if isinstance(image, PIL.Image.Image):
        width, height = image.size
    else:
        height, width = image.shape[:2]

    for idx, mask in enumerate(sorted(masks, key=lambda mask: mask["area"])[-TOP_K_OBJ:]):
        print(f"Mask {idx + 1}: Shape = {mask['segmentation'].shape}, Predicted IoU = {mask['predicted_iou']}, Stability = {mask['stability_score']}")

        if (
            mask["predicted_iou"] < predicted_iou_threshold
            and mask["stability_score"] < stability_score_threshold
            and (height, width) != mask["segmentation"].shape[:2]
        ):
            print(f"Mask {idx + 1} skipped due to size or score thresholds")
            continue

        # Instead of cropping, apply the mask directly to the full image
        full_image_with_mask = image * np.expand_dims(mask["segmentation"], -1)
        print(f"Full Image with Mask {idx + 1} Shape: {full_image_with_mask.shape}, Type: {full_image_with_mask.dtype}")

        # Continue with full image and mask

        # if query and get_score(full_image_with_mask, get_texts(query)) < clip_threshold:
        #     print(f"Mask {idx + 1} skipped due to CLIP score threshold")
        #     continue

        filtered_masks.append(mask)
    print("done Filtering")

    return filtered_masks
def draw_masks(image: np.ndarray, masks: List[np.ndarray], alpha: float = 0.7) -> np.ndarray:
    print("start drawing masks:")
    
    for mask in masks:
        color = [randint(127, 255) for _ in range(3)]
        colored_mask = np.expand_dims(mask["segmentation"], 0).repeat(3, axis=0)
        colored_mask = np.moveaxis(colored_mask, 0, -1)
        masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
        image_overlay = masked.filled()
        image = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)
        contours, _ = cv2.findContours(np.uint8(mask["segmentation"]), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(image, contours, -1, (0, 0, 255), 2)
    return image

def segment(predicted_iou_threshold: float, stability_score_threshold: float, clip_threshold: float, image: np.ndarray, query: str) -> PIL.Image.Image:
    """Segment the image using SAM and filter the masks."""
    mask_generator = load_sam_mask_generator()  # Load the SAM mask generator
    print("Loaded mask generator")

    # Check if image is a PyTorch tensor, if yes, convert it to a NumPy array
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()  # Convert PyTorch tensor to NumPy array
    
    # Debug: Check the shape and type of the image after conversion
    print(f"Image np before segmentation: {image.shape}")

    # Adjust size before segmentation
    image = adjust_image_size(image)  
    print(f"Adjusted image shape: {image.shape}")

    # Ensure the image is a float32 NumPy array for SAM
    image = image.astype(np.float32)  # Ensure the image is float32 NumPy array
    print(f"Image shape : {image.shape}")
    print(f"Image dtype : {image.dtype}")

    try:
        # Generate masks using SAM
        masks = mask_generator.generate(image)  # Generate masks with a NumPy array
        print("Generated")
    except Exception as e:
        print(f"Error during segmentation: {str(e)}")

    for idx, mask in enumerate(masks):
        print(f"Mask {idx + 1}:")
        print(f"  - Shape: {mask['segmentation'].shape}")
        print(f"  - Type: {mask['segmentation'].dtype}")
        print(f"  - Predicted IoU: {mask['predicted_iou']}")
        print(f"  - Stability Score: {mask['stability_score']}")
        break

    # Process and filter the masks
    print("Filtering...")
    masks = filter_masks(image, masks, predicted_iou_threshold, stability_score_threshold, query, clip_threshold)
    print("Filtered")
    
    # Draw masks on the image
    image_with_masks = draw_masks(image, masks)
    print("Drawn!")

    # Convert back to a PIL image for final output
    output_image = PIL.Image.fromarray(image_with_masks.astype(np.uint8))  # Convert to uint8 before making PIL image
    return output_image
