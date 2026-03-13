"""Mode: post - Denoise already-exported TIFFs/PNGs (post-edit).

Pipeline: TIFF/PNG in -> denoise -> TIFF/PNG out
Same as the original ai-denoise script but integrated into OpenDenoise.
"""

from pathlib import Path

import cv2
import numpy as np

from .engine import auto_tile_size, denoise

SUPPORTED_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".exr"}


def load_image(path: Path) -> tuple[np.ndarray, int]:
    """Load image as float32 RGB. Returns (image, original_bit_depth)."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Failed to load: {path}")

    # Detect bit depth
    if img.dtype == np.uint16:
        bit_depth = 16
    elif img.dtype == np.float32:
        bit_depth = 32
    else:
        bit_depth = 8

    # BGR(A) -> RGB(A)
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize to 0-1
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0
    elif img.dtype == np.uint16:
        img = img.astype(np.float32) / 65535.0
    elif img.dtype != np.float32:
        img = img.astype(np.float32)

    return img, bit_depth


def save_image(img: np.ndarray, path: Path, bit_depth: int = 16) -> None:
    """Save image, converting to appropriate bit depth."""
    if img.ndim == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGRA)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    img = np.clip(img, 0, 1)

    ext = path.suffix.lower()
    if ext in (".tif", ".tiff", ".png") and bit_depth == 16:
        img = (img * 65535).astype(np.uint16)
    elif ext == ".exr":
        pass  # keep float32
    else:
        img = (img * 255).astype(np.uint8)

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


def process_post(
    input_path: Path,
    output_path: Path,
    model,
    device,
    strength: float = 0.5,
    tile_size: int | None = None,
    fp16: bool = False,
) -> None:
    """Denoise an already-processed image."""
    img, bit_depth = load_image(input_path)
    h, w = img.shape[:2]

    if tile_size is None:
        tile_size = auto_tile_size(h, w)

    denoised = denoise(model, img, device, strength, tile_size, fp16)
    save_image(denoised, output_path, bit_depth)
