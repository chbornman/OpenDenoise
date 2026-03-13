"""Mode: raw - Demosaic RAW, denoise in linear RGB, output compressed TIFF.

Pipeline: ARW/CR3/NEF -> rawpy demosaic (linear) -> AI denoise -> 16-bit compressed TIFF
This is the Lightroom-style approach. Output is ~260 MB per image but proven quality.
"""

from pathlib import Path

import numpy as np
import rawpy
import tifffile

from .engine import auto_tile_size, denoise


def decode_raw(path: Path) -> tuple[np.ndarray, dict]:
    """Decode RAW file to linear float32 RGB. Returns (image, metadata)."""
    raw = rawpy.imread(str(path))

    # Extract metadata before processing
    meta = {
        "color_desc": raw.color_desc.decode()
        if isinstance(raw.color_desc, bytes)
        else str(raw.color_desc),
        "raw_pattern": raw.raw_pattern.tolist()
        if raw.raw_pattern is not None
        else None,
        "white_level": [int(w) for w in raw.camera_white_level_per_channel],
        "black_level": [int(b) for b in raw.black_level_per_channel],
        "rgb_xyz_matrix": raw.rgb_xyz_matrix.tolist(),
        "daylight_whitebalance": list(raw.daylight_whitebalance),
        "camera_whitebalance": list(raw.camera_whitebalance),
    }

    # Demosaic to linear RGB (no gamma, no auto-bright)
    rgb = raw.postprocess(
        use_camera_wb=True,
        no_auto_bright=True,
        output_bps=16,
        gamma=(1, 1),  # Linear
        output_color=rawpy.ColorSpace.sRGB,
    )

    raw.close()

    # Convert to float32 0-1
    img = rgb.astype(np.float32) / 65535.0
    return img, meta


def save_linear_tiff(
    img: np.ndarray,
    path: Path,
    compression: str = "zstd",
    meta: dict | None = None,
) -> None:
    """Save linear float32 RGB as 16-bit compressed TIFF."""
    # Convert back to uint16
    img_u16 = np.clip(img * 65535, 0, 65535).astype(np.uint16)

    path.parent.mkdir(parents=True, exist_ok=True)

    # Build extra tags for linear interpretation
    extratags = [
        # Mark as linear (gamma 1.0) so darktable doesn't apply tone curve
        # SampleFormat = uint
    ]

    if meta:
        desc = f"OpenDenoise | {meta.get('color_desc', '')}"
        extratags.append(("ImageDescription", "s", 0, desc, True))

    with tifffile.TiffWriter(str(path)) as tw:
        tw.write(
            img_u16,
            photometric="rgb",
            compression=compression,
            predictor=True,
            extratags=extratags if extratags else None,
        )


def process_raw(
    raw_path: Path,
    output_path: Path,
    model,
    device,
    strength: float = 0.5,
    tile_size: int | None = None,
    fp16: bool = False,
    compression: str = "zstd",
) -> None:
    """Full pipeline: RAW -> demosaic -> denoise -> compressed linear TIFF."""
    img, meta = decode_raw(raw_path)
    h, w = img.shape[:2]

    if tile_size is None:
        tile_size = auto_tile_size(h, w)

    denoised = denoise(model, img, device, strength, tile_size, fp16)
    save_linear_tiff(denoised, output_path, compression, meta)
