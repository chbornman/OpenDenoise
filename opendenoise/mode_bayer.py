"""Mode: bayer - Denoise in Bayer space, output real DNG.

Pipeline: ARW -> extract Bayer -> pack RGGB as 4-channel half-res ->
          denoise -> unpack -> write DNG with original metadata.

This preserves the original file size (~70 MB) and darktable treats the
output as a RAW file with full non-destructive editing. Experimental.

The key insight: a 2x2 Bayer pattern (RGGB) can be reshaped into a
half-resolution 4-channel image. The denoise model sees this as a
4-channel "image" and removes noise from each channel. We then unpack
back to Bayer and write a valid DNG.

For the 3-channel model, we run denoise on a pseudo-RGB image constructed
from (R, avg(Gr,Gb), B) and apply the corrections back to all 4 channels.
"""

from pathlib import Path

import numpy as np
import rawpy
import tifffile


def _make_thumbnail(raw, max_size: int = 256) -> np.ndarray:
    """Generate a small RGB thumbnail from the RAW file."""
    thumb = raw.postprocess(
        use_camera_wb=True,
        no_auto_bright=False,
        output_bps=8,
        half_size=True,
    )
    # Resize to max_size
    h, w = thumb.shape[:2]
    scale = max_size / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    # Simple nearest-neighbor resize
    row_indices = (np.arange(new_h) / scale).astype(int)
    col_indices = (np.arange(new_w) / scale).astype(int)
    thumb = thumb[row_indices][:, col_indices]
    return thumb


def extract_bayer(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Extract raw Bayer data, thumbnail, and metadata from a RAW file."""
    raw = rawpy.imread(str(path))

    # Extract Bayer data and pattern BEFORE postprocess (which mutates state)
    bayer = raw.raw_image_visible.copy().astype(np.float32)
    pattern = raw.raw_pattern.copy()

    meta = {
        "color_desc": raw.color_desc.decode()
        if isinstance(raw.color_desc, bytes)
        else str(raw.color_desc),
        "pattern": pattern.tolist(),
        "white_level": [int(w) for w in raw.camera_white_level_per_channel],
        "black_level_per_channel": [int(b) for b in raw.black_level_per_channel],
        "rgb_xyz_matrix": raw.rgb_xyz_matrix.tolist(),
        "daylight_whitebalance": list(raw.daylight_whitebalance),
        "camera_whitebalance": list(raw.camera_whitebalance),
        "raw_shape": bayer.shape,
        "sizes": {
            "raw_height": raw.sizes.raw_height,
            "raw_width": raw.sizes.raw_width,
            "height": raw.sizes.height,
            "width": raw.sizes.width,
            "top_margin": raw.sizes.top_margin,
            "left_margin": raw.sizes.left_margin,
            "flip": raw.sizes.flip,
        },
        "flip": raw.sizes.flip,
    }

    # Generate thumbnail AFTER extracting metadata (postprocess mutates raw state)
    # Need to re-open since postprocess consumes the raw data
    raw.close()
    raw2 = rawpy.imread(str(path))
    thumbnail = _make_thumbnail(raw2)
    raw2.close()

    return bayer, thumbnail, meta


def pack_bayer_to_4ch(bayer: np.ndarray, meta: dict) -> np.ndarray:
    """Pack RGGB Bayer array into half-resolution 4-channel image.

    Input:  (H, W) float32 Bayer
    Output: (H/2, W/2, 4) float32 [R, Gr, Gb, B] ordered by CFA pattern
    """
    h, w = bayer.shape
    # Ensure even dimensions
    h = h - (h % 2)
    w = w - (w % 2)
    bayer = bayer[:h, :w]

    # Extract 4 sub-images from 2x2 blocks
    ch0 = bayer[0::2, 0::2]  # top-left
    ch1 = bayer[0::2, 1::2]  # top-right
    ch2 = bayer[1::2, 0::2]  # bottom-left
    ch3 = bayer[1::2, 1::2]  # bottom-right

    return np.stack([ch0, ch1, ch2, ch3], axis=2)


def unpack_4ch_to_bayer(packed: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Unpack 4-channel half-res back to full Bayer pattern."""
    h, w = original_shape
    h = h - (h % 2)
    w = w - (w % 2)

    bayer = np.zeros((h, w), dtype=packed.dtype)
    bayer[0::2, 0::2] = packed[:, :, 0]
    bayer[0::2, 1::2] = packed[:, :, 1]
    bayer[1::2, 0::2] = packed[:, :, 2]
    bayer[1::2, 1::2] = packed[:, :, 3]

    return bayer


def bayer_4ch_to_pseudo_rgb(packed: np.ndarray, pattern: list) -> np.ndarray:
    """Convert 4-channel Bayer pack to 3-channel pseudo-RGB for the denoise model.

    Maps CFA pattern positions to R, G (averaged), B channels.
    Returns (H/2, W/2, 3) float32.
    """
    # Determine which channel index maps to R, G1, G2, B
    # rawpy pattern: 0=R, 1=G, 2=B, 3=G2 (second green)
    flat = [pattern[r][c] for r in range(2) for c in range(2)]

    r_indices = [i for i, c in enumerate(flat) if c == 0]
    g_indices = [i for i, c in enumerate(flat) if c in (1, 3)]  # both greens
    b_indices = [i for i, c in enumerate(flat) if c == 2]

    r = np.mean([packed[:, :, i] for i in r_indices], axis=0)
    g = np.mean([packed[:, :, i] for i in g_indices], axis=0)
    b = np.mean([packed[:, :, i] for i in b_indices], axis=0)

    return np.stack([r, g, b], axis=2)


def apply_rgb_denoise_to_4ch(
    packed_original: np.ndarray,
    packed_denoised_rgb: np.ndarray,
    pattern: list,
) -> np.ndarray:
    """Apply the RGB denoise result back to the 4-channel Bayer pack.

    We compute the per-pixel difference (noise) from the pseudo-RGB denoise
    and subtract it from each Bayer channel based on its color.
    """
    flat = [pattern[r][c] for r in range(2) for c in range(2)]

    # Map rawpy color index to RGB channel: 0->R(0), 1->G(1), 2->B(2), 3->G(1)
    color_to_rgb = {0: 0, 1: 1, 2: 2, 3: 1}

    original_rgb = bayer_4ch_to_pseudo_rgb(packed_original, pattern)
    result = packed_original.copy()

    for ch_idx in range(4):
        rgb_ch = color_to_rgb[flat[ch_idx]]
        noise = original_rgb[:, :, rgb_ch] - packed_denoised_rgb[:, :, rgb_ch]
        result[:, :, ch_idx] = packed_original[:, :, ch_idx] - noise

    return result


def _neutral_from_wb(camera_wb: list) -> list[int]:
    """Convert camera white balance to AsShotNeutral rationals.

    AsShotNeutral is the reciprocal of the WB multipliers (the color of a
    neutral object in camera-native RGB), stored as RATIONAL pairs.
    """
    # camera_wb is [R, G, B, G2] multipliers
    r, g, b = camera_wb[0], camera_wb[1], camera_wb[2]
    # AsShotNeutral = 1/multiplier for each channel
    inv_r, inv_g, inv_b = 1.0 / r, 1.0 / g, 1.0 / b
    # Normalize so the largest is 1.0
    mx = max(inv_r, inv_g, inv_b)
    inv_r, inv_g, inv_b = inv_r / mx, inv_g / mx, inv_b / mx
    scale = 10000
    return [
        int(inv_r * scale),
        scale,
        int(inv_g * scale),
        scale,
        int(inv_b * scale),
        scale,
    ]


def save_bayer_dng(
    bayer: np.ndarray,
    path: Path,
    meta: dict,
    thumbnail: np.ndarray | None = None,
) -> None:
    """Write denoised Bayer data as a DNG file.

    Uses tifffile to write a TIFF with DNG-compatible tags that darktable
    can open as a RAW file.
    """
    h, w = bayer.shape
    black = meta["black_level_per_channel"]
    white = max(meta["white_level"])

    # Convert float32 back to uint16 range
    bayer_u16 = np.clip(bayer, 0, white).astype(np.uint16)

    path.parent.mkdir(parents=True, exist_ok=True)

    # CFA pattern for DNG (TIFF tag 33422)
    # rawpy uses 0=R, 1=G, 2=B, 3=G2 -> DNG uses 0=R, 1=G, 2=B
    pattern = meta["pattern"]
    rawpy_to_dng = {0: 0, 1: 1, 2: 2, 3: 1}
    cfa_pattern = [rawpy_to_dng[pattern[r][c]] for r in range(2) for c in range(2)]

    # Color matrix from rgb_xyz_matrix (3x4 -> use first 3 columns as 3x3)
    # DNG ColorMatrix1 maps XYZ to camera RGB, stored as signed rationals
    cm = meta["rgb_xyz_matrix"]
    color_matrix = []
    for row in cm[:3]:
        for val in row[:3]:
            color_matrix.extend([int(val * 10000), 10000])

    # DNG requires SubIFD structure:
    # Main IFD: thumbnail + DNG version/camera tags
    # SubIFD: actual raw CFA data

    extratags_main = [
        (271, "s", 0, "OpenDenoise", True),  # Make
        (272, "s", 0, f"AI Denoised {meta.get('color_desc', '')}", True),  # Model
        (50706, "B", 4, [1, 4, 0, 0], True),  # DNGVersion
        (50707, "B", 4, [1, 1, 0, 0], True),  # DNGBackwardVersion
        (50708, "s", 0, f"OpenDenoise {meta.get('color_desc', '')}", True),
        (
            50728,
            5,  # RATIONAL type
            3,
            _neutral_from_wb(meta.get("camera_whitebalance", [1, 1, 1, 1])),
            True,
        ),
    ]

    # Add orientation tag if available (rawpy sizes.flip -> EXIF Orientation)
    flip = meta.get("flip", 0)
    flip_to_orientation = {0: 1, 3: 3, 5: 8, 6: 6}
    orientation = flip_to_orientation.get(flip, 1)
    if orientation != 1:
        extratags_main.append((274, "H", 1, orientation, True))  # Orientation

    extratags_raw = [
        # CFA
        (33421, "H", 2, [2, 2], True),  # CFARepeatPatternDim
        (33422, "B", 4, cfa_pattern, True),  # CFAPattern
        (50711, "H", 1, 1, True),  # CFALayout = 1 (rectangular)
        # Levels
        (50713, "H", 2, [2, 2], True),  # BlackLevelRepeatDim
        (50714, "H", 4, black, True),  # BlackLevel
        (50717, "I", 1, white, True),  # WhiteLevel
        # Color matrix (signed rationals: num, denom pairs)
        (50721, 10, 9, color_matrix, True),  # ColorMatrix1 (SRATIONAL)
        (50778, "H", 1, 21, True),  # CalibrationIlluminant1
    ]

    # Write DNG with SubIFD structure (rawspeed requires this)
    with tifffile.TiffWriter(str(path)) as tw:
        # Main IFD - RGB thumbnail
        if thumbnail is None:
            thumb = np.zeros((128, 128, 3), dtype=np.uint8)
        else:
            thumb = thumbnail
        tw.write(
            thumb,
            photometric="rgb",
            subifds=1,
            subfiletype=1,  # reduced-resolution thumbnail
            extratags=extratags_main,
        )
        # SubIFD - raw CFA data (tiled, as rawspeed requires)
        tw.write(
            bayer_u16,
            photometric=32803,  # CFA
            compression=None,
            tile=(256, 256),
            subfiletype=0,
            extratags=extratags_raw,
        )


def process_bayer(
    raw_path: Path,
    output_path: Path,
    model,
    device,
    strength: float = 0.5,
    tile_size: int | None = None,
    fp16: bool = False,
    gamma: bool = False,
) -> None:
    """Full pipeline: RAW -> Bayer denoise -> DNG.

    If gamma=True, apply sRGB-like gamma (x^1/2.2) before denoising and
    invert (x^2.2) after. This makes the linear sensor data look more like
    the sRGB images the model was trained on, potentially improving results.
    """
    from .engine import auto_tile_size, denoise

    bayer, thumbnail, meta = extract_bayer(raw_path)

    # Normalize Bayer to 0-1 range
    white = max(meta["white_level"])
    black = np.mean(meta["black_level_per_channel"])
    bayer_norm = (bayer - black) / (white - black)
    bayer_norm = np.clip(bayer_norm, 0, 1)

    # Pack to 4-channel half-res
    packed = pack_bayer_to_4ch(bayer_norm, meta)
    ph, pw = packed.shape[:2]

    if tile_size is None:
        tile_size = auto_tile_size(ph, pw, channels=3)

    # Convert to pseudo-RGB for the 3-channel denoise model
    packed = packed.astype(np.float32)
    pseudo_rgb = bayer_4ch_to_pseudo_rgb(packed, meta["pattern"]).astype(np.float32)

    # Apply gamma curve to make linear data look like sRGB for the model
    if gamma:
        pseudo_rgb = np.power(pseudo_rgb, 1.0 / 2.2)

    # Denoise the pseudo-RGB
    denoised_rgb = denoise(model, pseudo_rgb, device, strength, tile_size, fp16)

    # Invert gamma back to linear
    if gamma:
        pseudo_rgb = np.power(pseudo_rgb, 2.2)
        denoised_rgb = np.power(np.clip(denoised_rgb, 0, 1), 2.2)

    # Apply denoise corrections back to all 4 Bayer channels
    packed_denoised = apply_rgb_denoise_to_4ch(packed, denoised_rgb, meta["pattern"])

    # Unpack back to Bayer and rescale
    bayer_denoised = unpack_4ch_to_bayer(packed_denoised, bayer.shape)
    bayer_denoised = bayer_denoised * (white - black) + black

    # Save as DNG
    output_path = output_path.with_suffix(".dng")
    save_bayer_dng(bayer_denoised, output_path, meta, thumbnail)
