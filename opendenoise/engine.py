"""Denoise engine - model loading, tiled GPU inference, strength blending."""

from pathlib import Path

import numpy as np
import spandrel
import torch


def load_model(
    model_path: Path,
    device: torch.device,
    fp16: bool = False,
) -> spandrel.ImageModelDescriptor:
    """Load a spandrel-compatible denoise model."""
    model = spandrel.ModelLoader().load_from_file(str(model_path))
    assert isinstance(model, spandrel.ImageModelDescriptor)
    model.to(device).eval()
    if fp16 and device.type != "cpu":
        model.model.half()
    return model


def _to_tensor(
    img: np.ndarray,
    device: torch.device,
    fp16: bool,
) -> torch.Tensor:
    """HWC float32 numpy -> BCHW tensor on device."""
    if img.ndim == 2:
        t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    else:
        t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
    t = t.to(device)
    if fp16:
        t = t.half()
    return t


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """BCHW tensor -> HWC float32 numpy."""
    t = t.squeeze(0).float().cpu()
    if t.ndim == 2:
        return t.numpy()
    return t.permute(1, 2, 0).numpy()


def _pad_to(tensor: torch.Tensor, multiple: int = 64) -> tuple[torch.Tensor, int, int]:
    """Pad tensor to multiple of `multiple`, return padded tensor and pad amounts."""
    _, _, h, w = tensor.shape
    pad_h = (multiple - h % multiple) % multiple
    pad_w = (multiple - w % multiple) % multiple
    if pad_h or pad_w:
        tensor = torch.nn.functional.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor, pad_h, pad_w


def denoise_tiled(
    model: spandrel.ImageModelDescriptor,
    tensor: torch.Tensor,
    tile_size: int,
    overlap: int = 32,
) -> torch.Tensor:
    """Process image in overlapping tiles to avoid OOM."""
    _, c, h, w = tensor.shape
    output = torch.zeros_like(tensor)
    count = torch.zeros(1, 1, h, w, device=tensor.device, dtype=tensor.dtype)

    step = tile_size - overlap * 2

    for y in range(0, h, step):
        for x in range(0, w, step):
            y1 = max(0, y - overlap)
            x1 = max(0, x - overlap)
            y2 = min(h, y1 + tile_size)
            x2 = min(w, x1 + tile_size)

            tile = tensor[:, :, y1:y2, x1:x2]
            tile, ph, pw = _pad_to(tile)

            with torch.no_grad():
                result = model(tile)

            result = result[:, :, : y2 - y1, : x2 - x1]

            # Crop overlap from edges (except at image borders)
            ry1 = overlap if y1 > 0 else 0
            rx1 = overlap if x1 > 0 else 0
            ry2 = (y2 - y1) - (overlap if y2 < h else 0)
            rx2 = (x2 - x1) - (overlap if x2 < w else 0)

            oy1, oy2 = y1 + ry1, y1 + ry2
            ox1, ox2 = x1 + rx1, x1 + rx2

            output[:, :, oy1:oy2, ox1:ox2] = result[:, :, ry1:ry2, rx1:rx2]
            count[:, :, oy1:oy2, ox1:ox2] = 1.0

    return output


def denoise(
    model: spandrel.ImageModelDescriptor,
    img: np.ndarray,
    device: torch.device,
    strength: float = 0.5,
    tile_size: int | None = None,
    fp16: bool = False,
) -> np.ndarray:
    """Denoise an image (HWC float32, 0-1 range).

    Handles alpha channels, tiling, padding, and strength blending.
    Returns denoised image in same format.
    """
    # Separate alpha if present
    alpha = None
    if img.ndim == 3 and img.shape[2] == 4:
        alpha = img[:, :, 3:4]
        img = img[:, :, :3]

    tensor = _to_tensor(img, device, fp16)

    if tile_size:
        result = denoise_tiled(model, tensor, tile_size)
    else:
        tensor, pad_h, pad_w = _pad_to(tensor)
        with torch.no_grad():
            result = model(tensor)
        h, w = img.shape[:2]
        result = result[:, :, :h, :w]

    out = _to_numpy(result)

    # Strength blend
    if strength < 1.0:
        out = img * (1.0 - strength) + out * strength

    out = np.clip(out, 0.0, 1.0)

    if alpha is not None:
        out = np.concatenate([out, alpha], axis=2)

    return out


def auto_tile_size(h: int, w: int, channels: int = 3) -> int | None:
    """Determine tile size based on image dimensions. Returns None if no tiling needed."""
    if max(h, w) > 4096:
        return 1024
    return None


def get_device(force_cpu: bool = False) -> torch.device:
    """Get the best available device."""
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def print_device_info(device: torch.device) -> None:
    """Print GPU info if available."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"GPU: {name} ({vram:.0f} GB)")
    else:
        print("Device: CPU (slow)")
