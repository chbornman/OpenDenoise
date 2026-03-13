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


def _make_weight_mask(h: int, w: int, overlap: int, device, dtype) -> torch.Tensor:
    """Create a 2D weight mask that feathers from 0 at edges to 1 in the center.

    Uses a cosine ramp over the overlap region so overlapping tiles blend smoothly.
    """
    mask_y = torch.ones(h, device=device, dtype=dtype)
    mask_x = torch.ones(w, device=device, dtype=dtype)

    if overlap > 0 and h > overlap:
        ramp = 0.5 - 0.5 * torch.cos(
            torch.linspace(0, torch.pi, overlap, device=device, dtype=dtype)
        )
        mask_y[:overlap] = ramp
        mask_y[-overlap:] = ramp.flip(0)

    if overlap > 0 and w > overlap:
        ramp = 0.5 - 0.5 * torch.cos(
            torch.linspace(0, torch.pi, overlap, device=device, dtype=dtype)
        )
        mask_x[:overlap] = ramp
        mask_x[-overlap:] = ramp.flip(0)

    return mask_y.unsqueeze(1) * mask_x.unsqueeze(0)


def denoise_tiled(
    model: spandrel.ImageModelDescriptor,
    tensor: torch.Tensor,
    tile_size: int,
    overlap: int = 64,
) -> torch.Tensor:
    """Process image in overlapping tiles with cosine-feathered blending."""
    _, c, h, w = tensor.shape
    output = torch.zeros_like(tensor)
    weight_sum = torch.zeros(1, 1, h, w, device=tensor.device, dtype=tensor.dtype)

    step = tile_size - overlap

    for y in range(0, h, step):
        for x in range(0, w, step):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            y1 = max(0, y2 - tile_size)
            x1 = max(0, x2 - tile_size)

            tile = tensor[:, :, y1:y2, x1:x2]
            tile, ph, pw = _pad_to(tile)

            with torch.no_grad():
                result = model(tile)

            th, tw = y2 - y1, x2 - x1
            result = result[:, :, :th, :tw]

            # Build weight mask — full weight in center, feathered at edges
            # Don't feather at image borders (those edges have no neighbor to blend with)
            mask = torch.ones(th, tw, device=tensor.device, dtype=tensor.dtype)
            if overlap > 0:
                ramp_len = min(overlap, th // 2, tw // 2)
                if ramp_len > 1:
                    ramp = 0.5 - 0.5 * torch.cos(
                        torch.linspace(
                            0,
                            torch.pi,
                            ramp_len,
                            device=tensor.device,
                            dtype=tensor.dtype,
                        )
                    )
                    if y1 > 0:
                        mask[:ramp_len, :] *= ramp[:, None]
                    if y2 < h:
                        mask[-ramp_len:, :] *= ramp.flip(0)[:, None]
                    if x1 > 0:
                        mask[:, :ramp_len] *= ramp[None, :]
                    if x2 < w:
                        mask[:, -ramp_len:] *= ramp.flip(0)[None, :]

            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

            output[:, :, y1:y2, x1:x2] += result * mask
            weight_sum[:, :, y1:y2, x1:x2] += mask

    # Normalize by accumulated weights
    weight_sum = torch.clamp(weight_sum, min=1e-8)
    return output / weight_sum


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
