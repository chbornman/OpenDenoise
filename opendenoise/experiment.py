"""Experiment system for testing different pipeline configurations.

Usage:
    # Run a grid of experiments from YAML config
    python -m opendenoise.experiment experiments.yaml

    # Or programmatically
    from opendenoise.experiment import ExperimentConfig, run_grid
"""

from dataclasses import dataclass, field, asdict
from itertools import product
from pathlib import Path
from typing import Literal

import numpy as np
import time
import yaml


# ── Pipeline knob types ──────────────────────────────────────────────────────

PreTransform = Literal["none", "gamma", "srgb", "sqrt", "log"]
ChannelStrategy = Literal["pseudo_rgb", "per_channel", "rg1b_rg2b"]
AdaptiveStrength = Literal["off", "linear", "shadow_boost"]


@dataclass
class ExperimentConfig:
    """Every knob in the bayer denoise pipeline."""

    # Pre-transform: map linear sensor data to something the model understands
    pre: PreTransform = "none"

    # Model file (basename, resolved against models/ dir)
    model: str = "scunet_color_real_psnr.pth"

    # Denoise strength (0.0 = passthrough, 1.0 = full model output)
    strength: float = 0.5

    # Channel strategy for feeding data to the 3-channel model
    channels: ChannelStrategy = "pseudo_rgb"

    # Noise-adaptive strength
    adaptive: AdaptiveStrength = "off"

    # Tile size (None = auto)
    tile_size: int | None = None

    # FP16 inference
    fp16: bool = False

    def label(self) -> str:
        """Short human-readable label for this config."""
        parts = []
        if self.pre != "none":
            parts.append(self.pre)
        parts.append(Path(self.model).stem.replace("scunet_color_real_", ""))
        parts.append(f"s{int(self.strength * 100)}")
        if self.channels != "pseudo_rgb":
            parts.append(self.channels)
        if self.adaptive != "off":
            parts.append(self.adaptive)
        return "_".join(parts)


# ── Pre/post transforms ─────────────────────────────────────────────────────


def apply_pre_transform(img: np.ndarray, transform: PreTransform) -> np.ndarray:
    """Transform linear [0,1] data before feeding to denoise model."""
    if transform == "none":
        return img
    elif transform == "gamma":
        return np.power(np.clip(img, 0, 1), 1.0 / 2.2)
    elif transform == "srgb":
        # Full sRGB EOTF inverse: linear toe + gamma
        out = np.where(
            img <= 0.0031308,
            img * 12.92,
            1.055 * np.power(np.clip(img, 0.0031308, 1), 1.0 / 2.4) - 0.055,
        )
        return np.clip(out, 0, 1)
    elif transform == "sqrt":
        return np.sqrt(np.clip(img, 0, 1))
    elif transform == "log":
        # log1p with scale to spread shadows
        return np.log1p(np.clip(img, 0, 1) * 10.0) / np.log1p(10.0)
    else:
        raise ValueError(f"Unknown pre-transform: {transform}")


def apply_post_transform(img: np.ndarray, transform: PreTransform) -> np.ndarray:
    """Inverse of pre-transform to return to linear."""
    if transform == "none":
        return img
    elif transform == "gamma":
        return np.power(np.clip(img, 0, 1), 2.2)
    elif transform == "srgb":
        # Inverse sRGB: gamma toe + power
        out = np.where(
            img <= 0.04045,
            img / 12.92,
            np.power(np.clip((img + 0.055) / 1.055, 0, 1), 2.4),
        )
        return np.clip(out, 0, 1)
    elif transform == "sqrt":
        return np.square(np.clip(img, 0, 1))
    elif transform == "log":
        # inverse of log1p(x*10)/log1p(10)
        return (np.expm1(np.clip(img, 0, 1) * np.log1p(10.0))) / 10.0
    else:
        raise ValueError(f"Unknown post-transform: {transform}")


# ── Noise-adaptive strength ──────────────────────────────────────────────────


def compute_strength_map(
    img: np.ndarray,
    base_strength: float,
    mode: AdaptiveStrength,
) -> np.ndarray | float:
    """Compute per-pixel strength map based on signal level.

    Returns either a scalar (uniform) or an array broadcastable to img shape.
    """
    if mode == "off":
        return base_strength

    # Estimate signal level per pixel (mean across channels if multi-channel)
    if img.ndim == 3:
        signal = np.mean(img, axis=2, keepdims=True)
    else:
        signal = img

    if mode == "linear":
        # More denoising in shadows (low signal), less in highlights
        # strength varies from base_strength*1.5 (shadows) to base_strength*0.5 (highlights)
        weight = 1.5 - signal  # 1.5 at black, 0.5 at white
        return np.clip(base_strength * weight, 0, 1)

    elif mode == "shadow_boost":
        # Strong boost in deep shadows, normal elsewhere
        # Threshold at ~0.2 (mid-shadows in linear space)
        shadow_mask = np.clip(1.0 - signal / 0.2, 0, 1)
        boost = 1.0 + shadow_mask * 0.5  # up to 1.5x in deep shadows
        return np.clip(base_strength * boost, 0, 1)

    else:
        raise ValueError(f"Unknown adaptive mode: {mode}")


# ── Channel strategies ───────────────────────────────────────────────────────


def denoise_pseudo_rgb(
    packed: np.ndarray,
    pattern: list,
    model,
    device,
    config: ExperimentConfig,
) -> np.ndarray:
    """Current approach: average greens → 3ch pseudo-RGB → denoise → apply back."""
    from .engine import auto_tile_size, denoise
    from .mode_bayer import (
        bayer_4ch_to_pseudo_rgb,
        apply_rgb_denoise_to_4ch,
    )

    pseudo_rgb = bayer_4ch_to_pseudo_rgb(packed, pattern).astype(np.float32)

    # Pre-transform
    pseudo_rgb_t = apply_pre_transform(pseudo_rgb, config.pre)

    # Denoise
    tile_size = config.tile_size
    if tile_size is None:
        tile_size = auto_tile_size(packed.shape[0], packed.shape[1], channels=3)

    denoised_t = denoise(
        model, pseudo_rgb_t, device, config.strength, tile_size, config.fp16
    )

    # Post-transform back to linear
    pseudo_rgb_linear = apply_post_transform(pseudo_rgb_t, config.pre)
    denoised_linear = apply_post_transform(denoised_t, config.pre)

    # Adaptive strength: blend in linear space
    strength_map = compute_strength_map(pseudo_rgb, config.strength, config.adaptive)
    if config.adaptive != "off":
        # Re-blend with adaptive strength (overrides the uniform blend done in engine.denoise)
        # We need the raw model output, so we undo the engine's strength blend
        # denoised_t = original_t * (1-s) + model_out * s → model_out = (denoised_t - original_t*(1-s)) / s
        # Simpler: just re-blend in linear space
        if config.strength > 0:
            model_output = (
                denoised_linear - pseudo_rgb * (1 - config.strength)
            ) / config.strength
            model_output = np.clip(model_output, 0, 1)
            denoised_linear = (
                pseudo_rgb * (1 - strength_map) + model_output * strength_map
            )

    # Apply corrections back to 4 channels
    return apply_rgb_denoise_to_4ch(packed, denoised_linear, pattern)


def denoise_per_channel(
    packed: np.ndarray,
    pattern: list,
    model,
    device,
    config: ExperimentConfig,
) -> np.ndarray:
    """Denoise each of the 4 Bayer channels independently as grayscale.

    Each channel is duplicated to 3 channels (model expects 3ch), denoised,
    then the result is averaged back to 1 channel.
    """
    from .engine import auto_tile_size, denoise

    tile_size = config.tile_size
    if tile_size is None:
        tile_size = auto_tile_size(packed.shape[0], packed.shape[1], channels=3)

    result = packed.copy()
    for ch in range(4):
        single = packed[:, :, ch].astype(np.float32)

        # Pre-transform
        single_t = apply_pre_transform(single, config.pre)

        # Duplicate to 3 channels for the model
        triple = np.stack([single_t, single_t, single_t], axis=2)

        # Denoise
        denoised_triple = denoise(
            model, triple, device, config.strength, tile_size, config.fp16
        )

        # Average the 3 output channels back to 1
        denoised_t = np.mean(denoised_triple, axis=2)

        # Post-transform
        denoised_linear = apply_post_transform(denoised_t, config.pre)

        result[:, :, ch] = denoised_linear

    return result


def denoise_rg1b_rg2b(
    packed: np.ndarray,
    pattern: list,
    model,
    device,
    config: ExperimentConfig,
) -> np.ndarray:
    """Two passes: [R,G1,B] and [R,G2,B], average the overlapping R and B results.

    This gives the model real color context (unlike per_channel) while
    preserving the G1/G2 distinction (unlike pseudo_rgb which averages them).
    """
    from .engine import auto_tile_size, denoise
    from .mode_bayer import bayer_4ch_to_pseudo_rgb

    flat = [pattern[r][c] for r in range(2) for c in range(2)]
    color_to_rgb = {0: 0, 1: 1, 2: 2, 3: 1}

    # Find which packed channels are R, G1, G2, B
    r_idx = [i for i, c in enumerate(flat) if c == 0][0]
    g_indices = [i for i, c in enumerate(flat) if c in (1, 3)]
    b_idx = [i for i, c in enumerate(flat) if c == 2][0]
    g1_idx, g2_idx = g_indices[0], g_indices[1]

    tile_size = config.tile_size
    if tile_size is None:
        tile_size = auto_tile_size(packed.shape[0], packed.shape[1], channels=3)

    result = packed.copy()

    # Pass 1: [R, G1, B]
    rgb1 = np.stack(
        [packed[:, :, r_idx], packed[:, :, g1_idx], packed[:, :, b_idx]], axis=2
    ).astype(np.float32)
    rgb1_t = apply_pre_transform(rgb1, config.pre)
    den1_t = denoise(model, rgb1_t, device, config.strength, tile_size, config.fp16)
    den1 = apply_post_transform(den1_t, config.pre)

    # Pass 2: [R, G2, B]
    rgb2 = np.stack(
        [packed[:, :, r_idx], packed[:, :, g2_idx], packed[:, :, b_idx]], axis=2
    ).astype(np.float32)
    rgb2_t = apply_pre_transform(rgb2, config.pre)
    den2_t = denoise(model, rgb2_t, device, config.strength, tile_size, config.fp16)
    den2 = apply_post_transform(den2_t, config.pre)

    # Merge: average R and B from both passes, keep G1/G2 from respective passes
    result[:, :, r_idx] = (den1[:, :, 0] + den2[:, :, 0]) / 2
    result[:, :, g1_idx] = den1[:, :, 1]
    result[:, :, g2_idx] = den2[:, :, 1]
    result[:, :, b_idx] = (den1[:, :, 2] + den2[:, :, 2]) / 2

    return result


# ── Main experiment runner ───────────────────────────────────────────────────


def run_experiment(
    raw_path: Path,
    output_dir: Path,
    config: ExperimentConfig,
    model=None,
    device=None,
) -> dict:
    """Run a single experiment. Returns timing/size info."""
    from .engine import auto_tile_size, get_device, load_model
    from .mode_bayer import (
        extract_bayer,
        pack_bayer_to_4ch,
        unpack_4ch_to_bayer,
        save_bayer_dng,
    )

    start = time.time()

    # Load model if not provided
    model_dir = Path(__file__).parent.parent / "models"
    if model is None:
        device = device or get_device()
        model = load_model(model_dir / config.model, device, config.fp16)
    if device is None:
        device = get_device()

    # Extract
    bayer, thumbnail, meta = extract_bayer(raw_path)

    # Normalize
    white = max(meta["white_level"])
    black = np.mean(meta["black_level_per_channel"])
    bayer_norm = (bayer - black) / (white - black)
    bayer_norm = np.clip(bayer_norm, 0, 1)

    # Pack
    packed = pack_bayer_to_4ch(bayer_norm, meta).astype(np.float32)

    # Denoise with selected channel strategy
    if config.channels == "pseudo_rgb":
        packed_denoised = denoise_pseudo_rgb(
            packed, meta["pattern"], model, device, config
        )
    elif config.channels == "per_channel":
        packed_denoised = denoise_per_channel(
            packed, meta["pattern"], model, device, config
        )
    elif config.channels == "rg1b_rg2b":
        packed_denoised = denoise_rg1b_rg2b(
            packed, meta["pattern"], model, device, config
        )
    else:
        raise ValueError(f"Unknown channel strategy: {config.channels}")

    # Unpack and rescale
    bayer_denoised = unpack_4ch_to_bayer(packed_denoised, bayer.shape)
    bayer_denoised = bayer_denoised * (white - black) + black

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = (output_dir / raw_path.stem).with_suffix(".dng")
    save_bayer_dng(bayer_denoised, out_path, meta, thumbnail)

    elapsed = time.time() - start
    size_mb = out_path.stat().st_size / (1024**2)

    return {
        "config": config.label(),
        "file": out_path.name,
        "time_s": round(elapsed, 1),
        "size_mb": round(size_mb, 0),
        "path": str(out_path),
    }


def expand_grid(grid: dict) -> list[ExperimentConfig]:
    """Expand a grid dict into a list of ExperimentConfigs.

    grid = {"pre": ["none", "gamma"], "strength": [0.5, 0.75]}
    → 4 configs with all combinations
    """
    # Get all keys and their value lists
    keys = list(grid.keys())
    value_lists = [grid[k] if isinstance(grid[k], list) else [grid[k]] for k in keys]

    configs = []
    for values in product(*value_lists):
        kwargs = dict(zip(keys, values))
        configs.append(ExperimentConfig(**kwargs))

    return configs


def run_grid(
    raw_path: Path,
    output_base: Path,
    grid: dict,
    model_cache: dict | None = None,
) -> list[dict]:
    """Run a full grid of experiments.

    Each config gets its own subfolder under output_base.
    Models are cached across runs to avoid reloading.
    """
    from .engine import get_device, load_model

    configs = expand_grid(grid)
    model_dir = Path(__file__).parent.parent / "models"
    device = get_device()

    # Cache models by filename
    if model_cache is None:
        model_cache = {}

    results = []
    total = len(configs)

    print(f"Running {total} experiments on {raw_path.name}")
    print(f"Output: {output_base}/")
    print()

    for i, config in enumerate(configs, 1):
        label = config.label()
        print(f"  [{i}/{total}] {label} ...", end=" ", flush=True)

        # Load/cache model
        if config.model not in model_cache:
            model_cache[config.model] = load_model(
                model_dir / config.model, device, config.fp16
            )
        model = model_cache[config.model]

        out_dir = output_base / label
        result = run_experiment(raw_path, out_dir, config, model, device)
        results.append(result)

        print(f"{result['time_s']}s, {result['size_mb']:.0f}MB")

    print()
    print(f"Done. {total} experiments in {sum(r['time_s'] for r in results):.0f}s")

    # Write summary
    summary_path = output_base / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Input: {raw_path}\n")
        f.write(f"Date: {time.strftime('%Y-%m-%d %H:%M')}\n\n")
        f.write(f"{'Config':<40} {'Time':>6} {'Size':>6}\n")
        f.write("-" * 54 + "\n")
        for r in results:
            f.write(f"{r['config']:<40} {r['time_s']:>5.1f}s {r['size_mb']:>5.0f}MB\n")

    print(f"Summary: {summary_path}")
    return results


# ── CLI entry point ──────────────────────────────────────────────────────────


def main():
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Run denoise experiments with different pipeline configs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run from YAML config
  python -m opendenoise.experiment experiments.yaml

  # Quick comparison of pre-transforms
  python -m opendenoise.experiment photo.ARW --pre none gamma sqrt srgb

  # Strength sweep
  python -m opendenoise.experiment photo.ARW --strength 0.25 0.5 0.75 1.0

  # Full grid
  python -m opendenoise.experiment photo.ARW \\
      --pre none gamma sqrt \\
      --strength 0.5 0.75 \\
      --model scunet_color_real_psnr.pth scunet_color_real_gan.pth
        """,
    )
    parser.add_argument("input", type=Path, help="Input RAW file or YAML config")
    parser.add_argument(
        "-o", "--output", type=Path, default=None, help="Output base dir"
    )
    parser.add_argument("--pre", nargs="+", default=None, help="Pre-transforms to test")
    parser.add_argument("--strength", nargs="+", type=float, default=None)
    parser.add_argument("--model", nargs="+", default=None, help="Model files to test")
    parser.add_argument("--channels", nargs="+", default=None)
    parser.add_argument("--adaptive", nargs="+", default=None)

    args = parser.parse_args()

    # Check if input is a YAML config
    if args.input.suffix in (".yaml", ".yml"):
        with open(args.input) as f:
            cfg = yaml.safe_load(f)
        raw_path = Path(cfg["input"])
        output_base = Path(cfg.get("output", "experiments/"))
        grid = cfg.get("grid", {})
    else:
        # Build grid from CLI args
        raw_path = args.input
        output_base = args.output or Path("experiments/")
        grid = {}
        if args.pre:
            grid["pre"] = args.pre
        if args.strength:
            grid["strength"] = args.strength
        if args.model:
            grid["model"] = args.model
        if args.channels:
            grid["channels"] = args.channels
        if args.adaptive:
            grid["adaptive"] = args.adaptive

    if not grid:
        # Default experiment: compare pre-transforms at strength 0.5
        grid = {
            "pre": ["none", "gamma", "sqrt", "srgb"],
            "strength": [0.5],
        }

    if not raw_path.exists():
        print(f"Error: {raw_path} not found", file=sys.stderr)
        sys.exit(1)

    run_grid(raw_path, output_base, grid)


if __name__ == "__main__":
    main()
