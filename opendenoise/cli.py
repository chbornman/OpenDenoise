#!/usr/bin/env python3
"""OpenDenoise CLI - AI denoising for RAW photography.

Usage:
    opendenoise INPUT [OUTPUT] [--mode MODE] [--strength S] [--model PATH]

Modes:
    raw   - Demosaic RAW, denoise linear RGB, output compressed TIFF (~260 MB)
    bayer - Denoise in Bayer space, output DNG (~70 MB, experimental)
    post  - Denoise already-exported TIFFs/PNGs

Examples:
    opendenoise ~/Photos/*.ARW
    opendenoise ~/Photos/*.ARW --mode bayer
    opendenoise ~/Photos/*.ARW --mode raw --strength 0.7
    opendenoise ~/exports/*.tif --mode post
"""

import argparse
import sys
import time
from pathlib import Path

RAW_EXTENSIONS = {
    ".arw",
    ".cr2",
    ".cr3",
    ".nef",
    ".orf",
    ".raf",
    ".dng",
    ".rw2",
    ".pef",
}
POST_EXTENSIONS = {".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".exr"}
ALL_EXTENSIONS = RAW_EXTENSIONS | POST_EXTENSIONS

DEFAULT_MODEL = Path(__file__).parent.parent / "models" / "scunet_color_real_psnr.pth"


def detect_mode(files: list[Path]) -> str:
    """Auto-detect mode based on file extensions."""
    extensions = {f.suffix.lower() for f in files}
    if extensions & RAW_EXTENSIONS:
        return "raw"
    return "post"


def collect_files(inputs: list[Path]) -> list[Path]:
    """Collect files from paths (supports files and directories)."""
    files = []
    for p in inputs:
        if p.is_dir():
            files.extend(
                sorted(f for f in p.iterdir() if f.suffix.lower() in ALL_EXTENSIONS)
            )
        elif p.is_file() and p.suffix.lower() in ALL_EXTENSIONS:
            files.append(p)
        else:
            print(f"Warning: skipping {p}", file=sys.stderr)
    return files


def main():
    parser = argparse.ArgumentParser(
        description="OpenDenoise - AI denoising for RAW photography",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  raw    Decode RAW, denoise in linear RGB, output compressed TIFF.
         Best quality. ~260 MB output. (Lightroom-style)
  bayer  Denoise in Bayer space, output DNG. ~70 MB output.
         Experimental. Preserves RAW editability.
  post   Denoise already-edited TIFFs/PNGs. For post-export workflow.

Workflow:
  1. Import RAW into Darktable
  2. Run: opendenoise ~/Photos/*.ARW --mode bayer
  3. Import denoised DNGs into Darktable, edit non-destructively
        """,
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="+",
        help="Input files or directories",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output directory (default: INPUT_denoised/)",
    )
    parser.add_argument(
        "-m",
        "--mode",
        choices=["raw", "bayer", "post"],
        default=None,
        help="Processing mode (default: auto-detect from file type)",
    )
    parser.add_argument(
        "-s",
        "--strength",
        type=float,
        default=0.5,
        help="Denoise strength 0.0-1.0 (default: 0.5)",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL,
        help="Path to denoise model (.pth/.safetensors)",
    )
    parser.add_argument(
        "--tile",
        type=int,
        default=None,
        help="Tile size for large images (default: auto)",
    )
    parser.add_argument("--fp16", action="store_true", help="Half precision")
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument(
        "--gamma",
        action="store_true",
        help="Apply gamma curve before denoising (bayer mode). "
        "Makes linear data look like sRGB for the model.",
    )
    parser.add_argument(
        "--compression",
        choices=["zstd", "deflate", "lzw", "none"],
        default="zstd",
        help="TIFF compression for raw mode (default: zstd)",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default="_denoised",
        help="Filename suffix (default: _denoised)",
    )
    parser.add_argument(
        "--no-suffix",
        action="store_true",
        help="No filename suffix (use with separate output dir)",
    )

    args = parser.parse_args()

    # Collect input files
    files = collect_files(args.input)
    if not files:
        print("No supported files found.", file=sys.stderr)
        sys.exit(1)

    # Auto-detect mode
    mode = args.mode or detect_mode(files)

    # Validate mode vs file types
    if mode in ("raw", "bayer"):
        raw_files = [f for f in files if f.suffix.lower() in RAW_EXTENSIONS]
        if not raw_files:
            print(
                f"Error: mode '{mode}' requires RAW files, but none found.",
                file=sys.stderr,
            )
            sys.exit(1)
        files = raw_files
    elif mode == "post":
        post_files = [f for f in files if f.suffix.lower() in POST_EXTENSIONS]
        if not post_files:
            print(f"Error: mode 'post' requires TIFF/PNG/JPG files.", file=sys.stderr)
            sys.exit(1)
        files = post_files

    # Output directory
    if args.output:
        output_dir = args.output
    else:
        # Derive from first input
        first_input = args.input[0]
        if first_input.is_dir():
            output_dir = Path(str(first_input).rstrip("/") + "_denoised")
        else:
            output_dir = first_input.parent / (first_input.parent.name + "_denoised")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model check
    if not args.model.exists():
        print(f"Error: Model not found: {args.model}", file=sys.stderr)
        sys.exit(1)

    # Load model and setup device
    from .engine import get_device, load_model, print_device_info

    device = get_device(args.cpu)
    print_device_info(device)
    print(f"Model: {args.model.name}")
    print(f"Mode: {mode}")
    if args.strength < 1.0:
        print(f"Strength: {args.strength}")

    model = load_model(args.model, device, args.fp16)

    print(f"Processing {len(files)} files -> {output_dir}/")
    print()

    total_start = time.time()
    for i, f in enumerate(files, 1):
        start = time.time()

        # Determine output path
        if mode == "bayer":
            ext = ".dng"
        elif mode == "raw":
            ext = ".tif"
        else:
            ext = f.suffix

        if args.no_suffix:
            out_name = f.stem + ext
        else:
            out_name = f.stem + args.suffix + ext
        out_path = output_dir / out_name

        # Process based on mode
        if mode == "raw":
            from .mode_raw import process_raw

            process_raw(
                f,
                out_path,
                model,
                device,
                args.strength,
                args.tile,
                args.fp16,
                args.compression,
            )
        elif mode == "bayer":
            from .mode_bayer import process_bayer

            process_bayer(
                f,
                out_path,
                model,
                device,
                args.strength,
                args.tile,
                args.fp16,
                args.gamma,
            )
        elif mode == "post":
            from .mode_post import process_post

            process_post(
                f, out_path, model, device, args.strength, args.tile, args.fp16
            )

        elapsed = time.time() - start
        size_mb = out_path.stat().st_size / (1024**2)
        print(
            f"  [{i}/{len(files)}] {f.name} -> {out_name} ({size_mb:.0f} MB, {elapsed:.1f}s)"
        )

    total = time.time() - total_start
    avg = total / len(files) if files else 0
    print(f"\nDone. {len(files)} files in {total:.1f}s ({avg:.1f}s avg)")


if __name__ == "__main__":
    main()
