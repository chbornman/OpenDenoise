# OpenDenoise

AI denoising for RAW photography. Outputs DNG files with full non-destructive editing in darktable.

## The hack

We take an AI denoise model (SCUNet) that was trained on regular sRGB photos, and shove raw linear sensor data through it. Then we write the result back as a DNG file that darktable opens as a RAW.

**This shouldn't work well.** The model has never seen linear Bayer data — it was trained on gamma-curved sRGB images with synthetic noise. RAW sensor data is linear (darks are much darker, highlights much brighter), has signal-dependent Poisson-Gaussian noise (not the synthetic Gaussian/JPEG noise the model expects), and we're feeding it at half resolution through a fake RGB conversion.

**But it works.** At strength 0.5-0.75, you get visibly cleaner images with good detail preservation, and the output is a real DNG that darktable treats as a RAW file — full white balance, exposure, tone curves, color grading, all non-destructive.

The model is a swappable black box. The actual contribution is the **pipeline**: extracting Bayer data, packing it for a 3-channel model, writing valid DNG files that darktable/rawspeed actually accepts (which was harder than the denoising). When better RAW-trained models exist, they drop right in.

Long term, the goal is to compile a dataset of real RAW sensor data (paired noisy/clean captures at different ISOs) and train a model that understands linear Bayer data natively — 4-channel RGGB input, signal-dependent noise, the whole thing. The pipeline is ready for that; it's just the model that's borrowed.

No open-source tool currently does what Lightroom's "AI Denoise" does. This is a rough first pass at filling that gap.

## What it does

```
opendenoise ~/Photos/*.ARW --mode bayer --strength 0.5
```

Takes a noisy RAW file (Sony ARW, Canon CR2/CR3, Nikon NEF, etc.) → AI denoise on sensor data → DNG file → open in darktable with full RAW editing latitude.

~18 seconds per 61MP image on an AMD Radeon Pro V620 (ROCm).

## How it works (technical)

### The Bayer denoise pipeline (recommended)

A digital camera sensor captures light through a Color Filter Array (CFA) — a mosaic of red, green, and blue filters arranged in a 2x2 "Bayer" pattern:

```
R  G  R  G  R  G
G  B  G  B  G  B
R  G  R  G  R  G
G  B  G  B  G  B
```

Each pixel only records one color. RAW editors like darktable "demosaic" this pattern into a full-color image by interpolating the missing colors.

**The key insight**: we can denoise *before* demosaicing by reshaping the Bayer data.

#### Step 1: Extract raw Bayer data

Read the raw sensor values (uint16) from the ARW file using rawpy/libraw. Also extract metadata: CFA pattern, black/white levels, color matrix, white balance multipliers, orientation.

#### Step 2: Pack 2x2 Bayer blocks into a half-resolution 4-channel image

Each 2x2 block of the Bayer pattern contains one R, two G, and one B pixel. We reshape the full sensor array into four half-resolution channels by separating the four positions:

```
Full sensor (H x W):           Packed (H/2 x W/2 x 4):
  [R] [G1]                       Channel 0 (top-left):  all R pixels
  [G2] [B]                       Channel 1 (top-right): all G1 pixels
                                  Channel 2 (bot-left):  all G2 pixels
                                  Channel 3 (bot-right): all B pixels
```

A 9566x6374 sensor becomes a 4783x3187x4 image.

#### Step 3: Convert to pseudo-RGB for the 3-channel denoise model

SCUNet expects a 3-channel RGB input. We create pseudo-RGB by averaging the two green channels:

```
Pseudo-R = Channel 0 (R)
Pseudo-G = average(Channel 1, Channel 2)  (average of G1 and G2)
Pseudo-B = Channel 3 (B)
```

This gives us a 4783x3187x3 image — half resolution, but with real color information from the sensor.

#### Step 4: Run AI denoising (SCUNet)

The pseudo-RGB image is normalized to [0,1] range (using black/white levels), then fed through the SCUNet PSNR model. The model runs tiled inference (1024px tiles with 32px overlap) to fit in GPU memory.

**Strength blending**: the output is blended with the original:
```
result = original * (1 - strength) + denoised * strength
```

At strength 0.5, you get half the model's denoising effect — a good balance between noise reduction and detail preservation.

#### Step 5: Apply corrections back to all 4 Bayer channels

The model denoised 3 channels, but we have 4. We compute the per-pixel noise that was removed from each RGB channel:

```
noise_R = original_pseudo_R - denoised_R
noise_G = original_pseudo_G - denoised_G
noise_B = original_pseudo_B - denoised_B
```

Then subtract the appropriate noise from each of the 4 Bayer channels based on its color:
- Channel 0 (R): subtract noise_R
- Channel 1 (G1): subtract noise_G
- Channel 2 (G2): subtract noise_G
- Channel 3 (B): subtract noise_B

This preserves the subtle difference between G1 and G2 while applying the same denoising.

#### Step 6: Unpack back to Bayer and write DNG

The 4 channels are interleaved back into a full-resolution Bayer array, rescaled to uint16, and written as a DNG file with:

- **SubIFD structure**: thumbnail in main IFD (NewSubfileType=1), raw CFA data in SubIFD (NewSubfileType=0) — required by rawspeed (darktable's RAW decoder)
- **Uncompressed tiled storage**: 256x256 tiles, no compression (rawspeed only supports deflate for float data)
- **CFA tags**: CFAPattern, CFARepeatPatternDim, CFALayout
- **Color tags**: ColorMatrix1 (SRATIONAL), AsShotNeutral (RATIONAL), CalibrationIlluminant1
- **Level tags**: BlackLevel, BlackLevelRepeatDim, WhiteLevel
- **Orientation**: copied from original EXIF

The result is a ~120MB DNG that darktable opens as a RAW file with full non-destructive editing.

### Why denoise in Bayer space?

1. **Noise is simpler in linear RAW data** — sensor noise follows predictable Poisson/Gaussian distributions that haven't been warped by tone curves, gamma, white balance, or color grading.

2. **Preserves RAW editability** — the output is a real CFA image. Darktable applies its own demosaic, white balance, exposure, etc. You retain full editing freedom.

3. **4x faster** — operating at half resolution (due to 2x2 packing) means ~4x fewer pixels for the model to process. A 61MP image takes ~18s instead of ~70s.

4. **Smaller files** — ~120MB DNG vs ~250MB linear TIFF from the full-demosaic approach.

### The model: SCUNet

We use [SCUNet](https://github.com/cszn/SCUNet) (2022), specifically the `scunet_color_real_psnr` variant. It's a hybrid Swin Transformer + UNet architecture trained on synthetic noise patterns. The PSNR variant produces smoother, more conservative denoising compared to the GAN variant.

The model is loaded via [spandrel](https://github.com/chaiNNer-org/spandrel), which provides a universal interface for image restoration models.

**Important caveat**: SCUNet (like all current open-source denoise models) was trained on synthetic noise, not real camera sensor noise. It works surprisingly well, but a model fine-tuned on real RAW sensor data would likely produce significantly better results.

## Three processing modes

| Mode | Pipeline | Output | Speed | Size |
|------|----------|--------|-------|------|
| `bayer` | RAW → Bayer denoise → DNG | Editable RAW | ~18s | ~120MB |
| `raw` | RAW → demosaic → denoise linear RGB → TIFF | Linear TIFF | ~70s | ~250MB |
| `post` | TIFF/PNG → denoise → TIFF/PNG | Same format | ~25s | varies |

## Usage

```bash
# Bayer mode (recommended) - outputs DNG
opendenoise ~/Photos/*.ARW --mode bayer

# Adjust denoise strength (0.0 = none, 1.0 = full)
opendenoise photo.ARW -m bayer -s 0.75

# Output to specific directory
opendenoise photo.ARW -m bayer -o ~/denoised/

# Post-edit mode - denoise exported TIFFs
opendenoise ~/exports/*.tif -m post

# Full demosaic mode - outputs linear TIFF
opendenoise photo.ARW -m raw
```

### CLI options

```
opendenoise INPUT [INPUT...] [-m MODE] [-s STRENGTH] [-o OUTPUT]
  -m, --mode       bayer|raw|post (default: auto-detect)
  -s, --strength   0.0-1.0 (default: 0.5)
  -o, --output     Output directory
  --model          Path to .pth model file
  --tile           Tile size (default: auto)
  --fp16           Half precision inference
  --cpu            Force CPU
  --no-suffix      Don't add _denoised suffix
```

## Installation

Requires Python 3.11+, PyTorch (CUDA or ROCm).

```bash
pip install rawpy tifffile imagecodecs spandrel opencv-python-headless

# Clone and run
git clone https://github.com/...
cd OpenDenoise
python -m opendenoise.cli ~/Photos/photo.ARW -m bayer
```

## DNG compatibility notes

Writing DNG files that darktable/rawspeed actually accepts required solving several non-obvious issues:

- **SubIFD structure is mandatory** — rawspeed ignores raw data in the main IFD. The thumbnail goes in the main IFD with `NewSubfileType=1`, raw CFA data goes in a SubIFD with `NewSubfileType=0`.
- **No deflate for integer data** — rawspeed only supports deflate compression for floating-point DNG. Integer CFA data must be uncompressed or lossless JPEG.
- **Tags must be correct types** — ColorMatrix1 must be SRATIONAL (not SLONG), AsShotNeutral must be RATIONAL (not ULONG). Wrong types produce garbled colors.
- **rawpy.postprocess() mutates state** — calling it corrupts `raw_pattern` and other attributes. Must extract all metadata before postprocessing, or re-open the file.

## Project structure

```
opendenoise/
  cli.py          CLI entry point
  engine.py       Model loading, tiled inference, strength blending
  mode_bayer.py   Bayer denoise → DNG pipeline
  mode_raw.py     Full demosaic → denoise → TIFF pipeline
  mode_post.py    Post-export denoise pipeline
models/
  scunet_color_real_psnr.pth   Default model (smoother)
  scunet_color_real_gan.pth    Aggressive model
darktable/
  opendenoise.lua              Darktable Lua plugin
```
