# OpenDenoise -- TODO

## Needs doing before this is ready for others

### Verify what we've built actually works end-to-end

- [ ] Run the CLI bayer mode on a test image after the recent refactor and confirm the output DNG opens correctly in darktable.
- [ ] Test on the other 3 ARW files (CBR08393, DSC05932, test_original) with the default L60/C60 settings to make sure the defaults are reasonable across different images.
- [ ] Test on at least one non-Sony RAW file (Canon CR2/CR3, Nikon NEF) to verify camera-brand-agnostic operation.
- [ ] Test `raw` mode and `post` mode to make sure they still work after the experiment system refactoring.
- [ ] Test `pip install .` in a clean virtualenv and run through the full flow: install, download models, denoise, open in darktable.

### Darktable integration

- [ ] Test the existing Lua plugin (`darktable/opendenoise.lua`) with the current DNG output.
- [ ] Remove the hardcoded `HIP_VISIBLE_DEVICES=0` from the Lua plugin -- it's AMD-specific and will break for NVIDIA users. GPU selection should either be auto-detected or configurable in the plugin UI.
- [ ] Think through whether a proper darktable C module (compiled into darktable) would make more sense than a Lua script. A native module could operate directly on the pixel pipeline without file I/O round-trips. This is a bigger undertaking but would be the right long-term path.
- [ ] Consider XMP sidecar copying -- when denoising produces a new DNG, copy the original file's darktable edit history so edits carry over automatically.

## Known color issues

- [ ] **Green blooming in shadows.** The denoised output has a noticeable green shift in dark areas. This may be related to the model's sRGB training (green channel has 2x the samples in Bayer, and the model may be amplifying that imbalance in low-signal regions), or it could be an artifact of the luma/chroma blending in YCbCr space on linear data. Needs investigation.
- [ ] **Orange/red shift toward magenta.** Reds and oranges come out slightly more magenta than the original. Could be a side effect of the YCbCr decomposition (BT.601 coefficients on linear data aren't strictly correct), the model's color bias, or something in how we merge the two rg1b_rg2b passes. Worth comparing against the original with a color checker or known-color target.
- [ ] **GAN model green cast.** The SCUNet GAN variant has a strong green tint on linear data. Lower priority since we default to PSNR, but worth understanding whether the same mechanism causes the subtler green shadow issue in the PSNR model.

## Needs measuring

The current defaults (L60/C60, rg1b_rg2b, SCUNet PSNR) came from 9 rounds of eyeballing A/B comparisons on one test image. There's almost certainly low-hanging fruit we're missing.

- [ ] Run the pipeline on a wider set of test images at different ISOs, cameras, and subjects.
- [ ] Try proper image quality metrics (PSNR, SSIM, LPIPS) if we can get paired clean/noisy captures. Even synthetic pairs (add noise to clean RAW) would give us something to measure against.
- [ ] Sweep luma/chroma strength independently. We tested L60/C60 together but it's plausible that something like L40/C80 (preserve more detail, kill more color noise) would be better for high-ISO images.
- [ ] Test whether the `--gamma` pre-transform actually helps or hurts now that we're using rg1b_rg2b. Our earlier experiments with pre-transforms were on the old pseudo_rgb strategy.
- [ ] Profile memory usage and speed across different image sizes and GPUs. The auto tile size logic is very basic (>4096px = tile, otherwise don't).
- [ ] Measure the color shifts quantitatively. Compare original vs denoised in LAB or similar perceptual color space to see exactly where the hue rotation happens and how much it varies with strength.

## Model improvements

These are the biggest potential quality levers, roughly in order of expected impact:

- [ ] **Train on real sensor noise.** SCUNet was trained on synthetic noise. A model fine-tuned on paired real RAW data (short-exposure/long-exposure pairs from actual cameras) would be a massive quality improvement. This is the single biggest thing that could improve results.
- [ ] **Try NAFNet.** Higher SIDD benchmark scores than SCUNet, simpler architecture. Downloaded but never tested in our pipeline.
- [ ] **Try other architectures**: Restormer, SwinIR, HAT. All from the same 2022 era, some may handle linear data better.
- [ ] **Noise-level-aware models**: Instead of a fixed model + strength slider, use a model that takes noise level as an input parameter. Could estimate noise level per-image from the Bayer data.
- [ ] **GAN model green cast**: The SCUNet GAN variant has a green tint on linear data. Worth investigating whether this is fixable with a color correction step, or if it's a fundamental issue with the model on linear input.

## Pipeline improvements

- [ ] **Full-resolution processing**: Currently we pack 2x2 into half-res. Running at full resolution (each Bayer channel independently at full spatial resolution) would preserve finer detail but be ~4x slower.
- [ ] **Lossless JPEG DNG compression**: Output DNGs are uncompressed (~120 MB). Lossless JPEG (DNG compression type 7) would cut this to ~50-60 MB. Needs a LJPEG encoder.
- [ ] **Adaptive strength (revisit)**: Per-region strength based on noise level was shelved because it caused block artifacts at the packed half-resolution. Might work better with smoothed strength maps or full-res processing.
- [ ] **Copy full EXIF data**: Currently we write minimal DNG tags. Copying lens info, GPS, timestamps, copyright from the original would make the DNG more useful for organizing.
- [ ] **ColorMatrix2 / dual illuminant**: We only write ColorMatrix1 (D65). Adding a second matrix (Standard A / tungsten) would improve color accuracy under different lighting.
- [ ] **BaselineExposure**: Setting this correctly would make the DNG's default brightness match the original.

## Open-source project health

- [ ] **Tests**: There are currently zero tests. At minimum: a smoke test that runs the pipeline on a small synthetic Bayer array and verifies the output is a valid TIFF/DNG.
- [ ] **CI**: GitHub Actions to run tests, lint, and verify the package installs cleanly.
- [ ] **Sample images**: Can't include full RAW files in the repo (25MB+), but providing a link to a downloadable test image or a small synthetic test case would help people verify their install works.
- [ ] **Better error messages**: What happens when PyTorch isn't installed? When CUDA/ROCm isn't available? When the image format isn't supported? These should be helpful messages, not stack traces.
- [ ] **Changelog**: Start tracking changes as we make releases.
- [ ] **Contributing guide**: If anyone wants to help, they should know how to set up the dev environment and run experiments.
