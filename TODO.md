# OpenDenoise — potential improvements

## Model improvements

- **Train on real sensor noise**: SCUNet was trained on synthetic noise. A model fine-tuned on paired real RAW data (e.g., short-exposure/long-exposure pairs from actual cameras) would be a massive quality improvement. This is the single biggest lever.
- **Noise-level-aware models**: Instead of a fixed model + strength slider, use a model that takes noise level as an input parameter (like NAFNet or Restormer variants). Could estimate noise level per-image from the Bayer data.
- **Try other architectures**: NAFNet, Restormer, SwinIR, HAT — all from the same 2022 era. Some may handle RAW linear data better than SCUNet.
- **GAN vs PSNR model blending**: Instead of one or the other, blend GAN and PSNR model outputs for controllable sharpness/smoothness tradeoff.

## Bayer pipeline improvements

- **Process all 4 channels natively**: Currently we average G1/G2 to make pseudo-RGB for the 3-channel model. A 4-channel model (or running the model on each channel independently) would preserve the G1/G2 difference better and potentially catch channel-specific noise patterns.
- **Full-resolution processing**: We currently pack 2x2 → half-res for the model. This means the model sees noise at half the spatial resolution. Running at full res (processing each Bayer channel at full res independently, or using a model that accepts single-channel input) could preserve finer detail. Tradeoff: 4x slower.
- **Noise-adaptive strength**: Estimate per-region noise level and adjust strength spatially — denoise more in shadows (where noise is worst) and less in highlights (where there's less noise and more detail to preserve).
- **Preserve hot/dead pixel maps**: Some sensors have known hot pixels. The denoise model might smooth these out but darktable's hot pixel correction should handle them. Verify this doesn't cause issues.
- **Lossless JPEG compression**: Switch from uncompressed tiles (~120MB) to lossless JPEG (TIFF compression=7) for ~50-60MB output. Requires implementing LJPEG encoding or finding a library that does it.

## Color/metadata improvements

- **Copy full EXIF data**: Currently we only write the minimum DNG tags. Copying lens info, GPS, timestamps, etc. from the original would make the DNG more useful for organizing/searching.
- **ColorMatrix2 / dual illuminant**: We only write ColorMatrix1 (D65). Adding ColorMatrix2 (Standard A / tungsten) would improve color accuracy under different lighting.
- **ForwardMatrix1/2**: DNG supports forward matrices for better color rendering. Adobe DNG Converter writes these.
- **BaselineExposure**: Setting this correctly could make the DNG's default brightness match the original ARW.
- **Lens correction data**: Copy OpcodeList1/2/3 from original for vignetting/distortion correction.

## Workflow improvements

- **Darktable Lua plugin**: The plugin exists but hasn't been tested end-to-end since the DNG fixes. Need to verify it works with the current pipeline.
- **Batch processing with progress bar**: Add tqdm or similar for better progress feedback on large batches.
- **Watch mode**: Monitor a folder for new RAW files and auto-denoise them.
- **Side-by-side preview**: CLI tool that exports a small crop at multiple strengths for quick comparison without opening darktable.
- **XMP sidecar copying**: Copy darktable edit history from the original ARW's XMP to the denoised DNG, so edits carry over automatically.

## Performance improvements

- **FP16 inference**: The `--fp16` flag exists but hasn't been heavily tested. Could cut VRAM usage and speed up inference on GPUs with good FP16 throughput.
- **Batch GPU processing**: Process multiple tiles or images simultaneously if VRAM allows.
- **ONNX/TensorRT export**: Convert the model to ONNX for potentially faster inference, especially on NVIDIA GPUs via TensorRT.
- **Async I/O**: Overlap file reading/writing with GPU inference for better throughput on batches.

## Other modes

- **DNG output for raw mode**: Currently raw mode outputs TIFF. Could output a linear DNG (demosaiced but still linear) that darktable handles as RAW-like.
- **Selective denoise**: Denoise only specific regions (e.g., shadows) using a mask derived from the Bayer data.
- **Video support**: Denoise individual frames from RAW video formats (CinemaDNG, BRAW).
