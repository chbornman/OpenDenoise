# Experiments

## Context

We have a pipeline that takes RAW sensor data, packs 2x2 Bayer blocks into half-res pseudo-RGB, runs an sRGB-trained denoise model, and writes the result back as a DNG. Every stage of this pipeline has tunable parameters. We ran 9 rounds of A/B experiments to find reasonable defaults.

All experiments were run on a single test image (CBR08387.ARW, Sony 61MP, high ISO) and evaluated by eyeballing the results in darktable. This is not rigorous -- there are no quantitative metrics, no paired clean/noisy ground truth, and no testing across different cameras or scenes. The results are "what looked best to us on this image."

## Pipeline knobs

```
RAW -> extract Bayer -> normalize [0,1] -> pack 2x2 -> [pre-transform] -> [denoise] -> [post-transform] -> unpack -> DNG
                                                   ^                   ^                    ^
                                             channel strategy      model choice      strength / luma-chroma split
```

### Pre-transform

The model was trained on sRGB images (gamma-curved). Our data is linear. We can transform it before feeding the model.

| Value | Transform | Why try it |
|-------|-----------|------------|
| `none` | identity | Baseline. Feed linear data directly. |
| `gamma` | x^(1/2.2) | Simple gamma. Makes data look sRGB-like. |
| `srgb` | full sRGB EOTF inverse | More accurate sRGB simulation. |
| `sqrt` | x^0.5 | Variance-stabilizing transform for Poisson noise. |

### Channel strategy

How we feed 4-channel Bayer data to a 3-channel model.

| Value | Method | Speed | Tradeoff |
|-------|--------|-------|----------|
| `pseudo_rgb` | Average G1+G2 -> [R, G_avg, B], denoise once | 1x | Loses G1/G2 difference. |
| `rg1b_rg2b` | Two passes: [R,G1,B] and [R,G2,B], average R/B results | 2x | Preserves G1/G2. Real color context. |
| `per_channel` | Denoise each of R, G1, G2, B independently | 4x | No cross-channel context. |

### Strength

Blend between original and model output: `result = original * (1-s) + denoised * s`

### Luma/chroma split

Separate strength for luminance (Y) and chrominance (Cb, Cr) via YCbCr decomposition. Lets you kill color noise aggressively while preserving detail.

### Adaptive strength

Per-pixel strength based on signal level (more denoising in shadows, less in highlights).

## Results

All experiments on CBR08387.ARW. Output size is ~119 MB per DNG (uncompressed).

### Set 1 & 1v2: Pre-transform comparison

Tested `none`, `gamma`, `sqrt`, `srgb` at strength 0.5 with PSNR model.

| Config | Time |
|--------|------|
| `psnr_s50` (none) | ~18-21s |
| `gamma_psnr_s50` | ~18-21s |
| `sqrt_psnr_s50` | ~18-21s |
| `srgb_psnr_s50` | ~18-21s |

**Result**: No clear winner. Pre-transforms didn't obviously outperform plain linear in visual evaluation. We stuck with `none` as the default since it's simpler and there was no compelling reason to add a transform.

### Set 2: Pre-transform x strength x channel strategy grid

2x2x2 grid: {none, gamma} x {s50, s75} x {pseudo_rgb, rg1b_rg2b}.

| Config | Time |
|--------|------|
| Single-pass configs | ~21s |
| Two-pass (rg1b_rg2b) configs | ~38s |

**Result**: rg1b_rg2b was clearly better -- visibly cleaner with fewer artifacts, especially in areas with fine color detail. The 2x speed cost was worth it. Gamma vs none was still not decisive.

### Set 3: PSNR vs GAN model

| Config | Time |
|--------|------|
| `psnr_s50` | ~21s |
| `gan_s50` | ~21s |
| `psnr_s75` | ~21s |
| `gan_s75` | ~21s |

**Result**: GAN model has a green color cast on linear data. PSNR model is smoother and more neutral. Kept PSNR as default.

### Set 4: Adaptive strength

Tested `off`, `linear`, `shadow_boost` at s50 and s75.

**Result**: Both adaptive modes caused visible block artifacts. The strength map operates at the packed half-resolution, which creates blocky transitions. Shelved adaptive strength for now.

### Set 5: Strength sweep with rg1b_rg2b

Swept strength from 0.50 to 0.75 (in steps: 50, 60, 65, 70, 75) with rg1b_rg2b.

| Config | Time |
|--------|------|
| All configs | ~38s |

**Result**: s60-s65 looked like the sweet spot. Lower kept too much noise, higher started losing detail.

### Set 6: Adaptive + rg1b_rg2b

Tested adaptive modes at s65 with rg1b_rg2b.

**Result**: Same blocky artifacts as set 4. Adaptive strength doesn't work well at packed resolution. Confirmed the shelf decision.

### Set 7: Luma/chroma split (first test)

Introduced separate luma and chroma controls. Fixed chroma at 0.8, swept luma: 0.2, 0.3, 0.4.

| Config | Time |
|--------|------|
| `psnr_L20C80_rg1b_rg2b` | ~41s |
| `psnr_L30C80_rg1b_rg2b` | ~41s |
| `psnr_L40C80_rg1b_rg2b` | ~41s |

**Result**: Luma/chroma split is clearly useful. High chroma strength kills color noise effectively while lower luma preserves detail. L30 was the initial favorite.

### Set 8: Chroma sweep

Fixed luma at 0.3, swept chroma: 0.5, 0.7, 0.8, 0.9, 1.0.

| Config | Time |
|--------|------|
| `psnr_L30C50_rg1b_rg2b` | ~41s |
| `psnr_L30C70_rg1b_rg2b` | ~41s |
| `psnr_L30C80_rg1b_rg2b` | ~41s |
| `psnr_L30C90_rg1b_rg2b` | ~41s |
| `psnr_L30C100_rg1b_rg2b` | ~43s |

**Result**: C60-C70 range looked best. C80+ started to look plasticky in skin tones. C50 left too much color noise.

### Set 9: Sweet spot search

3x2 grid: luma {0.4, 0.5, 0.6} x chroma {0.6, 0.7}.

| Config | Time |
|--------|------|
| `psnr_L40C60_rg1b_rg2b` | ~41s |
| `psnr_L40C70_rg1b_rg2b` | ~41s |
| `psnr_L50C60_rg1b_rg2b` | ~41s |
| `psnr_L50C70_rg1b_rg2b` | ~41s |
| `psnr_L60C60_rg1b_rg2b` | ~38s |
| `psnr_L60C70_rg1b_rg2b` | ~41s |

**Result**: L60/C60 was chosen as the default. It's a balanced point that removes meaningful noise without losing too much detail or making things look overprocessed. L40/C60 preserved more texture but left noticeable noise; L60/C70 was slightly too aggressive on color.

## Summary of findings

1. **rg1b_rg2b channel strategy is the clear winner** over pseudo_rgb. Worth the 2x speed cost.
2. **Luma/chroma split is essential.** A single strength slider is too blunt.
3. **L60/C60 is a reasonable default.** But this was tuned on one image. Different ISOs and cameras may want different settings.
4. **Pre-transforms (gamma/srgb/sqrt) didn't clearly help.** Plain linear input works fine, which is surprising given the sRGB-trained model.
5. **Adaptive strength doesn't work at packed resolution.** Causes block artifacts. Would need to revisit at full resolution or with smoothed strength maps.
6. **GAN model has a green cast on linear data.** PSNR model is the safe choice.
7. **SCUNet PSNR is good enough.** We haven't tried other models yet (NAFNet, Restormer), and a model trained on real RAW noise would likely be much better.

## Caveats

- All evaluation was visual (eyeballing in darktable), not quantitative.
- All experiments on a single image from one camera (Sony A7R IV).
- We don't know how these settings generalize to other cameras, ISOs, or subjects.
- There may be obvious improvements we're not seeing because of the narrow test set.

## Running your own experiments

```bash
# Quick comparison of pre-transforms
python -m opendenoise.experiment originals/photo.ARW --pre none gamma sqrt srgb

# Strength sweep
python -m opendenoise.experiment originals/photo.ARW --strength 0.25 0.5 0.75 1.0

# Luma/chroma grid
python -m opendenoise.experiment originals/photo.ARW \
    --luma-strength 0.3 0.5 0.7 \
    --chroma-strength 0.5 0.7 0.9

# From YAML config
python -m opendenoise.experiment experiments.yaml
```

Each experiment outputs to `experiments/<label>/filename.dng`. A `summary.txt` with timing is written to the output directory.
