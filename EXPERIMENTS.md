# Experiment Plan

## What we're testing

We have a pipeline that takes RAW sensor data, packs 2x2 Bayer blocks into half-res pseudo-RGB, runs an sRGB-trained denoise model, and writes the result back as a DNG. Every stage of this pipeline has alternatives we can test.

## Pipeline stages and knobs

```
RAW → extract Bayer → normalize [0,1] → pack 2x2 → [pre-transform] → [denoise] → [post-transform] → unpack → DNG
                                                ↑                   ↑                    ↑
                                          channel strategy      model choice      adaptive strength
```

### 1. Pre-transform (`--pre`)

The model was trained on sRGB images (gamma-curved). Our data is linear. We can transform it before feeding the model.

| Value | Transform | Inverse | Why try it |
|-------|-----------|---------|------------|
| `none` | identity | identity | Baseline. Current approach. |
| `gamma` | x^(1/2.2) | x^2.2 | Simple gamma. Makes data look sRGB-like. Spreads shadow detail. |
| `srgb` | full sRGB EOTF inverse | full sRGB EOTF | More accurate sRGB simulation (linear toe + gamma). |
| `sqrt` | x^0.5 | x^2 | Variance-stabilizing transform for Poisson noise. Has theoretical justification for shot-noise-dominated data. |
| `log` | log1p(x*10)/log1p(10) | (expm1(x*log1p(10)))/10 | Aggressive shadow lift. Maximally spreads shadow detail for the model. |

**Hypothesis**: `gamma` or `srgb` will outperform `none` since the model expects gamma-curved data. `sqrt` might win in shadow-heavy images due to Poisson noise statistics.

### 2. Model (`--model`)

| Model | File | Training | Character |
|-------|------|----------|-----------|
| SCUNet PSNR | `scunet_color_real_psnr.pth` | Synthetic noise, L1 loss | Smooth, conservative. Current default. |
| SCUNet GAN | `scunet_color_real_gan.pth` | Synthetic noise, adversarial loss | Sharper, preserves texture, can hallucinate. |
| NAFNet | (need to download) | SIDD sRGB pairs | Higher PSNR scores. Simpler architecture. |

### 3. Strength (`--strength`)

Blend between original and model output: `result = original * (1-s) + denoised * s`

Test: 0.25, 0.5, 0.75, 1.0

### 4. Channel strategy (`--channels`)

How we feed 4-channel Bayer data to a 3-channel model.

| Value | Method | Speed | Tradeoff |
|-------|--------|-------|----------|
| `pseudo_rgb` | Average G1+G2, make [R, G_avg, B], denoise once | 1x | Loses G1/G2 difference. Current approach. |
| `per_channel` | Denoise each of R, G1, G2, B independently (duplicate to 3ch) | 4x | No cross-channel context. Model can't use color info. |
| `rg1b_rg2b` | Two passes: [R,G1,B] and [R,G2,B], average R/B results | 2x | Preserves G1/G2 difference. Model gets real color context. |

**Hypothesis**: `rg1b_rg2b` should be best quality (real color context + G1/G2 preserved) at 2x the cost. `per_channel` will likely be worst (no color context).

### 5. Adaptive strength (`--adaptive`)

Instead of uniform strength across the image, vary it based on signal level. Shadows are noisier than highlights in linear RAW data.

| Value | Behavior |
|-------|----------|
| `off` | Uniform strength everywhere. Current approach. |
| `linear` | Strength scales from 1.5x in shadows to 0.5x in highlights. |
| `shadow_boost` | Normal strength everywhere, up to 1.5x in deep shadows (<0.2 linear). |

## Experiment sets to run

### Set 1: Pre-transform comparison (quick, ~4 runs)
```bash
python -m opendenoise.experiment originals/CBR08387.ARW \
    --pre none gamma sqrt srgb \
    --strength 0.5
```
Answers: does transforming to gamma-space help?

### Set 2: Strength sweep with best pre-transform (~4 runs)
```bash
python -m opendenoise.experiment originals/CBR08387.ARW \
    --pre [best from set 1] \
    --strength 0.25 0.5 0.75 1.0
```
Answers: what's the optimal strength with the better pre-transform?

### Set 3: Model comparison (~2-4 runs)
```bash
python -m opendenoise.experiment originals/CBR08387.ARW \
    --pre [best] --strength [best] \
    --model scunet_color_real_psnr.pth scunet_color_real_gan.pth
```
Answers: PSNR vs GAN model character.

### Set 4: Channel strategy comparison (~3 runs)
```bash
python -m opendenoise.experiment originals/CBR08387.ARW \
    --pre [best] --strength [best] \
    --channels pseudo_rgb per_channel rg1b_rg2b
```
Answers: does preserving G1/G2 or per-channel denoising matter visually?

### Set 5: Adaptive strength (~3 runs)
```bash
python -m opendenoise.experiment originals/CBR08387.ARW \
    --pre [best] --strength [best] \
    --adaptive off linear shadow_boost
```
Answers: does spatially-varying strength help in shadows?

### Set 6: Full grid of best candidates (~6-12 runs)
Combine the winners from each set into a final comparison.

## How to evaluate

All outputs are DNG files. Import each experiment folder into darktable and compare:

1. **Overall color accuracy** — does it match the original?
2. **Shadow noise** — zoom into dark areas
3. **Detail preservation** — hair, fabric texture, text
4. **Artifacts** — haloing, color bleeding, banding, tile seams
5. **Highlight behavior** — does it clip or shift?

## Running experiments

```bash
# From CLI with flags
python -m opendenoise.experiment originals/CBR08387.ARW --pre none gamma sqrt

# From YAML config
python -m opendenoise.experiment experiments.yaml

# Example YAML:
# input: originals/CBR08387.ARW
# output: experiments/
# grid:
#   pre: [none, gamma, sqrt, srgb]
#   strength: [0.5, 0.75]
#   model: [scunet_color_real_psnr.pth, scunet_color_real_gan.pth]
```

Each experiment outputs to `experiments/<label>/filename.dng` where `<label>` describes the config (e.g., `gamma_psnr_s50`, `sqrt_gan_s75_rg1b_rg2b`).

A `summary.txt` is written with timing and file sizes for all runs.
