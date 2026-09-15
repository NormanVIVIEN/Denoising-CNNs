# Denoising CNNs — Autoencoder vs. DnCNN vs. U-Net

A small, from-scratch comparison of three convolutional architectures for
Gaussian image denoising: a plain convolutional **autoencoder**, **DnCNN**
(Zhang et al., 2017), and a **U-Net**-style autoencoder with skip
connections. The goal is a controlled, apples-to-apples comparison: the
same data split, the same noise model, and the same training procedure
across all three models.

> **Status note.** Of the three architectures, only the plain autoencoder
> has actually been trained and evaluated end-to-end so far (results
> below). The U-Net has not been trained yet, and the current `DnCNN.py`
> is an incomplete draft — see [Detailed model descriptions](#detailed-model-descriptions)
> and [Known issues / TODO](#known-issues--todo) for exactly what is and
> isn't verified.

## Problem statement

Learn a function `f` that removes additive Gaussian noise from a
corrupted image and recovers the clean image:

```
x_noisy = x + n,   n ~ N(0, sigma^2)
x_hat   = f(x_noisy) ≈ x
```

where `x` is a natural 128×128 grayscale image with pixel values
normalized to `[0, 1]`. (The loader function that produces these images
is named `_to_gray64` in the current code — a leftover name from an
earlier, smaller image size; it actually resizes to 128×128, not 64×64.)

## Method

1. **Data split** (`src/data.py`, currently `DataSplit.py`). `.jpg`
   files in the data directory are listed with `sorted(glob.glob(...))`
   for a deterministic file order, the first `n_train + n_test` are
   loaded and converted to grayscale/resized/normalized, and then
   `sklearn.model_selection.train_test_split(..., shuffle=True)` splits
   them into a training set and a held-out test set that are never
   mixed during training.
2. **Noise model** (`src/noise.py`). Additive Gaussian noise with a
   fixed standard deviation `sigma`, clipped back to `[0, 1]` after
   adding (real sensors can't represent values below black or above
   white). Two *separate, seeded* `torch.Generator` instances produce
   one **fixed** noise realization per set for evaluation only (in the
   reference notebook: seed `4321` for the test set, `1321` for the
   train-eval set) so that reconstruction error stays comparable across
   epochs and across models. During **training**, noise is instead
   re-sampled fresh at every batch from the unseeded default RNG —
   never the fixed evaluation draw — which stops the model from
   memorizing one noise pattern and acts as a form of data augmentation.
3. **Models** (`src/models.py`) — see the detailed section below.
4. **Training** (`src/train.py`, currently `Train.py`). Each model is
   trained with the same hybrid loss:

   ```
   loss = alpha * MSE(f(x_noisy), x) + (1 - alpha) * (1 - SSIM(f(x_noisy), x))
   ```

   (`alpha = 0.8` by default, using `pytorch_msssim.SSIM(data_range=1.0,
   size_average=True, channel=1)`). Per epoch: batches are drawn in a
   random order (`torch.randperm`), noise is resampled per batch, and a
   standard forward/backward/Adam-step updates the weights against the
   *clean* target. Every 5 epochs the model is scored on both
   fixed-noise sets (train and test); the test set never contributes a
   gradient. Checkpoints (model state, optimizer state, loss history)
   are saved every 50 epochs plus once at the end of training.
5. **Evaluation** (`src/evaluate.py`, currently split across `AECNN.py`'s
   `evaluate()` and the notebook). Each model's reconstruction is scored
   with MSE, RMSE, PSNR, and SSIM, and compared against the "do nothing"
   noisy-image baseline (`baseline()`) on both train and test. The fuller
   `evaluate()` in `AECNN.py` also reports three alpha-weighted composite
   scores (`Loss_80`, `Loss_60`, `Loss_40`), used for the alpha-sweep
   section of the notebook.

## Detailed model descriptions

All three models live in `src/models.py` (currently `AECNN.py`,
`UNetCNN.py`, `DnCNN.py`). The descriptions below are based on reading
the actual class definitions, not on the reference papers alone — where
the two disagree, that's called out explicitly.

### 1. `DenoisingAutoencoder` — plain convolutional autoencoder

A fully learned encoder–decoder with **no skip connections**; all
information reaching the decoder must pass through the bottleneck.

```
Encoder (stride-2 convs, kernel 3×3, padding 1):
  128×128×1 → 64×64×base → 32×32×2·base → 16×16×4·base → 8×8×8·base

Decoder (stride-2 transposed convs, kernel 4×4, padding 1, mirrored):
  8×8×8·base → 16×16×4·base → 32×32×2·base → 64×64×base → 128×128×1
```

With the notebook's default `base=64`, that's channel widths
64 → 128 → 256 → 512 and a bottleneck of 8×8×512 = 32,768 values —
larger than the 128×128 = 16,384-pixel input, so the code is an
**over-complete**, not compressive, autoencoder. Regularization instead
comes from the noise corruption itself and convolutional weight sharing.
This is confirmed independently by the parameter count: summing the
encoder/decoder layer sizes for `base=64` gives exactly **4,303,809**
parameters, matching the notebook's printed `Model parameters` line.

Design choices, as commented in the notebook:
- **Stride-2 convolutions instead of max-pooling** for downsampling — a
  learned combination of neighboring pixels averages out independent
  noise while preserving correlated image content, rather than imposing
  a fixed max rule.
- **Decoder kernel is 4×4 (not 3×3) with stride 2** — a kernel divisible
  by the stride gives uniform overlap of the transposed-convolution
  "stamps" at every output pixel, avoiding the checkerboard artifact
  described in Odena, Dumoulin & Olah, *"Deconvolution and Checkerboard
  Artifacts,"* Distill, 2016.
- **Sigmoid on the final layer** (targets lie in `[0, 1]`), **ReLU**
  everywhere else.

> **Correction vs. the notebook's prose.** The notebook's descriptive
> text calls this "three downsampling levels" with an 8×8×128 bottleneck
> and a 4,096-pixel input — those numbers correspond to a *3-layer,
> base=32* variant. The code actually instantiated and trained in the
> same notebook (`DenoisingAutoencoder()`, default `base=64`) has **four**
> stride-2 layers, a **128×128** input, and an **8×8×512** bottleneck —
> confirmed by the printed parameter count above. Recomputing the
> bottleneck's receptive field for the 3-layer description the notebook
> describes gives 15×15, matching its stated figure; for the 4-layer
> model actually trained, the receptive field at the bottleneck is
> **31×31**. We report the verified 4-layer numbers throughout this
> README rather than the notebook's prose figures.

### 2. `UNetDenoisingAutoencoder` — U-Net-style autoencoder with skip connections

Same 4-level encoder/decoder shape and kernel choices as the plain
autoencoder above (`enc1..enc4`: 3×3 stride-2 convs; `dec4..dec1`: 4×4
stride-2 transposed convs), but at each decoder stage the matching
encoder feature map is concatenated channel-wise before the transposed
convolution:

```
e1 = enc1(x)                    # base,   64×64
e2 = enc2(e1)                   # 2·base, 32×32
e3 = enc3(e2)                   # 4·base, 16×16
e4 = enc4(e3)                   # 8·base,  8×8   (bottleneck)

d4 = dec4(e4)                              # 4·base, 16×16
d3 = dec3(concat(d4, e3))                  # 2·base, 32×32
d2 = dec2(concat(d3, e2))                  # base,   64×64
out = dec1(concat(d2, e1))                 # 1,      128×128
```

This gives the decoder direct access to full-resolution detail at every
scale instead of forcing everything through the 8×8 bottleneck, which
should in principle reduce blur relative to the plain autoencoder — this
is a testable hypothesis for the model comparison, not yet a measured
result (see [Known issues](#known-issues--todo)).

The class also exposes a `residual_learning` flag (`False` by default):
- `residual_learning=False`: `torch.sigmoid(out)` is returned directly as
  the denoised image — the same output convention as the plain
  autoencoder.
- `residual_learning=True`: the network instead returns
  `clamp(x - out, 0, 1)` — i.e. the decoder's raw output is treated as
  something to *subtract* from the noisy input, and the final layer has
  no sigmoid in this mode. Note this is **not** identical to DnCNN's
  residual formulation: there's no batch normalization in this network,
  and `out` isn't constrained to look like a noise map during training —
  it's just whatever value makes `x - out` match the clean target.

### 3. `DnCNN` — status: incomplete draft, does not yet match Zhang et al. (2017)

**What the original DnCNN is.** Zhang, Zuo, Chen, Meng & Zhang, *"Beyond
a Gaussian Denoiser: Residual Learning of Deep CNN for Image
Denoising,"* IEEE Transactions on Image Processing, 26(7):3142–3155,
2017 — propose a deep, feed-forward CNN that keeps the spatial
resolution constant throughout (no down/upsampling), uses a constant
number of feature channels per hidden layer, applies batch normalization
in the hidden layers, and is trained to predict the **noise residual**
rather than the clean image directly (`x_hat = x_noisy − predicted_noise`).
The official repository (`cszn/DnCNN`) explains this choice: the
residual of an AWGN-corrupted image follows a constant Gaussian
distribution, which stabilizes batch normalization during training in a
way that a raw clean-image target does not.

**What `DnCNN.py` currently contains** — verified by reading the file:

```python
self.cnn = nn.Sequential(
    nn.Conv2d(1, base, 3, stride=1, padding=1), nn.ReLU(True),
    nn.Conv2d(base, base*2, 3, stride=1, padding=1), nn.ReLU(True),
    nn.Conv2d(base*2, base*4, 3, stride=1, padding=1), nn.ReLU(True),
    nn.Conv2d(base*4, base*8, 3, stride=1, padding=1), nn.ReLU(True),
    nn.Conv2d(base*8, base*16, 3, stride=1, padding=1), nn.ReLU(True),
    nn.Conv2d(base*16, base*32, 3, stride=1, padding=1), nn.ReLU(True),
)
def forward(self, x):
    return self.cnn(x)
```

- ✅ Matches the paper: stride-1, padding-1, 3×3 convolutions throughout,
  so spatial resolution stays at 128×128 the whole way through (no
  down/upsampling).
- ❌ Channel width **doubles at every layer** (`base → 2·base → 4·base →
  8·base → 16·base → 32·base`; with `base=64` that's 64 → 128 → 256 →
  512 → 1024 → **2048**) instead of staying constant, as in the paper.
- ❌ **Every** layer ends in `ReLU`, including the last — so the network's
  output has `32·base` channels (2048 with the default `base=64`) and is
  clamped to be non-negative, not a single-channel image.
- ❌ There is no final projection back to 1 output channel, no
  batch-normalization layers, and no subtraction from the input (`x -
  predicted_noise`) — so `forward()` returns a raw multi-channel feature
  map, not a denoised image comparable to the other two models.

**Practical consequence:** `DnCNN.py` cannot currently be dropped into
the same training/evaluation loop as the other two models — `evaluate()`
would fail comparing a `(B, 2048, 128, 128)` output against a `(B, 1,
128, 128)` clean target. To make it a fair third arm of the comparison it
needs, at minimum: a constant hidden width, a final `Conv2d(..., 1, 3,
padding=1)` with no activation, and either a residual subtraction
(`x - predicted_noise`) or a direct clean-image regression (to match the
other two models' output convention) before training. This is tracked in
[Known issues / TODO](#known-issues--todo) rather than silently assumed
to already work.

## Repository structure

```
denoising-autoencoder/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py         # _to_gray64, load_from_folder, train/test split
│   ├── noise.py         # add_gaussian_noise, seeded generators
│   ├── models.py         # DenoisingAutoencoder, UNetDenoisingAutoencoder, DnCNN
│   ├── evaluate.py       # evaluate(), baseline(), hybrid_loss(), comparison table
│   ├── train.py          # train_model(), CLI (argparse), checkpointing
│   └── visualize.py      # reconstruction grids, loss curves, metric bar charts
├── notebook.ipynb        # full comparison notebook, built on top of src/
└── outputs/
    ├── checkpoints/
    └── figures/
```

`src/` is the **target** layout described above (and requested for this
repository). As of this README, the code actually provided sits as flat
top-level scripts — `AECNN.py`, `DnCNN.py`, `UNetCNN.py`, `DataSplit.py`,
`Train.py`, `Result.py` — not yet merged into that package; see
[Known issues / TODO](#known-issues--todo).

## Notebook

`notebook.ipynb` is the main, narrative entry point. The reference run
documented in this README follows five sections: reproducibility setup
(fixed seeds, device selection), data split, noise model, the
`DenoisingAutoencoder`, training, and evaluation/interpretation. Planned
sections not yet reflected in the results below include a parameter-count
and inference-latency comparison across the three architectures, a
same-training-budget comparison, an `alpha` (MSE-vs-SSIM) sweep using the
`Loss_80` / `Loss_60` / `Loss_40` metrics already computed by
`evaluate()`, a batch-size sweep, and a comparison against non-learned
classical denoisers (Gaussian blur, median filter, Non-Local Means).
Point `args.data_dir` at your own image folder and run it top to bottom;
`src/train.py`'s planned CLI (below) is meant to cover the same ground
non-interactively once the `src/` migration is done.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires a folder of `.jpg` source images (any natural-image dataset;
`n_train + n_test` images are used, sorted by filename for a
deterministic selection).

## Usage (target CLI, once the `src/` migration is complete)

```bash
python -m src.train --model autoencoder --data-dir /path/to/images
python -m src.train --model dncnn       --data-dir /path/to/images
python -m src.train --model unet --residual-learning --data-dir /path/to/images
python -m src.train --compare-all --data-dir /path/to/images
```

| Flag | Default | Meaning |
|---|---|---|
| `--sigma` | `0.15` | Std. dev. of the additive Gaussian noise |
| `--epochs` | `500` | Training epochs |
| `--batch-size` | `100` | Mini-batch size |
| `--lr` | `1e-3` | Adam learning rate |
| `--alpha` | `0.8` | Weight of MSE in the hybrid loss |
| `--base` | `64` | Base channel width for all three architectures |
| `--eval-every` | `5` | Evaluate on train/test every N epochs |
| `--checkpoint-every` | `50` | Save a checkpoint every N epochs |
| `--device` | `auto` | `auto` / `cpu` / `cuda` / `mps` |

The current `Train.py`/`get_device()` only chooses between `mps` and
`cpu` (no `cuda` branch), so the `--device cuda` option above is part of
the target CLI, not something already implemented — it needs adding
alongside the `src/` migration.

Outputs land in `outputs/checkpoints/` (`.pth` files with model weights,
optimizer state, and loss history) and `outputs/figures/` (preview grid,
per-model reconstruction grids, per-model loss curves, and — with
`--compare-all` — a combined test-loss comparison plot).

## Reference results

The only model trained and evaluated end-to-end so far is
`DenoisingAutoencoder`, `sigma=0.15`, 500 epochs, batch size 100, Adam
`lr=1e-3`, `base=64`, on a **400/65** train/test split (this is the
number the notebook itself prints — `Training images : 400` / `Test
images : 65` — not the "~100 images" figure mentioned in one place in
the notebook's own discussion text, which appears to describe an earlier
run with different `args`).

```
RECONSTRUCTION ERROR TABLE (sigma = 0.15, 500 epochs, 964.9s)
-------------------------------------------------------------
Set                              MSE      RMSE   PSNR(dB)
-------------------------------------------------------------
Noisy (baseline) - train      0.01978    0.1407      17.04
Noisy (baseline) - test       0.01985    0.1409      17.02
Denoised - TRAIN               0.00556    0.0746      22.55
Denoised - TEST                0.00667    0.0817      21.76
-------------------------------------------------------------
Generalization gap (test MSE - train MSE): +0.00110
PSNR gain from the model: train +5.51 dB | test +4.74 dB
```

Epoch-by-epoch train/test MSE from the same run (printed every 50
epochs), which already shows the beginning of a train/test split around
epoch 300 even within this 500-epoch budget:

```
epoch    1 | train loss 0.06441 | test loss 0.06383 |    2.6s
epoch   50 | train loss 0.01030 | test loss 0.01044 |   84.9s
epoch  100 | train loss 0.00999 | test loss 0.00992 |  169.7s
epoch  150 | train loss 0.00886 | test loss 0.00875 |  257.4s
epoch  200 | train loss 0.00738 | test loss 0.00730 |  351.6s
epoch  250 | train loss 0.00672 | test loss 0.00682 |  449.7s
epoch  300 | train loss 0.00638 | test loss 0.00668 |  550.6s
epoch  350 | train loss 0.00631 | test loss 0.00680 |  653.4s
epoch  400 | train loss 0.00598 | test loss 0.00672 |  757.6s
epoch  450 | train loss 0.00593 | test loss 0.00685 |  861.1s
epoch  500 | train loss 0.00556 | test loss 0.00667 |  964.8s
```

The notebook's discussion section additionally reports — narratively,
without an accompanying code cell, chart, or table in the materials this
README is based on, so this is relayed as the notebook's own claim
rather than something independently re-verified here — that a separate
**5,000-epoch** run of the same model shows unambiguous overfitting:
train MSE keeps falling to ≈1.5×10⁻³ while test MSE bottoms out around
≈3.5×10⁻³ near epoch 800–1000 and then rises to ≈4.7×10⁻³ by epoch 5000.
If reproduced, the practical implication is to pick the epoch count from
the test-set minimum (early stopping) rather than training as long as
compute allows.

DnCNN and U-Net numbers are **not yet reported** — neither architecture
has been trained end-to-end with this codebase, and as noted above the
current `DnCNN.py` draft isn't structurally able to produce a comparable
denoised-image output yet. Run `python -m src.train --compare-all`
(once the `src/` migration and the `DnCNN` fix are done) to reproduce
this table for all three architectures on your own machine and dataset.

## Reproducibility notes

- `numpy` and `torch` seeds are fixed at the top of both the notebook
  (`SEED = 10`) and `Train.py` (`SEED = 1`) — these are **different
  values in the two entry points**, so don't expect identical numbers
  between them; if you need byte-identical runs across both, set them to
  the same value.
- Device selection is `mps` if available, else `cpu` — no CUDA branch is
  implemented in the code provided.
- The two *evaluation*-noise generators are seeded independently from
  training noise (`torch.Generator().manual_seed(4321)` for test,
  `.manual_seed(1321)` for the fixed train-eval set in the reference
  notebook) so that all models being compared see the identical
  corrupted images at evaluation time.

## Limitations (observed on the autoencoder run above)

- **Dataset size.** With 400 training images, the model has limited
  material from which to learn generalizable image statistics, which is
  a plausible driver of the overfitting reported (see above) past a few
  hundred epochs.
- **Fixed noise level.** Results are reported at a single `sigma`; model
  behavior at substantially lower or higher noise levels is not
  characterized here.
- **MSE-driven blur.** Pixel-wise MSE structurally favors smooth
  reconstructions over sharp ones, since it optimizes toward the
  conditional expectation of the clean image given the noisy input.
- **Over-complete latent code (autoencoder/U-Net).** At `8×8×(8·base)` =
  32,768 values (`base=64`), the bottleneck holds roughly twice as many
  values as the 128×128 = 16,384-pixel input — regularization comes from
  the noise corruption and convolutional weight sharing, not from a
  compressive bottleneck.

## Known issues / TODO

- [ ] **`DnCNN.py` is incomplete** — see the detailed breakdown above.
      Needs: constant hidden channel width, a final single-channel
      output conv with no activation, batch normalization (per the
      original paper), and a residual (or direct) output convention
      matching the other two models before it can be trained and
      compared.
- [ ] **U-Net not yet trained/evaluated** — `UNetDenoisingAutoencoder`
      exists and is architecturally complete, but no run/results for it
      are included in this README yet.
- [ ] **`src/` package not yet assembled** — the code currently exists
      as flat scripts (`AECNN.py`, `DnCNN.py`, `UNetCNN.py`,
      `DataSplit.py`, `Train.py`, `Result.py`); the `src/data.py`,
      `src/noise.py`, `src/models.py`, `src/evaluate.py`, `src/train.py`,
      `src/visualize.py` split and the `argparse`-based CLI described
      under [Usage](#usage-target-cli-once-the-src-migration-is-complete)
      are the target, not the current state.
- [ ] **Seed mismatch** between the notebook (`SEED=10`) and `Train.py`
      (`SEED=1`) — reconcile if bit-for-bit reproducibility across both
      entry points is desired.
