# Denoising CNNs — Autoencoder vs. DnCNN vs. U-Net

A small, from-scratch comparison of three convolutional architectures for
Gaussian image denoising: a plain convolutional **autoencoder**, **DnCNN**
(Zhang et al., 2017), and a **U-Net**-style autoencoder with skip
connections. The goal is a controlled, apples-to-apples comparison: the
same data split, the same noise model, and the same training procedure
across all three models.

## Problem statement

Learn a function `f` that removes additive Gaussian noise from a
corrupted image and recovers the clean image:

```
x_noisy = x + n,   n ~ N(0, sigma^2)
x_hat   = f(x_noisy) ≈ x
```

where `x` is a natural 128x128 grayscale image with pixel values
normalized to `[0, 1]`.

## Method

1. **Data split.** Images are split into a training set and a held-out
   test set that are never mixed during training (`src/data.py`).
2. **Noise model.** Additive Gaussian noise with a fixed standard
   deviation `sigma`. For *evaluation only*, one noise realization per
   set (train, test) is drawn once with a seeded generator and kept
   fixed, so reconstruction error is comparable across epochs and across
   models (`src/noise.py`). During *training*, noise is instead
   re-sampled fresh at every batch — this prevents the model from
   memorizing a specific noise pattern and acts as data augmentation.
3. **Models** (`src/models.py`):
   - `DenoisingAutoencoder` — stride-2 convolutional encoder/decoder,
     transposed convolutions for upsampling, no skip connections.
   - `UNetDenoisingAutoencoder` — the same encoder/decoder shape, but
     with U-Net-style skip connections concatenating encoder features
     into the decoder at every resolution.
   - `DnCNN` — a deep (17 layers by default), constant-width, residual
     CNN that predicts the noise itself rather than the clean image
     (`x_hat = x - predicted_noise`), following Zhang et al., *"Beyond a
     Gaussian Denoiser: Residual Learning of Deep CNN for Image
     Denoising"*, IEEE TIP 2017.
4. **Training** (`src/train.py`). Each model is trained with the same
   hybrid loss:

   ```
   loss = alpha * MSE(f(x_noisy), x) + (1 - alpha) * (1 - SSIM(f(x_noisy), x))
   ```

   (`alpha = 0.8` by default). Every few epochs the model is scored on
   both fixed-noise sets (train and test) with MSE / RMSE / PSNR / SSIM;
   the test set never contributes a gradient. Checkpoints are saved
   periodically and once at the end of training.
5. **Evaluation** (`src/evaluate.py`, `src/visualize.py`). Each model's
   reconstruction error is compared against the "do nothing" noisy-image
   baseline, on both train and test, plus reconstruction grids and
   loss-vs-epoch curves.

## Repository structure

```
denoising-autoencoder/
├── README.md
├── requirements.txt
├── src/
│   ├── data.py         # _to_gray64, load_from_folder, train/test split
│   ├── noise.py         # add_gaussian_noise, seeded generators
│   ├── models.py         # DenoisingAutoencoder, UNetDenoisingAutoencoder, DnCNN
│   ├── evaluate.py       # evaluate(), baseline(), classical_baselines(), hybrid_loss(), comparison table
│   ├── train.py          # train_model(), CLI (argparse), checkpointing
│   └── visualize.py      # reconstruction grids, loss curves, metric bar charts
├── notebook.ipynb        # full comparison notebook, built on top of src/
└── outputs/
    ├── checkpoints/
    └── figures/
```

## Notebook

`notebook.ipynb` is the main, narrative entry point — it imports everything from `src/`
(no duplicated logic) and walks through eleven sections: reproducibility setup, data/noise
loading, the three architectures with a parameter-count and inference-latency comparison,
a same-budget architecture comparison, an `alpha` (MSE vs. SSIM) sweep, a batch-size sweep,
an overfitting study, a comparison against non-learned classical denoisers (Gaussian blur,
median filter, Non-Local Means), qualitative reconstruction examples, a results template to
fill in, and further-work ideas. Point `args.data_dir` at your own image folder and run it
top to bottom; `src/train.py`'s CLI (below) covers the same ground non-interactively.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires a folder of `.jpg` source images (any natural-image dataset;
`n_train + n_test` images are used, sorted by filename for a
deterministic selection).

## Usage

Train a single architecture:

```bash
python -m src.train --model autoencoder --data-dir /path/to/images
python -m src.train --model dncnn       --data-dir /path/to/images
python -m src.train --model unet --residual-learning --data-dir /path/to/images
```

Train and compare all three on the same data split and noise draws:

```bash
python -m src.train --compare-all --data-dir /path/to/images
```

Key options (see `python -m src.train --help` for the full list):

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

Outputs land in `outputs/checkpoints/` (`.pth` files with model weights,
optimizer state, and loss history) and `outputs/figures/` (preview grid,
per-model reconstruction grids, per-model loss curves, and — with
`--compare-all` — a combined test-loss comparison plot).

## Reference results

The table below is from the exploratory notebook this project was built
from: `DenoisingAutoencoder`, `sigma=0.15`, 500 epochs, batch size 100,
Adam `lr=1e-3`, on a 400/65 train/test split.

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

DnCNN and U-Net numbers are not yet reported here — this repository's
`DnCNN` implementation replaces an earlier incomplete draft, and neither
architecture has been trained end-to-end with this exact codebase yet.
Run `python -m src.train --compare-all` to reproduce this table for all
three architectures on your own machine and dataset, and fill in the gap.

## Limitations (observed on the autoencoder run above)

- **Dataset size.** With only a few hundred training images, the model
  has limited material to learn generalizable image statistics from,
  which is a likely driver of overfitting past a few hundred epochs.
- **Fixed noise level.** Results are reported at a single `sigma`; model
  behavior at substantially lower or higher noise levels is not
  characterized here.
- **MSE-driven blur.** Pixel-wise MSE structurally favors smooth
  reconstructions over sharp ones, since it optimizes toward the
  conditional expectation of the clean image given the noisy input.
- **Over-complete latent code (autoencoder/U-Net).** At `8x8x(8*base)`,
  the bottleneck holds more values than the 128x128 input — regularization
  comes from the noise corruption and convolutional weight sharing, not
  from a compressive bottleneck.
