# Denoising CNNs: Autoencoder vs DnCNN vs UNet

A built from scratch comparison of three convolutional architectures for
Gaussian image denoising: a plain convolutional autoencoder, DnCNN
(Zhang et al., IEEE Transactions on Image Processing, 2017), and a UNet
style autoencoder with skip connections (Ronneberger, Fischer and Brox,
MICCAI 2015). Same data split, same noise model, same training loop and
same metrics across all three, so the comparison is fair by construction.

> Status note. The numbers and tables in this README come from
> `notebook.ipynb`'s own saved outputs, an actual run on the 400 train
> and 65 test image split, not from synthetic data or a template. The
> notebook's own section 10 ("Summary") has since been filled in from
> that same run; see [Results summary](#results-summary) below for the
> version reproduced here.

## Problem statement

Learn a function `f` that removes additive Gaussian noise from a
corrupted image and recovers the clean image:

```
x_noisy = clip(x + n, 0, 1),   n ~ N(0, sigma^2)
x_hat   = f(x_noisy) ~= x
```

where `x` is a natural 128 by 128 grayscale image with pixel values
normalized to `[0, 1]`.

## Method

1. **Data** (`data.py`). `load_from_folder` lists `.jpg` files in a
   deterministic, sorted order, converts each to grayscale, resizes to
   128 by 128 with bicubic interpolation, and normalizes to `[0, 1]`
   through `_to_gray64` (a name kept from an earlier notebook; the
   actual output size is controlled by an `IMAGE_SIZE = 128` constant,
   not by the function's name). `split_train_test`, backed by
   `sklearn.model_selection.train_test_split`, then produces a disjoint
   train and test set, done once, before any noise is added, so a later
   comparison between train and test metrics reflects seen versus unseen
   data rather than an artifact of the split.
2. **Noise** (`noise.py`). `add_gaussian_noise` adds independent, identically
   distributed Gaussian noise at a fixed standard deviation `sigma` and
   clips the result back to `[0, 1]`. `make_generator` builds a seeded
   `torch.Generator` for reproducible, fixed evaluation noise; training
   noise instead omits the generator, so it is redrawn fresh at every
   batch, which stops the model from memorizing one noise pattern and
   acts as a form of data augmentation.
3. **Models** (`models.py`), see [Detailed model descriptions]
   (#detailed-model-descriptions) below.
4. **Training** (`train.py`, function `train_model`). All three models
   share the same hybrid loss, implemented as `hybrid_loss` in
   `evaluate.py`:

   ```
   loss = alpha * MSE(f(x_noisy), x) + (1 - alpha) * (1 - SSIM(f(x_noisy), x))
   ```

   (`alpha = 0.8` by default). Each epoch shuffles the training batches
   (`torch.randperm`), corrupts every batch with fresh noise, and takes
   one Adam step. Every `eval_every` epochs the model is scored on both
   fixed noise sets with MSE, RMSE, PSNR and SSIM; the test set never
   contributes a gradient. Checkpoints (model state, optimizer state,
   loss history, hyperparameters) are written every `checkpoint_every`
   epochs plus once at the end of training.
5. **Evaluation** (`evaluate.py`). `evaluate(model, clean, noisy)` scores
   a model's reconstruction; `baseline(clean, noisy)` scores the raw
   noisy image, the "do nothing" floor; `classical_baselines(clean,
   noisy, sigma)` adds Gaussian blur, median filter, and Nonlocal Means
   (computed on CPU through scikit image and scipy, given the true
   `sigma` directly rather than estimating it); `format_comparison_table`
   renders any of these side by side.

## Detailed model descriptions

All three live in `models.py`, share one contract (a single channel 128
by 128 image in `[0, 1]` in, a same shaped denoised image in `[0, 1]`
out), and are registered under `MODEL_REGISTRY` so `build_model(name,
**kwargs)` can instantiate any of them by string name. `count_parameters`
reports the trainable parameter count of any of them. Every number
quoted below (parameter counts, latency) is the real, printed output of
the notebook's own section 3, not a recomputation from first principles,
though the parameter counts were independently checked by summing each
layer's weight and bias shapes and match exactly.

### 1. `DenoisingAutoencoder`, plain convolutional autoencoder

```
Encoder (stride 2 convolutions, kernel 3x3, padding 1):
  128x128x1 -> 64x64xbase -> 32x32x2base -> 16x16x4base -> 8x8x8base

Decoder (stride 2 transposed convolutions, kernel 4x4, padding 1, mirrored):
  8x8x8base -> 16x16x4base -> 32x32x2base -> 64x64xbase -> 128x128x1
```

No skip connections; every piece of information reaching the decoder
passes through the 8x8x8base bottleneck. With `base=64` that bottleneck
holds 32,768 values, more than the 16,384 pixel input, so this is an
overcomplete rather than compressive code; regularization instead comes
from the noise corruption itself and from convolutional weight sharing.
Design choices documented in the class docstring: stride 2 convolutions
instead of max pooling for downsampling (a learned combination of
neighboring pixels reduces the variance of independent noise while
preserving correlated image content, instead of imposing a fixed maximum
rule); a 4x4 decoder kernel with stride 2, not 3x3 (a kernel size
divisible by the stride gives uniform overlap of the transposed
convolution "stamps" at every output pixel, avoiding the checkerboard
artifact described by Odena, Dumoulin and Olah, Distill, 2016); sigmoid
on the final layer since targets lie in `[0, 1]`, ReLU everywhere else.

Measured: **4,303,809 parameters**; **0.33 ms per batch of 8** on the
device used for the notebook run (MPS), about **0.041 ms per image**.

### 2. `UNetDenoisingAutoencoder`, UNet style autoencoder with skip connections

Same four level encoder and decoder shape and kernel choices as the
plain autoencoder, but at each decoder stage the matching encoder
feature map is concatenated channel wise before the transposed
convolution, giving the decoder direct access to full resolution detail
at every scale instead of forcing everything through the bottleneck:

```
e1 = enc1(x)                    # base,   64x64
e2 = enc2(e1)                   # 2base,  32x32
e3 = enc3(e2)                   # 4base,  16x16
e4 = enc4(e3)                   # 8base,   8x8   (bottleneck)

d4 = dec4(e4)                              # 4base, 16x16
d3 = dec3(concat(d4, e3))                  # 2base, 32x32
d2 = dec2(concat(d3, e2))                  # base,  64x64
out = dec1(concat(d2, e1))                 # 1,    128x128
```

A `residual_learning` flag (`False` by default, and left at that default
in the notebook's own comparison) switches the output convention: with
`residual_learning=False`, `sigmoid(out)` is returned directly as the
denoised image; with `residual_learning=True`, the network instead
returns `clamp(x - out, 0, 1)`, mirroring DnCNN's formulation below,
though without DnCNN's batch normalization or constant width.

Measured: **4,960,193 parameters**, the largest of the three; **0.31 ms
per batch of 8**, about **0.038 ms per image**, essentially the same
speed as the plain autoencoder since the two share the same downsampling
schedule.

### 3. `DnCNN`, deep constant width residual CNN

Following Zhang, Zuo, Chen, Meng and Zhang, "Beyond a Gaussian Denoiser:
Residual Learning of Deep CNN for Image Denoising," IEEE Transactions on
Image Processing, volume 26, number 7, 2017 (pages 3142 to 3155). Unlike
the two models above, DnCNN never changes the spatial resolution: every
one of its `num_layers` layers keeps the full 128 by 128 feature map and
a constant channel width (`base`), stacking many small 3x3 receptive
fields to build a large effective receptive field instead of
downsampling. Layer layout, for `num_layers` total layers:

```
1.        Conv(1 -> base, 3x3) + ReLU
2..L-1.   [Conv(base -> base, 3x3, no bias) + BatchNorm + ReLU], repeated
L.        Conv(base -> 1, 3x3), no activation
```

It is trained with residual learning by default (`residual_learning=True`):
rather than predicting the clean image directly, the network predicts
the noise itself, and the clean estimate is recovered as `clamp(x -
predicted_noise, 0, 1)`, the formulation Zhang et al. found both easier
to optimize and empirically better than a direct clean image target.
Batch normalization runs on every middle layer, as in the original
paper, to stabilize training at this depth and work jointly with the
residual formulation.

The notebook instantiates it as `DnCNN(base=64, num_layers=17)`, the
same depth used in the original paper's single noise level, grayscale
configuration. Measured: **556,097 parameters**, by far the smallest of
the three (about an eighth of the autoencoder's), but **79.4 ms per
batch of 8**, about **9.9 ms per image**, roughly two hundred times
slower than the other two per image at inference. Parameter count alone
is a misleading proxy for cost here: DnCNN has no downsampling, so every
one of its 17 layers runs a full resolution 128 by 128 convolution,
while the other two only pay full resolution for one layer at each end.

## Repository structure

```
denoising-autoencoder/
|-- README.md
|-- requirements.txt
|-- __init__.py         # "Source package for the denoising-autoencoder project."
|-- data.py              # _to_gray64, load_from_folder, split_train_test, to_tensor, split_dataset, load_dataset
|-- noise.py              # add_gaussian_noise, make_generator
|-- models.py              # DenoisingAutoencoder, UNetDenoisingAutoencoder, DnCNN, MODEL_REGISTRY, build_model, count_parameters
|-- evaluate.py             # evaluate, baseline, hybrid_loss, classical_baselines, format_comparison_table
|-- train.py                 # set_seed, get_device, build_arg_parser, train_model, main
|-- visualize.py               # preview_clean_noisy, grid_combined, plot_loss_curves, plot_final_metric_bars
`-- notebook.ipynb              # the 11 section comparison described below, already executed once
```

This tree matches what is actually in the project. `requirements.txt`
itself was not among the files reviewed for this README, so its exact
contents are not verified here; based on the imports actually used
across these files, the project needs at least `torch`, `numpy`,
`pandas`, `matplotlib`, `pillow`, `scikit-learn`, `scipy`,
`scikit-image` and `pytorch-msssim`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Requires a folder of `.jpg` source images; `n_train + n_test` of them are
used, sorted by filename for a deterministic selection.

## Usage

`train.py` exposes a full `argparse` command line interface:

```bash
python -m src.train --model autoencoder --data-dir /path/to/images
python -m src.train --model dncnn       --data-dir /path/to/images
python -m src.train --model unet --residual-learning --data-dir /path/to/images
python -m src.train --compare-all --data-dir /path/to/images
```

| Flag | Default | Meaning |
|---|---|---|
| `--model` | none | One of `autoencoder`, `dncnn`, `unet`; ignored if `--compare-all` is set |
| `--compare-all` | off | Train every registered architecture on the same data and noise, then compare |
| `--data-dir` | required | Folder of source `.jpg` images |
| `--n-train` | `400` | Training images |
| `--n-test` | `65` | Test images |
| `--sigma` | `0.15` | Standard deviation of the additive Gaussian noise |
| `--epochs` | `500` | Training epochs |
| `--batch-size` | `100` | Mini batch size |
| `--lr` | `1e-3` | Adam learning rate |
| `--alpha` | `0.8` | Weight of MSE in the hybrid loss |
| `--base` | `64` | Base channel width for all three architectures |
| `--residual-learning` | off | For `unet` or `dncnn`: predict noise and subtract, instead of predicting the clean image directly |
| `--seed` | `10` | Seed for weight initialization, the data split and batch shuffling |
| `--test-noise-seed` | `4321` | Seed for the fixed test noise draw |
| `--train-eval-noise-seed` | `1321` | Seed for the fixed train evaluation noise draw |
| `--eval-every` | `5` | Evaluate on train and test every N epochs |
| `--log-every` | `50` | Print a log line every N epochs |
| `--checkpoint-every` | `50` | Save a checkpoint every N epochs |
| `--outputs-dir` | `outputs` | Root folder for checkpoints and figures |
| `--device` | `auto` | `auto` tries CUDA, then MPS, then CPU; or force `cpu`, `cuda`, `mps` |
| `--no-figures` | off | Skip saving preview, reconstruction and loss figures |

## The comparison, section by section

`notebook.ipynb` walks through eleven numbered sections built entirely on
top of the files above. The numbers quoted here are read straight from
its saved outputs; the run itself used a single Apple Silicon device
(`Device in use: mps`), `seed=10`, `sigma=0.15`, `base=64`, and the
400/65 train and test split.

1. **Configuration and reproducibility.** Seeds `numpy` and `torch` from
   one `args.seed`, resolves the device (CUDA first, then MPS, then
   CPU). Printed: `Device in use: mps`.
2. **Data loading and noise.** Loads the split and draws the two fixed
   evaluation noise realizations. Printed: `Training images : 400`,
   `Test images : 65 (disjoint from training)`, `Image tensor : (1, 128,
   128) (channels, H, W)`, plus a saved preview figure of clean versus
   noisy examples.
3. **The three architectures.** Instantiates all three at `base=64` and
   reports parameter counts and inference latency (batch of 8, 3 warmup
   passes, 20 timed repeats). See
   [Detailed model descriptions](#detailed-model-descriptions) for the
   numbers and what they mean.
4. **Architecture comparison, same loss and batch size.** All three
   trained for 500 epochs, batch size 100, `lr=1e-3`, `alpha=0.8`. This
   is the core comparison:

   | Model | Train MSE | Test MSE | Test RMSE | Test PSNR | Training time |
   |---|---|---|---|---|---|
   | autoencoder | 0.005725 | 0.006006 | 0.07750 | 22.21 dB | 1051.9 s (about 17.5 minutes) |
   | unet | 0.003574 | 0.004775 | 0.06910 | 23.21 dB | 1615.6 s (about 27 minutes) |
   | dncnn | 0.003944 | 0.003639 | 0.06033 | 24.39 dB | 13952.7 s (about 3 hours 53 minutes) |

   DnCNN reaches the lowest test error and highest PSNR with by far the
   fewest parameters, at the cost of a training and inference time an
   order of magnitude longer than the other two, for the reason given
   above (no downsampling). Note also that DnCNN's test MSE here is
   slightly *below* its train MSE, unlike the other two; a plausible
   explanation is batch normalization behaving differently between
   training mode and the `model.eval()` mode used for scoring, but this
   is a plausible reading of the number, not something confirmed by
   inspecting DnCNN's own running statistics here.
5. **Loss function comparison (`alpha`).** Sweeps `alpha` over `[0.0,
   0.4, 0.6, 0.8, 1.0]` on the autoencoder, 500 epochs each:

   | alpha | Test MSE | Test RMSE | Test PSNR |
   |---|---|---|---|
   | 0.0 (pure 1 minus SSIM) | 0.007240 | 0.08509 | 21.40 dB |
   | 0.4 | 0.006362 | 0.07977 | 21.96 dB |
   | 0.6 | 0.007166 | 0.08465 | 21.45 dB |
   | 0.8 (project default) | 0.006129 | 0.07829 | 22.13 dB |
   | 1.0 (pure MSE) | 0.005804 | 0.07619 | 22.36 dB |

   Reported as found, without smoothing it into a cleaner story: the
   result is not monotonic (`alpha=0.6` scores worse than `alpha=0.4`),
   and pure MSE (`alpha=1.0`) actually gives the best test MSE and PSNR
   here, ahead of the project's own `alpha=0.8` default. A single run
   per `alpha`, with no repeated seeds, means this could partly be
   run to run variance rather than a real trend; section 11 of the
   notebook itself proposes multiple seeds per configuration as a next
   step for exactly this reason.
6. **Effect of batch size.** Same model and loss, `batch_size` swept
   over `[16, 50, 100, 200]`, 500 epochs each:

   | Batch size | Test MSE | Test PSNR | Wall time |
   |---|---|---|---|
   | 16 | 0.006388 | 21.95 dB | 1091.1 s |
   | 50 | 0.005725 | 22.42 dB | 1308.1 s |
   | 100 | 0.006442 | 21.91 dB | 2087.7 s |
   | 200 | 0.006315 | 22.00 dB | 1038.3 s |

   Batch size 50 gives the best test PSNR here; batch size 200 is the
   fastest of the four (about 1038 s). The `batch_size=100` wall time is
   not a clean read on cost: its own epoch log jumps from 400.8 s at
   epoch 100 to 1449.6 s at epoch 150, a jump about five times larger
   than the roughly 200 s every other 50 epoch stretch takes in this same
   run and in the other batch sizes, consistent with the run being
   paused or sharing the machine partway through rather than batch size
   100 genuinely costing that much more. Treat every `wall_time_s` value
   in this table with that in mind; the MSE and PSNR values are not
   affected by it.
7. **Overfitting.** One model (autoencoder), 2000 epochs, batch size
   100, `alpha=0.8`:

   ```
   epoch    1 | train MSE 0.06442 | test MSE 0.06384
   epoch  200 | train MSE 0.00661 | test MSE 0.00652
   epoch  400 | train MSE 0.00538 | test MSE 0.00644
   epoch  600 | train MSE 0.00483 | test MSE 0.00696
   epoch  800 | train MSE 0.00449 | test MSE 0.00745
   epoch 1000 | train MSE 0.00413 | test MSE 0.00758
   epoch 1200 | train MSE 0.00394 | test MSE 0.00788
   epoch 1400 | train MSE 0.00387 | test MSE 0.00829
   epoch 1600 | train MSE 0.00371 | test MSE 0.00837
   epoch 1800 | train MSE 0.00360 | test MSE 0.00830
   epoch 2000 | train MSE 0.00353 | test MSE 0.00845
   ```

   Minimum test MSE, printed directly by the notebook: **epoch 335**
   (the log above only prints every 200 epochs, so the exact MSE value
   at epoch 335 itself is not visible, only the epoch number). Train MSE
   keeps falling for the full 2000 epochs while test MSE rises for most
   of the run after that early minimum, the textbook signature of
   overfitting. This is worth comparing to a reference point the
   notebook's own section 7 quotes from an earlier, separate run
   (`denoising_autoencoder.pdf`, 5000 epochs): that run's test MSE
   minimum landed around epoch 800 to 1000, much later than epoch 335
   here. The two are not directly comparable as is, since one ran for
   2000 epochs and the other for 5000; re running this notebook's own
   overfitting section at 5000 epochs would be needed to tell whether
   that gap is a real effect or an artifact of the shorter budget. The
   same log shows a second instance of the timing anomaly noted in
   section 6 above: 3287.4 s at epoch 1600 to 6168.5 s at epoch 1800, a
   jump far larger than any other 200 epoch stretch in that run.
8. **Classical, not learned, baselines.** Gaussian blur, median filter
   and Nonlocal Means (Buades, Coll and Morel, 2005), computed with the
   true `sigma` given directly rather than estimated, compared against
   the three CNNs on the test set:

   ```
   Test set, methods classiques vs CNN (sigma=0.15)
   ------------------------------------------------------------------------
   Set                                    MSE      RMSE   PSNR(dB)     SSIM
   ------------------------------------------------------------------------
   Noisy (baseline)                   0.01985    0.1409      17.02      n/a
   gaussian_blur                      0.00551    0.0742      22.59   0.5969
   median_filter                      0.00743    0.0862      21.29   0.4948
   non_local_means                    0.00594    0.0771      22.26   0.5804
   CNN, autoencoder                   0.00601    0.0775      22.21      n/a
   CNN, unet                          0.00477    0.0691      23.21      n/a
   CNN, dncnn                         0.00364    0.0603      24.39      n/a
   ------------------------------------------------------------------------
   ```

   Worth stating plainly since it runs against the intuitive story: a
   plain Gaussian blur, the simplest possible classical filter, beats
   the plain autoencoder here on both MSE and PSNR (22.59 versus 22.21
   dB), and is close to matching Nonlocal Means. Only UNet and, more
   clearly, DnCNN beat every classical method tested. SSIM is shown as
   not available (`n/a`, `nan` in the raw output) for the noisy baseline
   and the three CNN rows because this particular cell calls `evaluate`
   and `baseline` without passing an SSIM module, so this specific table
   cannot be used to compare CNN SSIM against the classical methods'
   SSIM, only MSE, RMSE and PSNR.
9. **Visual reconstruction examples.** A qualitative clean, noisy,
   denoised grid for two train and two test examples. The notebook
   prints `Meilleur modele selon test_mse (section 4) : dncnn`
   (DnCNN, correctly, per the section 4 table above), but the grid
   itself is drawn from `BEST_MODEL_FOR_VISUALS = "autoencoder"`, a
   variable left at its default rather than updated to match, so the
   saved figure shows the autoencoder's reconstructions, not the best
   performing model's.
10. **Summary.** Filled in directly from the run recorded in sections 3
    through 8, nothing invented or carried over from the earlier
    reference notebook except where explicitly labeled as such. See
    [Results summary](#results-summary) below, which reproduces it.
11. **Going further.** Concrete extension ideas grouped into four areas:
    * *Noise model*: Poisson (shot) noise instead of pure Gaussian; a
      single blind model trained across a range of `sigma` (about 0.05
      to 0.3), possibly conditioned on `sigma` as in FFDNet (Zhang et
      al., 2018); real noisy and clean pairs from an actual camera
      sensor dataset (for example SIDD or DND) instead of synthetic
      noise.
    * *Modeling and loss*: a perceptual loss term based on VGG features
      (Johnson et al., 2016) to reduce the blur characteristic of pure
      MSE; an ablation over DnCNN depth (9, 17, or 25 layers through
      `num_layers`), base width (32, 64, or 128), and kernel size (3x3
      versus 5x5); FLOPs as an efficiency axis alongside parameter count
      and latency.
    * *Training*: data augmentation (flips, 90 degree rotations, random
      crops); a learning rate scheduler with early stopping keyed to the
      test MSE minimum found in section 7; multiple random seeds per
      configuration to estimate variance, motivated directly by the non
      monotonic alpha sweep in section 5 above.
    * *Evaluation and comparison*: BM3D, not covered by scikit image or
      scipy, to complement the classical baselines in section 8; out of
      distribution tests on a different kind of image (faces, scanned
      documents, textures) to check whether the learned priors are
      generic rather than dataset specific.

## Results summary

Filling in the same seven questions the notebook's own section 10
template poses, using the numbers above:

* **Best architecture (test MSE and PSNR):** DnCNN, 0.003639 MSE, 24.39
  dB PSNR, well ahead of UNet (23.21 dB) and the plain autoencoder (22.21
  dB), despite having the fewest parameters of the three.
* **Best loss weighting (`alpha`):** pure MSE (`alpha=1.0`) gave the best
  test MSE and PSNR in this single run sweep; the project's own default,
  `alpha=0.8`, was second best; the sweep was not monotonic, so this
  should be treated as a single run result, not a settled conclusion,
  until it is repeated across seeds.
* **Effect of batch size:** `batch_size=50` gave the best test quality
  (22.42 dB test PSNR); `batch_size=200` was the fastest of the four
  (about 1038 s). Treat the wall time column with caution, especially
  for `batch_size=100`, see the note in section 6 above; it does not
  affect the MSE or PSNR values.
* **Overfitting:** test MSE bottomed out at epoch 335 in the 2000 epoch
  run reported here; by epoch 2000, train MSE had fallen to about
  0.00353 while test MSE had risen to about 0.00845, the expected
  overfitting signature. This minimum is much earlier than the epoch
  800 to 1000 minimum quoted from the separate, earlier 5000 epoch
  reference run; the two are not directly comparable as is because of
  the different epoch budgets, so confirming the gap would need
  rerunning this notebook's own overfitting section at 5000 epochs.
* **CNN versus classical methods:** DnCNN and UNet both clearly beat
  every classical baseline tested. The plain autoencoder did not: a
  plain Gaussian blur scored a higher test PSNR than it.
* **Parameters versus inference time:** not aligned. DnCNN has about an
  eighth of the autoencoder's parameters and less than an eighth of
  UNet's, yet is roughly two hundred times slower per image at
  inference and took about thirteen times as long to train for the same
  epoch budget, because it never downsamples.
* **Smallest, most stable train and test gap:** the autoencoder had the
  smallest generalization gap in section 4 (test MSE minus train MSE of
  about +0.00028), against UNet (+0.00120) and DnCNN (about minus
  0.00031, test error below train error, see the note in section 4
  above).

## Reproducibility notes

* `train.py`'s `--seed` defaults to `10`, matching `notebook.ipynb`'s
  `args.seed`, and both feed the same seed into weight initialization,
  the data split and batch shuffling; the two entry points agree here.
* Device selection matches too: both `train.py`'s `get_device` and the
  notebook's own `get_device` try CUDA, then Apple Silicon MPS, then
  fall back to CPU.
* The two evaluation noise generators are seeded independently from
  training noise (`4321` for the test set, `1321` for the fixed train
  evaluation set, both exposed as `--test-noise-seed` and
  `--train-eval-noise-seed` in `train.py`), so every model being
  compared sees identical corrupted images at evaluation time.
* The run summarized in this README used a single Apple Silicon (MPS)
  device; the latency numbers in particular (milliseconds per image or
  batch) are hardware dependent and would differ on a different machine,
  especially a CUDA GPU.

## Known issues

* **Section 9's visual grid does not show the best model.** It hardcodes
  `BEST_MODEL_FOR_VISUALS = "autoencoder"` instead of picking up the
  `dncnn` result its own preceding print statement identifies as best.
* **Section 8's comparison table is missing CNN SSIM.** `evaluate` and
  `baseline` are called there without an SSIM module, so the classical
  methods get an SSIM score and the CNNs and the noisy baseline do not;
  the MSE, RMSE and PSNR columns remain a fair comparison, SSIM does
  not.
* **Two separate wall time readings look contaminated** by an apparent
  pause or shared machine time: section 6's `batch_size=100` run (400.8 s
  at epoch 100 to 1449.6 s at epoch 150) and section 7's overfitting run
  (3287.4 s at epoch 1600 to 6168.5 s at epoch 1800). Every other epoch
  range in both runs, and the other three batch sizes in section 6, look
  internally consistent, so this reads as an interruption specific to
  those two stretches rather than a real cost difference.
* **The alpha sweep and the batch size sweep are single runs.** Neither
  has repeated seeds, so the specific rankings above (`alpha=1.0` best,
  `batch_size=50` best) carry real run to run uncertainty that the
  notebook's own section 11 already flags as a direction for future
  work.

## References

* Zhang, Zuo, Chen, Meng and Zhang, "Beyond a Gaussian Denoiser:
  Residual Learning of Deep CNN for Image Denoising," IEEE Transactions
  on Image Processing, volume 26, number 7, 2017, pages 3142 to 3155.
* Ronneberger, Fischer and Brox, MICCAI 2015 (UNet).
* Buades, Coll and Morel, 2005 (Nonlocal Means).
* Odena, Dumoulin and Olah, Distill, 2016 (checkerboard artifacts in
  transposed convolutions).
* Zhang, Zuo and Zhang, IEEE Transactions on Image Processing, volume
  27, number 9, 2018, pages 4608 to 4622 (FFDNet).
* Johnson et al., 2016 (perceptual loss from VGG features).
