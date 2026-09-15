"""
Training entry point.

    python -m src.train --model autoencoder --data-dir /path/to/images
    python -m src.train --model dncnn       --data-dir /path/to/images
    python -m src.train --model unet        --data-dir /path/to/images
    python -m src.train --compare-all       --data-dir /path/to/images

Training procedure, for a single model, mirrors the notebook this project
was built from:

1. Load the images and split them into a training set and a held-out test
   set (see ``src.data``); the two never overlap.
2. Draw one *fixed* noise realization per set (train, test), reused at
   every evaluation so reconstruction error is comparable across epochs
   (see ``src.noise``).
3. At every training batch, corrupt the clean batch with a *freshly*
   re-sampled Gaussian noise draw (never the fixed evaluation noise).
   This prevents the model from memorizing a specific noise pattern and
   acts as a form of data augmentation, important when only a few hundred
   images are available.
4. Every ``--eval-every`` epochs, score the model on both fixed-noise
   sets (train and test) with MSE / RMSE / PSNR / SSIM, and log the
   running history. The test set never contributes a gradient.
5. Periodically checkpoint the model (``--checkpoint-every``), plus one
   final checkpoint at the end of training.

``--compare-all`` trains all three architectures back to back on the
exact same data split and noise draws, then saves a combined loss-curve
comparison plot and a single comparison table across all three models -
this is the central comparison the project is built around.
"""

import argparse
import os
import time

import numpy as np
import torch
from pytorch_msssim import SSIM

from . import data as data_mod
from . import evaluate as eval_mod
from . import models as models_mod
from . import noise as noise_mod
from . import visualize as viz_mod

# Fixed seeds for the evaluation noise draws: kept separate from the
# training seed so that "which images ended up in train/test" and "what
# the fixed evaluation noise looks like" can be reasoned about
# independently.
DEFAULT_TEST_NOISE_SEED = 4321
DEFAULT_TRAIN_EVAL_NOISE_SEED = 1321


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device(preference="auto"):
    """Resolve the compute device.

    ``"auto"`` prefers CUDA, then Apple Silicon MPS, then falls back to
    CPU. Pass ``"cpu"``, ``"cuda"`` or ``"mps"`` to force one explicitly
    (raises if that backend isn't available).
    """
    if preference == "cpu":
        return torch.device("cpu")
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is not available.")
        return torch.device("cuda")
    if preference == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("--device mps requested but MPS is not available.")
        return torch.device("mps")

    # auto
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Train a denoising CNN and compare architectures.")

    p.add_argument("--model", choices=sorted(models_mod.MODEL_REGISTRY), default=None,
                    help="Architecture to train. Ignored if --compare-all is set.")
    p.add_argument("--compare-all", action="store_true",
                    help="Train all registered architectures on the same data/noise and compare them.")

    p.add_argument("--data-dir", type=str, required=True,
                    help="Folder containing the source .jpg images.")
    p.add_argument("--n-train", type=int, default=400)
    p.add_argument("--n-test", type=int, default=65)
    p.add_argument("--sigma", type=float, default=0.15, help="Std. dev. of the additive Gaussian noise.")

    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--alpha", type=float, default=0.8,
                    help="Weight of the MSE term in the hybrid loss (alpha*MSE + (1-alpha)*(1-SSIM)).")
    p.add_argument("--base", type=int, default=64, help="Base channel width for all architectures.")
    p.add_argument("--residual-learning", action="store_true",
                    help="For unet/dncnn: predict noise and subtract from input, instead of predicting the clean image directly.")

    p.add_argument("--seed", type=int, default=10, help="Seed for weight init, data split and batch shuffling.")
    p.add_argument("--test-noise-seed", type=int, default=DEFAULT_TEST_NOISE_SEED)
    p.add_argument("--train-eval-noise-seed", type=int, default=DEFAULT_TRAIN_EVAL_NOISE_SEED)

    p.add_argument("--eval-every", type=int, default=5, help="Evaluate on train/test every N epochs.")
    p.add_argument("--log-every", type=int, default=50, help="Print a log line every N epochs.")
    p.add_argument("--checkpoint-every", type=int, default=50, help="Save a checkpoint every N epochs.")

    p.add_argument("--outputs-dir", type=str, default="outputs",
                    help="Root folder for checkpoints/ and figures/ (created if missing).")
    p.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    p.add_argument("--no-figures", action="store_true", help="Skip saving preview/reconstruction/loss figures.")

    return p


def _build_model_for(name, args, device):
    kwargs = {"base": args.base}
    if name in ("unet", "dncnn"):
        kwargs["residual_learning"] = args.residual_learning
    return models_mod.build_model(name, **kwargs).to(device)


def _empty_history():
    return {
        "epoch": [], "t": [],
        "train_mse": [], "test_mse": [],
        "train_rmse": [], "test_rmse": [],
        "train_psnr": [], "test_psnr": [],
        "train_ssim": [], "test_ssim": [],
    }


def train_model(
    model, train_clean, test_clean, train_noisy_fixed, test_noisy,
    sigma, epochs, batch_size, lr, alpha=0.8,
    run_name="model", checkpoint_dir=None, checkpoint_every=None,
    eval_every=5, log_every=50, device=None, verbose=True,
):
    """Train one already-instantiated model and return its loss history.

    This is the low-level, notebook-friendly training loop: every
    hyperparameter is an explicit argument (no dependency on an
    ``argparse`` namespace), so it is easy to call in a sweep — over
    architectures, over ``alpha``, over ``batch_size`` — with only the
    swept value changing between calls.

    Checkpointing is opt-in and two-tiered: pass ``checkpoint_dir`` to
    save a *final* checkpoint at the end of training; additionally pass
    ``checkpoint_every`` to also save periodic snapshots every N epochs
    (useful for the main architecture comparison and an overfitting
    study, less so for a 5-way ``alpha`` sweep where only the final
    model usually matters).

    Returns a ``hist`` dict with ``"epoch"``, ``"t"`` (wall-clock
    seconds) and, for both ``train`` and ``test``, ``mse``/``rmse``/
    ``psnr``/``ssim`` lists logged every ``eval_every`` epochs.
    """
    device = device or train_clean.device
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    ssim_module = SSIM(data_range=1.0, size_average=True, channel=1).to(device)

    if verbose:
        n_params = models_mod.count_parameters(model)
        print(f"[{run_name}] parameters       : {n_params:,}")
        print(f"[{run_name}] optimizer        : Adam, lr = {lr}")
        print(f"[{run_name}] epochs / batch   : {epochs} / {batch_size}")

    hist = _empty_history()
    t0 = time.time()
    n = train_clean.shape[0]

    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)

    def _checkpoint(epoch):
        torch.save({
            "epoch": epoch,
            "run_name": run_name,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": opt.state_dict(),
            "loss_history": hist,
            "hyperparams": {"sigma": sigma, "epochs": epochs, "batch_size": batch_size,
                             "lr": lr, "alpha": alpha},
        }, os.path.join(checkpoint_dir, f"{run_name}_epoch{epoch}.pth" if epoch != epochs
                         else f"{run_name}_final.pth"))

    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            batch = train_clean[perm[i:i + batch_size]]
            # Fresh noise every batch (never the fixed evaluation noise).
            noisy = noise_mod.add_gaussian_noise(batch, sigma)

            output = model(noisy)
            loss, _mse, _ssim_val = eval_mod.hybrid_loss(output, batch, ssim_module, alpha=alpha)

            opt.zero_grad()
            loss.backward()
            opt.step()

        if ep % eval_every == 0 or ep == 1:
            tr, _ = eval_mod.evaluate(model, train_clean, train_noisy_fixed, ssim_module)
            te, _ = eval_mod.evaluate(model, test_clean, test_noisy, ssim_module)
            hist["epoch"].append(ep)
            hist["t"].append(time.time() - t0)
            for split, m in (("train", tr), ("test", te)):
                hist[f"{split}_mse"].append(m["MSE"])
                hist[f"{split}_rmse"].append(m["RMSE"])
                hist[f"{split}_psnr"].append(m["PSNR"])
                hist[f"{split}_ssim"].append(m["SSIM"])

            if verbose and (ep % log_every == 0 or ep == 1):
                print(f"[{run_name}] epoch {ep:4d} | train MSE {tr['MSE']:.5f} | "
                      f"test MSE {te['MSE']:.5f} | {time.time() - t0:6.1f}s")

        if checkpoint_dir and checkpoint_every and ep % checkpoint_every == 0:
            _checkpoint(ep)

    if verbose:
        print(f"[{run_name}] training finished in {time.time() - t0:.1f}s")

    if checkpoint_dir:
        _checkpoint(epochs)

    return hist


def main(argv=None):
    args = build_arg_parser().parse_args(argv)

    if not args.compare_all and args.model is None:
        raise SystemExit("Pass --model {autoencoder,dncnn,unet} or --compare-all.")

    set_seed(args.seed)
    device = get_device(args.device)
    print("Device in use:", device)

    figures_dir = os.path.join(args.outputs_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    train_clean, test_clean = data_mod.load_dataset(args, device)
    print(f"  Training images      : {train_clean.shape[0]}")
    print(f"  Test images          : {test_clean.shape[0]}  (disjoint from training)")
    print(f"  Image tensor shape   : {tuple(train_clean.shape[1:])}  (channels, H, W)")

    # Fixed noise draws, reused for every evaluation across every model so
    # results stay comparable.
    g_test = noise_mod.make_generator(device, args.test_noise_seed)
    test_noisy = noise_mod.add_gaussian_noise(test_clean, args.sigma, g_test)

    g_train_eval = noise_mod.make_generator(device, args.train_eval_noise_seed)
    train_noisy_fixed = noise_mod.add_gaussian_noise(train_clean, args.sigma, g_train_eval)

    if not args.no_figures:
        viz_mod.preview_clean_noisy(
            train_clean, train_noisy_fixed, args.sigma,
            save_path=os.path.join(figures_dir, "preview_clean_noisy.png"),
        )

    ssim_module_report = SSIM(data_range=1.0, size_average=True, channel=1).to(device)
    tr_baseline = eval_mod.baseline(train_clean, train_noisy_fixed, ssim_module_report)
    te_baseline = eval_mod.baseline(test_clean, test_noisy, ssim_module_report)

    model_names = sorted(models_mod.MODEL_REGISTRY) if args.compare_all else [args.model]

    histories = {}
    comparison_rows = [
        ("Noisy (baseline) - train", tr_baseline),
        ("Noisy (baseline) - test", te_baseline),
    ]

    checkpoint_dir = os.path.join(args.outputs_dir, "checkpoints")

    for name in model_names:
        model = _build_model_for(name, args, device)
        hist = train_model(
            model, train_clean, test_clean, train_noisy_fixed, test_noisy,
            sigma=args.sigma, epochs=args.epochs, batch_size=args.batch_size,
            lr=args.lr, alpha=args.alpha, run_name=name,
            checkpoint_dir=checkpoint_dir, checkpoint_every=args.checkpoint_every,
            eval_every=args.eval_every, log_every=args.log_every, device=device,
        )
        histories[name] = hist

        ssim_module_report = SSIM(data_range=1.0, size_average=True, channel=1).to(device)
        tr_m, tr_out = eval_mod.evaluate(model, train_clean, train_noisy_fixed, ssim_module_report)
        te_m, te_out = eval_mod.evaluate(model, test_clean, test_noisy, ssim_module_report)
        comparison_rows.append((f"Denoised - TRAIN ({name})", tr_m))
        comparison_rows.append((f"Denoised - TEST ({name})", te_m))

        print(
            f"[{name}] generalization gap (test MSE - train MSE): {te_m['MSE'] - tr_m['MSE']:+.5f}\n"
            f"[{name}] PSNR gain from the model: train {tr_m['PSNR'] - tr_baseline['PSNR']:+.2f} dB | "
            f"test {te_m['PSNR'] - te_baseline['PSNR']:+.2f} dB"
        )

        if not args.no_figures:
            viz_mod.grid_combined(
                train_clean, train_noisy_fixed, tr_out,
                test_clean, test_noisy, te_out,
                args.sigma, model_name=name,
                save_path=os.path.join(figures_dir, f"{name}_reconstructions.png"),
            )
            viz_mod.plot_loss_curves(
                {name: hist}, title=f"{name} — loss vs. epochs",
                save_path=os.path.join(figures_dir, f"{name}_loss_curve.png"),
            )

    table = eval_mod.format_comparison_table(
        comparison_rows,
        title=f"RECONSTRUCTION ERROR TABLE (sigma={args.sigma}, {args.epochs} epochs)",
    )
    print("\n" + table)

    if len(histories) > 1 and not args.no_figures:
        viz_mod.plot_loss_curves(
            histories, metric="mse",
            title="Model comparison — MSE vs. epochs",
            save_path=os.path.join(figures_dir, "comparison_test_loss.png"),
        )

    return histories, comparison_rows


if __name__ == "__main__":
    main()
