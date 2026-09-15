"""
Plotting helpers: preview grids, reconstruction grids, and loss curves.

Every function returns the ``matplotlib`` ``Figure`` it built (after an
optional save to disk) rather than calling ``plt.show()`` itself, so it
works the same way in a script, a notebook, or when saving straight to
``outputs/figures/``.
"""

import os

import matplotlib.pyplot as plt


def _maybe_save(fig, save_path):
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    return fig


def preview_clean_noisy(clean, noisy, sigma, n=4, save_path=None):
    """Show ``n`` clean images on top of their noisy counterparts.

    ``clean`` and ``noisy`` are (N, 1, H, W) tensors; only the first
    ``n`` images of each are plotted.
    """
    fig, axes = plt.subplots(2, n, figsize=(2.1 * n, 4.6))
    rows = ["Clean", f"Noisy (sigma={sigma})"]

    for i, (row, img) in enumerate(zip(rows, [clean, noisy])):
        for j in range(n):
            axes[i, j].imshow(img[j].detach().cpu().numpy().squeeze(), cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_title(f"{row} #{j}")
            axes[i, j].axis("off")

    fig.tight_layout()
    return _maybe_save(fig, save_path)


def grid_combined(
    train_clean, train_noisy, train_out,
    test_clean, test_noisy, test_out,
    sigma, k_train=2, k_test=2,
    model_name=None, save_path=None,
):
    """Side-by-side clean / noisy / reconstructed grid for train and test examples.

    Shows ``k_train`` examples from the training set and ``k_test``
    examples from the test set, each as a column with three rows:
    the clean target, the noisy input, and the model's reconstruction.
    """
    n_cols = k_train + k_test
    fig, axes = plt.subplots(3, n_cols, figsize=(2.1 * n_cols, 6.6))
    rows = ["Clean (target)", f"Noisy (sigma={sigma})", "Reconstructed"]

    columns = (
        [(train_clean[j], train_noisy[j], train_out[j], f"Train #{j}") for j in range(k_train)]
        + [(test_clean[j], test_noisy[j], test_out[j], f"Test #{j}") for j in range(k_test)]
    )

    for col_idx, (clean_img, noisy_img, out_img, col_label) in enumerate(columns):
        for r, im in enumerate([clean_img, noisy_img, out_img]):
            a = axes[r, col_idx]
            a.imshow(im[0].detach().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            a.axis("off")
            if r == 0:
                a.set_title(col_label, fontsize=9)
            if col_idx == 0:
                a.text(-0.15, 0.5, rows[r], transform=a.transAxes,
                       ha="right", va="center", fontsize=9, rotation=90)

    title = "Reconstruction examples — train vs. test"
    if model_name:
        title = f"{model_name} — {title}"
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_loss_curves(histories, metric="mse", title=None, save_path=None):
    """Plot train (dashed) vs. test (solid) curves for one or more runs.

    ``histories`` is a mapping ``{run_name: hist}`` where each ``hist``
    has ``"epoch"``, ``f"train_{metric}"`` and ``f"test_{metric}"`` lists,
    as produced by ``train.train_model``. Passing a single-entry dict
    (e.g. ``{"autoencoder": hist}``) plots one model's train/test curves;
    passing several entries overlays them for a direct comparison — this
    is the main plot for comparing architectures, loss weights, or batch
    sizes against each other.

    ``metric`` selects which logged quantity to plot: ``"mse"``,
    ``"rmse"``, ``"psnr"`` or ``"ssim"``.
    """
    fig, ax = plt.subplots(1, 1, figsize=(7, 4.2))
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, (name, hist) in enumerate(histories.items()):
        color = colors[i % len(colors)]
        ax.plot(hist["epoch"], hist[f"train_{metric}"], linestyle="--",
                 color=color, alpha=0.6, label=f"{name} (train)")
        ax.plot(hist["epoch"], hist[f"test_{metric}"], linestyle="-",
                 color=color, label=f"{name} (test)")

    ax.set_xlabel("epoch")
    ax.set_ylabel(metric.upper())
    if metric in ("mse", "rmse"):
        ax.set_yscale("log")
    ax.set_title(title or f"{metric.upper()} vs. epochs")
    ax.legend(fontsize=8, ncol=2 if len(histories) > 1 else 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return _maybe_save(fig, save_path)


def plot_final_metric_bars(final_metrics, metric="test_psnr", title=None,
                            higher_is_better=True, save_path=None):
    """Bar chart of one final metric across several runs/configs.

    ``final_metrics`` is a mapping ``{run_name: {metric_name: value, ...}}``
    (e.g. the ``final_metrics`` dict built while looping over models,
    alphas, or batch sizes in the comparison notebook). Bars are sorted
    best-to-worst according to ``higher_is_better``.
    """
    items = sorted(
        ((name, m[metric]) for name, m in final_metrics.items()),
        key=lambda kv: kv[1], reverse=higher_is_better,
    )
    names = [n for n, _ in items]
    values = [v for _, v in items]

    fig, ax = plt.subplots(1, 1, figsize=(max(4, 1.1 * len(names)), 4))
    bars = ax.bar(names, values)
    ax.set_ylabel(metric)
    ax.set_title(title or metric)
    ax.grid(axis="y", alpha=0.3)
    ax.tick_params(axis="x", rotation=20)

    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4g}",
                 ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    return _maybe_save(fig, save_path)
