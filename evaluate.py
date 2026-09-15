"""
Metrics and reporting shared across all three models.

Two things are computed against the same noisy/clean tensors for every
model, so results are directly comparable:

* ``baseline()``  - the error of the raw noisy image itself ("do nothing").
* ``evaluate()``  - the error of a model's denoised reconstruction.

The difference between the two is the actual gain a model provides, which
is more informative than an absolute MSE/PSNR number on its own.
"""

import numpy as np
import torch
import torch.nn.functional as F

_EPS = 1e-12


def _mse_rmse_psnr(diff_sq_mean):
    mse = diff_sq_mean
    rmse = mse ** 0.5
    psnr = 10.0 * np.log10(1.0 / max(mse, _EPS))
    return mse, rmse, psnr


@torch.no_grad()
def evaluate(model, clean, noisy, ssim_module=None):
    """Run ``model`` on ``noisy`` and score the reconstruction against ``clean``.

    Returns ``(metrics, output)`` where ``metrics`` has keys
    ``MSE``, ``RMSE``, ``PSNR`` and, if ``ssim_module`` is given, ``SSIM``.
    """
    model.eval()
    out = model(noisy)

    mse, rmse, psnr = _mse_rmse_psnr(torch.mean((out - clean) ** 2).item())
    metrics = {"MSE": mse, "RMSE": rmse, "PSNR": psnr}

    if ssim_module is not None:
        metrics["SSIM"] = ssim_module(out, clean).item()

    return metrics, out


@torch.no_grad()
def baseline(clean, noisy, ssim_module=None):
    """Score the untouched noisy image against ``clean`` (the "do nothing" baseline)."""
    mse, rmse, psnr = _mse_rmse_psnr(torch.mean((noisy - clean) ** 2).item())
    metrics = {"MSE": mse, "RMSE": rmse, "PSNR": psnr}

    if ssim_module is not None:
        metrics["SSIM"] = ssim_module(noisy, clean).item()

    return metrics


def hybrid_loss(output, target, ssim_module, alpha=0.8):
    """Training loss: ``alpha * MSE + (1 - alpha) * (1 - SSIM)``.

    With the default ``alpha=0.8`` the loss is mostly pixel-wise MSE, with
    a smaller SSIM-based term discouraging purely blurry reconstructions.
    Returns ``(loss, mse, ssim_value)`` so callers can log the components
    separately.
    """
    mse = F.mse_loss(output, target)
    ssim_value = ssim_module(output, target)
    loss = alpha * mse + (1 - alpha) * (1 - ssim_value)
    return loss, mse, ssim_value


def classical_baselines(clean, noisy, sigma=None, methods=("gaussian_blur", "median_filter", "non_local_means")):
    """Score a few *non-learned* denoisers against the same clean/noisy pair.

    Useful as a sanity-check floor for the trained CNNs: a CNN that fails
    to beat Non-Local Means, a decades-old classical method, has not
    learned anything a hand-designed filter didn't already capture.

    ``clean``/``noisy`` are ``(N, 1, H, W)`` tensors in ``[0, 1]``. Returns
    ``{method_name: metrics_dict}`` with the same ``MSE``/``RMSE``/``PSNR``/
    ``SSIM`` schema as ``evaluate()``/``baseline()``, so the result can be
    fed straight into ``format_comparison_table`` alongside CNN results.

    Runs on CPU via scikit-image/scipy (these methods are not
    GPU-accelerated); ``sigma`` overrides the per-image noise estimate
    used by Non-Local Means when known (as it is here, from the noise
    model), which is both faster and more accurate than estimating it.
    """
    from scipy.ndimage import median_filter as _median_filter
    from skimage.filters import gaussian as _gaussian_filter
    from skimage.metrics import structural_similarity as _ssim_fn
    from skimage.restoration import denoise_nl_means, estimate_sigma

    clean_np = clean.detach().cpu().numpy()[:, 0]  # (N, H, W)
    noisy_np = noisy.detach().cpu().numpy()[:, 0]

    def _score(denoised_np):
        denoised_np = np.clip(denoised_np, 0.0, 1.0)
        mse, rmse, psnr = _mse_rmse_psnr(float(np.mean((denoised_np - clean_np) ** 2)))
        ssim_vals = [
            _ssim_fn(c, d, data_range=1.0) for c, d in zip(clean_np, denoised_np)
        ]
        return {"MSE": mse, "RMSE": rmse, "PSNR": psnr, "SSIM": float(np.mean(ssim_vals))}

    results = {}

    if "gaussian_blur" in methods:
        out = np.stack([_gaussian_filter(img, sigma=1.0) for img in noisy_np])
        results["gaussian_blur"] = _score(out)

    if "median_filter" in methods:
        out = np.stack([_median_filter(img, size=3) for img in noisy_np])
        results["median_filter"] = _score(out)

    if "non_local_means" in methods:
        out = []
        for img in noisy_np:
            sigma_est = sigma if sigma is not None else float(np.mean(estimate_sigma(img)))
            out.append(denoise_nl_means(img, h=1.15 * sigma_est, fast_mode=True,
                                          patch_size=5, patch_distance=6))
        results["non_local_means"] = _score(np.stack(out))

    return results


def format_comparison_table(rows, title=None):
    """Build a fixed-width text table from ``rows``.

    ``rows`` is an iterable of ``(label, metrics_dict)`` pairs, where each
    ``metrics_dict`` has at least ``MSE``, ``RMSE``, ``PSNR`` and,
    optionally, ``SSIM``. Used to print/save the "noisy vs. each model"
    comparison table.
    """
    rows = list(rows)
    has_ssim = any("SSIM" in m for _, m in rows)

    if has_ssim:
        header = f"{'Set':<32}{'MSE':>10}{'RMSE':>10}{'PSNR(dB)':>11}{'SSIM':>9}"
    else:
        header = f"{'Set':<32}{'MSE':>10}{'RMSE':>10}{'PSNR(dB)':>11}"

    lines = []
    if title:
        lines.append(title)
    lines.append("-" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for name, m in rows:
        line = f"{name:<32}{m['MSE']:>10.5f}{m['RMSE']:>10.4f}{m['PSNR']:>11.2f}"
        if has_ssim:
            line += f"{m.get('SSIM', float('nan')):>9.4f}"
        lines.append(line)

    lines.append("-" * len(header))
    return "\n".join(lines)
