"""
Additive Gaussian noise model.

    x_noisy = clip(x + n, 0, 1),   n ~ N(0, sigma^2)

Values are clipped back to the valid [0, 1] pixel range after adding
noise, since a real sensor cannot represent values below black or above
white.

Two usage patterns matter for this project:

* Fixed evaluation noise: draw the corruption once with a seeded
  ``torch.Generator`` and reuse it for every evaluation pass, so that
  reconstruction error is comparable across epochs and across models
  (the models are being compared on the exact same noisy inputs).
* Fresh training noise: during training, re-sample noise at every batch
  (no generator / an unseeded draw) so the model cannot memorize a
  single noise pattern. This acts as a form of data augmentation and is
  a deliberate, separate choice from the fixed evaluation noise above.
"""

import torch


def add_gaussian_noise(x, sigma, generator=None):
    """Add i.i.d. Gaussian noise with std ``sigma`` to ``x`` and clip to [0, 1].

    Pass a seeded ``generator`` for a reproducible ("fixed") noise draw
    (evaluation sets); omit it for a fresh draw at every call (training).
    """
    noise = torch.randn(x.shape, generator=generator, device=x.device) * sigma
    return torch.clamp(x + noise, 0.0, 1.0)


def make_generator(device, seed):
    """Create a seeded ``torch.Generator`` on ``device`` for reproducible noise."""
    return torch.Generator(device=device).manual_seed(seed)
