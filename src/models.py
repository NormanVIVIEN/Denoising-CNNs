"""
The three architectures compared in this project. All three share the same
contract: a single-channel 128x128 image in [0, 1] in, a same-shaped
denoised image in [0, 1] out. That shared contract is what makes a direct
comparison between them meaningful.

- DenoisingAutoencoder : plain convolutional encoder/decoder, stride-2
  downsampling / transposed-conv upsampling, no skip connections.
- UNetDenoisingAutoencoder : the same encoder/decoder shape, but with
  U-Net-style skip connections concatenating encoder features into the
  decoder at every resolution.
- DnCNN : a deep, constant-width, residual (noise-predicting) CNN,
  following Zhang et al., "Beyond a Gaussian Denoiser: Residual Learning
  of Deep CNN for Image Denoising", IEEE TIP 2017.
"""

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 1. Convolutional autoencoder
# ---------------------------------------------------------------------------
class DenoisingAutoencoder(nn.Module):
    """Plain convolutional encoder/decoder denoiser.

    Encoder (downsampling, stride-2 convolutions):
        128x128x1 -> 64x64x(base) -> 32x32x(2*base) -> 16x16x(4*base) -> 8x8x(8*base)
    Decoder (upsampling, transposed convolutions, mirrored):
        8x8x(8*base) -> 16x16x(4*base) -> 32x32x(2*base) -> 64x64x(base) -> 128x128x1

    Design choices:
    * Stride-2 downsampling rather than max-pooling: a strided convolution
      learns how to combine neighboring pixels (which reduces the variance
      of independent noise while preserving correlated image content)
      instead of imposing a fixed max rule.
    * Decoder kernels are 4x4 with stride 2 (not 3x3): a kernel size
      divisible by the stride gives uniform overlap of the transposed
      convolution "stamps" at every output pixel, avoiding the
      checkerboard artifact described in Odena, Dumoulin & Olah
      (Distill, 2016).
    * Sigmoid on the final layer, since target pixels lie in [0, 1];
      ReLU everywhere else.
    * At 8x8x(8*base) the latent code is larger than the input
      (an over-complete code, not a compressive bottleneck) -
      regularization instead comes from weight sharing, reduced spatial
      resolution, and the noise corruption itself during training.
    """

    def __init__(self, base=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, base, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), nn.ReLU(True),
            nn.Conv2d(base * 4, base * 8, 3, stride=2, padding=1), nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base * 8, base * 4, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(base * 4, base * 2, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(base * 2, base, 4, stride=2, padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(base, 1, 4, stride=2, padding=1), nn.Sigmoid(),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# 2. U-Net style autoencoder (adds skip connections)
# ---------------------------------------------------------------------------
class UNetDenoisingAutoencoder(nn.Module):
    """U-Net variant of the autoencoder above.

    Same encoder/decoder resolutions and channel widths, but each decoder
    stage concatenates the matching encoder feature map (a skip
    connection) before upsampling, so fine spatial detail lost during
    downsampling can be recovered directly rather than only through the
    bottleneck.

    ``residual_learning=True`` switches the model from predicting the
    clean image directly (sigmoid output) to predicting the noise and
    subtracting it from the input (``out = clamp(x - predicted, 0, 1)``),
    mirroring the DnCNN formulation.
    """

    def __init__(self, base=64, residual_learning=False):
        super().__init__()
        self.residual_learning = residual_learning

        self.enc1 = nn.Sequential(nn.Conv2d(1, base, 3, stride=2, padding=1), nn.ReLU(True))
        self.enc2 = nn.Sequential(nn.Conv2d(base, base * 2, 3, stride=2, padding=1), nn.ReLU(True))
        self.enc3 = nn.Sequential(nn.Conv2d(base * 2, base * 4, 3, stride=2, padding=1), nn.ReLU(True))
        self.enc4 = nn.Sequential(nn.Conv2d(base * 4, base * 8, 3, stride=2, padding=1), nn.ReLU(True))

        self.dec4 = nn.Sequential(nn.ConvTranspose2d(base * 8, base * 4, 4, stride=2, padding=1), nn.ReLU(True))
        self.dec3 = nn.Sequential(nn.ConvTranspose2d(base * 4 + base * 4, base * 2, 4, stride=2, padding=1), nn.ReLU(True))
        self.dec2 = nn.Sequential(nn.ConvTranspose2d(base * 2 + base * 2, base, 4, stride=2, padding=1), nn.ReLU(True))
        self.dec1 = nn.ConvTranspose2d(base + base, 1, 4, stride=2, padding=1)

    def forward(self, x):
        # Encoder, keeping each intermediate output for the skip connections.
        e1 = self.enc1(x)   # (B, base,   64, 64)
        e2 = self.enc2(e1)  # (B, base*2, 32, 32)
        e3 = self.enc3(e2)  # (B, base*4, 16, 16)
        e4 = self.enc4(e3)  # (B, base*8,  8,  8)  <- bottleneck

        # Decoder, concatenating the matching skip connection at each stage.
        d4 = self.dec4(e4)                          # (B, base*4, 16, 16)
        d3 = self.dec3(torch.cat([d4, e3], dim=1))   # (B, base*2, 32, 32)
        d2 = self.dec2(torch.cat([d3, e2], dim=1))   # (B, base,   64, 64)
        out = self.dec1(torch.cat([d2, e1], dim=1))  # (B, 1,     128, 128)

        if self.residual_learning:
            out = x - out
            return torch.clamp(out, 0.0, 1.0)
        return torch.sigmoid(out)


# ---------------------------------------------------------------------------
# 3. DnCNN (Zhang et al., 2017)
# ---------------------------------------------------------------------------
class DnCNN(nn.Module):
    """Deep, constant-width, residual CNN denoiser.

    Unlike the two encoder/decoder models above, DnCNN never changes the
    spatial resolution: every layer keeps the full 128x128 feature map and
    a constant channel width (``base``), stacking many small 3x3
    receptive fields to build up a large effective receptive field
    instead. It is trained with residual learning: rather than predicting
    the clean image directly, the network predicts the noise itself, and
    the clean estimate is recovered as ``x - predicted_noise``. Zhang et
    al. found this residual formulation both easier to optimize and
    empirically better than predicting the clean image directly.

    Layer layout (``num_layers`` total):
        1.            Conv(1 -> base, 3x3) + ReLU
        2..L-1.       [Conv(base -> base, 3x3) + BatchNorm + ReLU]  (repeated)
        L.            Conv(base -> 1, 3x3), no activation (raw noise estimate)

    BatchNorm is used on every middle layer (as in the original paper) to
    stabilize training of a network this deep and to work jointly with
    residual learning to speed up convergence.
    """

    def __init__(self, channels=1, base=64, num_layers=17, residual_learning=True):
        super().__init__()
        if num_layers < 3:
            raise ValueError("DnCNN needs at least 3 layers (in, middle, out).")
        self.residual_learning = residual_learning

        layers = [nn.Conv2d(channels, base, 3, stride=1, padding=1), nn.ReLU(inplace=True)]
        for _ in range(num_layers - 2):
            layers += [
                nn.Conv2d(base, base, 3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(base),
                nn.ReLU(inplace=True),
            ]
        layers += [nn.Conv2d(base, channels, 3, stride=1, padding=1)]
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        predicted_noise = self.dncnn(x)
        if self.residual_learning:
            out = x - predicted_noise
        else:
            out = predicted_noise
        return torch.clamp(out, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
MODEL_REGISTRY = {
    "autoencoder": DenoisingAutoencoder,
    "unet": UNetDenoisingAutoencoder,
    "dncnn": DnCNN,
}


def build_model(name, **kwargs):
    """Instantiate one of the registered models by name.

    ``name`` is one of ``"autoencoder"``, ``"unet"``, ``"dncnn"``.
    Extra ``kwargs`` are forwarded to the model constructor (e.g.
    ``base=64``, or ``residual_learning=True`` for ``unet``/``dncnn``).
    """
    key = name.lower()
    if key not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available models: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key](**kwargs)


def count_parameters(model):
    """Total number of trainable parameters in ``model``."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
