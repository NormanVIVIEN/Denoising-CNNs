"""
Data loading utilities.

Pipeline: JPEG images on disk -> fixed-size grayscale numpy arrays in [0, 1]
-> a disjoint train/test split -> float32 torch tensors of shape
(N, 1, H, W) ready to be fed to a denoising model.

The train/test split is done once, before any noise is added, and the two
sets never overlap: images used for training gradients are never scored as
"test" and vice versa. This is what lets a later comparison between train
and test metrics be interpreted as "seen vs. unseen data" rather than an
artifact of the split.
"""

import glob
import os

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split

IMAGE_SIZE = 128


def _to_gray64(pil_img, size=IMAGE_SIZE):
    """Convert a PIL image to a normalized grayscale numpy array.

    The image is converted to single-channel grayscale, resized to
    ``size x size`` with bicubic interpolation, and rescaled from
    [0, 255] integer pixels to [0, 1] float32.

    Despite the name (kept for continuity with the original notebook),
    the output resolution is controlled by ``size`` (128x128 by default).
    """
    img = pil_img.convert("L").resize((size, size), Image.BICUBIC)
    return np.asarray(img, dtype=np.float32) / 255.0


def load_from_folder(data_dir, n_total, pattern="*.jpg"):
    """Load and preprocess up to ``n_total`` images from ``data_dir``.

    Files are sorted for a deterministic ordering before slicing, so the
    same ``n_total`` always selects the same images regardless of
    filesystem listing order.

    Returns a numpy array of shape (n_total, H, W).
    """
    files = sorted(glob.glob(os.path.join(data_dir, pattern)))
    if not files:
        raise FileNotFoundError(
            f"No images matching '{pattern}' found in '{data_dir}'. "
            "Check --data-dir / args.data_dir."
        )
    if len(files) < n_total:
        raise ValueError(
            f"Requested n_total={n_total} images but only found "
            f"{len(files)} matching '{pattern}' in '{data_dir}'."
        )
    imgs = [_to_gray64(Image.open(f)) for f in files[:n_total]]
    return np.stack(imgs)


def split_train_test(data, n_train, n_test, seed=None, shuffle=True):
    """Split an (N, H, W) array into disjoint train/test numpy arrays."""
    train_data, test_data = train_test_split(
        data,
        train_size=n_train,
        test_size=n_test,
        shuffle=shuffle,
        random_state=seed,
    )
    return train_data, test_data


def to_tensor(data, device):
    """Convert an (N, H, W) numpy array to an (N, 1, H, W) float tensor."""
    return torch.from_numpy(data).unsqueeze(1).to(device)


def split_dataset(data, n_train, n_test, seed, device):
    """Split an (N, H, W) numpy array straight into train/test tensors.

    Convenience wrapper combining ``split_train_test`` and ``to_tensor``:
    useful in a notebook where you already have ``data`` loaded and just
    want ``(train_clean, test_clean)`` tensors on ``device`` in one call.
    """
    train_data, test_data = split_train_test(data, n_train, n_test, seed=seed)
    return to_tensor(train_data, device), to_tensor(test_data, device)


def load_dataset(args, device):
    """End-to-end loader: folder -> disjoint train/test tensors.

    ``args`` must expose ``data_dir``, ``n_train``, ``n_test`` and,
    optionally, ``seed``. Returns ``(train_clean, test_clean)`` as
    ``(N, 1, H, W)`` float tensors on ``device``.
    """
    n_total = args.n_train + args.n_test
    data = load_from_folder(args.data_dir, n_total)
    seed = getattr(args, "seed", None)
    return split_dataset(data, args.n_train, args.n_test, seed, device)
