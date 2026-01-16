from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple
import numpy as np

@dataclass(frozen=True)
class BreastMNISTSplits:
    x_train: np.ndarray  # (N, 28, 28)
    y_train: np.ndarray  # (N,)
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray

def load_breastmnist(dataset_root: Path) -> BreastMNISTSplits:
    dataset_root = Path(dataset_root)
    npz_path = dataset_root / "breastmnist.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing file: {npz_path}")

    with np.load(npz_path) as d:
        key_sets = [("train_images", "train_labels", "val_images", "val_labels", "test_images", "test_labels"),
                    ("x_train", "y_train", "x_val", "y_val", "x_test", "y_test"),
                    ("train_image", "train_label", "val_image", "val_label", "test_image", "test_label")]
        keys = next((ks for ks in key_sets if all(k in d.files for k in ks)), None)
        if keys is None:
            raise KeyError(f"Unknown npz keys. Found keys={d.files}")
        x_train, y_train, x_val, y_val, x_test, y_test = (np.asarray(d[k]) for k in keys)

    # labels -> (N,) int
    y_train = np.asarray(y_train).reshape(-1).astype(np.int64)
    y_val   = np.asarray(y_val).reshape(-1).astype(np.int64)
    y_test  = np.asarray(y_test).reshape(-1).astype(np.int64)
    x_train = x_train.astype(np.float32) / 255.0
    x_val   = x_val.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32) / 255.0
    return BreastMNISTSplits(x_train, y_train, x_val, y_val, x_test, y_test)

def offline_augmentation(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
    repeats: int = 1,
    augmentation: bool = False, 
    do_flip: bool = True,
    noise_std: float = 0.02,
    brightness_delta: float = 0.15,
    clip: bool = True,
    clip_min: float = 0.0,
    clip_max: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x)
    y = np.asarray(y).reshape(-1)

    if (not augmentation) or repeats <= 0:
        return x, y
    rng = np.random.default_rng(seed)
    aug_x, aug_y = [x], [y]
    for _ in range(repeats):
        x_new = x.astype(np.float32, copy=True)
        for i in range(x_new.shape[0]):
            img = x_new[i]
            if do_flip and rng.random() < 0.5:
                img = np.fliplr(img).copy()  # ensure contiguous copy
            if noise_std > 0:
                img = img + rng.normal(0.0, noise_std, size=img.shape).astype(np.float32)
            if brightness_delta > 0:
                img = img * (1.0 + float(rng.uniform(-brightness_delta, brightness_delta)))
            if clip:
                img = np.clip(img, clip_min, clip_max)
            x_new[i] = img
        aug_x.append(x_new)
        aug_y.append(y)

    return np.concatenate(aug_x, axis=0), np.concatenate(aug_y, axis=0)
