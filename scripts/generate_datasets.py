#!/usr/bin/env python3
"""
Generate synthetic datasets as in the SHARK paper.

Each configuration (n x m - k) is generated `n_repeats` times.
For each dataset:
  - centers ~ N(0, I_m)
  - per-cluster variance sigma^2 ~ U(0.5, 1.5)  (so cluster_std = sqrt(sigma^2))
  - unequal cluster sizes with at least 20 samples per cluster
  - features range-normalized by default (optionally z-score)
  - 50% extra noise features ~ U(0,1) appended (saved as a separate array)

Outputs:
  out/
    1000x10-3/
      seed0000_repeat000/ dataset.npz, metadata.json
      ...
    ...
dataset.npz contains:
  X_clean : (n, m)
  y       : (n,) integer labels in [0, k-1]
  X_noisy : (n, m + nf)  (with nf = round(0.5*m))
"""

import argparse
import json
import math
import os
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.datasets import make_blobs


CONFIGS = [
    # (n, m, k)
    (1000, 10, 3),
    (1000, 10, 5),
    (1000, 10, 10),
    (2000, 20, 5),
    (2000, 20, 10),
    (2000, 20, 20),
    (2000, 30, 5),
    (2000, 30, 10),
    (2000, 30, 20),
    (5000, 50, 10),
    (5000, 50, 20),
    (5000, 50, 50),
]


def sample_cluster_sizes(n: int, k: int, min_size: int = 20, rng: np.random.Generator = None) -> np.ndarray:
    """Return an integer vector of length k summing to n, each >= min_size."""
    if rng is None:
        rng = np.random.default_rng()
    if n < k * min_size:
        raise ValueError(f"Cannot allocate {n} samples into {k} clusters with min_size={min_size}.")
    remaining = n - k * min_size
    # Randomly distribute the remaining samples across clusters
    extra = rng.multinomial(remaining, [1.0 / k] * k)
    return extra + min_size


def range_normalize(X: np.ndarray) -> np.ndarray:
    """Per-feature range normalization: (x - mean) / (max - min)."""
    X = X.astype(float, copy=True)
    mins = X.min(axis=0)
    maxs = X.max(axis=0)
    means = X.mean(axis=0)
    denom = (maxs - mins)
    # Avoid division by zero: if a feature is constant, leave it at 0
    denom[denom == 0] = 1.0
    X = (X - means) / denom
    return X


def zscore_normalize(X: np.ndarray) -> np.ndarray:
    """Per-feature z-score normalization."""
    X = X.astype(float, copy=True)
    means = X.mean(axis=0)
    stds = X.std(axis=0, ddof=0)
    stds[stds == 0] = 1.0
    return (X - means) / stds


def make_dataset(n: int, m: int, k: int, rng: np.random.Generator, normalize: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Build a single dataset according to the paper:
      - centers ~ N(0, I_m)
      - per-cluster variances sigma^2 ~ U(0.5, 1.5); cluster_std = sqrt(sigma^2)
      - unequal cluster sizes (min 20)
      - normalization (range or zscore)
      - append 50% extra noise features ~ U(0,1)
    """
    sizes = sample_cluster_sizes(n, k, min_size=20, rng=rng)

    # Centers ~ N(0, I_m)
    centers = rng.normal(loc=0.0, scale=1.0, size=(k, m))

    # Per-cluster std from sigma^2 ~ U(0.5, 1.5)
    sigma2 = rng.uniform(0.5, 1.5, size=k)
    cluster_std = np.sqrt(sigma2)

    # We need to pass a per-sample array-like to make_blobs for unequal sizes.
    # sklearn accepts a list of ints summing to n.
    # To ensure the sum is exactly n, sizes is already constructed that way.
    X, y = make_blobs(
        n_samples=sizes.tolist(),
        n_features=m,
        centers=centers,
        cluster_std=cluster_std.tolist(),  # one std per cluster (spherical)
        center_box=None,  # ignored because centers provided
        shuffle=True,
        random_state=rng.integers(0, 2**31 - 1),
    )

    # Normalization
    if normalize == "range":
        X_clean = range_normalize(X)
    elif normalize == "zscore":
        X_clean = zscore_normalize(X)
    else:
        raise ValueError("normalize must be 'range' or 'zscore'.")

    # Append 50% noise features ~ U(0,1)
    nf = int(round(0.5 * m))
    noise = rng.uniform(0.0, 1.0, size=(n, nf))
    X_noisy = np.concatenate([X_clean, noise], axis=1)

    meta = {
        "n": n,
        "m": m,
        "k": k,
        "noise_features": nf,
        "normalization": normalize,
        "cluster_sizes": sizes.tolist(),
        "sigma2_uniform": [0.5, 1.5],
        "cluster_std": cluster_std.tolist(),
        "centers_dist": "N(0, I)",
    }
    return X_clean, y.astype(int), X_noisy, meta


def main():
    ap = argparse.ArgumentParser(description="Generate SHARK paper-style synthetic datasets.")
    ap.add_argument("--out", type=str, required=True, help="Output directory.")
    ap.add_argument("--n-repeats", type=int, default=50, help="Datasets per configuration.")
    ap.add_argument("--normalize", choices=["range", "zscore"], default="range",
                    help="Feature normalization (paper uses range; z-score optional).")
    ap.add_argument("--seed", type=int, default=42, help="Global RNG seed.")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    global_rng = np.random.default_rng(args.seed)

    for (n, m, k) in CONFIGS:
        config_dir = out_dir / f"{n}x{m}-{k}"
        config_dir.mkdir(parents=True, exist_ok=True)

        for r in range(args.n_repeats):
            # Use independent seeds per repeat to make runs reproducible and resumable
            repeat_seed = int(global_rng.integers(0, 2**31 - 1))
            rng = np.random.default_rng(repeat_seed)

            X_clean, y, X_noisy, meta = make_dataset(n, m, k, rng, args.normalize)
            sample_dir = config_dir / f"repeat{r:03d}"
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Save arrays
            np.savez_compressed(
                sample_dir / "dataset.npz",
                X_clean=X_clean,
                y=y,
                X_noisy=X_noisy,
            )
            # Save metadata
            meta_out = {
                "repeat": r,
                "seed": repeat_seed,
                **meta,
            }
            with open(sample_dir / "metadata.json", "w", encoding="utf-8") as f:
                json.dump(meta_out, f, indent=2)

            print(f"[OK] {n}x{m}-{k} repeat={r:03d}  -> {sample_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()

