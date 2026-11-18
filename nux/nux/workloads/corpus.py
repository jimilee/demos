"""Numeric / regression corpus (plan A4).

No labeled evaluation set (e.g. ImageNet val) is available in this environment,
so task accuracy is `pending`. The corpus consists of deterministic synthetic
RGB images passed through the checkpoint's FIXED processor (resize/crop/
normalize), plus stress inputs. Tensors are regenerated deterministically and
identified by sha256 in the corpus manifest.

splits:  tune     candidate screening
         holdout  final re-verification only (plan §9.3)
         stress   boundary inputs (processor domain extremes and beyond-domain)
"""
from __future__ import annotations

import hashlib

import numpy as np
import torch

GENERATOR_VERSION = "synthetic-images-v1"


def _img(kind: str, seed: int, size=320):
    from PIL import Image

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / size
    if kind == "blobs":
        a = np.zeros((size, size, 3), np.float32)
        for _ in range(6):
            cx, cy, r = rng.uniform(0, 1, 2).tolist() + [rng.uniform(0.05, 0.3)]
            col = rng.uniform(0, 1, 3)
            m = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * r * r)))[..., None]
            a = a * (1 - m) + col * m
    elif kind == "gradient":
        c0, c1 = rng.uniform(0, 1, 3), rng.uniform(0, 1, 3)
        t = (xx * np.cos(seed) + yy * np.sin(seed))
        t = (t - t.min()) / (t.max() - t.min())
        a = c0 * (1 - t[..., None]) + c1 * t[..., None]
    elif kind == "texture":
        f = rng.uniform(4, 40, 2)
        a = 0.5 + 0.5 * np.stack([np.sin(2 * np.pi * (f[0] * xx + k)) * np.cos(2 * np.pi * f[1] * yy)
                                  for k in rng.uniform(0, 1, 3)], -1)
    elif kind == "white":
        a = np.ones((size, size, 3), np.float32)
    elif kind == "black":
        a = np.zeros((size, size, 3), np.float32)
    elif kind == "checker1px":
        a = np.repeat(((np.indices((size, size)).sum(0) % 2)[..., None]).astype(np.float32), 3, -1)
    elif kind == "uniform_noise":
        a = rng.uniform(0, 1, (size, size, 3)).astype(np.float32)
    else:
        raise ValueError(kind)
    return Image.fromarray((np.clip(a, 0, 1) * 255).astype(np.uint8), "RGB")


SPEC = (
    [("tune", k, s) for k, s in [("blobs", 1), ("blobs", 2), ("gradient", 3), ("texture", 4),
                                  ("blobs", 5), ("texture", 6)]]
    + [("holdout", k, s) for k, s in [("blobs", 101), ("gradient", 102), ("texture", 103), ("blobs", 104)]]
    + [("stress", k, s) for k, s in [("white", 0), ("black", 0), ("checker1px", 0), ("uniform_noise", 201)]]
)


def build_image_corpus(workload, limit_per_split: dict | None = None):
    out = []
    counts = {}
    for split, kind, seed in SPEC:
        counts[split] = counts.get(split, 0) + 1
        if limit_per_split and counts[split] > limit_per_split.get(split, 1 << 30):
            continue
        x = workload.preprocess(_img(kind, seed))
        out.append({"id": f"{split}_{kind}_{seed}", "split": split, "tensors": [x],
                    "sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(), "generator": GENERATOR_VERSION})
    # beyond-processor-domain stress (still a valid fp32 tensor): flag separately
    for name, val in (("const_zero", 0.0), ("const_pos3", 3.0), ("const_neg3", -3.0)):
        if limit_per_split and counts.get("stress", 0) >= limit_per_split.get("stress", 1 << 30):
            break
        x = torch.full((1, 3, 224, 224), val, dtype=getattr(workload, "input_dtype", torch.float32))
        out.append({"id": f"stress_{name}", "split": "stress", "tensors": [x], "beyond_processor_domain": True,
                    "sha256": hashlib.sha256(x.numpy().tobytes()).hexdigest(), "generator": GENERATOR_VERSION})
    return out
