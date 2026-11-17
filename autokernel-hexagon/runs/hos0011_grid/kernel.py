"""AutoKernel-Hexagon -- the ONLY file the agent edits (RMSNorm candidate).

Contract (owned by akh/contract.py, do not change here):
    y = x * rsqrt(mean(x^2, -1) + eps) * weight
    x, weight fp16 -> fp32 reduction/intermediates -> y fp16

The evaluator calls ``launches(x, w, y, eps, cfg)`` and executes the returned
Launch list in order. ``cfg`` is CONFIG merged with any search override.
Evaluator-owned Hexagon options (iterations, deviceCleanup, enableLWP) are
rejected if set here.

Strategies
  row             one program handles whole rows (grid-stride over rows).
                  block_h < H loops over H in chunks: two passes over x
                  (sum of squares, then normalize) -> x is re-read.
  split_redundant hidden dim split into SPLIT slices; each (row, slice)
                  program re-reduces the FULL row (redundant x reads, no
                  merge) and normalizes only its slice.
  split_two_pass  launch 1 writes per-slice partial sums of squares,
                  launch 2 merges SPLIT partials and normalizes its slice.
                  On the standalone Hexagon launcher each launch is a separate
                  run_main_on_hexagon process, so partials round-trip through
                  host files (counted in host time; DSP times are summed).
"""

KERNEL_TYPE = "rmsnorm"

HYPOTHESIS = "baseline: simplest row kernel (whole row per block, 1/sqrt, grid=1, backend default options)"

# Shapes the candidate handles correctly; tail = H not multiple of 64 / not pow2.
SUPPORTS_TAIL = True

# --- CONFIG BEGIN (search.py may rewrite this block) ---
CONFIG = {'block_h': 0,
 'hexagon': {'enableThreadedDispatch': True},
 'num_programs': 4,
 'split': 1,
 'strategy': 'row',
 'use_rsqrt': False}
# --- CONFIG END ---

import torch
import triton
import triton.language as tl

from akh.launch import Launch


@triton.jit
def rmsnorm_row_kernel(
    x_ptr, w_ptr, y_ptr,
    M: tl.constexpr, H: tl.constexpr, BLOCK_H: tl.constexpr,
    EPS: tl.constexpr, USE_RSQRT: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for row in range(pid, M, nprog):
        base = row * H
        if BLOCK_H >= H:
            offs = tl.arange(0, BLOCK_H)
            mask = offs < H
            x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
            ms = tl.sum(x * x, axis=0) / H
            if USE_RSQRT:
                inv = tl.math.rsqrt(ms + EPS)
            else:
                inv = 1.0 / tl.sqrt(ms + EPS)
            w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
            tl.store(y_ptr + base + offs, (x * inv * w).to(tl.float16), mask=mask)
        else:
            acc = tl.zeros([BLOCK_H], dtype=tl.float32)
            for start in range(0, H, BLOCK_H):
                offs = start + tl.arange(0, BLOCK_H)
                mask = offs < H
                x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
                acc += x * x
            ms = tl.sum(acc, axis=0) / H
            if USE_RSQRT:
                inv = tl.math.rsqrt(ms + EPS)
            else:
                inv = 1.0 / tl.sqrt(ms + EPS)
            for start in range(0, H, BLOCK_H):
                offs = start + tl.arange(0, BLOCK_H)
                mask = offs < H
                x = tl.load(x_ptr + base + offs, mask=mask, other=0.0).to(tl.float32)
                w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
                tl.store(y_ptr + base + offs, (x * inv * w).to(tl.float16), mask=mask)


@triton.jit
def rmsnorm_split_redundant_kernel(
    x_ptr, w_ptr, y_ptr,
    M: tl.constexpr, H: tl.constexpr, SPLIT: tl.constexpr,
    BLOCK_H: tl.constexpr, SLICE_LEN: tl.constexpr, SLICE_BLOCK: tl.constexpr,
    EPS: tl.constexpr, USE_RSQRT: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for t in range(pid, M * SPLIT, nprog):
        row = t // SPLIT
        part = t % SPLIT
        base = row * H
        offs = tl.arange(0, BLOCK_H)
        x = tl.load(x_ptr + base + offs, mask=offs < H, other=0.0).to(tl.float32)
        ms = tl.sum(x * x, axis=0) / H
        if USE_RSQRT:
            inv = tl.math.rsqrt(ms + EPS)
        else:
            inv = 1.0 / tl.sqrt(ms + EPS)
        local = tl.arange(0, SLICE_BLOCK)
        so = part * SLICE_LEN + local
        smask = (local < SLICE_LEN) & (so < H)
        xs = tl.load(x_ptr + base + so, mask=smask, other=0.0).to(tl.float32)
        ws = tl.load(w_ptr + so, mask=smask, other=0.0).to(tl.float32)
        tl.store(y_ptr + base + so, (xs * inv * ws).to(tl.float16), mask=smask)


@triton.jit
def rmsnorm_partial_kernel(
    x_ptr, p_ptr,
    M: tl.constexpr, H: tl.constexpr, SPLIT: tl.constexpr,
    SLICE_LEN: tl.constexpr, SLICE_BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for t in range(pid, M * SPLIT, nprog):
        row = t // SPLIT
        part = t % SPLIT
        local = tl.arange(0, SLICE_BLOCK)
        so = part * SLICE_LEN + local
        smask = (local < SLICE_LEN) & (so < H)
        xs = tl.load(x_ptr + row * H + so, mask=smask, other=0.0).to(tl.float32)
        tl.store(p_ptr + t, tl.sum(xs * xs, axis=0))


@triton.jit
def rmsnorm_apply_kernel(
    x_ptr, w_ptr, p_ptr, y_ptr,
    M: tl.constexpr, H: tl.constexpr, SPLIT: tl.constexpr,
    SLICE_LEN: tl.constexpr, SLICE_BLOCK: tl.constexpr,
    EPS: tl.constexpr, USE_RSQRT: tl.constexpr,
):
    pid = tl.program_id(0)
    nprog = tl.num_programs(0)
    for t in range(pid, M * SPLIT, nprog):
        row = t // SPLIT
        part = t % SPLIT
        ps = tl.load(p_ptr + row * SPLIT + tl.arange(0, SPLIT))
        ms = tl.sum(ps, axis=0) / H
        if USE_RSQRT:
            inv = tl.math.rsqrt(ms + EPS)
        else:
            inv = 1.0 / tl.sqrt(ms + EPS)
        local = tl.arange(0, SLICE_BLOCK)
        so = part * SLICE_LEN + local
        smask = (local < SLICE_LEN) & (so < H)
        xs = tl.load(x_ptr + row * H + so, mask=smask, other=0.0).to(tl.float32)
        ws = tl.load(w_ptr + so, mask=smask, other=0.0).to(tl.float32)
        tl.store(y_ptr + row * H + so, (xs * inv * ws).to(tl.float16), mask=smask)


def _grid(cfg, work_items):
    n = int(cfg.get("num_programs", 1))
    return (work_items if n <= 0 else max(1, min(n, work_items)),)


def launches(x: torch.Tensor, w: torch.Tensor, y: torch.Tensor, eps: float, cfg: dict):
    M, H = x.shape
    hex_opts = dict(cfg.get("hexagon", {}))
    use_rsqrt = bool(cfg.get("use_rsqrt", False))
    strategy = cfg.get("strategy", "row")
    full = triton.next_power_of_2(H)

    if strategy == "row":
        block_h = int(cfg.get("block_h", 0)) or full
        block_h = min(triton.next_power_of_2(block_h), full)
        return [Launch(
            "rmsnorm_row", rmsnorm_row_kernel, _grid(cfg, M), (x, w, y),
            dict(M=M, H=H, BLOCK_H=block_h, EPS=eps, USE_RSQRT=use_rsqrt),
            hex_opts, outputs=(y,))]

    split = int(cfg.get("split", 1))
    if split < 1 or split & (split - 1):
        raise ValueError(f"split must be a power of two, got {split}")
    slice_len = triton.cdiv(H, split)
    slice_block = triton.next_power_of_2(slice_len)
    work = M * split

    if strategy == "split_redundant":
        return [Launch(
            "rmsnorm_split_redundant", rmsnorm_split_redundant_kernel, _grid(cfg, work), (x, w, y),
            dict(M=M, H=H, SPLIT=split, BLOCK_H=full, SLICE_LEN=slice_len,
                 SLICE_BLOCK=slice_block, EPS=eps, USE_RSQRT=use_rsqrt),
            hex_opts, outputs=(y,))]

    if strategy == "split_two_pass":
        partials = torch.empty(M * split, dtype=torch.float32)
        common = dict(M=M, H=H, SPLIT=split, SLICE_LEN=slice_len, SLICE_BLOCK=slice_block)
        return [
            Launch("rmsnorm_partial", rmsnorm_partial_kernel, _grid(cfg, work), (x, partials),
                   dict(common), hex_opts, outputs=(partials,)),
            Launch("rmsnorm_apply", rmsnorm_apply_kernel, _grid(cfg, work), (x, w, partials, y),
                   dict(common, EPS=eps, USE_RSQRT=use_rsqrt), hex_opts, outputs=(y,)),
        ]

    raise ValueError(f"unknown strategy {strategy!r}")
