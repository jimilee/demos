"""HMX extension: small Linear y[M,N] = x[M,K] @ w[K,N] (fp16 in, fp32 out).

Same Triton source, two lowering paths selected by Hexagon options:
  hvx          default linalg -> HVX tiling/vectorization
  hexkl_micro  enableHexKL=True, hexKLMode="micro" -> hexkl_micro_* calls (HMX)

Path facts from hexagon-mlir rev 9ec15e8 (lib/Transforms/MatmulToHexKLPass.cpp,
LowerHexKLMatmulToMacroPass.cpp):
  * hexkl.matmul matches only 2-D static-shape linalg.matmul with f16 inputs and
    f16/f32 output; f32 output is produced as f16 then upcast.
  * "macro" mode requires CONSTANT weights preprocessed to the WH layout at
    compile time (init-time prepack). Triton kernel weights are runtime
    pointers, so from Triton only "micro" applies, and activation/output layout
    transforms happen on every call inside the timed function.
  * HexKL is not supported on v81 (user guide).
"""
KERNEL_TYPE = "linear"

import triton
import triton.language as tl

from akh.launch import Launch

PATHS = {
    "hvx": {"enableHexKL": False},
    "hexkl_micro": {"enableHexKL": True, "hexKLMode": "micro"},
}

# --- CONFIG BEGIN (search.py may rewrite this block) ---
CONFIG = {'hexagon': {}, 'path': 'hvx'}
# --- CONFIG END ---


@triton.jit
def linear_kernel(A, B, C, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    a_ptr = tl.make_block_ptr(base=A, shape=(M, K), strides=(K, 1), offsets=(0, 0),
                              block_shape=(M, K), order=(1, 0))
    b_ptr = tl.make_block_ptr(base=B, shape=(K, N), strides=(N, 1), offsets=(0, 0),
                              block_shape=(K, N), order=(1, 0))
    c_ptr = tl.make_block_ptr(base=C, shape=(M, N), strides=(N, 1), offsets=(0, 0),
                              block_shape=(M, N), order=(1, 0))
    a = tl.load(a_ptr)
    b = tl.load(b_ptr)
    acc = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(c_ptr, acc.to(C.type.element_ty))


@triton.jit
def linear_kernel_offsets(A, B, C, M: tl.constexpr, K: tl.constexpr, N: tl.constexpr):
    """Same op with plain pointer arithmetic. Upstream Triton >=3.6 (host
    interpreter) removed make_block_ptr; the hexagon-pinned Triton (a9ced83)
    still has it, and the official examples use it, so 'block' is the device
    default and 'offsets' is the portable variant."""
    rm = tl.arange(0, M)
    rk = tl.arange(0, K)
    rn = tl.arange(0, N)
    a = tl.load(A + rm[:, None] * K + rk[None, :])
    b = tl.load(B + rk[:, None] * N + rn[None, :])
    acc = tl.dot(a, b, out_dtype=tl.float32)
    tl.store(C + rm[:, None] * N + rn[None, :], acc.to(C.type.element_ty))


def launches(x, w, y, cfg):
    M, K = x.shape
    _, N = w.shape
    opts = dict(PATHS[cfg.get("path", "hvx")])
    opts.update(cfg.get("hexagon", {}))
    fn = linear_kernel if cfg.get("ptr_style", "block") == "block" else linear_kernel_offsets
    return [Launch(f"linear_{cfg.get('path', 'hvx')}", fn, (1,), (x, w, y),
                   dict(M=M, K=K, N=N), opts, outputs=(y,))]
