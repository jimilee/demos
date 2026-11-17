"""CPU PyTorch reference implementations (accuracy baseline)."""
import torch

from . import contract


def rmsnorm_ref(x: torch.Tensor, w: torch.Tensor, eps: float = contract.EPS) -> torch.Tensor:
    """fp16 in, fp32 reduction and intermediates, fp16 out."""
    xf = x.float()
    inv = torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (xf * inv * w.float()).to(contract.OUT_DTYPE)


def linear_ref(x: torch.Tensor, w_kn: torch.Tensor, out_dtype=torch.float32) -> torch.Tensor:
    """y[M,N] = x[M,K] @ w[K,N], fp16 inputs, fp32 accumulation."""
    return (x.float() @ w_kn.float()).to(out_dtype)
