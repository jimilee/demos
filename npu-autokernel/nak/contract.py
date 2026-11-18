"""Correctness contract (plan §10, §14 numeric draft). Evaluator-owned; agents cannot edit.

The values are the plan's INITIAL proposal for fp32 logits smoke checks, not
calibrated tolerances. They are versioned; calibration on the device baseline
produces a new contract id instead of silently changing this one.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class Contract:
    contract_id: str
    atol: float
    rtol: float
    out_dtype: str
    require_finite: bool = True
    integer_exact: bool = True
    task_accuracy: str = "pending"  # no labeled set yet; top-1 drop <= 0.1pp proposed (plan §14)
    note: str = ""


CONTRACTS = {
    "deit-tiny-fp32-logits-v0": Contract(
        "deit-tiny-fp32-logits-v0", atol=1e-4, rtol=1e-3, out_dtype="float32",
        note="plan v0.3 §14 smoke proposal; calibrate on device baseline before acceptance runs"),
    "toy-fp32-v0": Contract("toy-fp32-v0", atol=1e-5, rtol=1e-4, out_dtype="float32"),
}


def get(contract_id: str) -> Contract:
    return CONTRACTS[contract_id]


def compare(contract: Contract, out: torch.Tensor, ref: torch.Tensor) -> dict:
    res = {"contract_id": contract.contract_id, "ok": False}
    if out is None:
        return {**res, "reason": "no output"}
    out = torch.as_tensor(out)
    if tuple(out.shape) != tuple(ref.shape):
        return {**res, "reason": f"shape {tuple(out.shape)} != {tuple(ref.shape)}"}
    if str(out.dtype).replace("torch.", "") != contract.out_dtype:
        return {**res, "reason": f"dtype {out.dtype} != {contract.out_dtype}"}
    o, r = out.double(), ref.double()
    nonfinite = int((~torch.isfinite(o)).sum())
    diff = (o - r).abs()
    bound = contract.atol + contract.rtol * r.abs()
    bad = (diff > bound) | ~torch.isfinite(o)
    res.update(
        max_abs_err=float(diff[torch.isfinite(diff)].max()) if torch.isfinite(diff).any() else float("inf"),
        mean_abs_err=float(diff[torch.isfinite(diff)].mean()) if torch.isfinite(diff).any() else float("inf"),
        max_rel_err=float((diff / r.abs().clamp_min(1e-12)).max()),
        n_violations=int(bad.sum()), nonfinite=nonfinite,
        top1_match=bool(o.reshape(o.shape[0], -1).argmax(-1).eq(r.reshape(r.shape[0], -1).argmax(-1)).all()),
    )
    if o.ndim == 2 and o.shape[-1] >= 5:
        res["top5_overlap"] = float(len(set(o[0].topk(5).indices.tolist()) & set(r[0].topk(5).indices.tolist())) / 5)
    res["ok"] = res["n_violations"] == 0 and not (contract.require_finite and nonfinite)
    if not res["ok"]:
        res["reason"] = f"{res['n_violations']} elements outside atol={contract.atol} rtol={contract.rtol}; nonfinite={nonfinite}"
    return res


def summarize(per_input: list) -> dict:
    ok = all(r["ok"] for r in per_input)
    finite = [r.get("max_abs_err") for r in per_input if r.get("max_abs_err") is not None]
    return {"ok": ok, "n_inputs": len(per_input), "n_failed": sum(not r["ok"] for r in per_input),
            "max_abs_err": max(finite) if finite else None,
            "failed_inputs": [(r.get("input_id"), r.get("reason")) for r in per_input if not r["ok"]][:10],
            "task_accuracy": "pending"}


def as_dict(c: Contract) -> dict:
    return asdict(c)
