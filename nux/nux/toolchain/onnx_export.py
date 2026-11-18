"""Framework -> ONNX export: the vendor-neutral input artifact for vendor SDKs (QAIRT, AI Hub, ...).

Validated on the host with onnxruntime (CPU EP) against the framework reference
before it is handed to any vendor compiler, so vendor-side differences are not
confused with export errors.
"""
from __future__ import annotations

import collections
import hashlib
import time
from pathlib import Path

import numpy as np
import torch


def export(workload, out: Path, opset=18) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    m = workload.module().eval()
    x = workload.example_inputs()
    t = time.perf_counter()
    torch.onnx.export(m, tuple(x), str(out), input_names=["pixel_values" if workload.workload_id.startswith("deit") else "x"],
                      output_names=["logits"], opset_version=opset, dynamo=True, external_data=False)
    secs = time.perf_counter() - t
    import onnx

    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    ops = collections.Counter(n.op_type for n in model.graph.node)
    return {"path": str(out), "sha256": hashlib.sha256(out.read_bytes()).hexdigest(), "bytes": out.stat().st_size,
            "opset": opset, "export_s": round(secs, 1), "n_nodes": len(model.graph.node), "op_types": dict(ops.most_common()),
            "inputs": [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in model.graph.input],
            "torch": torch.__version__, "onnx": onnx.__version__}


def validate(workload, onnx_path: Path, contract, corpus, splits=("tune", "stress"), limit=None) -> dict:
    import onnxruntime as ort

    from ..core import contract as C

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    per = []
    for item in corpus:
        if item["split"] not in splits:
            continue
        y = sess.run(None, {name: item["tensors"][0].numpy()})[0]
        r = C.compare(contract, torch.from_numpy(np.asarray(y)), workload.reference(item["tensors"]))
        per.append({"input_id": item["id"], **{k: r.get(k) for k in ("ok", "max_abs_err", "top1_match", "top5_overlap")}})
        if limit and len(per) >= limit:
            break
    return {"ok": all(p["ok"] for p in per), "max_abs_err": max(p["max_abs_err"] for p in per), "n_inputs": len(per),
            "runtime": f"onnxruntime {ort.__version__} CPUExecutionProvider", "per_input": per}
