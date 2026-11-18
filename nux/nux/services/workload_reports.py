"""A0 (workload lock + framework reference) and A4 (corpus manifest) reports."""
from __future__ import annotations

import time

from ..corpus import GENERATOR_VERSION
from ..workloads import get
from . import ROOT, write_json


def lock_and_reference(workload_id: str) -> dict:
    import torch

    from ..workloads import deit

    w = get(workload_id)
    t = time.perf_counter()
    lock = deit.write_lock(w) if workload_id == "deit-tiny" else w.lock()
    load_s = time.perf_counter() - t
    ref = {}
    for item in w.corpus():
        y1 = w.reference(item["tensors"])
        y2 = w.reference(item["tensors"])
        top5 = y1[0].topk(5)
        ref[item["id"]] = {"split": item["split"], "input_sha256": item["sha256"], "logits_sha256": deit.tensor_sha256(y1),
                           "deterministic": bool(torch.equal(y1, y2)), "finite": bool(torch.isfinite(y1).all()),
                           "top5": top5.indices.tolist(), "top5_logits": [round(v, 4) for v in top5.values.tolist()],
                           "labels": [w.hf_model().config.id2label[i] for i in top5.indices.tolist()[:3]]}
    path = write_json(ROOT / "workloads" / f"{workload_id}.reference.json",
                      {"workload": workload_id, "reference": "framework eager fp32 (transformers ViTForImageClassification)",
                       "torch": torch.__version__, "inputs": ref,
                       "all_deterministic": all(r["deterministic"] for r in ref.values()),
                       "all_finite": all(r["finite"] for r in ref.values())})
    return {"load_s": load_s, "load_info": lock.get("load_info"), "path": path}


def corpus_manifest(workload_id: str) -> dict:
    w = get(workload_id)
    c1, c2 = w.corpus(), w.corpus()
    same = [x["sha256"] for x in c1] == [x["sha256"] for x in c2]
    items = [{"id": x["id"], "split": x["split"], "sha256": x["sha256"], "shape": list(x["tensors"][0].shape),
              "min": float(x["tensors"][0].min()), "max": float(x["tensors"][0].max()),
              "beyond_processor_domain": bool(x.get("beyond_processor_domain"))} for x in c1]
    splits = {}
    for x in items:
        splits[x["split"]] = splits.get(x["split"], 0) + 1
    path = write_json(ROOT / "workloads" / f"{workload_id}.corpus.json",
                      {"workload": workload_id, "generator": GENERATOR_VERSION, "regenerates_identically": same,
                       "splits": splits, "task_accuracy": "pending: no licensed labeled evaluation set in this environment",
                       "items": items})
    return {"path": path, "splits": splits, "regenerates_identically": same}
