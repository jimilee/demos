"""A1: linalg export, verifier, recipe survey, host JIT execution vs framework, reshape reproducers."""
from __future__ import annotations

import time

from .. import contract as C
from .. import mlir_host as H
from ..workloads import get
from . import REPORTS, ROOT, write_json


def a1_report(workload_id, contract_id, host_inputs=2, host_stress=1, jit_recipes=("fold-unit-extent-dims",), log=print):
    import torch

    w = get(workload_id)
    out = {"workload": workload_id, "stage": "linalg-on-tensors"}
    t = time.perf_counter()
    mod = H.export(w.module(), w.example_inputs(), w.func_name)
    out["export_s"] = round(time.perf_counter() - t, 2)
    out["verify"] = H.verify(mod)
    out["stats"] = H.ir_stats(mod)
    cache = ROOT / ".cache" / "ir"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{workload_id}.linalg.mlirbc").write_bytes(mod.bytecode)
    out["ir_file"] = f".cache/ir/{workload_id}.linalg.mlirbc (not committed; sha256 {mod.sha256})"
    repro_dir = REPORTS / "reproducers"
    repro_dir.mkdir(parents=True, exist_ok=True)
    out["rank5_reshapes_in_export"] = len(H.extract_reshape_reproducers(mod, min_rank=5))
    contract = C.get(contract_id)
    corpus = w.corpus()
    sel = [x for x in corpus if x["split"] == "tune"][:host_inputs] + [x for x in corpus if x["split"] == "stress"][:host_stress]
    out["recipes"] = {}
    variants = {"baseline": mod}
    for rid, r in H.RECIPES.items():
        ok, why = H.recipe_applicable(r, mod)
        entry = {"applicable": ok, "reason": why}
        if ok:
            try:
                m2 = H.apply_recipe(mod, r)
                entry.update(changed=m2.sha256 != mod.sha256, verify=H.verify(m2),
                             stats={k: v for k, v in H.ir_stats(m2).items() if k != "ops"})
                r5 = H.extract_reshape_reproducers(m2, min_rank=5)
                entry["rank5_reshapes"] = len(r5)
                for i, x in enumerate(r5[:10]):
                    (repro_dir / f"{workload_id}__{rid}__rank{x['max_rank']}_{i}.mlir").write_text(x["mlir"])
                if entry["changed"] and rid in jit_recipes:
                    variants[rid] = m2
            except Exception as e:  # noqa: BLE001
                entry["error"] = f"{type(e).__name__}: {e}"
        out["recipes"][rid] = entry
    unit = H.extract_reshape_reproducers(mod, min_rank=99, include_unit_dims=True, limit=8)
    for i, x in enumerate(unit):
        (repro_dir / f"{workload_id}__export__unitdim_{i}.mlir").write_text(x["mlir"])
    out["unit_dim_reproducers_written"] = len(unit)
    out["host_execution"] = {}
    for name, m in variants.items():
        t = time.perf_counter()
        jit = H.HostJIT(m)
        per = []
        for x in sel:
            ts = time.perf_counter()
            y = jit(*[t_.numpy() for t_ in x["tensors"]])
            r = C.compare(contract, torch.from_numpy(y), w.reference(x["tensors"]))
            r.update(input_id=x["id"], run_s=round(time.perf_counter() - ts, 2))
            per.append(r)
        out["host_execution"][name] = {
            "lower_s": round(jit.lower_s, 2), "jit_s": round(jit.jit_s, 2), "summary": C.summarize(per),
            "per_input": [{k: r.get(k) for k in ("input_id", "ok", "max_abs_err", "mean_abs_err", "top1_match",
                                                   "top5_overlap", "run_s")} for r in per],
            "total_s": round(time.perf_counter() - t, 1),
            "note": "host CPU loops lowering for correctness only; timings are not performance data"}
        s = out["host_execution"][name]["summary"]
        log(f"[host-jit {name}] ok={s['ok']} max_abs_err={s['max_abs_err']}")
    return {"path": write_json(REPORTS / f"a1_{workload_id}_ir.json", out), "report": out}
