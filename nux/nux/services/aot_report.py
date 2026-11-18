"""AOT report built by calling the real AOT backends (no separate build path).

  llvm-aot-host     MLIR's own LLVM -> x86-64 .so, executed via ctypes
  llvm-aot-hexagon  downgraded LLVM IR -> llc twin (x86-64, executed) + llc Hexagon object (not executable here)
"""
from __future__ import annotations

import time
from pathlib import Path

import torch

from .. import aot, frontend, registry
from .. import contract as C
from ..workloads import get
from . import REPORTS, rel, write_json


def _check(lib, w, contract, sel):
    per = []
    for x in sel:
        t = time.perf_counter()
        y = torch.from_numpy(lib(*[t_.numpy() for t_ in x["tensors"]]))
        r = C.compare(contract, y, w.reference(x["tensors"]))
        r.update(input_id=x["id"], run_s=round(time.perf_counter() - t, 2))
        per.append({k: r.get(k) for k in ("input_id", "ok", "max_abs_err", "top1_match", "run_s")})
    return {"ok": all(p["ok"] for p in per), "max_abs_err": max(p["max_abs_err"] for p in per), "per_input": per}


def _files(art):
    return {n: {"bytes": Path(p).stat().st_size, "file": aot._file_type(p), "role": art.roles.get(n),
                "sha256": art.artifact_hashes.get(n)} for n, p in art.files.items() if Path(p).exists()}


def aot_report(workload_id, contract_id, n_inputs=2, n_stress=1, keep_workdir=False, log=print) -> dict:
    w = get(workload_id)
    contract = C.get(contract_id)
    corpus = w.corpus()
    sel = [x for x in corpus if x["split"] == "tune"][:n_inputs] + [x for x in corpus if x["split"] == "stress"][:n_stress]
    mod, _ = frontend.export(w)
    rep = {"workload": workload_id, "ir_sha256": mod.sha256, "llc": aot.llc_path(), "llc_version": aot.llc_version(),
           "llc_has_hexagon": "hexagon" in aot.llc_targets(), "routes": {}}
    to_clean = []

    host = registry.resolve({"backend_id": "llvm-aot-host"})
    t = time.perf_counter()
    art = host.compiler.integrate_candidate(w, host.compiler.capture_baseline(w))
    to_clean += list(art.files.values())
    rep["routes"]["llvm-aot-host"] = {"build_s": round(time.perf_counter() - t, 1), "files": _files(art),
                                      "correctness": _check(art.payload, w, contract, sel)}
    log(f"[llvm-aot-host] ok={rep['routes']['llvm-aot-host']['correctness']['ok']}")

    hexb = registry.resolve({"backend_id": "llvm-aot-hexagon"})
    t = time.perf_counter()
    art = hexb.compiler.capture_baseline(w)
    entry = {"build_s": round(time.perf_counter() - t, 1), "target": hexb.manifest.compiler_profile}
    if art.pending:
        entry["pending"] = art.pending
    else:
        to_clean += list(art.files.values())
        entry.update(files=_files(art), provenance=art.provenance[-3:], llvm_ir_sha256=art.ir_hashes.get("llvm-ir-downgraded"),
                     symbols=aot.symbols(art.files["hexagon_object"], mod.func_name),
                     twin_correctness=_check(art.twin, w, contract, sel),
                     twin_note="x86-64 host build of the same downgraded IR as the Hexagon object (verification only)",
                     execution="pending: needs Hexagon SDK runtime (FastRPC skel / run_main_on_hexagon) and a device")
        log(f"[llvm-aot-hexagon] {entry['files']['hexagon_object']['file']} twin ok={entry['twin_correctness']['ok']}")
    rep["routes"]["llvm-aot-hexagon"] = entry
    path = write_json(REPORTS / f"aot_{workload_id}.json", rep)
    if keep_workdir:
        rep["workdirs"] = sorted({rel(Path(p).parent) for p in to_clean})
    else:
        aot.cleanup_workdirs(to_clean)
    return {"path": path, "report": rep}
