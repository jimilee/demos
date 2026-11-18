"""A2: compile smoke per backend (minimal op / one block / full model)."""
from __future__ import annotations

import time

from ..engine import registry
from ..toolchain import aot
from ..toolchain import mlir_host as H
from ..workloads import get
from . import REPORTS, write_json
from .environment import toolchain_lock


class _SubWorkload:
    """Minimal workload view of a sub-module for the compiler adapter."""

    def __init__(self, name, module, inputs):
        self.workload_id = self.func_name = name
        self._m, self._x = module, inputs

    def module(self):
        return self._m

    def example_inputs(self):
        return self._x


def compile_smoke(workload_id: str, backend_id: str, log=print) -> dict:
    w = get(workload_id)
    b = registry.resolve({"backend_id": backend_id})
    rows = []
    for name, mod_, inputs in w.submodules():
        r = {"level": name, "backend": backend_id}
        t = time.perf_counter()
        try:
            m = H.export(mod_, inputs, name)
            r["export"] = {"ok": True, "s": round(time.perf_counter() - t, 2), "verify": H.verify(m), "sha256": m.sha256,
                           "max_reshape_rank": H.ir_stats(m)["max_reshape_rank"]}
        except Exception as e:  # noqa: BLE001
            r["export"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            rows.append(r)
            continue
        try:
            art = b.compiler.capture_baseline(_SubWorkload(name, mod_, inputs))
            r["target_compile"] = ("pending: " + art.pending) if art.pending else (
                "passed" if art.target_compiled else "not a target backend")
            r["artifact_hashes"] = art.artifact_hashes
            aot.cleanup_workdirs(art.files.values())  # report-only build: do not leave .ll/.o behind
        except Exception as e:  # noqa: BLE001
            r["target_compile"] = f"failed: {type(e).__name__}: {e}"
        rows.append(r)
        log(f"[{name}] export={r['export'].get('ok')} target_compile={r['target_compile'][:90]}")
    path = write_json(REPORTS / f"a2_smoke_{backend_id}.json",
                      {"workload": workload_id, "backend": backend_id, "lock": toolchain_lock(), "levels": rows})
    return {"path": path, "levels": rows}
