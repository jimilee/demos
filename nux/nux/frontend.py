"""Framework front end: workload -> verified linalg-on-tensors IR (vendor-neutral, shared by all backends).

Backends must obtain the baseline IR here instead of from another backend.
Exports are cached per process by (workload_id, front-end version) because
they are deterministic and expensive for real models (DeiT-tiny: ~4-5 s).
"""
from __future__ import annotations

import time
from importlib.metadata import PackageNotFoundError, version

from . import mlir_host as H
from .adapters import CompileFailure

_CACHE: dict = {}


def frontend_version() -> str:
    try:
        return f"torch-mlir-{version('torch-mlir')}"
    except PackageNotFoundError:
        return "torch-mlir-unknown"


def frontend_fingerprint() -> dict:
    """What determines the exported IR besides the workload itself (whose weights/framework are in its lock)."""
    import hashlib
    from pathlib import Path

    try:
        torch_v = version("torch")
    except PackageNotFoundError:
        torch_v = "unknown"
    src = Path(H.__file__).read_bytes()
    return {"frontend": frontend_version(), "torch": torch_v, "mlir_host_sha": hashlib.sha256(src).hexdigest()[:16]}


def export(workload) -> tuple:
    """Returns (MLIRModule, export_seconds). Raises CompileFailure if the IR does not verify."""
    key = (workload.workload_id, frontend_version())
    if key not in _CACHE:
        t = time.perf_counter()
        try:
            mod = H.export(workload.module(), workload.example_inputs(), workload.func_name)
        except Exception as e:  # noqa: BLE001
            raise CompileFailure(f"front-end export failed: {type(e).__name__}: {e}") from e
        ok, why = H.verify(mod)
        if not ok:
            raise CompileFailure(f"exported IR does not verify: {why}")
        _CACHE[key] = (mod, time.perf_counter() - t)
    return _CACHE[key]


def clear_cache():
    _CACHE.clear()
