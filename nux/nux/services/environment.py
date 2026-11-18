"""Toolchain / device lock, built from every registered backend's declared manifest (vendor-neutral)."""
from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version

from ..engine import registry
from . import REPORTS, write_json

PACKAGES = ("torch", "torch-mlir", "transformers", "huggingface-hub", "numpy", "pillow", "onnx", "onnxruntime")


def _v(p):
    try:
        return version(p)
    except PackageNotFoundError:
        return None


def _plain(env: dict) -> dict:
    return {k: v for k, v in (env or {}).items() if isinstance(v, (str, bool, int, float, type(None)))}


def toolchain_lock() -> dict:
    lock = {"python": platform.python_version(), "platform": platform.platform(), **{p: _v(p) for p in PACKAGES}}
    backends = {}
    for bid in registry.available():
        try:
            m = registry.resolve({"backend_id": bid}).manifest
            backends[bid] = {"version": m.version, "compiler_profile": m.compiler_profile,
                             "runtime_profile": m.runtime_profile, "device_profile": m.device_profile,
                             "traits": m.traits(), "environment": _plain(m.environment)}
        except Exception as e:  # noqa: BLE001  (a broken backend must not break the lock)
            backends[bid] = {"error": f"{type(e).__name__}: {e}"}
    lock["backends"] = backends
    return lock


def write_env_report():
    return write_json(REPORTS / "env.json", toolchain_lock())


def capabilities(backend_id: str) -> dict:
    return registry.resolve({"backend_id": backend_id}).manifest.to_dict()
