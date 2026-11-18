"""BackendRegistry: backend_id -> adapter bundle factory (plan §6.1).

Adding a vendor = adding a module under nak/backends/ that calls ``register``,
plus configuration and conformance tests. No edits to controller/evaluator.
"""
from __future__ import annotations

import importlib
import pkgutil

from . import SCHEMA_VERSION

_FACTORIES: dict = {}
_LOADED = False


def register(backend_id: str, schema_version: str = SCHEMA_VERSION):
    def deco(factory):
        if schema_version != SCHEMA_VERSION:
            raise RuntimeError(f"{backend_id}: schema {schema_version} != {SCHEMA_VERSION}")
        if backend_id in _FACTORIES:
            raise RuntimeError(f"backend '{backend_id}' registered twice")
        _FACTORIES[backend_id] = factory
        return factory
    return deco


def _load_all():
    global _LOADED
    if _LOADED:
        return
    from . import backends

    for m in pkgutil.iter_modules(backends.__path__):
        importlib.import_module(f"{backends.__name__}.{m.name}")
    _LOADED = True


def available() -> list:
    _load_all()
    return sorted(_FACTORIES)


def resolve(config: dict):
    """Build the bundle fixed for one experiment. Rejects incompatible profile choices."""
    _load_all()
    bid = config["backend_id"]
    if bid not in _FACTORIES:
        raise KeyError(f"unknown backend_id '{bid}'; registered: {available()}")
    bundle = _FACTORIES[bid](config)
    m = bundle.manifest
    for key in ("compiler_profile", "runtime_profile"):
        want = config.get(key)
        if want and want != getattr(m, key):
            raise ValueError(f"{bid}: requested {key}={want!r} but bundle provides {getattr(m, key)!r}")
    if m.compatible_runtimes and m.runtime_profile not in m.compatible_runtimes:
        raise ValueError(f"{bid}: runtime {m.runtime_profile} incompatible with compiler {m.compiler_profile}")
    return bundle
