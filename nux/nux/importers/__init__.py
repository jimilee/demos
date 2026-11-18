"""Vendor profile importers (separate from the common layers).

Each importer reads files produced by a vendor's OWN documented tools (for
QNN/QAIRT: `qnn-profile-viewer` readers, as documented in Qualcomm's
open-source QAI AppBuilder) and returns a normalized `VendorProfile`.
Nothing here opens or disassembles vendor binaries.

Normalized layer row: {"name", "type", "compute_unit", "time_us", "cycles", "n_samples"}
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

_IMPORTERS = {}


def register(fmt: str, description: str):
    def deco(fn):
        _IMPORTERS[fmt] = (fn, description)
        return fn
    return deco


def formats() -> dict:
    from . import chrometrace, qnn_json  # noqa: F401  (registration)

    return {k: v[1] for k, v in _IMPORTERS.items()}


def load(fmt: str, path, **kw) -> "VendorProfile":
    formats()
    if fmt not in _IMPORTERS:
        raise KeyError(f"unknown profile format '{fmt}'; known: {sorted(_IMPORTERS)}")
    return _IMPORTERS[fmt][0](path, **kw)


@dataclass
class VendorProfile:
    source: str  # free-form label of where the profile came from (tool, SDK version, device)
    format: str
    layers: list  # normalized rows
    summary: dict = field(default_factory=dict)
    runs: int = 0
    time_basis: str = ""  # how time_us was obtained (measured, scaled from cycles, ...)
    found_keys: dict = field(default_factory=dict)  # which keys/columns were actually present (schema audit)
    warnings: list = field(default_factory=list)

    def to_dict(self):
        return asdict(self)
