"""Artifact store maintenance and leftover AOT work-directory cleanup."""
from __future__ import annotations

import shutil

from ..storage.artifact_store import ArtifactPolicy, ArtifactStore
from ..toolchain import aot
from . import ROOT


def store(max_gb=5.0, root=None) -> ArtifactStore:
    return ArtifactStore(root or ROOT / ".cache" / "artifacts", ArtifactPolicy(max_bytes=int(max_gb * 10**9)))


def clean_workdirs() -> dict:
    base = aot.workdir_base()
    dirs = [d for d in base.glob("nux_*") if d.is_dir()] if base.exists() else []
    size = sum(f.stat().st_size for d in dirs for f in d.rglob("*") if f.is_file())
    for d in dirs:
        shutil.rmtree(d, ignore_errors=True)
    return {"removed": len(dirs), "bytes": size, "base": str(base)}
