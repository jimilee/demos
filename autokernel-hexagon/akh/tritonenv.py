"""Triton install sanity for the Hexagon backend.

The Triton pinned by hexagon-mlir (a9ced83, version 3.7.0 + Qualcomm patches,
installed with `pip install -e .`) discovers backends through package entry
points (group "triton.backends") unless TRITON_BACKENDS_IN_TREE=1.
If another Triton distribution is also installed (typically a PyPI wheel such
as triton-3.7.1 pulled in by a CUDA build of torch), its dist-info advertises
backends ("amd", "nvidia") that do not exist in the imported source tree, and
`import triton` fails with ``No module named 'triton.backends.amd'``.

``prefer_in_tree_backends()`` makes discovery scan only the directories under
the imported ``triton/backends/`` (which include the qcom_hexagon_backend
symlink created by the hexagon-mlir build), so stale dist-info entry points
are ignored. Set TRITON_BACKENDS_IN_TREE=0 explicitly to opt out.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


def prefer_in_tree_backends() -> bool:
    """Must run before the first `import triton`. Returns True if it is in effect."""
    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")
    if "triton" in sys.modules and "triton.backends" in sys.modules:
        # discovery already ran; the setting applies only to new processes
        return False
    return os.environ["TRITON_BACKENDS_IN_TREE"] == "1"


def diagnose() -> dict:
    """Inspect installed Triton distributions and backend entry points without importing triton."""
    from importlib.metadata import distributions, entry_points

    spec = importlib.util.find_spec("triton")
    tdir = Path(spec.origin).parent if spec and spec.origin else None
    dists = []
    for d in distributions():
        name = (d.metadata.get("Name") or "").lower()
        if name in ("triton", "pytorch-triton", "triton-nightly"):
            try:
                loc = str(Path(d.locate_file("")).resolve())
            except Exception:  # noqa: BLE001
                loc = "?"
            direct = d.read_text("direct_url.json") or ""
            dists.append({"name": name, "version": d.version, "location": loc,
                          "editable": '"editable": true' in direct.replace(" ", "").replace('":true', '": true')})
    eps = []
    for ep in entry_points().select(group="triton.backends"):
        mod = ep.value
        sub = mod.split(".")[-1]
        resolvable = bool(tdir and mod.startswith("triton.backends.") and (tdir / "backends" / sub).is_dir())
        eps.append({"name": ep.name, "module": mod, "resolvable_in_imported_triton": resolvable})
    in_tree = sorted(p.name for p in (tdir / "backends").iterdir()
                     if p.is_dir() and not p.name.startswith("__")) if tdir and (tdir / "backends").is_dir() else []
    issues = []
    if len(dists) > 1:
        issues.append("multiple Triton distributions installed: "
                      + ", ".join(f"{d['name']}=={d['version']} ({'editable' if d['editable'] else d['location']})"
                                  for d in dists))
    broken = [e["name"] for e in eps if not e["resolvable_in_imported_triton"]]
    if broken:
        issues.append(f"entry points not present in the imported triton ({tdir}): {broken}")
    if "qcom_hexagon_backend" not in in_tree:
        issues.append(f"qcom_hexagon_backend not found under {tdir}/backends (triton not built with the hexagon-mlir plugin)")
    return {"imported_triton_dir": str(tdir) if tdir else None, "distributions": dists,
            "entry_points": eps, "in_tree_backends": in_tree,
            "TRITON_BACKENDS_IN_TREE": os.environ.get("TRITON_BACKENDS_IN_TREE"), "issues": issues}
