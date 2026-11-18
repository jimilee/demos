"""Vendor-reference services: ONNX input artifact and profile import."""
from __future__ import annotations

from pathlib import Path

from .. import contract as C
from .. import onnx_export as O
from ..importers import pipeline
from ..workloads import get
from . import REPORTS, ROOT, rel, write_json


def export_onnx(workload_id: str, contract_id: str) -> dict:
    w = get(workload_id)
    out = ROOT / ".cache" / "vendor" / f"{workload_id}.onnx"
    info = O.export(w, out)
    info["validation"] = O.validate(w, out, C.get(contract_id), w.corpus())
    info["path"] = rel(out) + " (not committed)"
    return {"path": write_json(REPORTS / f"vendor_input_{workload_id}.json", info), "info": info}


def import_profile(fmt, file, label, targets, workload_id="deit-tiny", onnx=None, build_config=None,
                   converter_args=None, precision="unknown") -> dict:
    meta = get(workload_id).meta() if workload_id else None
    onnx_path = Path(onnx) if onnx else ROOT / ".cache" / "vendor" / f"{workload_id}.onnx"
    return pipeline.run(fmt, Path(file), label, list(targets), onnx_path if onnx_path.exists() else None,
                        build_config, converter_args, precision, meta)
