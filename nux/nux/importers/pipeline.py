"""Vendor profile -> normalized profile -> hints -> capability-validated seed configs per target backend."""
from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

from . import hints as Hn
from . import load
from .onnx_map import name_to_type, source_graph

ROOT = Path(__file__).resolve().parents[2]
BASE_CONFIGS = {"hexagon-mlir": "deit_hexagon.json", "llvm-aot-hexagon": "deit_aot_hexagon.json",
                "host-linalg-jit": "deit_host_prepare.json", "llvm-aot-host": "deit_aot_host.json"}


def _rel(p: Path) -> str:
    try:
        return str(p.resolve().relative_to(ROOT))
    except ValueError:
        return str(p)


def build_config_hints(build_config: dict | None, converter_args: str | None, label: str):
    out = []
    caveat = ["vendor build setting; the target backend has its own knobs"]
    if build_config:
        g = (build_config.get("graphs") or [{}])[0]
        d = (build_config.get("devices") or [{}])[0]
        if "vtcm_mb" in g:
            out.append(Hn.Hint("build-vtcm", "memory", f"vendor reference built with vtcm_mb={g['vtcm_mb']}",
                               {"vtcm_mb": g["vtcm_mb"], "dsp_arch": d.get("dsp_arch")}, source=label,
                               confidence="medium", caveats=caveat, tags=["memory", "tiling"]))
        if "O" in g:
            out.append(Hn.Hint("build-opt-level", "memory", f"vendor reference built with HTP optimization level O={g['O']}",
                               {"O": g["O"]}, source=label, confidence="low", caveats=caveat, tags=["codegen"]))
    if converter_args is not None and "preserve_io" not in converter_args and "preserve-io" not in converter_args:
        out.append(Hn.Hint("build-native-io-layout", "layout",
                           "vendor reference converted without --preserve_io (native IO layout/dtype at graph edges)",
                           {"converter_args": converter_args}, source=label, confidence="medium",
                           caveats=caveat + ["changes the model I/O contract; E2E must include any host-side conversion"]))
    return out


def run(fmt, profile_path, label, targets, onnx_path=None, build_config_path=None, converter_args=None,
        precision="unknown", workload_meta=None, out_root=None):
    out_dir = Path(out_root or ROOT / "reports" / "vendor_ref") / label
    out_dir.mkdir(parents=True, exist_ok=True)
    prof = load(fmt, profile_path, source=label)
    src = source_graph(onnx_path) if onnx_path else {"op_types": {}, "nodes": []}
    mapping, how = name_to_type([l["name"] for l in prof.layers], src["nodes"]) if src["nodes"] else ({}, {})
    for l in prof.layers:
        if l["name"] in mapping:
            l["type"] = mapping[l["name"]]
            l["source_type_matched"] = True
    hints = Hn.hints_from_layer_table(prof.layers, prof.summary, src["op_types"], source=label)
    bc = json.loads(Path(build_config_path).read_text()) if build_config_path else None
    hints += build_config_hints(bc, converter_args, label)
    for h in hints:
        h.caveats.append(f"vendor precision: {precision}; target dtype contract is enforced separately (fp32 DeiT stays fp32)")
    meta = {"label": label, "format": fmt, "profile_file": str(profile_path), "onnx": str(onnx_path) if onnx_path else None,
            "precision": precision, "name_mapping": how, "imported_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "warnings": prof.warnings}
    (out_dir / "profile_normalized.json").write_text(json.dumps({"meta": meta, **prof.to_dict()}, indent=2, default=str))
    Hn.save(hints, out_dir / "hints.json", meta)
    seeds_out = {}
    from ..engine import registry

    for t in targets:
        m = registry.resolve({"backend_id": t}).manifest
        seeds, skipped = Hn.seed_candidates(hints, m, workload_meta)
        base_name = BASE_CONFIGS.get(t)
        base = json.loads((ROOT / "configs" / base_name).read_text()) if base_name else {"backend_id": t}
        cfg = {**base, "candidates": [{k: v for k, v in asdict(c).items() if k in
                                       ("action_id", "kind", "params", "hypothesis", "expected_savings", "possible_costs", "scope")}
                                      for c in seeds],
               "_seeded_from": {"label": label, "hints": _rel(out_dir / "hints.json")},
               "_skipped": skipped}
        p = out_dir / f"seeds_{t}.json"
        p.write_text(json.dumps(cfg, indent=2))
        seeds_out[t] = {"config": _rel(p), "n_seeds": len(seeds), "n_skipped": len(skipped)}
    summary = {"meta": meta, "n_layers": len(prof.layers), "summary": prof.summary,
               "hints": [(h.hint_id, h.category, h.observation) for h in hints], "seeds": seeds_out}
    (out_dir / "import_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
