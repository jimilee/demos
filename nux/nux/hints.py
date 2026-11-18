"""Vendor-neutral optimization hints and seed-candidate generation.

A vendor reference (e.g. a QAIRT-compiled AOT artifact profiled on a device)
is turned into `Hint`s by that vendor's backend module. Hints are OBSERVATIONS
with evidence, not facts about another compiler: every seed candidate derived
from them starts `proposed` / `unvalidated` and must climb the evidence ladder
on its own target. Vendor numbers are never used as the baseline of a
different dtype/runtime profile.

categories -> action tags (the shared vocabulary in capability.TAGS)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

from .capability import validate_action
from .schema import Candidate

CATEGORY_TAGS = {
    "hotspot": ["kernel", "tiling", "threading", "library_matmul", "vector"],
    "fusion": ["fusion"],
    "layout": ["layout", "reshape"],
    "placement": ["partition"],
    "precision": ["precision"],
    "memory": ["memory", "tiling"],
}


@dataclass
class Hint:
    hint_id: str
    category: str  # hotspot | fusion | layout | placement | precision | memory
    observation: str
    evidence: dict = field(default_factory=dict)  # raw numbers / op lists from the vendor output
    op_types: list = field(default_factory=list)  # source-graph op types concerned (for scoping)
    share: float | None = None  # fraction of vendor inference time, when known
    source: str = ""  # vendor reference id (e.g. qai-hub job id)
    confidence: str = "low"  # low | medium | high (how directly the evidence shows it)
    caveats: list = field(default_factory=list)
    tags: list = field(default_factory=list)  # optional override of CATEGORY_TAGS[category]

    def __post_init__(self):
        assert self.category in CATEGORY_TAGS, self.category


def save(hints, path, meta=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"meta": meta or {}, "hints": [asdict(h) for h in hints]}, indent=2, default=str))


def load(path):
    d = json.loads(path.read_text())
    return d.get("meta", {}), [Hint(**h) for h in d["hints"]]


def seed_candidates(hints, manifest, workload_meta=None, max_per_hint=3):
    """Map hints to capability-validated actions of ANY backend by tag. Returns (candidates, skipped)."""
    seeds, skipped, seen = [], [], set()
    ordered = sorted(hints, key=lambda h: -(h.share or 0))
    for h in ordered:
        tags = h.tags or CATEGORY_TAGS[h.category]
        acts = [a for a in manifest.usable_actions() if set(a.tags) & set(tags)]
        if not acts:
            skipped.append({"hint": h.hint_id, "reason": f"{manifest.backend_id} exposes no action tagged {tags}"})
            continue
        n = 0
        for a in acts:
            params = {}
            if a.params:
                p = a.params[0]
                if p.type == "bool":
                    params = {p.name: True}
                elif p.choices:
                    params = {p.name: p.choices[0]}
                else:
                    skipped.append({"hint": h.hint_id, "action": a.action_id,
                                    "reason": f"parameter '{p.name}' ({p.type}) needs a value chosen by the agent"})
                    continue
            c = Candidate(a.action_id, a.kind, params,
                          hypothesis=f"[{h.hint_id}] vendor reference shows: {h.observation}",
                          expected_savings=f"{h.category}" + (f" ({h.share:.0%} of vendor inference time)" if h.share else ""),
                          possible_costs="hint from a different compiler/runtime/precision; may not transfer")
            key = (c.action_id, json.dumps(c.params, sort_keys=True))
            ok, why = validate_action(manifest, c, workload_meta)
            if not ok:
                skipped.append({"hint": h.hint_id, "action": a.action_id, "reason": why})
                continue
            if key in seen:
                continue
            seen.add(key)
            seeds.append(c)
            n += 1
            if n >= max_per_hint:
                break
    return seeds, skipped


# ---------------------------------------------------------------------------------------------
# Vendor-neutral analysis of a NORMALIZED per-layer table
#   layers: [{"name", "type", "compute_unit", "time_us", "cycles"}]  (vendor modules normalize)
# Op-type names differ between frameworks/SDKs; they are mapped to coarse classes.
OP_CLASS = [  # order matters: first match wins ("convert" before "conv", "softmax" before "max")
    ("convert", ("convert", "cast", "quantize", "dequantize")),
    ("matmul", ("matmul", "fullyconnected", "fully_connected", "gemm", "batchmatmul", "conv", "linear")),
    ("norm", ("layernorm", "layer_norm", "layernormalization", "rmsnorm", "groupnorm", "batchnorm")),
    ("softmax", ("softmax",)),
    ("gelu", ("gelu", "erf")),
    ("layout", ("transpose", "permute", "reshape", "squeeze", "unsqueeze", "concat", "split", "gather", "slice",
                "strided", "pack", "unpack", "flatten", "expand", "tile")),
    ("elementwise", ("add", "sub", "mul", "div", "elementwise", "where", "isnan", "pow", "sqrt", "neg", "exp",
                     "relu", "sigmoid", "tanh", "max", "min")),
]
CLASS_TAGS = {"matmul": ["library_matmul", "tiling", "threading"], "norm": ["fusion", "kernel", "reduction"],
              "softmax": ["fusion", "kernel", "reduction"], "gelu": ["fusion", "kernel", "vector"],
              "layout": ["layout", "reshape"], "elementwise": ["fusion", "vector"], "convert": ["precision"],
              "other": ["kernel"]}


def op_class(t: str) -> str:
    k = (t or "").lower().replace("-", "").replace(" ", "")
    for cls, keys in OP_CLASS:
        if any(x in k for x in keys):
            return cls
    return "other"


def hints_from_layer_table(layers, summary, source_op_types: dict, source: str, npu_units=("NPU", "HTP", "DSP")):
    from collections import Counter, defaultdict

    hints = []
    times = [l.get("time_us") for l in layers if l.get("time_us") is not None]
    total = sum(times) if times else None
    by_cls_t, by_cls_n = defaultdict(float), Counter()
    for l in layers:
        c = op_class(l.get("type", ""))
        by_cls_n[c] += 1
        if l.get("time_us") is not None:
            by_cls_t[c] += l["time_us"]
    src_cls = Counter()
    for t, n in source_op_types.items():
        src_cls[op_class(t)] += n
    caveat = ["vendor graph, runtime and precision differ from the target backend; validate on target"]

    # hotspots: op classes by share of vendor time
    if total:
        for i, (c, t) in enumerate(sorted(by_cls_t.items(), key=lambda kv: -kv[1])):
            share = t / total
            if share < 0.05 or i >= 4:
                break
            top = sorted((l for l in layers if op_class(l.get("type", "")) == c and l.get("time_us") is not None),
                         key=lambda l: -l["time_us"])[:5]
            hints.append(Hint(f"hotspot-{c}", "hotspot", f"{c} ops take {share:.0%} of vendor inference time",
                              {"class_time_us": round(t, 2), "total_time_us": round(total, 2), "n_layers": by_cls_n[c],
                               "top_layers": [(l["name"], l["type"], l["time_us"]) for l in top]},
                              op_types=[c], share=share, source=source, confidence="high", caveats=caveat,
                              tags=CLASS_TAGS.get(c, ["kernel"])))
    # fusion: classes the vendor graph executes with fewer ops than the source graph
    n_src, n_ven = sum(source_op_types.values()), len(layers)
    for c in ("elementwise", "gelu", "norm", "softmax"):
        if src_cls[c] and by_cls_n[c] < src_cls[c]:
            hints.append(Hint(f"fusion-{c}", "fusion",
                              f"{c}: source graph has {src_cls[c]} ops, vendor graph executes {by_cls_n[c]} "
                              f"(fused or folded into producers/consumers)",
                              {"source_ops": src_cls[c], "vendor_layers": by_cls_n[c], "source_total": n_src,
                               "vendor_total": n_ven}, op_types=[c], source=source, confidence="medium", caveats=caveat,
                              tags=["fusion"]))
    # layout: transposes/reshapes removed by the vendor compiler
    if src_cls["layout"] and by_cls_n["layout"] < src_cls["layout"]:
        hints.append(Hint("layout-elimination", "layout",
                          f"layout ops: source {src_cls['layout']} -> vendor {by_cls_n['layout']} executed",
                          {"source_layout_ops": src_cls["layout"], "vendor_layout_layers": by_cls_n["layout"],
                           "vendor_layout_time_us": round(by_cls_t["layout"], 2) if total else None},
                          op_types=["layout"], share=(by_cls_t["layout"] / total) if total else None, source=source,
                          confidence="medium", caveats=caveat))
    # placement: anything not on the NPU
    units = Counter(l.get("compute_unit", "?") for l in layers)
    off = [l for l in layers if l.get("compute_unit") not in npu_units]
    if off:
        hints.append(Hint("placement-fallback", "placement",
                          f"{len(off)}/{len(layers)} vendor layers run outside the NPU ({dict(units)})",
                          {"units": dict(units), "layers": [(l["name"], l["type"], l.get("compute_unit")) for l in off[:20]]},
                          op_types=sorted({op_class(l.get('type', '')) for l in off}), source=source, confidence="high",
                          caveats=caveat))
    # precision: explicit conversions inserted by the vendor
    if by_cls_n["convert"]:
        hints.append(Hint("precision-converts", "precision",
                          f"vendor graph contains {by_cls_n['convert']} conversion layers",
                          {"n": by_cls_n["convert"]}, op_types=["convert"], source=source, confidence="medium",
                          caveats=caveat + ["the target's dtype contract may forbid precision changes"]))
    # memory: informational
    mem = {k: v for k, v in (summary or {}).items() if "memory" in k or "load_time" in k}
    if mem:
        hints.append(Hint("memory-summary", "memory", "vendor load/peak-memory figures (informational)", mem,
                          source=source, confidence="low", caveats=caveat))
    return hints
