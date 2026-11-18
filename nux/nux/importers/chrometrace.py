"""Chrome Trace JSON ({"traceEvents": [...]}): e.g. QAI AppBuilder's convert_qnn_to_trace.py output
or a profile-viewer Chrometrace reader output. Complete events (ph "X") with a duration are treated
as op executions; if a category is present, only "OP"-like categories are kept. Durations are in
microseconds (Chrome Trace convention). Schema-tolerant: what was found is recorded in found_keys.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from . import VendorProfile, register

LIFECYCLE = ("lifecycle", "execute", "graph", "context")


@register("chrometrace", "Chrome Trace JSON (traceEvents, ph=X durations in us)")
def load(path, source="", compute_unit="HTP", name_to_type=None):
    data = json.loads(Path(path).read_text())
    evs = data["traceEvents"] if isinstance(data, dict) else data
    cats = sorted({str(e.get("cat", "")) for e in evs})
    found = {"categories": cats, "phases": sorted({str(e.get("ph", "")) for e in evs}), "n_events": len(evs)}
    has_op_cat = any(c.lower() == "op" for c in cats)
    per_us, per_cyc, order, total = defaultdict(list), defaultdict(list), [], []
    for e in evs:
        if e.get("ph") != "X" or e.get("dur") is None:
            continue
        cat = str(e.get("cat", "")).lower()
        name = str(e.get("name", "?"))
        if (has_op_cat and cat != "op") or (not has_op_cat and any(k in cat for k in LIFECYCLE)):
            if "overall" in name.lower() or cat == "execute":
                total.append(float(e["dur"]))
            continue
        if name not in per_us:
            order.append(name)
        per_us[name].append(float(e["dur"]))
        cyc = (e.get("args") or {}).get("cycles")
        if cyc is not None:
            per_cyc[name].append(float(cyc))
    layers = [{"name": n, "type": (name_to_type or {}).get(n) or n, "compute_unit": compute_unit,
               "time_us": statistics.median(per_us[n]), "cycles": statistics.median(per_cyc[n]) if per_cyc[n] else None,
               "n_samples": len(per_us[n])} for n in order]
    warnings = [] if layers else ["no op duration events found"]
    return VendorProfile(source=source or str(path), format="chrometrace", layers=layers,
                         summary={"overall_execute_us": statistics.median(total) if total else None},
                         runs=max((r["n_samples"] for r in layers), default=0), time_basis="trace event durations (us)",
                         found_keys=found, warnings=warnings)
