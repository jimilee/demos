"""QNN JSON profile (output of `qnn-profile-viewer --reader libQnnJsonProfilingReader.so`).

Structure as used by Qualcomm's open-source QAI AppBuilder script
`tools/skills/qai-runner-skill/scripts/convert_qnn_to_trace.py` (BSD-3-Clause):

  {"messages": [
     {"method": "APP_CONTEXT_CREATE" | "APP_COMPOSE_GRAPHS" | "BACKEND_FINALIZE" | "BACKEND_EXECUTE" | ...,
      "appMetrics": {"startTime": .., "stopTime": ..},
      "profilingEvents": [
         {"identifier": "QNN accelerator (execute) time", "value": <us>},
         {"identifier": "Accelerator (execute) time (cycles)", "value": <cycles>,
          "sub-events": [{"identifier": "<op name> (cycles)", "value": <cycles>}, ...]}]}]}

One BACKEND_EXECUTE message per inference. Per-op values are CYCLES; time is
derived per run with that run's accelerator time / cycles ratio (the same
scaling AppBuilder uses). Per-op results are medians over runs.
Detailed profiling adds overhead: these numbers are diagnostic, not a latency KPI.
"""
from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from . import VendorProfile, register

EXEC_TIME = "QNN accelerator (execute) time"
EXEC_CYCLES = "Accelerator (execute) time (cycles)"


def _strip(ident: str) -> str:
    return ident[: -len(" (cycles)")] if ident.endswith(" (cycles)") else ident


@register("qnn-json", "qnn-profile-viewer + libQnnJsonProfilingReader output (per-op cycles per BACKEND_EXECUTE)")
def load(path, source="", compute_unit="HTP", name_to_type=None):
    data = json.loads(Path(path).read_text())
    msgs = data.get("messages", [])
    found = {"methods": sorted({m.get("method", "?") for m in msgs}),
             "event_identifiers": sorted({e.get("identifier", "") for m in msgs for e in m.get("profilingEvents", [])})[:50]}
    warnings = []
    phases = defaultdict(list)
    exec_us, exec_cyc = [], []
    per_op_cyc, per_op_us = defaultdict(list), defaultdict(list)
    order = []
    for m in msgs:
        meth = m.get("method", "")
        am = m.get("appMetrics") or {}
        if "startTime" in am and "stopTime" in am:
            phases[meth].append(am["stopTime"] - am["startTime"])
        if meth != "BACKEND_EXECUTE":
            continue
        t_us = cyc = None
        subs = []
        for e in m.get("profilingEvents", []):
            ident = e.get("identifier", "")
            if EXEC_CYCLES in ident:
                cyc = float(e.get("value", 0) or 0)
                subs = e.get("sub-events", []) or []
            elif EXEC_TIME in ident:
                t_us = float(e.get("value", 0) or 0)
        if cyc:
            exec_cyc.append(cyc)
        if t_us:
            exec_us.append(t_us)
        scale = (t_us / cyc) if (t_us and cyc) else None
        for s in subs:
            name = _strip(s.get("identifier", "?"))
            c = float(s.get("value", 0) or 0)
            if name not in per_op_cyc:
                order.append(name)
            per_op_cyc[name].append(c)
            if scale is not None:
                per_op_us[name].append(c * scale)
    if not exec_cyc:
        warnings.append(f"no BACKEND_EXECUTE '{EXEC_CYCLES}' events found; check the reader/profiling level")
    if exec_cyc and not exec_us:
        warnings.append(f"no '{EXEC_TIME}' events: per-op time_us unavailable (cycles only)")
    layers = []
    for name in order:
        cyc = per_op_cyc[name]
        us = per_op_us.get(name) or []
        layers.append({"name": name, "type": (name_to_type or {}).get(name) or name, "compute_unit": compute_unit,
                       "cycles": statistics.median(cyc), "time_us": statistics.median(us) if us else None,
                       "n_samples": len(cyc)})
    summary = {"runs": len(exec_cyc),
               "accelerator_execute_us_median": statistics.median(exec_us) if exec_us else None,
               "accelerator_execute_us_all": exec_us,
               "accelerator_execute_cycles_median": statistics.median(exec_cyc) if exec_cyc else None,
               "phase_durations": {k: statistics.median(v) for k, v in phases.items()},
               "phase_unit": "as reported in appMetrics (start/stop difference)",
               "note": "detailed profiling run: diagnostic, not a latency KPI"}
    if exec_cyc and exec_us:
        summary["us_per_cycle"] = statistics.median(exec_us) / statistics.median(exec_cyc)
    return VendorProfile(source=source or str(path), format="qnn-json", layers=layers, summary=summary,
                         runs=len(exec_cyc), time_basis="per-op cycles scaled by per-run accelerator time/cycles",
                         found_keys=found, warnings=warnings)
