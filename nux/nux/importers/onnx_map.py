"""Map vendor op names back to the source ONNX graph (op types), and summarize the source graph.

QNN op names usually derive from the source node names (often sanitized: '/', '.', ':' -> '_').
Matching is by exact name, then sanitized name, then longest source-name containment.
Unmatched vendor ops keep their own name as type and are classified by name heuristics.
"""
from __future__ import annotations

import re
from collections import Counter


def _san(s: str) -> str:
    return re.sub(r"[^0-9A-Za-z]+", "_", s).strip("_").lower()


def source_graph(onnx_path):
    import onnx

    m = onnx.load(str(onnx_path), load_external_data=False)
    nodes = [(n.name or f"{n.op_type}_{i}", n.op_type) for i, n in enumerate(m.graph.node)]
    return {"op_types": dict(Counter(t for _, t in nodes)), "nodes": nodes}


def name_to_type(vendor_names, src_nodes):
    exact = {n: t for n, t in src_nodes}
    san = {_san(n): t for n, t in src_nodes}
    by_len = sorted(((_san(n), t) for n, t in src_nodes if len(_san(n)) >= 6), key=lambda x: -len(x[0]))
    out, how = {}, Counter()
    for v in vendor_names:
        if v in exact:
            out[v] = exact[v]
            how["exact"] += 1
            continue
        sv = _san(v)
        if sv in san:
            out[v] = san[sv]
            how["sanitized"] += 1
            continue
        hit = next((t for s, t in by_len if s in sv), None)
        if hit:
            out[v] = hit
            how["contains"] += 1
        else:
            how["unmatched"] += 1
    return out, dict(how)
