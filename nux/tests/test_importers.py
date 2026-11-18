"""Vendor profile importers. Fixtures are SYNTHETIC files shaped like the documented QNN JSON profile
(structure from Qualcomm QAI AppBuilder's BSD-licensed convert_qnn_to_trace.py); they test parsing and
the hint/seed pipeline only and are never written to reports/."""
import json
from pathlib import Path

import pytest

from nux import importers
from nux.engine import registry
from nux.importers import hints as Hn
from nux.importers import pipeline
from nux.importers.onnx_map import name_to_type

OPS = [  # (vendor op name, cycles) - fewer ops than the source graph, like a fusing compiler
    ("node_conv2d", 900_000), ("node_layer_norm", 120_000), ("node_MatMul_10", 2_000_000),
    ("node_MatMul_20", 2_100_000), ("node_Softmax_74", 400_000), ("node_Erf_94", 150_000),
    ("node_transpose", 50_000), ("node_Add_5", 30_000),
]


def qnn_fixture(path: Path, runs=3):
    msgs = [{"method": "APP_CONTEXT_CREATE", "appMetrics": {"startTime": 0, "stopTime": 1200}},
            {"method": "BACKEND_FINALIZE", "appMetrics": {"startTime": 10, "stopTime": 30}}]
    for r in range(runs):
        total = sum(c for _, c in OPS)
        msgs.append({"method": "BACKEND_EXECUTE", "appMetrics": {"startTime": 100 + r, "stopTime": 4000 + r},
                     "profilingEvents": [
                         {"identifier": "QNN accelerator (execute) time", "value": 3000.0 + r},
                         {"identifier": "Accelerator (execute) time (cycles)", "value": total,
                          "sub-events": [{"identifier": f"{n} (cycles)", "value": c} for n, c in OPS]}]})
    path.write_text(json.dumps({"messages": msgs}))
    return path


def test_qnn_json_parse(tmp_path):
    p = importers.load("qnn-json", qnn_fixture(tmp_path / "p.json"), source="test")
    assert p.runs == 3 and len(p.layers) == len(OPS) and not p.warnings
    mm = next(l for l in p.layers if l["name"] == "node_MatMul_20")
    assert mm["cycles"] == 2_100_000 and mm["n_samples"] == 3
    total_c = sum(c for _, c in OPS)
    assert mm["time_us"] == pytest.approx(2_100_000 * 3001.0 / total_c)  # median run scale
    assert p.summary["accelerator_execute_us_median"] == 3001.0
    assert "BACKEND_EXECUTE" in p.found_keys["methods"]


def test_qnn_json_missing_events_warns(tmp_path):
    f = tmp_path / "empty.json"
    f.write_text(json.dumps({"messages": [{"method": "APP_CONTEXT_CREATE"}]}))
    p = importers.load("qnn-json", f)
    assert p.layers == [] and p.warnings


def test_chrometrace_parse(tmp_path):
    ev = [{"name": "HTP Execution (Overall)", "cat": "EXECUTE", "ph": "X", "ts": 0, "dur": 3000},
          {"name": "node_MatMul_10", "cat": "OP", "ph": "X", "ts": 0, "dur": 700, "args": {"cycles": 2e6}},
          {"name": "node_Softmax_74", "cat": "OP", "ph": "X", "ts": 700, "dur": 140}]
    f = tmp_path / "t.json"
    f.write_text(json.dumps({"traceEvents": ev}))
    p = importers.load("chrometrace", f)
    assert [l["name"] for l in p.layers] == ["node_MatMul_10", "node_Softmax_74"]
    assert p.summary["overall_execute_us"] == 3000


def test_name_mapping():
    src = [("node_MatMul_10", "MatMul"), ("/vit/layer.0/Add", "Add"), ("node_layer_norm_1", "LayerNormalization")]
    m, how = name_to_type(["node_MatMul_10", "_vit_layer_0_Add", "node_layer_norm_1_fused_relu", "zzz"], src)
    assert m["node_MatMul_10"] == "MatMul" and m["_vit_layer_0_Add"] == "Add"
    assert m["node_layer_norm_1_fused_relu"] == "LayerNormalization" and "zzz" not in m
    assert how == {"exact": 1, "sanitized": 1, "contains": 1, "unmatched": 1}


def test_pipeline_hints_and_seeds(tmp_path):
    onnx = Path(__file__).resolve().parents[1] / ".cache" / "vendor" / "deit-tiny.onnx"
    if not onnx.exists():
        pytest.skip("run `python -m nux export-onnx` first")
    bc = tmp_path / "htp.json"
    bc.write_text(json.dumps({"graphs": [{"graph_names": ["deit"], "vtcm_mb": 8, "O": 3}], "devices": [{"dsp_arch": "v75"}]}))
    s = pipeline.run("qnn-json", qnn_fixture(tmp_path / "p.json"), "synthetic-test", ["hexagon-mlir", "llvm-aot-hexagon"],
                     onnx_path=onnx, build_config_path=bc, converter_args="--input_network m.onnx", precision="fp16",
                     workload_meta={"dtype": "float32"}, out_root=tmp_path / "out")
    ids = {h[0] for h in s["hints"]}
    assert {"hotspot-matmul", "fusion-elementwise", "layout-elimination", "build-vtcm", "build-native-io-layout"} <= ids
    how = s["meta"]["name_mapping"]
    assert sum(how.values()) == len(OPS) and how.get("exact", 0) >= 4  # fixture names partly exist in the real ONNX
    hex_cfg = json.loads((tmp_path / "out" / "synthetic-test" / "seeds_hexagon-mlir.json").read_text())
    acts = {c["action_id"] for c in hex_cfg["candidates"]}
    assert "opt.fusion" in acts or "opt.fusionDoMultiUse" in acts
    assert "path.hexkl_matmul" not in acts  # filtered: fp32 workload, HexKL needs f16
    assert any("float16" in x.get("reason", "") for x in hex_cfg["_skipped"])
    assert all(c["hypothesis"].startswith("[") for c in hex_cfg["candidates"])


def test_seed_generation_respects_capability():
    h = [Hn.Hint("h1", "precision", "vendor inserted converts")]
    m = registry.resolve({"backend_id": "hexagon-mlir"}).manifest
    seeds, skipped = Hn.seed_candidates(h, m, {"dtype": "float32"})
    assert seeds == [] and skipped  # fp16 conversion is unavailable by contract
