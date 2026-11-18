"""Capability filtering, statistics, evaluator admissibility, contract, store immutability."""
import random
import re
from pathlib import Path

import pytest
import torch

from nux.backends import mock
from nux.core import contract as C
from nux.core import stats
from nux.core.capability import validate_action
from nux.core.schema import Candidate, CandidateRecord, Measurement
from nux.engine import evaluator
from nux.storage.store import Store

ROOT = Path(__file__).resolve().parents[1]


def test_capability_filtering_differs_by_backend():
    mlir, sdk = mock.mlir_manifest(), mock.sdk_manifest()
    rec = Candidate("recipe.fold-unit-extent-dims", "ir_recipe")
    assert validate_action(mlir, rec)[0] and not validate_action(sdk, rec)[0]
    assert not validate_action(mlir, Candidate("path.library_matmul", "lowering_path"))[0]
    assert not validate_action(mlir, Candidate("mock.tile", "config", {"tile": 48}))[0]  # not in choices
    assert not validate_action(mlir, Candidate("mock.tile", "config", {"bogus": 1}))[0]
    assert not validate_action(mlir, Candidate("mock.tile", "config", {"tile": 32}, scope="op:matmul"))[0]  # module-wide
    assert not validate_action(mlir, Candidate("nope", "config"))[0]
    assert not validate_action(mlir, Candidate("mock.tile", "kernel"))[0]  # kind mismatch


def test_hexagon_manifest_without_toolchain():
    from nux.backends import hexagon

    m = hexagon.manifest()
    assert m.execution.executes_on == "device" and m.execution.target_codegen and m.measurement.e2e == "unavailable"
    assert not m.acceptance_capable  # no session-reusing E2E runner yet
    assert all(a.validation == "unvalidated" for a in m.actions)
    hexkl = Candidate("path.hexkl_matmul", "lowering_path", {"mode": "micro"})
    ok, why = validate_action(m, hexkl, {"dtype": "float32"})
    assert not ok and "float16" in why  # fp32 DeiT baseline: HexKL needs f16, dtype change is forbidden
    assert not validate_action(m, hexkl, {"dtype": "float16", "arch": "81"})[0]
    assert not validate_action(m, Candidate("opt.enableConversionToFp16", "config", {"value": True}))[0]
    assert not validate_action(m, Candidate("kernel.triton_replace", "kernel"))[0]


def _blocks(center, n=10, k=20, jitter=0.01, seed=0):
    r = random.Random(seed)
    return [[center * (1 + r.uniform(-jitter, jitter)) for _ in range(k)] for _ in range(n)]


def test_stats_decide():
    pol = stats.Policy()
    assert stats.decide(_blocks(100), _blocks(90, seed=1), pol)["verdict"] == "improved"
    assert stats.decide(_blocks(100), _blocks(99, seed=1), pol)["verdict"] in ("not_improved", "inconclusive")
    assert stats.decide(_blocks(100), _blocks(110, seed=1), pol)["verdict"] == "not_improved"
    # median better but p95 much worse -> not improved
    cand = _blocks(90, seed=1)
    for b in cand:
        b[-2:] = [200, 200]
    d = stats.decide(_blocks(100), cand, pol)
    assert d["verdict"] != "improved" and any("p95" in r for r in d["reasons"])
    lo, hi = d["speedup_ci"]
    assert lo <= hi


def _manifest(executes_on, tier, e2e="supported"):
    from nux.core.capability import CapabilityManifest, ExecutionCaps, MeasurementCaps

    return CapabilityManifest("t", "1", "c", "r", "p", [], ["float32"], [], MeasurementCaps(e2e=e2e, tier=tier),
                              ExecutionCaps(executes_on=executes_on))


def test_evaluator_never_accepts_non_device_evidence():
    pol = stats.Policy()
    for src, man in (("synthetic", _manifest("host", "synthetic")), ("host", _manifest("host", "host")),
                     ("synthetic", _manifest("device", "device")), ("device", _manifest("host", "synthetic")),
                     ("device", _manifest("device", "device", e2e="unavailable")), ("device", _manifest("none", "device"))):
        b, c = Measurement(src, "x", "us", _blocks(100)), Measurement(src, "x", "us", _blocks(50, seed=2))
        assert evaluator.judge(b, c, man, pol, device_correct=True)["verdict"] == "not_admissible", (src, man.traits())
    dev = _manifest("device", "device")
    assert dev.acceptance_capable
    b, c = Measurement("device", "x", "us", _blocks(100)), Measurement("device", "x", "us", _blocks(50, seed=2))
    assert evaluator.judge(b, c, dev, pol, device_correct=True)["verdict"] == "accepted"
    assert evaluator.judge(b, c, dev, pol, device_correct=False)["verdict"] == "rejected"
    c.instrumentation = True
    assert evaluator.judge(b, c, dev, pol, device_correct=True)["verdict"] == "not_admissible"


def test_capability_values_validated_without_assert():
    from nux.core.capability import ActionSpec, ExecutionCaps, MeasurementCaps

    for bad in (lambda: ExecutionCaps(executes_on="cloud"), lambda: MeasurementCaps(tier="guess"),
                lambda: ActionSpec("x", "config", "d", "s", tags=["speed"])):
        with pytest.raises(ValueError):
            bad()


def test_contract_compare():
    con = C.get("deit-tiny-fp32-logits-v0")
    ref = torch.randn(1, 1000)
    assert C.compare(con, ref.clone(), ref)["ok"]
    bad = ref.clone()
    bad[0, 3] += 1e-2
    r = C.compare(con, bad, ref)
    assert not r["ok"] and r["n_violations"] == 1
    nan = ref.clone()
    nan[0, 0] = float("nan")
    assert not C.compare(con, nan, ref)["ok"]
    assert not C.compare(con, ref.double(), ref)["ok"]  # dtype is part of the contract
    assert not C.compare(con, ref[:, :10], ref)["ok"]


def test_store_records_are_immutable(tmp_path):
    s = Store(tmp_path)
    eid = s.new_experiment({"a": 1}, {})
    rec = CandidateRecord(eid, {"candidate_id": "abc", "action_id": "x", "kind": "config", "params": {}}, "b", "w", "c", "proposed",
                          stages=[{"stage": "proposed", "status": "passed"}])
    s.write_record(rec)
    with pytest.raises(FileExistsError):
        s.write_record(rec)
    s.set_best(Store.best_key("b1", "w", "c"), {"x": 1})
    s.set_best(Store.best_key("b2", "w", "c"), {"x": 2})
    assert s.get_best(Store.best_key("b1", "w", "c")) == {"x": 1}


COMMON = ["engine/controller", "engine/stages", "engine/evaluator", "engine/measurement", "engine/registry",
          "core/stats", "core/capability", "core/schema", "core/adapters", "core/contract", "storage/store",
          "storage/artifact_store"]


def test_no_vendor_branching_in_common_layers():
    # vendor names, and branching on backend identity/kind instead of declared traits
    pat = re.compile(r"hexagon|qualcomm|qnn|qairt|mock|host-linalg|host_reference|mediatek|samsung|exynos|apple"
                     r"|\b(?:b|bundle)\.kind\b|bundle_kind|backend_id\s*==", re.I)
    for name in COMMON:
        src = (ROOT / "nux" / f"{name}.py").read_text()
        code = "\n".join(l for l in src.splitlines() if not l.strip().startswith("#"))
        code = re.sub(r'"""[\s\S]*?"""', "", code)
        assert not pat.search(code), f"vendor-specific reference in common module nux/{name}.py: {pat.search(code).group(0)}"


def test_same_config_twice_in_one_second_does_not_collide(tmp_path):
    s = Store(tmp_path)
    assert s.new_experiment({"a": 1}, {}) != s.new_experiment({"a": 1}, {})


def test_config_is_strict():
    from nux.engine.config import ConfigError, parse

    base = {"backend_id": "mock-mlir", "workload_id": "toy-mlp", "evaluation_contract_id": "toy-fp32-v0"}
    assert parse({**base, "_note": "comments are allowed"}).comments == {"_note": "comments are allowed"}
    for bad in ({**base, "device_profile": "x"}, {**base, "backend_id": "nope"}, {**base, "correctness_splits": ["holdout"]},
                {**base, "budget": {"max_candidates": "10"}}, {**base, "candidates": [{"action_id": "a"}]},
                {**base, "candidates": [{"action_id": "a", "kind": "config", "parmas": {}}]}):
        with pytest.raises(ConfigError):
            parse(bad)


def test_record_validation_blocks_inconsistent_records(tmp_path):
    from nux.core.schema import RecordError

    s = Store(tmp_path)
    eid = s.new_experiment({}, {})
    ok_stage = [{"stage": "proposed", "status": "passed"}]
    cand = {"candidate_id": "c1", "action_id": "x", "kind": "config", "params": {}}
    for kw in ({"state": "rejected"},  # rejected without failure
               {"state": "host_correct", "failure": "compile_failure"},  # failure without rejected
               {"state": "weird"},
               {"state": "accepted"},  # accepted from a non-acceptance-capable backend
               {"state": "host_correct", "measurement": {"source": "guess", "boundary": "", "unit": "us", "blocks": []}}):
        rec = CandidateRecord(eid, dict(cand), "b", "w", "c", stages=ok_stage, **kw)
        with pytest.raises(RecordError):
            s.write_record(rec)


def test_contract_require_top1_and_backend_dtype_gate():
    c16 = C.get("deit-tiny-fp16-logits-v0")
    ref = torch.tensor([[1.000, 1.005, 0.0, 0.0, 0.0]])
    out = torch.tensor([[1.004, 1.001, 0.0, 0.0, 0.0]], dtype=torch.float16)  # within atol/rtol, top-1 flips
    r = C.compare(c16, out, ref)
    assert r["n_violations"] == 0 and not r["ok"] and "top-1" in r["reason"]
    assert C.compare(c16, out.float(), ref)["reason"].startswith("dtype")  # output dtype is part of the contract
    m = mock.mlir_manifest()  # declares float32 only
    ok, why = validate_action(m, Candidate("baseline", "baseline"), {"dtype": "float16"})
    assert not ok and "float16" in why
    assert validate_action(m, Candidate("baseline", "baseline"), {"dtype": "float32"})[0]
