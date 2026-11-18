"""A3: two capability mocks + host reference on ONE controller; acceptance path with a test-only device fixture."""
import json
import random

import pytest

from nux import contract, registry, stats
from nux.adapters import AdapterBundle
from nux.backends import mock
from nux.controller import Controller
from nux.schema import Candidate, Measurement
from nux.store import Store
from nux.workloads import get

POL = stats.Policy(warmup=1, pairs=4, requests_per_block=4)


def ctl(bid, root, bundle=None):
    return Controller(bundle or registry.resolve({"backend_id": bid}), get("toy-mlp"), contract.get("toy-fp32-v0"),
                      Store(root), POL, {"backend_id": bid})


def test_two_mocks_same_controller(tmp_path):
    a, b = ctl("mock-mlir", tmp_path), ctl("mock-sdk", tmp_path)
    for c in (a, b):
        assert c.run_baseline().state == "host_correct"
    rec = Candidate("recipe.fold-unit-extent-dims", "ir_recipe")
    assert a.run_candidate(rec).state == "host_correct"
    r = b.run_candidate(rec)
    assert r.state == "rejected" and r.failure == "unsupported"
    fast = a.run_candidate(Candidate("mock.tile", "config", {"tile": 32, "synthetic_speedup": "3.0"}))
    assert fast.state != "accepted" and fast.decision["verdict"] == "not_admissible"
    assert fast.decision["informational_stats"]["median_gain"] > 0.5  # stats computed, still not accepted
    # failures keep the previous best and are classified
    best_before = a.store.get_best(a.key)
    assert a.run_candidate(Candidate("mock.tile", "config", {"tile": 64, "fail": "compile"})).failure == "compile_failure"
    assert a.run_candidate(Candidate("mock.tile", "config", {"tile": 64, "fail": "runtime"})).failure == "runtime_failure"
    assert a.run_candidate(Candidate("mock.tile", "config", {"tile": 16, "perturb": "0.5"})).failure == "correctness_failure"
    assert a.store.get_best(a.key) == best_before
    # per-backend best isolation
    assert a.key != b.key and b.store.get_best(b.key)["record"] != best_before["record"]
    rows = (tmp_path / "results.tsv").read_text().splitlines()
    assert len(rows) == 1 + 2 + 6 and all("accepted" not in r.split("\t")[11] for r in rows[1:])


def test_host_reference_backend_on_toy(tmp_path):
    c = ctl("host-linalg-jit", tmp_path)
    assert c.run_baseline().state == "host_correct"
    r = c.run_candidate(Candidate("recipe.fuse-elementwise", "ir_recipe"))
    assert r.state == "host_correct" and r.decision["verdict"] == "pending"
    assert r.ir_hashes and r.ir_hashes != c.best_record.ir_hashes
    same = c.run_candidate(Candidate("recipe.canonicalize-cse", "ir_recipe"))
    assert same.state == "ir_valid" and "identical" in same.failure_detail


def test_hexagon_pending_without_toolchain(tmp_path):
    c = ctl("hexagon-mlir", tmp_path)
    r = c.run_baseline()
    if r.failure_detail.startswith("toolchain_unavailable"):
        assert r.state == "ir_valid"
        assert any(s["stage"] == "target_compile" and s["status"] == "pending" for s in r.stages)
        assert c.run_candidate(Candidate("opt.fusion", "config", {"value": False})).state == "ir_valid"
    else:
        pytest.skip("hexagon toolchain present: covered by device runs")


# ---------------- test-only device fixture (NOT registered, never used outside tests) -------------
class _DeviceProfiler(mock.Profiler):
    def measure_e2e(self, runtime, session, inputs, policy):
        rng = random.Random(len(json.dumps(session, sort_keys=True)))
        base = 1000.0 / float(session.get("synthetic_speedup") or 1.0)
        return Measurement("device", "TEST FIXTURE", "us",
                           [[base * (1 + rng.uniform(-0.005, 0.005)) for _ in range(policy.requests_per_block)]
                            for _ in range(policy.pairs)])


def _fixture_bundle():
    m = mock.mlir_manifest()
    m.kind = "device"
    m.backend_id = "test-device-fixture"
    b = mock.mlir_bundle({})
    return AdapterBundle(m, mock.Compiler("test-device-fixture"), mock.Runtime("test-device-fixture", None), _DeviceProfiler())


def test_acceptance_and_restore_path_with_device_fixture(tmp_path):
    c = ctl("x", tmp_path, bundle=_fixture_bundle())
    assert c.mode == "device"
    c.run_baseline()
    # device correctness: the fixture's runtime is the framework model, so it is correct
    win = c.run_candidate(Candidate("mock.tile", "config", {"tile": 32, "synthetic_speedup": "1.5"}))
    assert win.state == "accepted", win.decision
    lose = c.run_candidate(Candidate("mock.tile", "config", {"tile": 16, "synthetic_speedup": "0.8"}))
    assert lose.state != "accepted"
    best = c.store.get_best(c.key)
    assert [x["params"]["tile"] for x in best["chain"]] == [32]
    art = c.restore_best()
    assert art.payload["params"]["tile"] == 32
