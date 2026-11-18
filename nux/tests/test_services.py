"""Services are callable without the CLI (refactor 6/6)."""
import json

from nux import registry
from nux.services.environment import toolchain_lock
from nux.services.experiment import best_states, run_experiment


def test_toolchain_lock_covers_every_backend_without_vendor_code():
    lock = toolchain_lock()
    assert set(lock["backends"]) == set(registry.available())
    hx = lock["backends"]["hexagon-mlir"]
    assert "traits" in hx and "toolchain_importable" in hx["environment"]


def test_run_experiment_service(tmp_path):
    cfg = {"backend_id": "mock-mlir", "workload_id": "toy-mlp", "evaluation_contract_id": "toy-fp32-v0",
           "informational_measurement": True, "policy": {"warmup": 1, "pairs": 2, "requests_per_block": 2},
           "candidates": [{"action_id": "mock.tile", "kind": "config", "params": {"tile": 32}},
                          {"action_id": "nope", "kind": "config"}]}
    p = tmp_path / "c.json"
    p.write_text(json.dumps(cfg))
    events = []
    s = run_experiment(p, on_event=lambda k, x: events.append(k), store_root=tmp_path / "store",
                       artifacts_root=tmp_path / "art")
    assert [e for e in events] == ["start", "record", "record", "record", "done"]
    assert [r.state for r in s["records"]] == ["host_correct", "host_correct", "rejected"]
    assert best_states(tmp_path / "store")[0]["state"] == "host_correct"
