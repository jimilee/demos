"""Artifact store + retention policy (refactor 4/6)."""
import time

import pytest

from nux.core import contract, stats
from nux.core.schema import Candidate
from nux.engine import registry
from nux.engine.controller import Controller
from nux.storage.artifact_store import ArtifactPolicy, ArtifactStore
from nux.storage.store import Store
from nux.toolchain import aot
from nux.workloads import get


def _put(st, key, tmp_path, sizes, pins=()):
    files, roles = {}, {}
    for name, (role, n) in sizes.items():
        p = tmp_path / f"src_{key}_{name}.bin"
        p.write_bytes(b"x" * n)
        files[name], roles[name] = p, role
    st.put(key, files, roles, {"k": key})
    for r in pins:
        st.pin(key, r)


def test_put_get_verify_and_corruption(tmp_path):
    st = ArtifactStore(tmp_path / "s")
    _put(st, "k1" * 8, tmp_path, {"lib": ("primary", 10), "src": ("source", 5)})
    e = st.get("k1" * 8)
    assert set(e["paths"]) == {"lib", "src"} and e["meta"] == {"k": "k1" * 8}
    with open(e["paths"]["lib"], "ab") as f:  # corrupt a primary file
        f.write(b"!")
    assert st.get("k1" * 8) is None and st.manifest("k1" * 8) is None  # hash mismatch -> evicted -> miss


def test_record_time_policy(tmp_path):
    st = ArtifactStore(tmp_path / "s")
    _put(st, "aa" * 8, tmp_path, {"lib": ("primary", 10), "ll": ("intermediate", 100)})
    _put(st, "bb" * 8, tmp_path, {"lib": ("primary", 10), "ll": ("intermediate", 100)})
    st.after_record("aa" * 8, rejected=True)
    assert st.manifest("aa" * 8) is None  # rejected -> deleted
    st.after_record("bb" * 8, rejected=False)
    assert set(st.get("bb" * 8)["paths"]) == {"lib"}  # intermediates dropped, primary kept
    _put(st, "cc" * 8, tmp_path, {"lib": ("primary", 10), "twin": ("verification", 20), "ll": ("intermediate", 100)})
    st.after_record("cc" * 8, rejected=False)
    assert set(st.get("cc" * 8)["paths"]) == {"lib", "twin"}  # verification files survive the record


def test_pinned_entries_survive_lru_gc(tmp_path):
    st = ArtifactStore(tmp_path / "s", ArtifactPolicy(max_bytes=3500))  # each entry ~1.4 KB incl. manifest
    for i, k in enumerate(["p0", "u1", "u2", "u3"]):
        _put(st, k * 16, tmp_path, {"lib": ("primary", 1000)}, pins=("best",) if k == "p0" else ())
        time.sleep(0.01)
    st.get("u1" * 16)  # u1 becomes most recently used among unpinned
    r = st.gc()
    assert st.manifest("p0" * 16) is not None  # pinned, never evicted
    assert "u2" * 16 in r["evicted"] and st.manifest("u1" * 16) is not None  # LRU order: u2 goes before u1
    assert r["bytes"] <= 3500 and st.manifest("u3" * 16) is None


def _ctl(bid, root, store, cfg=None):
    return Controller(registry.resolve({"backend_id": bid}), get("toy-mlp"), contract.get("toy-fp32-v0"), Store(root / "rec"),
                      stats.Policy(), {"backend_id": bid, **(cfg or {})}, artifacts=store)


def test_controller_reuses_and_restores_host_aot(tmp_path):
    st = ArtifactStore(tmp_path / "art")
    c1 = _ctl("llvm-aot-host", tmp_path, st)
    b1 = c1.run_baseline()
    r1 = c1.run_candidate(Candidate("recipe.fuse-elementwise", "ir_recipe"))
    bad = c1.run_candidate(Candidate("host.nope", "config"))  # rejected before compile: no artifact
    assert not b1.cache_hit and not r1.cache_hit and bad.state == "rejected"
    assert st.stats()["pinned"] == 1  # baseline pinned; candidate not accepted (prepare mode) -> unpinned
    # a fresh controller (= restart) hits the store for identical inputs
    c2 = _ctl("llvm-aot-host", tmp_path, st)
    b2 = c2.run_baseline()
    assert b2.cache_hit and b2.cache_key == b1.cache_key and b2.state == "host_correct"
    assert b2.correctness["max_abs_err"] == b1.correctness["max_abs_err"]
    art, source = _ctl("llvm-aot-host", tmp_path, st).restore_best()
    assert source == "store" and art.kind == "shared_library"
    # intermediates were dropped after the records were written
    files = st.get(b1.cache_key)["paths"]
    assert "shared_library" in files and "source_ir" in files and "object" not in files


@pytest.mark.skipif(not aot.llc_path() or "hexagon" not in aot.llc_targets(), reason="llc with hexagon target needed")
def test_hexagon_object_cache_respects_twin_requirement(tmp_path):
    st = ArtifactStore(tmp_path / "art")
    b1 = _ctl("llvm-aot-hexagon", tmp_path, st).run_baseline()
    assert b1.state == "target_compiled" and not b1.cache_hit
    # twin .so is a `verification` file: kept, so re-evaluation hits AND still re-checks semantics
    b2 = _ctl("llvm-aot-hexagon", tmp_path, st).run_baseline()
    assert b2.cache_hit and b2.state == "target_compiled"
    assert any(s["stage"] == "twin_correctness" and s["status"] == "passed" for s in b2.stages)
    files = st.get(b1.cache_key)["paths"]
    assert {"hexagon_object", "twin_so", "source_ir"} <= set(files) and "llvm_ir" not in files and "twin_x86_object" not in files
    art, source = _ctl("llvm-aot-hexagon", tmp_path, st).restore_best()
    assert source == "store" and art.target_compiled and "hexagon_object" in art.files
    # if the twin is gone anyway (e.g. removed by hand), evaluation treats it as a miss and recompiles
    import os
    os.unlink(files["twin_so"])
    b3 = _ctl("llvm-aot-hexagon", tmp_path, st).run_baseline()
    assert not b3.cache_hit and b3.state == "target_compiled"


def test_disabled_store_keeps_old_behavior(tmp_path):
    r = _ctl("llvm-aot-host", tmp_path, None).run_baseline()
    assert r.state == "host_correct" and not r.cache_hit
    assert any(s["artifacts"].get("cache") == "disabled" for s in r.stages if s["stage"] == "compile")
