"""Architecture rules (refactor 3/6): dependency direction and the shared transform error contract."""
import ast
from pathlib import Path

import pytest
import torch

from nux.core import contract, stats
from nux.core.adapters import CompileFailure, Unsupported
from nux.core.schema import Candidate
from nux.engine import registry
from nux.engine.controller import Controller
from nux.storage.store import Store
from nux.toolchain import mlir_host, transforms
from nux.workloads import get

PKG = Path(__file__).resolve().parents[1] / "nux"
BACKENDS = sorted(p.stem for p in (PKG / "backends").glob("*.py") if p.stem != "__init__")
LAYERS = ["core", "storage", "toolchain", "engine"]
COMMON = sorted(f"{layer}/{p.stem}" for layer in LAYERS for p in (PKG / layer).glob("*.py") if p.stem != "__init__") \
    + ["importers/hints", "workloads/corpus"]
# Allowed nux sub-packages per layer (anything not listed is a violation). Lazy imports count too.
ALLOWED = {
    "core": set(),
    "storage": {"core"},
    "toolchain": {"core"},
    "engine": {"core", "storage", "toolchain", "workloads", "backends"},  # backends: registry discovery only
}


def _imports(path):
    tree = ast.parse(path.read_text())
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom):
            out.append(("." * n.level) + (n.module or ""))
        elif isinstance(n, ast.Import):
            out += [a.name for a in n.names]
    return out


@pytest.mark.parametrize("name", BACKENDS)
def test_backends_do_not_import_each_other(name):
    for imp in _imports(PKG / "backends" / f"{name}.py"):
        if imp.startswith(".") and not imp.startswith(".."):
            assert imp.lstrip(".") not in BACKENDS, f"backends/{name}.py imports sibling backend '{imp}'"
        assert not imp.startswith("nux.backends"), f"backends/{name}.py imports {imp}"


@pytest.mark.parametrize("name", COMMON)
def test_common_layers_do_not_import_backends_or_importers(name):
    for imp in _imports(PKG / f"{name}.py"):
        if name == "engine/registry" and imp == "..backends":
            continue  # the registry's dynamic backend discovery is the one allowed edge
        assert "backends" not in imp and "importers" not in imp, f"nux/{name}.py imports {imp}"


def _layer_targets(layer, path):
    """nux sub-packages that a module in nux/<layer>/ imports from (package-root names such as SCHEMA_VERSION skipped)."""
    subpkgs = {p.name for p in PKG.iterdir() if (p / "__init__.py").exists()}
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.ImportFrom) and n.level == 1:
            yield layer
        elif isinstance(n, ast.ImportFrom) and n.level == 2:
            if n.module:
                yield n.module.split(".")[0]
            else:
                yield from (a.name for a in n.names if a.name in subpkgs)
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and (n.module or "").startswith("nux."):
            yield n.module.split(".")[1]
        elif isinstance(n, ast.ImportFrom) and n.level > 2:
            raise AssertionError(f"{layer}/{path.name}: import reaches above the package")


@pytest.mark.parametrize("layer", LAYERS)
def test_layer_dependency_direction(layer):
    for path in sorted((PKG / layer).glob("*.py")):
        for tgt in _layer_targets(layer, path):
            assert tgt == layer or tgt in ALLOWED[layer], f"{layer}/{path.name} imports layer '{tgt}' (not allowed)"


def test_flat_package_has_only_entry_points():
    assert sorted(p.name for p in PKG.glob("*.py")) == ["__init__.py", "__main__.py"]


class _NoMatmul(torch.nn.Module):
    def forward(self, x):
        return torch.relu(x) * 2.0


def _mod():
    return mlir_host.export(_NoMatmul(), [torch.zeros(1, 4)], "nm")


def test_transform_error_contract(monkeypatch):
    m = _mod()
    with pytest.raises(Unsupported):
        transforms.apply_ir_recipe(m, "recipe.does-not-exist")
    with pytest.raises(Unsupported):
        transforms.apply_ir_recipe(m, "opt.fusion")  # not a recipe action
    with pytest.raises(Unsupported):
        transforms.apply_ir_recipe(m, "recipe.generalize-named-ops")  # precondition: no named matmul/transpose

    def boom(mod, r):
        raise RuntimeError("pass crashed")

    monkeypatch.setattr(mlir_host, "apply_recipe", boom)
    with pytest.raises(CompileFailure):
        transforms.apply_ir_recipe(m, "recipe.canonicalize-cse")


@pytest.mark.parametrize("bid", ["host-linalg-jit", "llvm-aot-host"])
def test_recipe_failure_is_recorded_not_raised(bid, tmp_path, monkeypatch):
    """Regression: llvm-aot used to let a failing recipe escape the controller as a raw exception."""
    c = Controller(registry.resolve({"backend_id": bid}), get("toy-mlp"), contract.get("toy-fp32-v0"),
                   Store(tmp_path), stats.Policy(), {"backend_id": bid})
    assert c.run_baseline().state == "host_correct"

    def boom(mod, r):
        raise RuntimeError("pass crashed")

    monkeypatch.setattr(mlir_host, "apply_recipe", boom)
    r = c.run_candidate(Candidate("recipe.fuse-elementwise", "ir_recipe"))
    assert (r.state, r.failure) == ("rejected", "compile_failure") and "pass crashed" in r.failure_detail


# ---- refactor 2/6: stage pipeline and measurement protocol ------------------------------------
def test_stage_order_follows_evidence_ladder():
    from nux.engine import stages

    assert [s.__name__ for s in stages.STAGES] == [
        "capability", "compile_", "ir_verify", "dedupe", "pending", "target_compiled", "twin_correctness",
        "session", "correctness", "measure"]


def test_exactly_one_record_per_candidate(tmp_path):
    import json

    c = Controller(registry.resolve({"backend_id": "mock-mlir"}), get("toy-mlp"), contract.get("toy-fp32-v0"),
                   Store(tmp_path), stats.Policy(warmup=1, pairs=2, requests_per_block=2),
                   {"backend_id": "mock-mlir", "informational_measurement": True})
    c.run_baseline()
    cands = [Candidate("nope", "config"),  # stops at capability
             Candidate("mock.tile", "config", {"tile": 64, "fail": "compile"}),  # stops at compile
             Candidate("mock.tile", "config", {"tile": 16, "perturb": "0.5"}),  # stops at correctness
             Candidate("mock.tile", "config", {"tile": 32})]  # full pipeline incl. measurement
    recs = [c.run_candidate(x) for x in cands]
    files = list((tmp_path / "experiments" / c.eid / "candidates").glob("*.json"))
    assert len(files) == 1 + len(cands)
    assert [r.failure for r in recs] == ["unsupported", "compile_failure", "correctness_failure", None]
    full = json.loads(next(f for f in files if json.loads(f.read_text())["candidate"]["params"] == {"tile": 32}).read_text())
    assert set(full["measurement"]) == {"base", "cand"} and full["decision"]["verdict"] == "not_admissible"
    assert full["measurement"]["cand"]["conditions"]["protocol"] == "abba-paired-v1"


def test_paired_protocol_abba_and_single_warmup():
    from nux.core.schema import Measurement
    from nux.engine.measurement import PairedProtocol

    calls = []

    class P:
        def measure_e2e(self, runtime, session, inputs, policy):
            calls.append((session, policy.warmup, policy.pairs))
            return Measurement("synthetic", "x", "us", [[1.0] * policy.requests_per_block])

    b, c = PairedProtocol(stats.Policy(warmup=5, pairs=4, requests_per_block=3)).run(P(), None, "B", "C", [])
    assert [s for s, _, _ in calls] == ["B", "C", "C", "B", "B", "C", "C", "B"]
    assert [w for _, w, _ in calls] == [5, 5, 0, 0, 0, 0, 0, 0] and all(p == 1 for _, _, p in calls)
    assert len(b.blocks) == len(c.blocks) == 4


# ---- refactor 6/6: thin CLI over services ------------------------------------------------------
STDLIB_OK = {"__future__", "argparse", "json", "sys"}


def test_cli_imports_only_services():
    for imp in _imports(PKG / "__main__.py"):
        if imp.startswith("."):
            assert imp == "." or imp.startswith(".services"), f"__main__.py imports {imp} (use nux.services)"
        else:
            assert imp in STDLIB_OK, f"__main__.py imports {imp}"


@pytest.mark.parametrize("name", sorted(p.stem for p in (PKG / "services").glob("*.py")))
def test_services_reach_backends_only_through_registry(name):
    for imp in _imports(PKG / "services" / f"{name}.py"):
        assert "backends" not in imp, f"services/{name}.py imports {imp}"
