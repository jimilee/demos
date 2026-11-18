"""Architecture rules (refactor 3/6): dependency direction and the shared transform error contract."""
import ast
from pathlib import Path

import pytest
import torch

from nux import contract, mlir_host, registry, stats, transforms
from nux.adapters import CompileFailure, Unsupported
from nux.controller import Controller
from nux.schema import Candidate
from nux.store import Store
from nux.workloads import get

PKG = Path(__file__).resolve().parents[1] / "nux"
BACKENDS = sorted(p.stem for p in (PKG / "backends").glob("*.py") if p.stem != "__init__")
COMMON = ["controller", "evaluator", "stats", "store", "capability", "schema", "adapters", "contract", "hints",
          "frontend", "transforms", "profilers", "mlir_host", "aot", "corpus", "onnx_export"]


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
        assert "backends" not in imp and "importers" not in imp, f"nux/{name}.py imports {imp}"


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
