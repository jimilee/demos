"""Target-independent MLIR tooling (plan A1, §7.2), built on the torch-mlir wheel.

* export: PyTorch -> torch-mlir FX importer -> linalg-on-tensors (same entry
  point the hexagon-mlir torch-mlir tutorials use: fx.export_and_import).
* verify: parse + MLIR verifier (structural validity only, not equivalence).
* recipes: external IR transformations as named upstream pass pipelines with
  match conditions and semantics notes. Applied OUTSIDE any vendor compiler.
* host lowering: linalg -> loops -> LLVM, JIT-executed on the host CPU via the
  MLIR ExecutionEngine. This checks that the converted IR computes the same
  thing as the framework. It is NOT a performance path and NOT target codegen.
"""
from __future__ import annotations

import ctypes
import hashlib
import re
import time
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

HOST_PIPELINE = (
    "builtin.module("
    "one-shot-bufferize{bufferize-function-boundaries function-boundary-type-conversion=identity-layout-map},"
    "func.func(convert-linalg-to-loops),"
    "expand-strided-metadata,lower-affine,convert-scf-to-cf,"
    "func.func(convert-math-to-llvm),convert-math-to-libm,"
    "finalize-memref-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,convert-cf-to-llvm,"
    "reconcile-unrealized-casts)"
)


@dataclass(frozen=True)
class Recipe:
    recipe_id: str
    pipeline: str
    stage: str
    meaning: str  # what the transform does, and the semantics it must preserve
    match: str  # regex that must occur in the IR for the recipe to be applicable
    numeric_change: str = "none"
    preconditions: tuple = ()
    notes: str = ""


RECIPES = {r.recipe_id: r for r in [
    Recipe("canonicalize-cse", "builtin.module(func.func(canonicalize,cse))", "linalg-on-tensors",
           "folding/canonical forms + common subexpression elimination", r"linalg\.|tensor\.|arith\."),
    Recipe("fold-unit-extent-dims", "builtin.module(func.func(linalg-fold-unit-extent-dims),func.func(canonicalize))",
           "linalg-on-tensors",
           "DropUnitDims: remove size-1 loop dims of linalg ops via collapse/expand reshapes; element order unchanged",
           r"tensor<1x", preconditions=("static shapes", "reassociation groups contiguous"),
           notes="related to the reported DeiT 5D collapse_shape assertion; result must be re-checked on target"),
    Recipe("fold-unit-extent-dims-slices",
           "builtin.module(func.func(linalg-fold-unit-extent-dims{use-rank-reducing-slices}),func.func(canonicalize))",
           "linalg-on-tensors", "DropUnitDims using rank-reducing extract/insert_slice instead of reshapes",
           r"tensor<1x", preconditions=("static shapes",)),
    Recipe("fuse-elementwise", "builtin.module(func.func(linalg-fuse-elementwise-ops),func.func(canonicalize))",
           "linalg-on-tensors", "producer/consumer fusion of elementwise linalg.generic ops (removes materialization)",
           r"linalg\.generic", notes="can raise buffer lifetime/register pressure on target"),
    Recipe("generalize-named-ops", "builtin.module(func.func(linalg-generalize-named-ops))", "linalg-on-tensors",
           "rewrite named ops (matmul, transpose, ...) as linalg.generic", r"linalg\.(matmul|batch_matmul|transpose)",
           notes="may hide library-path patterns (e.g. HexKL matmul matching) on targets"),
]}


def _ir():
    from torch_mlir import ir

    return ir


def sha(text_or_bytes) -> str:
    b = text_or_bytes.encode() if isinstance(text_or_bytes, str) else text_or_bytes
    return hashlib.sha256(b).hexdigest()


@dataclass
class MLIRModule:
    """Bytecode + text-level facts; re-parsed into a fresh Context when needed."""

    bytecode: bytes
    func_name: str
    stage: str = "linalg-on-tensors"
    provenance: list = field(default_factory=list)

    @property
    def sha256(self):
        return sha(self.bytecode)

    def parse(self):
        ir = _ir()
        ctx = ir.Context()
        return ctx, ir.Module.parse(self.bytecode, context=ctx)

    def text(self, large_elements_limit=None) -> str:
        ctx, m = self.parse()
        with ctx:
            return m.operation.get_asm(large_elements_limit=large_elements_limit)


def export(module, inputs, func_name) -> MLIRModule:
    from torch_mlir import fx

    m = fx.export_and_import(module, *inputs, output_type="linalg-on-tensors", func_name=func_name)
    return MLIRModule(m.operation.get_asm(binary=True), func_name,
                      provenance=["torch-mlir fx.export_and_import(output_type=linalg-on-tensors)"])


def verify(mod: MLIRModule) -> tuple:
    try:
        ctx, m = mod.parse()
        with ctx:
            ok = m.operation.verify()
        return bool(ok), "verifier passed" if ok else "verifier returned false"
    except Exception as e:  # noqa: BLE001
        return False, f"{type(e).__name__}: {str(e)[:500]}"


def recipe_applicable(recipe: Recipe, mod: MLIRModule) -> tuple:
    txt = mod.text(large_elements_limit=16)
    if not re.search(recipe.match, txt):
        return False, f"match pattern {recipe.match!r} not found"
    return True, "ok"


def apply_recipe(mod: MLIRModule, recipe: Recipe) -> MLIRModule:
    from torch_mlir.passmanager import PassManager

    ctx, m = mod.parse()
    with ctx:
        PassManager.parse(recipe.pipeline).run(m.operation)
        if not m.operation.verify():
            raise RuntimeError(f"verifier failed after {recipe.recipe_id}")
        bc = m.operation.get_asm(binary=True)
    return MLIRModule(bc, mod.func_name, mod.stage, mod.provenance + [f"recipe:{recipe.recipe_id} = {recipe.pipeline}"])


class HostJIT:
    """Linalg -> LLVM -> ExecutionEngine on the host CPU."""

    def __init__(self, mod: MLIRModule, opt_level=2):
        from torch_mlir.execution_engine import ExecutionEngine
        from torch_mlir.passmanager import PassManager

        ir = _ir()
        ctx, m = mod.parse()
        self._ctx = ctx
        with ctx:
            f = self._func(m, mod.func_name)
            self.result_types = [str(t) for t in ir.FunctionType(ir.TypeAttr(f.attributes["function_type"]).value).results]
            f.attributes["llvm.emit_c_interface"] = ir.UnitAttr.get()
            t = time.perf_counter()
            PassManager.parse(HOST_PIPELINE).run(m.operation)
            self.lower_s = time.perf_counter() - t
            t = time.perf_counter()
            self.engine = ExecutionEngine(m, opt_level=opt_level)
            self.jit_s = time.perf_counter() - t
        self.func_name = mod.func_name
        if len(self.result_types) != 1:
            raise NotImplementedError("HostJIT supports single-result functions")
        mt = re.match(r"tensor<((?:\d+x)*)(f32|f16|f64|i64|i32)>", self.result_types[0])
        if not mt:
            raise NotImplementedError(f"unsupported result type {self.result_types[0]}")
        self.out_shape = tuple(int(d) for d in mt.group(1).strip("x").split("x") if d)
        self.out_dtype = {"f32": np.float32, "f16": np.float16, "f64": np.float64, "i64": np.int64, "i32": np.int32}[mt.group(2)]

    @staticmethod
    def _func(m, name):
        for op in m.body.operations:
            if "sym_name" in op.attributes and str(op.attributes["sym_name"]).strip('"') == name:
                return op
        raise KeyError(name)

    def __call__(self, *arrays):
        from torch_mlir.runtime import (as_ctype, get_ranked_memref_descriptor, make_nd_memref_descriptor,
                                        ranked_memref_to_numpy)

        res = make_nd_memref_descriptor(len(self.out_shape), as_ctype(self.out_dtype))()
        keep = [np.ascontiguousarray(a) for a in arrays]
        args = [ctypes.pointer(ctypes.pointer(res))] + [ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(a))) for a in keep]
        self.engine.invoke(self.func_name, *args)
        return ranked_memref_to_numpy([res]).copy()


_RESHAPE = re.compile(r"(%[\w#.:]+) = tensor\.(collapse_shape|expand_shape) (%[\w#.:]+) (.*?) : (tensor<[^>]+>) into (tensor<[^>]+>)")


def _rank(t):
    body = t[len("tensor<"):-1]
    return len(body.split("x")) - 1


def ir_stats(mod: MLIRModule) -> dict:
    txt = mod.text(large_elements_limit=16)
    ops = Counter(re.findall(r"= ([a-z_]+\.[a-z_]+)", txt))
    reshapes = Counter()
    max_rank = 0
    for m in _RESHAPE.finditer(txt):
        a, b = m.group(5), m.group(6)
        reshapes[(m.group(2), a, b)] += 1
        max_rank = max(max_rank, _rank(a), _rank(b))
    return {"sha256": mod.sha256, "stage": mod.stage, "n_ops": sum(ops.values()), "ops": dict(ops.most_common()),
            "copies": ops.get("linalg.copy", 0) + ops.get("memref.copy", 0) + ops.get("tensor.insert_slice", 0),
            "reshape_signatures": len(reshapes), "max_reshape_rank": max_rank,
            "unit_dim_reshapes": sum(v for (k, a, b), v in reshapes.items() if "<1x" in a or "<1x" in b or "x1x" in a or "x1x" in b)}


def extract_reshape_reproducers(mod: MLIRModule, min_rank=5, include_unit_dims=False, limit=50):
    """Minimal standalone functions, one per distinct reshape signature (plan §13 collapse_shape issue)."""
    ir = _ir()
    txt = mod.text(large_elements_limit=16)
    seen, out = set(), []
    for m in _RESHAPE.finditer(txt):
        kind, attrs, a, b = m.group(2), m.group(4), m.group(5), m.group(6)
        sig = (kind, attrs, a, b)
        if sig in seen:
            continue
        hi = max(_rank(a), _rank(b)) >= min_rank
        unit = include_unit_dims and ("<1x" in a or "<1x" in b or "x1x" in a or "x1x" in b)
        if not (hi or unit):
            continue
        seen.add(sig)
        src = (f"func.func @repro(%arg0: {a}) -> {b} {{\n"
               f"  %0 = tensor.{kind} %arg0 {attrs} : {a} into {b}\n"
               f"  return %0 : {b}\n}}\n")
        ctx = ir.Context()
        try:
            ir.Module.parse(src, context=ctx)
            ok = True
        except Exception as e:  # noqa: BLE001
            ok = f"{type(e).__name__}: {str(e)[:200]}"
        out.append({"kind": kind, "from": a, "to": b, "reassociation": attrs, "max_rank": max(_rank(a), _rank(b)),
                    "parses": ok, "mlir": src})
        if len(out) >= limit:
            break
    return out
