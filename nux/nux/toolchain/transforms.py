"""Vendor-neutral IR transforms applied to candidates (one implementation for every backend).

Error contract (relied on by the controller):
  precondition not met / unknown recipe  -> Unsupported
  pass failure or verifier failure       -> CompileFailure
"""
from __future__ import annotations

from ..core.adapters import CompileFailure, Unsupported
from ..core.capability import ActionSpec
from . import mlir_host as H

RECIPE_PREFIX = "recipe."


def recipe_actions(evidence: str) -> list:
    """Capability entries for all registered recipes (backends add their own evidence text)."""
    return [ActionSpec(f"{RECIPE_PREFIX}{r.recipe_id}", "ir_recipe", r.meaning, r.stage, support="supported",
                       evidence=evidence, tags=list(r.tags), numeric_change=r.numeric_change)
            for r in H.RECIPES.values()]


def recipe_for(action_id: str) -> H.Recipe:
    if not action_id.startswith(RECIPE_PREFIX):
        raise Unsupported(f"'{action_id}' is not an IR recipe action")
    rid = action_id[len(RECIPE_PREFIX):]
    if rid not in H.RECIPES:
        raise Unsupported(f"unknown recipe '{rid}'")
    return H.RECIPES[rid]


def apply_ir_recipe(mod: H.MLIRModule, action_id: str) -> H.MLIRModule:
    r = recipe_for(action_id)
    ok, why = H.recipe_applicable(r, mod)
    if not ok:
        raise Unsupported(f"recipe precondition: {why}")
    try:
        out = H.apply_recipe(mod, r)
    except Exception as e:  # noqa: BLE001
        raise CompileFailure(f"{r.recipe_id}: {type(e).__name__}: {e}") from e
    ok, why = H.verify(out)
    if not ok:
        raise CompileFailure(f"{r.recipe_id}: verifier: {why}")
    return out


def source_ir(parent) -> H.MLIRModule:
    """The IR a candidate derives from (integrated artifacts keep it in `ir`)."""
    return parent.ir if parent.ir is not None else parent.payload
