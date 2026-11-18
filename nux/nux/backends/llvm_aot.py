"""Generic LLVM AOT bundles (no vendor SDK needed to compile).

  llvm-aot-host     host_reference. linalg -> LLVM (MLIR's own LLVM) -> .o -> .so,
                    executed through ctypes. AOT counterpart of host-linalg-jit.
  llvm-aot-hexagon  device kind, CODEGEN ONLY. linalg -> LLVM IR -> downgrade for
                    the system llc -> llvm.maximum/minimum expansion -> llc
                    -march=hexagon -> Hexagon ELF object (target_compiled).
                    The exact same downgraded IR is also compiled for x86-64 and
                    run against the framework reference ("twin check"), so the
                    IR given to the Hexagon backend is known to be correct.
                    This is upstream LLVM codegen, NOT the Hexagon-MLIR pipeline
                    (no VTCM tiling / DMA / HexKL), and it cannot be executed
                    here: running it needs the Hexagon SDK runtime (FastRPC skel
                    or run_main_on_hexagon) on a device.
"""
from __future__ import annotations

import os

import torch

from .. import aot
from .. import mlir_host as H
from ..adapters import (AdapterBundle, Artifact, CompilerAdapter, CompileFailure, DeviceUnavailable, ProfilerAdapter,
                        RuntimeAdapter, Unsupported)
from ..capability import ActionSpec, CapabilityManifest, MeasurementCaps, ParamSpec
from ..registry import register
from .host_linalg import Compiler as HostCompiler
from .host_linalg import Profiler as HostProfiler


def _recipe_actions(evidence):
    return [ActionSpec(f"recipe.{r.recipe_id}", "ir_recipe", r.meaning, r.stage, support="supported", evidence=evidence,
                       tags=list(r.tags))
            for r in H.RECIPES.values()]


def _apply(parent_mod, candidate, cfg):
    mod = parent_mod
    if candidate.kind == "ir_recipe":
        r = H.RECIPES[candidate.action_id.split(".", 1)[1]]
        ok, why = H.recipe_applicable(r, mod)
        if not ok:
            raise Unsupported(why)
        mod = H.apply_recipe(mod, r)
    elif candidate.kind == "config":
        cfg.update(candidate.params)
    elif candidate.kind != "baseline":
        raise Unsupported(candidate.kind)
    return mod, cfg


# --------------------------------------------------------------------- host
HOST_ID = "llvm-aot-host"


def host_manifest():
    return CapabilityManifest(
        backend_id=HOST_ID, kind="host_reference", version="mlir-own-llvm", compiler_profile="mlir-llvm-aot/x86_64",
        runtime_profile="ctypes-shared-library", profiler_profile="host-wallclock", input_stages=["linalg-on-tensors"],
        dtypes=["float32"], compatible_runtimes=["ctypes-shared-library"],
        actions=_recipe_actions("upstream MLIR pass; AOT shared library") + [
            ActionSpec("aot.opt_level", "config", "LLVM opt level", "aot", params=[ParamSpec("opt_level", "int", min=0, max=3)])],
        measurement=MeasurementCaps(e2e="supported", e2e_boundary="host CPU: ctypes call into AOT .so (NOT an NPU)", units="ms"),
        environment={"runtime_features": ["host-aot"]})


class HostAOTCompiler(CompilerAdapter):
    def __init__(self):
        self.front = HostCompiler()

    def capture_baseline(self, workload):
        a = self.front.capture_baseline(workload)
        return Artifact(HOST_ID, "ir", payload=a.payload, ir_hashes=a.ir_hashes, provenance=a.provenance, effective_config={"opt_level": 2})

    def compile_candidate(self, workload, candidate, parent):
        parent = parent or self.capture_baseline(workload)
        mod, cfg = _apply(parent.ir if parent.ir is not None else parent.payload, candidate, dict(parent.effective_config))
        return Artifact(HOST_ID, "ir", payload=mod, ir_hashes={mod.stage: mod.sha256}, provenance=mod.provenance,
                        requested_config=dict(candidate.params), effective_config=cfg)

    def integrate_candidate(self, workload, artifact):
        mod = artifact.payload
        d = aot.workdir()
        try:
            o = aot.host_object_via_mlir(mod, d / f"{mod.func_name}.o", int(artifact.effective_config.get("opt_level", 2)))
            so = aot.link_shared([d / f"{mod.func_name}.o"], d / f"lib{mod.func_name}.so")
            lib = aot.AOTLibrary(d / f"lib{mod.func_name}.so", mod.func_name, aot.result_type(mod))
        except Exception as e:  # noqa: BLE001
            raise CompileFailure(f"AOT build failed: {e}") from e
        return Artifact(HOST_ID, "shared_library", payload=lib, ir=mod, ir_hashes=artifact.ir_hashes,
                        files={"object": str(d / f"{mod.func_name}.o"), "shared_library": str(d / f"lib{mod.func_name}.so")},
                        artifact_hashes={"object": o["sha256"], "shared_library": so["sha256"]},
                        requested_config=artifact.requested_config, effective_config=artifact.effective_config,
                        provenance=artifact.provenance + ["AOT: MLIR own LLVM -> x86-64 .o -> cc -shared"],
                        log=f"aot object {o['seconds']}s {o['bytes']} B")


class HostAOTRuntime(RuntimeAdapter):
    def check_artifact(self, artifact):
        if artifact.backend_id != HOST_ID or artifact.kind != "shared_library":
            raise Unsupported(f"{artifact.backend_id}/{artifact.kind} not loadable by {HOST_ID}")

    def prepare_session(self, artifact):
        self.check_artifact(artifact)
        return artifact.payload

    def execute_and_wait(self, session, inputs):
        return torch.from_numpy(session(*[t.detach().contiguous().numpy() for t in inputs]))


@register(HOST_ID)
def host_bundle(config):
    return AdapterBundle(host_manifest(), HostAOTCompiler(), HostAOTRuntime(), HostProfiler())


# ------------------------------------------------------------------ hexagon
HEX_ID = "llvm-aot-hexagon"


def hex_target():
    t = os.environ.get("NUX_LLVM_HEXAGON_TARGET", "hexagon-v73")
    return t


def hex_manifest():
    v = aot.llc_version()
    has = "hexagon" in aot.llc_targets()
    t = hex_target()
    note = f"system llc {v}; hexagon target {'present' if has else 'MISSING'}"
    return CapabilityManifest(
        backend_id=HEX_ID, kind="device", version=f"llc-{v}", compiler_profile=f"upstream-llvm-{v}/{t}",
        runtime_profile="none (needs Hexagon SDK runtime)", profiler_profile="none", input_stages=["linalg-on-tensors"],
        dtypes=["float32"], compatible_runtimes=["none (needs Hexagon SDK runtime)"],
        actions=_recipe_actions("upstream MLIR pass; codegen by upstream LLVM Hexagon backend") + [
            ActionSpec("llc.opt", "config", "llc optimization level", "aot", params=[ParamSpec("opt", "enum", choices=["-O0", "-O1", "-O2", "-O3"])])],
        measurement=MeasurementCaps(e2e="unavailable", e2e_boundary="codegen-only backend"),
        device_profile={"arch": t.split("-v")[-1] if "-v" in t else None},
        environment={"llc": aot.llc_path(), "llc_version": v, "hexagon_target": has, "runtime_features": []},
        notes=[note, "upstream LLVM codegen, not the Hexagon-MLIR pipeline; LLVM 18 supports up to hexagonv73"])


class HexAOTCompiler(CompilerAdapter):
    def __init__(self, env):
        self.env = env
        self.front = HostCompiler()

    def capture_baseline(self, workload):
        a = self.front.capture_baseline(workload)
        return self._build(a.payload, {"opt": "-O2"}, list(a.provenance), {}, workload)

    def compile_candidate(self, workload, candidate, parent):
        parent = parent or self.capture_baseline(workload)
        mod, cfg = _apply(parent.ir if parent.ir is not None else parent.payload, candidate, dict(parent.effective_config))
        return self._build(mod, cfg, list(mod.provenance), dict(candidate.params), workload)

    def _build(self, mod, cfg, prov, requested, workload):
        if not self.env.get("llc") or not self.env.get("hexagon_target"):
            return Artifact(HEX_ID, "ir", payload=mod, ir_hashes={mod.stage: mod.sha256}, provenance=prov,
                            effective_config=cfg, pending="toolchain_unavailable: llc with hexagon target not found")
        d = aot.workdir("nux_hex_")
        try:
            ll, counts = aot.downgrade_llvm_ir(aot.to_llvm_ir(mod))
            ll, n_mm = aot.expand_fmaxmin(ll)
            (d / "model.ll").write_text(ll)
            twin = aot.llc_compile(d / "model.ll", "x86_64", d / "twin_x86.o", cfg.get("opt", "-O2"))
            aot.link_shared([d / "twin_x86.o"], d / "libtwin.so")
            hexo = aot.llc_compile(d / "model.ll", hex_target(), d / f"{mod.func_name}_hexagon.o", cfg.get("opt", "-O2"))
        except Exception as e:  # noqa: BLE001
            raise CompileFailure(f"{type(e).__name__}: {str(e)[:600]}") from e
        twin_lib = aot.AOTLibrary(d / "libtwin.so", mod.func_name, aot.result_type(mod))
        return Artifact(HEX_ID, "object", payload=None, ir=mod, twin=twin_lib,
                        twin_note="x86-64 build of the same downgraded LLVM IR given to the Hexagon backend",
                        files={"llvm_ir": str(d / "model.ll"), "hexagon_object": str(d / f"{mod.func_name}_hexagon.o"),
                               "twin_x86_object": str(d / "twin_x86.o")},
                        ir_hashes={mod.stage: mod.sha256, "llvm-ir-downgraded": H.sha(ll)},
                        artifact_hashes={"hexagon_object": hexo["sha256"], "twin_x86_object": twin["sha256"]},
                        requested_config=requested, effective_config=cfg, target_compiled=True,
                        provenance=prov + [f"downgrade rewrites {counts}", f"llvm.maximum/minimum expanded: {n_mm}",
                                           hexo["cmd"]],
                        log=f"hexagon .o {hexo['bytes']} B in {hexo['seconds']}s ({hexo['file']})")


class HexAOTRuntime(RuntimeAdapter):
    def check_artifact(self, artifact):
        if artifact.backend_id != HEX_ID or not artifact.target_compiled:
            raise Unsupported("not a Hexagon object artifact")

    def prepare_session(self, artifact):
        self.check_artifact(artifact)
        raise DeviceUnavailable("Hexagon object built; executing it needs the Hexagon SDK runtime "
                                "(FastRPC skel / run_main_on_hexagon) and a device")

    def execute_and_wait(self, session, inputs):
        raise DeviceUnavailable("no Hexagon runtime")


class NoProfiler(ProfilerAdapter):
    def measure_e2e(self, runtime, session, inputs, policy):
        raise Unsupported("codegen-only backend")


@register(HEX_ID)
def hex_bundle(config):
    m = hex_manifest()
    return AdapterBundle(m, HexAOTCompiler(m.environment), HexAOTRuntime(), NoProfiler())
