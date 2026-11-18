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
import platform

import torch

from ..core.adapters import (
    AdapterBundle,
    Artifact,
    CompileFailure,
    CompilerAdapter,
    DeviceUnavailable,
    RuntimeAdapter,
    Unsupported,
)
from ..core.capability import ActionSpec, CapabilityManifest, ExecutionCaps, MeasurementCaps, ParamSpec
from ..engine.registry import register
from ..toolchain import aot, frontend, transforms
from ..toolchain import mlir_host as H
from ..toolchain.profilers import HostWallClockProfiler, NoProfiler


def _apply(parent_mod, candidate, cfg):
    mod = parent_mod
    if candidate.kind == "ir_recipe":
        mod = transforms.apply_ir_recipe(mod, candidate.action_id)  # Unsupported / CompileFailure contract
    elif candidate.kind == "config":
        cfg.update(candidate.params)
    elif candidate.kind != "baseline":
        raise Unsupported(candidate.kind)
    return mod, cfg


# --------------------------------------------------------------------- host
HOST_ID = "llvm-aot-host"
TWIN_NOTE = "x86-64 build of the same downgraded LLVM IR given to the Hexagon backend"


def _ir_meta(mod):
    return {"func_name": mod.func_name, "stage": mod.stage, "result_type": aot.result_type(mod)}


def _load_ir(path, meta):
    from pathlib import Path

    cm = meta["cache_meta"]
    return H.MLIRModule(Path(path).read_bytes(), cm["func_name"], cm["stage"], list(meta.get("provenance", [])))


def host_manifest():
    return CapabilityManifest(
        backend_id=HOST_ID, version="mlir-own-llvm", compiler_profile="mlir-llvm-aot/x86_64",
        runtime_profile="ctypes-shared-library", profiler_profile="host-wallclock", input_stages=["linalg-on-tensors"],
        dtypes=["float32", "float16"], compatible_runtimes=["ctypes-shared-library"],
        actions=transforms.recipe_actions("upstream MLIR pass; AOT shared library") + [
            ActionSpec("aot.opt_level", "config", "LLVM opt level", "aot", params=[ParamSpec("opt_level", "int", min=0, max=3)])],
        execution=ExecutionCaps(target_codegen=False, executes_on="host"),
        compile_fingerprint={"host_pipeline": H.HOST_PIPELINE, **aot.code_fingerprint(), "machine": platform.machine()},
        measurement=MeasurementCaps(e2e="supported", tier="host", e2e_boundary="host CPU: ctypes call into AOT .so (NOT an NPU)", units="ms"),
        environment={"runtime_features": ["host-aot"]})


class HostAOTCompiler(CompilerAdapter):
    def capture_baseline(self, workload):
        mod, _ = frontend.export(workload)
        return Artifact(HOST_ID, "ir", payload=mod, ir_hashes={mod.stage: mod.sha256}, provenance=list(mod.provenance),
                        effective_config={"opt_level": 2})

    def compile_candidate(self, workload, candidate, parent):
        parent = parent or self.capture_baseline(workload)
        mod, cfg = _apply(transforms.source_ir(parent), candidate, dict(parent.effective_config))
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
        (d / "source.mlirbc").write_bytes(mod.bytecode)
        return Artifact(HOST_ID, "shared_library", payload=lib, ir=mod, ir_hashes=artifact.ir_hashes,
                        files={"object": str(d / f"{mod.func_name}.o"), "shared_library": str(d / f"lib{mod.func_name}.so"),
                               "source_ir": str(d / "source.mlirbc")},
                        roles={"object": "intermediate", "shared_library": "primary", "source_ir": "source"},
                        cache_meta=_ir_meta(mod),
                        artifact_hashes={"object": o["sha256"], "shared_library": so["sha256"]},
                        requested_config=artifact.requested_config, effective_config=artifact.effective_config,
                        provenance=artifact.provenance + ["AOT: MLIR own LLVM -> x86-64 .o -> cc -shared"],
                        log=f"aot object {o['seconds']}s {o['bytes']} B")

    def load_artifact(self, workload, entry, purpose):
        p, m = entry["paths"], entry["meta"]
        if "shared_library" not in p or "source_ir" not in p:
            return None
        mod = _load_ir(p["source_ir"], m)
        lib = aot.AOTLibrary(p["shared_library"], mod.func_name, m["cache_meta"]["result_type"])
        return Artifact(HOST_ID, "shared_library", payload=lib, ir=mod, ir_hashes=m["ir_hashes"], files=dict(p),
                        artifact_hashes=m["artifact_hashes"], requested_config=m["requested_config"],
                        effective_config=m["effective_config"], provenance=m["provenance"], log="artifact-store hit")


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
    return AdapterBundle(host_manifest(), HostAOTCompiler(), HostAOTRuntime(),
                         HostWallClockProfiler("host CPU: ctypes call into AOT .so (not NPU)"))


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
        backend_id=HEX_ID, version=f"llc-{v}", compiler_profile=f"upstream-llvm-{v}/{t}",
        runtime_profile="none (needs Hexagon SDK runtime)", profiler_profile="none", input_stages=["linalg-on-tensors"],
        dtypes=["float32", "float16"], compatible_runtimes=["none (needs Hexagon SDK runtime)"],
        actions=transforms.recipe_actions("upstream MLIR pass; codegen by upstream LLVM Hexagon backend") + [
            ActionSpec("llc.opt", "config", "llc optimization level", "aot", params=[ParamSpec("opt", "enum", choices=["-O0", "-O1", "-O2", "-O3"])])],
        execution=ExecutionCaps(target_codegen=True, executes_on="none"),
        compile_fingerprint={"host_pipeline": H.HOST_PIPELINE, **aot.code_fingerprint(), "target": t,
                             "llc_args": aot.TARGETS.get(t)},
        measurement=MeasurementCaps(e2e="unavailable", tier="none", e2e_boundary="codegen-only backend"),
        device_profile={"arch": t.split("-v")[-1] if "-v" in t else None},
        environment={"llc": aot.llc_path(), "llc_version": v, "hexagon_target": has, "runtime_features": []},
        notes=[note, "upstream LLVM codegen, not the Hexagon-MLIR pipeline; LLVM 18 supports up to hexagonv73"])


class HexAOTCompiler(CompilerAdapter):
    def __init__(self, env):
        self.env = env

    def capture_baseline(self, workload):
        mod, _ = frontend.export(workload)
        return self._build(mod, {"opt": "-O2"}, list(mod.provenance), {}, workload)

    def compile_candidate(self, workload, candidate, parent):
        parent = parent or self.capture_baseline(workload)
        mod, cfg = _apply(transforms.source_ir(parent), candidate, dict(parent.effective_config))
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
        (d / "source.mlirbc").write_bytes(mod.bytecode)
        return Artifact(HEX_ID, "object", payload=None, ir=mod, twin=twin_lib, twin_note=TWIN_NOTE,
                        files={"llvm_ir": str(d / "model.ll"), "hexagon_object": str(d / f"{mod.func_name}_hexagon.o"),
                               "twin_x86_object": str(d / "twin_x86.o"), "twin_so": str(d / "libtwin.so"),
                               "source_ir": str(d / "source.mlirbc")},
                        roles={"llvm_ir": "intermediate", "hexagon_object": "primary", "twin_x86_object": "intermediate",
                               "twin_so": "verification", "source_ir": "source"},
                        cache_meta=_ir_meta(mod),
                        ir_hashes={mod.stage: mod.sha256, "llvm-ir-downgraded": H.sha(ll)},
                        artifact_hashes={"hexagon_object": hexo["sha256"], "twin_x86_object": twin["sha256"]},
                        requested_config=requested, effective_config=cfg, target_compiled=True,
                        provenance=prov + [f"downgrade rewrites {counts}", f"llvm.maximum/minimum expanded: {n_mm}",
                                           hexo["cmd"]],
                        log=f"hexagon .o {hexo['bytes']} B in {hexo['seconds']}s ({hexo['file']})")


    def load_artifact(self, workload, entry, purpose):
        p, m = entry["paths"], entry["meta"]
        if "hexagon_object" not in p or "source_ir" not in p:
            return None
        twin = None
        if "twin_so" in p:
            twin = aot.AOTLibrary(p["twin_so"], m["cache_meta"]["func_name"], m["cache_meta"]["result_type"])
        elif purpose == "evaluate":
            return None  # twin dropped by retention policy: re-evaluation needs it, so treat as a miss
        return Artifact(HEX_ID, "object", payload=None, ir=_load_ir(p["source_ir"], m), twin=twin,
                        twin_note=TWIN_NOTE if twin else "", files=dict(p), ir_hashes=m["ir_hashes"],
                        artifact_hashes=m["artifact_hashes"], requested_config=m["requested_config"],
                        effective_config=m["effective_config"], target_compiled=True, provenance=m["provenance"],
                        log="artifact-store hit")


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




@register(HEX_ID)
def hex_bundle(config):
    m = hex_manifest()
    return AdapterBundle(m, HexAOTCompiler(m.environment), HexAOTRuntime(), NoProfiler("codegen-only backend"))
