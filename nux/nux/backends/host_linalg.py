"""host-linalg-jit: target-independent reference backend.

PyTorch -> (frontend) linalg-on-tensors -> (transforms) IR recipes -> LLVM ->
MLIR ExecutionEngine on the host CPU. Provides `ir_valid` and `host_correct`
evidence for IR candidates that can later be fed to a vendor compiler.
Its timings are host CPU timings (source="host") and can never be used for
acceptance.
"""
from __future__ import annotations

import torch

from ..core.adapters import AdapterBundle, Artifact, CompileFailure, CompilerAdapter, RuntimeAdapter, Unsupported
from ..core.capability import ActionSpec, CapabilityManifest, ExecutionCaps, MeasurementCaps, ParamSpec
from ..engine.registry import register
from ..toolchain import frontend, transforms
from ..toolchain import mlir_host as H
from ..toolchain.profilers import HostWallClockProfiler

BACKEND_ID = "host-linalg-jit"


def manifest():
    v = frontend.frontend_version()
    actions = transforms.recipe_actions(f"upstream MLIR pass pipeline in {v}") + [
        ActionSpec("host.opt_level", "config", "LLVM opt level of the host JIT (host-only knob)", "host-lowering",
                   params=[ParamSpec("opt_level", "int", min=0, max=3)], evidence="MLIR ExecutionEngine"),
        ActionSpec("kernel.replace", "kernel", "replace an op with a custom kernel", "kernel",
                   support="unavailable", evidence="host reference path has no kernel binding ABI", tags=["kernel"]),
    ]
    return CapabilityManifest(
        backend_id=BACKEND_ID, version=v, compiler_profile=f"{v}/linalg-on-tensors",
        runtime_profile="mlir-execution-engine-host", profiler_profile="host-wallclock",
        input_stages=["pytorch", "linalg-on-tensors"], dtypes=["float32", "float16"], actions=actions,
        compatible_runtimes=["mlir-execution-engine-host"],
        execution=ExecutionCaps(target_codegen=False, executes_on="host"),
        compile_fingerprint={"host_pipeline": H.HOST_PIPELINE},
        measurement=MeasurementCaps(e2e="supported", tier="host", units="ms",
                                    e2e_boundary="host CPU: JIT invoke of the whole model (NOT an NPU)",
                                    notes="host timings are reference-only; never acceptance evidence"),
        environment={"runtime_features": ["host-jit"]})


class Compiler(CompilerAdapter):
    def capture_baseline(self, workload):
        mod, secs = frontend.export(workload)
        return Artifact(BACKEND_ID, "ir", payload=mod, ir_hashes={mod.stage: mod.sha256},
                        provenance=list(mod.provenance), effective_config={"opt_level": 2}, log=f"export {secs:.1f}s")

    def compile_candidate(self, workload, candidate, parent):
        parent = parent or self.capture_baseline(workload)
        mod, cfg = transforms.source_ir(parent), dict(parent.effective_config)
        prov = list(parent.provenance)
        if candidate.kind == "ir_recipe":
            mod = transforms.apply_ir_recipe(mod, candidate.action_id)
            prov = mod.provenance
        elif candidate.kind == "config":
            cfg.update(candidate.params)
            prov.append(f"config:{candidate.params}")
        elif candidate.kind != "baseline":
            raise Unsupported(candidate.kind)
        return Artifact(BACKEND_ID, "ir", payload=mod, ir_hashes={mod.stage: mod.sha256}, provenance=prov,
                        requested_config=dict(candidate.params), effective_config=cfg)

    def integrate_candidate(self, workload, artifact):
        try:
            jit = H.HostJIT(artifact.payload, opt_level=int(artifact.effective_config.get("opt_level", 2)))
        except Exception as e:  # noqa: BLE001
            raise CompileFailure(f"host lowering failed: {e}") from e
        return Artifact(BACKEND_ID, "jit_module", payload=jit, ir=artifact.payload, ir_hashes=artifact.ir_hashes,
                        provenance=artifact.provenance + [f"host pipeline: {H.HOST_PIPELINE}"],
                        requested_config=artifact.requested_config, effective_config=artifact.effective_config,
                        target_compiled=False, log=f"lower {jit.lower_s:.1f}s jit {jit.jit_s:.1f}s")


class Runtime(RuntimeAdapter):
    def check_artifact(self, artifact):
        if artifact.backend_id != BACKEND_ID or artifact.kind != "jit_module":
            raise Unsupported(f"artifact {artifact.backend_id}/{artifact.kind} not loadable by {BACKEND_ID}")

    def prepare_session(self, artifact):
        self.check_artifact(artifact)
        return artifact.payload

    def execute_and_wait(self, session, inputs):
        return torch.from_numpy(session(*[t.detach().contiguous().numpy() for t in inputs]))


@register(BACKEND_ID)
def bundle(config):
    return AdapterBundle(manifest(), Compiler(), Runtime(), HostWallClockProfiler("host CPU JIT invoke (not NPU)"))
