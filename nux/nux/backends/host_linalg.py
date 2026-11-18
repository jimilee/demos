"""host-linalg-jit: target-independent reference backend.

PyTorch -> torch-mlir linalg-on-tensors -> (IR recipes) -> LLVM -> MLIR
ExecutionEngine on the host CPU. Provides `ir_valid` and `host_correct`
evidence for IR candidates that can later be fed to a vendor compiler.
Its timings are host CPU timings (source="host") and can never be used for
acceptance.
"""
from __future__ import annotations

import statistics
import time

import torch

from .. import mlir_host as H
from ..adapters import AdapterBundle, Artifact, CompilerAdapter, CompileFailure, ProfilerAdapter, RuntimeAdapter, Unsupported
from ..capability import ActionSpec, CapabilityManifest, ExecutionCaps, MeasurementCaps, ParamSpec
from ..registry import register
from ..schema import Measurement

BACKEND_ID = "host-linalg-jit"


def _torch_mlir_version():
    try:
        from importlib.metadata import version

        return version("torch-mlir")
    except Exception:  # noqa: BLE001
        return "unknown"


def manifest():
    v = _torch_mlir_version()
    actions = [ActionSpec(f"recipe.{r.recipe_id}", "ir_recipe", r.meaning, r.stage, numeric_change=r.numeric_change,
                          support="supported", validation="unvalidated",
                          evidence=f"upstream MLIR pass pipeline '{r.pipeline}' in torch-mlir {v}", tags=list(r.tags))
               for r in H.RECIPES.values()]
    actions += [
        ActionSpec("host.opt_level", "config", "LLVM opt level of the host JIT (host-only knob)", "host-lowering",
                   params=[ParamSpec("opt_level", "int", min=0, max=3)], evidence="MLIR ExecutionEngine"),
        ActionSpec("kernel.replace", "kernel", "replace an op with a custom kernel", "kernel",
                   support="unavailable", evidence="host reference path has no kernel binding ABI"),
    ]
    return CapabilityManifest(
        backend_id=BACKEND_ID, version=f"torch-mlir-{v}",
        compiler_profile=f"torch-mlir-{v}/linalg-on-tensors", runtime_profile="mlir-execution-engine-host",
        profiler_profile="host-wallclock", input_stages=["pytorch", "linalg-on-tensors"], dtypes=["float32"],
        actions=actions, compatible_runtimes=["mlir-execution-engine-host"],
        execution=ExecutionCaps(target_codegen=False, executes_on="host"),
        measurement=MeasurementCaps(e2e="supported", tier="host", e2e_boundary="host CPU: JIT invoke of the whole model (NOT an NPU)",
                                    units="ms", notes="host timings are reference-only; never acceptance evidence"),
        environment={"runtime_features": ["host-jit"]})


class Compiler(CompilerAdapter):
    def __init__(self):
        self._baseline_cache = {}

    def capture_baseline(self, workload):
        key = workload.workload_id
        if key not in self._baseline_cache:
            t = time.perf_counter()
            mod = H.export(workload.module(), workload.example_inputs(), workload.func_name)
            self._baseline_cache[key] = (mod, time.perf_counter() - t)
        mod, secs = self._baseline_cache[key]
        ok, why = H.verify(mod)
        if not ok:
            raise CompileFailure(f"baseline IR does not verify: {why}")
        return Artifact(BACKEND_ID, "ir", payload=mod, ir_hashes={mod.stage: mod.sha256},
                        provenance=list(mod.provenance), effective_config={"opt_level": 2}, log=f"export {secs:.1f}s")

    def compile_candidate(self, workload, candidate, parent):
        parent = parent or self.capture_baseline(workload)
        mod = parent.ir if parent.ir is not None else parent.payload
        cfg = dict(parent.effective_config)
        prov = list(parent.provenance)
        if candidate.kind == "ir_recipe":
            r = H.RECIPES[candidate.action_id.split(".", 1)[1]]
            ok, why = H.recipe_applicable(r, mod)
            if not ok:
                raise Unsupported(f"recipe precondition: {why}")
            try:
                mod = H.apply_recipe(mod, r)
            except Exception as e:  # noqa: BLE001
                raise CompileFailure(f"{r.recipe_id}: {e}") from e
            prov = mod.provenance
        elif candidate.kind == "config":
            cfg.update(candidate.params)
            prov.append(f"config:{candidate.params}")
        elif candidate.kind != "baseline":
            raise Unsupported(candidate.kind)
        ok, why = H.verify(mod)
        if not ok:
            raise CompileFailure(why)
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


class Profiler(ProfilerAdapter):
    def measure_e2e(self, runtime, session, inputs, policy):
        for _ in range(policy.warmup):
            runtime.execute_and_wait(session, inputs)
        blocks = []
        for _ in range(policy.pairs):
            b = []
            for _ in range(policy.requests_per_block):
                t = time.perf_counter()
                runtime.execute_and_wait(session, inputs)
                b.append((time.perf_counter() - t) * 1e3)
            blocks.append(b)
        return Measurement("host", "host CPU JIT invoke (not NPU)", "ms", blocks,
                           missing=["npu time", "dsp counters"],
                           conditions={"note": "host wall clock, median " + str(statistics.median(blocks[0]))})


@register(BACKEND_ID)
def bundle(config):
    return AdapterBundle(manifest(), Compiler(), Runtime(), Profiler())
