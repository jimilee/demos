"""Two capability MOCKS for the A3 structure check (plan §14 A3).

  mock-mlir  exposes config + IR recipe + kernel actions (an MLIR-style backend)
  mock-sdk   exposes only model-SDK options (no IR, no custom kernels)

Outputs are computed for real with the framework model (so correctness checks
run), but timings are SYNTHETIC (Measurement.source="synthetic") and the
bundle kind is "mock": the evaluator can never accept on them.
Fault injection params (tests only): fail="compile"|"runtime", perturb=<float>.
"""
from __future__ import annotations

import random

from ..adapters import AdapterBundle, Artifact, CompilerAdapter, CompileFailure, ProfilerAdapter, RuntimeAdapter, RuntimeFailure, Unsupported
from ..capability import ActionSpec, CapabilityManifest, ExecutionCaps, MeasurementCaps, ParamSpec
from ..registry import register
from ..schema import Measurement

FAULT = [ParamSpec("fail", "enum", choices=["none", "compile", "runtime"]), ParamSpec("perturb", "str"),
         ParamSpec("synthetic_speedup", "str")]


def _manifest(bid, actions, stages):
    return CapabilityManifest(
        backend_id=bid, version="mock-1", compiler_profile=f"{bid}-compiler",
        runtime_profile=f"{bid}-runtime", profiler_profile=f"{bid}-synthetic", input_stages=stages,
        dtypes=["float32"], actions=actions, compatible_runtimes=[f"{bid}-runtime"],
        execution=ExecutionCaps(target_codegen=False, executes_on="host"),  # outputs: real framework compute on host
        measurement=MeasurementCaps(e2e="supported", tier="synthetic", e2e_boundary="SYNTHETIC", units="us",
                                    notes="synthetic timings for controller tests only"),
        environment={"runtime_features": ["mock-session"]})


def mlir_manifest():
    return _manifest("mock-mlir", [
        ActionSpec("mock.tile", "config", "tile size", "compile-options",
                   params=[ParamSpec("tile", "int", choices=[16, 32, 64])] + FAULT),
        ActionSpec("recipe.fold-unit-extent-dims", "ir_recipe", "drop unit dims", "linalg-on-tensors", params=FAULT),
        ActionSpec("kernel.replace", "kernel", "custom kernel", "kernel", params=FAULT, support="experimental"),
        ActionSpec("path.library_matmul", "lowering_path", "library matmul", "compile-options",
                   support="unavailable", evidence="mock: not provided"),
    ], ["linalg-on-tensors", "kernel"])


def sdk_manifest():
    return _manifest("mock-sdk", [
        ActionSpec("sdk.opt_level", "config", "SDK optimization level", "model-sdk",
                   params=[ParamSpec("level", "int", min=0, max=3)] + FAULT),
        ActionSpec("recipe.fold-unit-extent-dims", "ir_recipe", "drop unit dims", "linalg-on-tensors",
                   support="unavailable", evidence="mock SDK takes a model file; no IR access"),
        ActionSpec("kernel.replace", "kernel", "custom kernel", "kernel", support="unavailable",
                   evidence="mock SDK has no custom kernel API"),
    ], ["model-file"])


class Compiler(CompilerAdapter):
    def __init__(self, bid):
        self.bid = bid

    def capture_baseline(self, workload):
        return Artifact(self.bid, "sdk_model", payload={"params": {}}, effective_config={})

    def compile_candidate(self, workload, candidate, parent):
        p = dict((parent.payload if parent else {"params": {}})["params"])
        if candidate.params.get("fail") == "compile":
            raise CompileFailure("injected compile failure")
        p.update(candidate.params)
        return Artifact(self.bid, "sdk_model", payload={"params": p}, requested_config=dict(candidate.params),
                        effective_config=p, target_compiled=True, provenance=[candidate.action_id])


class Runtime(RuntimeAdapter):
    def __init__(self, bid, workload_getter):
        self.bid = bid

    def check_artifact(self, artifact):
        if artifact.backend_id != self.bid:
            raise Unsupported(f"artifact for {artifact.backend_id} cannot load on {self.bid}")

    def prepare_session(self, artifact):
        self.check_artifact(artifact)
        return artifact.payload["params"]

    def bind_workload(self, workload):
        self.workload = workload

    def execute_and_wait(self, session, inputs):
        if session.get("fail") == "runtime":
            raise RuntimeFailure("injected runtime failure")
        y = self.workload.reference(inputs)
        if session.get("perturb"):
            y = y + float(session["perturb"])
        return y


class Profiler(ProfilerAdapter):
    def measure_e2e(self, runtime, session, inputs, policy):
        rng = random.Random(hash(tuple(sorted(map(str, session.items())))) & 0xFFFF)
        base = 1000.0 / float(session.get("synthetic_speedup") or 1.0)
        blocks = [[base * (1 + rng.uniform(-0.01, 0.01)) for _ in range(policy.requests_per_block)]
                  for _ in range(policy.pairs)]
        return Measurement("synthetic", "SYNTHETIC mock boundary", "us", blocks, missing=["everything real"])


def _bundle(bid, man):
    rt = Runtime(bid, None)
    return AdapterBundle(man, Compiler(bid), rt, Profiler())


@register("mock-mlir")
def mlir_bundle(config):
    return _bundle("mock-mlir", mlir_manifest())


@register("mock-sdk")
def sdk_bundle(config):
    return _bundle("mock-sdk", sdk_manifest())

