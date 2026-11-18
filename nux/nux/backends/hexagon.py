"""hexagon-mlir: Qualcomm Hexagon-MLIR adapter bundle (first real vendor backend).

Facts below were read from the hexagon-mlir source at rev 9ec15e8 and are
recorded as `evidence`; every action starts `unvalidated` until it runs on a
device with this toolchain.

Paths used (when the toolchain is importable):
  compile    triton._C.libtriton.qcom_hexagon_backend.translate_linalg_to_obj
             via TorchMLIRHexagonLauncher.mlir_to_obj(mlir_mod, options)
             -> object code for the target arch (state target_compiled)
  correctness  TorchMLIRHexagonLauncher.run_torch_mlir(path, inputs, func_name,
             options=...) -> device outputs (state device_correct)
  E2E        NOT AVAILABLE: the public launcher runs one standalone process
             per call (adb push / run_main_on_hexagon / pull), with no session
             reuse and no request-level timer matching the plan's §9.1
             boundary. A device runner with session reuse is an adapter
             prerequisite; until it exists measure_e2e raises Unsupported and
             no Hexagon candidate can reach `accepted`.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import torch

from .. import frontend, transforms
from .. import mlir_host as H
from ..adapters import (AdapterBundle, Artifact, CompilerAdapter, CompileFailure, DeviceUnavailable, ProfilerAdapter,
                        RuntimeAdapter, RuntimeFailure, Unsupported)
from ..capability import ActionSpec, CapabilityManifest, ExecutionCaps, MeasurementCaps, ParamSpec
from ..registry import register

BACKEND_ID = "hexagon-mlir"
SOURCE_REV = "9ec15e8"
EVID = f"qcom_hexagon_backend/backend/hexagon_options.py @ {SOURCE_REV}"

# (option, type, description, support, tags)  -- HexagonOptions fields relevant to the torch-mlir path
OPTIONS = [
    ("fusion", "bool", "linalg fusion", "supported", ["fusion"]),
    ("fusionAllowRecompute", "bool", "fusion may recompute producers", "supported", ["fusion"]),
    ("fusionDoMultiUse", "bool", "fuse producers with multiple uses", "supported", ["fusion"]),
    ("enableVectorization", "bool", "HVX vectorization", "supported", ["vector", "codegen"]),
    ("enableVTCMTiling", "bool", "tile linalg.generic into VTCM", "supported", ["tiling", "memory"]),
    ("enableConvertToHexagonmem", "bool", "rewrite memref alloc/copy to hexagonmem", "supported", ["memory"]),
    ("enableHexagonmemCopyToDMA", "bool", "hexagonmem copies via DMA", "supported", ["memory"]),
    ("enableDoubleBuffering", "bool", "double buffering", "experimental", ["memory"]),
    ("enableMultiThreading", "bool", "linalg-generic multithreading (FormVirtualThreads)", "supported", ["threading"]),
    ("enableSplitReduction", "bool", "split-reduction", "supported", ["reduction"]),
    ("enableHVXInlining", "bool", "HVX inlining", "supported", ["codegen"]),
    ("enableSCFLoopUnroll", "bool", "scf loop unroll", "supported", ["codegen"]),
    ("enableConversionToFp16", "bool", "convert to fp16", "unavailable", ["precision"]),  # violates fixed dtype contract
]


def _discover():
    env = os.environ
    info = {"HEXAGON_ARCH_VERSION": env.get("HEXAGON_ARCH_VERSION"), "HEXAGON_SDK_ROOT": env.get("HEXAGON_SDK_ROOT"),
            "HEXAGON_TOOLS": env.get("HEXAGON_TOOLS"), "HEXKL_ROOT": env.get("HEXKL_ROOT"),
            "ANDROID_SERIAL": env.get("ANDROID_SERIAL"), "runtime_features": []}
    os.environ.setdefault("TRITON_BACKENDS_IN_TREE", "1")  # ignore stale triton dist-info entry points
    try:
        from triton.backends.qcom_hexagon_backend.hexagon_options import HexagonOptions

        info["options_fields"] = sorted(HexagonOptions().__dict__)
        info["toolchain_importable"] = True
    except Exception as e:  # noqa: BLE001
        info["toolchain_importable"] = False
        info["toolchain_error"] = f"{type(e).__name__}: {e}"
    adb = shutil.which("adb")
    info["adb"] = adb
    if adb:
        cmd = [adb] + (["-H", env["ANDROID_HOST"]] if env.get("ANDROID_HOST") else []) + \
              (["-s", env["ANDROID_SERIAL"]] if env.get("ANDROID_SERIAL") else []) + ["get-state"]
        try:
            info["device_state"] = subprocess.run(cmd, capture_output=True, text=True, timeout=15).stdout.strip()
        except Exception as e:  # noqa: BLE001
            info["device_state"] = f"error: {e}"
    if info.get("device_state") == "device" and info["toolchain_importable"]:
        info["runtime_features"].append("standalone-launcher")
    return info


def manifest():
    env = _discover()
    fields = set(env.get("options_fields") or [])
    actions = []
    for name, typ, desc, sup, tags in OPTIONS:
        if fields and name not in fields:
            sup = "unavailable"
            ev = f"option missing from the importable HexagonOptions ({EVID} lists it)"
        elif name == "enableConversionToFp16":
            ev = "excluded: dtype/quantization must be preserved (plan §2)"
        else:
            ev = EVID
        actions.append(ActionSpec(f"opt.{name}", "config", desc, "compile-options",
                                  params=[ParamSpec("value", typ)], support=sup, evidence=ev, tags=tags))
    actions.append(ActionSpec("opt.tileSizes", "config", "user tile sizes (backend notes: debugging)", "compile-options",
                              params=[ParamSpec("value", "str")], support="experimental", evidence=EVID, tags=["tiling"]))
    actions.append(ActionSpec("path.hexkl_matmul", "lowering_path", "lower matmul via HexKL (HMX)", "compile-options",
                              params=[ParamSpec("mode", "enum", choices=["micro", "macro"])], support="experimental",
                              constraints={"dtypes": ["float16"], "arch_exclude": ["81"]},
                              evidence="MatmulToHexKLPass.cpp: f16 inputs only; user-guide: not valid on v81; "
                                       "macro mode needs constant weights", tags=["library_matmul", "kernel"]))
    # external input-IR recipes, applied before the vendor compiler
    actions += transforms.recipe_actions("upstream MLIR pass on the input IR; target effect unvalidated")
    actions.append(ActionSpec("kernel.triton_replace", "kernel", "replace a model op with a Triton kernel", "triton-kernel",
                              support="unavailable", requires_runtime=["kernel-binding"],
                              evidence="no proven symbol/ABI binding between Triton objects and the torch-mlir model path yet",
                              tags=["kernel"]))
    actions.append(ActionSpec("kernel.c_replace", "kernel", "replace a model op with a C/C++ kernel", "c-kernel",
                              support="unavailable", requires_runtime=["kernel-binding"], evidence="ABI integration test not done",
                              tags=["kernel"]))
    return CapabilityManifest(
        backend_id=BACKEND_ID, version=f"hexagon-mlir@{SOURCE_REV}",
        compiler_profile="hexagon-mlir/torch-mlir-linalg", runtime_profile="hexagon-standalone-launcher",
        profiler_profile="none", input_stages=["linalg-on-tensors"], dtypes=["float32", "float16"],
        actions=actions, compatible_runtimes=["hexagon-standalone-launcher"],
        execution=ExecutionCaps(target_codegen=True, executes_on="device"),
        measurement=MeasurementCaps(
            e2e="unavailable", tier="device", e2e_boundary="(none yet) plan §9.1 request boundary needs a session-reusing device runner",
            notes="launcher Perf: field is an in-process DSP loop average, not the plan's E2E metric"),
        device_profile={"arch": env.get("HEXAGON_ARCH_VERSION"), "serial_set": bool(env.get("ANDROID_SERIAL")),
                        "state": env.get("device_state")},
        environment=env,
        notes=["actions are unvalidated until compiled and run with this toolchain on this device"])


def _options(cfg):
    from triton.backends.qcom_hexagon_backend.hexagon_options import HexagonOptions

    o = HexagonOptions().__dict__.copy()
    o.update(cfg)
    return o


class Compiler(CompilerAdapter):
    def __init__(self, env):
        self.env = env

    def capture_baseline(self, workload):
        mod, _ = frontend.export(workload)  # shared, vendor-neutral front end
        return self._target_compile(mod, {}, list(mod.provenance), {})

    def compile_candidate(self, workload, candidate, parent):
        parent = parent or self.capture_baseline(workload)
        mod = transforms.source_ir(parent)
        cfg, prov = dict(parent.effective_config), list(parent.provenance)
        if candidate.kind == "ir_recipe":
            mod = transforms.apply_ir_recipe(mod, candidate.action_id)
            prov = mod.provenance
        elif candidate.kind == "config":
            cfg[candidate.action_id.split(".", 1)[1]] = candidate.params["value"]
        elif candidate.kind == "lowering_path":
            cfg["enableHexKL"] = True
            cfg["hexKLMode"] = candidate.params.get("mode", "micro")
        elif candidate.kind != "baseline":
            raise Unsupported(candidate.kind)
        return self._target_compile(mod, cfg, prov, candidate.params)

    def _target_compile(self, mod, cfg, prov, requested):
        why = None
        if not self.env.get("toolchain_importable"):
            why = f"toolchain_unavailable: hexagon backend not importable ({self.env.get('toolchain_error')})"
        elif not self.env.get("HEXAGON_ARCH_VERSION"):
            why = "toolchain_unavailable: HEXAGON_ARCH_VERSION not set"
        if why:  # the vendor-neutral input IR is still real evidence; target stages stay pending
            return Artifact(BACKEND_ID, "ir", payload=mod, ir_hashes={mod.stage: mod.sha256}, provenance=prov,
                            requested_config=requested, effective_config=dict(cfg), pending=why)
        from triton._C.libtriton import ir as tir, qcom_hexagon_backend as qhb  # noqa: F401
        from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher

        work = Path(tempfile.mkdtemp(prefix="nux_hex_"))
        src = work / f"{mod.func_name}.mlir"
        src.write_text(mod.text())  # text, not bytecode: torch-mlir and hexagon LLVM revisions may differ
        options = _options(cfg)
        try:
            ctx = tir.context()
            qhb.load_dialects(ctx)
            m = qhb.parse_mlir_module_from_file(str(src), ctx)
            objs = TorchMLIRHexagonLauncher().mlir_to_obj(m, options)
        except Exception as e:  # noqa: BLE001
            raise CompileFailure(f"translate_linalg_to_obj: {type(e).__name__}: {e}") from e
        files, hashes = {"input_ir": str(src)}, {}
        for i, b in enumerate(objs):
            p = work / f"{mod.func_name}_{i}.o"
            p.write_bytes(bytes(b))
            files[f"obj{i}"] = str(p)
            hashes[f"obj{i}"] = H.sha(bytes(b))
        (work / "source.mlirbc").write_bytes(mod.bytecode)
        files["source_ir"] = str(work / "source.mlirbc")
        # input_ir is read by the launcher at run time (Runtime.prepare_session), so it is primary, not intermediate
        roles = {n: ("source" if n == "source_ir" else "primary") for n in files}
        return Artifact(BACKEND_ID, "object", payload=mod, files=files, ir_hashes={mod.stage: mod.sha256},
                        artifact_hashes=hashes, requested_config=requested, roles=roles,
                        cache_meta={"func_name": mod.func_name, "stage": mod.stage},
                        effective_config={k: options[k] for k in sorted(options) if k in {o[0] for o in OPTIONS} or k in cfg},
                        provenance=prov + [f"translate_linalg_to_obj(arch=v{self.env['HEXAGON_ARCH_VERSION']})"],
                        target_compiled=True)

    def load_artifact(self, workload, entry, purpose):
        """UNVERIFIED in this environment (no toolchain): rehydrate a target-compiled artifact."""
        p, m = entry["paths"], entry["meta"]
        if "source_ir" not in p or not any(n.startswith("obj") for n in p) or "input_ir" not in p:
            return None
        cm = m["cache_meta"]
        mod = H.MLIRModule(Path(p["source_ir"]).read_bytes(), cm["func_name"], cm["stage"], list(m.get("provenance", [])))
        return Artifact(BACKEND_ID, "object", payload=mod, files=dict(p), ir_hashes=m["ir_hashes"],
                        artifact_hashes=m["artifact_hashes"], requested_config=m["requested_config"],
                        effective_config=m["effective_config"], provenance=m["provenance"], target_compiled=True,
                        log="artifact-store hit")


class Runtime(RuntimeAdapter):
    def __init__(self, env):
        self.env = env

    def check_artifact(self, artifact):
        if artifact.backend_id != BACKEND_ID or not artifact.target_compiled:
            raise Unsupported(f"artifact {artifact.backend_id}/{artifact.kind} is not a Hexagon target artifact")

    def prepare_session(self, artifact):
        self.check_artifact(artifact)
        if self.env.get("device_state") != "device":
            raise DeviceUnavailable(f"adb state: {self.env.get('device_state')!r} (ANDROID_SERIAL={self.env.get('ANDROID_SERIAL')!r})")
        return {"ir": artifact.files["input_ir"], "func": artifact.payload.func_name, "options": artifact.effective_config}

    def execute_and_wait(self, session, inputs):
        """One-shot launcher run: correctness only (compiles, pushes, runs one process, pulls)."""
        from triton.backends.qcom_hexagon_backend.torch_mlir_hexagon_launcher import TorchMLIRHexagonLauncher

        try:
            out = TorchMLIRHexagonLauncher().run_torch_mlir(session["ir"], inputs, session["func"],
                                                              options=_options(session["options"]))
        except SystemExit as e:  # the executor calls sys.exit(1) on a failed adb/link step
            raise RuntimeFailure(f"launcher exited ({e.code})") from e
        return torch.as_tensor(out[0])


class Profiler(ProfilerAdapter):
    def measure_e2e(self, runtime, session, inputs, policy):
        raise Unsupported("E2E measurement unavailable: no session-reusing device runner for the plan §9.1 boundary yet")


@register(BACKEND_ID)
def bundle(config):
    m = manifest()
    return AdapterBundle(m, Compiler(m.environment), Runtime(m.environment), Profiler())
