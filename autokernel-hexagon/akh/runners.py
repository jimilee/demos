"""Runners: execute a candidate's Launch list on a backend.

HexagonRunner     real hexagon-mlir HexagonDriver -> adb -> run_main_on_hexagon
                  on the Snapdragon cDSP. The only runner that produces
                  performance numbers.
HostInterpRunner  Triton interpreter on the host CPU (TRITON_INTERPRET=1).
                  Checks kernel *logic* (masks, splits, reductions) only.
                  Never produces DSP/NPU timings.

Neither runner uses ``tensor.device`` to decide where code ran: the Hexagon
driver keeps CPU torch tensors as host buffers. Device execution evidence is
(a) the executor log shows the run_main_on_hexagon step and "Ran
successfully", (b) a perf.txt pulled from the device with Result:Pass and the
expected kernel name, and (c) output buffers pre-filled with a NaN sentinel
were overwritten with what the launcher read back from the device.
"""
from __future__ import annotations

import contextlib
import glob
import io
import os
import subprocess
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import hexagon_timer as ht

# failure classes (also used in results.tsv `status`)
OK = "ok"
COMPILE_FAIL = "compile_fail"
LINK_FAIL = "link_fail"
DEVICE_UNAVAILABLE = "device_unavailable"
DEVICE_FAIL = "device_fail"
TIMING_PARSE_FAIL = "timing_parse_fail"
ACCURACY_FAIL = "accuracy_fail"
RUNTIME_ERROR = "runtime_error"


@dataclass
class LaunchResult:
    name: str
    status: str = OK
    error: str = ""
    compile_s: float | None = None  # JIT compile (first time only)
    host_call_ms: float | None = None  # python call -> results back (link+push+run+pull)
    host_run_ms: float | None = None  # adb shell run_main_on_hexagon wall time
    host_push_ms: float | None = None
    host_pull_ms: float | None = None
    dsp_avg_us: float | None = None  # steady-state per call, on DSP
    dsp_first_call_us: float | None = None
    dsp_avg_pcycles: float | None = None
    timer: str = "missing"
    evidence: dict = field(default_factory=dict)
    ir_hash: str | None = None
    ir_markers: dict = field(default_factory=dict)
    log_path: str | None = None


@dataclass
class RunResult:
    status: str
    launches: list
    error: str = ""

    def total(self, attr):
        vals = [getattr(l, attr) for l in self.launches]
        if not vals or any(v is None for v in vals):
            return None  # missing in any launch -> missing overall (excluded from ranking)
        return sum(vals)


def _first_bad(launch_results):
    for lr in launch_results:
        if lr.status != OK:
            return lr.status, lr.error
    return OK, ""


class HostInterpRunner:
    name = "host-interp"
    produces_perf = False

    def __init__(self, log_dir: Path):
        if os.environ.get("TRITON_INTERPRET") != "1":
            raise RuntimeError("HostInterpRunner requires TRITON_INTERPRET=1 before importing triton")
        self.log_dir = Path(log_dir)

    def run(self, launches, tag: str, measure: bool = False) -> RunResult:
        out = []
        for l in launches:
            lr = LaunchResult(l.name)
            try:
                l.fn[l.grid](*l.args, **l.consts)
                lr.evidence = {"executed_by": "triton-interpreter(host CPU)"}
            except Exception as e:  # noqa: BLE001
                lr.status, lr.error = RUNTIME_ERROR, f"{type(e).__name__}: {e}"
                out.append(lr)
                break
            out.append(lr)
        st, err = _first_bad(out)
        return RunResult(st, out, err)


def adb_state(host: str | None, serial: str | None) -> tuple[bool, str]:
    cmd = ["adb"]
    if host:
        cmd += ["-H", host]
    if serial:
        cmd += ["-s", serial]
    cmd += ["get-state"]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    except FileNotFoundError:
        return False, "adb not found on PATH"
    except subprocess.TimeoutExpired:
        return False, "adb get-state timed out"
    ok = p.returncode == 0 and p.stdout.strip() == "device"
    return ok, (p.stdout + p.stderr).strip()


class HexagonRunner:
    name = "hexagon"
    produces_perf = True

    def __init__(self, log_dir: Path, iterations: int = 100, patch_timer: bool = True):
        missing = [v for v in ("HEXAGON_SDK_ROOT", "HEXAGON_TOOLS", "HEXAGON_ARCH_VERSION",
                               "HEXAGON_MLIR_ROOT", "ANDROID_SERIAL") if not os.environ.get(v)]
        if missing:
            raise RuntimeError(f"missing environment variables for Hexagon execution: {missing}")
        if os.environ.get("RUN_ON_SIM", "0") != "0":
            raise RuntimeError("RUN_ON_SIM is set: simulator runs are not device measurements")
        import triton
        from triton.backends.qcom_hexagon_backend.driver import HexagonDriver

        triton.runtime.driver.set_active(HexagonDriver())
        self.patch_timer = patch_timer
        if patch_timer:
            ht.install_timer_patch()
        self.iterations = int(iterations)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.dump_dir = self.log_dir / "hexagon_dump"
        self.dump_dir.mkdir(exist_ok=True)
        os.environ["HEXAGON_MLIR_DUMP_DIR"] = str(self.dump_dir)
        self._compiled = set()

    def check_device(self):
        return adb_state(os.environ.get("ANDROID_HOST"), os.environ.get("ANDROID_SERIAL"))

    def _opts(self, l):
        opts = dict(l.hexagon_opts)
        opts["iterations"] = self.iterations
        return opts

    def run(self, launches, tag: str, measure: bool = True) -> RunResult:
        ok, msg = self.check_device()
        if not ok:
            return RunResult(DEVICE_UNAVAILABLE, [LaunchResult(l.name, DEVICE_UNAVAILABLE, msg) for l in launches], msg)
        results = []
        for i, l in enumerate(launches):
            lr = self._run_one(l, f"{tag}_{i}_{l.name}")
            results.append(lr)
            if lr.status != OK:
                break
        st, err = _first_bad(results)
        return RunResult(st, results, err)

    def _run_one(self, l, tag) -> LaunchResult:
        lr = LaunchResult(l.name)
        log_path = self.log_dir / f"{tag}.log"
        lr.log_path = str(log_path)
        opts = self._opts(l)
        buf = io.StringIO()

        # 1) JIT compile, separated from execution
        key = (id(l.fn), tuple(sorted((k, repr(v)) for k, v in l.consts.items())),
               tuple(sorted((k, repr(v)) for k, v in opts.items())), l.grid)
        compiled = None
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                t0 = time.perf_counter()
                compiled = l.fn.warmup(*l.args, grid=l.grid, **l.consts, **opts)
                if key not in self._compiled:
                    lr.compile_s = time.perf_counter() - t0
                    self._compiled.add(key)
        except (Exception, SystemExit) as e:  # noqa: BLE001
            lr.status, lr.error = COMPILE_FAIL, f"{type(e).__name__}: {e}"
            buf.write(traceback.format_exc())
            log_path.write_text(buf.getvalue())
            return lr
        lr.ir_hash, lr.ir_markers = ir_fingerprint(compiled)

        # 2) sentinel outputs -> proves the result came back from the device
        for t in l.outputs:
            t.fill_(float("nan"))
        before = set(glob.glob(str(self.dump_dir / "*")))

        # 3) execute: link .so, adb push, run_main_on_hexagon, adb pull
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                t0 = time.perf_counter()
                l.fn[l.grid](*l.args, **l.consts, **opts)
                lr.host_call_ms = (time.perf_counter() - t0) * 1e3
        except SystemExit as e:  # executor calls sys.exit(1) on a failed step
            text = buf.getvalue()
            lr.status = LINK_FAIL if "Compiling shared object file" in text and "Running the kernel on device" not in text else DEVICE_FAIL
            lr.error = f"SystemExit({e.code}): " + _last_error_line(text)
        except Exception as e:  # noqa: BLE001
            lr.status, lr.error = RUNTIME_ERROR, f"{type(e).__name__}: {e}"
            buf.write(traceback.format_exc())
        text = buf.getvalue()
        log_path.write_text(text)
        if lr.status != OK:
            return lr

        exe = ht.parse_executor_log(text)
        lr.host_run_ms = exe["run_s"] * 1e3 if exe["run_s"] is not None else None
        lr.host_push_ms = exe["push_s"] * 1e3
        lr.host_pull_ms = exe["pull_s"] * 1e3

        new_dirs = sorted(set(glob.glob(str(self.dump_dir / "*"))) - before)
        perf_files = [p for d in new_dirs for p in glob.glob(os.path.join(d, "*_perf.txt"))]
        rep = ht.parse_perf_text(Path(perf_files[-1]).read_text()) if perf_files else ht.PerfReport()
        lr.timer = rep.timer
        if rep.timer == ht.AK_TIMER_VERSION:
            lr.dsp_avg_us = rep.steady_avg_us
            lr.dsp_first_call_us = ht.ak_float(rep, "FirstCall_us")
            lr.dsp_avg_pcycles = ht.ak_float(rep, "Steady_avg_pcycles")
        elif rep.timer == "stock":
            lr.dsp_avg_us = rep.steady_avg_us  # integer-us average incl. first call

        sentinel_cleared = all(not torch.isnan(t).all().item() for t in l.outputs)
        lr.evidence = {
            "ran_on_device_log": exe["ran_on_device"],
            "perf_file": perf_files[-1] if perf_files else None,
            "perf_result": rep.result,
            "perf_name": rep.name,
            "sentinel_overwritten": sentinel_cleared,
            "dump_dirs": new_dirs,
        }
        if not (exe["ran_on_device"] and sentinel_cleared):
            lr.status, lr.error = DEVICE_FAIL, f"no device-execution evidence: {lr.evidence}"
        elif rep.result not in (None, "Pass"):
            lr.status, lr.error = DEVICE_FAIL, f"device report Result={rep.result}"
        elif lr.dsp_avg_us is None:
            lr.status, lr.error = TIMING_PARSE_FAIL, f"could not parse DSP time from {perf_files}"
        return lr


def _last_error_line(text: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].startswith("Error executing the command"):
            prev = lines[i - 1] if i > 0 else ""
            return f"{lines[i]} | after: {prev[:300]}"
    return lines[-1][:300] if lines else ""


IR_MARKERS = {
    "hexkl_micro": "hexkl_micro",
    "hexkl_macro": "hexkl_macro",
    "hvx_vector": "<32 x i32>",  # 128B HVX vector types in LLVM IR
    "hvx_f16x64": "<64 x half>",
    "scf_forall": "scf.forall",
    "async_runtime": "async",
    "dma": "dma",
    "vtcm": "vtcm",
}


def ir_fingerprint(compiled):
    """Hash the final LLVM-level IR, so 'different config, same lowering' is visible."""
    import hashlib

    asm = getattr(compiled, "asm", None) or {}
    key = "llir" if "llir" in asm else ("ttsharedir" if "ttsharedir" in asm else None)
    if key is None:
        return None, {}
    text = asm[key] if isinstance(asm[key], str) else asm[key].decode("latin1")
    full = "\n".join(str(asm[k]) for k in sorted(asm) if k in ("ttsharedir", "llir"))
    markers = {k: full.count(v) for k, v in IR_MARKERS.items()}
    return f"{key}:{hashlib.sha256(text.encode()).hexdigest()[:16]}", markers


def save_ir(compiled, out_dir: Path):
    asm = getattr(compiled, "asm", None) or {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in asm.items():
        if isinstance(v, (str, bytes)) and k != "o":
            (out_dir / f"kernel.{k}").write_text(v if isinstance(v, str) else v.decode("latin1"))
