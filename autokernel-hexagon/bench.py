#!/usr/bin/env python3
"""Fixed benchmark + correctness harness (evaluator-owned; the agent must not edit).

    python bench.py --backend hexagon                 # real device
    python bench.py --backend host-interp             # logic check only, no perf
    python bench.py --config '{"strategy":"split_redundant","split":4,"num_programs":4}'

Pipeline for one candidate:
  1. JIT compile (timed separately, excluded from steady state)
  2. accuracy: perf cases + evaluator-generated validation cases vs CPU PyTorch
  3. timing (hexagon only): `--samples` launcher invocations per shape; each
     invocation = 1 cold call (first_call_us) + `--iterations` timed calls on the
     DSP (dsp_avg_us). Host-side fields per invocation:
       host_call_ms  python call -> outputs back in host tensors. Includes
                     wrapper .so link (hexagon-clang), adb push of inputs/libs,
                     run_main_on_hexagon, adb pull of outputs/perf.txt.
       host_run_ms   wall time of the `adb shell run_main_on_hexagon` step only:
                     FastRPC session open, DSP library load, input file read,
                     cold call + iterations, output file write.
     Neither host field is a FastRPC per-call latency; this launcher has none.
     Every timing sample's output is re-checked against the reference.

Output: key: value lines on stdout and a JSON file (--out).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from akh import contract, reference  # noqa: E402
from akh import runners as R  # noqa: E402
from akh.launch import effective_config, file_sha256, load_kernel_module  # noqa: E402

BACKENDS = ("hexagon", "host-interp")


def make_runner(backend: str, log_dir: Path, iterations: int = 100):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if backend == "host-interp":
        os.environ["TRITON_INTERPRET"] = "1"  # must precede kernel import (triton.jit)
        return R.HostInterpRunner(log_dir)
    if backend == "hexagon":
        if os.environ.get("TRITON_INTERPRET") == "1":
            raise RuntimeError("TRITON_INTERPRET=1 is set; unset it for the hexagon backend")
        return R.HexagonRunner(log_dir, iterations=iterations)
    raise ValueError(backend)


def run_case(runner, mod, cfg, case: contract.Case, tag: str):
    x, w = contract.make_inputs(case)
    y = torch.empty_like(x, dtype=contract.OUT_DTYPE)
    try:
        launches = mod.launches(x, w, y, contract.EPS, cfg)
    except Exception as e:  # noqa: BLE001  (bad config / python error in kernel.py)
        return R.RunResult(R.COMPILE_FAIL, [], f"launches(): {type(e).__name__}: {e}"), None
    rr = runner.run(launches, tag)
    if rr.status != R.OK:
        return rr, None
    ref = reference.rmsnorm_ref(x, w, contract.EPS)
    return rr, contract.check(case, y, ref)


def accuracy_suite(runner, mod, cfg, shapes=None, tag="acc"):
    """Returns (status, error, [accuracy dicts], compile_s, ir)."""
    cases = contract.perf_cases(shapes) + contract.validation_cases()
    supports_tail = bool(getattr(mod, "SUPPORTS_TAIL", False))
    rows, compile_s, ir = [], 0.0, {}
    for c in cases:
        if c.needs_tail and not supports_tail:
            rows.append({"case": c.name, "ok": None, "note": "skipped: SUPPORTS_TAIL=False"})
            continue
        rr, acc = run_case(runner, mod, cfg, c, f"{tag}_{c.name}")
        for l in rr.launches:
            compile_s += l.compile_s or 0.0
            if l.ir_hash and c.perf:
                ir.setdefault(f"{c.shape[0]}x{c.shape[1]}", []).append(
                    {"launch": l.name, "ir_hash": l.ir_hash, "markers": l.ir_markers})
        if rr.status != R.OK:
            rows.append({"case": c.name, "ok": False, "note": f"{rr.status}: {rr.error}",
                         "logs": [l.log_path for l in rr.launches if l.log_path]})
            return rr.status, f"{c.name}: {rr.error}", rows, compile_s, ir
        rows.append(asdict(acc))
    failed = [r for r in rows if r.get("ok") is False]
    if failed:
        worst = max(failed, key=lambda r: r.get("max_abs_err") or float("inf"))
        return R.ACCURACY_FAIL, (f"{len(failed)} case(s) failed; worst {worst['case']} "
                                 f"max_abs_err={worst.get('max_abs_err')} {worst.get('note', '')}"), rows, compile_s, ir
    return R.OK, "", rows, compile_s, ir


def max_abs_err(rows):
    vals = [r.get("max_abs_err") for r in rows if r.get("ok") is not None and r.get("max_abs_err") is not None]
    return max(vals) if vals else None


def measure_once(runner, mod, cfg, shape, tag):
    """One timing sample = one launcher invocation per Launch (summed)."""
    case = contract.perf_cases([shape])[0]
    rr, acc = run_case(runner, mod, cfg, case, tag)
    s = {"status": rr.status, "error": rr.error,
         "dsp_us": rr.total("dsp_avg_us"), "first_call_us": rr.total("dsp_first_call_us"),
         "dsp_pcycles": rr.total("dsp_avg_pcycles"),
         "host_call_ms": rr.total("host_call_ms"), "host_run_ms": rr.total("host_run_ms"),
         "timer": ",".join(sorted({l.timer for l in rr.launches})) or "missing",
         "n_launches": len(rr.launches),
         "logs": [l.log_path for l in rr.launches if l.log_path],
         "evidence": [l.evidence for l in rr.launches]}
    if rr.status == R.OK and acc is not None and not acc.ok:
        s["status"], s["error"] = R.ACCURACY_FAIL, f"timing-sample output wrong, max_abs_err={acc.max_abs_err}"
    if s["status"] != R.OK:
        s["dsp_us"] = s["host_call_ms"] = s["host_run_ms"] = None  # never rank partial samples
    return s


def evaluate(kernel_path, override=None, backend="hexagon", shapes=None, samples=5,
             iterations=100, out_dir=None, runner=None):
    shapes = shapes or contract.PERF_SHAPES
    out_dir = Path(out_dir or HERE / "runs" / time.strftime("bench_%Y%m%d_%H%M%S"))
    runner = runner or make_runner(backend, out_dir / "logs", iterations)
    mod = load_kernel_module(kernel_path)
    res = {"kernel": str(kernel_path), "kernel_sha256": file_sha256(kernel_path),
           "backend": runner.name, "contract": contract.CONTRACT_VERSION,
           "hypothesis": getattr(mod, "HYPOTHESIS", "")}
    try:
        cfg = effective_config(mod, override)
    except ValueError as e:
        res.update(status=R.COMPILE_FAIL, error=str(e))
        return res
    res["config"] = cfg
    if runner.name == "hexagon":
        ok, msg = runner.check_device()
        if not ok:
            res.update(status=R.DEVICE_UNAVAILABLE, error=msg)
            return res
    st, err, rows, compile_s, ir = accuracy_suite(runner, mod, cfg, shapes)
    res.update(status=st, error=err, accuracy=rows, max_abs_err=max_abs_err(rows),
               compile_s_total=compile_s, ir=ir)
    res["perf"] = {}
    if st != R.OK or not runner.produces_perf:
        if not runner.produces_perf:
            res["perf_note"] = "not measured: host-interp backend produces no DSP/NPU timing"
        return res
    for m, h in shapes:
        key = f"{m}x{h}"
        ss = [measure_once(runner, mod, cfg, (m, h), f"perf_{key}_{i}") for i in range(samples)]
        res["perf"][key] = {"samples": ss, **summarize(ss)}
    return res


def summarize(ss):
    from akh.stats import median, rel_spread

    ok = [s for s in ss if s["status"] == R.OK]
    pick = lambda k: [s[k] for s in ok if s.get(k) is not None]  # noqa: E731
    return {"n_ok": len(ok), "n_total": len(ss),
            "dsp_us_median": median(pick("dsp_us")), "dsp_us_spread": rel_spread(pick("dsp_us")),
            "first_call_us_median": median(pick("first_call_us")),
            "host_call_ms_median": median(pick("host_call_ms")),
            "host_run_ms_median": median(pick("host_run_ms"))}


def print_summary(res):
    print(f"kernel_sha256: {res['kernel_sha256'][:16]}")
    print(f"backend: {res['backend']}")
    print(f"status: {res['status']}")
    if res.get("error"):
        print(f"error: {res['error']}")
    print(f"correctness: {'PASS' if res['status'] == R.OK else 'FAIL' if res['status'] == R.ACCURACY_FAIL else 'N/A'}")
    print(f"max_abs_err: {res.get('max_abs_err')}")
    for key, p in res.get("perf", {}).items():
        print(f"[{key}] dsp_us_median: {p['dsp_us_median']}  spread: {p['dsp_us_spread']}  "
              f"first_call_us: {p['first_call_us_median']}  host_call_ms: {p['host_call_ms_median']}  "
              f"host_run_ms: {p['host_run_ms_median']}  ok: {p['n_ok']}/{p['n_total']}")
    if res.get("perf_note"):
        print(f"perf: {res['perf_note']}")


def parse_shapes(s):
    return [tuple(int(v) for v in t.split("x")) for t in s.split(",")] if s else None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kernel", default=str(HERE / "kernel.py"))
    ap.add_argument("--backend", choices=BACKENDS, default="hexagon")
    ap.add_argument("--config", default=None, help="JSON override merged into kernel CONFIG")
    ap.add_argument("--shapes", default=None, help="e.g. 1x4096,32x4096")
    ap.add_argument("--samples", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    out_dir = Path(a.out or HERE / "runs" / time.strftime("bench_%Y%m%d_%H%M%S"))
    try:
        res = evaluate(a.kernel, json.loads(a.config) if a.config else None, a.backend,
                       parse_shapes(a.shapes), a.samples, a.iterations, out_dir)
    except RuntimeError as e:  # environment not usable for this backend
        res = {"kernel": a.kernel, "kernel_sha256": file_sha256(a.kernel), "backend": a.backend,
               "status": R.DEVICE_UNAVAILABLE, "error": str(e)}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "bench.json").write_text(json.dumps(res, indent=2, default=str))
    print_summary(res)
    print(f"json: {out_dir / 'bench.json'}")
    return 0 if res["status"] == R.OK else 1


if __name__ == "__main__":
    sys.exit(main())
