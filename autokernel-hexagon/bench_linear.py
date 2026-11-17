#!/usr/bin/env python3
"""HMX extension: compare HVX default vs HexKL (micro) lowering of the same Linear.

    python bench_linear.py --backend hexagon --m 1,32 --k 512 --n 512 --samples 6

Per M (decode M=1, prefill M>1) and per path: accuracy vs CPU fp32 reference,
lowering check from the compiled IR (hexkl_micro calls present / absent),
then interleaved (hvx, hexkl, hexkl, hvx) timing samples using the same
runner/timer as RMSNorm. Rows go to results_linear.tsv.

Weight layout cost accounting: with Triton, weights are runtime pointers, so
there is no init-time prepack; HexKL micro's layout conversion of activations
and outputs runs inside every timed call (weight_layout = per_call). The
init-time prepack path (macro mode, constant weights) is reachable only through
torch-mlir; see README "HMX / HexKL".
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import bench  # noqa: E402
from akh import env as envmod, reference, stats  # noqa: E402
from akh import runners as R  # noqa: E402
from akh.launch import file_sha256, load_kernel_module  # noqa: E402

ATOL, RTOL = 1e-2, 1e-2  # HexKL produces f16 then upcasts (MatmulToHexKLPass)
COLS = ["timestamp", "backend", "kernel_sha", "path", "phase", "M", "K", "N", "dtype", "ptr_style", "target", "triton_rev",
        "hexmlir_rev", "status", "max_abs_err", "lowering", "hexkl_calls", "ir_hash", "dsp_us_median",
        "dsp_spread", "first_call_us", "host_call_ms", "host_run_ms", "n_samples", "weight_layout", "note", "raw"]


def inputs(m, k, n, seed):
    g = torch.Generator().manual_seed(seed)
    return (torch.rand(m, k, generator=g).half(), torch.rand(k, n, generator=g).half())


PTR_STYLE = {"hexagon": "block", "host-interp": "offsets"}


def run(runner, mod, path, m, k, n, seed, tag, ptr):
    x, w = inputs(m, k, n, seed)
    y = torch.empty(m, n, dtype=torch.float32)
    rr = runner.run(mod.launches(x, w, y, {"path": path, "ptr_style": ptr}), tag)
    if rr.status != R.OK:
        return rr, None
    ref = reference.linear_ref(x, w)
    err = (y - ref).abs()
    ok = bool((err <= ATOL + RTOL * ref.abs()).all())
    return rr, (ok, err.max().item())


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=bench.BACKENDS, default="hexagon")
    ap.add_argument("--kernel", default=str(HERE / "ops" / "linear_kernel.py"))
    ap.add_argument("--m", default="1,32")
    ap.add_argument("--k", type=int, default=512)
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--paths", default="hvx,hexkl_micro")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--iterations", type=int, default=100)
    ap.add_argument("--ptr-style", choices=["block", "offsets"], default=None,
                    help="default: block on hexagon (official form), offsets on host-interp")
    a = ap.parse_args(argv)
    ptr = a.ptr_style or PTR_STYLE[a.backend]
    out_dir = HERE / "runs" / time.strftime("linear_%Y%m%d_%H%M%S")
    try:
        runner = bench.make_runner(a.backend, out_dir / "logs", a.iterations)
    except RuntimeError as e:
        print(f"status: device_unavailable\nerror: {e}")
        return 1
    mod = load_kernel_module(a.kernel)
    e = envmod.short(envmod.collect(adb_probe=a.backend == "hexagon"))
    paths = a.paths.split(",")
    raw, rows = {}, []
    for m in (int(v) for v in a.m.split(",")):
        phase = "decode" if m == 1 else "prefill"
        per = {}
        for p in paths:
            st, note, maxerr, ir_hash, markers = R.OK, "", None, None, {}
            for seed in (0, 1):
                rr, acc = run(runner, mod, p, m, a.k, a.n, seed, f"acc_{p}_M{m}_s{seed}", ptr)
                if rr.status != R.OK:
                    st, note = rr.status, rr.error
                    break
                ir_hash, markers = rr.launches[0].ir_hash, rr.launches[0].ir_markers
                maxerr = max(maxerr or 0, acc[1])
                if not acc[0]:
                    st, note = R.ACCURACY_FAIL, f"seed {seed} max_abs_err={acc[1]}"
                    break
            if markers:
                lowering = "hexkl" if markers.get("hexkl_micro", 0) + markers.get("hexkl_macro", 0) else "hvx"
                if p.startswith("hexkl") and lowering != "hexkl":
                    note += " | enableHexKL set but no hexkl calls in IR (pattern did not match)"
            else:
                lowering = "n/a (no compiled IR on this backend)"
            per[p] = {"status": st, "note": note, "max_abs_err": maxerr, "ir_hash": ir_hash,
                      "markers": markers, "lowering": lowering, "samples": []}
        timed = [p for p in paths if per[p]["status"] == R.OK] if runner.produces_perf else []
        for i in range(a.samples // 2 if len(timed) > 1 else a.samples):
            order = timed + timed[::-1] if len(timed) > 1 else timed
            for p in order:
                x, w = inputs(m, a.k, a.n, 0)
                y = torch.empty(m, a.n, dtype=torch.float32)
                rr = runner.run(mod.launches(x, w, y, {"path": p, "ptr_style": ptr}), f"perf_{p}_M{m}_{i}")
                per[p]["samples"].append({"status": rr.status, "dsp_us": rr.total("dsp_avg_us"),
                                          "first_call_us": rr.total("dsp_first_call_us"),
                                          "host_call_ms": rr.total("host_call_ms"),
                                          "host_run_ms": rr.total("host_run_ms"), "error": rr.error})
        raw[f"M{m}"] = per
        for p in paths:
            d = per[p]
            ok = [s for s in d["samples"] if s["status"] == R.OK]
            pick = lambda key: [s[key] for s in ok if s[key] is not None]  # noqa: E731
            rows.append({"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "backend": runner.name,
                         "kernel_sha": file_sha256(a.kernel)[:16], "path": p, "phase": phase, "M": m, "K": a.k,
                         "N": a.n, "dtype": "fp16xfp16->fp32", "ptr_style": ptr, **{k: e.get(k, "") for k in ("target", "triton_rev", "hexmlir_rev")},
                         "status": d["status"], "max_abs_err": d["max_abs_err"], "lowering": d["lowering"],
                         "hexkl_calls": d["markers"].get("hexkl_micro", "") if d["markers"] else "",
                         "ir_hash": d["ir_hash"], "dsp_us_median": stats.median(pick("dsp_us")),
                         "dsp_spread": stats.rel_spread(pick("dsp_us")),
                         "first_call_us": stats.median(pick("first_call_us")),
                         "host_call_ms": stats.median(pick("host_call_ms")),
                         "host_run_ms": stats.median(pick("host_run_ms")), "n_samples": len(ok) if runner.produces_perf else "",
                         "weight_layout": "per_call (runtime weight pointer; no init prepack)" if p.startswith("hexkl") else "row-major, no conversion",
                         "note": d["note"] or ("" if runner.produces_perf else "host-interp: accuracy only, no perf"),
                         "raw": str((out_dir / "linear.json").relative_to(HERE))})
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "linear.json").write_text(json.dumps(raw, indent=2, default=str))
    tsv = HERE / "results_linear.tsv"
    new = not tsv.exists()
    with tsv.open("a") as f:
        if new:
            f.write("\t".join(COLS) + "\n")
        for r in rows:
            f.write("\t".join("" if r.get(c) is None else (f"{r[c]:.6g}" if isinstance(r[c], float) else str(r[c])).replace("\t", " ")
                              for c in COLS) + "\n")
    for r in rows:
        print(f"[{r['phase']} M={r['M']}] {r['path']:<12} status={r['status']} lowering={r['lowering']} "
              f"max_abs_err={r['max_abs_err']} dsp_us={r['dsp_us_median']} host_call_ms={r['host_call_ms']} {r['note']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
