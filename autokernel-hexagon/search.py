#!/usr/bin/env python3
"""Experiment driver: baseline, finite grid search, agent iterations, restore.

    python search.py baseline            # first device-verified simple kernel = perf baseline
    python search.py grid --space small  # finite search over CONFIG overrides (greedy, vs best)
    python search.py try --hypothesis "..."   # evaluate the agent's edited kernel.py vs best
    python search.py final               # interleaved initial-baseline vs best, all shapes
    python search.py status | restore | adopt <exp_id> --reason "..."

State is kept per backend under state/<backend>/ so host-interp checks can
never become a device performance baseline. Best is only replaced after a
complete, verified comparison; an interrupted experiment leaves an
``inflight.json`` marker that the next command records as `interrupted`.
"""
from __future__ import annotations

import argparse
import difflib
import json
import shutil
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bench  # noqa: E402
from akh import contract, env as envmod, stats  # noqa: E402
from akh import runners as R  # noqa: E402
from akh.launch import (config_key, effective_config, file_sha256, load_kernel_module,  # noqa: E402
                        write_config_block)

TSV_COLUMNS = [
    "exp_id", "timestamp", "kind", "backend", "parent_best", "kernel_sha", "diff", "config_key",
    "config_json", "hypothesis", "shape", "dtype", "contract", "target", "soc", "triton_rev",
    "hexmlir_rev", "sdk", "status", "max_abs_err", "failing_case", "dsp_us_median",
    "base_dsp_us_median", "base_dsp_spread", "dsp_gain_vs_best", "dsp_gain_vs_initial",
    "first_call_us", "host_call_ms", "base_host_call_ms", "host_run_ms", "n_samples",
    "ir_hash", "verdict", "adopted", "reason", "raw",
]

SPACES = {
    # Greedy, one override at a time on top of the CURRENT best CONFIG.
    "small": [
        ("rsqrt instead of 1/sqrt: one HVX rsqrt vs sqrt+div", {"use_rsqrt": True}),
        ("chunked H=2048: smaller live vector, but x re-read", {"strategy": "row", "block_h": 2048}),
        ("chunked H=1024: more chunks, x re-read", {"strategy": "row", "block_h": 1024}),
        ("HVX inlining + scf loop unroll (official softmax flags)",
         {"hexagon": {"enableHVXInlining": True, "enableSCFLoopUnroll": True}}),
        ("split-reduction pass for the H reduction", {"hexagon": {"enableSplitReduction": True}}),
        ("split-reduce-generic pass", {"hexagon": {"enableSplitReduceGeneric": True}}),
        ("no VTCM tiling: 8 KB row may not need VTCM staging", {"hexagon": {"enableVTCMTiling": False}}),
        ("DMA for hexagonmem copies (grid==1 only)", {"hexagon": {"enableHexagonmemCopyToDMA": True}}),
        ("rows over 4 programs, serial dispatch (control: no real threads)",
         {"strategy": "row", "num_programs": 4, "hexagon": {"enableThreadedDispatch": False}}),
        ("rows over 4 programs, real qurt threads",
         {"strategy": "row", "num_programs": 4, "hexagon": {"enableThreadedDispatch": True}}),
        ("M=1 hidden split x2, redundant full-row reduce, threaded",
         {"strategy": "split_redundant", "split": 2, "num_programs": 0, "hexagon": {"enableThreadedDispatch": True}}),
        ("M=1 hidden split x4, redundant full-row reduce, threaded",
         {"strategy": "split_redundant", "split": 4, "num_programs": 0, "hexagon": {"enableThreadedDispatch": True}}),
        ("M=1 hidden split x4, two-pass partial merge (2 launches), threaded",
         {"strategy": "split_two_pass", "split": 4, "num_programs": 0, "hexagon": {"enableThreadedDispatch": True}}),
        ("linalg multithreading (FormVirtualThreads) with grid=4",
         {"strategy": "row", "num_programs": 4, "hexagon": {"enableMultiThreading": True}}),
    ],
    "tiny": [
        ("rsqrt instead of 1/sqrt", {"use_rsqrt": True}),
        ("chunked H=1024: x re-read", {"strategy": "row", "block_h": 1024}),
        ("M=1 hidden split x4, redundant reduce, threaded",
         {"strategy": "split_redundant", "split": 4, "num_programs": 0, "hexagon": {"enableThreadedDispatch": True}}),
    ],
}


def _atomic_write(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _clean(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v).replace("\t", " ").replace("\n", " ")


class Workspace:
    def __init__(self, root: Path, backend: str, runner=None, samples=5, iterations=100,
                 max_rounds=3, shapes=None, env_info=None):
        self.root = Path(root)
        self.backend = backend
        self.kernel = self.root / "kernel.py"
        self.state = self.root / "state" / backend
        self.state.mkdir(parents=True, exist_ok=True)
        self.runs = self.root / "runs"
        self.runs.mkdir(exist_ok=True)
        self.tsv = self.root / "results.tsv"
        self.samples, self.iterations, self.max_rounds = samples, iterations, max_rounds
        self.shapes = shapes or contract.PERF_SHAPES
        self._runner = runner
        self._env = env_info

    # ---------------------------------------------------------------- infra
    @property
    def runner(self):
        if self._runner is None:
            self._runner = bench.make_runner(self.backend, self.runs / "_runner_logs", self.iterations)
        return self._runner

    @property
    def env(self):
        if self._env is None:
            self._env = envmod.short(envmod.collect(adb_probe=self.backend == "hexagon"))
        return self._env

    def _json(self, name):
        p = self.state / name
        return json.loads(p.read_text()) if p.exists() else None

    def best(self):
        return self._json("best.json")

    def initial(self):
        return self._json("initial.json")

    def best_kernel(self) -> Path:
        return self.state / "best_kernel.py"

    def next_exp_id(self, kind):
        c = self.state / "counter"
        n = int(c.read_text()) + 1 if c.exists() else 1
        _atomic_write(c, str(n))
        return f"{self.backend[:3]}{n:04d}_{kind}"

    def append_rows(self, rows):
        new = not self.tsv.exists()
        with self.tsv.open("a") as f:
            if new:
                f.write("\t".join(TSV_COLUMNS) + "\n")
            for r in rows:
                f.write("\t".join(_clean(r.get(c)) for c in TSV_COLUMNS) + "\n")

    def check_inflight(self):
        p = self.state / "inflight.json"
        if p.exists():
            info = json.loads(p.read_text())
            print(f"WARNING: experiment {info['exp_id']} was interrupted; best is unchanged.")
            self.append_rows([{**info, "status": "interrupted", "verdict": "keep_previous", "adopted": "no",
                               "reason": "process ended before a decision; best kept"}])
            p.unlink()

    def protect_user_kernel(self):
        """Back up kernel.py if it differs from best (never silently overwrite user edits)."""
        bk = self.best_kernel()
        if bk.exists() and self.kernel.exists() and file_sha256(bk) != file_sha256(self.kernel):
            dst = self.runs / f"_kernel_backup_{time.strftime('%Y%m%d_%H%M%S')}.py"
            shutil.copy2(self.kernel, dst)
            return dst
        return None

    def restore(self, backup=True):
        bk = self.best_kernel()
        if not bk.exists():
            raise SystemExit("no best kernel recorded for this backend yet (run `search.py baseline`)")
        backup = self.protect_user_kernel() if backup else None
        shutil.copy2(bk, self.kernel)
        return backup

    def _base_row(self, exp_id, kind, cand_path, cfg, hypothesis, shape):
        best = self.best() or {}
        return {"exp_id": exp_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind,
                "backend": self.backend, "parent_best": best.get("exp_id", ""),
                "kernel_sha": file_sha256(cand_path)[:16], "config_key": config_key(cfg) if cfg else "",
                "config_json": json.dumps(cfg, sort_keys=True) if cfg else "", "hypothesis": hypothesis,
                "shape": f"{shape[0]}x{shape[1]}", "dtype": "fp16->fp32acc->fp16",
                "contract": contract.CONTRACT_VERSION, **self.env}

    def _write_diff(self, exp_dir: Path, cand: Path):
        bk = self.best_kernel()
        if not bk.exists():
            return ""
        d = "".join(difflib.unified_diff(bk.read_text().splitlines(True), cand.read_text().splitlines(True),
                                         "best/kernel.py", "candidate/kernel.py"))
        (exp_dir / "diff.patch").write_text(d)
        return str((exp_dir / "diff.patch").relative_to(self.root)) if d else "(identical)"

    # ---------------------------------------------------------- evaluation
    def _accuracy(self, path, override, tag):
        mod = load_kernel_module(path)
        try:
            cfg = effective_config(mod, override)
        except ValueError as e:
            return mod, None, (R.COMPILE_FAIL, str(e), [], 0.0, {})
        return mod, cfg, bench.accuracy_suite(self.runner, mod, cfg, self.shapes, tag=tag)

    def _measure_pair(self, base, cand, rounds):
        """ABBA-interleaved samples. base/cand = (mod, cfg). Returns {shape: {'base': [...], 'cand': [...]}}"""
        out = {f"{m}x{h}": {"base": [], "cand": []} for m, h in self.shapes}
        for r in range(rounds):
            for m, h in self.shapes:
                key = f"{m}x{h}"
                per_round = max(1, self.samples // 2)
                for i in range(per_round):
                    order = [("base", base), ("cand", cand), ("cand", cand), ("base", base)]
                    for who, (mod, cfg) in order:
                        s = bench.measure_once(self.runner, mod, cfg, (m, h), f"{who}_{key}_r{r}_{i}")
                        out[key][who].append(s)
        return out

    def baseline(self, hypothesis=None, force=False):
        self.check_inflight()
        if self.initial() and not force:
            raise SystemExit(f"initial baseline already recorded ({self.initial()['exp_id']}); use --force")
        exp_id = self.next_exp_id("baseline")
        exp_dir = self.runs / exp_id
        exp_dir.mkdir()
        cand = exp_dir / "kernel.py"
        shutil.copy2(self.kernel, cand)
        mod, cfg, (st, err, rows, compile_s, ir) = self._accuracy(cand, None, f"{exp_id}_acc")
        hypothesis = hypothesis or getattr(mod, "HYPOTHESIS", "baseline")
        perf = {}
        if st == R.OK and self.runner.produces_perf:
            for m, h in self.shapes:
                key = f"{m}x{h}"
                ss = [bench.measure_once(self.runner, mod, cfg, (m, h), f"{exp_id}_{key}_{i}")
                      for i in range(self.samples)]
                perf[key] = {"samples": ss, **bench.summarize(ss)}
        raw = {"accuracy": rows, "compile_s": compile_s, "ir": ir, "perf": perf, "config": cfg}
        (exp_dir / "result.json").write_text(json.dumps(raw, indent=2, default=str))
        primary = f"{contract.PRIMARY_SHAPE[0]}x{contract.PRIMARY_SHAPE[1]}"
        timed = bool(perf) and perf.get(primary, {}).get("dsp_us_median") is not None
        if st == R.OK and self.runner.produces_perf and not timed:
            st, err = R.TIMING_PARSE_FAIL, "baseline passed accuracy but no DSP timing was collected"
        adopted = st == R.OK
        verdict = ("initial_baseline" if timed else "accuracy_only_baseline") if adopted else "rejected"
        reason = ("first device-verified simple kernel = perf baseline" if timed else
                  "host-interp: accuracy-verified logic baseline; NO performance baseline" if adopted else err)
        rows_out = []
        for m, h in self.shapes:
            key = f"{m}x{h}"
            p = perf.get(key, {})
            rows_out.append({**self._base_row(exp_id, "baseline", cand, cfg, hypothesis, (m, h)),
                             "diff": "(baseline)", "status": st, "max_abs_err": bench.max_abs_err(rows),
                             "failing_case": _failing(rows), "dsp_us_median": p.get("dsp_us_median"),
                             "first_call_us": p.get("first_call_us_median"),
                             "host_call_ms": p.get("host_call_ms_median"), "host_run_ms": p.get("host_run_ms_median"),
                             "n_samples": p.get("n_ok"), "ir_hash": _ir(ir, key), "verdict": verdict,
                             "adopted": "yes" if adopted else "no", "reason": reason,
                             "raw": str((exp_dir / "result.json").relative_to(self.root))})
        self.append_rows(rows_out)
        if adopted:
            metrics = {k: {x: v[x] for x in v if x != "samples"} for k, v in perf.items()}
            rec = {"exp_id": exp_id, "kernel_sha": file_sha256(cand), "config": cfg, "metrics": metrics,
                   "ir": ir, "timed": timed, "hypothesis": hypothesis}
            shutil.copy2(cand, self.best_kernel())
            _atomic_write(self.state / "best.json", json.dumps(rec, indent=2, default=str))
            _atomic_write(self.state / "initial.json", json.dumps(rec, indent=2, default=str))
            shutil.copy2(cand, self.state / "initial_kernel.py")
        print(f"{exp_id}: status={st} verdict={verdict} {reason}")
        return exp_id, st, verdict

    def compare(self, cand_src: Path, override, kind, hypothesis):
        """Evaluate candidate (file + optional CONFIG override) against the current best."""
        self.check_inflight()
        best = self.best()
        if not best:
            raise SystemExit("no best yet: run `search.py baseline` first")
        exp_id = self.next_exp_id(kind)
        exp_dir = self.runs / exp_id
        exp_dir.mkdir()
        cand = exp_dir / "kernel.py"
        if override:
            src = cand_src.read_text()
            mod0 = load_kernel_module(cand_src)
            src = write_config_block(src, effective_config(mod0, override))
            cand.write_text(src)
        else:
            shutil.copy2(cand_src, cand)
        diff = self._write_diff(exp_dir, cand)
        _atomic_write(self.state / "inflight.json", json.dumps(
            {"exp_id": exp_id, "kind": kind, "hypothesis": hypothesis, "backend": self.backend,
             "kernel_sha": file_sha256(cand)[:16], "raw": str(exp_dir.relative_to(self.root))}))

        result = {"exp_id": exp_id, "hypothesis": hypothesis}
        if diff == "(identical)":  # nothing to evaluate
            mod, cfg = None, best["config"]
            st, err, acc_rows, compile_s, ir = R.OK, "", [], 0.0, best.get("ir") or {}
        else:
            mod, cfg, (st, err, acc_rows, compile_s, ir) = self._accuracy(cand, None, f"{exp_id}_acc")
        result.update(accuracy=acc_rows, compile_s=compile_s, ir=ir, config=cfg)
        verdict, reason, pairs, dec = None, "", {}, None
        primary = f"{contract.PRIMARY_SHAPE[0]}x{contract.PRIMARY_SHAPE[1]}"
        if diff == "(identical)":
            verdict, reason = "keep_previous", "candidate source identical to best"
        elif st != R.OK:
            verdict, reason = "rejected", err
        elif not self.runner.produces_perf or not best.get("timed"):
            verdict = "not_judged"
            reason = ("accuracy PASS on host-interp; performance not judged without a device; best kept"
                      if not self.runner.produces_perf else "best has no DSP timing; cannot compare")
        elif best.get("ir") and ir and _same_ir(best["ir"], ir):
            verdict = "keep_previous"
            reason = "final IR identical to best for all perf shapes (same lowering) -> no perf change possible"
        else:
            bmod = load_kernel_module(self.best_kernel())
            bcfg = best["config"]
            rounds, samples = 0, {}
            while True:
                rounds += 1
                new = self._measure_pair((bmod, bcfg), (mod, cfg), 1)
                for k, v in new.items():
                    samples.setdefault(k, {"base": [], "cand": []})
                    samples[k]["base"] += v["base"]
                    samples[k]["cand"] += v["cand"]
                b = samples[primary]["base"]
                c = samples[primary]["cand"]
                bad = next((s for s in c if s["status"] != R.OK), None)
                if bad:
                    st, verdict, reason = bad["status"], "rejected", f"timing run failed: {bad['error']}"
                    break
                dec = stats.decide([s["dsp_us"] for s in b], [s["dsp_us"] for s in c],
                                   [s["host_call_ms"] for s in b], [s["host_call_ms"] for s in c],
                                   rounds_done=rounds, max_rounds=self.max_rounds)
                if dec.verdict != "need_more":
                    break
            pairs = samples
            if verdict is None:
                verdict, reason = dec.verdict, dec.reason
                if dec.verdict == "no_timing":
                    st = R.TIMING_PARSE_FAIL
        result.update(status=st, verdict=verdict, reason=reason, samples=pairs,
                      decision=dec.__dict__ if dec else None)
        (exp_dir / "result.json").write_text(json.dumps(result, indent=2, default=str))

        adopted = verdict == "improved"
        init = self.initial() or {}
        rows = []
        for m, h in self.shapes:
            key = f"{m}x{h}"
            p = pairs.get(key, {})
            ok_c = [s for s in p.get("cand", []) if s["status"] == R.OK]
            ok_b = [s for s in p.get("base", []) if s["status"] == R.OK]
            cm = stats.median([s["dsp_us"] for s in ok_c])
            bm = stats.median([s["dsp_us"] for s in ok_b])
            im = (init.get("metrics") or {}).get(key, {}).get("dsp_us_median")
            rows.append({**self._base_row(exp_id, kind, cand, cfg, hypothesis, (m, h)), "diff": diff,
                         "status": st, "max_abs_err": bench.max_abs_err(acc_rows),
                         "failing_case": _failing(acc_rows), "dsp_us_median": cm, "base_dsp_us_median": bm,
                         "base_dsp_spread": stats.rel_spread([s["dsp_us"] for s in ok_b]),
                         "dsp_gain_vs_best": (bm - cm) / bm if cm and bm else None,
                         "dsp_gain_vs_initial": (im - cm) / im if cm and im else None,
                         "first_call_us": stats.median([s["first_call_us"] for s in ok_c]),
                         "host_call_ms": stats.median([s["host_call_ms"] for s in ok_c]),
                         "base_host_call_ms": stats.median([s["host_call_ms"] for s in ok_b]),
                         "host_run_ms": stats.median([s["host_run_ms"] for s in ok_c]),
                         "n_samples": len(ok_c) if p else None, "ir_hash": _ir(ir, key),
                         "verdict": verdict + ("" if key == primary else " (primary-shape decision)"),
                         "adopted": "yes" if adopted else "no", "reason": reason,
                         "raw": str((exp_dir / "result.json").relative_to(self.root))})
        self.append_rows(rows)
        if adopted:
            metrics = {}
            for k, v in pairs.items():
                ok_c = [s for s in v["cand"] if s["status"] == R.OK]
                metrics[k] = {"dsp_us_median": stats.median([s["dsp_us"] for s in ok_c]),
                              "host_call_ms_median": stats.median([s["host_call_ms"] for s in ok_c]),
                              "first_call_us_median": stats.median([s["first_call_us"] for s in ok_c])}
            self._set_best(exp_id, cand, cfg, metrics, ir, hypothesis)
        (self.state / "inflight.json").unlink(missing_ok=True)
        print(f"{exp_id}: status={st} verdict={verdict} adopted={adopted} :: {reason}")
        return exp_id, st, verdict

    def _set_best(self, exp_id, cand, cfg, metrics, ir, hypothesis):
        shutil.copy2(cand, self.best_kernel())
        rec = {"exp_id": exp_id, "kernel_sha": file_sha256(cand), "config": cfg, "metrics": metrics,
               "ir": ir, "timed": True, "hypothesis": hypothesis}
        _atomic_write(self.state / "best.json", json.dumps(rec, indent=2, default=str))

    def adopt(self, exp_id, reason):
        exp_dir = self.runs / exp_id
        res = json.loads((exp_dir / "result.json").read_text())
        if res.get("verdict") not in ("hold_host_regression",):
            raise SystemExit(f"{exp_id} verdict is {res.get('verdict')}; only hold_host_regression can be adopted manually")
        metrics = {}
        for k, v in res["samples"].items():
            ok_c = [s for s in v["cand"] if s["status"] == R.OK]
            metrics[k] = {"dsp_us_median": stats.median([s["dsp_us"] for s in ok_c]),
                          "host_call_ms_median": stats.median([s["host_call_ms"] for s in ok_c])}
        self._set_best(exp_id, exp_dir / "kernel.py", res["config"], metrics, res.get("ir"), res["hypothesis"])
        self.append_rows([{"exp_id": exp_id, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": "manual_adopt",
                           "backend": self.backend, "verdict": "improved", "adopted": "yes", "reason": reason,
                           "raw": str((exp_dir / "result.json").relative_to(self.root))}])

    def grid(self, space):
        results = []
        for hyp, override in SPACES[space]:
            exp = self.compare(self.best_kernel(), override, "grid", hyp)
            results.append(exp)
        return results

    def final(self):
        """Same-session interleaved initial vs best, so the cumulative gain is not cross-session."""
        init, best = self.initial(), self.best()
        if not (init and best and best.get("timed")):
            raise SystemExit("need a timed initial baseline and best")
        imod = load_kernel_module(self.state / "initial_kernel.py")
        bmod = load_kernel_module(self.best_kernel())
        pairs = self._measure_pair((imod, init["config"]), (bmod, best["config"]), self.max_rounds)
        exp_id = self.next_exp_id("final")
        exp_dir = self.runs / exp_id
        exp_dir.mkdir()
        (exp_dir / "result.json").write_text(json.dumps(pairs, indent=2, default=str))
        rows = []
        for key, v in pairs.items():
            okb = [s for s in v["base"] if s["status"] == R.OK]
            okc = [s for s in v["cand"] if s["status"] == R.OK]
            bm, cm = stats.median([s["dsp_us"] for s in okb]), stats.median([s["dsp_us"] for s in okc])
            hb, hc = stats.median([s["host_call_ms"] for s in okb]), stats.median([s["host_call_ms"] for s in okc])
            m, h = (int(t) for t in key.split("x"))
            rows.append({**self._base_row(exp_id, "final", self.best_kernel(), best["config"],
                                          f"initial {init['exp_id']} vs best {best['exp_id']}", (m, h)),
                         "status": R.OK if cm else R.TIMING_PARSE_FAIL, "dsp_us_median": cm,
                         "base_dsp_us_median": bm, "base_dsp_spread": stats.rel_spread([s["dsp_us"] for s in okb]),
                         "dsp_gain_vs_initial": (bm - cm) / bm if bm and cm else None,
                         "host_call_ms": hc, "base_host_call_ms": hb, "n_samples": len(okc),
                         "verdict": "report", "adopted": "", "reason": "cumulative vs initial baseline, same session",
                         "raw": str((exp_dir / "result.json").relative_to(self.root))})
            print(f"[{key}] initial {bm} us -> best {cm} us DSP; host_call {hb} -> {hc} ms")
        self.append_rows(rows)

    def status(self):
        init, best = self.initial(), self.best()
        print(f"backend: {self.backend}")
        print(f"initial: {init and init['exp_id']}  best: {best and best['exp_id']}  timed={best and best.get('timed')}")
        if best:
            print(f"best config: {json.dumps(best['config'], sort_keys=True)}")
        if init and best and best.get("timed"):
            for k, m in best["metrics"].items():
                i = init["metrics"].get(k, {}).get("dsp_us_median")
                b = m.get("dsp_us_median")
                g = f"{(i - b) / i:+.1%}" if i and b else "n/a"
                print(f"  [{k}] DSP initial {i} us -> best {b} us ({g}); host_call best {m.get('host_call_ms_median')} ms")
        if self.tsv.exists():
            lines = self.tsv.read_text().splitlines()
            hdr = lines[0].split("\t")
            rows = [dict(zip(hdr, ln.split("\t"))) for ln in lines[1:]]
            rows = [r for r in rows if r.get("backend") == self.backend]
            print(f"experiments: {len({r['exp_id'] for r in rows})}  rows: {len(rows)}")
            for shape in {r["shape"] for r in rows if r.get("shape")}:
                ranked = sorted((r for r in rows if r["shape"] == shape and r["dsp_us_median"]
                                 and r["status"] == R.OK), key=lambda r: float(r["dsp_us_median"]))
                if ranked:
                    w = ranked[0]
                    print(f"  fastest measured at {shape}: {w['exp_id']} {w['dsp_us_median']} us ({w['hypothesis']})")
                else:
                    print(f"  {shape}: no timed rows (ranking excludes missing timings)")


def _failing(rows):
    bad = [r for r in rows if r.get("ok") is False]
    if not bad:
        return ""
    return ";".join(f"{r['case']}(err={r.get('max_abs_err')},{r.get('note', '')})" for r in bad[:4])


def _ir(ir, key):
    return ",".join(x["ir_hash"] for x in ir.get(key, [])) if ir else ""


def _same_ir(a, b):
    keys = set(a) | set(b)
    return bool(keys) and all([x["ir_hash"] for x in a.get(k, [])] == [x["ir_hash"] for x in b.get(k, [])]
                              and a.get(k) for k in keys)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=bench.BACKENDS, default="hexagon")
    ap.add_argument("--samples", type=int, default=6, help="samples per side per round (ABBA pairs)")
    ap.add_argument("--iterations", type=int, default=100, help="timed DSP calls per launcher invocation")
    ap.add_argument("--max-rounds", type=int, default=3)
    ap.add_argument("--shapes", default=None)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("baseline")
    b.add_argument("--force", action="store_true")
    g = sub.add_parser("grid")
    g.add_argument("--space", choices=sorted(SPACES), default="small")
    t = sub.add_parser("try")
    t.add_argument("--hypothesis", required=True)
    t.add_argument("--keep-file", action="store_true", help="do not restore kernel.py to best after a rejection")
    sub.add_parser("final")
    sub.add_parser("status")
    sub.add_parser("restore")
    a_ = sub.add_parser("adopt")
    a_.add_argument("exp_id")
    a_.add_argument("--reason", required=True)
    a = ap.parse_args(argv)

    ws = Workspace(HERE, a.backend, samples=a.samples, iterations=a.iterations,
                   max_rounds=a.max_rounds, shapes=bench.parse_shapes(a.shapes))
    if a.cmd in ("baseline", "grid", "try", "final"):
        try:
            ws.runner  # noqa: B018  (fail fast if the backend cannot run here)
            if a.backend == "hexagon":
                ok, msg = ws.runner.check_device()
                if not ok:
                    raise RuntimeError(f"adb: {msg}")
        except RuntimeError as e:
            ws.append_rows([{"exp_id": f"{a.backend[:3]}-env", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                             "kind": a.cmd, "backend": a.backend, "status": R.DEVICE_UNAVAILABLE,
                             "kernel_sha": file_sha256(ws.kernel)[:16], "verdict": "not_run", "adopted": "no",
                             "reason": str(e)}])
            print(f"status: {R.DEVICE_UNAVAILABLE}\nerror: {e}\n(recorded in results.tsv; state unchanged)")
            sys.exit(3)
    if a.cmd == "baseline":
        ws.baseline(force=a.force)
    elif a.cmd == "grid":
        backup = ws.protect_user_kernel()
        if backup:
            print(f"kernel.py differs from best; saved to {backup}")
        ws.grid(a.space)
        ws.restore(backup=False)
        print("kernel.py restored to best")
    elif a.cmd == "try":
        _, st, verdict = ws.compare(ws.kernel, None, "agent", a.hypothesis)
        if verdict != "improved" and not a.keep_file:
            ws.restore(backup=False)
            print("kernel.py restored to best (candidate kept in runs/)")
    elif a.cmd == "final":
        ws.final()
    elif a.cmd == "status":
        ws.status()
    elif a.cmd == "restore":
        backup = ws.restore()
        print(f"kernel.py restored to best{'; previous saved to ' + str(backup) if backup else ''}")
    elif a.cmd == "adopt":
        ws.adopt(a.exp_id, a.reason)


if __name__ == "__main__":
    main()
