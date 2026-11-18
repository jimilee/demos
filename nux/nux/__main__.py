"""CLI:  python -m nux <command>

  env                         toolchain / device lock          -> reports/env.json
  capabilities --backend B    capability manifest              -> stdout
  workload [--id deit-tiny]   A0: pinned fetch, strict load, host reference -> workloads/*.lock.json, *.reference.json
  corpus                      A4: numeric/regression corpus    -> workloads/*.corpus.json
  ir [--host-inputs N]        A1: linalg export, verifier, recipes, host JIT vs eager, reshape reproducers
  smoke --backend B           A2: compile smoke (min op / one block / full model) per backend
  aot [--targets ...]         AOT: host .so (MLIR LLVM) + llc twin x86 + llc Hexagon object, with correctness
  export-onnx [--id ...]      vendor input artifact: ONNX export validated with onnxruntime
  import-profile ...          vendor profile (e.g. QNN JSON from qnn-profile-viewer) -> hints -> seed configs
  run --config C              controller: baseline + candidates (prepare mode unless device E2E is admissible)
  status                      best-known state per (backend, workload, contract)
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def _w(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    print(f"wrote {path.relative_to(ROOT)}")


def toolchain_lock():
    from importlib.metadata import PackageNotFoundError, version

    def v(p):
        try:
            return version(p)
        except PackageNotFoundError:
            return None

    lock = {"python": platform.python_version(), "platform": platform.platform(),
            **{p: v(p) for p in ("torch", "torch-mlir", "transformers", "huggingface-hub", "numpy", "pillow")}}
    from .backends import hexagon

    env = hexagon._discover()
    lock["hexagon"] = {k: env.get(k) for k in ("toolchain_importable", "toolchain_error", "HEXAGON_ARCH_VERSION",
                                               "HEXAGON_SDK_ROOT", "HEXAGON_TOOLS", "HEXKL_ROOT", "adb", "device_state")}
    lock["hexagon"]["source_rev_assumed"] = hexagon.SOURCE_REV
    return lock


def cmd_env(a):
    _w(REPORTS / "env.json", toolchain_lock())


def cmd_capabilities(a):
    from . import registry

    m = registry.resolve({"backend_id": a.backend}).manifest
    print(json.dumps(m.to_dict(), indent=2, default=str))


def cmd_workload(a):
    import torch

    from .workloads import deit, get

    w = get(a.id)
    t = time.perf_counter()
    lock = deit.write_lock(w) if a.id == "deit-tiny" else w.lock()
    print(f"loaded {a.id} in {time.perf_counter() - t:.1f}s: load_info={lock.get('load_info')}")
    corpus = w.corpus()
    ref = {}
    for item in corpus:
        y1 = w.reference(item["tensors"])
        y2 = w.reference(item["tensors"])
        top5 = y1[0].topk(5)
        ref[item["id"]] = {"split": item["split"], "input_sha256": item["sha256"], "logits_sha256": deit.tensor_sha256(y1),
                           "deterministic": bool(torch.equal(y1, y2)), "finite": bool(torch.isfinite(y1).all()),
                           "top5": top5.indices.tolist(), "top5_logits": [round(v, 4) for v in top5.values.tolist()],
                           "labels": [w.hf_model().config.id2label[i] for i in top5.indices.tolist()[:3]]}
    _w(ROOT / "workloads" / f"{a.id}.reference.json",
       {"workload": a.id, "reference": "framework eager fp32 (transformers ViTForImageClassification)",
        "torch": torch.__version__, "inputs": ref,
        "all_deterministic": all(r["deterministic"] for r in ref.values()), "all_finite": all(r["finite"] for r in ref.values())})


def cmd_corpus(a):
    from .corpus import GENERATOR_VERSION
    from .workloads import get

    w = get(a.id)
    c1, c2 = w.corpus(), w.corpus()
    same = [x["sha256"] for x in c1] == [x["sha256"] for x in c2]
    items = [{"id": x["id"], "split": x["split"], "sha256": x["sha256"], "shape": list(x["tensors"][0].shape),
              "min": float(x["tensors"][0].min()), "max": float(x["tensors"][0].max()),
              "beyond_processor_domain": bool(x.get("beyond_processor_domain"))} for x in c1]
    splits = {}
    for x in items:
        splits[x["split"]] = splits.get(x["split"], 0) + 1
    _w(ROOT / "workloads" / f"{a.id}.corpus.json",
       {"workload": a.id, "generator": GENERATOR_VERSION, "regenerates_identically": same, "splits": splits,
        "task_accuracy": "pending: no licensed labeled evaluation set in this environment", "items": items})


def cmd_ir(a):
    import numpy as np

    from . import contract as C
    from . import mlir_host as H
    from .workloads import get

    w = get(a.id)
    out = {"workload": a.id, "stage": "linalg-on-tensors"}
    t = time.perf_counter()
    mod = H.export(w.module(), w.example_inputs(), w.func_name)
    out["export_s"] = round(time.perf_counter() - t, 2)
    out["verify"] = H.verify(mod)
    out["stats"] = H.ir_stats(mod)
    cache = ROOT / ".cache" / "ir"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / f"{a.id}.linalg.mlirbc").write_bytes(mod.bytecode)
    out["ir_file"] = f".cache/ir/{a.id}.linalg.mlirbc (not committed; sha256 {mod.sha256})"
    repro_dir = REPORTS / "reproducers"
    repro_dir.mkdir(parents=True, exist_ok=True)
    rep = H.extract_reshape_reproducers(mod, min_rank=5)
    out["rank5_reshapes_in_export"] = len(rep)
    contract = C.get(a.contract)
    corpus = w.corpus()
    sel = [x for x in corpus if x["split"] == "tune"][: a.host_inputs] + [x for x in corpus if x["split"] == "stress"][: a.host_stress]
    out["recipes"] = {}
    variants = {"baseline": mod}
    for rid, r in H.RECIPES.items():
        ok, why = H.recipe_applicable(r, mod)
        entry = {"applicable": ok, "reason": why}
        if ok:
            try:
                m2 = H.apply_recipe(mod, r)
                entry.update(changed=m2.sha256 != mod.sha256, verify=H.verify(m2), stats={k: v for k, v in H.ir_stats(m2).items() if k != "ops"})
                r5 = H.extract_reshape_reproducers(m2, min_rank=5)
                entry["rank5_reshapes"] = len(r5)
                for i, x in enumerate(r5[:10]):
                    (repro_dir / f"{a.id}__{rid}__rank{x['max_rank']}_{i}.mlir").write_text(x["mlir"])
                if entry["changed"] and rid in a.jit_recipes.split(","):
                    variants[rid] = m2
            except Exception as e:  # noqa: BLE001
                entry["error"] = f"{type(e).__name__}: {e}"
        out["recipes"][rid] = entry
    unit = H.extract_reshape_reproducers(mod, min_rank=99, include_unit_dims=True, limit=8)
    for i, x in enumerate(unit):
        (repro_dir / f"{a.id}__export__unitdim_{i}.mlir").write_text(x["mlir"])
    out["unit_dim_reproducers_written"] = len(unit)
    out["host_execution"] = {}
    for name, m in variants.items():
        t = time.perf_counter()
        jit = H.HostJIT(m)
        per = []
        for x in sel:
            ts = time.perf_counter()
            y = jit(*[t_.numpy() for t_ in x["tensors"]])
            r = C.compare(contract, __import__("torch").from_numpy(y), w.reference(x["tensors"]))
            r.update(input_id=x["id"], run_s=round(time.perf_counter() - ts, 2))
            per.append(r)
        out["host_execution"][name] = {"lower_s": round(jit.lower_s, 2), "jit_s": round(jit.jit_s, 2),
                                       "summary": C.summarize(per),
                                       "per_input": [{k: r.get(k) for k in ("input_id", "ok", "max_abs_err", "mean_abs_err", "top1_match", "top5_overlap", "run_s")} for r in per],
                                       "total_s": round(time.perf_counter() - t, 1),
                                       "note": "host CPU loops lowering for correctness only; timings are not performance data"}
        print(f"[host-jit {name}] ok={out['host_execution'][name]['summary']['ok']} max_abs_err={out['host_execution'][name]['summary']['max_abs_err']}")
    _ = np
    _w(REPORTS / f"a1_{a.id}_ir.json", out)


def cmd_smoke(a):
    from . import mlir_host as H
    from . import registry
    from .schema import Candidate
    from .workloads import get

    w = get(a.id)
    b = registry.resolve({"backend_id": a.backend})
    rows = []
    for name, mod_, inputs in w.submodules():
        r = {"level": name, "backend": a.backend}
        t = time.perf_counter()
        try:
            m = H.export(mod_, inputs, name)
            r["export"] = {"ok": True, "s": round(time.perf_counter() - t, 2), "verify": H.verify(m), "sha256": m.sha256,
                           "max_reshape_rank": H.ir_stats(m)["max_reshape_rank"]}
        except Exception as e:  # noqa: BLE001
            r["export"] = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            rows.append(r)
            continue
        class _Sub:  # minimal workload view for the compiler adapter
            workload_id, func_name = name, name
            def module(s): return mod_
            def example_inputs(s): return inputs
        try:
            art = b.compiler.capture_baseline(_Sub())
            r["target_compile"] = ("pending: " + art.pending) if art.pending else (
                "passed" if art.target_compiled else "not a target backend")
            r["artifact_hashes"] = art.artifact_hashes
        except Exception as e:  # noqa: BLE001
            r["target_compile"] = f"failed: {type(e).__name__}: {e}"
        rows.append(r)
        print(f"[{name}] export={r['export'].get('ok')} target_compile={r['target_compile'][:90]}")
    _ = Candidate
    _w(REPORTS / f"a2_smoke_{a.backend}.json", {"workload": a.id, "backend": a.backend, "lock": toolchain_lock(), "levels": rows})


def cmd_aot(a):
    import shutil
    import subprocess

    import torch

    from . import aot
    from . import contract as C
    from . import mlir_host as H
    from .workloads import get

    w = get(a.id)
    contract = C.get(a.contract)
    mod = H.export(w.module(), w.example_inputs(), w.func_name)
    corpus = w.corpus()
    sel = [x for x in corpus if x["split"] == "tune"][: a.inputs] + [x for x in corpus if x["split"] == "stress"][: a.stress]
    d = aot.workdir()
    rt = aot.result_type(mod)
    rep = {"workload": a.id, "ir_sha256": mod.sha256, "llc": aot.llc_path(), "llc_version": aot.llc_version(),
           "llc_has_hexagon": "hexagon" in aot.llc_targets(), "workdir": str(d.relative_to(ROOT)), "routes": {}}

    def check(lib):
        per = []
        for x in sel:
            t = time.perf_counter()
            y = torch.from_numpy(lib(*[t_.numpy() for t_ in x["tensors"]]))
            r = C.compare(contract, y, w.reference(x["tensors"]))
            r.update(input_id=x["id"], run_s=round(time.perf_counter() - t, 2))
            per.append({k: r.get(k) for k in ("input_id", "ok", "max_abs_err", "top1_match", "run_s")})
        return {"ok": all(p["ok"] for p in per), "max_abs_err": max(p["max_abs_err"] for p in per), "per_input": per}

    o = aot.host_object_via_mlir(mod, d / "host_mlir.o")
    so = aot.link_shared([d / "host_mlir.o"], d / "libhost_mlir.so")
    rep["routes"]["host-mlir-llvm"] = {**o, "so_sha256": so["sha256"],
                                       "correctness": check(aot.AOTLibrary(d / "libhost_mlir.so", mod.func_name, rt))}
    print(f"[host-mlir-llvm] ok={rep['routes']['host-mlir-llvm']['correctness']['ok']}")
    t = time.perf_counter()
    ll, counts = aot.downgrade_llvm_ir(aot.to_llvm_ir(mod))
    ll, nmm = aot.expand_fmaxmin(ll)
    (d / "model.ll").write_text(ll)
    rep["downgrade"] = {"llvm_ir_bytes": len(ll), "rewrites": counts, "fmaxmin_expanded": nmm,
                        "seconds": round(time.perf_counter() - t, 1), "llvm_ir_sha256": H.sha(ll)}
    for tgt in a.targets.split(","):
        out = d / f"model_{tgt}.o"
        try:
            r = aot.llc_compile(d / "model.ll", tgt, out)
        except Exception as e:  # noqa: BLE001
            rep["routes"][f"llc-{tgt}"] = {"ok": False, "error": str(e)[:800]}
            print(f"[llc {tgt}] FAILED {str(e)[:120]}")
            continue
        entry = dict(r)
        nm = shutil.which("llvm-nm-18") or shutil.which("llvm-nm")
        if nm:
            syms = subprocess.run([nm, "--defined-only", str(out)], capture_output=True, text=True).stdout
            entry["entry_symbol"] = [l.split()[-1] for l in syms.splitlines() if mod.func_name in l][:4]
            undef = subprocess.run([nm, "--undefined-only", str(out)], capture_output=True, text=True).stdout
            entry["undefined_symbols"] = sorted({l.split()[-1] for l in undef.splitlines()})
        if tgt == "x86_64":
            aot.link_shared([out], d / "libtwin.so")
            entry["correctness"] = check(aot.AOTLibrary(d / "libtwin.so", mod.func_name, rt))
            entry["note"] = "twin build of the exact downgraded IR used for the Hexagon object"
        else:
            entry["execution"] = "pending: needs Hexagon SDK runtime (FastRPC skel / run_main_on_hexagon) and a device"
        rep["routes"][f"llc-{tgt}"] = entry
        print(f"[llc {tgt}] {entry['file']} {entry['bytes']} B {entry['seconds']}s"
              + (f" ok={entry['correctness']['ok']}" if "correctness" in entry else ""))
    _w(REPORTS / f"aot_{a.id}.json", rep)


def cmd_export_onnx(a):
    from . import contract as C
    from . import onnx_export as O
    from .workloads import get

    w = get(a.id)
    out = ROOT / ".cache" / "vendor" / f"{a.id}.onnx"
    info = O.export(w, out)
    info["validation"] = O.validate(w, out, C.get(a.contract), w.corpus())
    info["path"] = str(out.relative_to(ROOT)) + " (not committed)"
    _w(REPORTS / f"vendor_input_{a.id}.json", info)
    print(f"onnx {info['n_nodes']} nodes sha256={info['sha256'][:16]} validation ok={info['validation']['ok']} "
          f"max_abs_err={info['validation']['max_abs_err']:.3g}")


def cmd_import_profile(a):
    from .importers import pipeline
    from .workloads import get

    meta = get(a.id).meta() if a.id else None
    onnx_path = Path(a.onnx) if a.onnx else (ROOT / ".cache" / "vendor" / f"{a.id}.onnx")
    s = pipeline.run(a.format, Path(a.file), a.label, a.targets.split(","), onnx_path if onnx_path.exists() else None,
                     a.build_config, a.converter_args, a.precision, meta)
    print(f"layers={s['n_layers']} runs={s['summary'].get('runs')} mapping={s['meta']['name_mapping']}")
    for h in s["hints"]:
        print(f"  hint {h[0]:24} [{h[1]}] {h[2]}")
    for t, v in s["seeds"].items():
        print(f"  seeds for {t}: {v['n_seeds']} candidates ({v['n_skipped']} skipped) -> {v['config']}")
    for wmsg in s["meta"]["warnings"]:
        print("  WARNING:", wmsg)


def cmd_run(a):
    from . import contract as C
    from . import registry, stats
    from .controller import Controller
    from .schema import Candidate
    from .store import Store
    from .workloads import get

    cfg = json.loads(Path(a.config).read_text())
    w = get(cfg["workload_id"])
    corpus = w.corpus()
    lim = cfg.get("corpus_limit")
    if lim:
        keep, n = [], {}
        for x in corpus:
            n[x["split"]] = n.get(x["split"], 0) + 1
            if n[x["split"]] <= lim.get(x["split"], 1 << 30):
                keep.append(x)
        corpus = keep
    b = registry.resolve(cfg)
    pol = stats.Policy(**cfg.get("policy", {}))
    ctl = Controller(b, w, C.get(cfg["evaluation_contract_id"]), Store(ROOT / cfg.get("store", "store")), pol, cfg,
                     corpus=corpus, toolchain_lock=toolchain_lock())
    print(f"experiment {ctl.eid}: backend={b.backend_id} traits={b.traits} mode={ctl.mode}")
    r = ctl.run_baseline()
    print(f"  baseline -> {r.state} {r.failure or ''} {r.failure_detail[:90]}")
    for cd in cfg.get("candidates", []):
        r = ctl.run_candidate(Candidate(**cd))
        print(f"  {cd['action_id']:32} -> {r.state:13} {r.failure or '':20} {(r.decision or {}).get('verdict', '')} {r.failure_detail[:70]}")


def cmd_status(a):
    store = ROOT / a.store
    for p in sorted((store / "best").glob("*.json")):
        b = json.loads(p.read_text())
        print(f"{p.stem}: state={b.get('state')} mode={b.get('mode')} chain={len(b.get('chain', []))} {b.get('note', '')}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="python -m nux", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("env")
    s = sub.add_parser("capabilities"); s.add_argument("--backend", required=True)
    for n in ("workload", "corpus"):
        s = sub.add_parser(n); s.add_argument("--id", default="deit-tiny")
    s = sub.add_parser("ir"); s.add_argument("--id", default="deit-tiny")
    s.add_argument("--contract", default="deit-tiny-fp32-logits-v0")
    s.add_argument("--host-inputs", type=int, default=2); s.add_argument("--host-stress", type=int, default=1)
    s.add_argument("--jit-recipes", default="fold-unit-extent-dims")
    s = sub.add_parser("smoke"); s.add_argument("--id", default="deit-tiny"); s.add_argument("--backend", default="hexagon-mlir")
    s = sub.add_parser("aot"); s.add_argument("--id", default="deit-tiny")
    s.add_argument("--contract", default="deit-tiny-fp32-logits-v0")
    s.add_argument("--targets", default="x86_64,hexagon-v73")
    s.add_argument("--inputs", type=int, default=2); s.add_argument("--stress", type=int, default=1)
    s = sub.add_parser("export-onnx"); s.add_argument("--id", default="deit-tiny")
    s.add_argument("--contract", default="deit-tiny-fp32-logits-v0")
    s = sub.add_parser("import-profile")
    s.add_argument("--format", required=True, help="qnn-json | chrometrace")
    s.add_argument("--file", required=True); s.add_argument("--label", required=True)
    s.add_argument("--id", default="deit-tiny"); s.add_argument("--onnx", default=None)
    s.add_argument("--targets", default="hexagon-mlir,llvm-aot-hexagon")
    s.add_argument("--build-config", default=None, help="HTP backend config JSON used for the context binary")
    s.add_argument("--converter-args", default=None, help="qairt-converter arguments used (to detect preserve_io)")
    s.add_argument("--precision", default="unknown", help="fp16 | int8 | ... (vendor build precision)")
    s = sub.add_parser("run"); s.add_argument("--config", required=True)
    s = sub.add_parser("status"); s.add_argument("--store", default="store")
    a = ap.parse_args(argv)
    globals()[f"cmd_{a.cmd.replace('-', '_')}"](a)


if __name__ == "__main__":
    sys.exit(main())
