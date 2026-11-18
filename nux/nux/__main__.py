"""CLI:  python -m nux <command>      (argument parsing and printing only; logic lives in nux/services/)

  env                         toolchain / device lock (all registered backends) -> reports/env.json
  capabilities --backend B    capability manifest              -> stdout
  workload [--id deit-tiny]   A0: pinned fetch, strict load, host reference -> workloads/*.lock.json, *.reference.json
  corpus                      A4: numeric/regression corpus    -> workloads/*.corpus.json
  ir [--host-inputs N]        A1: linalg export, verifier, recipes, host JIT vs eager, reshape reproducers
  smoke --backend B           A2: compile smoke (min op / one block / full model) per backend
  aot                         AOT report through the llvm-aot-host and llvm-aot-hexagon backends
  export-onnx [--id ...]      vendor input artifact: ONNX export validated with onnxruntime
  import-profile ...          vendor profile (e.g. QNN JSON from qnn-profile-viewer) -> hints -> seed configs
  run --config C              controller: baseline + candidates (prepare mode unless device E2E is admissible)
  cache stats|gc|clean-workdirs  artifact store (.cache/artifacts) and leftover AOT work directories
  status                      best-known state per (backend, workload, contract)
"""
from __future__ import annotations

import argparse
import json
import sys

from . import services as S


def _wrote(path):
    print(f"wrote {S.rel(path)}")


def cmd_env(a):
    from .services.environment import write_env_report

    _wrote(write_env_report())


def cmd_capabilities(a):
    from .services.environment import capabilities

    print(json.dumps(capabilities(a.backend), indent=2, default=str))


def cmd_workload(a):
    from .services.workload_reports import lock_and_reference

    r = lock_and_reference(a.id)
    print(f"loaded {a.id} in {r['load_s']:.1f}s: load_info={r['load_info']}")
    _wrote(r["path"])


def cmd_corpus(a):
    from .services.workload_reports import corpus_manifest

    _wrote(corpus_manifest(a.id)["path"])


def cmd_ir(a):
    from .services.ir_report import a1_report

    _wrote(a1_report(a.id, a.contract, a.host_inputs, a.host_stress, tuple(a.jit_recipes.split(",")))["path"])


def cmd_smoke(a):
    from .services.smoke import compile_smoke

    _wrote(compile_smoke(a.id, a.backend)["path"])


def cmd_aot(a):
    from .services.aot_report import aot_report

    r = aot_report(a.id, a.contract, a.inputs, a.stress, a.keep_workdir)
    _wrote(r["path"])
    if a.keep_workdir:
        print("kept work directories:", ", ".join(r["report"]["workdirs"]))


def cmd_export_onnx(a):
    from .services.vendor import export_onnx

    r = export_onnx(a.id, a.contract)
    i = r["info"]
    _wrote(r["path"])
    print(f"onnx {i['n_nodes']} nodes sha256={i['sha256'][:16]} validation ok={i['validation']['ok']} "
          f"max_abs_err={i['validation']['max_abs_err']:.3g}")


def cmd_import_profile(a):
    from .services.vendor import import_profile

    s = import_profile(a.format, a.file, a.label, a.targets.split(","), a.id, a.onnx, a.build_config,
                       a.converter_args, a.precision)
    print(f"layers={s['n_layers']} runs={s['summary'].get('runs')} mapping={s['meta']['name_mapping']}")
    for h in s["hints"]:
        print(f"  hint {h[0]:24} [{h[1]}] {h[2]}")
    for t, v in s["seeds"].items():
        print(f"  seeds for {t}: {v['n_seeds']} candidates ({v['n_skipped']} skipped) -> {v['config']}")
    for wmsg in s["meta"]["warnings"]:
        print("  WARNING:", wmsg)


def _print_event(kind, p):
    if kind == "start":
        print(f"experiment {p['experiment_id']}: backend={p['backend_id']} traits={p['traits']} mode={p['mode']}")
        if p["traits"].get("measurement_tier") == "synthetic":
            print("  WARNING: synthetic-tier backend (test/structure mock). Timings are not measurements and outputs "
                  "are the framework reference; see nux/CLAUDE.md 'Mock backends - caveats'.")
    elif kind == "record":
        hit = " [cache hit]" if p.cache_hit else ""
        name = p.candidate["action_id"]
        print(f"  {name:32} -> {p.state:15} {p.failure or '':20} {(p.decision or {}).get('verdict', '')} "
              f"{p.failure_detail[:70]}{hit}")
    elif kind == "done" and p["artifact_store"]:
        s = p["artifact_store"]
        print(f"  artifact store: {s['entries']} entries ({s['pinned']} pinned), {s['bytes'] / 1e6:.0f} MB / "
              f"{s['max_bytes'] / 1e9:.1f} GB cap")


def cmd_run(a):
    from .services.experiment import run_experiment

    run_experiment(a.config, on_event=_print_event)


def cmd_cache(a):
    from .services import cache

    st = cache.store(a.max_gb)
    if a.action == "gc":
        print(json.dumps(st.gc(), indent=2))
    elif a.action == "clean-workdirs":
        r = cache.clean_workdirs()
        print(f"removed {r['removed']} leftover AOT work directories ({r['bytes'] / 1e9:.2f} GB) under {r['base']}")
    s = st.stats()
    print(f"artifact store: {s['entries']} entries ({s['pinned']} pinned), {s['bytes'] / 1e6:.0f} MB, "
          f"cap {s['max_bytes'] / 1e9:.1f} GB")


def cmd_status(a):
    from .services.experiment import best_states

    for b in best_states(S.ROOT / a.store):
        print(f"{b['key']}: state={b['state']} mode={b['mode']} chain={b['chain']} {b['note']}")


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
    s.add_argument("--inputs", type=int, default=2); s.add_argument("--stress", type=int, default=1)
    s.add_argument("--keep-workdir", action="store_true")
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
    s = sub.add_parser("cache"); s.add_argument("action", choices=["stats", "gc", "clean-workdirs"])
    s.add_argument("--max-gb", type=float, default=5.0)
    a = ap.parse_args(argv)
    globals()[f"cmd_{a.cmd.replace('-', '_')}"](a)


if __name__ == "__main__":
    sys.exit(main())
