"""Run one experiment config: baseline + candidates through the controller."""
from __future__ import annotations

from ..core import contract as C
from ..engine import registry
from ..engine.config import load as load_config
from ..engine.controller import Controller
from ..storage.artifact_store import ArtifactStore
from ..storage.store import Store
from ..workloads import get
from . import ROOT
from .environment import toolchain_lock


def limit_corpus(corpus, limits):
    if not limits:
        return corpus
    keep, n = [], {}
    for x in corpus:
        n[x["split"]] = n.get(x["split"], 0) + 1
        if n[x["split"]] <= limits.get(x["split"], 1 << 30):
            keep.append(x)
    return keep


def run_experiment(config_path, on_event=None, store_root=None, artifacts_root=None) -> dict:
    """on_event(kind, payload) receives 'start', 'record' and 'done' events (the CLI prints them)."""
    emit = on_event or (lambda kind, payload: None)
    xc = load_config(config_path)  # strict: unknown keys, bad ids, bad candidates are errors here
    cfg = xc.as_runtime_dict()
    w = get(xc.workload_id)
    b = registry.resolve(cfg)
    ac = xc.artifact_cache
    artifacts = ArtifactStore(artifacts_root or ROOT / ac.root, ac.policy()) if ac.enabled else None
    ctl = Controller(b, w, C.get(xc.evaluation_contract_id), Store(store_root or ROOT / xc.store), xc.policy, cfg,
                     corpus=limit_corpus(w.corpus(), xc.corpus_limit), toolchain_lock=toolchain_lock(), artifacts=artifacts)
    emit("start", {"experiment_id": ctl.eid, "backend_id": b.backend_id, "traits": b.traits, "mode": ctl.mode})
    records = [ctl.run_baseline()]
    emit("record", records[-1])
    for cand in xc.candidates:
        records.append(ctl.run_candidate(cand))
        emit("record", records[-1])
    summary = {"experiment_id": ctl.eid, "backend_id": b.backend_id, "mode": ctl.mode, "records": records,
               "artifact_store": artifacts.stats() if artifacts is not None else None}
    emit("done", summary)
    return summary


def best_states(store_dir) -> list:
    import json
    from pathlib import Path

    out = []
    for p in sorted((Path(store_dir) / "best").glob("*.json")):
        b = json.loads(p.read_text())
        out.append({"key": p.stem, "state": b.get("state"), "mode": b.get("mode"), "chain": len(b.get("chain", [])),
                    "note": b.get("note", "")})
    return out
