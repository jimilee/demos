"""Experiment store (plan §12): write-once records, per-target best isolation, cache keys."""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

from .schema import dump

TSV_COLUMNS = ["experiment_id", "candidate_id", "created_at", "backend_id", "backend_kind", "workload_id",
               "contract_id", "action_id", "kind", "params", "hypothesis", "state", "failure", "failure_detail",
               "ir_hash", "artifact_hashes", "max_abs_err", "n_inputs", "measurement_source", "median",
               "unit", "decision", "cache_key", "record"]


def _atomic(path: Path, text: str):
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(text)
    tmp.replace(path)


def cache_key(**parts) -> str:
    """Includes backend/profile/IR/options/toolchain/arch/dtype/shape (plan §12)."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]


class Store:
    def __init__(self, root):
        self.root = Path(root)
        (self.root / "experiments").mkdir(parents=True, exist_ok=True)
        (self.root / "best").mkdir(parents=True, exist_ok=True)
        self.tsv = self.root / "results.tsv"

    def new_experiment(self, config: dict, lock: dict) -> str:
        eid = time.strftime("%Y%m%d-%H%M%S") + "-" + hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:6]
        d = self.root / "experiments" / eid
        d.mkdir(parents=True)
        (d / "experiment.json").write_text(dump({"experiment_id": eid, "config": config, "lock": lock}))
        return eid

    def exp_dir(self, eid) -> Path:
        return self.root / "experiments" / eid

    def write_record(self, rec) -> Path:
        d = self.exp_dir(rec.experiment_id) / "candidates"
        d.mkdir(parents=True, exist_ok=True)
        p = d / f"{rec.candidate['candidate_id']}.json"
        if p.exists():
            raise FileExistsError(f"record {p} is immutable")
        p.write_text(rec.to_json())
        self._tsv(rec, p)
        return p

    def _tsv(self, rec, path):
        m = rec.measurement or {}
        blocks = m.get("blocks") or []
        flat = sorted(x for b in blocks for x in b)
        row = {
            "experiment_id": rec.experiment_id, "candidate_id": rec.candidate["candidate_id"], "created_at": rec.created_at,
            "backend_id": rec.backend_id, "backend_kind": rec.effective_config.get("_backend_kind", ""),
            "workload_id": rec.workload_id, "contract_id": rec.contract_id, "action_id": rec.candidate["action_id"],
            "kind": rec.candidate["kind"], "params": json.dumps(rec.candidate["params"], sort_keys=True),
            "hypothesis": rec.candidate.get("hypothesis", ""), "state": rec.state, "failure": rec.failure or "",
            "failure_detail": rec.failure_detail, "ir_hash": ",".join(f"{k}:{v[:12]}" for k, v in rec.ir_hashes.items()),
            "artifact_hashes": ",".join(f"{k}:{v[:12]}" for k, v in rec.artifact_hashes.items()),
            "max_abs_err": rec.correctness.get("max_abs_err", ""), "n_inputs": rec.correctness.get("n_inputs", ""),
            "measurement_source": m.get("source", ""), "median": flat[len(flat) // 2] if flat else "",
            "unit": m.get("unit", ""), "decision": (rec.decision or {}).get("verdict", ""), "cache_key": rec.cache_key,
            "record": str(path.relative_to(self.root)),
        }
        new = not self.tsv.exists()
        with self.tsv.open("a") as f:
            if new:
                f.write("\t".join(TSV_COLUMNS) + "\n")
            f.write("\t".join(str(row[c]).replace("\t", " ").replace("\n", " ") for c in TSV_COLUMNS) + "\n")

    # ---- best-known state, isolated per (backend, workload, contract) ----
    @staticmethod
    def best_key(backend_id, workload_id, contract_id):
        return f"{backend_id}__{workload_id}__{contract_id}"

    def get_best(self, key):
        p = self.root / "best" / f"{key}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def set_best(self, key, value: dict):
        _atomic(self.root / "best" / f"{key}.json", dump(value))
