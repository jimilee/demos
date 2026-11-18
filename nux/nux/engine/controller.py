"""Experiment controller (plan §11). Vendor-neutral: talks only to the adapter bundle.

Responsibilities (and nothing else):
  * run the stage pipeline (nux/stages.py) for one candidate,
  * write exactly one immutable record from the final CandidateContext,
  * maintain the best-known state (update only on `accepted`; otherwise the
    previous best stays active = restore), budgets, and restart recovery.

Modes (derived from declared backend traits, never from backend names)
  prepare   the backend cannot produce acceptance evidence (outputs not computed on a
            device, or timings not device-tier E2E). Candidates climb the evidence
            ladder as far as this environment allows; nothing is accepted.
  device    manifest.acceptance_capable: device outputs + device-tier E2E timings.
            Paired ABBA blocks (nux/measurement.py), evaluator decision.

Measurement in prepare mode happens only when the experiment config sets
`informational_measurement: true`. Such numbers are recorded but never admissible.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, replace

from ..core import stats
from ..core.schema import Candidate, CandidateRecord, Evidence, Failure, State
from ..storage.artifact_store import ArtifactStore, compile_key
from ..storage.store import Store
from ..toolchain import frontend
from . import stages


class Controller:
    def __init__(self, bundle, workload, contract, store: Store, policy: stats.Policy, config: dict,
                 corpus=None, toolchain_lock=None, artifacts: ArtifactStore | None = None):
        self.b, self.w, self.c, self.store, self.policy, self.cfg = bundle, workload, contract, store, policy, config
        self.corpus = corpus if corpus is not None else workload.corpus()
        self.lock = toolchain_lock or {}
        self.b.runtime.bind_workload(workload)
        self.key = Store.best_key(bundle.backend_id, workload.workload_id, contract.contract_id)
        self.eid = store.new_experiment(config, {"toolchain": self.lock, "workload": workload.lock(),
                                                 "capabilities": bundle.manifest.to_dict(), "policy": policy.to_dict()})
        budget = config.get("budget", {})
        self.max_candidates = int(budget.get("max_candidates", 50))
        self.max_consecutive_failures = int(budget.get("max_consecutive_failures", 5))
        self.n_run = 0
        self.n_consecutive_fail = 0
        self.best_artifact = None
        self.best_session = None
        self.best_record = None
        self.artifacts = artifacts  # None disables the artifact store (no lookups, nothing kept)
        wl = workload.lock()
        self._workload_hash = hashlib.sha256(json.dumps(wl, sort_keys=True, default=str).encode()).hexdigest()

    # ---- artifact keys ------------------------------------------------------
    @property
    def best_key_value(self):
        return self.best_record.cache_key if self.best_record else None

    def compile_key(self, cand: Candidate, parent_key: str | None) -> str:
        m = self.b.manifest
        return compile_key(backend=self.b.backend_id, compiler=m.compiler_profile, runtime=m.runtime_profile,
                           arch=m.device_profile.get("arch"), frontend=frontend.frontend_version(), toolchain=self.lock,
                           workload=self._workload_hash, parent=parent_key or "root",
                           candidate={"action_id": cand.action_id, "kind": cand.kind, "params": cand.params,
                                      "scope": cand.scope, "payload": cand.content_hash() if cand.payload_files else None})

    # ---- derived policy ---------------------------------------------------
    @property
    def mode(self):
        return "device" if self.b.manifest.acceptance_capable else "prepare"

    @property
    def measure_enabled(self) -> bool:
        return self.b.manifest.measures_e2e and (self.mode == "device" or bool(self.cfg.get("informational_measurement", False)))

    def perf_inputs(self):
        tune = [c for c in self.corpus if c["split"] == "tune"]
        return (tune[0] if tune else self.corpus[0])["tensors"]

    # ---- record -----------------------------------------------------------
    def _write(self, ctx: stages.CandidateContext):
        art = ctx.art
        eff = dict(art.effective_config) if art else {}
        rec = CandidateRecord(
            experiment_id=self.eid, candidate=asdict(ctx.cand), backend_id=self.b.backend_id,
            workload_id=self.w.workload_id, contract_id=self.c.contract_id, state=ctx.state.value,
            failure=ctx.failure.value if ctx.failure else None, failure_detail=ctx.detail,
            stages=[asdict(s) for s in ctx.stages],
            requested_config=dict(art.requested_config) if art else dict(ctx.cand.params),
            effective_config=eff, ir_hashes=dict(art.ir_hashes) if art else {},
            artifact_hashes=dict(art.artifact_hashes) if art else {}, correctness=ctx.corr or {},
            measurement=stages.measurement_for_record(ctx.meas), decision=ctx.decision,
            toolchain_lock=self.lock, backend_traits=self.b.manifest.traits(),
            cache_key=ctx.key or "", cache_hit=bool(art and art.cache_hit))
        path = self.store.write_record(rec)
        ctx.record, ctx.record_path = rec, path
        if self.artifacts is not None and ctx.key:
            self.artifacts.after_record(ctx.key, rejected=ctx.rejected)
        return rec

    def _keep(self, ctx, reason):
        if self.artifacts is not None and ctx.key:
            self.artifacts.pin(ctx.key, reason)
            self.artifacts.gc()

    def _rel(self, path):
        try:
            return str(path.relative_to(self.store.root))
        except ValueError:
            return str(path)

    def evaluate(self, cand: Candidate, baseline=False) -> stages.CandidateContext:
        ctx = stages.run(self, stages.CandidateContext(cand, baseline=baseline))
        self._write(ctx)
        return ctx

    # ---- public API ---------------------------------------------------------
    def run_baseline(self):
        ctx = self.evaluate(Candidate("baseline", "baseline", hypothesis="backend defaults"), baseline=True)
        rec = ctx.record
        if not ctx.rejected:
            self.best_artifact, self.best_session, self.best_record = ctx.art, ctx.session, rec
            note = ("baseline is the best-known artifact until a device-accepted candidate exists" if ctx.session is not None
                    else f"baseline stopped at {rec.state}: {rec.failure_detail}")
            self.store.set_best(self.key, {"record": self._rel(ctx.record_path), "chain": [], "state": rec.state,
                                           "mode": self.mode, "note": note, "artifact_key": ctx.key})
            self._keep(ctx, f"baseline:{self.key}")
        return rec

    def run_candidate(self, cand: Candidate):
        if self.n_run >= self.max_candidates or self.n_consecutive_fail >= self.max_consecutive_failures:
            ctx = stages.CandidateContext(cand)
            ctx.add("budget", Evidence.FAILED, "budget exhausted")
            ctx.stop(State.REJECTED, Failure.BUDGET_EXHAUSTED, "budget")
            return self._write(ctx)
        self.n_run += 1
        parent = self.best_record.candidate["candidate_id"] if self.best_record else None
        ctx = self.evaluate(replace(cand, parent_id=parent, candidate_id=""))
        rec = ctx.record
        self.n_consecutive_fail = self.n_consecutive_fail + 1 if ctx.rejected else 0
        if ctx.state == State.ACCEPTED:  # otherwise the previous best stays active (= restore)
            prev = self.store.get_best(self.key) or {"chain": []}
            self.best_artifact, self.best_session, self.best_record = ctx.art, ctx.session, rec
            self.store.set_best(self.key, {"record": self._rel(ctx.record_path), "chain": prev["chain"] + [rec.candidate],
                                           "state": rec.state, "mode": self.mode, "artifact_key": ctx.key})
            self._keep(ctx, f"best:{self.key}")
        elif self.artifacts is not None:
            self.artifacts.gc()
        return rec

    def restore_best(self):
        """Rebuild the best artifact after a restart: from the artifact store when the pinned entries are
        intact, otherwise by replaying the stored chain through the compiler. Returns (artifact, source)."""
        best = self.store.get_best(self.key) or {"chain": []}
        chain = [Candidate("baseline", "baseline")] + [Candidate(**cd) for cd in best.get("chain", [])]
        key, art, source = None, None, "store"
        for i, c in enumerate(chain):
            key = self.compile_key(c, key if i else None)
            entry = self.artifacts.get(key) if self.artifacts is not None else None
            loaded = self.b.compiler.load_artifact(self.w, entry, "restore") if entry else None
            if loaded is not None:
                art = loaded
                continue
            source = "recompiled"
            art = self.b.compiler.capture_baseline(self.w) if i == 0 else self.b.compiler.compile_candidate(self.w, c, art)
        self.best_artifact = art
        return art, source
