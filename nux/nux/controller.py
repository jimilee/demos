"""Experiment controller (plan §11). Vendor-neutral: talks only to the adapter bundle.

Modes (derived from declared backend traits, never from backend names)
  prepare   the backend cannot produce acceptance evidence (outputs not computed on a
            device, or timings not device-tier E2E). Candidates climb the evidence
            ladder as far as this environment allows; nothing is accepted.
  device    manifest.acceptance_capable: device outputs + device-tier E2E timings.
            Paired ABBA blocks, evaluator decision, best update or restore.

Measurement in prepare mode happens only when the experiment config sets
`informational_measurement: true` (e.g. structure checks with synthetic timings).
Such numbers are recorded but can never be admissible.
"""
from __future__ import annotations

import time
from dataclasses import asdict, replace

from . import evaluator, stats
from .adapters import BackendError, DeviceUnavailable, Unsupported
from .capability import validate_action
from .schema import Candidate, CandidateRecord, Evidence, Failure, Measurement, StageRecord, State
from .store import Store, cache_key


class Controller:
    def __init__(self, bundle, workload, contract, store: Store, policy: stats.Policy, config: dict,
                 corpus=None, toolchain_lock=None):
        self.b, self.w, self.c, self.store, self.policy, self.cfg = bundle, workload, contract, store, policy, config
        self.corpus = corpus if corpus is not None else workload.corpus()
        self.lock = toolchain_lock or {}
        self.b.runtime.bind_workload(workload)
        self.key = Store.best_key(bundle.backend_id, workload.workload_id, contract.contract_id)
        self.eid = store.new_experiment(config, {"toolchain": self.lock, "workload": workload.lock(),
                                                 "capabilities": bundle.manifest.to_dict(), "policy": policy.to_dict()})
        self.max_candidates = int(config.get("budget", {}).get("max_candidates", 50))
        self.max_consecutive_failures = int(config.get("budget", {}).get("max_consecutive_failures", 5))
        self.n_run = 0
        self.n_consecutive_fail = 0
        self.best_artifact = None
        self.best_session = None
        self.best_record = None

    # ------------------------------------------------------------------
    @property
    def mode(self):
        return "device" if self.b.manifest.acceptance_capable else "prepare"

    @property
    def _measure(self) -> bool:
        m = self.b.manifest
        return m.measures_e2e and (self.mode == "device" or bool(self.cfg.get("informational_measurement", False)))

    @property
    def _outputs_on_device(self) -> bool:
        return self.b.manifest.execution.executes_on == "device"

    def _record(self, cand, state, stages, art=None, failure=None, detail="", corr=None, meas=None, decision=None):
        eff = dict(art.effective_config) if art else {}
        rec = CandidateRecord(
            experiment_id=self.eid, candidate=asdict(cand), backend_id=self.b.backend_id,
            workload_id=self.w.workload_id, contract_id=self.c.contract_id, state=state.value if isinstance(state, State) else state,
            failure=failure.value if isinstance(failure, Failure) else failure, failure_detail=detail,
            stages=[asdict(s) for s in stages], requested_config=dict(art.requested_config) if art else dict(cand.params),
            effective_config=eff, ir_hashes=dict(art.ir_hashes) if art else {},
            artifact_hashes=dict(art.artifact_hashes) if art else {}, correctness=corr or {},
            measurement=asdict(meas) if isinstance(meas, Measurement) else meas, decision=decision,
            toolchain_lock=self.lock, backend_traits=self.b.manifest.traits(),
            cache_key=cache_key(backend=self.b.backend_id, compiler=self.b.manifest.compiler_profile,
                                runtime=self.b.manifest.runtime_profile, ir=art.ir_hashes if art else None,
                                cfg=eff, toolchain=self.lock, arch=self.b.manifest.device_profile.get("arch"),
                                meta=self.w.meta()))
        path = self.store.write_record(rec)
        return rec, path

    def _correctness(self, session, splits):
        return evaluator.check_correctness(lambda ts: self.b.runtime.execute_and_wait(session, ts),
                                           self.w, self.c, self.corpus, splits=splits)

    # ------------------------------------------------------------------
    def evaluate(self, cand: Candidate, baseline=False):
        """Run one candidate up the evidence ladder. Returns (record, artifact, session)."""
        stages = [StageRecord("proposed", Evidence.PASSED.value)]
        state = State.PROPOSED
        ok, why = validate_action(self.b.manifest, cand, self.w.meta())
        if not ok:
            stages.append(StageRecord("capability", Evidence.FAILED.value, why))
            return self._record(cand, State.REJECTED, stages, failure=Failure.UNSUPPORTED, detail=why)[0], None, None
        stages.append(StageRecord("capability", Evidence.PASSED.value, why))

        t = time.perf_counter()
        try:
            art = (self.b.compiler.capture_baseline(self.w) if baseline else
                   self.b.compiler.compile_candidate(self.w, cand, self.best_artifact))
            if not art.pending:
                art = self.b.compiler.integrate_candidate(self.w, art)
        except BackendError as e:
            stages.append(StageRecord("compile", Evidence.FAILED.value, str(e), time.perf_counter() - t))
            fail = Failure(e.failure) if e.failure in Failure._value2member_map_ else Failure.COMPILE_FAILURE
            detail = str(e)
            return self._record(cand, State.REJECTED, stages, failure=fail, detail=detail)[0], None, None
        stages.append(StageRecord("compile", Evidence.PASSED.value, art.log, time.perf_counter() - t,
                                  {"kind": art.kind, "provenance": art.provenance[-3:]}))
        if art.ir_hashes:
            state = State.IR_VALID
            stages.append(StageRecord("ir_verify", Evidence.PASSED.value, "MLIR verifier (structural only)"))
        if self.best_record and art.ir_hashes and art.ir_hashes == self.best_record.ir_hashes and \
                art.effective_config == self.best_record.effective_config:
            stages.append(StageRecord("dedupe", Evidence.PASSED.value, "IR and effective config identical to best: no change possible"))
            return self._record(cand, state, stages, art, detail="identical to best")[0], art, None
        if art.pending:  # environment limit, not a candidate failure: later evidence stays pending
            stages.append(StageRecord("target_compile", Evidence.PENDING.value, art.pending))
            if baseline:
                self.best_artifact = art
            return self._record(cand, state, stages, art, detail=art.pending)[0], art, None
        if art.target_compiled and self.b.manifest.execution.target_codegen:
            state = State.TARGET_COMPILED
        if art.twin is not None:  # host build of the exact target-bound IR: semantics check, not device evidence
            import torch

            tc = evaluator.check_correctness(
                lambda ts: torch.from_numpy(art.twin(*[t_.detach().contiguous().numpy() for t_ in ts])),
                self.w, self.c, self.corpus, splits=tuple(self.cfg.get("correctness_splits", ["tune", "stress"])))
            if not tc["ok"]:
                stages.append(StageRecord("twin_correctness", Evidence.FAILED.value, str(tc["failed_inputs"][:3])))
                return self._record(cand, State.REJECTED, stages, art, Failure.CORRECTNESS_FAILURE,
                                    "twin: " + str(tc["failed_inputs"][:3]), _slim(tc))[0], art, None
            stages.append(StageRecord("twin_correctness", Evidence.PASSED.value,
                                      f"{art.twin_note}: max_abs_err={tc['max_abs_err']:.3g} on {tc['n_inputs']} inputs"))
            self._twin_corr = _slim(tc)
        else:
            self._twin_corr = None

        # ---- correctness ----
        try:
            session = self.b.runtime.prepare_session(art)
        except DeviceUnavailable as e:
            stages.append(StageRecord("device_correctness", Evidence.PENDING.value, str(e)))
            return self._record(cand, state, stages, art, detail=str(e), corr=self._twin_corr)[0], art, None
        except BackendError as e:
            stages.append(StageRecord("load", Evidence.FAILED.value, str(e)))
            return self._record(cand, State.REJECTED, stages, art, Failure(e.failure), str(e))[0], art, None
        splits = tuple(self.cfg.get("correctness_splits", ["tune", "stress"]))
        corr = self._correctness(session, splits)
        cstage = "device_correctness" if self._outputs_on_device else "host_correctness"
        if not corr["ok"]:
            runtime_err = any(r.get("reason", "").startswith(("RuntimeFailure", "BackendError")) for r in corr["per_input"])
            stages.append(StageRecord(cstage, Evidence.FAILED.value, str(corr["failed_inputs"][:3])))
            fail = Failure.RUNTIME_FAILURE if runtime_err else Failure.CORRECTNESS_FAILURE
            return self._record(cand, State.REJECTED, stages, art, fail, str(corr["failed_inputs"][:3]), _slim(corr))[0], art, None
        state = State.DEVICE_CORRECT if self._outputs_on_device else State.HOST_CORRECT
        stages.append(StageRecord(cstage, Evidence.PASSED.value, f"max_abs_err={corr['max_abs_err']:.3g} on {corr['n_inputs']} inputs"))
        if not self.b.manifest.execution.target_codegen:
            stages.append(StageRecord("target_compile", Evidence.PENDING.value, "backend has no target codegen"))
        return self._record_with(cand, state, stages, art, corr), art, session

    def _record_with(self, cand, state, stages, art, corr):
        self._pending = (cand, state, stages, art, corr)
        return None  # record written after measurement decision

    # ------------------------------------------------------------------
    def run_baseline(self):
        cand = Candidate("baseline", "baseline", hypothesis="backend defaults")
        rec, art, session = self.evaluate(cand, baseline=True)
        if rec is not None:  # stopped early (rejected / pending)
            if rec.state != State.REJECTED.value:
                self.best_artifact, self.best_record = art, rec
                self.store.set_best(self.key, {"record": f"experiments/{self.eid}/candidates/{rec.candidate['candidate_id']}.json",
                                               "chain": [], "state": rec.state, "mode": self.mode,
                                               "note": f"baseline stopped at {rec.state}: {rec.failure_detail}"})
            return rec
        cand, state, stages, art, corr = self._pending
        meas = None
        if self._measure:
            try:
                meas = self.b.profiler.measure_e2e(self.b.runtime, session, self._perf_inputs(), self.policy)
                if meas.is_real_device:
                    state = State.E2E_MEASURED
                stages.append(StageRecord("e2e", Evidence.PASSED.value if meas.is_real_device else Evidence.PENDING.value,
                                          f"source={meas.source}"))
            except Unsupported as e:
                stages.append(StageRecord("e2e", Evidence.PENDING.value, str(e)))
        else:
            stages.append(StageRecord("e2e", Evidence.PENDING.value, f"mode={self.mode}: no admissible E2E measurement"))
        rec, path = self._record(cand, state, stages, art, corr=_slim(corr), meas=meas,
                                 decision={"verdict": "baseline"})
        self.best_artifact, self.best_session, self.best_record = art, session, rec
        self.store.set_best(self.key, {"record": str(path), "chain": [], "state": rec.state, "mode": self.mode,
                                       "note": "baseline is the best-known artifact until a device-accepted candidate exists"})
        return rec

    def _perf_inputs(self):
        tune = [c for c in self.corpus if c["split"] == "tune"]
        return (tune[0] if tune else self.corpus[0])["tensors"]

    def run_candidate(self, cand: Candidate):
        if self.n_run >= self.max_candidates or self.n_consecutive_fail >= self.max_consecutive_failures:
            stages = [StageRecord("budget", Evidence.FAILED.value, "budget exhausted")]
            return self._record(cand, State.REJECTED, stages, failure=Failure.BUDGET_EXHAUSTED, detail="budget")[0]
        self.n_run += 1
        cand = replace(cand, parent_id=self.best_record.candidate["candidate_id"] if self.best_record else None, candidate_id="")
        rec, art, session = self.evaluate(cand)
        if rec is not None:
            self.n_consecutive_fail = self.n_consecutive_fail + 1 if rec.state == State.REJECTED.value else 0
            return rec  # best untouched -> previous best stays active (restore)
        cand, state, stages, art, corr = self._pending
        meas_b = meas_c = None
        decision = None
        if self._measure:
            try:
                meas_b, meas_c = self._paired(self.best_session, session)
                decision = evaluator.judge(meas_b, meas_c, self.b.manifest, self.policy,
                                           device_correct=state == State.DEVICE_CORRECT)
                if meas_c.is_real_device and meas_b.is_real_device:
                    state = State.E2E_MEASURED
                stages.append(StageRecord("e2e", Evidence.PASSED.value if meas_c.is_real_device else Evidence.PENDING.value,
                                          f"source={meas_c.source} verdict={decision['verdict']}"))
            except Unsupported as e:
                stages.append(StageRecord("e2e", Evidence.PENDING.value, str(e)))
        else:
            stages.append(StageRecord("e2e", Evidence.PENDING.value, f"mode={self.mode}: no admissible E2E measurement"))
            decision = {"verdict": "pending", "reasons": [f"{self.mode} mode: acceptance requires device E2E"]}
        accepted = bool(decision and decision.get("verdict") == "accepted")
        if accepted:
            state = State.ACCEPTED
        rec, path = self._record(cand, state, stages, art, corr=_slim(corr),
                                 meas={"base": asdict(meas_b), "cand": asdict(meas_c)} if meas_c else None,
                                 decision=decision)
        if accepted:
            prev = self.store.get_best(self.key) or {"chain": []}
            self.best_artifact, self.best_session, self.best_record = art, session, rec
            self.store.set_best(self.key, {"record": str(path), "chain": prev["chain"] + [rec.candidate], "state": rec.state,
                                           "mode": self.mode})
        self.n_consecutive_fail = 0
        return rec

    def _paired(self, base_session, cand_session):
        one = replace(self.policy, pairs=1)
        warm = replace(one, warmup=self.policy.warmup)
        cold = replace(one, warmup=0)
        inputs = self._perf_inputs()
        bb, cb, src = [], [], set()
        meas_b = meas_c = None
        for i in range(self.policy.pairs):
            order = [("b", base_session), ("c", cand_session)] if i % 2 == 0 else [("c", cand_session), ("b", base_session)]
            for who, s in order:
                m = self.b.profiler.measure_e2e(self.b.runtime, s, inputs, warm if i == 0 else cold)
                (bb if who == "b" else cb).extend(m.blocks)
                src.add(m.source)
                if who == "b":
                    meas_b = m
                else:
                    meas_c = m
        meas_b = replace(meas_b, blocks=bb)
        meas_c = replace(meas_c, blocks=cb)
        return meas_b, meas_c

    def restore_best(self):
        """Rebuild the best artifact from the stored chain (after restart)."""
        best = self.store.get_best(self.key)
        self.best_artifact = self.b.compiler.capture_baseline(self.w)
        for cd in (best or {}).get("chain", []):
            c = Candidate(**cd)
            self.best_artifact = self.b.compiler.compile_candidate(self.w, c, self.best_artifact)
        return self.best_artifact


def _slim(corr):
    return {k: v for k, v in corr.items() if k != "per_input"} | {
        "per_input": [{k: r.get(k) for k in ("input_id", "ok", "max_abs_err", "mean_abs_err", "top1_match",
                                              "top5_overlap", "nonfinite", "reason", "beyond_processor_domain")}
                      for r in corr.get("per_input", [])]}
