"""Candidate evaluation as an explicit stage pipeline (refactor 2/6).

Each stage is a plain function `stage(ctl, ctx)` that reads/writes only the
CandidateContext and may end the pipeline with `ctx.stop(...)`. The
controller runs STAGES in order and writes exactly one immutable record from
the final context. There is no hidden controller state between stages.

Stage order follows the evidence ladder:
  capability -> compile -> ir_verify -> dedupe -> pending -> target_compiled
  -> twin_correctness -> session -> correctness -> measure
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field

from . import evaluator
from .adapters import Artifact, BackendError, DeviceUnavailable, Unsupported
from .capability import validate_action
from .measurement import PairedProtocol, SingleProtocol
from .schema import Candidate, Evidence, Failure, Measurement, StageRecord, State


@dataclass
class CandidateContext:
    cand: Candidate
    baseline: bool = False
    stages: list = field(default_factory=list)
    state: State = State.PROPOSED
    art: Artifact | None = None
    session: object = None
    corr: dict | None = None  # correctness summary attached to the record (twin or target)
    meas: object = None  # Measurement (baseline) or {"base": ..., "cand": ...} (paired)
    decision: dict | None = None
    failure: Failure | None = None
    detail: str = ""
    done: bool = False
    record: object = None
    record_path: object = None

    def add(self, stage, status, detail="", seconds=None, artifacts=None):
        self.stages.append(StageRecord(stage, status.value if isinstance(status, Evidence) else status, detail,
                                       seconds, artifacts or {}))

    def stop(self, state: State | None = None, failure: Failure | None = None, detail: str = "", corr=None):
        if state is not None:
            self.state = state
        self.failure, self.detail, self.done = failure, detail or self.detail, True
        if corr is not None:
            self.corr = corr

    @property
    def rejected(self):
        return self.state == State.REJECTED


def slim(corr):
    keep = ("input_id", "ok", "max_abs_err", "mean_abs_err", "top1_match", "top5_overlap", "nonfinite", "reason",
            "beyond_processor_domain")
    return {k: v for k, v in corr.items() if k != "per_input"} | {
        "per_input": [{k: r.get(k) for k in keep} for r in corr.get("per_input", [])]}


def _splits(ctl):
    return tuple(ctl.cfg.get("correctness_splits", ["tune", "stress"]))


# ------------------------------------------------------------------ stages
def capability(ctl, ctx):
    ctx.add("proposed", Evidence.PASSED)
    ok, why = validate_action(ctl.b.manifest, ctx.cand, ctl.w.meta())
    if not ok:
        ctx.add("capability", Evidence.FAILED, why)
        return ctx.stop(State.REJECTED, Failure.UNSUPPORTED, why)
    ctx.add("capability", Evidence.PASSED, why)


def compile_(ctl, ctx):
    t = time.perf_counter()
    try:
        art = (ctl.b.compiler.capture_baseline(ctl.w) if ctx.baseline else
               ctl.b.compiler.compile_candidate(ctl.w, ctx.cand, ctl.best_artifact))
        if not art.pending:
            art = ctl.b.compiler.integrate_candidate(ctl.w, art)
    except BackendError as e:
        ctx.add("compile", Evidence.FAILED, str(e), time.perf_counter() - t)
        fail = Failure(e.failure) if e.failure in Failure._value2member_map_ else Failure.COMPILE_FAILURE
        return ctx.stop(State.REJECTED, fail, str(e))
    ctx.art = art
    ctx.add("compile", Evidence.PASSED, art.log, time.perf_counter() - t, {"kind": art.kind, "provenance": art.provenance[-3:]})


def ir_verify(ctl, ctx):
    if ctx.art.ir_hashes:
        ctx.state = State.IR_VALID
        ctx.add("ir_verify", Evidence.PASSED, "MLIR verifier (structural only)")


def dedupe(ctl, ctx):
    best = ctl.best_record
    if best and ctx.art.ir_hashes and ctx.art.ir_hashes == best.ir_hashes and ctx.art.effective_config == best.effective_config:
        ctx.add("dedupe", Evidence.PASSED, "IR and effective config identical to best: no change possible")
        ctx.stop(detail="identical to best")


def pending(ctl, ctx):
    if ctx.art.pending:  # environment limit, not a candidate failure: later evidence stays pending
        ctx.add("target_compile", Evidence.PENDING, ctx.art.pending)
        ctx.stop(detail=ctx.art.pending)


def target_compiled(ctl, ctx):
    if ctx.art.target_compiled and ctl.b.manifest.execution.target_codegen:
        ctx.state = State.TARGET_COMPILED


def twin_correctness(ctl, ctx):
    """Host build of the exact target-bound IR: semantics check, not device evidence."""
    art = ctx.art
    if art.twin is None:
        return
    import torch

    tc = evaluator.check_correctness(lambda ts: torch.from_numpy(art.twin(*[t.detach().contiguous().numpy() for t in ts])),
                                     ctl.w, ctl.c, ctl.corpus, splits=_splits(ctl))
    if not tc["ok"]:
        ctx.add("twin_correctness", Evidence.FAILED, str(tc["failed_inputs"][:3]))
        return ctx.stop(State.REJECTED, Failure.CORRECTNESS_FAILURE, "twin: " + str(tc["failed_inputs"][:3]), slim(tc))
    ctx.add("twin_correctness", Evidence.PASSED, f"{art.twin_note}: max_abs_err={tc['max_abs_err']:.3g} on {tc['n_inputs']} inputs")
    ctx.corr = slim(tc)


def session(ctl, ctx):
    try:
        ctx.session = ctl.b.runtime.prepare_session(ctx.art)
    except DeviceUnavailable as e:
        ctx.add("device_correctness", Evidence.PENDING, str(e))
        return ctx.stop(detail=str(e))
    except BackendError as e:
        ctx.add("load", Evidence.FAILED, str(e))
        return ctx.stop(State.REJECTED, Failure(e.failure), str(e))


def correctness(ctl, ctx):
    corr = evaluator.check_correctness(lambda ts: ctl.b.runtime.execute_and_wait(ctx.session, ts), ctl.w, ctl.c,
                                       ctl.corpus, splits=_splits(ctl))
    on_device = ctl.b.manifest.execution.executes_on == "device"
    name = "device_correctness" if on_device else "host_correctness"
    if not corr["ok"]:
        runtime_err = any(r.get("reason", "").startswith(("RuntimeFailure", "BackendError")) for r in corr["per_input"])
        ctx.add(name, Evidence.FAILED, str(corr["failed_inputs"][:3]))
        fail = Failure.RUNTIME_FAILURE if runtime_err else Failure.CORRECTNESS_FAILURE
        return ctx.stop(State.REJECTED, fail, str(corr["failed_inputs"][:3]), slim(corr))
    ctx.state = State.DEVICE_CORRECT if on_device else State.HOST_CORRECT
    ctx.add(name, Evidence.PASSED, f"max_abs_err={corr['max_abs_err']:.3g} on {corr['n_inputs']} inputs")
    if not ctl.b.manifest.execution.target_codegen:
        ctx.add("target_compile", Evidence.PENDING, "backend has no target codegen")
    ctx.corr = slim(corr)


def measure(ctl, ctx):
    if not ctl.measure_enabled:
        ctx.add("e2e", Evidence.PENDING, f"mode={ctl.mode}: no admissible E2E measurement")
        ctx.decision = {"verdict": "baseline"} if ctx.baseline else \
            {"verdict": "pending", "reasons": [f"{ctl.mode} mode: acceptance requires device E2E"]}
        return ctx.stop()
    inputs = ctl.perf_inputs()
    try:
        if ctx.baseline:
            m = SingleProtocol(ctl.policy).run(ctl.b.profiler, ctl.b.runtime, ctx.session, inputs)
            if m.is_real_device:
                ctx.state = State.E2E_MEASURED
            ctx.meas = m
            ctx.add("e2e", Evidence.PASSED if m.is_real_device else Evidence.PENDING, f"source={m.source}")
            ctx.decision = {"verdict": "baseline"}
        else:
            mb, mc = PairedProtocol(ctl.policy).run(ctl.b.profiler, ctl.b.runtime, ctl.best_session, ctx.session, inputs)
            ctx.decision = evaluator.judge(mb, mc, ctl.b.manifest, ctl.policy, device_correct=ctx.state == State.DEVICE_CORRECT)
            if mb.is_real_device and mc.is_real_device:
                ctx.state = State.E2E_MEASURED
            ctx.meas = {"base": asdict(mb), "cand": asdict(mc)}
            ctx.add("e2e", Evidence.PASSED if mc.is_real_device else Evidence.PENDING,
                    f"source={mc.source} verdict={ctx.decision['verdict']}")
            if ctx.decision.get("verdict") == "accepted":
                ctx.state = State.ACCEPTED
    except Unsupported as e:
        ctx.add("e2e", Evidence.PENDING, str(e))
        if ctx.baseline:
            ctx.decision = {"verdict": "baseline"}
    ctx.stop()


STAGES = (capability, compile_, ir_verify, dedupe, pending, target_compiled, twin_correctness, session, correctness, measure)


def run(ctl, ctx):
    for stage in STAGES:
        stage(ctl, ctx)
        if ctx.done:
            break
    return ctx


def measurement_for_record(meas):
    return asdict(meas) if isinstance(meas, Measurement) else meas
