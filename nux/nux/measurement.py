"""Measurement protocols (evaluator-owned): how baseline and candidate are sampled.

The profiler adapter only knows how to time one session for a given number of
blocks; ordering, warm-up and pairing are policy decisions and live here.
"""
from __future__ import annotations

from dataclasses import replace

from .stats import Policy


class PairedProtocol:
    """ABBA-interleaved blocks: pair i runs (base, cand) when i is even and (cand, base) when odd.
    Warm-up happens once per session, before its first block."""

    name = "abba-paired-v1"

    def __init__(self, policy: Policy):
        self.policy = policy

    def run(self, profiler, runtime, base_session, cand_session, inputs):
        cold = replace(self.policy, pairs=1, warmup=0)
        warm = replace(self.policy, pairs=1)
        warmed = set()
        base_blocks, cand_blocks = [], []
        last = {}
        for i in range(self.policy.pairs):
            order = (("b", base_session), ("c", cand_session)) if i % 2 == 0 else (("c", cand_session), ("b", base_session))
            for who, session in order:
                pol = cold if who in warmed else warm
                warmed.add(who)
                m = profiler.measure_e2e(runtime, session, inputs, pol)
                (base_blocks if who == "b" else cand_blocks).extend(m.blocks)
                last[who] = m
        return (replace(last["b"], blocks=base_blocks, conditions={**last["b"].conditions, "protocol": self.name}),
                replace(last["c"], blocks=cand_blocks, conditions={**last["c"].conditions, "protocol": self.name}))


class SingleProtocol:
    """Baseline-only measurement (no pairing partner yet)."""

    name = "single-v1"

    def __init__(self, policy: Policy):
        self.policy = policy

    def run(self, profiler, runtime, session, inputs):
        m = profiler.measure_e2e(runtime, session, inputs, self.policy)
        return replace(m, conditions={**m.conditions, "protocol": self.name})
