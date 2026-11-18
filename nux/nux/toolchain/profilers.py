"""Vendor-neutral profilers that backends can reuse instead of importing each other."""
from __future__ import annotations

import statistics
import time

from ..core.adapters import ProfilerAdapter, Unsupported
from ..core.schema import Measurement


class HostWallClockProfiler(ProfilerAdapter):
    """Wall-clock timing of runtime.execute_and_wait on the HOST. Never device evidence."""

    def __init__(self, boundary: str, unit="ms"):
        self.boundary = boundary
        self.unit = unit

    def measure_e2e(self, runtime, session, inputs, policy):
        scale = 1e3 if self.unit == "ms" else 1e6
        for _ in range(policy.warmup):
            runtime.execute_and_wait(session, inputs)
        blocks = []
        for _ in range(policy.pairs):
            b = []
            for _ in range(policy.requests_per_block):
                t = time.perf_counter()
                runtime.execute_and_wait(session, inputs)
                b.append((time.perf_counter() - t) * scale)
            blocks.append(b)
        return Measurement("host", self.boundary, self.unit, blocks, missing=["npu time", "device counters"],
                           conditions={"note": f"host wall clock, first-block median {statistics.median(blocks[0]):.4g} {self.unit}"})


class NoProfiler(ProfilerAdapter):
    def __init__(self, reason: str):
        self.reason = reason

    def measure_e2e(self, runtime, session, inputs, policy):
        raise Unsupported(self.reason)
