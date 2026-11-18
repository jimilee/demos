"""Candidate / result schema shared by every backend (plan §12, §14 candidate states)."""
from __future__ import annotations

import enum
import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from . import SCHEMA_VERSION


class State(str, enum.Enum):
    """Evidence ladder. Passing one step never implies the next (plan §14)."""

    PROPOSED = "proposed"
    IR_VALID = "ir_valid"
    HOST_CORRECT = "host_correct"
    TARGET_COMPILED = "target_compiled"
    DEVICE_CORRECT = "device_correct"
    E2E_MEASURED = "e2e_measured"
    ACCEPTED = "accepted"
    REJECTED = "rejected"


LADDER = [State.PROPOSED, State.IR_VALID, State.HOST_CORRECT, State.TARGET_COMPILED,
          State.DEVICE_CORRECT, State.E2E_MEASURED, State.ACCEPTED]


class Failure(str, enum.Enum):
    """Distinct failure classes (plan §6.2)."""

    DEVICE_UNAVAILABLE = "device_unavailable"
    UNSUPPORTED = "unsupported"
    COMPILE_FAILURE = "compile_failure"
    RUNTIME_FAILURE = "runtime_failure"
    CORRECTNESS_FAILURE = "correctness_failure"
    MEASUREMENT_INVALID = "measurement_invalid"
    BUDGET_EXHAUSTED = "budget_exhausted"


class Evidence(str, enum.Enum):
    PASSED = "passed"
    FAILED = "failed"
    PENDING = "pending"  # not verifiable in this environment / not attempted
    NOT_APPLICABLE = "n/a"


@dataclass
class Candidate:
    """One change relative to a parent (baseline or previous best).

    kind: config | ir_recipe | kernel | lowering_path | baseline
    """

    action_id: str
    kind: str
    params: dict = field(default_factory=dict)
    hypothesis: str = ""
    expected_savings: str = ""
    possible_costs: str = ""
    scope: str = "module"  # module | op:<pattern> | subgraph:<id>
    parent_id: str | None = None
    payload_files: dict = field(default_factory=dict)  # name -> path (kernel source, recipe, IR)
    candidate_id: str = ""

    def __post_init__(self):
        if not self.candidate_id:
            self.candidate_id = self.content_hash()[:12]

    def content_hash(self) -> str:
        blob = {"action_id": self.action_id, "kind": self.kind, "params": self.params,
                "scope": self.scope, "parent": self.parent_id,
                "payload": {k: _file_hash(v) for k, v in sorted(self.payload_files.items())}}
        return hashlib.sha256(json.dumps(blob, sort_keys=True, default=str).encode()).hexdigest()


def _file_hash(path) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.sha256(f.read()).hexdigest()
    except OSError:
        return "missing"


@dataclass
class Measurement:
    """Raw timing samples with an explicit statement of what they are.

    source: device (real target measurement) | host (host CPU, never NPU
    performance) | synthetic (mock, controller tests only).
    blocks: list of blocks; each block is a list of per-request latencies.
    """

    source: str
    boundary: str
    unit: str
    blocks: list = field(default_factory=list)
    instrumentation: bool = False
    conditions: dict = field(default_factory=dict)  # power/thermal/charging/etc.
    invalid_blocks: list = field(default_factory=list)  # (index, reason)
    missing: list = field(default_factory=list)  # quantities the profiler cannot provide

    @property
    def is_real_device(self) -> bool:
        return self.source == "device"


@dataclass
class StageRecord:
    stage: str
    status: str  # Evidence value
    detail: str = ""
    seconds: float | None = None
    artifacts: dict = field(default_factory=dict)


@dataclass
class CandidateRecord:
    """Immutable experiment record (plan §12)."""

    experiment_id: str
    candidate: dict
    backend_id: str
    workload_id: str
    contract_id: str
    state: str
    failure: str | None = None
    failure_detail: str = ""
    stages: list = field(default_factory=list)
    requested_config: dict = field(default_factory=dict)
    effective_config: dict = field(default_factory=dict)
    ir_hashes: dict = field(default_factory=dict)
    artifact_hashes: dict = field(default_factory=dict)
    correctness: dict = field(default_factory=dict)
    measurement: dict | None = None
    decision: dict | None = None
    toolchain_lock: dict = field(default_factory=dict)
    backend_traits: dict = field(default_factory=dict)  # executes_on / target_codegen / measurement tier
    cache_key: str = ""
    created_at: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    schema_version: str = SCHEMA_VERSION

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=_json_default)


def _json_default(o: Any):
    if isinstance(o, enum.Enum):
        return o.value
    if hasattr(o, "__dataclass_fields__"):
        return asdict(o)
    return str(o)


def dump(obj) -> str:
    return json.dumps(obj, indent=2, default=_json_default)
