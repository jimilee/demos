"""Capability manifest and action validation (plan §6.1, §7.4).

Two independent axes per action:
  support     supported | experimental | unavailable   (what the backend/toolchain documents or exposes)
  validation  validated | unvalidated | failed          (what was proven on THIS device/toolchain/workload)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

SUPPORT = ("supported", "experimental", "unavailable")
VALIDATION = ("validated", "unvalidated", "failed")
ACTION_KINDS = ("config", "ir_recipe", "kernel", "lowering_path", "baseline")
# shared vocabulary between vendor hints and backend actions
TAGS = ("fusion", "layout", "reshape", "tiling", "memory", "threading", "vector", "reduction", "codegen",
        "library_matmul", "kernel", "partition", "precision", "cleanup")


@dataclass
class ParamSpec:
    name: str
    type: str  # bool | int | str | enum | list
    choices: list | None = None
    min: int | None = None
    max: int | None = None


@dataclass
class ActionSpec:
    action_id: str
    kind: str
    description: str
    stage: str  # e.g. "linalg-on-tensors", "compile-options", "triton-kernel", "model-sdk"
    params: list = field(default_factory=list)  # [ParamSpec]
    scope: str = "module"  # module-wide options must not be presented as per-op
    numeric_change: str = "none"  # none | reassociation | approximation | ...
    requires_runtime: list = field(default_factory=list)
    constraints: dict = field(default_factory=dict)  # dtype/shape/arch constraints
    support: str = "supported"
    validation: str = "unvalidated"
    evidence: str = ""  # where the support claim comes from (source file, doc, run id)
    tags: list = field(default_factory=list)  # optimization categories (fusion, layout, tiling, ...) used to map hints

    def __post_init__(self):
        _check(self.kind, ACTION_KINDS, f"{self.action_id}.kind")
        _check(self.support, SUPPORT, f"{self.action_id}.support")
        _check(self.validation, VALIDATION, f"{self.action_id}.validation")
        for t in self.tags:
            _check(t, TAGS, f"{self.action_id}.tags")
        self.params = [p if isinstance(p, ParamSpec) else ParamSpec(**p) for p in self.params]


EXECUTES_ON = ("device", "host", "none")
TIERS = ("device", "host", "synthetic", "none")


def _check(value, allowed, what):
    if value not in allowed:
        raise ValueError(f"{what}={value!r} not in {allowed}")


@dataclass
class ExecutionCaps:
    """What the backend can do with an artifact (independent of how trustworthy its timings are)."""

    target_codegen: bool = False  # produces code for a non-host target architecture
    executes_on: str = "none"  # where outputs are computed: device | host | none

    def __post_init__(self):
        _check(self.executes_on, EXECUTES_ON, "execution.executes_on")


@dataclass
class MeasurementCaps:
    e2e: str = "unavailable"  # supported | experimental | unavailable
    tier: str = "none"  # evidence tier of timings: device | host | synthetic | none
    e2e_boundary: str = ""
    units: str = "us"
    op_profile: str = "unavailable"
    counters: list = field(default_factory=list)
    notes: str = ""

    def __post_init__(self):
        _check(self.e2e, SUPPORT, "measurement.e2e")
        _check(self.tier, TIERS, "measurement.tier")


@dataclass
class CapabilityManifest:
    backend_id: str
    version: str
    compiler_profile: str
    runtime_profile: str
    profiler_profile: str
    input_stages: list
    dtypes: list
    actions: list  # [ActionSpec]
    measurement: MeasurementCaps
    execution: ExecutionCaps = field(default_factory=ExecutionCaps)
    # Inputs that change what THIS backend's compiler produces (versions, target, lowering code hashes).
    # Artifact-store keys use this, never the full environment: device state, adb paths or other
    # backends' environments must not invalidate or alias cached artifacts.
    compile_fingerprint: dict = field(default_factory=dict)
    compatible_runtimes: list = field(default_factory=list)
    device_profile: dict = field(default_factory=dict)
    environment: dict = field(default_factory=dict)  # discovered toolchain / device facts
    notes: list = field(default_factory=list)

    def __post_init__(self):
        self.actions = [a if isinstance(a, ActionSpec) else ActionSpec(**a) for a in self.actions]
        if isinstance(self.measurement, dict):
            self.measurement = MeasurementCaps(**self.measurement)
        if isinstance(self.execution, dict):
            self.execution = ExecutionCaps(**self.execution)

    # ---- derived properties: the ONLY way common code asks "what kind of backend is this" ----
    @property
    def measures_e2e(self) -> bool:
        return self.measurement.e2e in ("supported", "experimental")

    @property
    def acceptance_capable(self) -> bool:
        """Can this backend ever produce acceptance evidence? (device outputs + device E2E timings)"""
        return self.execution.executes_on == "device" and self.measurement.tier == "device" and self.measures_e2e

    def traits(self) -> dict:
        return {"executes_on": self.execution.executes_on, "target_codegen": self.execution.target_codegen,
                "measurement_tier": self.measurement.tier, "e2e": self.measurement.e2e,
                "acceptance_capable": self.acceptance_capable}

    def action(self, action_id: str) -> ActionSpec | None:
        return next((a for a in self.actions if a.action_id == action_id), None)

    def usable_actions(self, include_experimental=True) -> list:
        ok = ("supported", "experimental") if include_experimental else ("supported",)
        return [a for a in self.actions if a.support in ok and a.validation != "failed"]

    def to_dict(self):
        return asdict(self)


def validate_action(manifest: CapabilityManifest, candidate, workload_meta: dict | None = None):
    """Return (ok, reason). Filters candidates BEFORE submission (plan §6.2, §7.4)."""
    wdt = (workload_meta or {}).get("dtype")
    if wdt and manifest.dtypes and wdt not in manifest.dtypes:  # applies to the baseline too
        return False, f"workload dtype {wdt} not supported by {manifest.backend_id} (dtypes {manifest.dtypes})"
    if candidate.kind == "baseline":
        return True, "baseline"
    spec = manifest.action(candidate.action_id)
    if spec is None:
        return False, f"action '{candidate.action_id}' not in {manifest.backend_id} capability manifest"
    if spec.kind != candidate.kind:
        return False, f"kind mismatch: candidate {candidate.kind} vs capability {spec.kind}"
    if spec.support == "unavailable":
        return False, f"action unavailable on {manifest.backend_id}: {spec.evidence or spec.description}"
    if spec.validation == "failed":
        return False, f"action previously failed validation on this toolchain/device: {spec.evidence}"
    if spec.scope == "module" and candidate.scope != "module":
        return False, f"'{spec.action_id}' is module-wide; per-op scope '{candidate.scope}' not supported"
    known = {p.name: p for p in spec.params}
    for k, v in candidate.params.items():
        p = known.get(k)
        if p is None:
            return False, f"unknown parameter '{k}' for {spec.action_id}"
        if p.choices is not None and v not in p.choices:
            return False, f"{k}={v!r} not in {p.choices}"
        if p.type == "int" and (not isinstance(v, int) or (p.min is not None and v < p.min)
                                or (p.max is not None and v > p.max)):
            return False, f"{k}={v!r} outside [{p.min}, {p.max}]"
        if p.type == "bool" and not isinstance(v, bool):
            return False, f"{k} must be bool"
    meta = workload_meta or {}
    c = spec.constraints
    if "dtypes" in c and meta.get("dtype") and meta["dtype"] not in c["dtypes"]:
        return False, f"dtype {meta['dtype']} not in {c['dtypes']}"
    if "arch_exclude" in c and meta.get("arch") in c["arch_exclude"]:
        return False, f"not supported on arch {meta.get('arch')}"
    for r in spec.requires_runtime:
        if r not in manifest.environment.get("runtime_features", []):
            return False, f"requires runtime feature '{r}' not available"
    return True, "ok"
