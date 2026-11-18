"""Experiment configuration schema (refactor 5/6): strict validation at load time.

Unknown keys are errors (they used to be silently ignored). Keys starting with
"_" are comments/provenance and are kept but not interpreted. Identifiers
(backend, contract, workload) are checked against their registries, and every
candidate entry is checked against the Candidate fields. Standard library only.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, fields
from pathlib import Path

from .artifact_store import ArtifactPolicy
from .schema import Candidate
from .stats import Policy

SPLITS = ("tune", "holdout", "stress")


class ConfigError(ValueError):
    pass


def _strict(name, d: dict, allowed: set, required: set = frozenset()):
    if not isinstance(d, dict):
        raise ConfigError(f"{name}: expected an object, got {type(d).__name__}")
    unknown = sorted(k for k in d if k not in allowed and not k.startswith("_"))
    if unknown:
        raise ConfigError(f"{name}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")
    missing = sorted(required - set(d))
    if missing:
        raise ConfigError(f"{name}: missing required key(s) {missing}")


@dataclass(frozen=True)
class Budget:
    max_candidates: int = 50
    max_consecutive_failures: int = 5


@dataclass(frozen=True)
class ArtifactCacheConfig:
    enabled: bool = True
    root: str = ".cache/artifacts"
    max_gb: float = 5.0
    delete_rejected: bool = True
    drop_intermediates_after_record: bool = True
    verify_on_get: bool = True

    def policy(self) -> ArtifactPolicy:
        return ArtifactPolicy.from_config(self.__dict__)


@dataclass(frozen=True)
class ExperimentConfig:
    backend_id: str
    workload_id: str
    evaluation_contract_id: str
    compiler_profile: str = ""
    runtime_profile: str = ""
    correctness_splits: tuple = ("tune", "stress")
    corpus_limit: dict | None = None
    budget: Budget = field(default_factory=Budget)
    policy: Policy = field(default_factory=Policy)
    artifact_cache: ArtifactCacheConfig = field(default_factory=ArtifactCacheConfig)
    store: str = "store"
    informational_measurement: bool = False
    candidates: tuple = ()
    comments: dict = field(default_factory=dict)  # "_"-prefixed keys, kept verbatim

    def as_runtime_dict(self) -> dict:
        """The plain dict the controller and records see (validated values, defaults filled in)."""
        return {"backend_id": self.backend_id, "workload_id": self.workload_id,
                "evaluation_contract_id": self.evaluation_contract_id, "compiler_profile": self.compiler_profile,
                "runtime_profile": self.runtime_profile, "correctness_splits": list(self.correctness_splits),
                "corpus_limit": self.corpus_limit, "budget": dict(self.budget.__dict__),
                "policy": self.policy.to_dict(), "artifact_cache": dict(self.artifact_cache.__dict__),
                "store": self.store, "informational_measurement": self.informational_measurement,
                "candidates": [c.__dict__ for c in self.candidates], **self.comments}


_TOP = {f.name for f in fields(ExperimentConfig)} - {"comments"}
_CAND = {f.name for f in fields(Candidate)} - {"candidate_id", "parent_id"}
_REQ_CAND = {"action_id", "kind"}


def _typed(name, value, typ):
    if typ is bool and not isinstance(value, bool):
        raise ConfigError(f"{name}: expected true/false, got {value!r}")
    if typ is int and (isinstance(value, bool) or not isinstance(value, int)):
        raise ConfigError(f"{name}: expected an integer, got {value!r}")
    if typ is float and (isinstance(value, bool) or not isinstance(value, (int, float))):
        raise ConfigError(f"{name}: expected a number, got {value!r}")
    if typ is str and not isinstance(value, str):
        raise ConfigError(f"{name}: expected a string, got {value!r}")
    return float(value) if typ is float else value


def _section(name, d, cls):
    _strict(name, d, {f.name for f in fields(cls)})
    kw = {}
    for f in fields(cls):
        if f.name in d:
            typ = {"int": int, "float": float, "bool": bool, "str": str}.get(str(f.type), None)
            kw[f.name] = _typed(f"{name}.{f.name}", d[f.name], typ) if typ else d[f.name]
    return cls(**kw)


def parse(d: dict, source: str = "config", check_ids: bool = True) -> ExperimentConfig:
    _strict(source, d, _TOP, {"backend_id", "workload_id", "evaluation_contract_id"})
    for k in ("backend_id", "workload_id", "evaluation_contract_id", "compiler_profile", "runtime_profile", "store"):
        if k in d:
            _typed(f"{source}.{k}", d[k], str)
    if check_ids:
        from . import contract, registry, workloads

        if d["backend_id"] not in registry.available():
            raise ConfigError(f"{source}.backend_id: '{d['backend_id']}' not registered; known: {registry.available()}")
        if d["evaluation_contract_id"] not in contract.CONTRACTS:
            raise ConfigError(f"{source}.evaluation_contract_id: unknown '{d['evaluation_contract_id']}'")
        if d["workload_id"] not in workloads.available():
            raise ConfigError(f"{source}.workload_id: unknown '{d['workload_id']}'")
    splits = tuple(d.get("correctness_splits", ("tune", "stress")))
    bad = [s for s in splits if s not in SPLITS]
    if bad or not splits:
        raise ConfigError(f"{source}.correctness_splits: {list(splits)} (allowed {SPLITS})")
    if "holdout" in splits:
        raise ConfigError(f"{source}.correctness_splits: holdout is reserved for final re-verification (plan §9.3)")
    lim = d.get("corpus_limit")
    if lim is not None:
        _strict(f"{source}.corpus_limit", lim, set(SPLITS))
        for k, v in lim.items():
            _typed(f"{source}.corpus_limit.{k}", v, int)
    cands = []
    for i, c in enumerate(d.get("candidates", [])):
        _strict(f"{source}.candidates[{i}]", c, _CAND, _REQ_CAND)
        if not isinstance(c.get("params", {}), dict):
            raise ConfigError(f"{source}.candidates[{i}].params: expected an object")
        cands.append(Candidate(**c))
    return ExperimentConfig(
        backend_id=d["backend_id"], workload_id=d["workload_id"], evaluation_contract_id=d["evaluation_contract_id"],
        compiler_profile=d.get("compiler_profile", ""), runtime_profile=d.get("runtime_profile", ""),
        correctness_splits=splits, corpus_limit=lim,
        budget=_section(f"{source}.budget", d.get("budget", {}), Budget),
        policy=_section(f"{source}.policy", d.get("policy", {}), Policy),
        artifact_cache=_section(f"{source}.artifact_cache", d.get("artifact_cache", {}), ArtifactCacheConfig),
        store=d.get("store", "store"),
        informational_measurement=_typed(f"{source}.informational_measurement", d.get("informational_measurement", False), bool),
        candidates=tuple(cands), comments={k: v for k, v in d.items() if k.startswith("_")})


def load(path) -> ExperimentConfig:
    p = Path(path)
    try:
        d = json.loads(p.read_text())
    except json.JSONDecodeError as e:
        raise ConfigError(f"{p}: invalid JSON: {e}") from e
    return parse(d, source=p.name)
