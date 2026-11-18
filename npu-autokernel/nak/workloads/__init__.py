"""Workload registry: model + weights + I/O contract + reference (plan §5, §13)."""
from __future__ import annotations

_WORKLOADS = {}


def register(workload_id):
    def deco(cls):
        _WORKLOADS[workload_id] = cls
        cls.workload_id = workload_id
        return cls
    return deco


def get(workload_id, **kw):
    from . import deit, toy  # noqa: F401  (registration side effects)

    if workload_id not in _WORKLOADS:
        raise KeyError(f"unknown workload '{workload_id}'; known: {sorted(_WORKLOADS)}")
    return _WORKLOADS[workload_id](**kw)


def available():
    from . import deit, toy  # noqa: F401

    return sorted(_WORKLOADS)


class Workload:
    """Interface every workload implements."""

    workload_id = "?"
    func_name = "model"

    def module(self):  # torch.nn.Module taking the model inputs, returning one tensor
        raise NotImplementedError

    def example_inputs(self):  # list of tensors with the fixed shapes/dtypes
        raise NotImplementedError

    def corpus(self):  # list of dicts {id, split, tensors, sha256}
        raise NotImplementedError

    def meta(self) -> dict:
        raise NotImplementedError

    def lock(self) -> dict:  # provenance recorded with every experiment
        return {"workload_id": self.workload_id}
