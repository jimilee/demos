"""Adapter contract (plan §6, §6.1). These are this project's interfaces, not vendor APIs.

A backend registers an AdapterBundle = CompilerAdapter + RuntimeAdapter +
ProfilerAdapter + CapabilityManifest. The controller/evaluator only talk to
these interfaces.
"""
from __future__ import annotations

import abc
from dataclasses import dataclass, field

from .capability import CapabilityManifest
from .schema import Measurement


class BackendError(Exception):
    failure = "runtime_failure"


class DeviceUnavailable(BackendError):
    failure = "device_unavailable"


class Unsupported(BackendError):
    failure = "unsupported"


class CompileFailure(BackendError):
    failure = "compile_failure"


class RuntimeFailure(BackendError):
    failure = "runtime_failure"


@dataclass
class Artifact:
    """Output of compile/integrate: whatever the runtime adapter can load."""

    backend_id: str
    kind: str  # ir | object | shared_library | jit_module | sdk_model
    payload: object = None  # in-memory handle (e.g. JIT engine), not persisted
    ir: object = None  # source IR the artifact was built from (candidates derive from this)
    files: dict = field(default_factory=dict)  # name -> path
    ir_hashes: dict = field(default_factory=dict)  # stage -> sha256
    artifact_hashes: dict = field(default_factory=dict)
    requested_config: dict = field(default_factory=dict)
    effective_config: dict = field(default_factory=dict)  # after backend normalization
    provenance: list = field(default_factory=list)  # applied transforms, in order
    target_compiled: bool = False  # True only if code for the target architecture was produced
    pending: str = ""  # why a later stage could not run in this environment (e.g. toolchain_unavailable)
    log: str = ""


class CompilerAdapter(abc.ABC):
    @abc.abstractmethod
    def capture_baseline(self, workload) -> Artifact:
        """Baseline IR/artifact for the workload with backend defaults."""

    @abc.abstractmethod
    def compile_candidate(self, workload, candidate, parent: Artifact | None) -> Artifact:
        """Apply candidate change; return IR/binary with requested AND effective config."""

    def integrate_candidate(self, workload, artifact: Artifact) -> Artifact:
        """Produce the full-model executable artifact (kernel binding etc.)."""
        return artifact


class RuntimeAdapter(abc.ABC):
    def bind_workload(self, workload) -> None:
        """Optional hook: runtimes that need the workload object (e.g. mocks) receive it here."""

    @abc.abstractmethod
    def check_artifact(self, artifact: Artifact) -> None:
        """Reject artifacts built for another backend/profile (raise Unsupported)."""

    @abc.abstractmethod
    def prepare_session(self, artifact: Artifact):
        """Load model, constants and persistent buffers. Returns a session handle."""

    @abc.abstractmethod
    def execute_and_wait(self, session, inputs: list):
        """Run one request to completion; return host-readable outputs (list of tensors)."""

    def release(self, session) -> None:
        pass


class ProfilerAdapter(abc.ABC):
    @abc.abstractmethod
    def measure_e2e(self, runtime: RuntimeAdapter, session, inputs: list, policy) -> Measurement:
        """Measure the E2E request boundary declared in the manifest."""

    def profile(self, runtime, session, inputs) -> dict:
        return {"status": "unavailable"}


@dataclass
class AdapterBundle:
    manifest: CapabilityManifest
    compiler: CompilerAdapter
    runtime: RuntimeAdapter
    profiler: ProfilerAdapter

    @property
    def backend_id(self):
        return self.manifest.backend_id

    @property
    def kind(self):
        return self.manifest.kind
