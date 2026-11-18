"""Content-addressed artifact store with an explicit retention policy (refactor 4/6).

Layout:  <root>/<key[:2]>/<key>/{manifest.json, <files...>}

File roles
  primary       needed to load/run the artifact (target object, shared library)
  source        needed to reproduce it (input IR bytecode)
  verification  not deployed, but needed to RE-EVALUATE without recompiling (host twin .so of the
                exact target-bound IR); kept with the entry, evicted with it under the size cap
  intermediate  only needed while building/evaluating once (LLVM IR text, twin .o); regenerable

Default policy (ArtifactPolicy)
  * pinned entries (baseline, accepted best chain) are never evicted
  * rejected candidates are deleted right after their record is written
  * intermediate files are dropped right after the record is written (regenerable);
    verification files are kept
  * total size capped (default 5 GB); unpinned entries evicted least-recently-used first
  * every get() verifies the sha256 of every file; a mismatch evicts the entry (= miss)

Records keep artifact keys and file hashes permanently, so evicted artifacts
remain identifiable and can be regenerated from the candidate chain.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from dataclasses import dataclass
from pathlib import Path

ROLES = ("primary", "source", "verification", "intermediate")


@dataclass(frozen=True)
class ArtifactPolicy:
    max_bytes: int = 5 * 10**9  # decimal GB, as documented
    delete_rejected: bool = True
    drop_intermediates_after_record: bool = True
    verify_on_get: bool = True

    @classmethod
    def from_config(cls, cfg: dict | None):
        c = cfg or {}
        return cls(max_bytes=int(float(c.get("max_gb", 5)) * 10**9), delete_rejected=bool(c.get("delete_rejected", True)),
                   drop_intermediates_after_record=bool(c.get("drop_intermediates_after_record", True)),
                   verify_on_get=bool(c.get("verify_on_get", True)))


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rmdir_if_empty(d: Path):
    try:
        d.rmdir()
    except OSError:
        pass


def compile_key(**parts) -> str:
    """Deterministic key computed BEFORE compiling (inputs only)."""
    return hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:32]


class ArtifactStore:
    def __init__(self, root, policy: ArtifactPolicy | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.policy = policy or ArtifactPolicy()
        self.events = []  # (time, action, key, detail) for this process; tests and reports read it

    # ---- paths / manifests ----------------------------------------------------
    def _dir(self, key) -> Path:
        return self.root / key[:2] / key

    def _manifest_path(self, key) -> Path:
        return self._dir(key) / "manifest.json"

    def manifest(self, key) -> dict | None:
        p = self._manifest_path(key)
        return json.loads(p.read_text()) if p.exists() else None

    def _save(self, key, m):
        p = self._manifest_path(key)
        tmp = p.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(m, indent=2, default=str))
        tmp.replace(p)

    def _log(self, action, key, detail=""):
        self.events.append((time.time(), action, key, detail))

    def entries(self):
        for p in self.root.glob("*/*/manifest.json"):
            try:
                yield json.loads(p.read_text())
            except (OSError, json.JSONDecodeError):
                continue

    # ---- put / get --------------------------------------------------------------
    def put(self, key, files: dict, roles: dict, meta: dict) -> dict:
        """Move files into the store. files: name -> path; roles: name -> role. Returns {name: new path}."""
        d = self._dir(key)
        d.mkdir(parents=True, exist_ok=True)
        m = self.manifest(key) or {"key": key, "created": time.time(), "pins": [], "files": {}}
        out = {}
        for name, src in files.items():
            role = roles.get(name, "intermediate")
            if role not in ROLES:
                raise ValueError(f"unknown role {role!r}")
            src = Path(src)
            if not src.exists():
                continue
            dst = d / f"{name}{''.join(src.suffixes) or ''}"
            if src.resolve() != dst.resolve():
                shutil.move(str(src), dst)
                _rmdir_if_empty(src.parent)
            m["files"][name] = {"path": dst.name, "role": role, "sha256": sha256_file(dst), "bytes": dst.stat().st_size}
            out[name] = str(dst)
        m.update(meta=meta, last_used=time.time())
        self._save(key, m)
        self._log("put", key, f"{len(out)} files")
        return out

    def get(self, key) -> dict | None:
        """Manifest with absolute file paths, or None (miss / failed verification)."""
        m = self.manifest(key)
        if m is None:
            self._log("miss", key)
            return None
        d = self._dir(key)
        paths = {}
        for name, f in m["files"].items():
            p = d / f["path"]
            if not p.exists():
                continue  # dropped intermediates are expected to be absent
            if self.policy.verify_on_get and sha256_file(p) != f["sha256"]:
                self._log("corrupt", key, name)
                self.evict(key, reason=f"hash mismatch on {name}", force=True)
                return None
            paths[name] = str(p)
        m["last_used"] = time.time()
        self._save(key, m)
        self._log("hit", key)
        return {**m, "paths": paths}

    # ---- retention --------------------------------------------------------------
    def pin(self, key, reason):
        m = self.manifest(key)
        if m and reason not in m["pins"]:
            m["pins"].append(reason)
            self._save(key, m)
            self._log("pin", key, reason)

    def drop_role(self, key, role):
        m = self.manifest(key)
        if not m:
            return 0
        freed = 0
        for name, f in list(m["files"].items()):
            p = self._dir(key) / f["path"]
            if f["role"] == role and p.exists():
                freed += p.stat().st_size
                p.unlink()
                f["dropped"] = time.time()
        self._save(key, m)
        if freed:
            self._log("drop", key, f"{role} {freed} B")
        return freed

    def evict(self, key, reason="", force=False) -> bool:
        m = self.manifest(key)
        if m is None:
            return False
        if m["pins"] and not force:
            return False
        shutil.rmtree(self._dir(key), ignore_errors=True)
        self._log("evict", key, reason)
        return True

    def size_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file())

    def gc(self) -> dict:
        """Evict unpinned entries, least recently used first, until under the size cap."""
        total = self.size_bytes()
        evicted = []
        if total > self.policy.max_bytes:
            for m in sorted((m for m in self.entries() if not m["pins"]), key=lambda m: m.get("last_used", 0)):
                if total <= self.policy.max_bytes:
                    break
                size = sum(p.stat().st_size for p in self._dir(m["key"]).glob("*") if p.is_file())
                if self.evict(m["key"], "lru over size cap"):
                    total -= size
                    evicted.append(m["key"])
        return {"bytes": total, "max_bytes": self.policy.max_bytes, "evicted": evicted}

    def after_record(self, key, rejected: bool) -> None:
        """Apply the record-time part of the policy."""
        if key is None or self.manifest(key) is None:
            return
        if rejected and self.policy.delete_rejected and self.evict(key, "rejected candidate"):
            return
        if self.policy.drop_intermediates_after_record:
            self.drop_role(key, "intermediate")

    def stats(self) -> dict:
        es = list(self.entries())
        return {"entries": len(es), "pinned": sum(bool(m["pins"]) for m in es), "bytes": self.size_bytes(),
                "max_bytes": self.policy.max_bytes}
