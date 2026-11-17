"""Launch description shared by kernel.py and the runners, plus candidate I/O.

kernel.py returns a list of ``Launch`` objects. A runner decides how to
execute them: the host interpreter ignores ``hexagon_opts``; the Hexagon
runner passes them to the Hexagon-MLIR backend as compile/runtime options.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib.util
import json
import pprint
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

# Options the evaluator owns. kernel.py may not set them.
EVALUATOR_OWNED_OPTS = {"iterations", "deviceCleanup", "enableLWP", "enableETM"}

CONFIG_BEGIN = "# --- CONFIG BEGIN (search.py may rewrite this block) ---"
CONFIG_END = "# --- CONFIG END ---"


@dataclass
class Launch:
    name: str
    fn: object  # triton.jit function
    grid: tuple
    args: tuple  # positional runtime args (tensors / scalars)
    consts: dict = field(default_factory=dict)  # tl.constexpr kwargs
    hexagon_opts: dict = field(default_factory=dict)
    # Tensors the evaluator must treat as outputs of this launch (for
    # sentinel/evidence checks and for host-roundtrip accounting).
    outputs: tuple = ()


def file_sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_kernel_module(path: str | Path):
    """Import a candidate file under a unique module name (no caching)."""
    path = Path(path).resolve()
    sha = file_sha256(path)[:12]
    name = f"akh_candidate_{sha}_{abs(hash(str(path))) % 10**6}"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod  # triton.jit needs inspectable source
    spec.loader.exec_module(mod)
    for attr in ("CONFIG", "launches", "KERNEL_TYPE"):
        if not hasattr(mod, attr):
            raise AttributeError(f"{path} is missing required attribute {attr}")
    return mod


def effective_config(mod, override: dict | None = None) -> dict:
    cfg = copy.deepcopy(mod.CONFIG)
    if override:
        for k, v in override.items():
            if k == "hexagon" and isinstance(v, dict):
                cfg.setdefault("hexagon", {}).update(v)
            else:
                cfg[k] = v
    bad = EVALUATOR_OWNED_OPTS & set(cfg.get("hexagon", {}))
    if bad:
        raise ValueError(f"kernel config may not set evaluator-owned options: {sorted(bad)}")
    return cfg


def read_config_block(source: str) -> dict:
    m = re.search(re.escape(CONFIG_BEGIN) + r"\n(.*?)\n" + re.escape(CONFIG_END), source, re.S)
    if not m:
        raise ValueError("CONFIG markers not found")
    tree = ast.parse(m.group(1))
    for node in tree.body:
        if isinstance(node, ast.Assign) and node.targets[0].id == "CONFIG":
            return ast.literal_eval(node.value)
    raise ValueError("CONFIG assignment not found in marked block")


def write_config_block(source: str, cfg: dict) -> str:
    """Return source with the marked CONFIG block replaced by ``cfg``."""
    body = "CONFIG = " + pprint.pformat(cfg, sort_dicts=True, width=88)
    pat = re.escape(CONFIG_BEGIN) + r"\n.*?\n" + re.escape(CONFIG_END)
    new, n = re.subn(pat, lambda _: f"{CONFIG_BEGIN}\n{body}\n{CONFIG_END}", source, flags=re.S)
    if n != 1:
        raise ValueError("CONFIG markers not found exactly once")
    return new


def config_key(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:12]
