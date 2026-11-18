"""Use-case services (refactor 6/6). The CLI (nux/__main__.py) only parses arguments and prints;
everything it does is a function here, callable from tests and other tools.

Rules (tests/test_architecture.py): services reach backends only through the registry, never by
importing nux.backends.*; the CLI imports only nux.services.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "reports"


def write_json(path: Path, obj) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))
    return path


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(ROOT))
    except ValueError:
        return str(path)
