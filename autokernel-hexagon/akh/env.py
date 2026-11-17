"""Toolchain / target / device inventory recorded with every experiment."""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import subprocess
from pathlib import Path


def _git_rev(path):
    if not path or not Path(path).exists():
        return None
    try:
        out = subprocess.run(["git", "-C", str(path), "rev-parse", "--short=12", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except Exception:  # noqa: BLE001
        pass
    return None


def _cmd(cmd, timeout=15):
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return (p.stdout + p.stderr).strip()
    except Exception as e:  # noqa: BLE001
        return f"<{type(e).__name__}: {e}>"


def _sdk_version(root):
    if not root:
        return None
    m = re.search(r"(\d+\.\d+\.\d+(?:\.\d+)?)", str(root))
    for f in ("version.json", "incs/version.h"):
        p = Path(root) / f
        if p.exists():
            return (m.group(1) if m else "") + f" ({f} present)"
    return m.group(1) if m else None


def collect(adb_probe: bool = True) -> dict:
    info = {"host": {"python": platform.python_version(), "platform": platform.platform()}}
    try:
        import torch
        info["torch"] = torch.__version__
    except Exception as e:  # noqa: BLE001
        info["torch"] = f"<missing: {e}>"
    try:
        import triton
        tdir = Path(triton.__file__).resolve().parent
        info["triton"] = {"version": triton.__version__, "path": str(tdir),
                          "git_rev": _git_rev(tdir) or _git_rev(tdir.parent)}
    except Exception as e:  # noqa: BLE001
        info["triton"] = {"version": f"<missing: {e}>"}
    try:
        from triton.backends.qcom_hexagon_backend import driver  # noqa: F401
        info["hexagon_backend_importable"] = True
    except Exception as e:  # noqa: BLE001
        info["hexagon_backend_importable"] = f"no: {type(e).__name__}: {e}"

    env = os.environ
    info["hexagon_mlir"] = {"root": env.get("HEXAGON_MLIR_ROOT"), "git_rev": _git_rev(env.get("HEXAGON_MLIR_ROOT"))}
    info["sdk"] = {"HEXAGON_SDK_ROOT": env.get("HEXAGON_SDK_ROOT"),
                   "HEXAGON_SDK_VERSION": env.get("HEXAGON_SDK_VERSION"),
                   "detected": _sdk_version(env.get("HEXAGON_SDK_ROOT"))}
    tools = env.get("HEXAGON_TOOLS")
    clang = Path(tools, "bin", "hexagon-clang") if tools else None
    info["hexagon_tools"] = {"root": tools,
                             "clang_version": _cmd([str(clang), "--version"]).splitlines()[0]
                             if clang and clang.exists() else None}
    info["target_arch"] = env.get("HEXAGON_ARCH_VERSION")
    info["hexkl_root"] = env.get("HEXKL_ROOT")
    info["linalg_hexagon_opt"] = shutil.which("linalg-hexagon-opt")
    info["run_on_sim"] = env.get("RUN_ON_SIM", "0")

    dev = {"adb": shutil.which("adb"), "ANDROID_HOST": env.get("ANDROID_HOST"),
           "ANDROID_SERIAL": env.get("ANDROID_SERIAL")}
    if adb_probe and dev["adb"]:
        base = ["adb"] + (["-H", env["ANDROID_HOST"]] if env.get("ANDROID_HOST") else []) \
            + (["-s", env["ANDROID_SERIAL"]] if env.get("ANDROID_SERIAL") else [])
        dev["devices"] = _cmd(["adb"] + (["-H", env["ANDROID_HOST"]] if env.get("ANDROID_HOST") else []) + ["devices", "-l"])
        dev["state"] = _cmd(base + ["get-state"])
        if dev["state"] == "device":
            for prop in ("ro.soc.model", "ro.board.platform", "ro.product.model", "ro.build.fingerprint"):
                dev[prop] = _cmd(base + ["shell", "getprop", prop])
            dev["cdsp_verinfo"] = _cmd(base + ["shell", "cat /vendor/firmware_mnt/verinfo/ver_info.txt 2>/dev/null | grep -i cdsp"])
    info["device"] = dev
    info["can_run_on_device"] = bool(info["hexagon_backend_importable"] is True and dev.get("state") == "device"
                                     and env.get("HEXAGON_SDK_ROOT") and tools and info["target_arch"])
    return info


def short(info: dict) -> dict:
    """Flattened fields for results.tsv."""
    t = info.get("triton", {})
    return {
        "target": f"v{info.get('target_arch')}" if info.get("target_arch") else "",
        "soc": info.get("device", {}).get("ro.soc.model", ""),
        "triton_rev": f"{t.get('version', '')}@{t.get('git_rev') or '?'}",
        "hexmlir_rev": (info.get("hexagon_mlir") or {}).get("git_rev") or "",
        "sdk": (info.get("sdk") or {}).get("HEXAGON_SDK_VERSION") or (info.get("sdk") or {}).get("detected") or "",
    }


if __name__ == "__main__":
    print(json.dumps(collect(), indent=2))
