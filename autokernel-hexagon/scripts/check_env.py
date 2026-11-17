#!/usr/bin/env python3
"""Record toolchain revisions, SDK, target arch and device state to env/env.json.

Lists exactly what is missing for on-device execution.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from akh import env  # noqa: E402

REQUIRED = {
    "hexagon backend importable (triton built with hexagon-mlir plugin)": lambda i: i["hexagon_backend_importable"] is True,
    "HEXAGON_MLIR_ROOT checkout": lambda i: bool(i["hexagon_mlir"]["root"]),
    "HEXAGON_SDK_ROOT (Hexagon SDK 6.4.0.2)": lambda i: bool(i["sdk"]["HEXAGON_SDK_ROOT"]),
    "HEXAGON_TOOLS (hexagon-clang)": lambda i: bool(i["hexagon_tools"]["clang_version"]),
    "HEXAGON_ARCH_VERSION (73/75/79/81)": lambda i: bool(i["target_arch"]),
    "adb on PATH": lambda i: bool(i["device"]["adb"]),
    "ANDROID_SERIAL set": lambda i: bool(i["device"]["ANDROID_SERIAL"]),
    "device state == device": lambda i: i["device"].get("state") == "device",
}


def main():
    info = env.collect()
    info["collected_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    info["missing_for_device"] = [k for k, f in REQUIRED.items() if not f(info)]
    info["hexkl_available"] = bool(info.get("hexkl_root")) and info.get("target_arch") != "81"
    out = ROOT / "env" / "env.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"\nwritten: {out}")
    print("device execution possible:", not info["missing_for_device"])
    for m in info["missing_for_device"]:
        print("  missing:", m)
    ti = info.get("triton_install", {})
    if ti.get("issues"):
        print("\ntriton install (TRITON_BACKENDS_IN_TREE=%s, imported from %s):"
              % (ti.get("TRITON_BACKENDS_IN_TREE"), ti.get("imported_triton_dir")))
        for d in ti.get("distributions", []):
            print(f"  dist: {d['name']}=={d['version']}  editable={d['editable']}  {d['location']}")
        for i in ti["issues"]:
            print("  issue:", i)
        print("  hint: a non-editable triton wheel next to the hexagon-mlir editable build (version 3.7.0) is"
              " usually pulled in by a CUDA torch. In-tree discovery (default here) ignores its entry points;"
              " to remove it: `pip uninstall triton` only if `pip show triton` does NOT point into"
              " hexagon-mlir/triton, then reinstall the editable build (scripts/build_triton.sh) and"
              " use torch==2.13.0+cpu (ci/torch-requirements.txt).")
    return 0 if not info["missing_for_device"] else 2


if __name__ == "__main__":
    sys.exit(main())
