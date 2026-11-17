#!/usr/bin/env python3
"""Step 0: run Qualcomm's official Triton softmax example unchanged on the device.

Runs  pytest -sv $HEXAGON_MLIR_ROOT/test/python/triton/test_softmax.py
with HEXAGON_MLIR_DUMP_DIR pointed at runs/softmax_smoke/<ts>/dump, then
collects: pytest verdict per dtype (the test compares the device output with
torch.softmax), the device Test_Info blocks (stock timer: integer-us average
over `iterations`=10 calls incl. the first), and the adb step times.
Writes runs/softmax_smoke/<ts>/smoke.json and env/softmax_smoke.json.
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from akh import env as envmod, hexagon_timer as ht  # noqa: E402


def main():
    root = os.environ.get("HEXAGON_MLIR_ROOT")
    out_dir = ROOT / "runs" / "softmax_smoke" / time.strftime("%Y%m%d_%H%M%S")
    dump = out_dir / "dump"
    dump.mkdir(parents=True)
    info = envmod.collect()
    res = {"env": envmod.short(info), "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    test = Path(root or "", "test/python/triton/test_softmax.py")
    if not root or not test.exists():
        res.update(status="device_unavailable", error=f"official test not found: {test}")
    elif info["device"].get("state") != "device":
        res.update(status="device_unavailable", error=f"adb state: {info['device'].get('state')!r}")
    elif info["hexagon_backend_importable"] is not True:
        res.update(status="compile_fail", error=f"hexagon backend not importable: {info['hexagon_backend_importable']}")
    else:
        e = dict(os.environ, HEXAGON_MLIR_DUMP_DIR=str(dump), TRITON_ALWAYS_COMPILE="1")
        e.pop("TRITON_INTERPRET", None)
        t0 = time.perf_counter()
        p = subprocess.run([sys.executable, "-m", "pytest", "-sv", str(test)], cwd=root, env=e,
                           capture_output=True, text=True)
        log = p.stdout + p.stderr
        (out_dir / "pytest.log").write_text(log)
        verdicts = dict(re.findall(r"test_softmax\[(\w+)\]\s+(PASSED|FAILED|ERROR)", log))
        reports = [ht.parse_perf_text(Path(f).read_text()).__dict__ for f in sorted(glob.glob(str(dump / "*/*_perf.txt")))]
        exe = ht.parse_executor_log(log)
        res.update(status="ok" if p.returncode == 0 else "failed", pytest_returncode=p.returncode,
                   verdicts=verdicts, device_reports=reports, wall_s=time.perf_counter() - t0,
                   adb_run_s_first=exe["run_s"], ran_on_device=exe["ran_on_device"],
                   log=str(out_dir / "pytest.log"))
    (out_dir / "smoke.json").write_text(json.dumps(res, indent=2, default=str))
    (ROOT / "env").mkdir(exist_ok=True)
    (ROOT / "env" / "softmax_smoke.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=2, default=str))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
