"""Hexagon timing: wrapper-template patch and parsers.

What the stock hexagon-mlir launcher measures (checked at rev 9ec15e8, files
backend/hexagon_launcher_base.py and test/utils/dsp/include/hexagon_benchmark.h):

  * The generated C++ wrapper runs ON THE DSP (run_main_on_hexagon). It reads
    inputs from files, then times ``iterations`` back-to-back kernel calls with
    HAP_perf_get_time_us() and writes ``Perf:<avg>`` to perf.txt.
  * The average is integer division -> 1 us resolution, and the first (cold)
    call is inside the timed loop.
  * Thread-pool setup (multithreaded SPMD) happens before the timed block; the
    timed block contains tm.exec()/tm.exec_serial() dispatch + kernel bodies.
  * File I/O, FastRPC session setup, library load and adb are NOT included.

The AK patch (``install_timer_patch``) replaces only the benchmarking block:
  1 untimed-for-steady-state first call (reported separately as FirstCall_us),
  then ``iterations`` timed calls; average as double; pcycles alongside.
The stock ``Test_Info`` block is still written first, so the launcher's own
pass/fail check (get_test_result) keeps working.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

AK_TIMER_VERSION = "ak-timer-v1"

AK_BENCH_TEMPLATE = """
uint64_t ak_t0 = HAP_perf_get_time_us();
uint64_t ak_c0 = HAP_perf_get_pcycles();
{function_call}
uint64_t ak_t1 = HAP_perf_get_time_us();
uint64_t ak_c1 = HAP_perf_get_pcycles();
for (int ak_i = 0; ak_i < {iterations}; ++ak_i) {{
    {function_call}
}}
uint64_t ak_t2 = HAP_perf_get_time_us();
uint64_t ak_c2 = HAP_perf_get_pcycles();
double avg_time_us = (double)(ak_t2 - ak_t1) / (double){iterations};
TestReport tr("{func_name}", avg_time_us, "us", Result::Pass, "{save_path}");
tr.save();
{{
    FILE *ak_fp = fopen("{save_path}", "a");
    if (ak_fp) {{
        fprintf(ak_fp, "AK_Timer:{timer_version}\\n");
        fprintf(ak_fp, "AK_Iterations:%d\\n", (int){iterations});
        fprintf(ak_fp, "AK_FirstCall_us:%llu\\n", (unsigned long long)(ak_t1 - ak_t0));
        fprintf(ak_fp, "AK_FirstCall_pcycles:%llu\\n", (unsigned long long)(ak_c1 - ak_c0));
        fprintf(ak_fp, "AK_Steady_total_us:%llu\\n", (unsigned long long)(ak_t2 - ak_t1));
        fprintf(ak_fp, "AK_Steady_avg_pcycles:%f\\n", (double)(ak_c2 - ak_c1) / (double){iterations});
        fclose(ak_fp);
    }}
}}
"""


def render_ak_block(function_call: str, iterations: int, func_name: str, save_path: str) -> str:
    return AK_BENCH_TEMPLATE.format(function_call=function_call, iterations=int(iterations),
                                    func_name=func_name, save_path=save_path,
                                    timer_version=AK_TIMER_VERSION)


_PATCHED = False


def install_timer_patch():
    """Monkeypatch HexagonWrapperGenerator.generate_benchmarking_and_reporting."""
    global _PATCHED
    if _PATCHED:
        return True
    from triton.backends.qcom_hexagon_backend import hexagon_launcher_base as hlb

    def generate_benchmarking_and_reporting(self, function_call, exec_dir):
        return render_ak_block(function_call, self.iterations, self.func_name, f"{exec_dir}/perf.txt")

    hlb.HexagonWrapperGenerator.generate_benchmarking_and_reporting = generate_benchmarking_and_reporting
    _PATCHED = True
    return True


@dataclass
class PerfReport:
    name: str | None = None
    result: str | None = None
    perf: float | None = None
    units: str | None = None
    ak: dict = field(default_factory=dict)
    timer: str = "missing"  # ak-timer-v1 | stock | missing

    @property
    def steady_avg_us(self):
        if self.perf is None:
            return None
        return self.perf if self.units in ("us", "microseconds") else None


_KV = re.compile(r"^\s*([A-Za-z_]+)\s*:\s*(.*?)\s*$")


def parse_perf_text(text: str) -> PerfReport:
    rep = PerfReport()
    for line in text.splitlines():
        m = _KV.match(line)
        if not m:
            continue
        k, v = m.group(1), m.group(2)
        if k == "Name":
            rep.name = v
        elif k == "Result":
            rep.result = v
        elif k == "Perf":
            try:
                rep.perf = float(v)
            except ValueError:
                rep.perf = None
        elif k == "Units":
            rep.units = v
        elif k.startswith("AK_"):
            rep.ak[k[3:]] = v
    if rep.perf is not None:
        rep.timer = rep.ak.get("Timer", "stock") if rep.ak else "stock"
    return rep


def ak_float(rep: PerfReport, key: str):
    try:
        return float(rep.ak[key])
    except (KeyError, ValueError):
        return None


_CMD_TOOK = re.compile(r"Command took ([0-9.]+) seconds")


def parse_executor_log(text: str) -> dict:
    """Extract per-adb-step wall times printed by HexagonExecutor.

    Returns {'run_s': float|None, 'push_s': float, 'pull_s': float,
             'steps': [(cmd_head, seconds), ...], 'ran_on_device': bool}
    """
    steps = []
    last_cmd = None
    for line in text.splitlines():
        m = _CMD_TOOK.search(line)
        if m:
            if last_cmd is not None:
                steps.append((last_cmd, float(m.group(1))))
            last_cmd = None
        elif line.startswith("adb ") or line.startswith("mkdir ") or "run_main_on_hexagon" in line:
            last_cmd = line.strip()
    run_s = next((s for c, s in steps if "run_main_on_hexagon 3" in c), None)
    push_s = sum(s for c, s in steps if " push " in c)
    pull_s = sum(s for c, s in steps if " pull " in c)
    return {
        "run_s": run_s,
        "push_s": push_s,
        "pull_s": pull_s,
        "steps": steps,
        "ran_on_device": ("==> Running the kernel on device" in text and run_s is not None
                          and "==> Ran successfully" in text),
    }
