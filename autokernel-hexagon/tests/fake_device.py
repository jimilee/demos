"""TEST-ONLY scripted device. Never exposed as a bench/search backend.

Outputs are computed for real by the Triton interpreter (so accuracy checks
are genuine); DSP/host timings are synthetic and scripted per candidate so the
adoption state machine can be exercised without hardware.
Failure injection via hexagon opts: fakeFail = compile | device | no_timing.
"""
import random

from akh import runners as R


class FakeDeviceRunner:
    name = "hexagon"  # behaves like the device backend for search.py
    produces_perf = True

    def __init__(self, timing_fn, seed=0, noise=0.01):
        self.interp = R.HostInterpRunner("/tmp")
        self.timing_fn = timing_fn  # (launch) -> (dsp_us, host_ms)
        self.rng = random.Random(seed)
        self.noise = noise
        self.calls = 0

    def check_device(self):
        return True, "device"

    def run(self, launches, tag, measure=True):
        self.calls += 1
        fail = next((l.hexagon_opts.get("fakeFail") for l in launches if l.hexagon_opts.get("fakeFail")), None)
        if fail == "compile":
            return R.RunResult(R.COMPILE_FAIL, [R.LaunchResult(launches[0].name, R.COMPILE_FAIL, "injected")], "injected")
        if fail == "device":
            return R.RunResult(R.DEVICE_FAIL, [R.LaunchResult(launches[0].name, R.DEVICE_FAIL, "injected")], "injected")
        rr = self.interp.run(launches, tag)
        if rr.status != R.OK:
            return rr
        for lr, l in zip(rr.launches, launches):
            if fail == "no_timing":
                lr.status, lr.error = R.TIMING_PARSE_FAIL, "injected: perf.txt missing"
                rr.status, rr.error = R.TIMING_PARSE_FAIL, lr.error
                continue
            dsp, host = self.timing_fn(l)
            j = lambda v: v * (1 + self.rng.uniform(-self.noise, self.noise))  # noqa: E731
            lr.dsp_avg_us, lr.dsp_first_call_us, lr.host_call_ms, lr.host_run_ms = j(dsp), j(dsp * 3), j(host), j(host / 2)
            lr.ir_hash = None
        return rr
