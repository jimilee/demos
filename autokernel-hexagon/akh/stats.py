"""Adoption rule (owned by the evaluator).

Samples are per-invocation steady-state DSP averages (one launcher process each),
collected with baseline/candidate interleaved in ABBA order on the same device.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass

MIN_GAIN = 0.01  # never adopt below 1% even if the baseline looks noise-free
HOST_REGRESSION_FLOOR = 0.05  # host regressions under 5% are ignored


def median(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else None


def rel_spread(xs):
    """Observed relative variation of a sample set: (p90-p10)/median for n>=5,
    else (max-min)/median."""
    xs = sorted(x for x in xs if x is not None)
    if len(xs) < 2:
        return None
    med = statistics.median(xs)
    if med == 0:
        return None
    if len(xs) >= 5:
        q = statistics.quantiles(xs, n=10, method="inclusive")
        lo, hi = q[0], q[-1]
    else:
        lo, hi = xs[0], xs[-1]
    return (hi - lo) / med


@dataclass
class Decision:
    verdict: str  # improved | need_more | keep_previous | hold_host_regression | no_timing
    gain: float | None
    noise: float | None
    host_change: float | None
    reason: str


def decide(base_dsp, cand_dsp, base_host=None, cand_host=None, rounds_done=1, max_rounds=3) -> Decision:
    mb, mc = median(base_dsp), median(cand_dsp)
    if mc is None:
        return Decision("no_timing", None, None, None, "candidate has no DSP timing; excluded from ranking")
    if mb is None:
        return Decision("no_timing", None, None, None, "baseline has no DSP timing in this comparison")
    gain = (mb - mc) / mb
    noise = rel_spread(base_dsp)
    threshold = max(noise or 0.0, MIN_GAIN)

    host_change = None
    hb, hc = median(base_host or []), median(cand_host or [])
    host_noise = rel_spread(base_host or []) or 0.0
    host_regressed = False
    if hb and hc:
        host_change = (hc - hb) / hb
        host_regressed = host_change > max(host_noise, HOST_REGRESSION_FLOOR)

    if gain > threshold:
        if host_regressed:
            return Decision("hold_host_regression", gain, noise, host_change,
                            f"DSP {gain:+.1%} > noise {threshold:.1%} but host latency {host_change:+.1%} "
                            f"(> {max(host_noise, HOST_REGRESSION_FLOOR):.1%}); investigate before adopting")
        return Decision("improved", gain, noise, host_change,
                        f"DSP median {mb:.3f}->{mc:.3f} us ({gain:+.1%}) exceeds baseline spread {threshold:.1%}")
    if gain > 0 and rounds_done < max_rounds:
        return Decision("need_more", gain, noise, host_change,
                        f"gain {gain:+.1%} within baseline spread {threshold:.1%}; measuring more")
    why = "slower" if gain <= 0 else f"gain {gain:+.1%} not above baseline spread {threshold:.1%} after {rounds_done} rounds"
    return Decision("keep_previous", gain, noise, host_change, f"{why}; previous best kept")
