"""Paired-block statistics and acceptance rule (plan §9.2, §9.3).

Blocks, not individual requests, are the resampling unit: requests within a
block are correlated. The thresholds are the plan's initial proposals and must
be fixed per device before an acceptance run (self-vs-self noise check).
"""
from __future__ import annotations

import random
import statistics
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Policy:
    policy_id: str = "e2e-paired-v0"
    warmup: int = 20
    pairs: int = 10
    requests_per_block: int = 20
    min_median_gain: float = 0.03
    ci_level: float = 0.95
    ci_lower_gt: float = 1.0
    max_p95_regression: float = 0.02
    max_regression_shape: float = 0.02
    bootstrap: int = 2000
    seed: int = 0

    def to_dict(self):
        return asdict(self)


def pct(xs, q):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def summarize_blocks(blocks):
    flat = [x for b in blocks for x in b]
    return {"n_blocks": len(blocks), "n_requests": len(flat), "median": statistics.median(flat) if flat else None,
            "p95": pct(flat, 0.95), "block_medians": [statistics.median(b) for b in blocks if b]}


def paired_speedups(base_blocks, cand_blocks):
    """speedup_i = median(base block i) / median(cand block i)  (>1 means candidate faster)."""
    return [statistics.median(b) / statistics.median(c) for b, c in zip(base_blocks, cand_blocks) if b and c]


def bootstrap_ci(values, level=0.95, n=2000, seed=0):
    if len(values) < 2:
        return None, None
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(values, k=len(values))) for _ in range(n))
    a = (1 - level) / 2
    return meds[int(a * n)], meds[min(int((1 - a) * n), n - 1)]


def decide(base_blocks, cand_blocks, policy: Policy):
    """Pure statistics verdict. Whether the data is admissible (real device etc.) is the evaluator's job."""
    b, c = summarize_blocks(base_blocks), summarize_blocks(cand_blocks)
    sp = paired_speedups(base_blocks, cand_blocks)
    lo, hi = bootstrap_ci(sp, policy.ci_level, policy.bootstrap, policy.seed)
    gain = (b["median"] - c["median"]) / b["median"] if b["median"] and c["median"] else None
    p95_reg = (c["p95"] - b["p95"]) / b["p95"] if b["p95"] and c["p95"] else None
    reasons = []
    if gain is None or lo is None:
        verdict = "inconclusive"
        reasons.append("not enough paired blocks")
    else:
        if gain < policy.min_median_gain:
            reasons.append(f"median gain {gain:+.2%} < {policy.min_median_gain:.0%}")
        if lo <= policy.ci_lower_gt:
            reasons.append(f"speedup CI lower {lo:.4f} <= {policy.ci_lower_gt}")
        if p95_reg is not None and p95_reg > policy.max_p95_regression:
            reasons.append(f"p95 regression {p95_reg:+.2%} > {policy.max_p95_regression:.0%}")
        if not reasons:
            verdict = "improved"
        elif gain > 0 and lo is not None and lo <= policy.ci_lower_gt < (hi or 0):
            verdict = "inconclusive"  # measure more blocks; never 'improved'
        else:
            verdict = "not_improved"
    return {"verdict": verdict, "reasons": reasons, "median_gain": gain, "speedup_median": statistics.median(sp) if sp else None,
            "speedup_ci": [lo, hi], "p95_regression": p95_reg, "base": b, "cand": c, "paired_speedups": sp,
            "policy": policy.to_dict()}
