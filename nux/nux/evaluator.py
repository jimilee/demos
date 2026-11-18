"""Read-only evaluator: correctness over the corpus, measurement admissibility, acceptance.

Hard rules (plan §2, §9, §14):
  * Acceptance needs a device-kind backend, real device measurements
    (Measurement.source == "device") without instrumentation, for BOTH
    baseline and candidate, plus device correctness. Host or synthetic timings
    can never produce `accepted`.
  * Missing quantities stay missing; nothing is imputed.
"""
from __future__ import annotations

from . import contract as C
from . import stats


def check_correctness(run_fn, workload, contract: C.Contract, corpus, splits=("tune", "stress")):
    """run_fn(tensors) -> output tensor. Compares with the workload's framework reference."""
    per = []
    for item in corpus:
        if item["split"] not in splits:
            continue
        ref = workload.reference(item["tensors"])
        try:
            out = run_fn(item["tensors"])
            r = C.compare(contract, out, ref)
        except Exception as e:  # noqa: BLE001
            r = {"ok": False, "reason": f"{type(e).__name__}: {e}"}
        r["input_id"] = item["id"]
        r["beyond_processor_domain"] = bool(item.get("beyond_processor_domain"))
        per.append(r)
    s = C.summarize(per)
    s["per_input"] = per
    s["splits"] = list(splits)
    return s


def admissible(measurement, bundle_kind: str):
    """Return (ok, reason) for using a measurement in an acceptance decision."""
    if measurement is None:
        return False, "no measurement"
    if bundle_kind != "device":
        return False, f"backend kind '{bundle_kind}' cannot produce acceptance evidence"
    if measurement.source != "device":
        return False, f"measurement source '{measurement.source}' is not a real device measurement"
    if measurement.instrumentation:
        return False, "instrumented run; final decisions use uninstrumented artifacts"
    if not measurement.blocks or any(len(b) == 0 for b in measurement.blocks):
        return False, "empty measurement blocks"
    return True, "ok"


def judge(base_meas, cand_meas, bundle_kind, policy: stats.Policy, device_correct: bool):
    ok_b, why_b = admissible(base_meas, bundle_kind)
    ok_c, why_c = admissible(cand_meas, bundle_kind)
    if not device_correct:
        return {"verdict": "rejected", "reasons": ["device correctness not established"]}
    if not (ok_b and ok_c):
        # still compute stats for information, but never accept
        info = stats.decide(base_meas.blocks, cand_meas.blocks, policy) if base_meas and cand_meas and base_meas.blocks and cand_meas.blocks else {}
        return {"verdict": "not_admissible", "reasons": [r for r in (why_b, why_c) if r != "ok"],
                "informational_stats": info}
    d = stats.decide(base_meas.blocks, cand_meas.blocks, policy)
    return {**d, "verdict": "accepted" if d["verdict"] == "improved" else d["verdict"]}
