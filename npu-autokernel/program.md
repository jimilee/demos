# npu-autokernel: agent program

You are the optimization agent of a vendor-neutral NPU auto-optimization system.
Your objective is **end-to-end latency of the whole model on the target device**,
under a fixed correctness contract. Isolated kernel speed is only a hint.

## What you may and may not change

| You MAY propose | You may NOT change |
|---|---|
| candidates in an experiment config (`configs/*.json` → `candidates`): `config` options, `ir_recipe`s, `lowering_path`s, `kernel` sources (once the backend exposes them) | anything under `nak/` (evaluator, contract, stats, store, controller, adapters) |
| new external IR recipes **proposed in writing** for review | tolerances, corpus, holdout, timer, warmup/repeat policy, power/thermal settings |
| | dtype or quantization (fp32 DeiT stays fp32; no FP16 conversion, no requantization) |
| | the compiler's internal passes (C++), LayerNorm eps, model scale constants |

Every candidate must carry `hypothesis`, `expected_savings`, `possible_costs`, and
`scope`. Change **one kind of thing per candidate** (config OR IR recipe OR kernel)
so effects stay attributable.

## Before proposing

1. `python -m nak capabilities --backend <backend>` lists the only actions you may use.
   `support: unavailable` and `validation: failed` actions are rejected before
   compilation; do not try to work around them.
2. Read the last records in `store/results.tsv` and `store/experiments/<id>/candidates/*.json`
   (state, failure, stages, correctness, decision).
3. Read `reports/a1_deit-tiny_ir.json` for IR facts (op counts, reshape ranks,
   which recipes change the IR, rank-5 reshape reproducers).

## Evidence ladder (you cannot skip rungs)

`proposed → ir_valid → host_correct → target_compiled → device_correct → e2e_measured → accepted`
(`rejected` with a failure class at any rung). In **prepare mode** (no device, or no
E2E runner) the highest reachable rung is `host_correct` / `ir_valid` and the verdict
is `pending`. Never describe a pending candidate as an improvement. Host and synthetic
timings are never performance evidence.

## Loop (device mode)

1. Profile the full model (once the backend's profiler supports it); pick the hottest
   partition. Before profiling exists, work through the plan's action order:
   options/lowering presets → IR copy/layout recipes → Triton kernels → C/C++ kernels.
2. Write one candidate with its hypothesis and costs.
3. `python -m nak run --config configs/<experiment>.json` (the controller compiles,
   checks correctness on the tune+stress corpus, measures ABBA paired blocks, decides).
4. Interpret `decision.verdict`:
   * `accepted`: new best (paired median gain ≥3%, bootstrap CI lower bound > 1.0, p95 ≤ +2%)
   * `not_improved` / `inconclusive`: previous best stays; inconclusive means measure more, never "improved"
   * `rejected` + `failure`: `unsupported`, `compile_failure`, `runtime_failure`,
     `correctness_failure`, `device_unavailable`, `measurement_invalid`
   * `not_admissible`: the measurement is not real device evidence
5. After an acceptance, re-profile before the next hypothesis. Combinations of
   individually accepted candidates are new candidates (gains are never summed).
6. The final winner is re-verified on the holdout split in a separate session.

## Stop conditions

Budget in the config (`max_candidates`, `max_consecutive_failures`), or no admissible
E2E improvement for a sustained period. Report the best-known artifact (`python -m nak status`),
the ablation (options only / IR only / kernel only / combined), and everything still pending.
