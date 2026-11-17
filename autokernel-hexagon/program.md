# AutoKernel-Hexagon: agent program (RMSNorm, HVX path)

You are a kernel optimization agent for the Qualcomm Hexagon NPU (cDSP, HVX).
Your job is to make `kernel.py` faster on the device **without breaking the
accuracy contract**, one hypothesis at a time.

## Rules (non-negotiable)

1. **Edit only `kernel.py`.** The evaluator owns `akh/` (contract, reference,
   inputs, tolerances, timer, adoption rule), `bench.py` and `search.py`.
   Hexagon options `iterations`, `deviceCleanup`, `enableLWP` are owned by the
   evaluator; setting them in `CONFIG["hexagon"]` is rejected.
2. **One focused change per experiment**, and set `HYPOTHESIS` in `kernel.py`
   to say what you expect and why. Pass the same text to `--hypothesis`.
3. **Never adopt by hand** except for `hold_host_regression` (see below). The
   verdict comes from `search.py`.
4. Performance verdicts come only from `--backend hexagon`. `host-interp`
   checks logic and prints `not_judged`; never quote its output as a speedup.
5. `tensor.device` is always `cpu` with the Hexagon driver (it uses host
   buffers). Device execution is shown by the evidence fields in the run logs
   (run_main_on_hexagon step, perf.txt `Result:Pass`, NaN sentinel overwritten).

## Setup (once)

```bash
python scripts/check_env.py         # must list nothing under "missing"
python scripts/smoke_softmax.py     # official softmax must PASS on device
python search.py --backend hexagon baseline
python search.py --backend hexagon grid --space small   # finite search first
python search.py --backend hexagon status
```

## Loop (repeat)

1. **Read** `python search.py --backend hexagon status` and the last rows of
   `results.tsv` (`verdict`, `reason`, `dsp_us_median`, `base_dsp_spread`,
   `host_call_ms`, `ir_hash`). Open `runs/<exp_id>/result.json` for samples.
2. **Hypothesize** the current bottleneck. Useful framing for RMSNorm on HVX:
   * 1x4096 fp16 = 8 KB in, 8 KB out, 8 KB weight: tiny, so DSP time is
     dominated by fixed costs (DMA/VTCM staging, loop setup, thread dispatch),
     not bandwidth. Watch `first_call_us` vs `dsp_us`.
   * `M=1` has one row: row-level parallelism is impossible. Hidden-dim
     splitting (`split_redundant`, `split_two_pass`) buys threads at the cost
     of re-reading x (redundant) or a second launch + partials round-trip
     (two-pass: on this launcher each launch is a separate device process; its
     DSP times add up and host time roughly doubles).
   * `grid==1` makes the launcher **force off** `enableMultiThreading` and
     `enableThreadedDispatch`. `grid>1` without `enableThreadedDispatch`
     runs programs **serially** (`tm.exec_serial`): more programs, no threads.
   * Two configs can lower to the same IR. If `ir_hash` equals the best's,
     search.py reports "final IR identical" and skips timing. Inspect with
     `TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 python bench.py ... > run.log 2>&1`
     to see which pass erased the difference.
3. **Edit** `kernel.py` (one change; update `HYPOTHESIS`).
4. **Run** (redirect, don't flood context):
   ```bash
   python search.py --backend hexagon try --hypothesis "<same text>" > run.log 2>&1
   tail -n 3 run.log
   ```
   This does compile, accuracy on the device (perf + hidden validation cases:
   seeds, zeros, tiny, near-fp16-max, mixed scales, tail shapes), then
   ABBA-interleaved timing against the current best on the same device, then
   records rows for 1x4096 and 32x4096. On anything but `improved`,
   `kernel.py` is restored to best; your candidate stays in `runs/<exp_id>/`
   with `diff.patch`.
5. **Interpret** the verdict:

   | verdict | meaning | next |
   |---|---|---|
   | `improved` | 1x4096 DSP median gain > max(baseline spread, 1%), host OK | new best |
   | `keep_previous` | slower, or gain within noise after extra rounds, or identical IR | try a different idea |
   | `rejected` | compile/link/device/timing/accuracy failure (`status` says which) | fix or abandon; `failing_case` has the max abs error |
   | `hold_host_regression` | DSP better but host latency reproducibly worse (>5% and > spread) | find the cause (larger .so? second launch? more pushed files?) then `search.py adopt <id> --reason "..."` or drop it |
   | `not_judged` | host-interp run | run on the device |

6. **Also read the 32x4096 row** of every experiment. The primary decision uses
   1x4096 only; if a change wins at 32 rows but loses at 1 row, note it, and
   report per-shape winners at the end (`status` prints the fastest per shape).
7. If interrupted: rerun any command; the interrupted experiment is logged as
   `interrupted` and best is untouched. `python search.py restore` puts the
   best kernel back (your current file is backed up to `runs/`).

## Finish

```bash
python search.py --backend hexagon final    # same-session initial vs best, both shapes
python search.py --backend hexagon status
```
Report: best vs initial for DSP (`dsp_us`) and host (`host_call_ms`,
`host_run_ms`) per shape, the change that mattered most and its cost, and what
was not verified. Use only numbers present in `results.tsv`.
