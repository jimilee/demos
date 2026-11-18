# nux — notes for new sessions

Vendor-neutral NPU auto-optimization layer (plan: `mobile_compiler_autokernel_plan_v0_3.md`, preparation stage A0–A4).
Read `README.md` §1 (structure, diagrams) before changing code.

## Environment

- Python venv: `../.venv-npu` (`. ../.venv-npu/bin/activate`). torch 2.15 (PyPI), torch-mlir 20261101 (PyPI), transformers 5.23.
- `export HF_HUB_DISABLE_XET=1` before anything that downloads DeiT weights (classic LFS path).
- No Snapdragon device, no Hexagon SDK, no Hexagon-MLIR build in the cloud environment. System `llc` 18 has the Hexagon target (max hexagonv73).
- Tests: `python -m pytest -q tests -p no:logging` (75 passed at commit 6bdd744). Tests set `NUX_AOT_WORKDIR` to a temp dir.
- Committed experiment stores (`store/`, `store_mock/`) are results: re-running a config APPENDS to them. For behavior checks, copy a config, set `"store": ".cache/verify_store"`, and compare `results.tsv` rows (state, failure, decision).

## Mock backends — caveats (read before using or changing them)

`mock-mlir` and `mock-sdk` (`nux/backends/mock.py`) exist to check the common layers (capability filtering,
failure classes, best restore, per-backend isolation) against two backend shapes without hardware (plan A3).

- **Timings are synthetic.** They declare `measurement.tier = "synthetic"`; the evaluator returns `not_admissible`,
  never `accepted`. Never quote mock timings or "speedups" as performance results.
- **Correctness passes trivially.** Mock outputs ARE the framework reference, so the correctness stage only fails when a
  fault is injected (`perturb`). Passing on a mock says nothing about a real compiler.
- **Fault-injection params** (`fail=compile|runtime`, `perturb`, `synthetic_speedup`) are for tests and the A3 configs only.
- **They are registered in the production registry**, so they appear in the CLI and in `registry.available()`.
  `run` prints a WARNING for synthetic-tier backends. Moving them to a test-only registry is an open decision
  (it would remove the CLI reproduction of `configs/a3_mock_*.json`).
- **They affect other backends' artifact keys today** (see open defect 2): the toolchain lock includes every registered
  backend, so editing a mock manifest changes the compile keys of real backends.
- The accept/restore path is tested with a **test-only device fixture** in `tests/test_controller.py`
  (mock bundle declaring device traits, not registered). It is the only coverage of `accepted`; no real backend has
  reached it.

## Open defects (from the 2nd architecture review, reproduced)

1. **Restart does not restore the best state.** `Controller.restore_best()` is never called by any run path, and even
   when called it only sets `best_artifact`; `best_record`/`best_session` stay `None`. After a restart, candidate keys
   are computed with parent "root" instead of the real best, so once a non-baseline best exists a wrong artifact could
   be served from the cache, and device-mode paired measurement would get a `None` base session.
2. **Compile keys include volatile, unrelated data.** `Controller.compile_key()` hashes the full toolchain lock
   (all backends' environments, including Hexagon `device_state` and adb paths). Plugging a device changes the key
   of unrelated backends. Groundwork exists: every manifest declares `compile_fingerprint`, plus
   `frontend.frontend_fingerprint()`; the fix is to key on those instead of `toolchain=self.lock` /
   `frontend_version()`. Add regression tests reproducing both scenarios.

Other open items: backend conformance suite (refactor step 5) not written; vendor knowledge still in common
`aot.py`; backend ids hard-coded in `services/aot_report.py` and `importers/pipeline.py`; `Workload` base class lacks
`submodules()`/`preprocess()` that services call; contracts are a Python dict in `contract.py`.

## Rules for this project

- Common modules (`controller`, `stages`, `evaluator`, `capability`, `store`, ...) must not name vendors or branch on
  backend identity; decide from manifest traits. `tests/test_architecture.py` and `tests/test_core.py` enforce parts of this.
- Do not open, decompile or disassemble vendor binaries (QAIRT/QNN context binaries, runtimes). Vendor references come
  only from the vendor's documented outputs (e.g. `qnn-profile-viewer` JSON, see `nux/importers/`).
- Only real device measurements can justify `accepted`; host and synthetic timings are never performance evidence.
- DeiT stays fp32 (no dtype/quantization changes); vendor fp16 profiles are hints only.
