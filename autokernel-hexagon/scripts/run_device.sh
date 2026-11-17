#!/usr/bin/env bash
# Full on-device sequence. Run on the host that has the Hexagon-MLIR build,
# the Hexagon SDK and adb access to the Snapdragon device.
#
#   cd $HEXAGON_MLIR_ROOT && source scripts/set_local_env.sh   # official env
#   export ANDROID_SERIAL=<serial from `adb devices`>          # ANDROID_HOST optional
#   bash autokernel-hexagon/scripts/run_device.sh
set -euo pipefail
cd "$(dirname "$0")/.."
unset TRITON_INTERPRET RUN_ON_SIM

python scripts/check_env.py                      # env/env.json, exits 2 if anything is missing
python scripts/smoke_softmax.py                  # official softmax: compile + device run + compare + timing
python bench.py --backend hexagon --samples 5    # single-kernel benchmark of current kernel.py
python search.py --backend hexagon baseline      # initial perf baseline (first device-verified kernel)
python search.py --backend hexagon grid --space small
python search.py --backend hexagon final         # same-session initial vs best, both shapes
python search.py --backend hexagon status
python bench_linear.py --backend hexagon --m 1,32 --k 512 --n 512   # HVX vs HexKL (needs HEXKL_ROOT, not v81)
