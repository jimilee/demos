# AutoKernel-Hexagon — Snapdragon NPU(HVX) 커널 자동 최적화 루프

[RightNow-AI/AutoKernel](https://github.com/RightNow-AI/autokernel)의 구조
(`program.md` + 에이전트가 고치는 `kernel.py` + 고정 벤치마크 + keep/revert +
`results.tsv`)를 [qualcomm/hexagon-mlir](https://github.com/qualcomm/hexagon-mlir)
백엔드에 맞춰 다시 구현한 프로젝트다. 첫 타깃은 **FP16 RMSNorm의 HVX 경로**이며,
주 채택 지표는 `(M=1, H=4096)`의 DSP 실행 시간이다.

> **이 저장소의 현재 상태 (2026-11-17 세션)**
> 이 세션의 실행 환경(클라우드 컨테이너)에는 **Snapdragon 단말, adb, Hexagon SDK,
> Hexagon-MLIR 빌드가 없다.** SDK 다운로드는 네트워크 정책(403)으로 막혔고, 연결된
> 사용자 머신도 없었다. 따라서 **이 저장소에 실기기 성능 수치는 하나도 없다.**
> 아래 "실행한 것 / 확인 못 한 것"에 실제로 실행한 범위를 그대로 적었다.
> 단말이 있는 호스트에서는 `scripts/run_device.sh` 한 번으로 전체 흐름이 실행된다.

---

## 1. 구조

| 파일 | 소유 | 책임 |
|---|---|---|
| `kernel.py` | **에이전트** | RMSNorm 후보 (Triton) + `launches()` wrapper + `CONFIG` + `HYPOTHESIS` |
| `bench.py` | 평가기 | CPU 참조, 입력 생성, Hexagon 실행, 정확도 검증, 시간 수집 (단일 커널 벤치마크 CLI) |
| `search.py` | 평가기 | baseline / 유한 탐색 / 에이전트 1회 실험 / 비교·채택·기각 / 기록 / 최선 후보 복원 |
| `akh/contract.py` | 평가기 | eps, dtype, seed, shape, 허용 오차, 검증 케이스 (버전 `rmsnorm-v2`) |
| `akh/reference.py` | 평가기 | CPU PyTorch 기준 구현 (RMSNorm, Linear) |
| `akh/runners.py` | 평가기 | `HexagonRunner`(실기기), `HostInterpRunner`(Triton 인터프리터, 로직 검증 전용), 실패 분류, 실행 근거 |
| `akh/hexagon_timer.py` | 평가기 | DSP 타이머 패치, `perf.txt`·executor 로그 파서 |
| `akh/stats.py` | 평가기 | 중앙값/변동 폭, 채택 규칙 |
| `akh/env.py`, `scripts/check_env.py` | 평가기 | 툴체인·SDK·타깃·단말 상태 기록 (`env/env.json`) |
| `scripts/smoke_softmax.py` | 평가기 | Qualcomm 공식 softmax 예제를 그대로 실행해 컴파일·단말 실행·비교·타이밍 확인 |
| `ops/linear_kernel.py`, `bench_linear.py` | 에이전트 / 평가기 | HMX 확장: 같은 Linear를 HVX vs HexKL(micro)로 비교 |
| `program.md` | — | 에이전트 실험 절차와 결과 해석 지침 |
| `results.tsv`, `results_linear.tsv`, `runs/<exp>/`, `state/<backend>/` | 평가기 | 실험 로그, 원시 결과·후보 스냅샷·diff, 최선/최초 후보 |

한 실험의 흐름: **후보 변경 → JIT 컴파일(별도 측정) → 실기기 정확도 검증 → ABBA 교차 반복 측정 → 비교 → 기록 → (채택 시) best 갱신 / (그 외) `kernel.py` 복원**.
유한 탐색(`grid`)과 에이전트 후보(`try`) 모두 같은 `bench.accuracy_suite`와
`bench.measure_once`를 거친다. 탐색 설정은 best `kernel.py`의 `CONFIG` 블록을 바꾼
완전한 소스 파일로 `runs/<exp>/kernel.py`에 저장되므로, 모든 후보는 파일 해시 하나로
식별된다.

## 2. 확인한 Hexagon-MLIR 동작 (로컬 소스 리비전 `9ec15e8`, 2026-10-13)

문서가 아니라 체크아웃 소스에서 확인했다. 리비전이 바뀌면 다시 확인할 것.

* **드라이버 등록**: `triton.runtime.driver.set_active(HexagonDriver())`
  (`backend/driver.py`). `get_active_torch_device()`는 `cpu`를 돌려준다. 텐서는
  호스트 버퍼이므로 **`tensor.device`로 NPU 실행을 판정할 수 없다.**
* **런처**: 매 호출마다 C++ wrapper 생성 → `hexagon-clang`으로 `.so` 링크 →
  `adb push` → `adb shell ... ./run_main_on_hexagon 3 <lib>.so` → 출력 `.raw`와
  `perf.txt`를 `adb pull`. 실패하면 executor가 `sys.exit(1)`을 호출한다. 그래서
  `HexagonRunner`가 `SystemExit`을 잡아 `link_fail`/`device_fail`로 나눈다.
* **런처 인자/옵션**: 커널 호출 kwargs가 `HexagonOptions`의 컴파일 옵션이 된다
  (`enableMultiThreading`, `enableThreadedDispatch`, `enableVTCMTiling`,
  `enableConvertToHexagonmem`, `enableHexagonmemCopyToDMA`, `enableHVXInlining`,
  `enableSCFLoopUnroll`, `enableSplitReduction`, `enableSplitReduceGeneric`,
  `enableHexKL`, `hexKLMode`, `num_threads`, `iterations`, `enableLWP`, ...).
* **런처 안전 게이트** (`triton_hexagon_launcher.py`):
  `grid==1`이면 `enableMultiThreading`과 `enableThreadedDispatch`를 **강제로 끈다**.
  `grid>1`이면 `enableHexagonmemCopyToDMA`를 끈다. `grid>1`인데
  `enableThreadedDispatch=False`이면 `tm.exec_serial`, 즉 **직렬 실행**이다.
  M=1 병렬화를 해석할 때 핵심 사실이다.
* **DSP 시간 (stock)**: wrapper가 DSP 위에서 `HAP_perf_get_time_us()`로
  `iterations`회 연속 호출을 재고, **정수 나눗셈** 평균(1 µs 해상도, 첫 cold 호출
  포함)을 `perf.txt`의 `Perf:`에 쓴다. 파일 I/O, FastRPC 세션, 라이브러리 로드,
  adb는 포함하지 않는다. 멀티스레드 SPMD의 thread-pool 생성은 측정 구간 밖이고,
  `tm.exec()` 디스패치는 안에 들어간다.
* **프로파일링 옵션**: `enableLWP`(하드웨어 lightweight profiling → `lwp.json`),
  ETM 트레이스(`HexagonProfiler`, v75/lanai 전용, 단말 재부팅 필요). 이 프로젝트의
  기본 측정에는 둘 다 쓰지 않는다 (평가기 소유 옵션).
* **문서와 소스의 차이**: user guide는 Hexagon SDK 6.4.0.2를 적었지만
  `scripts/set_local_env.sh`는 `HEXAGON_SDK_VERSION=6.6.0.0`을 export한다. 실제 값은
  `env/env.json`에 기록된다. HexKL 튜토리얼이 가리키는
  `test/python/triton/test_matmul_hexkl.py`는 이 리비전에 없다.
  Triton 서브모듈 핀은 `a9ced836…`이다 (`ci/setup_submodules.sh`).

### AK 타이머 패치 (`akh/hexagon_timer.py`)

`HexagonWrapperGenerator.generate_benchmarking_and_reporting`만 교체한다.
측정 블록은 다음과 같다.

1. **cold 호출 1회** → `AK_FirstCall_us` / `AK_FirstCall_pcycles` (초기화·첫 VTCM/DMA 비용, steady-state에서 제외)
2. **`iterations`회 timed 호출** → `Perf:` = 평균 µs (**double**), `AK_Steady_avg_pcycles`

stock `Test_Info` 블록을 먼저 그대로 쓰기 때문에 런처 자체의 `Result:` 판정은 그대로
동작한다. 이 패치는 실제 hexagon-mlir wrapper 생성기에 적용해 생성된 C++로 검증했다
(`tests/test_wrapper_patch.py`: grid=1 직접 호출, grid=4 `tm.exec`, grid=4
`tm.exec_serial`). **단말 위에서 컴파일·실행한 적은 아직 없다.**

## 3. 측정 정의

| 필드 | 구간 | 포함되는 것 |
|---|---|---|
| `dsp_us` | DSP 위, steady-state | 커널 본체 × `iterations`의 평균 (+ SPMD 디스패치). cold 호출 제외 |
| `first_call_us` | DSP 위 | 첫 호출 1회 (cold cache, 첫 VTCM/DMA 설정) |
| `host_run_ms` | 호스트 wall | `adb shell run_main_on_hexagon` 한 단계: FastRPC 세션 열기, DSP 라이브러리 로드, 입력 파일 읽기, cold + `iterations` 호출, 출력 파일 쓰기 |
| `host_call_ms` | 호스트 wall | Python 커널 호출 → 결과가 호스트 텐서에 돌아올 때까지: wrapper `.so` 링크, adb push(입력·라이브러리), run, adb pull(출력·perf) |
| `compile_s` | 호스트 | Triton→MLIR→object JIT (최초 1회, `fn.warmup`으로 분리) |

* 이 런처에는 **호출 1회당 FastRPC 지연이라는 개념이 없다** (커널마다 독립 프로세스).
  `host_*` 필드는 "이 실행 방식에서 호스트가 본 비용"으로 읽어야 한다.
* 다중 launch 후보(`split_two_pass`)의 시간은 launch별 합이다. 중간 partials는
  호스트 파일을 거쳐 왕복한다.
* **실행 근거** (`runs/.../logs/*.log`, `result.json`의 `evidence`):
  executor 로그의 `run_main_on_hexagon` 단계와 `Ran successfully`,
  단말에서 pull한 `perf.txt`의 `Result:Pass`와 커널 이름, 출력 버퍼에 미리 채운
  **NaN sentinel이 단말 결과로 덮였는지**. 하나라도 없으면 `device_fail`이다.

## 4. 정확도 계약 (`akh/contract.py`, `rmsnorm-v2`)

* `y = x * rsqrt(mean(x², -1) + eps) * w`, eps = 1e-6, x/w fp16 → fp32 reduction → y fp16.
* 기준: CPU PyTorch fp32 계산 후 fp16 반올림. 판정:
  요소별 `|y-ref| <= 1e-2 + 1e-2·|ref|` **그리고** `‖y-ref‖₂/‖ref‖₂ <= 2e-3`.
* 케이스: 성능 케이스(1x4096, 32x4096, seed 0)와 평가기가 만드는 별도 검증 입력
  (seed 11/12/13/14, 0, 1e-4 크기, fp16 최댓값 근처, 행마다 스케일이 다른 입력,
  tail shape 3x4000·1x4100·2x1000). 실패하면 `failing_case`에 케이스별 최대 절대 오차와
  rel-RMS 오차가 남는다. 모든 타이밍 샘플의 출력도 다시 검사한다.
* v1 → v2 변경 이유: 에이전트 실험 `hos0016`(평균을 `H` 대신 2의 거듭제곱인
  `BLOCK_H`로 나눔)이 v1에서 `3x4000`과 `2x1000`을 **통과했다.** 약 1.2% 스케일 오차가
  요소별 허용 오차 안에 들어갔기 때문이다. 집계 bound를 추가해 세 tail 케이스가 모두
  잡히게 했고, 회귀 테스트를 넣었다.

## 5. 채택 규칙 (`akh/stats.py`, `search.py`)

* baseline(현재 best)과 candidate를 **같은 runner·같은 단말에서 ABBA 순서로 교차**
  측정한다. 라운드당 shape별로 base·cand 각각 `--samples`개를 모은다. 개별 샘플은
  `runs/<exp>/result.json`에 전부 남는다.
* 1x4096 DSP **중앙값 이득 > max(baseline 변동 폭, 1%)**이면 `improved`다.
  변동 폭은 n≥5일 때 (p90−p10)/중앙값, 그 외에는 (max−min)/중앙값이다.
  이득이 0보다 크지만 변동 폭 이하이면 `need_more`로 라운드를 추가한다
  (`--max-rounds`, 기본 3). 그래도 불분명하면 `keep_previous`다.
* DSP는 좋아졌지만 `host_call_ms`가 max(5%, host 변동 폭)보다 재현 가능하게 나빠지면
  `hold_host_regression`이다. 원인을 조사한 뒤 `search.py adopt <id> --reason`으로만
  채택할 수 있다.
* 최종 IR(`asm["llir"]`) 해시가 best와 모든 shape에서 같으면 측정하지 않고
  "same lowering"으로 기록한다. 프런트엔드 설정 차이가 lowering에서 사라진 경우를
  보이게 하려는 것이다.
* 실패 분류 (`status`): `compile_fail`, `link_fail`, `device_unavailable`,
  `device_fail`, `timing_parse_fail`, `accuracy_fail`, `runtime_error`, `interrupted`.
  빠진 시간은 빈 칸으로 남기고 순위에서 제외한다.
* 상태는 backend별로 분리된다 (`state/hexagon/`, `state/host-interp/`). host 결과는
  절대로 성능 baseline이 될 수 없다. best는 완료된 비교 뒤에만 원자적으로 교체된다.
  중단되면 `inflight.json`이 남고, 다음 명령이 그 실험을 `interrupted`로 기록한다.
  `kernel.py`가 best와 다르면 덮어쓰기 전에 `runs/_kernel_backup_*.py`로 백업한다.

## 6. 탐색 범위 (`search.py` `SPACES["small"]`, `kernel.py` 전략)

* `row`: 행 단위. 전체 행을 한 블록(`block_h=0`)으로 하거나 `block_h` 청크로 나눈다
  (청크로 나누면 sum 패스와 normalize 패스에서 x를 **재로드**).
* `split_redundant`: hidden 차원을 S개로 나눈다. 각 프로그램이 **전체 행을 다시
  reduce**하고(x 중복 로드) 자기 slice만 정규화한다. 병합 비용이 없다.
* `split_two_pass`: launch 1은 slice별 부분합을 쓰고, launch 2는 S개 부분합을 병합해
  정규화한다(x slice 재로드). 런처 특성상 launch가 둘이면 단말 프로세스도 둘이다.
* 스레드: `num_programs`(grid), `enableThreadedDispatch`(실제 qurt 스레드 vs 직렬),
  `enableMultiThreading`(linalg FormVirtualThreads).
* 컴파일러 플래그: `use_rsqrt`, HVX inlining + SCF unroll(공식 softmax 플래그),
  split-reduction, split-reduce-generic, VTCM tiling off, DMA copy.
* 탐색은 greedy 방식이다. 각 override를 **현재 best CONFIG 위에** 하나씩 적용하고,
  채택되면 이후 후보의 기준이 바뀐다.

## 7. 재현 명령

### 7.1 단말 호스트 준비 (Hexagon-MLIR 공식 절차)

```bash
git clone https://github.com/qualcomm/hexagon-mlir && cd hexagon-mlir
./scripts/build_hexagon_mlir.sh        # triton(a9ced83)+triton_shared+backend, SDK/Tools/HexKL 다운로드
source scripts/set_local_env.sh        # HEXAGON_MLIR_ROOT, HEXAGON_SDK_ROOT, HEXAGON_TOOLS, HEXAGON_ARCH_VERSION, HEXKL_ROOT, PYTHONPATH ...
export ANDROID_SERIAL=<adb devices 출력>   # 원격 adb 서버면 ANDROID_HOST=<host>
pip install pytest numpy               # torch는 hexagon-mlir 환경 것을 그대로 사용
cd /path/to/autokernel-hexagon
```
`HEXAGON_ARCH_VERSION`은 단말의 cDSP 버전(v73/v75/v79/v81)과 맞아야 한다.
HexKL은 v81에서 지원하지 않는다.

### 7.2 단계별

```bash
python scripts/check_env.py          # env/env.json: triton/hexagon-mlir rev, SDK, arch, adb 상태, 빠진 항목
python scripts/smoke_softmax.py      # 공식 softmax 예제: 컴파일+단말 실행+torch 비교+Test_Info 타이밍
python bench.py --backend hexagon --samples 5                     # 단일 커널 벤치마크 (현재 kernel.py)
python bench.py --backend hexagon --config '{"strategy":"split_redundant","split":4,"num_programs":0,"hexagon":{"enableThreadedDispatch":true}}'
python search.py --backend hexagon baseline                       # 최초 성능 기준
python search.py --backend hexagon grid --space small             # 유한 탐색
python search.py --backend hexagon try --hypothesis "..."         # 에이전트 1회 실험
python search.py --backend hexagon final                          # 같은 세션 initial vs best (1x4096, 32x4096)
python search.py --backend hexagon status | restore
python bench_linear.py --backend hexagon --m 1,32 --k 512 --n 512 # HVX vs HexKL
bash scripts/run_device.sh                                        # 위 전체
```
IR 확인: `TRITON_ALWAYS_COMPILE=1 MLIR_ENABLE_DUMP=1 LLVM_IR_ENABLE_DUMP=1 python bench.py ... > run.log 2>&1`.
wrapper·`.so`·`perf.txt`는 `runs/_runner_logs/hexagon_dump/`
(`HEXAGON_MLIR_DUMP_DIR`)에 남는다.

### 7.3 에이전트 반복 실행

이 세션에는 단말이 없어서 에이전트 루프를 실기기에서 돌리지 못했다. 단말 호스트에서는
Claude Code를 이 디렉터리에서 열고 다음 프롬프트를 넘긴다.

```bash
claude "program.md를 읽고 Setup을 실행한 뒤, Loop를 한 번에 kernel.py 변경 하나씩 반복해. 결과는 results.tsv 숫자만 인용해."
```
수동으로 진행하려면 `program.md`의 Loop 1–7을 따른다
(`kernel.py` 편집 → `search.py try --hypothesis ...` → verdict 해석).

### 7.4 단말 없이 (이 세션에서 실제로 실행한 것)

```bash
python -m venv .venv && . .venv/bin/activate && pip install torch triton numpy pytest
python -m pytest -q tests                                  # 13 passed (+3 skipped)
AKH_HEXMLIR_SRC=/path/to/hexagon-mlir python -m pytest -q tests   # 16 passed (wrapper patch 포함)
python scripts/check_env.py
python search.py --backend host-interp baseline
python search.py --backend host-interp grid --space small
python search.py --backend host-interp try --hypothesis "..."
python bench_linear.py --backend host-interp
```

### 7.5 문제 해결: `No module named 'triton.backends.amd'`

Hexagon-MLIR이 고정한 Triton(`a9ced83`)은 **3.7.0**이고, `pip install -e .`로
editable 설치된다. 이 Triton은 entry point(`triton.backends` 그룹)로 backend를 찾는다.
같은 환경에 **다른 Triton 배포판**(예: CUDA torch가 끌어온 PyPI `triton-3.7.1.dist-info`)이
있으면, 그 dist-info의 `amd`/`nvidia` entry point가 섞여 import가 실패한다.

* 이 프로젝트는 Triton을 import하기 전에 `TRITON_BACKENDS_IN_TREE=1`
  (`akh/tritonenv.py`)을 기본값으로 둔다. 그러면 import된 `triton/backends/` 아래의
  디렉터리만 탐색하므로 stale entry point는 무시된다. 끄려면 `TRITON_BACKENDS_IN_TREE=0`.
  공식 pytest를 직접 돌릴 때는 셸에서 `export TRITON_BACKENDS_IN_TREE=1`.
* `python scripts/check_env.py`는 설치된 Triton 배포판과 깨진 entry point를
  `env/env.json`의 `triton_install`과 콘솔에 보여 준다.
* 3.7.1 wheel 자체에는 Hexagon 플러그인이 없어서 **단말 실행에 쓸 수 없다.**
  근본적으로 정리하려면 `pip show triton`이 `hexagon-mlir/triton`을 가리키지 **않을 때만**
  `pip uninstall triton`으로 wheel을 지운다. 그다음 `scripts/build_triton.sh`로 editable
  build를 다시 설치하고, torch는 `torch==2.13.0+cpu`(`ci/torch-requirements.txt`)를 쓴다.

## 8. 결과 (이 세션)

### 8.1 환경 기록 (`env/env.json`)

| 항목 | 값 |
|---|---|
| 호스트 | Linux x86_64 컨테이너, Python 3.13.18 |
| torch / triton (호스트) | 2.15.0 / 3.9.0 (PyPI, upstream. **Hexagon 플러그인 없음**) |
| Hexagon-MLIR | 소스 조사용으로만 clone `9ec15e8` (빌드 안 함) |
| Hexagon SDK / Tools / HexKL | 없음 (softwarecenter.qualcomm.com 다운로드가 프록시에서 403) |
| 타깃 아키텍처 | 미설정 |
| 단말 | adb 없음, 연결된 단말 없음 |
| 공식 softmax 예제 | `device_unavailable` (`env/softmax_smoke.json`) |

### 8.2 실기기 성능

**없음.** `results.tsv`의 `hex-env` 행이 `device_unavailable`로 기록되어 있다.
DSP·호스트 지연, 최초 기준 대비 개선율, shape별 우승 후보는 **측정하지 않았다.**

### 8.3 host-interp로 확인한 것 (성능 아님)

| 실험 | 내용 | 결과 |
|---|---|---|
| `hos0001_baseline` | 단순 row 커널 | 정확도 PASS (max abs err 1.95e-3), `accuracy_only_baseline` |
| `hos0002–0015_grid` | small space 14개 설정 전부 (row/chunk/split_redundant/split_two_pass/플래그) | 전부 정확도 PASS, `not_judged` |
| `hos0016_agent` | 평균을 `/BLOCK_H`로 계산 | `accuracy_fail` → 기각, `kernel.py` 복원. tail 3케이스 실패, 최대 오차 1.69 |
| `hos0017_agent` | w 로드를 reduction 앞으로 이동 | 정확도 PASS, `not_judged`, best 유지 |
| Linear | hvx / hexkl_micro 설정 × M=1, 32 (K=N=512) | 정확도 PASS. lowering은 인터프리터에 IR이 없어 판정 불가 |

host-interp는 **Hexagon 옵션을 무시하므로** 플래그만 다른 후보들은 호스트에서 서로
구별되지 않는다. 플래그 후보에 대한 정확도 PASS는 "Triton 로직이 맞다"는 의미이고,
Hexagon lowering 뒤의 정확도는 아니다.

상태 머신은 `tests/test_workflow.py`에서 **테스트 전용 가짜 단말**로 검증했다. 출력은
실제 인터프리터로 계산하고, 시간은 스크립트로 정한 합성값이다. 검증한 경로는
채택, 느린 후보 기각, compile/device 실패, 타이밍 누락(빈 칸·순위 제외), 노이즈 이내
이득(라운드 추가 후 유지), 호스트 회귀 보류 + 수동 채택, 동일 소스, 중단 복구, 사용자
변경 백업, 최초 대비 누적 개선이다. 이 합성 시간은 어디에도 기록되지 않는다.

## 9. HMX / HexKL 확장

* `ops/linear_kernel.py`: 같은 Triton `tl.dot`(fp16×fp16→fp32)을 `path=hvx`
  (`enableHexKL=False`)와 `path=hexkl_micro`(`enableHexKL=True, hexKLMode="micro"`)로
  컴파일한다. 기본 포인터 형식은 공식 예제와 같은 `make_block_ptr`이다.
  upstream Triton ≥3.6(호스트 인터프리터)에서 block pointer가 제거되어, 호스트에서는
  같은 연산의 offset 버전을 쓴다 (`ptr_style` 컬럼).
* **경로 확인**: 컴파일된 IR에서 `hexkl_micro` 호출 수를 센다(`lowering`,
  `hexkl_calls`). `enableHexKL`을 켰는데도 호출이 없으면 note에 "pattern did not match"를
  남긴다.
* 소스에서 확인한 지원 조건 (`MatmulToHexKLPass.cpp`): 2-D, static shape,
  f16 입력, f16/f32 출력. f32 출력은 f16으로 계산한 뒤 upcast하므로 허용 오차는
  `atol=rtol=1e-2`다. **macro 모드는 컴파일 타임 상수 weight(WH 레이아웃 prepack)가
  필요하다.** Triton 커널의 weight는 런타임 포인터라서 Triton 경로에서는 micro만
  가능하고, activation·output 레이아웃 변환이 **매 호출 timed 구간 안에서** 일어난다
  (`weight_layout=per_call`). 초기화 때 한 번 prepack하는 경로는 torch-mlir 쪽
  공식 예제 `test/python/torch-mlir/test_hexkl_macro_matmul.py`(macro, 상수 weight)로
  재현해야 한다.
* decode(M=1)와 prefill(M=32)을 별도 행으로 기록한다. HexKL 실행 조건:
  `HEXKL_ROOT` 설정, `HEXAGON_SDK_VERSION`에 맞는 HexKL 패키지, v81 아님.

## 10. 아직 확인하지 못한 것 (단말에서 먼저 볼 것)

1. 공식 softmax 예제의 단말 실행 (`scripts/smoke_softmax.py`).
2. `HexagonRunner` 전체 경로. 특히 `fn.warmup(..., **hexagon_opts)`이 pinned
   Triton(`a9ced83`)에서 컴파일만 수행하는지, `CompiledKernel.asm`에 `llir`가 있는지
   (IR 해시·HexKL 판정이 여기에 의존한다).
3. AK 타이머 패치가 단말 툴체인에서 컴파일되는지 (`HAP_perf_get_pcycles`,
   `fprintf`). 실패하면 `link_fail`로 분류된다. stock 타이머 파싱 경로도 남아 있다.
4. fp16 → fp32 upcast reduction의 lowering. 공식 rms_norm 테스트에는
   `.to(tl.float32)`가 주석 처리되어 있고, "fp16 sum reductions don't compile"이라는
   주석이 있다.
5. 스칼라 `tl.store` (two-pass 부분합)와 `tl.math.rsqrt`의 Hexagon lowering.
6. 실제 DSP/호스트 수치, 반복 측정 변동 폭, M=1에서 split 병렬화의 손익,
   (1,4096)과 (32,4096)의 우승 후보 차이.
7. HexKL micro의 M=1 지원 여부와 HVX 대비 성능.
