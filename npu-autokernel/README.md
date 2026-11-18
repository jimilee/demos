# npu-autokernel — 멀티 벤더 모바일 NPU 자동 최적화 시스템 (초기 프로젝트)

`mobile_compiler_autokernel_plan_v0_3.md`(기획서 v0.3)를 바탕으로 만든 **준비 단계(A0–A4)**
구현이다. 공통 계층(controller·후보/결과 schema·정확성 계약·채택 정책·실험 저장소)과
벤더별 adapter bundle(compiler·runtime·profiler·capability)을 분리했다.
첫 백엔드는 Hexagon-MLIR이고, 첫 검증 모델은 **DeiT-tiny**(`facebook/deit-tiny-patch16-224`,
non-distilled)다. 기존 `../autokernel-hexagon/`(RMSNorm 커널 루프)와는 별도 프로젝트다.

> **현재 상태 (2026-11-18)**: 실행 환경에 Snapdragon 단말, Hexagon SDK, Hexagon-MLIR 빌드가
> 없다. 그래서 이 저장소의 모든 결과는 **준비 모드**에서 나온 것이다. `accepted` 후보와
> NPU·E2E 지연 수치는 **하나도 없다.** host 시간은 host CPU 시간이고 성능 근거가 아니다.

## 1. 구조

```
nak/
  schema.py        후보·측정·기록 schema, 상태 사다리, 실패 분류
  capability.py    CapabilityManifest / ActionSpec (support × validation), validate_action
  adapters.py      CompilerAdapter / RuntimeAdapter / ProfilerAdapter / AdapterBundle (본 프로젝트 인터페이스)
  registry.py      BackendRegistry (backend_id → bundle, profile 호환성 검사)
  controller.py    실험 루프 (prepare / device 모드, ABBA paired 측정, 예산, best 복원)
  evaluator.py     정확성 검사, 측정 적격성, 채택 판정 (읽기 전용 규칙)
  stats.py         paired block 통계, bootstrap CI, 채택 정책
  contract.py      정확성 계약 (버전 관리)
  store.py         불변 기록, backend별 best 격리, cache key, results.tsv
  mlir_host.py     벤더 중립 MLIR 도구: export, verifier, IR recipe, host LLVM JIT, reshape reproducer
  corpus.py        수치·회귀 corpus (tune / holdout / stress)
  workloads/       deit.py (DeiT-tiny), toy.py (테스트용)
  backends/        hexagon.py, host_linalg.py, mock.py   ← 벤더별 코드는 여기에만 둔다
configs/           실험 설정 (backend_id, profiles, workload, contract, candidates, budget, policy)
reports/           A1·A2 보고서, reshape reproducer (.mlir)
workloads/         DeiT lock / reference / corpus manifest
store/, store_mock/  실험 기록 (immutable JSON + results.tsv + best/)
program.md         에이전트 지침
```

공통 모듈에는 벤더 이름이 없다. `tests/test_core.py::test_no_vendor_branching_in_common_layers`가
이를 검사한다. 새 벤더를 추가하려면 `nak/backends/<vendor>.py`에서 `@register("<id>")`로
bundle을 등록하고, config와 conformance test만 추가하면 된다.

### 등록된 backend

| backend_id | kind | 역할 | E2E 측정 |
|---|---|---|---|
| `hexagon-mlir` | device | Hexagon-MLIR (rev `9ec15e8` 기준 capability). linalg → `translate_linalg_to_obj` → target object. 단말 정확성은 공식 one-shot launcher(`run_torch_mlir`) | **unavailable**: 공개 launcher는 호출마다 독립 프로세스로 실행되고 session 재사용이 없다. 계획 §9.1의 요청 경계 runner는 adapter 선행 과제 |
| `host-linalg-jit` | host_reference | torch-mlir linalg → LLVM → MLIR ExecutionEngine (host CPU). IR 후보의 `ir_valid`/`host_correct` 증거 | host 시간만 (채택 불가) |
| `mock-mlir`, `mock-sdk` | mock | A3 구조 검증. 출력은 실제 framework 결과, 시간은 synthetic | synthetic (채택 불가) |

### 상태 사다리와 채택 게이트

`proposed → ir_valid → host_correct → target_compiled → device_correct → e2e_measured → accepted`,
실패는 `rejected`와 실패 분류(`unsupported`, `compile_failure`, `runtime_failure`,
`correctness_failure`, `device_unavailable`, `measurement_invalid`, `budget_exhausted`)로 기록한다.
환경 때문에 못 한 단계(SDK 없음, 단말 없음)는 실패가 아니라 `pending`이다.

`accepted`가 되려면 다음을 모두 만족해야 한다.
1. backend kind가 `device`다.
2. baseline과 후보 모두 `Measurement.source == "device"`이고 instrumentation이 없다.
3. device 정확성을 통과했다.
4. paired block 판정을 통과했다: median 이득 ≥3%, speedup bootstrap 95% CI 하한 >1.0, p95 회귀 ≤2%.

이 숫자들은 계획 §9.3의 초기 제안값이다. self-vs-self 측정 뒤 `stats.Policy`로 고정한다.

## 2. 이번 세션에서 실제로 실행한 것

| 단계 | 명령 | 결과 |
|---|---|---|
| env | `python -m nak env` | `reports/env.json`: torch 2.15.0, torch-mlir 20261101, transformers 5.23.0. Hexagon backend import 불가, adb 없음 |
| **A0** workload | `python -m nak workload` | Hub 커밋 `b3428f1`의 `pytorch_model.bin`(sha256 `e1a51b0c…`)을 고정했다. `from_pretrained` 결과 missing/unexpected/mismatched key 0개. 구조 확인: 12 blocks, hidden 192, 3 heads, 197 tokens, 1000 classes. corpus 17개 입력 전부 출력이 결정적이고 유한하다 (`workloads/deit-tiny.reference.json`) |
| **A4** corpus | `python -m nak corpus` | tune 6 / holdout 4 / stress 7. 고정 processor(PIL backend)로 전처리하며 재생성 hash가 일치한다. task accuracy는 라벨 있는 평가셋이 없어 `pending` |
| **A1** IR | `python -m nak ir --jit-recipes fold-unit-extent-dims,fuse-elementwise` | linalg export 2209 ops, verifier 통과, 최대 reshape rank 4. **baseline·DropUnitDims·elementwise fusion IR을 host LLVM JIT로 실행해 eager 대비 max abs err 1.26e-5, top-1 일치** (tune 2 + stress 1, 계약 1e-4 + 1e-3·\|ref\| 이내) |
| **A2** compile smoke | `python -m nak smoke --backend hexagon-mlir` | LayerNorm(192) / encoder block 0 / 전체 모델 모두 export·verify 통과. target compile은 **pending (toolchain 없음)** |
| **A3** 구조 | `python -m nak run --config configs/a3_mock_{mlir,sdk}.json`, `pytest` | 같은 controller에 두 mock을 연결: 지원 action 필터가 다르게 동작하고, synthetic 1.5x 후보는 `not_admissible`, 주입한 compile·runtime·정확성 실패는 각각 분류되며 best는 유지된다 |
| DeiT 준비 실험 | `python -m nak run --config configs/deit_host_prepare.json` | baseline `host_correct`. `fold-unit-extent-dims`·`fuse-elementwise` → `host_correct`, verdict `pending`. `canonicalize-cse` → IR이 동일해 측정 생략. `kernel.replace` → capability로 차단 |
| DeiT Hexagon 실험 | `python -m nak run --config configs/deit_hexagon.json` | baseline과 옵션·recipe 후보 → `ir_valid` (target compile pending). `path.hexkl_matmul` → **fp32 모델이라 차단** (HexKL은 f16 입력만 지원, dtype 변경 금지). Triton kernel 교체 → ABI 미검증으로 차단 |
| 테스트 | `python -m pytest -q tests` | 15 passed |

### DeiT `collapse_shape` 이슈 관련 관찰 (계획 §13)

* torch-mlir 20261101의 linalg export에는 rank ≥5 reshape가 **없다** (최대 rank 4).
* 외부 recipe `linalg-fuse-elementwise-ops`를 적용하면 attention score에서 **rank-5 reshape 2개**가
  생긴다: `tensor<1x1x197x1x197xf32>` collapse, `tensor<1x3x197x197xf32>` → `1x3x197x1x197` expand.
  최소 reproducer는 `reports/reproducers/deit-tiny__fuse-elementwise__rank5_*.mlir`에 있다.
* host에서는 이 fused IR이 verifier를 통과하고 결과도 정확하다(위 표). 보고된 assertion은
  아직 **재현하지 못했다**. Hexagon 파이프라인(DropUnitDims 등)에 이 IR을 넣어 재현을 시도하는 것이
  toolchain이 생기면 첫 작업이다.

## 3. 재현

```bash
python -m venv .venv-npu && . .venv-npu/bin/activate
pip install torch torch-mlir==20261101 transformers safetensors pillow numpy pytest
export HF_HUB_DISABLE_XET=1          # 클래식 LFS 다운로드 경로 사용
cd npu-autokernel
python -m nak env
python -m nak workload               # A0 (가중치는 .cache/에 받는다. 커밋하지 않음)
python -m nak corpus                 # A4
python -m nak ir --jit-recipes fold-unit-extent-dims,fuse-elementwise   # A1 (~3분)
python -m nak smoke --backend hexagon-mlir                               # A2
python -m nak run --config configs/a3_mock_mlir.json                     # A3
python -m nak run --config configs/deit_host_prepare.json
python -m nak run --config configs/deit_hexagon.json
python -m nak status
python -m pytest -q tests
```
네트워크 허용이 필요한 host: `huggingface.co`, `us.aws.cdn.hf.co`, PyPI.

### Hexagon toolchain과 단말이 있는 호스트에서

```bash
cd hexagon-mlir && source scripts/set_local_env.sh && export ANDROID_SERIAL=<serial>
export TRITON_BACKENDS_IN_TREE=1     # stale triton dist-info 대응 (../autokernel-hexagon README 7.5)
cd npu-autokernel
python -m nak env                    # hexagon.toolchain_importable=true, device_state=device 확인
python -m nak capabilities --backend hexagon-mlir   # 실제 HexagonOptions 필드로 capability 재계산
python -m nak smoke --backend hexagon-mlir          # target_compiled 여부 (최소 op → block → 전체)
python -m nak run --config configs/deit_hexagon.json
```
* Hexagon 쪽 torch-mlir(`20260915.877`, hexagon-mlir `ci/torch-requirements.txt`)과 이 프로젝트의
  torch-mlir 버전이 다르면 IR 호환성 문제가 생길 수 있다. 그래서 adapter는 IR을 bytecode가 아니라
  **text**로 넘기고, provenance에 버전을 기록한다. 단말 실험에서는 hexagon 환경의 torch-mlir로
  export하는 것을 권장한다.
* 단말이 있어도 E2E 채택 루프는 아직 켜지지 않는다. `hexagon-mlir`의 ProfilerAdapter가 session을
  재사용하는 device runner를 갖출 때까지 `measure_e2e`가 `Unsupported`를 반환하기 때문이다.
  이 runner는 단말 위의 C/C++ runner로, 계획 §5.2와 §9.1의 경계를 구현해야 한다.
  그 전까지 단말에서 확인할 수 있는 최고 단계는 `device_correct`다.

## 4. 남은 일 (계획 단계 기준)

| 단계 | 상태 |
|---|---|
| A0 workload 고정 | 완료 (가중치·config·processor hash, 구조, 결정성) |
| A1 IR·host 검증 | 완료: linalg export + host JIT 실행 비교 (corpus 일부). 중간 activation 비교는 미구현 |
| A2 compile smoke | 대상 독립 단계까지만 완료. target object 생성은 toolchain 필요 |
| A3 공통 구조 | 완료: 두 mock + host + Hexagon adapter를 같은 controller에 연결. mock 채택 금지를 검증 |
| A4 corpus | 합성 이미지·stress·holdout 완료. 라벨 있는 평가셋(task accuracy)은 pending |
| B0–B3 단말 성능 | 미착수: 단말, SDK, E2E device runner 필요 |
| C0–C2 두 번째 벤더 | 미착수: 벤더·SDK 접근성 조사가 먼저 필요 |

알려진 한계:
* host JIT는 naive loop lowering이라 첫 실행이 약 11초, 이후 약 1.1초 걸린다. 정확성 확인용일 뿐이다.
* 정확성 계약 `deit-tiny-fp32-logits-v0`는 계획의 smoke 제안값이다. device baseline으로 calibration한 뒤
  새 contract id로 고정한다.
* DeiT 가중치는 `pytorch_model.bin`(pickle 형식)뿐이다. transformers가 `weights_only` 경로로
  로드하지만, 추후 safetensors 변환본을 만들어 hash로 고정하는 것이 안전하다.
