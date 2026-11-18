# 설계안: 정밀도(dtype)를 최적화 탐색 대상으로 만들기

상태: **제안 (미구현, 결정 대기)** · 작성 2026-11-18

현재 nux는 workload의 dtype을 고정하고, 후보가 dtype을 바꾸거나 양자화하는 것을 막는다.
이 문서는 그 제한을 풀어 정밀도 변경(fp16, 혼합 정밀도, 이후 int8)을 후보로 탐색하려면
무엇이 필요한지, 어떤 순서로 만들지, 무엇을 먼저 결정해야 하는지 정리한다.

---

## 1. 현재 상태와 그 이유

| 항목 | 현재 |
|---|---|
| 정책 | 계획 §2 "dtype·양자화 보존". `hexagon.py`의 `opt.enableConversionToFp16`은 이 근거로 `unavailable` |
| fp16 실험 | 후보가 아니라 **별도 workload** `deit-tiny-fp16` (가중치 `.half()`, 계약 `deit-tiny-fp16-logits-v0`) |
| 정확성 계약 | 출력 logits를 원소 단위로 비교 (fp32: atol 1e-4 / rtol 1e-3, fp16: 2e-2 / 5e-3 + top-1 일치) |
| 과제 정확도 | `task_accuracy: "pending"`. 라벨 평가셋이 없음. 계획 §14 제안값은 top-1 하락 ≤ 0.1%p |
| best 기록 | (backend, workload, contract) 단위 |

dtype을 고정한 이유는 네 가지다.

1. 정밀도를 낮춘 이득과 컴파일러 최적화의 이득을 구분할 수 없다.
2. 계약이 dtype에 묶여 있다. 정밀도를 바꾸는 후보를 허용하려면 계약을 느슨하게 해야 하고,
   그러면 후보가 자기 평가 기준을 바꾸는 셈이 된다.
3. 정밀도를 얼마나 낮춰도 되는지는 과제 정확도로 판단해야 하는데, 이를 잴 수단이 없다.
4. 같은 체인 안에서 dtype이 바뀌면 짝 측정이 같은 계산끼리의 비교가 아니게 된다.

아래 설계는 이 네 가지를 각각 해결하는 방식으로 구성했다.

---

## 2. 목표와 범위

**목표**
* 고정된 원본 모델(fp32 DeiT-tiny)에서 출발해, 정밀도 변경을 포함한 후보가 속도와 과제 정확도
  둘 다의 기준을 통과하면 채택될 수 있게 한다.
* 정밀도로 얻은 이득과 컴파일러 최적화로 얻은 이득을 기록과 보고서에서 분리한다.
* 기존의 dtype 고정 실험(fp32·fp16 workload)과 그 기록은 그대로 둔다.

**1차 범위에 포함**
* 모듈 전체 fp16 변환
* op 종류·layer 단위 혼합 정밀도 (예: LayerNorm·softmax는 fp32 유지, matmul은 fp16)

**1차 범위에서 제외** (별도 설계 필요)
* int8 PTQ·QAT: 보정(calibration) 데이터셋, 양자화 파라미터 저장, 벤더별 양자화 형식이 추가로 필요하다
* 가중치 재학습이나 정밀도 이외의 모델 변경

---

## 3. 핵심 원칙

| # | 원칙 | 해결하는 문제 |
|---|---|---|
| P1 | 계약은 **후보의 dtype과 무관한 과제 기준**으로 정의한다. 모든 후보에 같은 기준을 적용한다 | 이유 2 |
| P2 | 계약 기준은 **라벨 평가셋의 과제 정확도**다. logits 원소 비교는 빠른 사전 필터로만 쓴다 | 이유 3 |
| P3 | 모든 기록에 후보의 **실효 정밀도**를 남기고, 성능 이득을 `precision`과 `compiler`로 나누어 보고한다 | 이유 1 |
| P4 | 정밀도 탐색은 **별도 계약 ID**로 실행한다. best 기록이 dtype 고정 체인과 섞이지 않는다 | 이유 4 |
| P5 | 계약은 지금처럼 evaluator 소유다. 에이전트는 정밀도 후보를 제안할 수 있지만 기준은 바꿀 수 없다 | 이유 2 |
| P6 | 채택 판단에 쓰는 정확도는 **단말 실제 출력**으로 잰다. host twin 정확도는 사전 필터일 뿐이다 | 벤더 fp16 누적 방식 차이 |

---

## 4. 구성 요소

### 4.1 정밀도 계획 (precision plan)

후보가 정밀도를 바꾸는 방법을 하나의 데이터 구조로 표현한다.

```json
{
  "action_id": "precision.mixed",
  "kind": "precision",
  "params": {
    "default": "float16",
    "keep_fp32": ["layernorm", "softmax", "classifier"],
    "accumulate": "float32"
  },
  "hypothesis": "matmul/GELU를 fp16으로: HMX/HVX fp16 경로 사용, 정규화·softmax는 수치 안정성 위해 fp32"
}
```

* `kind: "precision"`을 새 후보 종류로 추가한다. 기존 `config`·`ir_recipe`·`lowering_path`·`kernel`과
  구분해야 보고서에서 이득을 분리할 수 있다 (P3).
* op 이름 기준 지정(`layernorm` 등)은 workload가 제공하는 **op 분류표**로 해석한다. DeiT라면
  `workloads/deit.py`에 "op 종류 → torch 모듈 경로" 매핑을 둔다.
* 정밀도 계획은 컴파일 key의 일부다. 같은 IR recipe라도 정밀도 계획이 다르면 다른 산출물이다.
  (2차 리뷰 #2의 fingerprint 기반 key 수정이 먼저 되어 있어야 깔끔하다.)

### 4.2 정밀도 적용 위치

| 위치 | 방식 | 장점 | 단점 |
|---|---|---|---|
| **A. front end 이전 (torch 모듈)** | 지정한 submodule을 `.half()`하고 경계에 cast 삽입 후 export | 모든 backend에 공통. 구현이 단순하고 검증이 쉬움 | 후보마다 export 필요 (DeiT 약 10초, 캐시로 완화) |
| B. linalg IR 변환 | MLIR pass로 연산 타입 변경 | export 1회 | upstream에 범용 pass가 없음. 직접 구현·검증 부담 |
| C. 벤더 컴파일러 옵션 | 예: Hexagon-MLIR `enableConversionToFp16` | 벤더가 최적 경로 선택 | 벤더마다 다르고 범위 제어가 거침. 실효 정밀도를 확인하기 어려움 |

**권고: A를 기본으로 하고 C는 backend action으로 허용한다.** C를 쓰는 후보도 기록에는 실효 정밀도를
남겨야 하므로(P3), backend는 컴파일 결과에서 실제 연산 타입을 보고해야 한다(예: IR의 element type 집계).
보고할 수 없는 backend의 C 후보는 `unsupported`로 둔다.

### 4.3 계약: 과제 정확도 기반 (P1, P2)

새 계약 형식 예시 (`contracts/deit-tiny-task-top1-v0.json`, 2차 리뷰 #7의 계약 데이터화와 함께):

```json
{
  "contract_id": "deit-tiny-task-top1-v0",
  "kind": "task",
  "reference_model": "deit-tiny (fp32, pinned)",
  "eval_set": {"id": "<라벨 평가셋>", "sha256": "...", "splits": {"tune": 10000, "holdout": 10000}},
  "metric": "top1",
  "max_drop_pp": 0.1,
  "decision": "paired: candidate vs fp32 reference on the same images, one-sided 95% bound on drop <= max_drop_pp",
  "prefilter": {"finite": true, "corpus_top1_agreement_min": 0.99}
}
```

**평가 단계 (비용 순)**

| 단계 | 대상 | 비용 | 통과 조건 |
|---|---|---|---|
| 1 사전 필터 | 기존 corpus 17장, host twin | 수 초 | 유한값, top-1 일치율 |
| 2 속도 | 단말 짝 측정 (기존 PairedProtocol) | 수 분 | 기존 통계 기준 (3% 이득, CI 하한 > 1) |
| 3 과제 정확도 | 라벨 평가셋 tune split, **단말 출력** | 큼 | 하락의 단측 95% 상한 ≤ 0.1%p |
| 4 확인 | holdout split | 큼 | 같은 기준. 최종 채택 직전 1회만 |

정확도 평가는 비싸므로 속도 기준을 통과한 후보에만 실행한다.

**필요 표본 수 (중요).** 같은 이미지로 원본과 후보를 비교하면 차이의 표준오차는 대략
√(불일치율 / N)이다. fp16에서 top-1 불일치율이 1% 정도라면:

| N | 표준오차 | 0.1%p 판정 |
|---|---|---|
| 1,000 | 0.32%p | 불가 |
| 10,000 | 0.10%p | 경계 |
| 50,000 | 0.045%p | 가능 |

0.1%p 기준을 의미 있게 쓰려면 **수만 장 규모의 라벨 평가셋**이 필요하다. 기준값과 평가셋 규모는
함께 정해야 한다 (6장 결정 사항).

**평가셋 과적합 방지.** 탐색 중에는 tune split만 보고, holdout은 채택 직전 확인에만 쓴다.
holdout 결과는 다음 후보 생성에 쓰지 않는다 (기존 corpus의 tune/holdout 규칙과 같다).

### 4.4 기록과 보고 (P3)

* `CandidateRecord`에 `precision` 필드 추가: 정밀도 계획과, backend가 보고한 **실효 정밀도**
  (op 종류별 element type 집계).
* `decision`에 `gain_attribution` 추가:
  * `precision_only`: 같은 정밀도 계획을 backend 기본 설정으로 컴파일한 기준 대비 이득
  * `compiler`: 나머지 (recipe·옵션·커널)
* 보고서는 두 이득을 따로 표시하고, 정밀도가 바뀐 후보에는 정확도 하락량을 항상 함께 표시한다.
* schema 버전을 올린다 (2차 리뷰 #10의 schema 진화 규칙이 먼저 필요).

### 4.5 best 체인 분리 (P4)

* 정밀도 탐색은 새 계약 ID(`deit-tiny-task-top1-v0`)로 실행한다. best key가
  (backend, `deit-tiny`, `deit-tiny-task-top1-v0`)이 되어 기존 fp32·fp16 체인과 저절로 분리된다.
* baseline은 fp32 원본이다. 정밀도 후보는 fp32 baseline과 같은 입력으로 짝 측정한다.
* 참고값으로 `deit-tiny-fp16` workload의 best와도 비교해 보고서에 싣는다.

### 4.6 capability와 에이전트 규칙 (P5)

* backend manifest에 정밀도 능력을 선언한다:
  `precision: {"supported": ["float32", "float16"], "mixed": true, "reports_effective": true}`.
* `validate_action`은 `kind: "precision"` 후보를 **계약 `kind`가 `task`일 때만** 허용한다.
  logits 계약(`deit-tiny-fp32-logits-v0` 등)으로 실행하면 지금처럼 막힌다.
* `program.md`에 추가: 정밀도 후보는 가설에 예상 정확도 영향을 적는다. 계약·평가셋은 수정할 수 없다.
* 벤더 프로파일 importer의 `precision` hint(벤더가 실제로 fp16으로 실행한 op)를 정밀도 후보 seed로
  쓸 수 있다.

---

## 5. 구현 순서

| 단계 | 작업 | 선행 조건 |
|---|---|---|
| P0 | 6장 결정 (평가셋, 기준값, 계획 §2 개정) | 사용자 결정 |
| P1 | 2차 리뷰 #1·#2 (재시작 복원, fingerprint 기반 컴파일 key), #7 (계약 데이터화), #10 (schema 진화) | — |
| P2 | 라벨 평가셋 workload화: 고정 hash, tune/holdout 분할, host fp32 reference 정확도 기록 | P0 |
| P3 | `kind: "precision"`, 정밀도 계획 해석기(방식 A), 실효 정밀도 보고, 기록 필드, 이득 분리 | P1 |
| P4 | `task` 계약과 4단계 평가 (사전 필터 → 속도 → tune 정확도 → holdout) | P2, P3 |
| P5 | host에서 전체 경로 검증: fp32 vs 모듈 전체 fp16 vs 혼합 정밀도의 host 정확도를 비교하고, 판정 로직을 테스트용 device fixture로 확인 | P4 |
| P6 | 단말에서 실제 실행: 단말 출력 기반 정확도와 속도로 첫 채택 판정 | 단말·SDK·E2E runner (2차 리뷰 #3) |

P1–P5는 이 클라우드 환경에서 진행할 수 있다. P6은 Hexagon toolchain과 단말이 있어야 한다.

---

## 6. 결정이 필요한 사항

| # | 질문 | 선택지·고려점 |
|---|---|---|
| D1 | 계획 §2 개정 여부와 문구 | 제안 문구: "dtype·양자화는 기본적으로 보존한다. 단 `task` 계약으로 실행하는 정밀도 탐색 실험에서는 정밀도 변경 후보를 허용하며, 채택은 과제 정확도 기준과 단말 측정을 모두 통과해야 한다." |
| D2 | 라벨 평가셋 | ImageNet-1k val (50,000장)은 라이선스상 연구·비상업 용도 제한이 있어 사용 조건 확인이 필요하다. 대안은 사용 조건이 맞는 다른 공개 라벨셋이나 사내 데이터. 규모는 4.3의 표본 수 표 참고 |
| D3 | 정확도 기준 | 계획 §14의 top-1 하락 ≤ 0.1%p를 유지할지. 평가셋 규모가 작다면 기준을 완화하거나 판정 방식을 바꿔야 한다 |
| D4 | 1차 범위 | 모듈 전체 fp16과 혼합 정밀도만 할지, int8까지 포함할지 (int8은 별도 설계 권장) |
| D5 | 벤더 옵션 경로(4.2 C) 허용 여부 | 실효 정밀도를 보고할 수 있는 backend에서만 허용할지 |

---

## 7. 위험

* **host와 단말의 정확도 차이**: 벤더 fp16 경로는 누적 정밀도·rounding이 다를 수 있다.
  host twin 정확도로 채택하지 않는다 (P6 원칙).
* **평가 비용**: 단말에서 수만 장을 추론하는 시간이 후보당 크다. 속도 기준 통과 후보로 한정하고,
  평가 결과를 (산출물 hash, 평가셋 hash) 단위로 캐시한다.
* **평가셋 과적합**: tune 결과로 탐색이 반복되면 tune split에 맞춰질 수 있다. holdout 확인을 생략하지 않는다.
* **이득 귀속 오류**: 정밀도 변경이 다른 최적화를 가능하게 하는 경우(예: fp16이어야 HexKL 사용 가능)
  이득을 깔끔하게 나눌 수 없다. 이 경우 보고서에 "결합 이득"으로 표시하고 억지로 나누지 않는다.
