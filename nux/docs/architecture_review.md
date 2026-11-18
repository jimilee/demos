# nux 아키텍처 리뷰와 리팩토링 항목

최종 갱신: 2026-11-18 (커밋 `0b8c1ff` 기준)

이 문서는 nux 구조에 대한 두 차례의 비판적 리뷰와, 그에 따른 리팩토링 진행 상태를 기록한다.
새 세션에서 작업을 이어받을 때는 [`../CLAUDE.md`](../CLAUDE.md)의 "Open defects"와 함께 본다.

상태 표기: ✅ 해결 · ⚠️ 부분 해결 · ❌ 미해결

---

## 1. 1차 리뷰와 리팩토링 로드맵 (완료 현황)

### 1.1 1차 리뷰 지적 사항

| # | 심각도 | 문제 | 상태 |
|---|---|---|---|
| 1 | 높음 | controller가 숨은 상태(`_pending`, `_twin_corr`)로 흐름을 제어하는 god object | ✅ 단계 pipeline(`engine/stages.py`)과 `CandidateContext` |
| 2 | 높음 | 테스트 전용 개념(mock 분기)이 운영 로직에 섞임 | ✅ `informational_measurement` config 정책 |
| 3 | 높음 | backend `kind` 하나에 실행 위치·코드 생성·측정 등급이 섞임 | ✅ 직교 traits(`executes_on`, `target_codegen`, `measurement.tier`) |
| 4 | 높음 | backend끼리 import하고 front end가 backend 안에 있음 | ✅ `toolchain/{frontend,transforms,profilers}.py` |
| 5 | 중간 | `Artifact`가 타입 없는 만능 컨테이너 | ⚠️ 파일 역할·store manifest는 도입, 런타임 핸들 분리는 미완 (2차 #6) |
| 6 | 중간 | 캐시 key를 계산만 하고 조회 안 함, 작업 디렉토리 누수(4.83 GB) | ⚠️ content-addressed store와 보존 정책 도입, key 구성에 새 결함 (2차 #1·#2) |
| 7 | 중간 | 기록·config schema 미검증, 모르는 key 무시 | ⚠️ config 엄격 검증·record 검증 완료, store manifest 미검증 |
| 8 | 중간 | 벤더 격리를 정규식 테스트로만 보장 | ⚠️ import 방향은 AST로 검사, 이름 검사는 일부 모듈만 |
| 9 | 중간 | registry가 import 부작용에 의존, 장애 격리 없음 | ❌ (2차 #8) |
| 10 | 낮음–중간 | CLI에 비즈니스 로직과 중복 빌드 경로 | ✅ `services/`, CLI는 파싱·출력만 |
| 11 | 낮음 | 측정 프로토콜 책임 분산 | ✅ `engine/measurement.py` (`PairedProtocol`) |
| 12 | 낮음 | 힌트 추출이 이름 휴리스틱에 취약 | ❌ (op 분류 순서 버그 1건은 수정) |

### 1.2 리팩토링 단계

| 단계 | 내용 | 커밋 | 비고 |
|---|---|---|---|
| 1 | backend `kind` → 직교 traits | `b03bb10` | schema 0.2 |
| 2 | 단계 pipeline, 측정 프로토콜 | `ea872af` | warm-up 회귀를 새 테스트가 잡아 커밋 전 수정 |
| 3 | 공유 frontend / transforms / profilers | `785ad6a` | `llvm-aot` recipe 오류 미포착 버그 수정 |
| 4 | content-addressed 산출물 저장소와 보존 정책 | `982dc89`, `a9db6dd` | schema 0.3, 실험 ID 충돌 버그 수정, twin `.so`를 `verification` 역할로 보존 |
| 5 | config·record schema 검증 | `060affc` | **conformance suite는 미완료** |
| 6 | CLI → services 분리 | `1892fc1` | AOT 보고서가 backend를 직접 호출 |
| 7 | 평면 모듈 → 계층 패키지 (`core` / `storage` / `toolchain` / `engine`, `hints`→`importers/`, `corpus`→`workloads/`) | 이 문서 다음 커밋 | 계층 의존 방향 AST 테스트 추가, 동작 변화 없음 (29행 동일) |

각 단계마다 6개 실험 config를 임시 저장소에 다시 돌려, 커밋된 기록과 29개 결과 행(상태·실패 분류·판정)이
같음을 확인했다. 테스트는 26개에서 75개로, 7단계 후 85개로 늘었다.

---

## 2. 2차 리뷰 (현재 미해결 항목)

### 2.1 총평

1차 리뷰의 구조 문제는 대부분 해결됐다. 남은 문제는 두 갈래다.

* 리팩토링으로 새로 만든 **캐시·재시작 경로에 정확성 결함**이 있다.
* 시스템의 핵심 목적인 **실기기 채택이 한 번도 실제로 동작한 적이 없다.**

### 2.2 지적 사항

#### #1 [높음, 재현됨] 재시작 시 best 상태 미복원 — ❌

* **근거**: `Controller.restore_best()`는 어떤 실행 경로에서도 호출되지 않는다 (테스트에서만 사용).
  호출해도 `best_artifact`만 채우고 `best_record`·`best_session`은 `None`이다.
* **결과**: 재시작 뒤 후보의 컴파일 key가 실제 부모(best)가 아니라 root 기준으로 계산된다.
  * best가 baseline이 아닌 상황(device 모드에서 채택 이후)에서는 다른 부모로 빌드된 산출물이 cache hit로
    잘못 쓰일 수 있다.
  * device 모드의 짝 측정은 기준 session이 `None`이 된다.
* **권고**: controller 초기화 시 명시적인 resume 단계를 둔다 (best 기록 로드 → artifact 복원 → session 준비).
  복원한 key를 저장된 key와 대조해, 컴파일 입력이 바뀌었으면 옛 증거를 신뢰하지 않고 stale로 보고한다.
  컴파일 key의 부모는 메모리 상태가 아니라 저장된 best 포인터에서 가져온다.

#### #2 [높음, 재현됨] 컴파일 key에 휘발성·무관한 정보 포함 — ⚠️ 기반만 완료

* **근거**: `Controller.compile_key()`가 toolchain lock 전체를 hash한다. lock에는 등록된 모든 backend의 환경
  (Hexagon `device_state`, adb 경로 포함)이 들어 있다. 재현 결과 Hexagon 단말 연결 상태만 바꿔도
  무관한 `llvm-aot-host`의 key가 바뀌었다.
* **결과**: 단말 연결 변화마다 캐시가 무효화된다. mock manifest 변경도 실제 backend의 key를 바꾼다.
* **진행**: 각 backend manifest가 `compile_fingerprint`(컴파일 결과를 바꾸는 입력만)를 선언하고,
  `frontend.frontend_fingerprint()`와 `aot.code_fingerprint()`가 공유 부분을 제공한다 (`6bdd744`).
* **남은 일**: `compile_key()`에서 `toolchain=self.lock`을 `fingerprint=m.compile_fingerprint`로,
  `frontend=frontend.frontend_version()`을 `frontend=frontend.frontend_fingerprint()`로 바꾸고,
  두 재현 시나리오를 회귀 테스트로 추가한다.

#### #3 [높음] 핵심 경로가 실물로 미검증 — ❌

* `hexagon-mlir`의 컴파일·실행·`load_artifact` 코드는 실행된 적이 없다 (toolchain·단말 없음).
* 채택(`accepted`) 경로는 테스트 전용 device fixture로만 검증됐다.
* E2E device runner(계획 §9.1의 요청 경계, session 재사용)가 없어서 어떤 실제 backend도
  `acceptance_capable`이 될 수 없다.
* backend conformance suite(리팩토링 5단계 일부)가 없다.
* **권고**: conformance suite를 마무리하고, Hexagon-MLIR 환경이 있는 호스트에서 `smoke`·`run`을 한 번 실행해
  실제 출력으로 adapter를 고친다.

#### #4 [중간] 공통 계층에 벤더 지식 잔존 — ❌

* 공통 `toolchain/aot.py`에 Hexagon 타깃 정의와 "LLVM 18 Hexagon이 `fmaximum`을 선택하지 못함" 우회가 있다.
* `services/aot_report.py`와 `importers/pipeline.py`에 backend ID가 하드코딩되어 있다.
* CLI `smoke`의 기본 backend가 `hexagon-mlir`이다.
* 이름 검사 테스트는 일부 공통 모듈만 대상이라 이를 잡지 못한다.
* **권고**: 타깃별 지식(triple·cpu·우회)은 해당 backend로 옮기고 `toolchain/aot.py`는 "LLVM IR → llc"만 남긴다.
  보고서·seed 생성은 traits로 대상 backend를 고른다.

#### #5 [중간] Workload 인터페이스 불일치 — ❌

* 기반 클래스(`module`, `example_inputs`, `corpus`, `meta`, `lock`)에 없는 `submodules()`·`preprocess()`를
  service와 corpus 생성기가 호출한다. toy workload로 `smoke`를 실행하면 `AttributeError`가 난다.
* **권고**: 선택 기능을 명시적 인터페이스로 선언하고(예: `SupportsSubmodules`), 미지원이면 "unsupported"로 보고한다.

#### #6 [중간] 산출물 표현 이원화 — ❌

* 메모리의 `Artifact`에 런타임 핸들(`payload: object`, `twin: callable`)이 섞여 있다.
* 저장소 manifest(`meta`)는 검증 없는 dict라, backend 3곳의 `load_artifact`가 key를 손으로 꺼낸다.
* **권고**: 직렬화·검증되는 `ArtifactManifest`와 런타임 핸들 `LoadedArtifact`를 분리한다.

#### #7 [중간] 계약·정책이 코드에 박혀 있음 — ❌

* 정확성 계약이 `core/contract.py`의 `CONTRACTS` dict다. calibration 후 새 contract id를 만드는 운영 절차가 코드 수정이 된다.
* **권고**: contract를 버전 붙은 데이터 파일(`contracts/*.json`)로 옮기고 기록에 파일 hash를 남긴다.

#### #8 [중간] registry 장애 격리·discovery 부작용 — ❌

* `registry._load_all()`은 backend 모듈 하나라도 import에 실패하면 전체가 깨진다.
* `hexagon.manifest()`가 resolve마다 adb subprocess를 실행하고, toolchain lock이 모든 backend를 resolve한다.
* **권고**: backend별 import 오류는 "unavailable"로 격리하고, discovery는 프로세스당 한 번 캐시한다.

#### #9 [낮음] 단일 프로세스 가정 — ❌

* `results.tsv` 추가 쓰기와 best 포인터 교체에 잠금이 없다. 병렬 실험(계획 C단계)에서 경합이 생긴다.
* **권고**: 병렬화 전에 실험 단위 잠금 또는 append-only 이벤트 로그.

#### #10 [낮음] schema 진화 경로 없음 — ❌

* `SCHEMA_VERSION`을 올리면 이전 기록을 읽거나 마이그레이션할 수단이 없다 (이번 리팩토링에서 기록을 두 번 재생성).
* **권고**: 실기기 측정이 쌓이기 전에 읽기 호환 규칙과 마이그레이션을 정한다.

#### #11 [낮음] config 이중 표현 — ❌

* 검증된 `ExperimentConfig`를 `as_runtime_dict()`로 다시 dict로 바꿔 `cfg.get()`으로 읽는다. 타입 정보가 사라진다.
* **권고**: controller가 `ExperimentConfig`를 직접 받는다.

### 2.3 권고 순서

| 순서 | 작업 | 이유 |
|---|---|---|
| 1 | #1 재시작 복원, #2 fingerprint 기반 key, 재현 시나리오 회귀 테스트 | 정확성 결함. device 모드에서 채택이 시작되면 드러남 |
| 2 | conformance suite + Hexagon-MLIR 실환경 1회 실행 (#3) | 가장 큰 위험이 실행된 적 없는 코드에 있음 |
| 3 | 벤더 지식 이동 (#4), workload 인터페이스 (#5) | 두 번째 벤더 연결(C1) 전에 필요 |
| 4 | ArtifactManifest (#6), contract 데이터화 (#7), registry 격리 (#8) | 운영 안정성 |
| 5 | 동시성 (#9), schema 진화 (#10), config 일원화 (#11) | 규모가 커지기 전에 |

---

## 3. 관련 주의사항

### mock backend

`mock-mlir`·`mock-sdk`의 주의점(합성 시간은 절대 채택 불가, 정확성은 고장 주입 없이는 항상 통과,
운영 registry에 등록됨, #2로 인해 실제 backend의 key에 영향)은 README 1.6.0과 `CLAUDE.md`에 있다.

### 작업 이력의 특이사항

이 세션에서 다음 작업은 응답이 자동 안전 분류기에 의해 중간에 중단되어 완료하지 못했다.
코드나 도구 오류는 아니며, 분류기는 판단 근거를 제공하지 않는다.

* 리팩토링 5단계의 backend conformance suite
* 2차 리뷰 #1·#2 수정 중 controller 변경 (fingerprint 선언만 커밋됨)

새 세션에서 이어서 진행하는 것을 권한다.
