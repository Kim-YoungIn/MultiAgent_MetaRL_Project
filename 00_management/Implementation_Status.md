# Implementation Status — 2026-11-21

## D-031 실험 모델

봉인된 실행을 별도 디렉터리에서 재개했다. 기존 3/375회 기록은 시작 당시의 스냅샷이다. 카탈로그별 완료 건수와 원본 소스 복원 방법은 [재개 기록](../research/d031_completion/README.md)에 있다.

[실험 설계](../04_design/Adaptation_Experiment.md)의 실행 진입점은 `python -m experiment`다. 게임 엔진·규칙 Compiler·온라인 학습자를 공유하며, 모델·훈련 목표·평가·통계는 `experiment/`에 있다. [사용법](../README.md)

| 질문 | 측정 |
|---|---|
| 학생의 적응이 빨라지는가 | 같은 규칙·학습 상대·난수에서 후속 학생 성능을 k=0/3/6/12/24에 측정하고 M−P 면적을 계산 |
| 어떤 경험이 도움이 되는가 | 첫 관측 이후 입력을 가린 O, 학생 Q 제거, 수집 이력과 상대 상태의 3×3 교차, full/binned Q 및 hidden reset |
| 게임이나 상대를 바꿔도 유지되는가 | 규칙 수와 생성 가족, 단일/혼합 학습 상대, 새 상대 파라미터, 예외 조합, n3/4/5 전이 |
| 어느 범위까지 결론을 낼 수 있는가 | 사전 고정한 25개 모델 쌍에서 catalog와 훈련 seed를 계층 재표집하고 평가 규칙 표집 민감도를 별도 산출 |

v1 hidden 가족의 가능한 n3 표현 1,728개를 전수 열거했다. 정규화·행동 재명명 후 52개 중 과거 및 학습 규칙과 중복되지 않는 보상표는 2개였다. 이 결과를 근거로 hidden 보너스의 상대적 크기를 학습과 평가에 공통으로 표집하는 v2를 채택했다. [전수 결과와 지문](../runs/d031_acceptance_20261121/v1_capacity.json)

개발 실행은 7개 훈련 조건·15개 모델·51개 평가 파일·1,530시행을 끝냈다. 2턴 수학 대조는 정보 활용 2.9와 무정보 2.0, 1턴 및 우세 safe 조건에서는 정보 가치 0을 재현했다. 독립 검증에서 n3/n5 모델의 n3/4/5 전이와 경험 교차 대각선을 원시 시행과 대조했다. CPU와 CUDA 개발 실행의 패널 SHA256이 일치했고 CUDA에서도 주 조건 세 방법의 훈련·평가가 완료됐다. 전체 회귀 검사는 실제 WSL GPU에서 206개가 통과했다. 선택 의존성이 없는 환경에서는 206개 중 90개를 건너뛰고 나머지가 통과했다. [개발 보고서](../runs/d031_v2_dev/summary.md)

봉인된 최종 패널에는 n=3/4/5 각각 fresh 64개(네 가족당 16개)와 exception 64개가 있다. 각 n에서 128개 모두 고유했고 과거·훈련·개발 보상표와의 중복은 0개였다. n3 fresh 규칙을 같은 가족 안에서 짝지은 2턴·균등 상대 검사에서는 32쌍 중 21쌍에서 첫 보상 관측의 가치가 양수였고, 1쌍에서는 즉시 최선의 첫 행동만 허용할 때 기대 성능이 낮아졌다. 이 값은 주어진 상대와 2턴 예산에서의 정보 가치다. 실제 학습 상대에 대한 효과는 주 평가에서 추정한다. [패널 중복 검사](../runs/d031_acceptance_20261121/panel_verification.json) · [수학 감사](../runs/d031_v2/audit_n3.json)

### 정규 예산으로 완료한 첫 비교

주 조건 c=1, j=1(훈련 seed 41101)에서 M·P·O 각각 800 updates × 32 trials를 실행했다. 세 방법의 훈련 규칙·상대 노출은 일치했고 제거 대조 포함 총 5,529,600턴을 사용했다. n3 최종 패널에서 세 모델과 네 대조 정책을 같은 3,072개 조건으로 평가했다. 21,504개 시행 기록에서 k=0 일치와 면적을 다시 계산했고 같은 재개 명령이 동일한 보고서를 생성했다.

미학습 fresh 규칙·두 학습 상대에서 적응 곡선 면적은 M 0.10997, P 0.11257, O 0.10183이었다. M−P는 −0.00260이다. 3턴에서 학생의 평균 후속 보상은 M 0.1782, P 0.1934였고 24턴에서는 각각 0.3353과 0.3363이었다. 사전 지정된 25개 모델 쌍 중 첫 한 쌍의 결과다. 주 조건 72회와 확장 300회 학습이 남았고 주가설 판정은 보류했다. [정규 실행 보고서](../runs/d031_v2/summary.md) · [시행별 재계산 검증](../runs/d031_acceptance_20261121/v2_primary_cell_verification.json)

## 선행 구현과 공통 기반

[Baseline](Project_Baseline.md)의 목적은 강화학습한 탐색 전략이 새 MA 과제의 적응을 더 빠르게 하는가다.
학습 상대·관측 네 필드·Memory-One·대칭 공통F·무작위 초기화·비공개 horizon을 유지한다.
실행법은 [README](../README.md)에 있다. D-030은 행동 수와 생성 규칙의 범위를 확대한 선행 실험이다.

| 부분 | 구현/해석 경계 |
|---|---|
| 행동 공간 | 관계형 메타 CLI에서 n3/4/5 학습·저장·재평가. logical/probe 메타는 n3 유지 |
| 생성 분포 | diverse_relational의 가변 수8/32/128,focused contest/broad4가족,기존 관계의 예외 조합 분할 |
| 분할/재현 | 같은 n/seed/평가개수에서 공통 heldout을 먼저 예약,훈련 prefix·전체 canonical 중복 제거·source 재생 검증 |
| 탐색/적응 | action_set 탐색자→동일Q학생→온라인 Q/Expected SARSA 상대,G 기본/절대 후속목표 비교 |
| 전이 평가 | 같은 checkpoint의 n3/4/5 평가,각 n 안에서 random과 paired 비교;initial checkpoint 별도 동일 평가 |
| 경험 효과 | 동일History×학생 표현,History×상대 상태3×3·학생 online/frozen. frozen에서도 상대는 학습 |
| 실행 기반 | CPU/CUDA·배치·update 재개·별도K/H. 작은H32는 측정상CPU가 빨랐음 |

## D-030 실제 실행과 검증

`runs/action_rule_expansion_20261118/`에 다음 공학 인수를 저장했다.

- n3/4/5×focused/broad의6catalog를 각각128train/16fresh/16예외로 생성·검증했다.
  각catalog160개에서 canonical 중복0. source seed·거부 후보·정규화·실제 tensor·사회적 유인/단계별BR audit를 검사했다.
- 별도로6개 조건에서8/4/4규칙·seed7·2updates×4trials·K6/H12·H8·batch4로 실제 훈련했다.
  훈련 제어 포함1,440steps이며,각 저장 checkpoint×4평가집합의24회 평가(240trials)를 정확히 재현했다.
- 짧게 학습한 n3 checkpoint와 그 initial을 n3/4/5의 같은 평가 task/seed에서 각각 평가했다.
  각 실행2,880simulation steps,가중치 불변·같은 random 대조·행동 범위·상대 학습을 확인했다.
- 독립 validator의5개 계약 검증을 포함해 전체186tests,108.054s,OK(skip 없음,실제CUDA 포함).
  Torch 없는 Python186tests,8.527s,OK(skipped=78). 증거/로그는 실행 폴더의 `verification.json`에 있다.

이번 실행은 기능 인수이며 성능/전략 다양성/행동 공간 일반화의 증거가 아니다.
focused/broad는 생성 recipe 범위와 가족별 노출을 함께 바꾸므로 순수한 전략 다양성 조작으로 부르지 않는다.
n5 훈련의 labelled stage BR signature는focused93/broad80으로 가족 수와 단조 관계도 아니었다.
상성·최선반응·정보 난도·학생 재사용을 함께 보고 실제 연구 분포를 확정해야 한다.
같은H에서도 hidden은 n×H이며,n4 상성의 추가 무승부 등은 행동 수 효과와 구분한다.

## 이전 연구 결과의 의미

D-029의2목표×3seed200update 개발 비교는 `runs/objective_control_dev_20261118/`에 있다.
절대 후속목표는 G보다 새 상대 설정에서 높고 재조합에서 낮았으며,학습곡선은 비단조였다.
기본G를 바꿀 일관된 근거는 없었다. 교차 상대 상태에 따라 History 순위가 반전돼 순수 경험 품질 순위로 단정하지 않는다.
그 이전 probe의 재조합 개선도 새 파라미터/상대에서의 보편적 탐색 우위를 뜻하지 않았다.
수치/검증 상세는 [보존 상태](archive/Implementation_Status.md)와 각 실행의 report/study/verification에 있다.
H1/H2와 학습된 행동 공간 일반화는 아직 미검증이다.

## 실행 기반과 근거

실호스트RTX5060Laptop8GB·기존torch2.15+cu132/runtime13.2를 사용한다. 제한된프로세스의GPU차단은
장치부재와 구별한다. 작은H32는CPU가빨랐고,H256/batch256측정에서CUDA가약1.23배빨랐다.
[GPU 측정](../runs/gpu_runtime_check_20261116/verification.json),[메타 계약](../04_design/Meta_Exploration_Contract.md)을 따른다.
Colab/브라우저 자체 실행은 미검증이다. [이전 상태](archive/Implementation_Status.md),
[v1](sessions/10_meta_exploration_review.md),[관계형v2](sessions/11_relational_rules_review.md)는 당시 근거다.
