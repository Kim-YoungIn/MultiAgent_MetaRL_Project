# Implementation Status — 2026-11-17

## 현재 구현

[Baseline](Project_Baseline.md)의 탐색→새 MA 과제 적응 목적을 유지한다. 실행법은 [README](../README.md)다.
학습 상대는 처음부터 유지하며,상대 학습·변화에 대한 적응 자체를 분리 조작하는 실험만 후속이다.

| 부분 | 현재 경로·근거 |
|---|---|
| 환경 | 동시 행동·Memory-One·랜덤 초기화·네 관측·비공개 horizon,[환경 계약](../04_design/Interface_Contract.md) |
| 공통 게임 | safe/risk/marker Template·θ/파라미터/역할 변화·단독 요소와 재조합,[공통 Template 계약](../04_design/Probe_Template_Contract.md) |
| 폭넓은 생성기 | 기존 관계형28규칙과 legacy 생성기를 비교/재현 경로로 보존 |
| 메타 학습 | 탐색자→동일Q 학생→Q/Expected SARSA 상대→G/PPO,별도K/H·배치·장치 선택·재개 |
| 경험 효과 | 동일History×Q표현3종,서로다른탐색×동일학생,자연MA/동일reference상대 조건부 점수 |
| 평가 | 공통 manifest의 학습 prefix 제한·고정 평가/seed,연구자θ식별성·검열을 포함한threshold 기능 |

## 최신 개발 실행

`runs/probe_template_dev_20261116/`:32학습/16새파라미터/16요소재조합,3seeds7/17/27,
200updates×32trials,K12/H48,H32,batch32,CPU1thread,reps1을 실제 실행했다.
실제384,000+제어307,200 steps/seed,3seed 총2,073,600(평가 별도),기존 실제 예산의50배다.
새 최종 평가를 사용하지 않은 개발 실행이며,규칙 다양성 수준 간 H1 비교는 아직 아니다.

| 후속 평가 | 초기 | 학습 탐색 | 랜덤 | ε-greedy |
|---|---:|---:|---:|---:|
| 새 파라미터 | .270037 | .271545 | .268849 | .245894 |
| 요소 재조합 | .267712 | .310767 | .284088 | .309214 |
| 새 상대 설정 | .317521 | .287267 | .292094 | .328400 |

재조합은 초기/랜덤 대비3seed 모두 양수지만 ε-greedy와 비슷했다. 새 파라미터는 차이가 작고 seed별 부호가 섞였고,
새 상대 설정은 초기 대비3seed 모두 하락했다. 보편적인 탐색 우위나H1/H2 성공은 확인하지 못했다.
θ짝32/32의 단계별 최선반응은 다르지만 temporal 보너스를 추가해 최선반응이 달라진 규칙은0/64였다.
재조합을 새로운 전략 구조로 부르지 않는다. 보너스 합계 .15→최대 .30과 정규화의 교란도 남는다.

## History 효과와 학생 활용

각 seed의 `history_seed*/report.html`은 같은 fresh 32 tasks/reps1/평가 seed에서 세 훈련 checkpoint를 paired 비교한다.
random-reference 상대에서 full Q의 learned-history 평균은 .243832, initial .251839, random .268849였다.
이 reference는 random-history에는 자연 쌍, learned에는 교차 쌍이므로 순수 경험 품질 순위로 단정하지 않는다.
같은 random-history를 제공한 학생 점수는 full .268849, 보상 제외/sign 표현 각각 .304245였다.
이는 Q 표현 비교이며 다른 학습 알고리즘의 성공이나 학생 기본값 변경의 근거는 아니다.
θ 짝 식별성은 random 100%, learned 87.5%였지만 알려진 prior의 연구자 reward-consistency 진단이며
정책 내부 규칙 분류를 보여주지 않는다. threshold는 이번 실행에서 지정하지 않았다.
고정 상대 zero-bonus 양성 대조의 정보 가치 .9는 정확 계산했지만 실제 학습 상대/보너스 조건에 자동 상속하지 않는다.

## 검증

실제 GPU를 포함한 전체 168 tests,91.578s,OK(skip 없음). Torch 없는 Python 168 tests,3.508s,OK(skipped=66).
독립 validator의 11개 계약 검증과 research_critic의 설계/결과 검토를 반영했다.
6 checkpoint×4집합의 24회 전체 평가 재현,1,920 평가 trial 비용/clone/업데이트 audit,
History 전체 hash/동일 상대/업데이트 수,seed7 전체 row 재생성,source snapshot/HTML 일치를 확인했다.
증거는 실행 폴더의 `verification.json`과 `strategy_audit.json`이다. Colab/브라우저 자체 실행은 미검증이다.

## 다음 작업

1. reference상대를random/initial/learned로교차해History순위의민감도를확인한다. 조건부진단을인과증거로확대하지않는다.
2. 개발집합에서학생/탐색자/목표를제한비교후고정한다. 실제규칙추론은연구자후보검사와분리한다.
3. 공통manifest·다중catalog·고정총예산/규칙당노출대조로H1을실행한다. 규칙수와전략차이를분리하고최종평가를봉인한다.
4. 총보너스/정규화/정보난도대조를추가한뒤재조합H2와상대변화적응분석을확장한다. 작은성공을상시관문으로삼지않는다.

## 실행 기반과 과거 근거

실호스트RTX5060Laptop8GB·기존torch2.15+cu132/runtime13.2를사용한다. 제한된프로세스의GPU차단은
장치부재와구별한다. 작은H32는CPU가빨랐고,H256/batch256측정에서CUDA가약1.23배빨랐다.
[GPU 측정](../runs/gpu_runtime_check_20261116/verification.json),[메타 계약](../04_design/Meta_Exploration_Contract.md)을따른다.
[이전전체이력](archive/Implementation_Status.md),[v1](sessions/10_meta_exploration_review.md),
[관계형v2](sessions/11_relational_rules_review.md)는당시근거이며현재지시가아니다.
