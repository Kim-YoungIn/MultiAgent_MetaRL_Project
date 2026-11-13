# Project Baseline

## 연구 목적 (FIXED)

Multi-Agent 반복 게임에서, Meta-Policy 학습이 새로운 규칙과 상대 전략에 대한 **Exploration** 및 **빠른 Adaptation**에 어떤 영향을 주는지 연구한다. 최고 보상·새 알고리즘 개발 자체보다 적응 과정의 실험적 분석을 중시한다. 첫 번째 참고 논문(Mali, 2026)의 Self-Play Meta-RL 구조를 출발점으로 삼되 그대로 재현할 필요는 없다.

## 현재 실험 작업 가정 (CURRENT; 변경 가능)

- 2인 동시 행동, 각 에이전트의 이산 행동 수 `n=3`에서 시작. 환경·논리 규칙·RandomPolicy는 가변 `n>=2`; `n=4,5`도 검증한다. 학습된 정책의 행동 공간 일반화는 별도 연구 과제다.
- 기본 게임 규칙은 `R[player, previous_action_0, previous_action_1, current_action_0, current_action_1]`로 표현. 텐서 크기 `(2,n,n,n,n)`; 보상 인덱스 수는 플레이어당 `n^4`.
- 현재 Generator는 자기 관점의 공통 함수 F를 사용한 대칭 게임이다. 공통 표는 `n^4`, 양쪽을 펼친 저장은 `2n^4`이며, 양쪽의 실현 보상이 같다는 뜻은 아니다. 환경은 비대칭 텐서도 수용하지만 생성 범위 확장은 별도 결정이다.
- 게임의 관측 가능한 상태는 직전 joint action이며, 보상은 직전 joint action과 이번 joint action으로 계산.
- 에이전트는 직전 자신/상대 행동, 직전 자기 보상, episode 경계 표시를 관측. Rule ID와 상대 policy ID는 숨긴다.
- Memory-One을 현재 구현 범위로 고정한다. 초기 joint action은 두 행동을 독립·균등 추출한다. 초기 자기 보상은 0, episode_start_flag=True; 첫 보상은 일반 텐서로 계산한다.
- Episode는 내부 설정인 고정 양의 길이에서 절단(truncated)한다. 길이·남은 턴은 정책에 직접 제공하지 않는다. 경험을 통한 길이 추정까지 금지하는 의미는 아니다.
- 규칙은 논리적 명세를 먼저 생성한 뒤 결정적으로 텐서로 컴파일한다. 규칙 생성 시 seed 기반 무작위성을 사용하되 episode 내 텐서는 고정한다. 임의 보상 텐서를 직접 샘플링하지 않는다.
- 정책은 관측·자기 내부 상태·행동 수만 받는다. 단일 episode runner는 매번 상태를 초기화한다.
  별도 recurrent 학습 수집기는 같은 규칙의 여러 episode를 유한 trial로 묶어 hidden을 유지하고 trial 시작에 초기화한다.
- 메타 학습의 출발점은 RL² 스타일 Recurrent Policy + PPO. Independent Self-Play는 우선 검토할 학습 구조이며 아직 최종 확정은 아니다.
- 첫 학습 구현은 P0 recurrent PPO learner 대 P1 고정 RandomPolicy다. Self-Play는 아직 구현하지 않았다.
  중간 episode 종료의 최종 관측은 관측 전용 GRU 이벤트로 전달하며 행동/discount timestep을 추가하지 않는다.
  return/GAE는 episode 경계를 넘고, 유한 trial의 최종 bootstrap은 0이다.

## 가까운 작업 목표

**결과 확인:** 단일 episode의 버전 있는 JSON 기록과 로컬 HTML 리포트.
실행 조건·보상·행동·규칙별 기여·상태별 보상표를 연구자가 확인한다. 정책 입력은 확장하지 않는다.
현재 RandomPolicy 실행 리포트를 학습/적응 성능이나 서로 다른 게임의 직접 비교 지표로 해석하지 않는다.

**Toy A:** 상태를 명시적으로 전달하는 Policy 인터페이스와 RandomPolicy 및 가변 n의 recurrent actor-critic을 연결했다.
유한 trial의 full-sequence PPO, frozen 평가, 체크포인트와 학습 리포트를 구현했다.
현재 기본 action_bonus 진단은 상대 행동에 의존하지 않아 상대 전략 적응의 증거가 아니다.
40-update 단일 seed 실행에서는 보상 향상과 episode 간 기억 이득을 관찰하지 못했다.
후속 seed 3개의 진단에서 고정 target0/2는 학습했지만 target1은 Random 수준에 머물렀다.
가변 target120updates에서 평균 보상이 소폭 상승했으나 뚜렷한 적응/기억 이득은 확인하지 못했다.
즉시 관측에 대한 지도학습 표현력 검사는 별도이며 PPO 성공이나 장기 기억의 증거가 아니다.
후속27개 번호 교환 대조에서 정체가 물리 행동이 아니라 내부 번호1을 따라갔다.
현재 숫자형 인코딩의 학습 편향이 확인됐으나 actor/GRU 각각의 원인과 기억 정체 전체는 미확정이다.
ordinal_v1은 비교 기준으로 보존했다. 새 action_set_v1은 각 행동에 동일 GRU/actor를 적용하고
숫자 크기 입력을 제거했다. 임의 행동 순열과 n=3/4/5 실행을 검증했으며 기억 크기는 H→nH로 달라진다.
학습 CLI는 새 구조를 기본으로 선택하고, 라이브러리 TrainConfig 기본은 과거 재현을 위해 ordinal_v1이다.
새 구조와 매 행동 hidden0 비교군의 3가족 파일럿에서 일부 보상 상승은 있으나 일관된 기억/적응 이득은 미확인이다.
이 결과는 학습기 진단으로 보존한다. 추가 최적화 진단을 연구 분포 설계의 필수 관문으로 삼지 않는다(D-018).
GPU는 후속 아이디어 I-011로만 보존한다.

**Toy B (초기 연결용):** 작은 논리 primitive의 가산 합성, RuleSpec 생성과 Compiler 분리, 정적 matrix 변환, 파라미터·구조 variation. 전략적 다양성은 구현 완료와 별도로 검증한다.
기능 벤치마크로 hidden_action, probe_memory, opponent_response를 추가했다.
probe_memory는 동일 현재 관측/다른 최적 행동의 도달 가능한 이력 witness를 검증했다.
상대 반응 과제는 coordination rule 고정 및 copy_previous/avoid_previous 상대를 사용한다.
기존 generator의 기본 sampling 분포는 이번 운영 조정으로 바뀌지 않는다.
다음 중심 작업은 통제된 논리 게임 분포·MA 상대군·규칙 차이와 held-out 평가의 설계다(D-018).
이후 공통 정책의 첫 전이 비교를 수행한다. 최초 Toy A/B 최소 구현·통합은 완료했지만 H1/H2는 직접 검증하지 않았다.
세부 인수 기준과 아직 확정하지 않은 후보는 `sessions/07_independent_research_review.md`, 동료 전달 초안은 `sessions/08_colleague_progress_note.md`를 따른다.

## 주의할 모델링 구분

- `n^4`는 State Space나 Observation Space의 크기가 아니라 **보상 함수의 state–joint-action 입력 조합 수**.
- 직전 joint action 공간은 `n^2`다. 숨은 task와 이력 반응형 상대의 메모리까지 포함한 전체 Markov state 크기와 구별한다. 보상 함수의 Memory-One 제약은 상대의 기억 길이를 제한하지 않는다.
- 보상 텐서만으로 실제 묵찌빠의 지속적인 공격권 같은 추가 상태를 일반적으로 완전 표현할 수는 없다.
- Self-Play와 fixed-opponent 환경은 다른 실험 조건. 적응 동안의 hidden-state 갱신과 학습 파라미터 갱신을 구분한다.
- 학습 동안 task를 바꾸는 것과 평가 중 게임 안에서 규칙을 바꾸는 것은 별개의 연구 설계다.

## 미확정 사항

연구용 task/trial 길이, 규칙 변경 시점, 최종 PPO 하이퍼파라미터, Exploration 목적함수/정책 분리, 평가 지표·상대 정책 구성은 미확정이다.
실행용 기본값은 별도 `Recurrent_PPO_Contract.md`의 n=3, 2 episodes×12 steps, 40 updates이며 연구적 최적 조건이 아니다.
단일 episode CLI의 기본 32 step도 실행 예시다. `Open_Questions.md`를 참조한다.
