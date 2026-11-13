# Project Baseline

## 연구 목적 (FIXED)

Multi-Agent 반복 게임에서, Meta-Policy 학습이 새로운 규칙과 상대 전략에 대한 **Exploration** 및 **빠른 Adaptation**에 어떤 영향을 주는지 연구한다. 최고 보상·새 알고리즘 개발 자체보다 적응 과정의 실험적 분석을 중시한다. 첫 번째 참고 논문(Mali, 2026)의 Self-Play Meta-RL 구조를 출발점으로 삼되 그대로 재현할 필요는 없다.

## 현재 실험 작업 가정 (CURRENT; 변경 가능)

- 2인 동시 행동, 각 에이전트의 이산 행동 수 `n=3`에서 시작.
- 기본 게임 규칙은 `R[player, previous_action_0, previous_action_1, current_action_0, current_action_1]`로 표현. 텐서 크기 `(2,n,n,n,n)`; 보상 인덱스 수는 플레이어당 `n^4`.
- 게임의 관측 가능한 상태는 직전 joint action이며, 보상은 직전 joint action과 이번 joint action으로 계산.
- 에이전트는 직전 자신/상대 행동, 직전 자기 보상, episode 경계 표시를 관측. Rule ID와 상대 policy ID는 숨긴다.
- 첫 턴 처리 방식은 별도 설계 사항이며 임의 초기 joint action+START 플래그 방식이 후보.
- 메타 학습의 출발점은 RL² 스타일 Recurrent Policy + PPO. Independent Self-Play는 우선 검토할 학습 구조이며 아직 최종 확정은 아니다.

## 가까운 작업 목표

**Toy A:** Agent/Policy 인터페이스, 두 플레이어의 동시 행동, 상태·보상 처리, 학습 알고리즘 연결 지점, 최소 테스트.

**Toy B:** 기본 게임 Rule 및 Variation 정의, 파라미터 변화, 게임 규칙→보상 텐서 변환, 생성 규칙에 대한 테스트.

## 주의할 모델링 구분

- `n^4`는 State Space나 Observation Space의 크기가 아니라 **보상 함수의 state–joint-action 입력 조합 수**.
- 직전 joint action에 의한 Memory-One 게임의 환경 State Space는 `n^2` (별도 상태가 필요한 게임은 예외).
- 보상 텐서만으로 실제 묵찌빠의 지속적인 공격권 같은 추가 상태를 일반적으로 완전 표현할 수는 없다.
- Self-Play와 fixed-opponent 환경은 다른 실험 조건. 적응 동안의 hidden-state 갱신과 학습 파라미터 갱신을 구분한다.
- 학습 동안 task를 바꾸는 것과 평가 중 게임 안에서 규칙을 바꾸는 것은 별개의 연구 설계다.

## 미확정 사항

Task/episode 길이, 규칙 변경 시점, PPO 구현체·하이퍼파라미터, Exploration 목적함수/정책 분리, 평가 지표, 게임 가족의 범위 및 상대 정책 구성은 확정하지 않았다. `Open_Questions.md`를 참조한다.
