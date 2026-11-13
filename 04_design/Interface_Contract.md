# Two Toy Models — Shared Interface Contract (Draft)

## 공통 표기

- Agent IDs: `0`, `1`. Action: 정수 `0..n-1`; 초기 `n=3`.
- Game rule의 표준 텐서 축: `R[player, prev_0, prev_1, now_0, now_1]`.
- Tensor shape: `(2,n,n,n,n)`. **모든 축의 플레이어 순서는 글로벌 ID 기준**이다.
- 현재 환경 state: `s_t = (prev_0, prev_1)`; 선택된 joint action `(a0,a1)`이 다음 state가 된다.
- Reward: `r_i = R[i, prev_0, prev_1, a0, a1]`.
- Observation for player `i`: `previous_own_action`, `previous_opponent_action`, `previous_own_reward`, `episode_start_flag`.
- `episode_start_flag`는 정상 게임 상태의 일부가 아니라 경계 구분을 위한 관측 신호.

## Toy B → Toy A 최소 계약

`generate_rule(n, seed, variant_spec) -> (2,n,n,n,n)` 형식으로 전달하는 것을 현재 권장한다. 정확한 Python API는 구현 후 합의하고 여기 갱신한다.

Rule Generator는 학습 알고리즘을 호출하지 않는다. Environment는 게임 유형명을 하드코딩하지 않고 전달받은 텐서를 사용한다. Rule ID와 상대 정책 ID는 에이전트의 관측에 노출하지 않는다.

## 첫 턴 (미결정)

출발 후보: reset 때 가상 초기 joint action을 균등 추출, `previous_own_reward = 0`, `episode_start_flag = True`. 첫 턴 보상은 일반 텐서에서 계산. 단, `START` 전용 state와의 비교 여지는 유지한다.

## 최소 검증

- 정적 `n×n` matrix를 previous-action 축에 복제한 경우 이전 행동과 무관한 보상.
- Global / agent-local 행동 순서가 꼬이지 않는지 검증.
- Seed 재현성, 텐서 shape, simultaneous action, state update, reward 수치.
- 하나의 trial 내 episode 리셋과 task 리셋의 구분은 추후 학습 프로토콜에 연결.
