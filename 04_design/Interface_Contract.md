# Two Toy Models — Shared Interface Contract

## 현재 범위

Memory-One, 2인 동시 행동, 기본 `n=3`; 환경·규칙·RandomPolicy는 가변 `n>=2`.
Python 패키지는 `05_toy_models/mamrl/`. 표준 라이브러리만 사용한다.
이 문서는 코어 환경/단일 episode 계약이다. 선택적 recurrent 정책·유한 trial PPO는
`Recurrent_PPO_Contract.md`를 따른다. episode 도중 규칙 변경은 아직 범위 밖이다.

## 공통 값과 보상 텐서

- 플레이어 ID는 `0,1`, 행동은 정수 `0..n-1`.
- 글로벌 축 순서: `R[player, prev_0, prev_1, now_0, now_1]`.
- 게임 상태 `(prev_0,prev_1)`는 `n²`개, 플레이어별 보상 항목은 `n⁴`개.
- `RewardTensor(n, values)`는 위 축 순서(row-major)의 flat iterable을 복사한 불변 값.
  `.shape == (2,n,n,n,n)`, `tensor[i,p0,p1,a0,a1]`로 접근한다.
- `n`은 2 이상의 정수, seed는 `None` 또는 음이 아닌 정수. bool은 정수로 허용하지 않는다.
- 보상은 유한한 실수이며 bool은 허용하지 않는다. 잘못된 자료형은 TypeError,
  범위·shape·비유한 값 오류는 ValueError. 텐서 인덱스 범위 오류는 IndexError.

## Toy B: 명세 생성과 컴파일

- `RuleTerm(kind, coefficient=1.0, action=None)`와 `RuleSpec(n, terms)`는 불변 명세.
- 지원 kind와 플레이어 i의 보상 기여:
  - `coordination`: 현재 두 행동이 같으면 coefficient.
  - `repeat_own`: 현재 자기 행동이 직전 자기 행동과 같으면 coefficient.
  - `copy_opponent`: 현재 자기 행동이 직전 상대 행동과 같으면 coefficient.
  - `action_bonus`: 현재 자기 행동이 지정 action이면 coefficient.
  - `cyclic_zero_sum`: d=(now_i-now_other) mod n. 0<d<n/2이면 +coefficient,
    d>n/2이면 -coefficient, 나머지(동일 행동 및 짝수 n의 반대편)는 0.
  - 명시적 벤치 template `probe_repeat`/`probe_copy`: 기준 행동이면+1, 다른 기준이면-3,
    나머지는-.25/-.5, 전체에 coefficient를 곱함. 두 기준이 같으면 정답 우선.
    정확한 의미는 `Nominal_Policy_Benchmark_Contract.md`를 따른다. 기존 기본 sampler는 바꾸지 않는다.
- 한 명세의 모든 term 기여를 더한다. 부호·크기·term 선택으로 작은 논리적 variation을 만든다.
  배율/대체/범용 DSL은 아직 지원하지 않는다. 임의의 지속 상태는 추가하지 않는다.
- `generate_rule(n=3, seed=None, family="mixed") -> RuleSpec`.
  family는 `zero_sum`, `coordination`, `memory_one`, `mixed`.
  계수 크기 후보는 `{0,1,2,3,4}`; mixed는 부호도 선택해 페널티를 만든다.
  규칙은 생성 때만 샘플링하며 결과 명세가 실제 게임을 정의한다.
- `evaluate_rule(spec, previous, actions) -> (r0,r1)`는 직접 규칙 평가.
- `compile_rule(spec) -> RewardTensor`는 RNG 없이 결정적으로 전 조합을 계산.
- `matrix_to_tensor(payoffs) -> RewardTensor`: 글로벌 행동 축의 `(2,n,n)` 보상을
  previous-action 축 전체에 복제한다.
- RuleSpec은 정책 관측에 포함하지 않는다. 게임의 식별 가능성/전략적 다양성은 별도 연구 검증 대상.

## Toy A: 환경 및 경계

- `MemoryOneEnv(rule: RewardTensor, *, max_steps: int)`.
- `.n`은 공개된 행동 수. `max_steps`는 양의 정수이며 bool은 거부한다.
- `reset(*, seed=None, rule=None) -> (Observation, Observation)`.
  rule을 바꾸면 같은 n만 허용한다. 같은 rule과 명시 seed는 같은 초기 관측을 재현한다.
  seed=None은 환경의 로컬 RNG를 계속 사용한다. 전역 RNG를 변경하지 않는다.
- 초기 joint action은 각 플레이어별 균등·독립 추출. 초기 자기 보상은 0,
  `episode_start_flag=True`. 전용 START 상태는 사용하지 않는다.
- `step(action_0, action_1) -> StepResult`는 reset 후에만 가능하다.
  두 행동을 검증한 후 이전 상태와 두 현재 행동으로 보상을 계산하고,
  다음 상태를 현재 joint action으로 갱신한다. 유효하지 않은 행동은 상태/시간을 바꾸지 않는다.
- Observation은 frozen dataclass의 네 필드만 포함:
  `previous_own_action`, `previous_opponent_action`, `previous_own_reward`, `episode_start_flag`.
  다음 관측은 이번 행동과 자기 보상, flag=False를 가진다.
- StepResult는 `observations`, `rewards`, `terminated`, `truncated`를 가진다.
  게임 자체의 종료는 없어 terminated=False. 정확히 max_steps번째 step에서 truncated=True.
  이후 step은 RuntimeError; reset해야 다시 진행할 수 있다.
- 길이·남은 턴·rule/spec/상대 policy ID를 정책에 전달하지 않는다.
  비공개는 직접 비노출을 뜻하며 경험으로 길이를 추정하는 것까지 막지는 않는다.
- 종료용 내부 카운터는 게임 상태 n²와 구별한다. 한 episode 내 텐서는 고정된다.

## Policy와 실행기

- `Policy.initial_state(*, n, seed=None) -> state`.
- `Policy.act(observation, state, *, n) -> PolicyOutput(action, state)`.
  정책은 자기 관측과 n, 자기 내부 상태만 받는다. 환경·종료 길이를 받지 않는다.
  상태는 runner가 플레이어별로 따로 보관하고 episode 시작에 초기화한다.
- `RandomPolicy`는 플레이어별 로컬 RNG 상태를 사용하며 학습하지 않는다.
- 선택적 `RecurrentPolicy`도 같은 API로 checkpoint 기반 단일 episode를 실행한다.
  이 runner의 상태 초기화는 유지하며, episode 간 hidden을 유지하는 학습 trial은 별도 수집기를 사용한다.
- `run_episode(env, policies, *, seed=None, policy_seeds=(None,None)) -> EpisodeResult`.
  정책은 정확히 두 개. 각 턴 두 정책을 같은 이전 step의 관측에서 호출한 뒤 env.step에 전달한다.
  둘째 정책에 첫째 정책의 이번 행동을 넘기지 않는다.
- EpisodeResult는 `transitions`, `total_rewards`를 보존한다. 각 Transition은
  `observations`, `actions`, `rewards`, `next_observations`, `terminated`, `truncated`를 기록한다.
  이것은 향후 학습용 연결 지점이며 PPO rollout buffer나 recurrent trial 구현은 아니다.

## 독립 검증 기준

손계산 기대 보상과 글로벌 인덱스, 플레이어별 관측, first/reset/truncation,
불법 입력, 비노출, 로컬 RNG, seed 재현성, 정책 상태 격리 및 동시 행동을 검증한다.
규칙 평가 함수만 재사용한 비교를 독립 검증으로 간주하지 않는다.
`n=3,4,5`의 고정 예제 전 조합을 확인한다. 정책의 학습된 n 간 일반화는 주장하지 않는다.

## 연구자용 실행 기록과 리포트

정책에 추가 정보를 전달하지 않고 episode 종료 후 별도 기록을 만든다.
저장 스키마·검증·로컬 HTML은 `Run_Report_Contract.md`를 따른다.

공유 행동 집합 정책, 신규 checkpoint v2와 세 기능 벤치마크는
`Nominal_Policy_Benchmark_Contract.md`를 따른다. 관측 필드와 환경 state 표현은 확장하지 않는다.
