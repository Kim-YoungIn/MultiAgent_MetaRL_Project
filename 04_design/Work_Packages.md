# Toy Model Work Packages

## Toy A — Agent + Environment

목표: Rule Tensor를 주입하면 두 정책이 simultaneous action을 제출하고 각자 보상을 수신하는 최소 실행 환경. 이후 Recurrent PPO / Self-Play로 확장 가능한 분리된 인터페이스.

최소 인수 조건:
- `reset(seed, rule)` / `step(action_0, action_1)`에 준하는 환경 API
- 두 플레이어 관측과 개별 보상
- Episode 및 첫 턴 경계 처리
- Random Agent 두 개로 smoke test
- Tensor shape 오류 및 seed 오류를 명확히 검출

**주의:** 지금은 학습 알고리즘의 선정이나 PPO 하이퍼파라미터 확정을 요구하지 않는다.

## Toy B — Rule + Variation + Tensor Conversion

목표: 반복 게임의 보상 규칙을 명시적으로 정의하고 `(2,n,n,n,n)` 텐서로 변환·변형하는 최소 생성기.

최소 인수 조건:
- 정적 matrix game을 4차 action tensor로 올리는 변환기
- RPS 또는 zero-sum 게임 예제 1개
- Coordination/mixed-motive 예제 1개
- 직전 joint action에 의존하는 memory-one 예제 1개
- 변형 규칙의 seed 재현성 및 보상값 테스트

**주의:** 실제 묵찌빠의 지속 공격권은 extra state가 필요할 수 있으므로 Toy B의 첫 구현에서 완전 구현을 강제하지 않는다.

## 통합

Toy B가 생성한 텐서를 Toy A에 전달하고, Random Agent 두 개로 1개 episode를 실행하여 state/reward를 검증한다. 일치 검증이 완료되면 별도로 학습 policy를 연결한다.
