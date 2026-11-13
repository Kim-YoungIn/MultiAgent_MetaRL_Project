# Toy B: Rules, Variations and Tensor Conversion

**상태:** 최소 구현 완료. 실제 코드는 `../mamrl/rules.py`, `tensor.py`.

작업 범위는 `../../04_design/Work_Packages.md`와 `../../04_design/Interface_Contract.md`를 따른다. 게임의 보상 규칙을 텐서로 변환하고 변형을 생성하되, Agent/Policy 학습 코드는 포함하지 않는다.

불변 RuleSpec 생성 → 결정적 Compiler → 불변 RewardTensor로 분리한다.
coordination, repeat_own, copy_opponent, action_bonus, cyclic_zero_sum의 가산 합성을 지원한다.
생성 시 조건·계수·대상 행동을 선택하며 게임 진행 중 텐서는 고정된다.
기본 n=3, 가변 n>=2. 계수 0 또는 합성으로 퇴화 게임이 나올 수 있으며 전략적 다양성은 보장하지 않는다.
범용 DSL, 배율/조건부 대체, Novel Rule 및 연구용 난이도 통제는 후속이다.

기능 벤치마크용 명시적 probe_repeat/probe_copy template를 추가했다. 기존 mixed 기본 sampling은
유지한다. 숨은 mode 식별과 이력의 유용성 witness는 `benchmarks.py`에서 검증한다.
일반 Generator 확장은 이 작은 과제들의 학습 검증 이후 진행한다.
