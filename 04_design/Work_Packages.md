# Toy Model Work Packages

## 현재 구현 순서 (2026-11-13)

1. **최초 구현 완료:** Memory-One 환경, 명시적 상태 Policy 인터페이스, RandomPolicy, episode 실행기와 transition 기록.
2. **최초 연결 완료:** 작은 논리 규칙 명세와 결정적 컴파일, 정적 게임 및 이력 의존 예제.
3. **검증:** n=3/4/5, 손계산 보상·관측·경계·재현성·정책 격리·오류 입력을 독립 테스트.
4. **우선순위 상향:** 실행 JSON 저장·검증과 오프라인 HTML 리포트. `Run_Report_Contract.md`를 따른다.
5. **첫 실행 완료:** recurrent 학습 정책·유한 trial PPO와 학습/평가 리포트. `Recurrent_PPO_Contract.md` 참조.
6. **첫 분리 진단 완료:** 고정/가변 target 다중 seed PPO 및 별도 지도학습 probe. `Learning_Diagnostics_Contract.md` 참조.
7. **분석 완료:** reward/gradient 감사와 cyclic 행동 번호 교환27run. `Action_Label_Audit_Contract.md` 참조.
8. **교정/파일럿 완료:** action_set 정책의 순열 대칭성·v1/v2 호환 및 3개 기능 벤치마크.
   `Nominal_Policy_Benchmark_Contract.md` 참조. 신경망 기억 학습 성공을 뜻하지 않는다.
9. **다음 M1: 통제된 생성 분포와 연구 계약.** 논리 문법·변형·대칭성·중복/퇴화·거리 후보·MA 상대군과
   학습/평가 manifest를 설계한다. 필요한 구별의 도달 가능성과 일부 하위집합의 정보 선택 가치를 확인한다.
10. **다음 M2: 첫 실제 전이 비교.** 공통 정책의 혼합 학습과 held-out 규칙/상대/조합 평가,
    초기 성적과 경험 후 개선, 제한된 예산·대조군을 사전 명세해 실행한다.

D-018에 따라 작은 기능 과제의 학습 성공을 M1의 필수 관문에서 해제했다.
구현 오류는 우선 수정하지만 무기한 최적화 탐색은 하지 않는다. 기존 진단/결과는 보존한다.
M1/M2의 구체적 문법·모델·상대·규모는 아직 미확정이며 코드/기본값은 이번 조정에서 바꾸지 않았다.
상세 인수 기준은 `../00_management/sessions/07_independent_research_review.md`를 따른다.

정확한 API와 경계는 `Interface_Contract.md`를 따른다. 현재 코드는 `05_toy_models/mamrl/`,
독립 검증은 `tests/`에 둔다. 작은 구현의 기본값은 연구 조건의 확정이 아니다.

## Toy A — Agent + Environment

목표: Rule Tensor를 주입하면 두 정책이 simultaneous action을 제출하고 각자 보상을 수신하는 최소 실행 환경. 이후 Recurrent PPO / Self-Play로 확장 가능한 분리된 인터페이스.

최소 인수 조건:
- `reset(seed, rule)` / `step(action_0, action_1)`에 준하는 환경 API
- 두 플레이어 관측과 개별 보상
- Episode 및 첫 턴 경계 처리
- Random Agent 두 개로 smoke test
- Tensor shape 오류 및 seed 오류를 명확히 검출
- 플레이어별 정책 상태를 별도로 보관하고 episode 시작에 초기화
- 두 행동을 같은 이전 step의 관측에서 결정한 뒤 환경에 동시 제출
- fixed horizon에서 truncated 반환, 정책에 길이/남은 턴/규칙 명세 비노출
- transition 및 총 보상 기록; 학습 가능한 정책의 입력/출력 연결 지점

**주의:** 지금은 학습 알고리즘의 선정이나 PPO 하이퍼파라미터 확정을 요구하지 않는다.

## Toy B — Rule + Variation + Tensor Conversion

목표: 반복 게임의 보상 규칙을 명시적으로 정의하고 `(2,n,n,n,n)` 텐서로 변환·변형하는 최소 생성기.

최소 인수 조건:
- 정적 matrix game을 4차 action tensor로 올리는 변환기
- RPS 또는 zero-sum 게임 예제 1개
- Coordination/mixed-motive 예제 1개
- 직전 joint action에 의존하는 memory-one 예제 1개
- 변형 규칙의 seed 재현성 및 보상값 테스트
- 불변 RuleSpec을 반환하는 Generator와 결정적 Compiler 분리
- 논리 primitive의 작은 가산 합성 및 파라미터/조건 선택; 범용 DSL은 요구하지 않음
- 정적 zero-sum은 cyclic_zero_sum, coordination은 coordination, Memory-One은 repeat_own/copy_opponent로 검증

**주의:** 실제 묵찌빠의 지속 공격권은 extra state가 필요할 수 있으므로 Toy B의 첫 구현에서 완전 구현을 강제하지 않는다.

## 통합

Toy B가 생성한 텐서를 Toy A에 전달하고, Random Agent 두 개로 1개 episode를 실행하여 state/reward를 검증한다. 일치 검증이 완료되면 별도로 학습 policy를 연결한다.

정적 게임 하나의 통합 성공만으로 Toy B 전체 최소 조건을 충족했다고 간주하지 않는다.
규칙 식별 가능성·적응 필요성, Novel Rule, 장기 지속 상태, n 간 학습 일반화와 규칙 scheduler는 별도 연구 단계다.
