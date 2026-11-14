# 현재 한계와 근거

현재 결론과 다음 작업은 [Implementation Status](Implementation_Status.md),미결 질문은
[Open Questions](Open_Questions.md)에서 관리한다. 이 문서는 한계의 근거를 찾는 색인이다.

| 해석상의 한계 | 근거/상세 |
|---|---|
| 새 규칙에서 탐색 우위 미확인,새 상대 절대 성능과G의 차이 | [관계형 v2 결과](sessions/11_relational_rules_review.md) |
| G는 조건부 retained-Q 효과이며 순수 정보량/절대 개선의 보장이 아님 | [메타 학습 계약](../04_design/Meta_Exploration_Contract.md) |
| 한 catalog seed·Q 학생·짧은 예산·제한된 sampler와3 learning seeds | [관계형 실행 조건](../04_design/Relational_Game_Contract.md) |
| 보상 축 의존성·이름·텐서 거리는 전략 다양성을 보장하지 않음 | 관계형 계약의 compiled audit와 거리 정의 |
| fresh/exception 분할은 계수만/깊이만 전이가 아님 | 관계형 v2 결과의 사전 분포 |
| v1/v2는 생성 분포와 상대 학습법이 함께 달라 단일 요인 효과 비교가 아님 | 관계형 v2 결과 |
| n4/5 실행 지원은 학습된 행동 공간 일반화가 아님 | Baseline과 환경/정책 계약 |

[정리 전 검토](archive/Limitations_Review.md)는 v1 당시의 진단을 보존한다.
그 안의 고정 sampler·Q-only 등의 제약을 현재 구현 부재로 해석하지 않는다.
