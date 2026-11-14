# Work Packages

현재 상태·다음 작업은 [Implementation Status](../00_management/Implementation_Status.md) 한 곳에서 관리한다.
이 문서는 구현 책임과 검증 경계를 정한다. 과거 완료 목록을 다시 누적하지 않는다.

| 작업 단위 | 범위 | 계약·인수 근거 |
|---|---|---|
| 환경 | 보상 텐서·동시 행동·관측·초기화·경계 | [Interface Contract](Interface_Contract.md) |
| 규칙 | 의미 명세·Compiler·catalog·중복·거리·audit | [Relational Game Contract](Relational_Game_Contract.md) |
| 메타 학습 | 탐색자·학생·학습 상대·PPO·paired 대조 | [Meta Exploration Contract](Meta_Exploration_Contract.md) |
| 결과 | 상세파일·집계·체크포인트·HTML·재현 | 위 계약의 artifacts/verification |

메인은 범위·통합·최종 검증을 책임진다. 분리 가능한 구현과 독립 계약 검증에만 subagent를 사용하고
같은 파일을 동시에 수정하지 않는다. 실행하지 않은 검증을 통과로 기록하지 않는다.
연구 해석을 바꾸는 선택은 Decisions/관련 계약에서 먼저 분리한다.

과거 최초 Toy A/B 인수 조건과 완료 순서는 [이전 작업 기록](archive/Work_Packages.md)에 보존한다.
기존 Recurrent/Generated/Diagnostics 계약은 해당 비교 실험 재현에만 적용한다.
