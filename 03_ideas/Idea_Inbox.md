# Idea Inbox — 미채택 후보 목록

이 문서의 항목들은 브레인스토밍에서 나온 **후보**로, 결정이나 구현 요구사항이 아니다. 검토 전에 독립 대안을 먼저 생성하고 비교한다.

| ID | 후보 | 기대하는 연구적 가치 | 먼저 확인할 점 | 상태 |
|---|---|---|---|---|
| I-001 | Memory-One `n^4` Reward Tensor | 정적 게임과 이전 행동 의존 게임을 공통 표현 | 실제 필요한 추가 상태, 보상 범위 및 테스트 | CURRENT 설계 가정 |
| I-002 | Structured vs Random Tensor | 공유 규칙의 전이 가능성(H2) | 난이도·Oracle 성능·정보량 통제 | CANDIDATE |
| I-003 | Rule Diversity × Opponent Diversity | 규칙 변화와 상대 변화의 효과 분리(H1) | Self-Play 상대 변화와 고정 정책 상대의 구분 | CANDIDATE |
| I-004 | RL² vs E-RL² | 암묵적 탐색과 명시적 탐색 목적 비교 | 탐색 비용, 전체 return, 평가 예산의 정의 | CANDIDATE |
| I-005 | Separate Exploration & Exploitation Policies | 정보 수집 전략과 활용 전략 분리 | 연구 목표 대비 구현 부담 | CANDIDATE |
| I-006 | On-policy PPO vs Off-policy Recurrent RL | 경험 재사용과 적응 학습의 효율 비교 | replay의 상대 정책 비정상성 및 RNN state 관리 | CANDIDATE |
| I-007 | Fixed Opponents vs Co-training vs Snapshot Pool | 상대 변화 속도가 적응에 미치는 영향 | 상대 다양성과 업데이트 방식의 교란 | CANDIDATE |
| I-008 | Within-task Rule Switch | 게임 중 규칙 변화에 대한 지속 적응 | Task 자체의 의미와 RNN reset 정책 재정의 필요 | CANDIDATE |
| I-009 | Information-Probe Games | 탐색 행동의 정보적 가치 및 비용 통제 | 관측 보상만으로 규칙 식별 가능성 보장 | CANDIDATE |
| I-010 | START state / 초기 무작위 joint action | 첫 턴 보상 정의 | 가상 이전 행동의 노출과 첫 턴 표시 | CANDIDATE |

아이디어를 채택한 후에만 `00_management/Decisions.md` 및 실제 설계 문서를 수정한다.
