# Project Baseline

## 연구 목적 (FIXED)

Multi-Agent 반복 게임에서 **강화학습한 탐색 전략이 새로운 규칙·상대에 대한 에이전트의 적응을 더 빠르게 만드는가**를 연구한다(D-021/D-022).
같은 적응 학습기·초기 조건·상호작용 예산에서 수집 경험의 유용성과 후속 성능을 비교한다.
상대도 자기 경험으로 전략을 학습해야 한다. 고정 반응식은 기능 검사·대조군이다.
H1/H2의 다양성·재사용 구조는 이 효과를 분석하는 축이다. self-play·RL²·부모/생존 선택 구조는 필수가 아니다.

## 현재 실행 기준 (CURRENT)

- 2인 동시 행동, 기본 n=3, Memory-One. 환경·규칙의 n4/5 실행은 검증했으나 학습된 행동 공간 일반화는 미검증이다.
- 공통 자기 관점 함수 F를 양쪽에 적용하는 대칭 게임. 공통 보상 항목은 n⁴, 저장은 `(2,n,n,n,n)`이다.
  관측 가능한 직전 joint action은 n²개이며 보상 항목 수와 구별한다.
- 초기 joint action은 독립·균등 무작위, 초기 자기 보상0, 시작 flag=True.
- 관측은 직전 자기/상대 행동·직전 자기 보상·episode 시작 표시다. 규칙/상대 ID와 종료 길이는 숨긴다.
- episode는 고정 길이에서 truncated 처리한다. 경험으로 길이를 추정하는 것은 금지하지 않는다.
- 논리 명세 생성→결정적 Compiler→고정 보상 텐서. 임의 tensor cell을 직접 샘플링하지 않는다.
- 현재 경로는 `mamrl.meta_exploration --catalog relational --opponent-suite mixed_learning`이다(D-024).
  인자 생략 시 과거 logical/Q 기본값을 유지한다. 실행법은 [README](../README.md)를 따른다.
- 관계형28규칙은12학습/8새규칙/8예외구조. 상성·관계·조건·보상·예외를 분리하고 현재 행동 일치를 강제하지 않는다.
  원식/정규화/seed/중복·재명명/PD·최선반응 audit를 기록한다. 이름·텐서 거리가 전략 다양성을 보장하지 않는다.
- action_set_v1 탐색자(H32)가12턴 경험을 수집하고 같은 온라인 Q 학생(alpha=.5,gamma=.95)이 학습한다.
  학생은 새 episode에서12턴 greedy로 플레이하며 계속 학습한다.
- 상대는 자기 보상으로 Q(.3,.15) 또는 Expected SARSA(.5,.2)를 학습한다(gamma=.95).
  heldout은 이미 훈련한 Expected SARSA의 새 설정(.7,.1)이다. 미사용 알고리즘 전이가 아니다.
- 학생/상대는 trial 시작에 초기화하고 episode 사이에는 유지한다. 탐색자 가중치는 inner trial/최종 평가에서 고정한다.
- 같은 상대 상태·난수·reset에서 학생 Q만 지운12턴 대조를 계산한다. 실제24턴+대조12턴=36 simulation steps.
- 메타G=후속 평균−경험 제거 평균으로 탐색자 PPO를 학습한다. 탐색 중 보상을 더하지 않는다.
  G는 조건부 경험 유지 효과이며 순수 정보량·절대 성능 개선이 아니다. 절대 성능과 함께 판단한다.
- 현재 파일럿은3seeds×40updates×8trials,H32,reps2. heldout으로 checkpoint 선택/튜닝하지 않는다.

## 문서의 역할

- [Implementation Status](Implementation_Status.md): 현재 구현·최신 결과 결론·검증·다음 작업의 진입점.
- [Decisions](Decisions.md): 채택/보류한 선택의 근거와 이력.
- [Open Questions](Open_Questions.md): 아직 결정하거나 검증할 질문만 관리.
- [환경 계약](../04_design/Interface_Contract.md), [메타 탐색 계약](../04_design/Meta_Exploration_Contract.md),
  [관계형 확장 계약](../04_design/Relational_Game_Contract.md): 구현과 재현의 상세 조건.
- [Research Log](Research_Log.md): 탐구 경과. [Idea Inbox](../03_ideas/Idea_Inbox.md): 아이디어 출처·상태 색인.

현재 완료 여부와 결과 수치는 이 문서에 누적하지 않는다. 규칙 변경 scheduler·비대칭 생성·긴 추가 상태·
다른 학생/예산·새 목적함수는 미채택이며 [미해결 질문](Open_Questions.md)을 따른다.
