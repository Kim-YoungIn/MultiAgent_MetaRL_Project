# Idea Inbox — 후보와 검토 상태

이 문서는 아이디어 색인이다. CANDIDATE/REVIEWED/DEFERRED는 결정이나 구현 요구사항이 아니다.
채택·구현된 항목도 추적을 위해 남기며 결정 ID/계약과 구분한다. 검토 전에 독립 대안도 비교한다.
현재 대화 전체의 축어록은 아니며 주요 아이디어·관찰·판단 변경을 요약해 기록한다.
탐구 경과는 [Research Log](../00_management/Research_Log.md), 생성기 상세 후보는
[Rule Generator 연구 노트](Rule_Generator_Research_Notes.md)를 따른다.

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
| I-010 | START state / 초기 무작위 joint action | 첫 턴 보상 정의 | 전용 START 상태 대신 랜덤 이전 joint action·초기보상0·start flag 사용 | ADOPTED — D-005; 전용 START 상태는 미채택 |
| I-011 | 모델·실험 규모 확대 시 GPU 배치 실행 | 여러 trial의 모델 계산을 묶어 학습 시간 단축 | 먼저 환경/추론/PPO 시간을 측정하고 동일 조건 CPU/GPU 비교; 현재 설정 변경 없음 | DEFERRED — 사용자 요청으로 후속 아이디어만 보존 |
| I-012 | 논리 조건·조합을 기록하는 Rule Generator | parameter/composition/novel-relation을 분리 통제 | 최소 문법과 전략적 차이; 범용 DSL 자체가 목표가 아님 | IMPLEMENTED SUBSET — D-019; 제한된 Boolean/if/add/scale |
| I-013 | 규칙 구조·텐서·전략 차이 측정 | H1/H2의 유사성/다양성 분석 | 배율·재명명·중복·상대/horizon 의존, 사후 거리 정의의 순환 논리 | IMPLEMENTED SUBSET — 원시/정규화/재명명 텐서·구조 token 차이; 전략 거리 미구현 |
| I-014 | 공유 행동별 recurrent scorer | 숫자형 인코딩 편향 교정과 가변 n 실행 | 후보 간 정보 교환·nH 상태 변화; 범용성/학습 일반화는 별도 | IMPLEMENTED — D-016; 진단 결과 보존 |
| I-015 | 통제된 이력 반응형 상대군 | self-play 없이 상대 적응·행동 영향 분석 | 상대의 유인·관측·episode/trial 메모리 경계 | IMPLEMENTED SUBSET — Random/copy/avoid와 reward-reactive win_stay, D-020 |
| I-016 | n=3 one-hot + 단일 GRU 참조 모델 | 현재 구조에 종속되지 않는 학습 비교 | 라벨 재명명 통제·예산·모델 용량 | REVIEWED — 미구현 |
| I-017 | LLM을 exploration 보조로 활용 | 논리 사전지식의 정보 수집 활용 | 비용·정보 공개·사전학습 prior 차이; 동료 메모 제안 | CANDIDATE — 우선 구현 아님 |
| I-018 | 탐색/활용 보상 혼합 및 off-policy 시도 | 동료 보고의 정보 활용·학습 신호 병목 검토 | 별도 구현의 원자료 없음; 목적함수 변경은 새 결정 필요 | EXTERNAL REPORT — 우리 코드 적용/재현 없음 |
| I-019 | 학습/평가 manifest와 미학습 조합 | 암기·파라미터 전이·구조 전이 구분 | seed만 분리하지 않고 규칙·동등 게임 누수 점검 | IMPLEMENTED — D-020, 조건별 seen flags 포함 |
| I-020 | 연결된 전체 실험판 우선 완성 | 생성→MA 구성→혼합 학습→held-out 평가→분석 연결 | 필수 범위와 선택 확장 구분, 단계가 영구 장벽이 되지 않게 함 | IMPLEMENTED FIRST PILOT — D-019/D-020; 연구 성공을 뜻하지 않음 |
| I-021 | H1 실패/부정적 전이 가능성 | 성공을 전제하지 않는 연구 | 학습 바닥 성능·노출량·유사성 교란과 구분 | REVIEWED — H1/H2 직접 검증 전 |
| I-022 | 완성된 보상 규칙 목록 대신 관계·조건·보상 연산의 문법 | 생성 가능한 게임 구조 자체를 통제하고 조합 전이 분석 | predicate/연산/깊이 제한, 합성 의미와 공통 구조; 사용자 2026-11-14 생각노트 검토 | IMPLEMENTED SUBSET — 후속 프로젝트 진행 위임으로 D-019 채택 |
| I-023 | 부모/메타 과정의 탐색 전략 생성·선택·학습 | 수집 경험이 새 에이전트의 적응에 주는 효과 | population 선택과 RL 최적화, 게임 안 MA와 게임 밖 meta 과정 구분 | REVIEWED — 목적 예시이며 literal 구조 미채택 |
| I-024 | Teacher exploration / student improvement meta reward | 탐색 경험의 downstream 학습 가치 직접 최적화 | Xu2018의 단일-agent 연속제어를 MA·새 task 전이와 구분; 상대 학습에 의한 점수 변화 | IMPLEMENTED FIRST PROTOCOL — D-023; 동일 학생/상대 상태의 retained−erased G, 원 논문 재현은 아님 |
| I-025 | 행동 연결의 의미 관계와 상성·조건·보상·예외 분리 | 승패/반복/모방/예측과 공동 보너스·숨은 행동·예외의 조합 전이 | 현재/이전 결과 구분, 예외 충돌·대칭·상성 의미, 실제 sampler 제약 | IMPLEMENTED SUBSET — D-024; 상성/조건/보상/예외와28규칙 v2 |

| I-026 | 반복=협력/변경=배신의 사회적 게임 표현 | 의미 관계와 보상 순서에 따른 게임 분류 | PD는 최종 보상 T>R>P>S 및 모든 행동 변형의 동등성 필요; n3의 C/D 표본수 차이 | IMPLEMENTED SUBSET — D-024; PD/조정/반조정/협력 우세, balanced/change 대조 |

아이디어를 채택한 후에만 `00_management/Decisions.md` 및 실제 설계 문서를 수정한다.
