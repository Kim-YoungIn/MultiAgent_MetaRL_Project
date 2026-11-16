# Decision Log

이 문서에는 **채택되거나 명시적으로 보류·기각한 결정만** 기록한다. 과거 대화에서 나온 제안을 자동으로 채택하지 않는다.

| ID | 상태 | 결정/범위 | 근거/비고 |
|---|---|---|---|
| D-001 | 현재 작업 기준 | 연구 초점은 Multi-Agent Meta-RL의 Exploration 및 Adaptation | 사용자 정리 및 제안서 |
| D-002 | 현재 작업 기준 | 2개 Toy Model을 분리해 구현·통합 | 사용자 요청 |
| D-003 | 현재 작업 가정 | 3개 이산 행동, Memory-One `n^4` 보상 표현 | 확장 가능한 출발점이며 보편적 게임 표현은 아님 |
| D-004 | 채택 (2026-11-13) | 현재 구현은 Memory-One, 초기 n=3, 환경·규칙·RandomPolicy는 가변 n | 사용자 생각 노트와 설계 위임. 외부 개정안의 n=4 기본값은 채택하지 않음 |
| D-005 | 채택 (2026-11-13) | 랜덤 초기 state; 독립 균등 초기 행동, 초기 보상 0, 시작 flag 사용 | 랜덤 초기화는 사용자 결정. 분포·보상·flag는 위임된 초기 구현 설계 |
| D-006 | 채택 (2026-11-13) | 고정 길이 episode 절단, 길이/남은 턴 비노출 | 사용자 결정. 종료는 truncated=True; 실제 학습 길이는 미정 |
| D-007 | 채택 (2026-11-13) | 논리 RuleSpec → 결정적 Compiler → 고정 RewardTensor | 사용자 원칙과 설계 위임. 작은 가산 primitive 합성으로 시작; 범용 DSL/배율/조건부 대체는 후속 |
| D-008 | 채택 (2026-11-13) | 에이전트 인터페이스·환경·실행기 우선, 명시적 플레이어별 상태와 transition 기록 | RandomPolicy로 우선 검증. episode마다 정책 상태 초기화; trial/RL²/PPO는 후속 |
| D-009 | 보류 (2026-11-13) | 최소 체류 길이 이후 확률적 규칙 변경 | 사용자 아이디어이며 최초 구현에는 적용하지 않음. episode 종료와 분리해 검토 |
| D-010 | 채택 (2026-11-13) | 다음 우선 작업은 단일 실행 JSON 저장과 오프라인 HTML 리포트 | 결과 확인 기능 논의 후 사용자 작업 순위 조정·계속 요청. 연구자용 시각화만 추가, 정책 관측·학습/평가 조건은 유지 |
| D-011 | 현재 실행 기준 (2026-11-13) | Recurrent PPO의 첫 baseline은 P0 learner vs P1 고정 RandomPolicy | 설계 위임과 계속 진행 요청. Self-Play 확정 아님; 초기/Random/학습/hidden-reset을 같은 외생 draws로 평가 |
| D-012 | 현재 실행 기준 (2026-11-13) | 유한 trial 동안 hidden 유지·trial 시작 초기화, 중간 episode return 유지, 최종 bootstrap 0 | 기본 2 episodes×12 steps. 마지막 보상은 observe-only 이벤트로 전달; 전체 trial PPO 재생. 단일 episode runner 계약은 유지 |
| D-013 | 현재 실행 기준 (2026-11-13) | action_bonus diagnostic + shared candidate scorer로 가변 n 실행 검증 | 실제 학습 경로의 공학 검증. 상대 전략 적응/미지 규칙/행동 공간 일반화 검증과 구별. 평가 중 가중치 갱신 없음 |
| D-014 | 진단 실행 기준 (2026-11-13) | 고정 target 학습·지도학습 조건부 출력·가변 target 40/120update를 분리 진단 | 사용자 계속 진행 요청. 모델/PPO/GPU 기본값 유지, learning seeds 7/17/27, target층화 paired 평가. 결과에 따른 원인 후보만 좁히며 모델 수정은 별도 채택 |
| D-015 | 분석 실행 기준 (2026-11-13) | 보상/수치 감사와 cyclic 물리→내부 행동 번호 교환의 완전 교차 대조 | 사용자 분석·점검 요청. seed3×physical target3×mapping3, 각40updates. 모델 기본 구조 변경은 채택하지 않음. 분석 checkpoint는 internal 좌표와 permutation을 함께 복원 |
| D-016 | 현재 구현 기준 (2026-11-13) | 비서열적 action_set_v1과 매 행동 hidden0 비교군 추가, 학습 CLI 기본 선택 | 사용자 계획·작업 속행 요청. 공유 per-action GRU/actor로 label-equivariance 검증. nH 기억 용량 변화 명시. ordinal_v1/v1 checkpoint 보존; 라이브러리 TrainConfig 기본은 과거 재현용 ordinal 유지 |
| D-017 | 파일럿 기준 (2026-11-13) | 숨은 행동·probe 관계 기억·상대 copy/avoid 반응의 3개 기능 벤치마크 | Memory-One 고정 텐서 및 4필드 관측 유지. probe payoff +1/-3/-.25 또는-.5, 상대별 상한1/1/(n-1). informed prior/Q 온라인 갱신을 구분. H1/H2 검증은 아님 |
| D-018 | 연구 운영 조정 (2026-11-13) | 다음 중심 작업을 통제된 논리 게임 분포·MA 상대군·규칙 차이·held-out 평가 설계로 전환 | 사용자 독립 검토·연구 분석·조정 요청. 작은 진단의 학습 성공을 Generator 연구의 필수 관문에서 해제. 최초 Toy A/B 인수 조건 완료와 H1/H2 미검증을 구분. 새 문법/모델/상대/예산은 후속 계약에서 확정. `sessions/07_independent_research_review.md` 참조 |
| D-019 | 채택 (2026-11-14) | bounded Predicate/Payoff 문법과 공통 F의 대칭 Compiler 추가 | 사용자 프로젝트 진행 요청과 설계 위임. eq/ne/AND/OR/NOT, if/add/scale, 최대 깊이6·노드63. 기존 가산 CLI 보존. `Generated_Transfer_Contract.md` 참조 |
| D-020 | 첫 전이 실행 기준 (2026-11-14) | 생성 규칙24개·학습 규칙4/12×상대1/3의 공통 정책 비교와 공통 held-out 평가 | 총예산 일치, training-only validation, 최종 checkpoint 고정. 정규화가 실제 task 정의. 상대 메모리 episode reset/learner trial reset. seed3의 기술적 파일럿이며 H1/H2 확증 아님 |
| D-021 | 연구 목적 명료화 (2026-11-14) | 핵심은 강화학습된 탐색 전략이 같은 학습기·초기 조건·예산에서 더 유용한 경험을 만들고 적응을 빠르게 하는가 | 사용자 직접 명료화. self-play/단일 RL²/PPO는 필수 아님. 기존 전체 trial 보상 플레이 정책과 탐색의 downstream 효과를 구분. dual 신경망/특정 inner learner/새 관측은 아직 미채택 |
| D-022 | 연구 조건 명료화 (2026-11-14) | 본 실험에는 상호작용 경험·자기 보상으로 전략을 학습/갱신하는 상대를 포함 | 사용자 직접 지적. 고정 scripted 상대는 기능 검사/대조군으로 제한. task 간 상대 교체·단순 반응·무작위 전환만으로 대체하지 않음. 구체 상대 학습법/속도/reset은 후속 통합 계약에서 확정 |
| D-023 | 첫 본 목적 파일럿 (2026-11-14) | 메타 탐색자→온라인 Q 적응 학습기→온라인 Q 상대의 downstream 탐색 학습 경로 구현 | 사용자 프로젝트 진행/조정 위임. 탐색12·후속12, 같은 상대 상태에서 학생 Q만 지운 대조12. meta reward=후속−대조, 탐색 즉시보상 미포함. 학생/상대 trial reset·episode 간 유지. 상대 업데이트와 실제 비균등 행동 사용을 기록. `Meta_Exploration_Contract.md` 참조. G와 절대성능 모두 보고하며 이득 보장 아님 |
| D-024 | 채택·실행 (2026-11-14) | 의미 관계/상성표/조건/보상/예외를 분리한 관계형 생성기와28규칙 v2 catalog, Q+Expected SARSA 학습 상대 | 사용자 독립 탐구·조정 후 진행 요청과 설계 위임. 기존 관측·대칭F·Memory-One·n3·학생Q·메타G·K/H 유지. 반복C/변경D의 최종 compiled PD 판정,5개 사전 사회적 유인층,balanced/change 대조군. v1 보존; 두 축 동시 변경이라 단일 요인 효과 아님. `../04_design/Relational_Game_Contract.md` 참조 |
| D-025 | 확대 계획 채택 (2026-11-16; 미구현) | 비판 보고서를 바탕으로 한정 진단과 실행 기반을 병행하고 다중catalog·대규모학습의 Rule Diversity 본실험으로 이동 | 사용자 계획조정/본격규모 요청. D-021/022 목적·학습상대 유지,작은 성공 필수관문 미채택. 목표8/32/128×3catalog×5seeds,실행 전 개발/처리량 검증과 학생/모델/목표/평가 봉인. 상세는 Relational_Game_Contract의2026-11-16절; v2변경/실험완료 아님 |
| D-026 | 구현 (2026-11-16) | 메타 탐색 CPU/CUDA 장치 선택·독립 trial 배치·update 경계 재개·Colab 실행 경로 | 사용자 GPU/Colab 구현 요청. 환경/Q는CPU,신경망추론/PPO 배치가속. 연구 관측/보상/G 유지. GPU 실측은 장치에서 별도 검증; 상세 Meta_Exploration_Contract/README |
| D-027 | 구현·실GPU 검증 (2026-11-16) | 실행범위별 runtime 진단,CUDA 수집의 배치 전송,탐색K/후속H 분리 | 사용자 확인/구현 속행 요청. 호스트 RTX5060Laptop·기존torch2.15+cu132 runtime13.2 확인,별도설치없음. H생략시기존K유지,관측/G불변. 실제비용K+H/제어포함K+2H; 확대H1실행완료 아님 |
| D-028 | 채택·구현 (2026-11-16) | 공통 probe Template·정확 정보가치 대조·파라미터/요소 재조합 분할·동일History/학생 표현 비교 | 사용자 계획/구현 메모와 후속 정정. 학습상대 유지; 상대변화적응의 독립조작만후속. θ의 재명명동등성 marker로교정,역할교환다양성미계산. 세학생은Q표현비교. 핵심 관측/G불변;32/16/16·3seed200update개발실행,최종H1/H2아님. Probe_Template_Contract 참조 |

※ `현재 작업 가정`은 변경 가능한 baseline이고, 연구 가설이 검증되었다는 의미는 아니다.
