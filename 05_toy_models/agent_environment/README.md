# Toy A: Agent + Multi-Agent Environment

**상태:** 최소 구현 완료. 실제 코드는 `../mamrl/environment.py`, `policy.py`, `rollout.py`.

작업 범위는 `../../04_design/Work_Packages.md`와 `../../04_design/Interface_Contract.md`를 따른다. Rule Generator에서 전달받은 `(2,n,n,n,n)` tensor를 사용하고, 해당 모듈의 구체적인 게임 이름은 알 필요가 없다.

명시적 상태 Policy API, RandomPolicy, fixed-horizon truncation, episode 실행기와 transition 기록을 제공한다.
관측에는 길이·남은 턴·규칙 명세를 포함하지 않는다. 현재 정책 상태는 episode마다 초기화한다.
선택 의존성을 사용하는 `../mamrl/recurrent.py`, `ppo.py`, `training.py`는 고정 Random 상대와
유한 trial의 recurrent PPO를 제공한다. 같은 규칙의 episode 간 hidden 유지, trial 시작 초기화,
관측 전용 경계 이벤트와 full-trial replay를 지원한다. 정확한 계약은 `../../04_design/Recurrent_PPO_Contract.md`.
Self-Play/연구용 일반화 검증은 후속이다. 실행/테스트는 루트 README를 따른다.

새 학습 CLI는 `action_set_v1`을 사용한다. `../mamrl/action_set.py`는 행동별 공유 GRU/actor와
순열 대칭성을 제공하며, ordinal_v1은 참조로 보존한다. `benchmarks.py`의 작은 기능 과제와
Random/수작업/온라인Q/가족 지식 기준으로 학습과 기억을 분리해 평가한다.
