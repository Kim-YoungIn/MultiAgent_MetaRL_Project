# Toy A: Agent + Multi-Agent Environment

**상태:** 구현 전.

작업 범위는 `../../04_design/Work_Packages.md`와 `../../04_design/Interface_Contract.md`를 따른다. Rule Generator에서 전달받은 `(2,n,n,n,n)` tensor를 사용하고, 해당 모듈의 구체적인 게임 이름은 알 필요가 없다.

권장 코드 분리: `environment`, `agent_api`, `smoke_test`. PPO/RL²는 이 기본 상호작용이 검증된 뒤 연결한다.
