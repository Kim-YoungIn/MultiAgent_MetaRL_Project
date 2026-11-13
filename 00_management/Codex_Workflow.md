# Codex 작업 구성

루트 `AGENTS.md`는 공통 지침이며 `.codex/config.toml`은 최대 동시 Subagent 수를 지정한다. `.codex/agents/`에는 필요할 때 호출할 검증 역할만 정의한다. 모델은 프로젝트에서 고정하지 않고 실행 세션 설정을 따른다.

## 책임과 위임

- **Coordinator (메인 스레드)**: 관련 명세 파악, 작업 범위 결정, 구현/검증 결과 통합, 최종 테스트 및 결정 사항 정리. 작은 작업은 직접 수행한다.
- **worker (Codex 기본 역할)**: 범위가 정해진 구현을 담당한다. 구현을 무조건 위임하지 않는다.
- **validator (사용자 정의)**: 계약에서 독립적으로 예상 결과를 도출해 테스트하거나, 실패 사례를 찾아낸다. 생산 코드 수정은 맡지 않는다.
- **research_critic (사용자 정의)**: 연구 가설, 실험 조건, 결과 해석을 검토한다. 일상적인 코드 수정에는 필요하지 않다.

구현과 검증을 병렬로 할 때 두 역할은 동일한 계약을 출발점으로 삼는다. 작업 파일이 겹치면 한쪽은 읽기/검증만 하거나 순차적으로 수행한다. Coordinator가 검증 결과를 검토하고 수정 및 재검증을 마무리한다. Subagent의 동의 자체를 검증 증거로 간주하지 않는다.

## 시작과 재개

프로젝트 루트에서 `codex`를 실행한다. Codex가 이 프로젝트를 신뢰한 경우 프로젝트 설정을 읽는다. 필요하면 `/subagents`로 실행 중인 작업을 확인한다.

- 새 프로젝트 구현: `00_management/sessions/06_first_milestone.md`
- 상태 확인 및 이어서 작업: `sessions/01_project_management.md`
- 기능 변경: `sessions/02_implementation.md`
- 실험: `sessions/03_experiment.md`
- 검수: `sessions/04_review.md`
- 외부 ChatGPT 아이디어 반영 검토: `sessions/05_external_idea_intake.md`

세션 문서는 해당 요청에서 선택해 사용한다. 모두 매번 읽을 필요는 없다.
