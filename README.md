# Multi-Agent Meta-RL — MAMRL

Multi-Agent 반복 게임에서 Meta-Policy 학습이 새로운 규칙과 상대 전략에 대한 Exploration 및 빠른 Adaptation에 미치는 영향을 연구하기 위한 저장소다. 현재는 연구 명세와 Toy Model 구현 준비 단계이며, 학습 코드나 실험 결과는 포함하지 않는다.

## 구조

- `AGENTS.md` — 로컬 Codex의 지속적인 작업 원칙
- `.codex/` — 선택적 Multi-Agent 설정과 검증 역할
- `00_management/` — 연구 기준선, 결정/미결정 사항, Codex 운영 및 세션별 지시문
- `01_originals/` — 제안서와 과제 지침 원본 (수정하지 않음)
- `02_references/` — 참고 논문 서지·링크·독서 메모
- `03_ideas/` — 검토 후보; 실행 명령이 아님
- `04_design/` — Toy Model 인터페이스 및 구현 범위
- `05_toy_models/` — 환경/에이전트와 규칙 생성기의 구현 작업 공간
- `FILE_MANIFEST.tsv` — 파일별 크기 및 SHA-256

## 시작

1. 루트에서 `codex`를 실행한다. 저장소가 신뢰된 프로젝트인지 확인한다.
2. 최초 구현은 `00_management/sessions/06_first_milestone.md`, 일반 작업은 `sessions/01`~`05` 중 알맞은 지시문을 사용한다.
3. 작업의 현재 기준은 `00_management/Project_Baseline.md` 및 관련 설계 문서다. `03_ideas/Idea_Inbox.md`의 후보는 별도 승인 없이 구현하지 않는다.
4. 구현과 검증을 분리해야 할 때 `.codex/agents/validator.toml`을 사용한다. 연구 타당성 검토가 필요할 때만 `research_critic`을 호출한다.

자세한 운영 구분은 `00_management/Codex_Workflow.md`를 따른다. 메인 에이전트가 작업 분해 및 통합의 책임을 갖는다.

참고 논문 PDF 바이너리는 제공되지 않으며 `02_references/`에는 원문 링크가 있다. 과제 평가 계획 PDF는 `01_originals/`에 포함된다.
