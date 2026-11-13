# R01 — Self-Play Meta-Reinforcement Learning in Multi-Agent Games

- **Authors:** Imre Gergely Mali
- **Year / Venue:** 2026 / Acta Universitatis Sapientiae, Informatica
- **Role:** CORE
- **Official page:** https://link.springer.com/article/10.1007/s44427-026-00021-y
- **Official PDF:** https://link.springer.com/content/pdf/10.1007/s44427-026-00021-y.pdf

## 핵심 내용

다양한 normal-form 게임에서 RL²식 메타학습을 Self-Play로 확장. Independent co-training 및 과거 정책 상대 구조를 비교하는 핵심 출발점.

## 우리 프로젝트에 연결되는 지점

Meta-RL 에이전트, multi-agent 학습 루프, 상대 정책 변화의 설계 참조.

## 적용 시 주의

논문은 기본적으로 현재 joint action의 payoff matrix 중심이다. 본 프로젝트의 직전 joint action 의존 텐서로 확장하는 부분은 자체 제안.

**지위:** 참고 자료이며 구현 결정이 아니다.
