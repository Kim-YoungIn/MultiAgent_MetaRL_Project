# 연구 현황 · 2026-11-27

## 현재 단계

공동 투자가 상태를 바꾸는 게임에서 메타학습된 탐색 정책의 전이 실험을 완료했다. 이전 D-031도 봉인된 예산 전체의 학습·평가·독립 검증을 마쳤다. 새 실행 대기열은 비어 있다.

## 검증된 결과

- **규칙 전이 실험** — 기존 v2 훈련 블록 36개, 새 기본 게임 2,304개, 게임당 6개 변형 × 2개 리셋. K=8 이후 평가에서 성공 확률을 .6 낮췄을 때 전체−차단 피드백 이득의 변화는 턴당 −.000118 [−.001682, +.001424]였다. [프로토콜](research/dynamics_transfer/PROTOCOL.md) · [보고서와 재현 명령](research/dynamics_transfer/REPORT.md)
- 원래 시간 조건에서 96턴 Teacher−Student 누적 이득은 같은 조작으로 −.399 [−.625, −.166] 변했다. 호혜 게임의 감소가 가장 컸다. [계열별 수치](research/dynamics_transfer/evidence/summary.csv) · [시드별 그림](research/dynamics_transfer/evidence/primary_by_seed.png)
- 864개 전체 궤적을 스칼라 환경에서 재현하고 주 효과와 구간을 독립 재계산했다. 새 궤적·행동·체크섬은 로컬 `runs/dynamics_transfer_v1/`에 있다. [검증 영수증](research/dynamics_transfer/evidence/verification.json)
- **D-031** — 375/375회 학습, 175/175개 자체 행동 수 평가 셀, 50/50개 교차 평가 셀. 1,275개 영수증과 M/P 시험 곡선 563,200개를 확인했다. 25쌍 주 비교 M−P는 −.01274 [−.01608, −.00954]였다. [완료 보고서](research/d031_completion/README.md) · [원자료 독립 재집계](research/d031_completion/evidence/completion.json)
- 저장소 회귀 검사 248개 중 244개 통과, CUDA 의존 검사 4개를 건너뛰었다. 기존 24블록과 36블록 연구는 [Fast Exploration 색인](research/fast_exploration/README.md)에 있다.

## 재개 기준

[연구 목록](research/project_control/studies.json)에 15개 연구의 상태·프로토콜·근거·재현 경로가 있다. `python3 -m research.project_control --check`로 경로와 지정된 로컬 해시를 검사한다. 상세 원자료와 체크포인트는 로컬 `runs/`에 유지한다.

## 다음 판단

같은 전이 거리에서도 공동 투자 기회와 보상 회수율은 계열과 상대 행동에 따라 달랐다. 다음 과학적 판단은 **협력 자체의 수익성을 맞춘 상태에서, 새 정보를 얻는 행동의 가치만 달라지는 비교를 식별할 수 있는가**이다. 이번 연구는 성공 확률을 낮추는 한 방향만 시험했다. [행동을 고정한 진단](research/dynamics_transfer/evidence/exposure.csv)
