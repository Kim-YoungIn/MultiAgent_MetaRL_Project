# Implementation Status — 2026-11-14

## 현재 구현

현재 경로는 관계형 규칙 생성→학습하는 MA 환경→탐색 정책 학습→미학습 평가→HTML 분석까지 연결됐다.
연구 성공과 실행 경로 완성을 구분한다. 현행 가정은 [Baseline](Project_Baseline.md), 실행법은 [README](../README.md)다.

| 부분 | 구현 수준 | 근거 |
|---|---|---|
| 환경·정책 경계 | 동시 행동·Memory-One·네 관측·랜덤 초기화·비공개 horizon | [환경 계약](../04_design/Interface_Contract.md) |
| 규칙 | 의미 관계·상성·조건·보상·예외,28규칙과 중복/PD/최선반응 audit | [관계형 계약](../04_design/Relational_Game_Contract.md) |
| 학습 | 메타 탐색자→Q 학생·Q/Expected SARSA 상대·후속G→PPO | [메타 계약](../04_design/Meta_Exploration_Contract.md) |
| 평가 | 초기/학습 탐색과5대조군·동일 예산·가족/미학습 분할 | [최신 실험 검토](sessions/11_relational_rules_review.md) |
| 결과 확인 | trial 원자료·체크포인트·후속 곡선·prefix AUC·로컬 HTML | README의 메타 탐색 실행 |

## 최신 연구 결과

관계형 v2의3seed×40update 실행을 완료했다. 새 규칙에서 학습 탐색은 초기/랜덤보다 낮았고,
예외 구조에서는 안정된 우위를 확인하지 못했다. 새 상대 설정에서는 절대 성능이 상승했지만
학생 경험 유지G는 초기보다 소폭 낮았다. 더 유용한 경험을 학습했다는 결론은 미확정이다.
정확한 수치·조건·한계는 [결과 검토 한 곳](sessions/11_relational_rules_review.md)에 둔다.
로컬 원자료는 `runs/meta_relational_v2/`에 있으며 공유 사본 없이 동료가 독립 실행한다.

## 검증

리팩터링 후 전체143 tests,46.779s,OK; Torch 없는 Python143 tests,3.109s,OK(skipped=55).
독립 validator의9개 저장 계약 검증을 포함한다. v2 checkpoint24회는 전체 결과가 정확히 일치했고,
v1의24회도 기존 필드 값이 일치했다(v2에서 추가한 알고리즘/prefix/기울기 metadata는 별도).
정리 전 source와2update 학습의 이력·최종 가중치도 정확히 일치했다.
기존 실험의1496 trial 검증은 해당 실행 폴더,이번 증거는 `runs/refactor_verification_20261114/verification.json`에 있다.
브라우저 시각 검수는 미실행이다.

## 다음 작업

1. 새 튜닝 전, 학생이 얻은 경험과 탐색이 만든 상대 상태의 효과를 분리할 비교를 설계한다.
   초기/학습 상태의2×2 교차 평가는 후보이며 아직 실행하지 않았다.
2. 생성 seed·학생·상대·예산 확장을 사전 계약하고 노출량·유인 구조·정보 가치를 구분한다.
   작은 진단의 성공을 확장 연구의 필수 관문으로 두지 않는다.
3. 목적함수 변경·self-play·population·게임 내 규칙 전환은 별도 결정한다.

## 정리 (2026-11-14)

현재 문서에서 과거 지시·중복 수치를 제거하고,공통 규칙 metric과 리포트/아티팩트 저장 책임을 분리했다.
새 실행은 상세 평가를 한 번 저장하고 요약 집계에서 hash가 있는 참조로 읽는다. 기존 실행 파일은 보존한다.
같은 v2 자료의 집계 파일은185,120,204→1,260,467 bytes이며,상세 복원과 HTML 재생성이 일치했다.
기존 실행은 소급 압축/삭제하지 않았다. 보관 문서는 과거 근거로만 읽고 현재 문서를 다시 누적하지 않는다.

## 과거 근거

[전체 구현 이력](archive/Implementation_Status.md)은 정리 전 기록이며 현재 작업 지시가 아니다.
[첫 전이 실험](sessions/09_generated_transfer_review.md), [메타 v1](sessions/10_meta_exploration_review.md),
[메타 v2](sessions/11_relational_rules_review.md)를 개별 근거로 사용한다.
고정 상대·기존 생성기·학습 진단 CLI와 체크포인트는 비교/재현 경로로 유지한다.
