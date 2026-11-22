# D-031 정규 예산 재개

기존 결과는 `runs/d031_v2/`에 있다. 완료된 학습 3회와 열린 평가 패널을 복제한 재개 위치는 `runs/d031_completion_v2/study/`이다. 재개 코드는 원래 실행의 소스별 SHA-256을 확인한 뒤, 현재 소스 또는 보관된 과거 소스에서 모든 모듈을 복원한다. 과거 `meta_exploration_rollout.py`는 봉인 당시 보관본을 사용한다. Python 3.14.4, PyTorch 2.15.0+cu132의 CPU 실행, 최초 평가 패널, 800×32 훈련 예산은 봉인과 일치한다. 검증된 체크포인트는 그대로 사용하고 진행 중 학습은 마지막 갱신 지점부터 복원한다.

원래 실행기에서 하나의 작업 프로세스가 하나의 카탈로그·반복 조합을 담당한다. 주 실험 M/P/O의 25쌍을 먼저 평가하고 보고서를 만든다. 이후 규칙 수, 상대 유형, 생성 계열, 행동 수와 교차 행동 수 평가를 이어간다. 정규 계획은 총 375회 학습이다. 단계 완료 뒤 `study/summary.md`와 `study/report.json`을 갱신한다. 원본 `runs/d031_v2/`에는 복제 및 소스 확인을 위한 읽기만 수행한다.

```bash
# 실행 중인 프로세스는 별도 작업 잠금을 보유한다. 중단 후 같은 경로로 재개한다.
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u research/d031_completion/run.py \
  runs/d031_v2 runs/d031_completion_v2 --workers 8
.venv/bin/python -m research.d031_completion.status runs/d031_completion_v2/study
```

생성된 원자료로 유사도를 갱신하려면 새 분석 출력 경로를 지정한다.

```bash
PYTHONPATH=05_toy_models:. OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m research.rule_similarity.d031 \
  runs/d031_completion_v2/study runs/rule_similarity_d031_continuation
```

`run.log`에는 진행 중인 작업의 카탈로그·반복 번호가 기록된다. 완료 집계에는 원자적으로 발행된 학습 원장과 평가 영수증만 포함된다. 학생·상대의 설정과 기존 보고 규칙은 [D-031 프로토콜](../../04_design/Adaptation_Experiment.md)에 있다.

## 25쌍 주 비교 결과

미학습 규칙 64개를 두 학습 상대와 8개 반복에서 짝지었다. 다섯 카탈로그마다 학습 정책을 다섯 번 적합했다. K=0을 기준으로 한 적응 면적에서 M은 0.10431, P는 0.11705였다. **M−P는 −0.01274 [−0.01608, −0.00954]**였으며 다섯 카탈로그의 평균 차이가 모두 음수였다. 50개 M/P 평가 영수증의 파일 해시를 확인하고 시행별 곡선 51,200개에서 적응 면적과 카탈로그·반복 수준의 부트스트랩 구간을 다시 계산했다. `study/report.json`의 주 결과와 일치한다. [25쌍 유사도 분석](../rule_similarity/REPORT.md#주-실험-25쌍으로-확장)

주 조건은 75개 학습과 다른 행동 수로 옮긴 25개 평가 조합까지 완료했다. 나머지 조건의 실행률은 위의 `status` 명령이 발행된 원장·영수증에서 읽는다. `study/summary.md`는 완료된 단계의 스냅샷이다.

## 첫 훈련 쌍의 행동 수 전이

기존 3행동 M/P 체크포인트를 그대로 사용한 4·5행동 평가를 완료했다. 각 행동 수에서 미학습 규칙과 원래 학습 상대의 1,024개 짝지은 시험을 사용했다. 모든 평가 파일 해시와 시행별 적응 면적을 원자료에서 다시 확인했다.

| 평가 행동 수 | M | P | M−P |
|---|---:|---:|---:|
| 3 | .10997 | .11257 | −.00260 |
| 4 | .07003 | .07878 | −.00875 |
| 5 | .06071 | .06760 | −.00688 |

이 표는 첫 카탈로그·반복의 기술 통계다. 재개 디렉터리의 `first_transfer_verification.json`에 원자료 집계가 있다.

## CUDA 개발 검증

장치 검증을 위해 일부만 실행했던 CUDA 개발 패널을 같은 소스 해시와 개발 설정으로 재개했다. 15개 학습과 51개 평가 파일을 모두 완료하고 검증했다. 결과는 `runs/d031_completion_cuda_dev/study/summary.md`에 있다. 이 경로에는 원래 개발용의 2×3 훈련 예산이 적용된다. CUDA 장치에 접근 가능한 Linux/WSL 프로세스에서 실행한다.

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u research/d031_completion/run.py \
  runs/d031_v2_cuda runs/d031_completion_cuda_dev --workers 1
```
