# D-031 정규 예산 결과와 재현

기존 결과는 `runs/d031_v2/`에 있다. 완료된 학습 3회와 열린 평가 패널을 복제한 재개 위치는 `runs/d031_completion_v2/study/`이다. 재개 코드는 원래 실행의 소스별 SHA-256을 확인한 뒤, 현재 소스 또는 보관된 과거 소스에서 모든 모듈을 복원한다. 과거 `meta_exploration_rollout.py`는 봉인 당시 보관본을 사용한다. Python 3.14.4, PyTorch 2.15.0+cu132의 CPU 실행, 최초 평가 패널, 800×32 훈련 예산은 봉인과 일치한다. 검증된 체크포인트는 그대로 사용하고 진행 중 학습은 마지막 갱신 지점부터 복원한다.

원래 실행기에서 하나의 작업 프로세스가 하나의 카탈로그·반복 조합을 담당한다. 주 실험 M/P/O의 25쌍을 먼저 평가하고 보고서를 만든다. 이후 규칙 수, 상대 유형, 생성 계열, 행동 수와 교차 행동 수 평가를 이어간다. 정규 계획은 총 375회 학습이다. 단계 완료 뒤 `study/summary.md`와 `study/report.json`을 갱신한다. 원본 `runs/d031_v2/`에는 복제 및 소스 확인을 위한 읽기만 수행한다.

```bash
# 봉인된 PyTorch wheel과 라이브러리를 로컬 캐시에서 복원한 환경이다.
LD_LIBRARY_PATH="$PWD/runs/archive/runtime_d031_cu132/nvidia/nvshmem/lib" \
PYTHONPATH="$PWD/runs/archive/runtime_d031_cu132" \
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -u research/d031_completion/run.py \
  runs/d031_v2 runs/d031_completion_v2 --workers 5
.venv/bin/python -m research.d031_completion.status runs/d031_completion_v2/study
```

봉인된 실행 환경은 Python 3.14.4와 PyTorch 2.15.0+cu132이다. 위 로컬 런타임에는 `nvidia-nvshmem-cu13==3.7.2`도 포함된다. 다른 머신에서는 같은 Python과 CUDA 13.2용 PyTorch 버전을 별도 환경에 설치해야 한다. `run.py`는 실제 버전과 봉인된 소스 해시가 다르면 재개를 거부한다. 학습과 평가는 CPU로 실행한다.

생성된 원자료로 유사도를 갱신하려면 새 분석 출력 경로를 지정한다.

```bash
PYTHONPATH=05_toy_models:. OPENBLAS_NUM_THREADS=1 \
  .venv/bin/python -m research.rule_similarity.d031 \
  runs/d031_completion_v2/study runs/rule_similarity_d031_continuation
```

`run.log`에는 진행 중인 작업의 카탈로그·반복 번호가 기록된다. 완료 집계에는 원자적으로 발행된 학습 원장과 평가 영수증만 포함된다. 학생·상대의 설정과 기존 보고 규칙은 [D-031 프로토콜](../../04_design/Adaptation_Experiment.md)에 있다.

## 25쌍 주 비교 결과

미학습 규칙 64개를 두 학습 상대와 8개 반복에서 짝지었다. 다섯 카탈로그마다 학습 정책을 다섯 번 적합했다. K=0을 기준으로 한 적응 면적에서 M은 0.10431, P는 0.11705였다. **M−P는 −0.01274 [−0.01608, −0.00954]**였으며 다섯 카탈로그의 평균 차이가 모두 음수였다. 50개 M/P 평가 영수증의 파일 해시를 확인하고 시행별 곡선 51,200개에서 적응 면적과 카탈로그·반복 수준의 부트스트랩 구간을 다시 계산했다. `study/report.json`의 주 결과와 일치한다. [25쌍 유사도 분석](../rule_similarity/REPORT.md#주-실험-25쌍으로-확장)

## 정규 예산 전체 결과

375회 학습과 175개 훈련 셀의 자체 행동 수 평가, 50개 셀의 양방향 교차 행동 수 평가를 마쳤다. 마지막 다섯 모델은 기존 봉인과 동일한 런타임에서 이어서 학습했고, 중단된 평가 파일은 마지막으로 발행된 영수증부터 재개했다. 새 평가 패널은 열지 않았다.

M은 탐색 후 학생 Q를 지운 분기 대비 이익을, P는 탐색 단계 보상을 학습 목표로 사용한다. 아래 수치는 두 방법의 **탐색 0턴 기준 적응 면적 차이(M−P)**가 훈련 조건에 따라 얼마나 바뀌었는지 보여준다. 카탈로그 다섯 개와 카탈로그별 반복 다섯 개를 짝지어 계층 재표집한 95% 구간이다.

| 훈련 조건의 변경 | M−P의 변화 | 95% 구간 |
|---|---:|---:|
| 훈련 규칙 8개 → 128개 | −.01017 | [−.01996, −.00243] |
| 8개 → 32개 | −.00676 | [−.01696, +.00293] |
| 32개 → 128개 | −.00342 | [−.00843, +.00183] |
| 집중 계열 → 넓은 계열 | −.00200 | [−.00591, +.00188] |
| 혼합 상대 − 두 단일 상대 조건의 평균 | −.00192 | [−.00645, +.00340] |

8개와 128개에서 M의 평균 적응 면적은 각각 .10439, .10292였고 P는 .11037, .11908이었다. 총 학습 예산은 같아서 규칙을 늘리면 규칙당 노출이 줄어든다. 이 결과는 이 특정 예산에서 M과 P의 상대 성능이 어떻게 바뀌었는지를 보여준다. 일반적인 훈련 다양성의 적응 효과는 별도 검증이 필요하다. [다양성 및 조건별 그림](evidence/diversity.png)

3행동에서 훈련한 25쌍의 4·5행동 평가에서 M−P는 각각 −.01995, −.01841이었다. 5행동 훈련의 3·4·5행동 평가에서도 각각 −.01022, −.01839, −.01728이었다. 아래의 초기 한 쌍 결과를 포함한 전체 재평가다.

독립 검사에서는 봉인 소스·평가 패널, 학습 원장과 체크포인트, 1,275개 평가 영수증을 확인했다. 275개 셀·행동 수 조합의 M/P 시험 곡선 563,200개에서 적응 면적을 다시 적분해 11개 그룹과 원래 보고서의 5개 조건 대비를 대조했다. 이 집계는 fresh 규칙·두 훈련 상대에 해당한다. [검증 기록](evidence/completion.json) · [카탈로그·반복별 결과](evidence/cells.csv)

```bash
OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.d031_completion.verify_completion \
  runs/d031_completion_v2/study --output research/d031_completion/evidence
```

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
