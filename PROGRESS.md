# Research status, 2026-11-22

Current objective and stage
- Completed the requested scaled Fast Exploration study, preserving the Student prior surrogate and Teacher actor–critic.
- Results and evidence in research/fast_exploration_v1/RESULTS.md. Reproduction and comparison details in that directory's README.md, AUDIT.md and SCALE_PROTOCOL.md.

Verified outcomes
- 24 independent blocks (8 × 3 master seeds), 400 updates ×32 episodes per Teacher, 128 prior games and 128 test games ×2 resets per block. K=0/4/8/16/24, H=32.
- Learned / blind / random / greedy absolute AUC .03889/.03844/.03486/.03407. Matched Learned differences +.00045 [−.00034,+.00123] vs blind, +.00403 [.00182,.00649] vs random, +.00482 [.00134,.00841] vs greedy.
- Opponent restored and both learners frozen during assessment: Learned minus random +.00484 [.00256,.00733]. At K24 the greedy policy earned more total natural reward. Prior ablations with the same learned Teacher favored pooled and zero over the trained prior.
- Vectorized environment, Student and opponent and batched GRU. CPU baseline 27.45→3.27 s for a paired short fit, evaluation 13.44→1.01 s. CUDA short fits 5.34–5.74 s. Full first block 104.4 s, remaining 23 on eight workers in 554.4 s.
- 14 focused tests passed on GPU host. Full CPU-context suite ran 237, 233 passed and four skipped. Replay verified all raw-to-report outcomes and 768 scalar trajectories. Identical dev NPZ and weights across worker counts.
- Full raw artifacts and SHA-256 manifests retained under runs/fast_exploration_scale_final/. Compact evidence is versioned. Existing patched files and separate D-031 results remain available.

Next research decision
- Treat the descriptive absolute-AUC advantage and the lower total return against greedy as separate outcomes. A fresh protocol is required to compare the two training utilities in the same full-horizon design.
## Current stage

Fast Exploration v2, its 96-turn extension and the exploratory 96-turn UCB replay are complete. New experiment design and new runs are on hold by user request. The study index is research/fast_exploration/README.md.

## Verified evidence

- v2 trained 252 Teachers in 36 blocks across three master seeds. The fresh main panel had 4,608 games and two resets each. See research/fast_exploration/results/v2/README.md.
- Pooled full minus masked feedback controlled absolute AUC was +.00097 [.00037,.00159]; immediate-reward Teacher and UCB had higher AUC than the downstream Teacher.
- At 48 turns and K=8, the total-reward Teacher exceeded the matched downstream objective by +.1700 and Student-only by +.0985. Its paired difference from tuned UCB was −.1202. At K=24 it trailed Student-only by −.4382.
- On another 4,608-game panel, the K=8 total-reward Teacher was +.4456 [.2742,.6148] vs Student-only at 96 turns. Exploratory same-panel UCB comparison was −.1760 [−.3080,−.0382].
- Actual joint actions are archived for v2 training and evaluation. Scalar replay checked 2,592 main trajectories and 180 horizon trajectories. Key effects and confidence intervals were independently recalculated from raw files.
- 239 regression tests ran, 235 passed and four CUDA-dependent tests skipped. The final run resumed successfully after execution termination. Raw runs are in runs/fast_exploration_v2_{final,horizon_final,horizon_ucb}/.
- Historical v1 and D-031 paths remain indexed in README.md. The v2 development observations and follow-up hypotheses are documented in research/fast_exploration/architecture/EXPERIMENT_PLAN.md.

## 다음 세션 중점 · 게임 유형, 복잡도, 규칙 거리

- 게임 계열별로 어떤 탐색 행동이 유용하고 비용을 회수하는 데 몇 턴이 걸리는지 다시 살핀다. 보상의 시간 지연, 상태 의존성, 관측으로 줄일 수 있는 불확실성, 상대의 반응과 학습을 서로 구분한다.
- **검토할 가설** — 지금의 게임은 단순해서 학습된 탐색의 이득이 작게 나타났을 수 있다. 장기 최적 행동에 필요한 정보가 있고 그 정보를 얻는 행동이 중요할수록 메타 탐색의 추가 이득이 커지는가? v2의 96턴 결과는 이 질문의 동기다. 같은 정책을 오래 평가한 결과에서 게임 자체의 복잡도나 탐색의 정보 가치를 조작한 것은 아니다.
- **규칙 거리의 핵심 문제** — 규칙의 선형 결합 계수가 일정하게 변해도 최적 행동, 도달 가능한 상태, 정보의 가치가 일정하게 변한다는 보장은 없다. 복잡한 게임에서도 거리를 연속적·정량적으로 조절하고, 그 거리가 실제 전략·정보 구조의 차이를 반영하는지 검증할 방법을 깊이 탐구한다. 계열 내·계열 간 비교, 행동 이름 변경에 대한 불변성, 유사도와 복잡도의 분리가 쟁점이다.
- 재개할 때 v2의 계열별 행동·보상 기록과 유사도 결과부터 확인한다. v2에서 보상 유사도와 피드백 이득의 상관은 거의 0이었고, 계열 내 전이 유사도는 상수였다. 이번에는 아이디어를 인수인계에만 기록한다.
