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
