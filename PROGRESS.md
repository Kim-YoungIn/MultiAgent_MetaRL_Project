# Research status, 2026-11-21

Current objective and stage
- Measure how rule similarity relates to adaptation after exploration, and finish outstanding D-031 work at the original budget.
- Similarity analysis and reward-stakes experiment complete. D-031 full-budget execution active in runs/d031_completion_v2/study/; the original run remains preserved.

Verified outcomes
- D-031 original first pair, 64 fresh games: within-family Spearman strategy similarity vs M−P adaptation AUC −.090, payoff similarity vs M−P −.067. Recomputed paired seeds, trial AUCs, strategy losses and ranks. This is one trained pair.
- Structured games, 24 independent blocks and 3,072 games: reward/transition similarity vs 8-probe retained-experience gain Pearson −.042 [−.072,−.012]. Extra gain vs ordinary pretraining −.008 [−.042,+.025]. Exploratory, opened panel. See research/rule_similarity/REPORT.md and evidence/similarity.png.
- Reward-stakes proposal: 16 blocks, 4,096 games and 32 shuffles. At terminal payoff weight eight, sensitivity contrast +.00188 [+.00080,+.00294]. Genuine full weighted return favored the myopic opener over the specialized opener. See research/probe_stakes/REPORT.md.
- D-031 CUDA development completed 15/15 fits and all 51 evaluation receipts verified under the original seal. See runs/d031_completion_cuda_dev/verification.json. D-031 regression tests 20 passed.
- D-031 full-budget continuation restored every sealed source and the original CPU runtime. 3→4/5 action transfer c1/j1 completed with M−P −.00875 and −.00688, verified from raw trials. At 19:09 KST 19/375 fit ledgers and 1/25 primary pairs were complete. CPU continuation first evaluates all 25 primary pairs, then runs the extensions.
- Earlier matched counterpart-learning study and original H1/H2 evidence remain in research/opponent_adaptation/REPORT.md and research/REPORT.md.

Next action
- Check live counts with .venv/bin/python -m research.d031_completion.status runs/d031_completion_v2/study. Completion requires 375 ledgers and the planned evaluation shards, with no missing files. Use the sealed continuation command in research/d031_completion/README.md if the run is interrupted.
