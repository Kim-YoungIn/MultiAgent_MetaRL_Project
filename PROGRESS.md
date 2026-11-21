# Research status, 2026-11-21

Current objective and stage
- Completed the independent study of when a learned population prior helps multi-agent adaptation. Start at research/REPORT.md and research/REPRODUCE.md.

Verified outcomes
- D-031 raw records reproduce M−P −0.002601 for one of 25 planned primary pairs. H1/H2 for D-031 remain unresolved. See research/adapt_v1/HISTORICAL_AUDIT.md.
- Two-arena study: 384 fits, 2,048 held-out games, 32 independent blocks. Four-probe matched-prior advantage +0.06798; balanced vs narrow training +0.05352. Reversed concordance harmed transfer. All fits, outcomes and primary intervals independently reconstructed in runs/adapt_v1_final/.
- Supplied structured extension: 512 fresh games, Q-learning opponents and continuous state. Mixed empirical-Bayes minus scratch gain to eight probes −0.01510 [−0.02818, −0.00290]. Raw replay in runs/structured_transfer_v1/.
- Material v1 deviations: two motifs have one public state duplicated into two model features, and stochastic gates occur only in resource/mixed. The original run and launch protocol are preserved.
- Two-turn exact diagnostic: three actions/depth three, 73/96 pairs gained from observing first reward, 40/96 from allowing a nongreedy opening. These measure different opportunities. Verified counts and 12 complete brute-force comparisons in runs/structured_oracle_v1/.
- Fresh cross-size study: trained at three actions/depth three, transferred without refitting to six actions/depth five. At eight probes learned prior minus adaptive pooled +0.00376 [0.00071, 0.00702]; ordering reversed at sixteen. See runs/structured_scale_v1/.
- Fresh runs reproduced all raw numerical blocks in the three learning studies. Repository suite 220 passed, 3 GPU-only skipped; seven new oracle/size tests passed. Compact outputs and preservation hashes in research/evidence/.

Active interpretation and next action
- The studies identify budget- and population-dependent transfer under supplied representations and myopic adaptive policies. Discovery of strategic primitives, general H1/H2 and a causal threshold in complexity remain open.
- All four evaluation panels are open. For a next research iteration, use a new protocol to compare a belief-conditioned planning policy with its myopic counterpart on structured tasks with measured two-turn opportunity.
