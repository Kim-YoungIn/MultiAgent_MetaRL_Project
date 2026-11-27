# Prior Work Synthesis (Legacy v0.1–v0.9)

> **Archive pointer:** All pre-refactor code and 15 full study reports are preserved verbatim at git tag **`archive/v0.9-legacy-studies`** (commit `290dabf`). Inspect any historical file via `git show archive/v0.9-legacy-studies:<path>`. Preserved key figures and summary tables live in `results/legacy_summary/`.

## 1. What Was Tried

Across v0.1–v0.9, the project progressed through four generations of multi-agent meta-exploration setups:

1. **Memory-One Logical / Relational Games (`05_toy_models/mamrl/`, v0.1–v0.5):**
   2-player simultaneous games (`n = 3..5`) with an `ActionSet` recurrent PPO explorer (Teacher), online Q-learning Student, and online Q / Expected SARSA opponent. Optimized retained-gain objective $G = R_{\text{assessment}} - R_{\text{erased}}$.
2. **D-031 Sealed Diversity & Transfer Study (`experiment/`, `d031_completion`, 375 fits / 175 cells):**
   Full-budget comparison of retained-gain Teacher (`M`), exploration-reward Teacher (`P`), and masked Teacher (`O`) across training rule counts (`8 / 32 / 128`), families (`focused` vs `broad`), opponent mixtures, and cross-action transfer (`n = 3 ↔ 5`).
3. **Cross-Context & Structured Transfer Pilots (`adapt_v1`, `structured_*`, `opponent_adaptation`):**
   Bayesian filtering across two-arena games (varying sign concordance $q$) and empirical-Bayes / ridge-regression reward priors on 4-family stateful structured games (`resource`, `sequence`, `reciprocity`, `mixed`; `n = 3` and `n = 6`).
4. **Fast Exploration v1/v2 & Dynamics Transfer (`fast_exploration*`, `dynamics_transfer`, 36 blocks, 2,304–4,608 test games):**
   Recurrent Teacher + Bayesian Linear Student + Q-learning opponent over 48/96 turns (`K = 4, 8, 16, 24`), comparing Full-Feedback Teacher, Masked Teacher, Immediate-Reward Teacher, UCB, and Student-only, plus dynamics shifts (reducing joint-investment transition reliability by $d \in \{0, 0.3, 0.6\}$).

---

## 2. What Failed and Why (Verified Empirical Findings)

A retrospective re-aggregation of all 15 completed studies (`archive/v0.9-legacy-studies:research/retrospective/REPORT.md`) uncovered five concrete reasons why meta-learned exploration failed to show general fast-adaptation superiority:

| Failure Mode | Verified Evidence (Legacy Runs) | Root Cause |
|---|---|---|
| **1. Mean vs. Median Illusion** | On two 96-turn panels (4,608 & 2,304 games), Teacher − Student **mean** was `+0.446 / +0.428`, but **median** was **`−0.144 / −0.142`** and win-rate was only **40.1% / 40.6%**. Excluding `reciprocity` games turned the mean negative (`−0.058 / −0.082`). | Heavy right-tail payoff in one family (`reciprocity`: `+1.96` mean, `−0.199` median) masked losses on >59% of games. |
| **2. Meta-Explorer Lost to Simple Baselines** | 96-turn Teacher − UCB mean was **`−0.176 / −0.094`** (43.5% win-rate). Feedback conditioning (Full − Masked Teacher) added only `+0.00097` AUC (below the `+0.002` practical bar). In D-031, `M − P = −0.01274` (`[−0.01608, −0.00954]`), and in competitive rules both lost to Random (`−0.01901` / `−0.01191`). | The explorer did not learn an adaptive information-seeking strategy; in earlier Q-Student runs, PPO collapsed to action repetition while reducing rule identifiability below Random. |
| **3. Passive Info ≠ Active Exploration Need** | Exact 2-turn Oracle (`results/legacy_summary/oracle.csv`): in 6-action depth-5 games, **62 / 96 pairs** benefited from observing the first reward, yet **only 6 / 96 (6.3%)** required changing the opening action away from a masked/greedy optimum. | If greedy actions already reveal the hidden rule, there is almost no margin for a meta-learned explorer to beat UCB or greedy play. |
| **4. Exploration Confounded with Opponent Shaping** | Replacing Student-only with Teacher raised the *opponent's* 96-turn return by **`+0.365 / +0.371`** and increased joint-investment frequency in reciprocity games (`+3.06%p`). Lowering joint-investment reliability ($d = 0.6$) erased `−0.399` of Teacher's 96-turn gain while leaving controlled feedback gain unchanged (`−0.00012`). | Long-horizon gains came largely from nudging game state and the learning opponent into cooperative trajectories, not from gathering superior information for the Student. |
| **5. Task Diversity Confounded with Exposure Dilution** | In D-031 under fixed total budget, increasing training rules from 8 → 128 worsened `M − P` by **`−0.01017`** (`[−0.01996, −0.00243]`). In `adapt_v1`, a learned cross-context prior helped when test concordance matched training (`+0.068` at $q=0.75$) and hurt symmetrically when reversed (`−0.272` at $q=0.25$). | Increasing rule count at fixed compute cuts exposure per rule; diversity helps only when reusable structure matches test deployment and exposure is controlled. |

---

## 3. Preserved Legacy Evidence (Git-Tracked, Compact)

- `results/legacy_summary/distribution.png` — Empirical CDF of game-level 96-turn Teacher − Student return differences (positive mean, negative median).
- `results/legacy_summary/joint_returns.png` — Focal vs. counterpart 96-turn return shifts and family reweighting sensitivity.
- `results/legacy_summary/oracle.csv` — Exact 2-turn Value-of-Information vs. `must_change_opening` counts across 3×96 game pairs.
- `results/legacy_summary/family_weights.csv` — Reweighted Teacher/UCB/Student returns across family mixes.
- `results/legacy_summary/diversity.png` — D-031 diversity and transfer contrasts across 375 completed fits.

---

## 4. Direct Implication for Path A

We do **not** treat the prior work as wasted effort. Together, these findings establish **Part 1 of our paper**: *standard multi-agent meta-exploration benchmarks silently fail because passive/greedy actions already leak task identity, mean metrics hide median degradation, and opponent shaping masks information gain.*

**Part 2 (`docs/SPEC.md`)** builds a minimal, Oracle-verified benchmark where **costly active probing is strictly required** (`must_change_opening = 100%` in the Active regime vs. `0%` in the Passive control), cleanly testing when meta-learning genuinely improves exploration and adaptation.
