# TASK-004 — Interim Scientific Cross-Audit
**Date:** 2026-11-27 · **Role:** research_critic · **Status:** Proposal for Lead review; not a final freeze.

## 1. Executive Verdict

The tables support conditional benefits, meaningful reversals under distribution shift, and a shared Exp 4 training failure. They do **not** establish that auxiliary prediction is universally necessary, that equivariance alone caused Exp 3B’s gains, or that a bootstrap Catch-22 caused Exp 4’s failure.

- **Masking:** Broad conclusions persist, but several endpoint verdicts change. R1’s information-value evidence changes substantially; “no conclusions changed” would be inaccurate.
- **Exp 3A:** The welfare reversal is supported at the tested stress levels. Robustness remains comparator-specific, and information acquisition is not separated causally from coordination.
- **Exp 3B:** Strong architecture-package gains are supported on three suites. Unseen-family success depends on the counterpart. Auxiliary benefit remains **INCONCLUSIVE**, not equivalent to zero.
- **Exp 4:** No evaluated neural method earns positive higher-tier exploitation rewards. Catch-22 is a plausible explanation requiring intervention.

**Scope and checks:** Used compact tables and the two permitted source files; no `runs/` or full `summary.json`. Missing brief filenames were replaced by existing Exp 3A `r3_obs_reliability.csv`, `coop_trap_dynamics.csv`, `paired_contrasts.csv`, and Exp 3B `matchups.csv`. Five read-only `unittest` checks passed: 620 contrast-row verdict/rate checks, Exp 3A/3B/4 mean reconciliation, tier totals, and default-network equivariance across all 24 permutations. No files were modified.

## 2. Masking Fix & Exp 3A Audit

**Masking evidence:** Current compact contrasts were compared with the same tables at `ac0766a^`.

| Endpoint | Before → current | Interpretation |
|---|---|---|
| Exp 1 total return, R0/R1/R3, full minus masked | +2.40/+1.60/+1.57 → +3.20/+2.40/+2.345 | All remain `SUPPORTED_POS`. |
| R1 K4 commitment/adaptive value | 0/0, resolved null → +13.0/+1.9, supported positive | Material information-endpoint revision. |
| R1 K4 net adaptive value | −1.4, supported negative → +0.5, supported positive | Directional verdict reversal. |
| R3 K4 commitment/adaptive/net value | +10.4/+2.02/+0.84 → +13.0/+3.425/+1.945 | All remain supported positive. |

Exp 2A matched full-minus-masked total return remains supported positive against Q-learning: continuous **+1.7518 [1.0406, 2.4598]**, restored **+1.8818 [1.4509, 2.3875]**. Against unseen SARSA, total-return contrasts remain inconclusive, while post-prefix contrasts become supported positive: **+1.2211 [0.6388, 1.8384]** and **+1.4318 [0.6173, 2.2694]**. Zero-shot restored SARSA total return changes from inconclusive to **supported negative**, −3.7938 [−6.9503, −1.2857].

Exp 2B competitive masking contrasts remain inconclusive. Cooperation gains mostly persist; unseen-rule focal return against Q-learning changes from inconclusive to supported positive: **+0.8576 [0.3728, 1.3432]**. These are revisions to table evidence; this audit does not independently certify Exp 1/2 retraining provenance or encoder boundaries.

**Exp 3A corruption:** `mt` identifies the Exp 2A training condition; the compact evidence does not establish corruption-matched training.

| Corruption d | `mt_aux_yr` mean / median | Difference vs matched structural | Difference vs blind prior |
|---|---|---|---|
| 0.0 | 16.060 / 16.9 | −1.145, supported negative | +11.260, supported positive |
| 0.3 | 10.292 / 15.0 | −2.565, supported negative | +5.492, supported positive |
| 0.6 | 0.269 / −0.8 | +4.663, supported positive | −4.531, supported negative |

At d=0.6, `mt_aux_yr` also loses to matched vanilla: **−3.034 [−3.897, −2.033]**. Its win-rate against structural is 59.3%, but only 32.2% against the blind prior. Its early AUC is −0.0887 versus the prior’s +0.2; counterpart returns are 50.7904 versus 48. Thus “more robust than structural at high corruption” is justified; general robustness superiority is not.

Mean/median divergence matters: fixed-trained `exp1_aux_yr` at d=0.3 has mean **0.340**, median **14.5**, and IQR **37.5**. Its negative mean contrast against the prior coexists with 62.2% wins. Also, `shifted_bayes_ref` scores **2.7045 < 4.8** for the feasible blind policy at d=0.6: it cannot be presented as an optimal ceiling there.

**Cooperation slip:** The headline measures **joint welfare**, cross-seed meta×meta minus UCB×UCB:

- d=0: **+12.317 [4.412, 20.300]**, median difference +11.05, win-rate 65.9%.
- d=0.3: **+2.465 [0.725, 4.093]**, median +1.7, win-rate 59.5%.
- d=0.6: **−3.853 [−4.568, −3.266]**, median −2.0, win-rate 40.5%.

This supports a reversal at the three tested points, without proving population-wide monotonicity or an information mechanism. Late mutual cooperation falls 0.6162→0.4099→0.2279; slip changes realized coordination rewards and both players’ trajectories. At d=0.6, welfare is 3.948 versus UCB’s 7.801; focal return is 2.127 versus 3.900. Restored evaluation removes prefix-carried opponent state, but subsequent shaping can recur.

## 3. Exp 3B Equivariant/D-031 Confound Audit

**Supported result:** Cross-seed auxiliary equivariant minus auxiliary One-Hot welfare is **+12.628 [9.016, 16.173]**, **+13.614 [10.585, 16.532]**, and **+12.682 [9.507, 15.912]** on in-distribution, unseen-permutation, and unseen-parameter suites. Median differences are +5.6/+8.5/+1.125; win-rates are 57.0%/58.7%/51.6%, with approximately 25% ties.

**Architecture confounds:** `equivariant_drqn.py` changes token encoding, recurrent-state organization, pooling, nonlinear heads, parameter sharing, and tie-breaking. Its default model has **14,307 parameters**, including **9,025 recurrent-plus-Q parameters**, and **4×32=128 recurrent-state scalars**, versus the nominal global GRU-64 reference. Actual fitted dimensions and One-Hot counts are not recorded in these tables. Neither parameter count nor effective expressivity is controlled.

Shared token processing and mean pooling support permutation equivariance; the numerical Q/auxiliary checks passed. This establishes a network property, not a zero empirical return gap between independently sampled suites. Random tie-breaking supports symmetry in distribution, without guaranteeing identical relabeled trajectories under the same random draws.

**Auxiliary interpretation:** Equivariant auxiliary-minus-vanilla welfare is −0.911 [−2.261, 0.661], −0.990 [−2.406, 0.543], −0.342 [−1.491, 0.766], and +0.531 [−4.723, 5.899]. All are **INCONCLUSIVE**. Vanilla’s architecture gains show that auxiliary prediction is not required to obtain a strong result here; they do not establish equivalence or explain why the methods differ.

**Unseen fifth family:** **30.7078 is absolute joint welfare against UCB**, not a gain. It exceeds UCB×UCB by **+8.022 [6.520, 9.484]**, with median difference +3.1 and 59.8% wins. Focal mean/median/IQR are **15.055/17.3/11.6**, early AUC **0.2246**, and counterpart mean **15.6528**. The cross-seed meta×meta comparison against UCB×UCB is instead **−3.891 [−8.367, 0.559], INCONCLUSIVE**; against One-Hot it is also inconclusive.

**D-031:** Doubling n4 budget improves welfare by +4.408/+4.089 on the first two suites and +6.304 on the unseen family, but only +0.236, inconclusive, on unseen parameters. Several supported mean improvements have zero median differences. More compute also changes optimization, replay history, and exploration scheduling; it does not isolate exposure dilution.

The n2 condition beats n4_b2 on the unseen family by **+5.370 [1.582, 9.191]**. This supports a distribution-specific transfer advantage. “Prior alignment” remains an interpretation, and “diversity decomposed into exposure dilution plus prior alignment” overstates causal identification. Train/test disjointness and bootstrap handling of both reused player identities cannot be certified from these aggregate tables.

## 4. Exp 4 Scaling Wall & Catch-22 Critique

All eight neural methods have **tier0_only=1.0** and zero observed positive higher-tier exploitation. This is not identical to all-safe behavior: auxiliary-y mean return is **10.01**, and discovery **12.74**, below all-safe **12.8**.

Discovery-minus-vanilla is **−0.06 [−0.18, 0], RESOLVED_NULL** at δ=0.3, with 80% ties and no wins. Discovery-minus-uniform is **+0.105 [0, 0.3156], INCONCLUSIVE**. Discovery’s median/IQR are 12.8/0, early AUC 0.2, and counterpart mean 128. The privileged policy reaches 120.625 and tier 3 in every evaluated episode.

**Mechanistic limitation:** The implemented discovery rule updates gate logits using cosine alignment between replay auxiliary gradients and a separate replay minibatch’s TD gradient. It does not differentiate through an updated inner optimizer or optimize held-out episodic return directly. Its final `gate_y=0.0817` and combined noise weight 0.5542 do not establish absence of useful extrinsic gradients.

Plausible alternatives include insufficient successful replay coverage, epsilon decay, replay eviction, TD target quality, recurrent credit assignment, clipping, `aux_scale`, `meta_gate_lr`, training duration, and discounting. The asserted warmup success frequency “<0.1%” needs a defined event denominator and measured visitation; evaluation tier counts cannot establish training coverage.

The table reports 1,075,200 training transitions and 316 discovery outer updates, without full fitted hyperparameters or matched computational cost. The driver calls Gate 0 before fits, but exact-DP correctness was not independently recertified. Its resume path checks an evaluation-array hash without checking requested config/git compatibility, so follow-ups need distinct output paths or stronger cache validation.

**Defensible wording:** “Under the tested training configuration, all evaluated neural methods failed to exploit higher tiers; sparse successful experience and local TD-based gate selection are candidate explanations.”

## 5. Required Caveats & Top 3 Next Experiments

Treat the many suite/method contrasts as exploratory pending a declared primary comparison. Report training-seed uncertainty, median/IQR, wins/ties/losses, early AUC, cost-inclusive total return, and counterpart return. Arithmetic checks do not independently reproduce bootstrap intervals, seed independence, split integrity, or run provenance.

1. **Highest priority: Exp 4 coverage × auxiliary-objective intervention.** Cross vanilla/fixed-yr/discovery with ordinary warmup versus a predefined public, type-independent repeat-biased warmup, keeping transitions and subsequent schedules matched. Log successful-tier training visits, replay retention, gradient norms/alignments, and gate trajectories. Recovery across methods supports a coverage bottleneck; discovery-specific failure after coverage restoration weakens the Catch-22 account and implicates gate selection. Failure despite verified coverage points toward optimization or credit assignment.
2. **Exp 3B architecture and feedback controls.** Compare equivariance with capacity/state-matched One-Hot and permutation-augmented controls; train masked/prior controls under their own observation restrictions. Report actual parameters and compute. Use a common fixed counterpart to isolate focal-policy effects, then report meta×meta separately. Predeclare unseen-suite focal return and early adaptation as primary endpoints.
3. **Complete the diversity × budget factorial.** Add n2_b2, evaluate multiple balanced two-family subsets, and retain every test suite without selecting favorable mixtures. This tests diversity at matched total budget and budget effects within each mixture; prior alignment remains a hypothesis unless independently manipulated.

Lock these protocols on development seeds, retain at least five independent training seeds, and evaluate new sealed seeds without tuning against the existing stress-test outcomes.