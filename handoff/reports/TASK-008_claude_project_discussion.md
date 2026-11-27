# TASK-008: Project-wide synthesis and pre-freeze position (Antigravity-Claude, `claude_analyst`)

**Date:** 2026-11-28 · **Status:** A proposal for the Lead to review. It has not been adopted. I changed no files.

**Inputs:**
- `context_pack_slim.md`, `INTERIM_SYNTHESIS_EXP1_TO_4.md` and `TASK-006`.
- All 5 Exp 5A CSVs and all 4 Exp 4B CSVs.
- `study_exp4b_compact.py`, read in three places: aux heads and gate (L148–186), training loss (L340–390) and `structural_model_sl` (L555–720).
- `envs/stage_lock.py`, `configs/exp4b_compact_multibarrier.json` and `configs/core_v3_sealed.json`.

## 1. TASK-006 thesis checked against Exp 5A and Exp 4B

**Confirmed:**
- **Exp 4 failed on credit assignment, not on missing data (TASK-006 decision rule 2).** Diag 1D found 508–602 positive Tier-3 warmup events, 8/8 cells observed, and `structural_model_mb` unlocked Tier 3 on 5/5 seeds.
  - **New supporting evidence:** Stage-Lock learned from *fewer* top-tier positives. It had `warmup_hits_s2 = 311` in about 207k warmup transitions (90 × 48 × 48), against about 550 in 276k for Exp 4.
  - **So** the raw count of positive events is not what decides success. The event definitions differ slightly, so treat this as directional.
- **Coverage unlocks all learners alike (P1, the "alike" half).** In Stage-Lock, vanilla, y, r, yr and discovery all reach `mean_max_stage` 1.56–1.75. Every aux-vs-vanilla contrast is `INCONCLUSIVE`: yr −3.07 (2/5 seeds), y −6.59 (1/5), r −5.71, discovery −8.10 (1/5).
  - **So** the aux objectives do not cause the unlock. Decision rule 5 (aux improves sample efficiency) is **unsupported**.
- **Surprise switch (Diag 1B), read-out rule applied.** It helps at d=0.6 (+1.56, 5/5 seeds) but loses to `prior_only` (−2.80, 0/5). It *hurts* at d=0.3 (p95 −0.99, `SUPPORTED_NEG`) and slightly at d=0. **Drop the GRU-surprise 2×2.**
- **Gate validity.** `drqn_aux_discovery` puts 95.3% (R3+distractor) and 87.2% (Stage-Lock) of its weight on (y, r). The mechanism passes. (In Exp 4 it ended with `gate_y` 0.08 and noise weight 0.55.)

**Falsified or revised:**
- **My VoI-collapse alternative is falsified (Diag 1A).** The PBVI lower bound keeps VoI = +7.20 at d=0.6 (12.00 vs 4.80), and the opening must change (best action 4).
  - `mt_aux_yr` (0.44) and `prior_only` (4.80) are both far below 12.0. **The safe floor is the wrong target.**
  - **Better reading:** the OOD failure is **miscalibrated inference** (overconfident exploitation). Neither a fallback rule nor more probing fixes it.
- **"Matches the structural learner" holds only for shallow chains.** In Exp 1 R3 it ties (17.52 vs 17.50). In Stage-Lock, `aux_yr − structural_model_sl` = **−25.90, 0/5 seeds**.
  - **Revised thesis:** recurrent DRQN amortizes Bayes-adaptive probing when two conditions hold:
    1. probe→payoff credit paths are short and cheap relative to trap costs;
    2. the test lies inside the training support.
  - The amortization gap to a correctly specified structural learner grows with depth.
  - Aux prediction stabilizes the **shallow** regime (Exp 1; R3+distractor yr−vanilla +4.56, 4/5 seeds) and is neutral in the deeper Stage-Lock.
- **Exp 3B is not about changing the opening action (Diag 1C).** VoI(H=4) is positive (+0.86 to +1.31), but `symmetric_opening_tie=True` and `blind_opt_count=4`, so "must change opening" is undefined.
  - Any Exp 3B exploration gain is within-episode response adaptation after turn 0.
  - `comp_rps` has VoI = 0, which makes it a **negative control**, not an exploration test.

## 2. Hidden flaws and fairness asymmetries (audit)

1. **`structural_model_sl` does not see the data with privilege, but its model class is privileged.** It cannot leak `env.keys`, but it hard-codes:
   - the meaning of `SL_ACT_P`;
   - the env's exact stage and unlock transition law (`a==P ∧ ul==1 ∧ st<2 → st+1`, `r>0.25 → ul=1`);
   - K = `SL_N_KEYS` = 4 clusters, deterministic `argmax` responses, `f` shared across stages, and a key-belief reset at each crossing.

   It matches `bayes_optimal` on **every** column, including `early_auc` 0.0125 and `opp_return` 115.5, with std 0. **Label it a "correctly specified structural learner (model class known)"**, not a planner-free baseline. The same applies to Exp 1's `n_clusters = N_TYPES`. The DRQN-vs-structural gap therefore measures *the cost of not knowing the model class*. Framed that way, it is a clean, publishable contrast. It also used about 98k random transitions against the DRQN's 1.04M.
2. **R3+distractor is not a replication of Exp 1** (`aux_yr` 10.75 vs 17.52; vanilla 6.19 vs 4.59). Several settings differ at once:

   | Setting | Exp 1 | Exp 4B |
   |---|---|---|
   | Iterations | 450 | 400 |
   | Warmup | 100 | 80 |
   | Effective aux weight, y / r | 2.0 / 1.0 (total 3.0) | 1.34 / 0.66 (`aux_scale` 2.0) |
   | r-loss | — | `mse(pred, r/r_scale)·r_scale` (may also differ from Exp 1) |

   **So** the gap cannot be pinned on either iterations or scale alone. Also, `uniform` (14.13) ≥ `aux_yr` (10.75) and ≈ `discovery` (13.69; diff −0.44, `INCONCLUSIVE`), with `aux_yr` seed std 3.99 against median 14.7. The R3 track looks under-converged. The +7.50 discovery−vanilla result is valid within the study, but **do not set R3 returns from Exp 4B beside Exp 1 returns.**
3. **Discovery vs uniform (+23.52) is harm-avoidance against a strawman.** `uniform` puts 0.5 × 2.0 = 1.0 of the loss weight on pure-noise heads and collapses (35.65, std 35.5, 2/5 seeds reach Stage 2). Discovery − vanilla is −8.10 `INCONCLUSIVE`. In R3, noise heads did *not* hurt (uniform beat vanilla). **Allowed claim:** "the gate correctly suppresses harmful distractor losses." **Not allowed:** "discovery improves adaptation."
4. **The jump from 22 to 6 actions changes four things at once:**
   - action count;
   - horizon (64 → 48);
   - miss cost (−10 → −0.6 to −1.2);
   - probe depth: in Stage-Lock, one P (−0.2) reveals the key exactly (`y = 1 + k`), and arms can also be found by repeat-trial.

   **So** "slimming the action space breaks the wall" is over-attributed. Say instead: "a compact, cheap-probe 3-layer variant is learnable; the responsible factor is not identified."
5. **Exploration is not attributed in Stage-Lock.** There is no prefix-swap, neutral audit or probe-use metric (AGENTS §3).
   - `early_auc_1_8` is degenerate: about 0 for every method, Bayes-opt included.
   - A DRQN can find stage keys by repeating arms instead of probing, so the 67 return is not yet evidence of *probing*.
6. **Control strength:**
   - In Stage-Lock, `masked` hides only 8 of 48 steps, so `aux_yr − masked` is +22.05 but `INCONCLUSIVE` (3/5 seeds).
   - `rl2_ppo` ran on 264k transitions vs 691k for the DRQN. This is a fairness asymmetry *against* the canonical meta-RL baseline.
   - In Diag 1D, seeds 1002 and 1003 have mean **31.15** against median 120.55. The cross-seed mean is about **84.8**, not 120. The no-mean-only rule requires reporting it.
7. **Exp 1/4/4B use a fixed, honest opponent**, so `opp_return` only tracks hits and no opponent shaping is possible. These studies are contextual POMDPs, and the paper must say so. The multi-agent claims rest on 2A, 2B, 3A and 3B only.

## 3. Minimum decisive final block (freeze around Dec 8)

**Must run (protects the largest claims; about 2–3 days of CPU):**
- **A. Exp 3B One-Hot controls:** a permutation-augmented One-Hot DRQN (primary, my P3) and a capacity- and state-matched One-Hot (secondary). Run 4 suites plus the unseen family, 5 seeds each. Report focal return against **fixed UCB/Q partners** as the primary metric, and meta×meta as secondary (because of convention sharing).
- **B. Stage-Lock probe attribution, eval-only on existing checkpoints (no retraining):** per stage, report
  - the rate of P before the first exploit;
  - turns until the key is identified, using the neutral posterior (exact, since P is fully revealing);
  - turns to the first hit;
  - the rate of repeat-trial key discovery.

  These replace the degenerate early AUC and satisfy the §3 decomposition.
- **C. Fix or match `rl2_ppo`:** either rerun it on R3 at 691k transitions (5 fits), or remove every "beats RL²" claim.

**Should run (cheap; about 20 fits):**
- **D. Re-run R3+distractor at Exp 1's exact hyperparameters:** 450 / 100 iterations, y = 2.0 / r = 1.0, i.e. `aux_scale` 3.0 with yr at 0.67/0.33. Otherwise, footnote the non-replication explicitly.

**Cut or freeze (report as limitations or future work):**
- GRU-surprise retraining and the noise-matched 2×2 (P2).
- Further 22-action Exp 4 interventions (ε-z-greedy, oracle seeding). Diag 1D plus Exp 4B are enough.
- A factorial on the 22→6 confound. A single padded-action Stage-Lock arm is a stretch goal only.
- D-031 `n2_b2` and balanced subsets. Report D-031 descriptively, with no dilution-vs-alignment claim.
- More `comp_rps` work (freeze it as the zero-VoI negative control).
- New environments.

**Hygiene before freeze (no compute):**
- One pre-declared primary contrast per study, with k/5 shown beside the bootstrap CI.
- Rename Exp 4 `RESOLVED_NULL` to "not testable at floor".
- Report Diag 1D's mean (about 84.8).
- Run a key-mapped CSV↔JSON reconciliation script.
- Label `structural_*` as correctly specified and model-class-privileged.

## 4. Top 3 reviewer attack vectors

1. **"Is this exploration, or just exploitation and coordination?"** The weak spots: Stage-Lock has no probe-use metric and key-finding by trial works; Exp 3B/2B have symmetric opening ties; meta×meta conventions are shared; and fixed-opponent studies cannot separate information gain from shaping. *Defense:* items B and A, fixed-partner focal return, and Gate-0 tables (Diag 1A/1C) printed beside every result.
2. **"Your baselines are asymmetric."** Each one cuts a different way:
   - the structural learners know the model class (gap overstated as a "planner ceiling");
   - `rl2_ppo` was under-budgeted;
   - `uniform` is a strawman for discovery;
   - Stage-Lock's `masked` is weak;
   - Exp 3B's `EquivariantDRQN` is not capacity-matched.

   *Defense:* item C, item A, honest labels, and recasting discovery as harm-avoidance.
3. **"Your attributions are confounded and the results don't replicate."** The examples: 22→6 changes four factors at once; R3 `aux_yr` scores 10.75 vs 17.52; 5 seeds give bimodal outcomes (vanilla 4/5 reach Stage 2; uniform std 35) and narrow percentile-bootstrap CIs. *Defense:* item D, k/5 counts plus median and IQR everywhere, and restrained wording ("a compact variant is learnable; the factor is unidentified").

## 5. Paper narrative spine (high level only)

1. **Part 1 diagnosis and the Gate-0 method.** Earlier meta-exploration raised the mean but lowered the median because most tasks did not *require* probing. We therefore pre-certify every suite with exact or PBVI VoI and the must-change-opening test before training. This methodology is itself a contribution.
2. **Shallow, required probing (Exp 1 R3; R3+distractor).** Recurrent DRQN with public (y, r) prediction reaches 99.7% of Bayes-opt. Masked feedback separates feedback-driven adaptation. It *ties* a correctly specified structural learner: amortization, not superiority. (RL² is shown at a matched budget.)
3. **Reliability and non-stationarity boundaries (Exp 2A, 3A, 5A-1A/1B).** Meta-RL beats a *misspecified* structural model but is overconfident OOD. PBVI shows VoI persists at d=0.6 (+7.2), so the failure is miscalibrated inference. An unlearned surprise switch recovers only part of it.
4. **Multi-agent transfer through inductive bias (Exp 2B, 3B, 5A-1C).** Label equivariance, not aux losses, drives cross-family transfer, pending the augmented and capacity controls. Openings are symmetric ties, so the adaptation happens within the episode. `comp_rps` (VoI = 0) and the `coop_trap` slip reversal mark the limits.
5. **Depth scaling (Exp 4, 4B, 5A-1D).** With 22 actions and harsh traps, the failure is model-free credit assignment even though the data suffices. A compact cheap-probe variant is learnable by *all* DRQN variants: aux is neutral, and the gate only suppresses harmful signals. The structural learner hits Bayes-opt, so the gap grows with depth.

   **Conclusion:** meta-learning's gain in exploration depends on credit-path length, training-support coverage and inductive bias, not on auxiliary objectives as such.
