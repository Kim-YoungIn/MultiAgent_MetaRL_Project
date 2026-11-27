# TASK-006 — Second-opinion interim design review (Antigravity-Claude, `claude_analyst`)
**Date:** 2026-11-28 · **Status:** This is a proposal for the Lead to review. It is not a final freeze and has not been adopted.
**Inputs:** `context_pack_slim.md`, `INTERIM_SYNTHESIS_EXP1_TO_4.md`, the TASK-004 and TASK-005 audits, the compact CSVs (Exp 3A `r3_obs_reliability`, Exp 4 `methods`), and targeted read-only reads of `envs/multi_barrier.py`, `study_exp4_multibarrier.py` and `baselines_v3.py`. I did not read `runs/` or the full `summary.json`, and I changed no files.

## 1. Unifying thesis

**Candidate thesis (falsifiable).** The meta-learned recurrent policies *amortize* Bayes-adaptive probing **only inside the support of their training experience**. Coverage matters at three levels:
- **within-episode trajectory coverage** (Exp 4),
- **across-task/opponent coverage** (Exp 2A, 3A, 2B),
- **label-permutation coverage** (Exp 3B).

Inside that support, they match a correctly specified structural learner but do not exceed it. Outside it, they extrapolate **commitment rather than caution**.

**Evidence anchors:**
- Exp 1 R3: `aux_yr` scores 17.52 and the structural model 17.50 (`RESOLVED_NULL`). This is the ceiling of amortization, not a win over model-based inference.
- The only cases where meta-learning beats the structural model come from structural misspecification: Exp 2A Restored (+1.86) and Exp 3A at d=0.6 (+4.66). Both still lose to the safe floor of 4.8 at d=0.6 and on unseen SARSA.
- Exp 3A at d=0.6 (`r3_obs_reliability.csv`) separates exploration from exploitation:
  - The neutral auditor's posterior entropy is about the same for both methods (`mt_aux_yr` 1.886, `mt_vanilla` 1.928), and MAP accuracy is 0.366 vs 0.331. So both **collect similar information**.
  - But `mt_aux_yr` commits to the wrong exploit 3× as often (`expt_X_mismatch` 0.145 vs 0.048) and falls back to S less often (0.475 vs 0.764).
  - So the OOD failure is **overconfident exploitation given the information, not over-exploration**. The synthesis should rename it accordingly.
- Exp 4: all 8 neural methods sit at the S-floor. No higher-tier payoff trajectory ever shaped their values.

**Pre-registered predictions that would falsify the thesis:**
- **P1 (Exp 4):** Restoring coverage, with no objective change, unlocks tier ≥ 1 for vanilla, yr and discovery alike. *Falsified if* tiers stay locked even with privileged replay seeding.
- **P2 (Exp 3A):** Training on d ∈ {0, 0.3} produces a graded fallback at held-out d=0.6 with no new input feature. *Falsified if* the fallback appears only once a surprise input is added.
- **P3 (Exp 3B):** A permutation-*augmented* One-Hot DRQN closes most of the +12.6 gap. *Falsified if* it does not; the gain would then be architectural sample efficiency or capacity, not coverage.

**What not to claim:**
- That "meta-learning improves exploration" in general.
- That aux-y is necessary. It is `INCONCLUSIVE` in Exp 1 and Exp 3B, and floor-tied in Exp 4.
- That diversity decomposes causally into dilution plus alignment.

## 2. Exp 4: breaking the Tier-0 wall and separating the catch-22

**Mechanics that matter (from `multi_barrier.py`):**
- An exploit pays only when it is **repeated** (`q == a and y == 1`). The first play costs −1, −3 or −10 depending on tier.
- Under uniform ε, 8 of the 22 actions are T3 actions. Almost every T3 play is penalized at −10, so value estimates for exploits are pushed down hard during warmup.
- The response `y = f_θ(q)` is observed after **every** action, with or without reward. Random warmup (about 4.3k episodes × 64 steps) therefore already contains enough data to identify `P(y | q, θ)` and `r(q, a, y)`.

**Implication:** the "no positive trajectories" form of the catch-22 may be wrong. The data may be **informationally sufficient**, with the failure lying in model-free value propagation or credit assignment across probe chains of 3 or more steps.

**Interventions compared:**

| Intervention | Manipulates | Confound | Verdict |
|---|---|---|---|
| Episodic `(q,a)` count bonus | reward | Changes the objective. It also rewards repeated *wrong* T3 entries (−10). Needs β tuning. | secondary |
| Prediction-error curiosity | reward | Collinear with the aux-y factor, so it contaminates the aux arm. Public noise streams `z1`/`z2` create a noisy-TV risk. | **reject** for separation |
| Barrier curriculum | task distribution | Confounds diversity with per-task exposure (an AGENTS §1 concern). | feasibility check only |
| `aux_discovery` on R3 + 2 noise distractors | gate validity | Ground truth is known (y useful, noise useless). Does not test coverage. | **cheap precondition** |
| ε-z-greedy (temporally extended random repeats, Dabney et al. 2020) | behavior data only | Repeat bias matches the env's confirmation mechanic, which must be disclosed. Use the default ζ (μ=2), untuned. | **core arm** |
| Privileged Bayes-opt replay seeding (ρ=10%) | coverage ceiling | Privileged, so diagnostic only and never a baseline. | **anchor arm** |

**Recommended single design (Exp 4B):** learner {vanilla, fixed-yr, discovery} × data source {ε-greedy, ε-z-greedy, oracle-seed ρ=10% (diagnostic)}.
- Matched transitions (1.075M) and identical schedules, 5 seeds each: 45 fits, about 1.1× the Exp 4 cost.
- One extra unprivileged row, **`structural_model_mb`**: EM with M=8 on the *same* ε-greedy warmup data, plus a full-horizon belief DP over the *learned* model. Reuse the Gate-0 belief-mask solver; the default `plan_depth=4` is too short for a 3-probe chain followed by the −10 entry cost.
- **Mediators to log:**
  - training-time positive-tier events per 10k transitions (public: `q == a`, `y == 1`, tier ≥ 1);
  - replay retention of those events;
  - gate trajectories;
  - TD and aux gradient norms.
- **Primary endpoint:** tier ≥ 1 exploitation rate and total return, reported per seed (k/5).

**Decision rules:**
1. Oracle-seed fails for every learner → representation or optimization failure. Stop pursuing exploration fixes.
2. `structural_model_mb` unlocks on ε-greedy data but DRQN does not → the data is sufficient; refine the catch-22 into a **credit-assignment** failure.
3. ε-z unlocks all three learners equally → **coverage bottleneck**; aux is irrelevant here.
4. ε-z unlocks vanilla and yr but not discovery → **gate-selection failure**. This would be consistent with the current final gates, where noise weight is 0.554 and `gate_y` is 0.082.
5. yr unlocks at lower *measured* coverage than vanilla → aux improves sample efficiency, which is the only positive aux claim available.

Run the R3 + distractor discovery check first (one seed sweep). If discovery cannot up-weight y where y is known to help, its Exp 4 null says nothing about catch-22.

## 3. Exp 2A/3A: is online surprise fed into the GRU a clean test?

- **Unprivileged:** yes. `pred_y(h, a)` already exists in the `aux_yr` checkpoints, so `−log p(y_t | h_{t−1}, a_{t−1})` uses only public signals.
- **Not clean as specified, for four reasons:**
  1. **No new information.** Surprise is a deterministic function of quantities the GRU already sees. Any gain is an inductive-bias or capacity effect. It needs a control with a *shuffled* surprise scalar of the same dimension.
  2. **Extrapolation.** If training uses only d=0 or the Q-learner, high surprise values never occur during training. The policy's response to them is undefined, so surprise alone cannot teach "fall back when surprised".
  3. **Matching confound.** Training with noise to cover high-surprise values makes it noise-matched training. That must be crossed with the surprise input as a 2×2: {surprise, shuffled} × {train d=0, train d ∈ {0, 0.3}}, evaluated at held-out d=0.6 and on SARSA.
  4. **Feedback loop.** Use stop-gradient on the surprise input, or the aux head can game its own input.
- **Missing target.** Even at d=0.6, `shifted_bayes_ref` (2.70) is below the blind floor (4.8), so the correct behavior there is unknown. Compute the exact Bayes-optimal (Gate-0) under corruption at d ∈ {0.3, 0.6} first. If VoI ≈ 0 at d=0.6, the test reduces to "detect VoI collapse → stop exploiting".
- **Cheapest decisive step (eval-only, near-zero cost):** add a non-learned **surprise-threshold switch** to the existing `mt_aux_yr` checkpoints.
  - If cumulative surprise exceeds τ, play S for the rest of the episode.
  - Calibrate τ as the 99th percentile on the *training* distribution only, never on test.
  - Write outputs to a new, unsealed directory.
  - Read-out: if it beats both `mt_aux_yr` and `prior_only` at d=0.6 and on SARSA without losing at d=0, surprise carries actionable OOD signal, and the GRU-input 2×2 becomes worth running. If not, drop the idea.

## 4. Blind spots not raised in TASK-004/005

1. **Floor-effect "equivalence".** In Exp 4, discovery − vanilla `RESOLVED_NULL` (80% ties) is measured with both methods pinned at the S-policy. A δ-equivalence verdict at a floor is uninformative. Label it "not testable at floor", not `RESOLVED_NULL`.
2. **CI validity with 5 clusters.** A percentile hierarchical bootstrap over 5 training seeds tends to give intervals that are too narrow. Add a seed-level sign count (k/5) or a seed-level t-interval for every headline claim.
3. **Exp 4 conflates three scaling axes.** Moving from Exp 1 to Exp 4 changes actions (8 → 22), horizon (24 → 64) and probe depth (1 → 3) at once, so "scaling wall" is unattributed. Gate-0 says T3 is optimal from T=16, so a T=16 run cheaply isolates horizon and credit assignment from action count and depth.
4. **Gate-0/VoI is missing for Exp 2B, 3A and 3B.** None of `study_exp2b.py`, `study_exp3b.py` or `study_reliability_sweep.py` calls the oracle or computes VoI. AGENTS §3 Gate 0 requires the VoI > 0 fraction and the fraction of tasks that require changing the opening action, per suite. Without these, the Exp 3B gains cannot be called *exploration* gains.
5. **Exploration/exploitation decomposition is missing outside Exp 1/2A.** Prefix-swap and restored-commitment exist only for Exp 1 and 2A (`layer1_k248.csv`, `study_exp2a.py`). Exp 2B/3B report welfare only, so the equivariant gain could be pure exploitation or coordination.
6. **Convention sharing in cross-seed meta×meta.** Exp 2B `coop_trap` +31.70 and the Exp 3B meta×meta results pair two agents trained on the same distribution. Shared conventions explain part of this (the zero-shot-coordination cross-play issue). For exploration claims, make the primary numbers focal return against fixed heterogeneous partners (UCB, Q).
7. **Structural-model asymmetries:**
   - It knows the number of types (`n_clusters = N_TYPES`), a mild structural prior that should be disclosed.
   - It is absent from Exp 3B and Exp 4, which are exactly the settings where it would arbitrate between "data insufficient" and "learner insufficient" (see §2).
8. **Permutation augmentation as the cleanest Exp 3B control.** No augmentation exists in `study_exp3b.py`. Under the coverage thesis (P3), an augmented One-Hot DRQN is a more diagnostic control than a capacity-matched One-Hot alone. Run both if compute allows.

## 5. Recommended priority (time-boxed to a ~Dec 8 freeze)

1. **Now, near-zero cost:** the eval-only surprise-threshold switch on existing checkpoints, and a VoI-under-corruption Gate-0 at d ∈ {0.3, 0.6}. Use new unsealed output paths.
2. **Exp 4B as specified in §2**, preceded by the R3 + distractor gate-validity check. This has the highest scientific leverage on the core research question.
3. **Minimal Exp 3B controls:** permutation-augmented and capacity-matched One-Hot, plus Gate-0 VoI fractions per suite. This protects the largest current positive claim, which is also the most confounded.
4. Pre-declare **one primary contrast per study** before running anything. Report every other contrast as exploratory, with k/5 seed counts alongside the bootstrap CIs.
