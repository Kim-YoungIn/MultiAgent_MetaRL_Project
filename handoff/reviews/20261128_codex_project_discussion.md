# TASK-007 — Project-wide scientific position

**Date:** 2026-11-28 · **Role:** `research_critic`  
**Status:** Proposal for Lead review; no repository files modified.  
**Evidence:** Required governance/context documents, TASK-006, all nine requested CSVs, `stage_lock.py`, and targeted portions of both study drivers. No `runs/` or full `summary.json` inspected.

## 1. Scientific answer

**Recurrent meta-RL can learn useful costly probing, but its advantage depends on the information structure, training distribution, representation, and optimization. The evidence supports conditional adaptation benefits; it does not establish general exploration superiority or auxiliary-task necessity.**

- **Exp 1 remains the strongest exploration result:** `aux_yr` approaches the exact R3 ceiling and improves audited information value over feedback controls. Its tie with the structural learner establishes successful amortization, while the PPO budget imbalance limits algorithm comparisons.
- **Exp 2A/3A establish deployment boundaries:** matched training helps against learning opponents, yet observation shifts can produce harmful commitment. Similar auditor entropy plus greater wrong-arm exploitation supports an exploitation/calibration interpretation, without proving the policy’s internal mechanism.
- **Exp 5A-1A rules out “nothing useful remains to learn” at high corruption:** the achievable PBVI value is **12.00**, versus the blind **4.80**, so feedback benefit is **at least 7.20**. Rewards remain uncorrupted; this is not evidence that noisy opponent symbols alone retain that value.
- **Exp 5A-1D weakens a coverage-only explanation:** equal-budget random data contain positive events and support successful structured planning. This demonstrates informational feasibility, but does not isolate model-free credit assignment from replay retention, representation, optimization, or planning differences.
- **Exp 4B demonstrates feasibility in a redesigned sequential environment:** vanilla reaches mean **67.27**, median **72.20**, and final-stage unlock rate **79.1%**. Auxiliary methods do not reliably improve on vanilla.
- **Exp 2B/3B support coordination and transfer claims:** their welfare gains do not independently establish improved information acquisition. Exp 3B’s architectural-package interpretation remains necessary pending controls.

## 2. New-result audit and required qualifications

### A. Recurrent bootstrap history mismatch — resolve before sealing Exp 4B

In `study_exp4b_compact.py:344`, both next-action selection and target evaluation run on `tx[:, 1:, :]` with a fresh recurrent state. Current-state values use the full episode sequence. The next-state target therefore omits the episode’s initial observation.

An in-memory unittest comparing full-history and shifted-history Q-values **failed**, with maximum discrepancy **0.04878** for a deterministic initialization. This establishes a history mismatch; its effect on trained returns remains unmeasured.

**Lead action:** validate the intended recurrent Bellman target using full-history unrolls or equivalent carried states. Preserve existing results and record any correction separately. This finding concerns the inspected Exp 4B trainer; I have not established the same defect in Exp 4.

### B. Structural-model results are useful but require narrower wording

- **22-action arbitration:** seed means are **120.625, 31.15, 31.15, 120.625, 120.625**; their overall mean is **84.835**. Each seed’s median is **120.55**, but only **3/5** reach the optimal mean.
- `tier3_rate_late=1.0` counts **Tier-3 actions**, not successful confirmations (`study_diagnostics_v3.py:774`). It cannot alone establish successful Tier-3 exploitation on every trajectory.
- Diag 1D **generates fresh uniform-action episodes** with the warmup budget. It does not load the original DRQN replay. Say “equal-budget warmup-distribution data,” unless trajectory identity is independently verified.
- **Stage-Lock structural planning:** emissions and rewards are learned, but the planner receives the transition structure: three stages, unlock detection at `r>0.25`, the meaning of `P`, four latent clusters, and belief reset after crossing (`study_exp4b_compact.py:589`, `:649`).
- That structure is reconstructible from public history, and I found no hidden-key input in the inspected policy path. Nevertheless, DRQN must learn the corresponding state representation and transitions. Label this a **structure-informed learned-model planner**, and disclose its 2,048 offline episodes and planning cost.

### C. Gate-0 additions establish opportunity more strongly than necessity

- PBVI supplies achievable **lower bounds**. A positive difference between two approximate opening values does **not** prove their exact ordering. The `must_change_opening` flag at `study_diagnostics_v3.py:304` needs a bound separating the candidate opening from the exact best myopic opening.
- Thus, retain **positive feedback-value lower bound** at noisy R3; qualify exact opening necessity until certified.
- Diag 1C solves a finite-bank, **uniform-partner**, short-horizon diagnostic. Positive VoI there does not demonstrate that learned policies acquire better information against Q-learning, UCB, or another meta-policy.
- All four actions are myopically tied in those suites. Strictly leaving the myopic-optimal action set is therefore impossible. Positive VoI must not be conflated with R3-style costly-opening necessity.
- `positive_voi_rule_frac_h2` is a conditional policy-advantage calculation using posterior tie-breaking and averaged openings. Its “100%” should be described under that diagnostic definition.
- These are retrospective audits; they do not retroactively satisfy the original requirement to verify Gate 0 before training.

### D. Auxiliary selection succeeds conditionally

| Setting | Distribution and comparison | Interpretation |
|---|---|---|
| R3 distractors | Discovery **13.69**, median **15.55**, IQR **3.05**; versus vanilla **+7.50**, CI **[5.51, 9.57]**, **5/5** positive seeds, **75%** episode wins | Supports the registered gate-validity contrast; **95.3%** final weight goes to `(y,r)` |
| R3 distractors | Discovery minus uniform **−0.435**, `INCONCLUSIVE` | Useful-signal weighting does not establish superiority over uniform auxiliaries |
| Stage-Lock | Discovery **59.17**, median **62.15**, IQR **32.80**; versus uniform **+23.52**, CI **[7.28, 40.14]**, **5/5** positive seeds, **74.4%** wins | Supports the registered bootstrap contrast; selection mechanism remains unisolated |
| Stage-Lock | Discovery minus vanilla **−8.10**; `aux_yr` minus vanilla **−3.07**; both `INCONCLUSIVE` | No reliable auxiliary advantage over pure-reward DRQN |

The Stage-Lock discovery–uniform **seed-level t-interval is [−2.54, 49.57]**, showing sensitivity to inference with five seeds. Keep the registered bootstrap verdict, accompanied by this sensitivity and seed counts. Five positive signs alone yield two-sided sign-test \(p=0.0625\).

Early AUC also prevents a blanket “faster adaptation” claim: R3 discovery is **−0.204** versus vanilla **+0.113**; Stage-Lock discovery is **−0.013** versus vanilla **−0.011**. These windows include purposeful exploration costs.

Counterpart returns are **55.85 versus 49.70** on R3 and **114.99 versus 115.24** on Stage-Lock, discovery versus vanilla. With fixed honest counterparts, their success bonuses are mechanically coupled to focal exploitation; they are not independent evidence of opponent shaping.

### E. Surprise falsification and environment redesign

The switch improves high-noise mean **0.44→2.00**, median **−0.50→4.20**, but remains **2.80 below** the blind policy. This fails the registered escalation criterion: **stop surprise retraining**. It does not logically falsify every possible GRU-surprise method.

Diag 1B’s SARSA raw return is **11.60**, versus **3.29** in the earlier synthesis’s noisy unseen condition. Reconcile configurations before describing this as a replication of that failure.

Stage-Lock changes action count, horizon, penalties, latent structure, and progression mechanics together. Replace “22→6 actions breaks the wall” with **“a compact sequential redesign permits higher-tier learning.”** Action count has not been isolated causally.

## 3. Single final experiment block

**Choose Exp 3B representation and feedback controls.** This protects the strongest new-game transfer claim, while further barrier variants would add scope without resolving its attribution.

- Run One-Hot **capacity × permutation augmentation** controls: original versus enlarged recurrent capacity, each without/with consistent joint relabeling of action-related inputs, targets, and outputs.
- Match the equivariant model’s **128 recurrent scalars** in the enlarged control; report parameter counts separately. A dense GRU generally cannot simultaneously match both recurrent state and parameter count.
- Reuse the existing original-capacity/unaugmented cell where protocol-compatible: three new full-feedback cells × five seeds.
- Add equivariant **masked-feedback and prior-only** controls: ten further fits. Total proposed addition: **25 fits**, with identical transition budgets and update schedules.
- Register one primary contrast: **equivariant versus augmented, state-matched One-Hot focal return against UCB on the unseen family**. Report welfare, median/IQR, early AUC, counterpart return, episode win/tie/loss, and seed signs alongside it.
- If augmentation/capacity closes the gap, revise the architectural claim accordingly. If the gap persists, support an architectural inductive-bias benefit; call it improved exploration only with direct information-value evidence.
- Tune exclusively on development data, freeze endpoints before execution, and disclose that these controls follow inspection of the existing suites.

**Stop:** surprise retraining, new barrier/environment variants, full D-031 expansion, and additional auxiliary searches. Resolve the target-history issue and reporting qualifications as integrity work; preserve all earlier negative findings. Freeze by approximately December 8.

## 4. Three strongest reviewer attacks

| Attack | Pre-emptive defense |
|---|---|
| **“You redesigned toy tasks after failures and attribute return gains to exploration.”** | Make Exp 1’s exact information-value audit the primary causal evidence. Present Stage-Lock as a disclosed redesign, symmetric welfare as coordination/transfer, and retrospective diagnostics as exploratory. |
| **“Capacity, supplied structure, and unequal budgets explain the wins.”** | Execute the Exp 3B controls; disclose structural transition priors and offline/planning costs; retain the PPO budget caveat. Separate absence of hidden-task leakage from equality of inductive assumptions. |
| **“Five seeds, many contrasts, and implementation/provenance gaps make the headlines fragile.”** | Resolve recurrent target alignment; predeclare the final primary contrast; report seed-level sensitivity, distributions, ties, and failures. Correct the 22-action mean and distinguish reconstructed warmup data from original replay. |

## 5. Position relative to TASK-006

I agree with Claude’s conditional-amortization framing and warning about harmful OOD commitment. I disagree with treating **coverage as the single unifying cause**: structured inference can exploit equal-budget random data, while the remaining neural failure mechanisms have not been separated experimentally.

The defensible thesis is: **useful information can exist without reliable learned adaptation; representation, planning, and optimization determine whether policies convert that information into cost-inclusive return.** Auxiliary selection helps in some comparisons, but is neither necessary nor uniformly beneficial here.

**Validation:** In-memory `unittest`: **3 passed, 1 failed**—exact Stage-Lock ceilings, 864 scalar/vector transition cases, and table arithmetic passed; recurrent bootstrap-history alignment failed as described above. No edits or commits performed.