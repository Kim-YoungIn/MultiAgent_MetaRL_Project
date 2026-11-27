# Experimental Design Specification — Design Verification Draft (v2.1)

> **Status:** Under design verification before locking implementation and running final experiments.
> **Guiding Principle:** Stay strictly anchored to the core research question. Every hypothesis below is framed symmetrically so positive, conditional, and negative/null outcomes are equally informative. We do **not** tune environments solely to force Meta-RL to win.

---

## 1. Core Research Question & Scope

> **멀티에이전트 반복 게임 환경에서 메타 러닝이 탐색(exploration)을 개선하여 새로운 게임 규칙과 학습 상대에 대한 빠른 적응(fast adaptation)에 어떤 영향을 미치는가?**

### Two-Part Research Structure (Path A, aligned with Dec 17/20 Deliverables)
- **Part 1 (Completed — `docs/PRIOR_FINDINGS.md`, tag `archive/v0.9-legacy-studies`):**
  Empirical diagnosis of 15 prior studies showing why standard multi-agent meta-exploration benchmarks failed to beat UCB or improve median returns: passive/greedy actions already leaked rules (only 6/96 pairs required changing the opening move), mean gains hid median losses (`+0.446` mean vs `−0.144` median), long-horizon gains were confounded with opponent shaping (`+0.365` counterpart return), and task-count scaling at fixed compute diluted per-task exposure.
- **Part 2 (Active Design Under Verification):**
  Controlled multi-agent study testing **when—if ever—meta-learned exploration outperforms standard exploration**, cleanly separating (i) observability / structural complexity, (ii) pure information gain vs. opponent shaping, and (iii) task diversity vs. per-task exposure.

---

## 2. Neutral, Falsifiable Research Questions

### RQ1 — Observability & Exploration Complexity
*How does the relative adaptation performance of meta-learned exploration (vs. Greedy, Random, and Arm-Independent UCB) vary across regimes of observability and structural coupling?*

We evaluate **four matched benchmark regimes** (same action count $n$, horizon $T$, and exploitation reward scale, verified by Gate-0 exact Oracle before training):

| Regime | Rule Identification Structure | Expected Gate-0 Oracle | What This Tests |
|---|---|---|---|
| **R0: Passive Deterministic** | Greedy/safe action deterministically leaks latent rule $\theta$ (low complexity, mirrors legacy studies) | `VoI > 0`, `must_change = 0%` | Baseline control: confirms Greedy/UCB suffice when passive play reveals the rule. |
| **R1: Active Deterministic (1-step)** | Safe action is uninformative; 1 costly probe reveals $\theta$ deterministically | `VoI > 0`, `must_change = 100%` | Tests whether 1-step active necessity alone favors Meta-RL, or whether simple UCB still matches/beats it. |
| **R2: Active Noisy (i.i.d. Noise Control)** | Same as R1, plus symmetric Gaussian reward noise across actions | `VoI > 0`, `must_change = 100%` | **Anti-bias control:** tests whether pure stochasticity alone helps Meta-RL or hurts Meta-RL *more* than UCB via policy-gradient variance. |
| **R3: Structured Partial Observability (Aliased / Multi-Step)** | $M \ge 4$ latent rule/opponent regimes over $n \ge 6$ actions; risky high-stakes actions incur steep mismatch penalties and are aliased/noisy, while low-cost probes (or 2-step feedback-contingent probe pairs + opponent reaction) carry high-SNR cross-action information about unvisited actions | `VoI > 0`, `must_change = 100%`, 1-step probe insufficient | Tests the core hypothesis: when $(Q, V)$ cannot be solved independently per arm in 1 step, does **feedback-conditioned Meta-Exploration** outperform arm-independent UCB and open-loop Masked exploration? |

### RQ2 — Multi-Agent Non-Stationarity: Information Gain vs. Opponent Shaping
*Against fixed reactive opponents vs. online learning opponents (`Q-learning` / `Expected SARSA`), how much of any adaptation gain is pure information acquisition versus shaping the opponent and game state?*

- Every regime is evaluated under **two paired protocols** on identical test tasks and seeds:
  1. **Controlled (Opponent-Restored, Frozen Assessment):** Snapshot opponent & env state before exploration; after $K$ exploration turns, keep *only* the focal Student's updated state, restore opponent/env to pre-exploration snapshot, and evaluate. Isolates pure information value.
  2. **Continuous Deployment (Both Agents Learning):** Keep live opponent state and online learning running over $T$ turns; log both **Focal Return** and **Counterpart Return** (`both_up`, `both_down`, action shift).

### RQ3 — Training Diversity (H1): Diversity vs. Exposure Dilution
*In the regime(s) where active exploration matters, how does increasing rule & opponent diversity (`Low / Medium / High`) affect held-out adaptation?*

- Explicitly separate **two compute controls**:
  1. **Fixed Total Budget:** Total training interactions held constant while task diversity increases (mirrors realistic compute cap, tests diversity vs. exposure dilution).
  2. **Fixed Per-Task Exposure:** Interactions per training task held constant (isolates generalization benefit of diversity from under-training).

---

## 3. Fair Baselines, Learners & Controls (Anti-Leakage Design)

A critical requirement from our design audit: **no baseline or learned explorer may receive privileged access to hidden task specs (`spec` / payoff tables).**

### A. Unprivileged Evaluated Methods (Primary Comparison)
All methods share the **exact same observation interface** `(prev_self_action, prev_opp_action, prev_self_reward, is_start)` and **identical Exploitation Student** (unless end-to-end):
1. **Greedy (`greedy`)**: Student plays greedily from turn 0 (no separate exploration).
2. **Random (`random`)**: Uniform random actions for $K$ turns, then Student exploits.
3. **Arm-Independent UCB (`ucb`)**: Empirical per-action mean + count uncertainty bonus (`mean_a + beta * sqrt(log(t+1)/(N_a+1))`) for $K$ turns (tuned *only* on dev seeds). Does **not** know cross-action latent structure.
4. **Meta-Explorer — Full Feedback (`meta_full`)**: Recurrent GRU exploration policy trained across training tasks (optimizing total return including exploration cost, and checked against assessment-only objective on dev).
5. **Meta-Explorer — Masked Feedback (`meta_masked`)**: Identical architecture, optimizer, and training budget as `meta_full`, but opponent action & reward feedback are masked after turn 0. Isolates **adaptive feedback-conditioned probing** from memorized open-loop action sequences.
6. **End-to-End Recurrent Meta-RL (`rl2`)**: Single GRU policy acting for all $T$ turns without explicit Teacher–Student split (connects directly back to the original proposal and checks whether decoupling helps or hurts).

### B. Exploitation Student (Model-Free Primary + Diagnostic Ceiling)
- **Primary Student (Unprivileged, Fair):** Fast empirical online learner (e.g., contextual / action-value ridge or tabular Q learner **without** privileged task tables, OR a shared meta-learned / empirical-Bayes prior fitted strictly on training tasks—identical across `greedy`, `random`, `ucb`, `meta_full`, `meta_masked`).
- **Diagnostic Ceiling Only (Privileged):**
  - **Privileged Bayesian Student** (uses true task-family likelihood) + **Oracle Explorer**: reported *strictly* as an upper-bound diagnostic to separate "Explorer failed to collect info" from "Student failed to use info."

---

## 4. Mandatory Metrics & Pre-Registered Decision Criteria

Every evaluation runs across **$\ge 5$ independent training seeds** on a **sealed test split** (disjoint from dev tuning seeds), paired per game and reset, reporting:
- **Focal Performance:** Mean, **Median**, IQR, and **Win-Rate (%)** vs. `greedy`, `ucb`, and `meta_masked`.
- **Adaptation Speed:** Early-assessment AUC, Time-to-Threshold (with explicit right-censoring rate).
- **Economic Cost:** Exploration Return ($1..K$), Assessment Return ($K+1..T$), and **Total Return** ($1..T$).
- **Multi-Agent Shaping Check:** Counterpart Return, Controlled vs. Continuous gap.

### Pre-Registered Interpretation Rules (Preventing Goalpost Moving)
- **Evidence that Meta-Exploration genuinely accelerates adaptation:** `meta_full` outperforms `greedy`, `ucb`, AND `meta_masked` in **both Mean and Median Total Return** (95% seed/block bootstrap CI > 0, win-rate > 50%) **and** retains positive Controlled (Opponent-Restored) gain.
- **Evidence of Conditional / Regime-Bounded Utility:** `meta_full` beats baselines in R3 (Structured Partial Observability) while matching/losing to `ucb` in R0/R1/R2 — establishing *when* meta-exploration is worth its cost.
- **Honest Null / Negative Finding:** If `meta_full` fails to beat `ucb` or `meta_masked` even in R3 (while Oracle succeeds), diagnose via the Privileged-Student ceiling whether the bottleneck is **meta-optimization/credit assignment** vs. **Student representation**, and report it transparently.

---

## 5. Open Design Questions for Independent Verification (GPT / Critic Checklist)

Before freezing `src/mamrl/` for final multi-seed execution, we explicitly seek critique on:
1. **Student fairness vs. bottleneck:** In R3 (cross-action structural coupling), an arm-independent empirical Student cannot infer unvisited action B's value from probing action A unless the Student itself has cross-action generalization (e.g., meta-learned / shared-prior Student or end-to-end `RL²`). How should we pair `Meta-Explorer` with a Student in R3 so **both** (a) UCB gets a fair comparison and (b) cross-action probe information can actually be exploited?
2. **Design of R3 (`Structured Partial Observability`):** What is the cleanest, simplest 2-player repeated game where (a) 1-step probing is insufficient or risky, (b) the opponent's action/learning genuinely matters (not a disguised single-agent bandit), and (c) exact Gate-0 Oracle remains tractable?
3. **Training algorithm stability:** Is baseline-reduced REINFORCE sufficient for `meta_full` / `meta_masked` / `rl2`, or should we use PPO with GAE/value baseline to prevent high-variance failure in R2/R3?
4. **Scope vs. Dec 17 Deadline:** Given 20 days until the 5-min video + quiz (Dec 17) and 23 days until the 5-page NeurIPS report (Dec 20), is the `R0–R3 × Controlled/Continuous × H1` matrix appropriately scoped, or should any axis be tightened?
