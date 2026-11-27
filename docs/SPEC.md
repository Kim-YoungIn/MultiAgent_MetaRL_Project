# Experimental Design Specification — Design Verification Draft (v2.2)

> **Status:** Architecture adopted (`RL²`-PPO + 3-Layer Exploration/Exploitation Decomposition); under final design verification before locking implementation and running experiments.
> **Guiding Principle:** Stay strictly anchored to the core research question. Every hypothesis is framed symmetrically so positive, conditional, and negative/null outcomes are equally informative. We never tune environments solely to make Meta-RL win.

---

## 1. Core Research Question & Scope

> **멀티에이전트 반복 게임 환경에서 메타 러닝이 탐색(exploration)을 개선하여 새로운 게임 규칙과 학습 상대에 대한 빠른 적응(fast adaptation)에 어떤 영향을 미치는가?**

### Two-Part Research Structure (Path A, aligned with Dec 17/20 Deliverables)
- **Part 1 (Completed — `docs/PRIOR_FINDINGS.md`, tag `archive/v0.9-legacy-studies`):**
  Empirical diagnosis of 15 prior studies showing why standard multi-agent meta-exploration setups failed to beat UCB or improve median returns: passive/greedy actions already leaked rules (only 6/96 pairs required changing the opening move), mean gains hid median losses (`+0.446` mean vs `−0.144` median), long-horizon gains were confounded with opponent shaping (`+0.365` counterpart return), and task-count scaling at fixed compute diluted per-task exposure.
- **Part 2 (Active Design Under Verification):**
  Controlled multi-agent study using an unprivileged **End-to-End `RL²` (Recurrent PPO+GAE)** meta-learner paired with a **3-Layer Exploration vs. Exploitation Decomposition** to test *when—if ever—meta-learning improves exploration itself*, cleanly separating (i) observability / exploration complexity, (ii) exploration vs. exploitation vs. zero-shot prior, (iii) pure information gain vs. opponent shaping, and (iv) task diversity vs. per-task exposure.

---

## 2. Neutral, Falsifiable Research Questions

### RQ1 — Observability & Exploration Complexity
*How does the relative adaptation performance—and specifically the exploration quality—of `RL²` (vs. Greedy, Random, Arm-Independent UCB, and Memory-Reset) vary across regimes of observability and structural coupling?*

We evaluate **four matched benchmark regimes** (same action count $n$, horizon $T$, and exploitation payoff scale, verified by Gate-0 exact Oracle before training):

| Regime | Rule Identification Structure | Gate-0 Oracle Property | What This Tests |
|---|---|---|---|
| **R0: Passive Deterministic** | Safe/greedy action deterministically leaks latent rule $\theta$ (mirrors legacy studies) | `VoI > 0`, `must_change = 0%` | Baseline control: confirms Greedy/UCB suffice when passive play reveals the rule. |
| **R1: Active Deterministic (1-step)** | Safe action is uninformative; 1 costly probe reveals $\theta$ deterministically | `VoI > 0`, `must_change = 100%` | Tests whether 1-step active necessity alone favors Meta-RL, or whether simple UCB still matches/beats it. |
| **R2: Active Noisy (i.i.d. Noise Control)** | Same as R1, plus symmetric Gaussian reward noise across actions | `VoI > 0`, `must_change = 100%` | **Anti-bias control:** tests whether pure stochasticity alone helps Meta-RL or hurts Meta-RL *more* than UCB via optimization variance. |
| **R3: Structured Partial Observability (Aliased / Multi-Step)** | $M \ge 4$ latent rule/opponent regimes over $n \ge 6$ actions; risky high-stakes actions incur steep mismatch penalties and are aliased/noisy, while low-cost probes (or 2-step feedback-contingent probing + opponent reaction) carry high-SNR cross-action clues | `VoI > 0`, `must_change = 100%`, 1-step probe insufficient | Tests the core hypothesis: when $(Q, V)$ cannot be solved independently per arm in 1 step, does **feedback-conditioned `RL²`** learn superior active exploration over UCB and open-loop exploration? |

### RQ2 — Multi-Agent Non-Stationarity: Information Gain vs. Opponent Shaping
*Against fixed reactive opponents vs. online learning opponents (`Q-learning` / `Expected SARSA`), how much of any adaptation gain is pure information acquisition versus shaping the opponent and game state?*

- Every regime is evaluated under **both** paired protocols on identical test tasks and seeds:
  1. **Controlled (Opponent-Restored):** Snapshot opponent & env state before the first $K$ turns; evaluate post-prefix exploitation with opponent restored to its initial state. Isolates pure information value.
  2. **Continuous Deployment (Both Agents Learning):** Keep live opponent state and online learning running over $T$ turns; log both **Focal Return** and **Counterpart Return** (`both_up`, `both_down`, action shift).

### RQ3 — Training Diversity (H1): Diversity vs. Exposure Dilution
*In the regime(s) where active exploration matters, how does increasing rule & opponent diversity (`Low / Medium / High`) affect held-out exploration quality and adaptation?*

- Explicitly separate **two compute controls**:
  1. **Fixed Total Budget:** Total training interactions constant while diversity increases (tests diversity vs. exposure dilution).
  2. **Fixed Per-Task Exposure:** Interactions per training task constant (isolates generalization effect of diversity).

---

## 3. Core Architecture: Unprivileged `RL²` + 3-Layer Exploration/Exploitation Decomposition

### A. Evaluated Agents (Strictly Unprivileged — Zero Access to Hidden `spec`)
All evaluated agents observe only `(prev_self_action, prev_opp_action, prev_self_reward, is_start, turn_frac)`:
1. **`rl2` (Primary Meta-Learner):** End-to-end Recurrent Actor-Critic (GRU + PPO with GAE value baseline) acting across all $T$ turns. No fixed-$K$ handoff constraint; learns *when* to explore, *when* to exploit, and *when* to re-explore if the opponent adapts.
2. **`rl2_masked` (Open-Loop Control):** Identical GRU + PPO architecture and training budget, but opponent action and reward feedback are masked during the early exploration window ($t < K_{\text{audit}}$) and unmasked afterward. Isolates **feedback-conditioned active exploration** from memorized open-loop probing sequences.
3. **`memory_reset` (Zero-Shot Prior Control):** Identical policy trained/evaluated with hidden state reset every turn (or feedforward MLP). Isolates **zero-shot prior / static policy** from within-task adaptation.
4. **`ucb` (Arm-Independent UCB):** Empirical per-action mean + count bonus (`mean_a + beta * sqrt(log(t+1)/(N_a+1))`), tuned strictly on dev seeds.
5. **`greedy` (Empirical Greedy):** Empirical per-action greedy baseline (no exploration bonus).
6. **`random_then_exploit` (Random Prefix Control):** Uniform random exploration for $K_{\text{audit}}$ turns, followed by greedy empirical exploitation (or neutral exploiter in prefix-swap).
7. **`oracle` (Diagnostic Ceiling Only — Privileged):** Exact Bayes-optimal policy with access to the task-family likelihood; used strictly to calibrate headroom, never as a baseline competitor.

### B. Solving the Exploration vs. Exploitation Decomposition Problem (Without OOD Bias)

A known pitfall of naive Prefix-Swap on `RL²` is **off-policy distribution shift**: if `RL²`'s GRU was trained only on its own trajectories, feeding it a `Random` or `UCB` prefix at test time could hurt `RL²` simply because the prefix is out-of-distribution (OOD), not because it contains less information.

To make Exploration vs. Exploitation attribution **airtight and free of OOD bias**, we apply **three complementary, cross-checked decomposition layers** at audit horizons $K \in \{2, 4, 8\}$:

| Layer | Mechanism | What It Measures | Confound Eliminated |
|---|---|---|---|
| **Layer 1: Neutral Bayes Info & Value Audit (Policy-Agnostic)** | Feed the exact first-$K$ trajectory $\tau_{1:K}$ collected by each method (`rl2`, `rl2_masked`, `ucb`, `random`, `greedy`, `oracle`) into an exact **policy-agnostic Bayesian posterior evaluator** over hidden $\theta$ / opponent type. Measure (a) posterior entropy $H(\theta \mid \tau_{1:K})$ and rule ID accuracy, and (b) Bayes-optimal post-prefix value $V^*(\tau_{1:K})$ (opponent-restored). | **Objective Exploration Quality** (how much decision-relevant information $\tau_{1:K}$ actually contains). | **Zero OOD bias** and **zero Student bottleneck**! |
| **Layer 2: Neutral Mixed-Prefix Exploiter (Model-Free Prefix-Swap)** | Train a separate frozen recurrent **Neutral Exploiter** on a *balanced mixture* of prefixes generated by `random`, `ucb`, `greedy`, and `rl2` (so no prefix distribution is OOD). At test time, swap only the first-$K$ prefix across methods and evaluate turns $K+1..T$ with this shared Neutral Exploiter (under both Opponent-Restored and Continuous settings). | **Model-Free Exploration Quality** (can an unprivileged learner convert $\tau_{1:K}$ into higher downstream reward?). | Separates Exploration from Exploitation **and** removes on-policy OOD preference for `rl2`. |
| **Layer 3: Exploitation & Belief Decoding Audit (`RL²` Self-Check)** | (a) Compare `rl2`'s own deployed return against `rl2` fed an `oracle` prefix (and against Layer 1 $V^*(\tau_{1:K})$) to measure **exploitation efficiency**. (b) Fit a lightweight post-hoc linear probe on `rl2`'s hidden state $h_t$ to measure how fast `rl2` internally decodes hidden $\theta$ and predicts the opponent's next action. | **Exploitation & Inference Efficiency** (did `rl2` actually use the information it gathered?). | Separates "failed to gather info" from "gathered info but failed to decode/exploit it". |

*(Optional ablation: include a Decoupled Teacher–Student arm on R3 to test whether explicit decoupling improves over end-to-end `RL²`.)*

---

## 4. Mandatory Metrics & Pre-Registered Decision Criteria

Every final experiment runs across **$\ge 5$ independent training seeds** on a **sealed test split** (disjoint from dev seeds), paired per game and reset, reporting:
- **End-to-End Performance:** Mean, **Median**, IQR, and **Win-Rate (%)** vs. `greedy`, `ucb`, `memory_reset`, and `rl2_masked`.
- **Exploration Quality (Layers 1 & 2):** Posterior entropy reduction $\Delta H(\theta \mid \tau_{1:K})$, rule ID accuracy, Neutral Bayes $V^*(\tau_{1:K})$, and Mixed-Prefix Neutral Exploiter return (Opponent-Restored & Continuous).
- **Adaptation Speed & Cost:** Early-turn AUC, Time-to-Threshold (with right-censoring rate), Exploration-window return, Total Return ($1..T$).
- **Multi-Agent Shaping Check:** Counterpart Return, Controlled vs. Continuous gap.

### Pre-Registered Interpretation Rules (Preventing Goalpost Moving)
- **Meta-Learning Truly Improves Exploration:** `rl2` beats `ucb`, `random`, and `rl2_masked` in **both** end-to-end Mean & Median Total Return **AND** in **Layer 1 (Neutral Bayes Info/Value)** + **Layer 2 (Mixed-Prefix Neutral Exploiter, Opponent-Restored)** (95% bootstrap CI > 0, win-rate > 50%).
- **Prior/Exploitation Gain Without Better Exploration:** `rl2` beats baselines in end-to-end Total Return, **but** fails to beat `ucb`/`random` on Layer 1 & Layer 2 → honest finding that meta-learning improved *exploitation/prior*, not *exploration*.
- **Shaping-Driven Multi-Agent Gain:** `rl2` gains in Continuous Deployment (+ Counterpart Return shift) while showing near-zero Controlled (Opponent-Restored) gain → honest finding that gains stem from *opponent shaping*, not pure information gathering.
- **Honest Null / Regime-Bounded Finding:** `rl2` loses/ties in R0/R1/R2 and succeeds only in R3 (or fails across all regimes, diagnosed via Layer 3) → reported transparently as the empirical boundary of meta-learned exploration.

---

## 5. Open Design Questions for Independent Verification (GPT / Critic Checklist)

Before freezing `src/mamrl/` and running final multi-seed experiments:
1. **R3 Game Design (Truly Multi-Agent + Tractable Oracle):** What is the cleanest 2-player repeated game for R3 where (a) 1-step probing is insufficient (needs 2-step feedback-contingent probing or risky vs. safe-probe discrimination), (b) the opponent's hidden type/learning genuinely interacts with both the probe signal and optimal exploitation (not a disguised single-agent bandit), and (c) Gate-0 exact Oracle & Layer-1 Neutral Bayes posterior remain fast and exact?
2. **Action-Label Permutation & Network Architecture:** Under `high` diversity (action relabeling), should `rl2` use a standard `GRU` or an **Action-Equivariant Attention / Set-GRU** (or compare both on dev) so permutation invariance isn't a hidden confound?
3. **Training Objective & Horizon:** Should `rl2` train strictly on full-episode undiscounted/discounted Total Return ($1..T$), and what $(T, K_{\text{audit}})$ cleanly separates exploration cost recovery from asymptotic saturation without blowing up compute?
4. **Scope vs. 2026-12-17 Deadline:** Is `R0–R3 × (Reactive vs. Q-Learning Opponent) × 3-Layer Decomposition + H1 (Fixed Budget vs. Fixed Exposure)` well-scoped for 20 days to the 5-min video + 5-page NeurIPS report, or should we prioritize a tighter core matrix?
