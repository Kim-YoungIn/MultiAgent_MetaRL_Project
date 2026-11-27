# Active Experiment Specification (Path A — v2 Clean Core)

## 1. Central Thesis & Experimental Structure

**Thesis:** Meta-learning improves multi-agent exploration and fast adaptation *if and only if* (1) optimal exploitation depends on hidden rule/opponent structure that passive/greedy actions cannot identify (**Active Probing Necessity**), (2) the explorer conditions on interaction feedback rather than memorizing fixed open-loop sequences, and (3) training diversity covers the structural probe-to-rule mapping without starving per-task exposure.

We test this in two tightly coupled experiments runnable on a single machine well before the **2026-12-17** deadline:

- **Exp 1 — Necessity Gate (Passive vs. Active Regime):** Contrast two matched multi-agent game distributions that share identical horizon, action count, and exploitation payoff scale, differing only in whether greedy play already reveals the hidden regime (`Passive`, Oracle `must_change_opening ≈ 0%`) or strictly hides it behind a costly probe (`Active`, Oracle `must_change_opening = 100%`).
- **Exp 2 — H1 Diversity × Exposure & Opponent Dynamics:** Within the `Active` regime, vary training task diversity (`Low / Medium / High`) under **both Fixed Total Budget and Fixed Per-Task Exposure**, against **Fixed Reactive Opponents** and **Online Q-Learning Opponents**, evaluating both **Opponent-Restored (pure information)** and **Continuous Deployment**.

---

## 2. Benchmark Environment: `ActiveProbeGame`

A clean 2-player repeated strategic game (`src/mamrl/envs/active_probe_game.py`) designed specifically to pass **Gate 0 (Oracle Verification)**:

- **Action Space:** $n = 4$ (configurable) discrete actions per player per turn:
  - `SAFE` (1 action): Low-variance baseline payoff $r_{\text{safe}}$, **uninformative** about the hidden regime in the Active mode.
  - `EXPLOIT_A`, `EXPLOIT_B` (2 actions): High-stakes strategic actions whose optimal choice depends on the hidden task regime $\theta \in \{0, 1, \dots, M-1\}$ **and** the opponent's action/type. Choosing the wrong exploit action incurs a heavy mismatch penalty; choosing the right one yields high reward.
  - `PROBE` (1 action, or multiple typed probes in higher diversity): Pays an explicit immediate exploration cost $-c_{\text{probe}} < 0$, and emits a diagnostic payoff/reaction signal depending on $(\theta, b_t)$ — i.e., **the opponent's response/action jointly shapes whether the probe succeeds and what signal is observed**.
- **Action-Label Permutation (Diversity & Anti-Memorization):**
  Each task instance samples a latent rule $\theta$, a probe-cost / payoff parameterization, and (optionally in `Medium`/`High` diversity) a permutation or structural mapping of probe-to-exploit roles so the agent cannot hardcode a single action index.
- **Observation (No Leakage):**
  At turn $t$, each player observes only `(prev_self_action, prev_opp_action, prev_self_reward, step_start_flag)` (or turn index if enabled). Hidden $\theta$, opponent identity, and horizon remain unobserved.
- **Passive vs. Active Mode Toggle:**
  - `mode="active"`: `SAFE` returns constant $r_{\text{safe}}$ regardless of $\theta$; only `PROBE` (or risky trial-and-error) separates $\theta$. Exact Oracle verifies `VoI > 0` and `must_change_opening = True`.
  - `mode="passive"`: `SAFE` leaks $\theta$ through a small diagnostic payoff signature without needing a costly probe. Exact Oracle verifies `VoI > 0` and `must_change_opening = False`.

---

## 3. Agents & Controls

All methods share the **exact same Exploitation Student** after an exploration budget $K$ (e.g., $K \in \{2, 4, 8\}$, total horizon $T = 32$ or $48$):

1. **Exploitation Student (`src/mamrl/agents/students.py`):**
   - Primary: **Bayesian Belief / Linear-Reward Student** (fast, zero-bottleneck posterior update over task hypotheses or action-context reward estimates — avoids the tabular Q-state sparsity bottleneck diagnosed in v0.4).
   - Secondary check: **Online Q-Learning Student** (to verify robustness across learner classes).
2. **Opponents (`src/mamrl/agents/opponents.py`):**
   - `reactive`: Strategic opponent whose response depends on its hidden type and focal action (clean multi-agent interaction).
   - `q_learning`: Online tabular Q-learning opponent that adapts simultaneously (tests non-stationarity and shaping).
   - Every opponent supports `snapshot()` / `restore()` for controlled counterfactual evaluation.
3. **Exploration Policies (`src/mamrl/agents/explorers.py`):**
   - `greedy`: No separate exploration (Student plays greedily from turn 0).
   - `random`: Uniform random exploration for $K$ turns.
   - `ucb`: Uncertainty-bonus exploration over Student estimates for $K$ turns.
   - `meta_full`: Meta-learned recurrent (GRU) Explorer trained via policy gradient / PPO on held-in training tasks to maximize post-exploration adaptation return (or total return incl. exploration cost).
   - `meta_masked`: Identical architecture and training budget as `meta_full`, but **reward/opponent feedback is masked** after turn 0 (isolates true feedback-conditioned exploration from open-loop sequence memorization).
   - `oracle`: Exact belief/VoI reference upper bound (not a competitor; calibrates benchmark headroom).

---

## 4. Mandatory Metrics & Evaluation Protocol

Every run (`src/mamrl/runner.py`) pairs all methods on the exact same held-out test games, initial seeds, and opponent states, reporting:

1. **Distribution-Aware Performance:** Mean, **Median**, IQR, and **Win-Rate (%)** vs. `greedy` and `ucb`.
2. **Adaptation Speed:** Early-turn AUC (first $H_{\text{early}}$ assessment turns) and **Time-to-Threshold** (with right-censoring logged explicitly).
3. **Full Economic Cost:** Exploration Return (turns $1..K$), Assessment Return (turns $K+1..T$), and **Total Return** (turns $1..T$).
4. **Multi-Agent Shaping Decomposition:**
   - **Controlled Evaluation:** Snapshot opponent state before exploration, restore opponent state after exploration (keeping only Student posterior), freeze both learners, measure pure information gain.
   - **Continuous Deployment:** Keep real opponent state & online learning running, and report **Counterpart Return** alongside Focal Return.

---

## 5. Pre-Registered Milestones Toward Dec 17/20

- **M1 (Done now):** Clean core architecture, Oracle gate, unit tests, and GPT handoff pipeline.
- **M2 (Nov 28–Dec 1):** Run Gate 0 + Exp 1 (Passive vs. Active Necessity, multi-seed) and review with GPT (`handoff/reviews/`).
- **M3 (Dec 2–Dec 7):** Run Exp 2 (H1 Diversity × Exposure × Learning Opponent + Shaping Decomposition).
- **M4 (Dec 8–Dec 13):** Final locked multi-seed runs, publication figures, and 5-page NeurIPS report draft.
- **M5 (Dec 14–Dec 17):** Slides, 5-min video, 3 quiz questions, and <5MB code submission zip check.
