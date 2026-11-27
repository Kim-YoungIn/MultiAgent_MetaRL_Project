# TASK-003: Deep Results Interpretation & Legacy-to-v3 Verification Gap Analysis

- **Task ID:** `TASK-003`
- **Assigned Role:** `results_analyst` (`.codex/agents/results_analyst.toml`)
- **Status:** `OPEN`
- **Dispatched By:** Lead Agent (`Antigravity`)

## Objective
Mine the sealed Exp 1 / Exp 2A / Exp 2B CSV tables to uncover non-obvious behavioral/mechanistic patterns and design concrete clean-v3 experiments that directly test the **unverified legacy hypotheses** from `docs/PRIOR_FINDINGS.md` (v0.1–v0.9).

## Required Reading (Compact Files Only — Token Budget)
- `AGENTS.md`, `PROGRESS.md`, `docs/PRIOR_FINDINGS.md`, `handoff/BOARD.md`
- `results/core_v3_sealed/tables/{methods,paired_contrasts,by_theta,behavior_repetition}.csv`
- `results/exp2a_online_opp/tables/{methods,paired_contrasts,restore_vs_continuous,behavior_repetition}.csv`
- `results/exp2b_symmetric_ma/tables/{matchups,paired_contrasts}.csv`

## Analysis Questions
1. **Legacy-to-v3 Coverage Matrix:** Map all 5 legacy failure modes/hypotheses (`docs/PRIOR_FINDINGS.md` §2: Mean-vs-Median skew, Baseline loss & feedback conditioning, Passive vs Active VoI, Opponent shaping vs dynamics shift $d$, and D-031 Diversity vs Per-Rule Exposure dilution + `adapt_v1` concordance reversal) to what is ALREADY verified in v3 vs. what still needs a targeted clean v3 experiment.
2. **Hidden Patterns & Anomalies in Exp 1 / 2A / 2B:**
   - Why did `mt_drqn_vanilla` beat `mt_drqn_aux_yr` on `expected_sarsa_unseen` (`+4.24` vs `+3.29`)?
   - Why did `meta_aux_yr` lose to `q_learner` in `comp_rps` even `in_dist` (`-0.98` margin) while `meta_vanilla` won (`+0.83`), yet `meta_aux_yr` dominated `coop_trap` (`+31.70` vs `+20.70`)?
   - What explains the `Meta vs Non-Meta` negative margin (`-1.83 / -1.15`) alongside huge positive social welfare (`+6.22 / +15.91`) in `coop_trap`?
3. **Top 3 Next Experiments (Ranked by Scientific Leverage per Compute Cost):**
   Give concrete, compact v3 experiment proposals (e.g., D-031 diversity-vs-exposure in `coop_trap`, dynamics/reliability shift $d$ in `R3`/`coop_trap`, `comp_rps` balanced rule split or stochastic policy, and Auxiliary Task Discovery) with explicit falsification criteria.

## Allowed Write Paths
- `handoff/reports/TASK-003_legacy_bridge_and_insights.md`

## Forbidden Paths
- All other files (`src/`, `configs/`, `tests/`, `results/`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`).

## Expected Report Output
Write a concise, evidence-grounded report (`<= 120` lines) to `handoff/reports/TASK-003_legacy_bridge_and_insights.md`. Do not run `git commit`.
