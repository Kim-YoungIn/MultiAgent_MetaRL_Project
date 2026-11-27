# TASK-004: Interim Cross-Audit of Exp 3A, Exp 3B, Exp 4, and Masking Fix

- **Task ID:** `TASK-004`
- **Assigned Role:** `research_critic` (`.codex/agents/research_critic.toml`)
- **Status:** `OPEN`
- **Dispatched By:** Lead Agent (`Antigravity`)

## Objective
Conduct an unbiased, rigorous **interim** scientific cross-audit of the newly completed studies (**Exp 3A**, **Exp 3B**, **Exp 4**) plus the post-`TASK-002` **strict masking fix** (`ac0766a`, `step_t > mask_k`) across Exp 1/2A/2B. Note: this is an **interim checkpoint**, not a final paper freeze.

Focus on:
1. **Masking Fix Verification (Exp 1 / 2A / 2B):** Confirm the corrected strict-masked contrasts in `results/core_v3_sealed/tables/paired_contrasts.csv`, `results/exp2a_online_opp/tables/paired_contrasts.csv`, and `results/exp2b_symmetric_ma/tables/paired_contrasts.csv`. Did any prior conclusion change?
2. **Exp 3A (`results/exp3a_reliability_sweep/tables/`):** Audit the $d \in \{0.0, 0.3, 0.6\}$ stress sweep (`R3` obs corruption & `coop_trap` slip). Are the `mt_drqn_aux_yr` vs `structural_model` vs `prior_only` claims and the `coop_trap` monotonic reversal (`+12.32 -> +2.47 -> -3.85`) properly bounded and free of confounds?
3. **Exp 3B (`results/exp3b_equivariant_semantic/tables/`):** Critically scrutinize the headline **`EquivariantDRQN`** breakthrough (`+12.63/+13.61/+12.68` over One-Hot; `+30.71` on unseen 5th family `dual_eq_battle`). Check potential confounds:
   - Parameter count / expressivity / architecture differences (`src/mamrl/agents/equivariant_drqn.py`)
   - Why `equiv_vanilla` tied `equiv_aux_yr` (`INCONCLUSIVE`)
   - Whether D-031 (`n2` vs `n4_b1` vs `n4_b2`) claims overreach (including `n2` beating `n4_b2` on `dual_eq_battle`)
4. **Exp 4 (`results/exp4_multibarrier_aux_discovery/tables/`):** Audit the 3-layer multi-barrier (`T=64`) scaling wall and the "Bootstrap Catch-22" interpretation of `drqn_aux_discovery`. Is the diagnosis scientifically airtight, or are there alternative confounds (e.g., hyperparameters, `meta_gate_lr`, replay buffer, `aux_scale`, `n_iters`) that should be explicitly caveated and tested next?
5. **Top 3 Next-Experiment Recommendations** before the final freeze (what single experiment would most sharpen the scientific story?).

## Required Reading (Token-Efficient)
- `AGENTS.md`, `PROGRESS.md`, `handoff/BOARD.md`
- `results/exp3a_reliability_sweep/tables/r3_obs_corruption_methods.csv`, `results/exp3a_reliability_sweep/tables/r3_obs_corruption_contrasts.csv`
- `results/exp3a_reliability_sweep/tables/coop_trap_slip_methods.csv`, `results/exp3a_reliability_sweep/tables/coop_trap_slip_contrasts.csv`
- `results/exp3b_equivariant_semantic/tables/methods.csv`, `results/exp3b_equivariant_semantic/tables/paired_contrasts.csv`
- `results/exp4_multibarrier_aux_discovery/tables/methods.csv`, `results/exp4_multibarrier_aux_discovery/tables/paired_contrasts.csv`
- Spot-check masking rows in `results/core_v3_sealed/tables/paired_contrasts.csv`, `results/exp2a_online_opp/tables/paired_contrasts.csv`, `results/exp2b_symmetric_ma/tables/paired_contrasts.csv`
- Inspect `src/mamrl/agents/equivariant_drqn.py` and `src/mamrl/study_exp4_multibarrier.py` only for confound checks (skip `runs/`).

## Allowed Write Paths
- `handoff/reviews/20261127_exp3_4_interim_audit.md`

## Forbidden Paths
- All other files (`AGENTS.md`, `docs/SPEC.md`, `PROGRESS.md`, `handoff/BOARD.md`, `src/`, `configs/`, `tests/`, sealed `results/`).

## Verification & Expected Report Output
1. Write a concise, evidence-anchored audit (`<= 120` lines) to `handoff/reviews/20261127_exp3_4_interim_audit.md` with sections: (1) Executive Verdict, (2) Masking Fix & Exp 3A Audit, (3) Exp 3B Equivariant/D-031 Confound Audit, (4) Exp 4 Scaling Wall & Catch-22 Critique, (5) Required Caveats & Top 3 Next Experiments.
2. Do NOT modify any code or run `git commit`.
