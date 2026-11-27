# TASK-005: Exp 3–4 Contract Audit

**Verdict: PARTIAL PASS.** The tested masking boundaries, equivariance, matrix separation, and Exp 4 Gate-0 checks passed. Two claims need qualification: the Exp 4 gate update is a cosine-alignment heuristic rather than the stated bilevel meta-gradient, and the table reconciliation did not cover every cell or verify row identities.

| Check | Result | Evidence and limitation |
|---|---|---|
| Masking | **PASS, scoped** | The PPO, Exp 2B, and Exp 4 encoders use `step_t > mask_k`; prior-only encoders hide feedback. Targeted PPO and Exp 2B boundary tests passed. Exp 3B uses the unmasked encoder and has no masked variant in its training path. |
| Auxiliary masking | **PASS, scoped** | Exp 2B and Exp 4 masked auxiliary targets begin at `mask_k`, excluding feedback generated on earlier turns. Exp 4 prior-only disables auxiliary weights. Exp 2B’s configured methods do not include a prior-only variant. |
| Non-leakage | **PASS, code inspection** | Inspected evaluated neural and online policy paths consume public observations. Reliability-sweep reset passes true type only when `policy.is_privileged`; privileged diagnostics remain separate. This is not an exhaustive runtime information-flow proof. |
| S₄ equivariance | **PASS** | The existing test exercised all 24 permutations for Q-values, opponent-prediction logits, and reward prediction. Random tie-breaking in action selection gives distributional symmetry; it does not guarantee identical actions under a fixed, unpermuted random stream. |
| Exp 3B matrix separation | **PASS** | The test found zero payoff-matrix overlap between the four-family training bank and each unseen suite; player-two matrices are transposes. |
| Exp 4 exact Gate 0 | **PASS, with scope limit** | Recomputed all 60 reported fields at T=6, 10, 16, 24, 48, and 64 against the sealed summary: zero mismatches. Optimal tiers were 0, 2, 3, 3, 3, 3; T=64 value was 120.625. The test also matched the T=64 DP value to vectorized-environment rollouts. This checks selected horizons, not every integer T=6–64 or an independent DP implementation. |
| Exp 4 gate mathematics | **FAIL for claimed meta-gradient** | [study_exp4_multibarrier.py](/home/archant/projects/MultiAgent_MetaRL_Project/src/mamrl/study_exp4_multibarrier.py:272) computes cosine similarities of validation TD and auxiliary gradients, then adds them to clipped gate logits. This is a gradient-alignment heuristic; it does not differentiate validation loss through an inner update or account for the softmax Jacobian and gradient magnitudes. Describe it accordingly or implement and verify the claimed bilevel update. |
| CSV completeness | **PASS, qualified** | No non-finite numeric values found across the 19 tables. Exp 2A has 256 blanks: 128 each in `both_up_rate` and `both_down_rate`, intentionally omitted for non-total-return metrics. They are not evidence of computed joint rates. |
| CSV ↔ JSON | **PARTIAL** | Anchor-based matching found zero discrepancies among **9,323 shared numeric fields** across **1,108 matched rows**, with absolute tolerance `5.1e-4` and relative tolerance `1e-5`. Matching used numeric anchors rather than full row keys; behavior, by-type, renamed, and other fields were not fully reconciled. Do not interpret this as a complete integrity pass. |
| Snapshot/restore | **PASS, targeted** | Existing Exp 2A snapshot/restore determinism and partial-mask test passed. |
| Submission budget | **PASS** | `git archive --format=zip HEAD` produced **743,382 bytes**, below 5 MB. Largest tracked working-tree file reported was 255,404 bytes. The archive measures `HEAD`, not uncommitted files. |
| Provenance | **REVIEW** | Core, Exp 2A, and Exp 2B summaries record `e65ccaa`, preceding masking-fix commit `ac0766a`, although that commit changed their sealed outputs. The recorded commit alone does not establish which code produced each updated fit or evaluation; verify receipts and checkpoint provenance before claiming all masked controls were retrained under the fix. |

**Validation:** Seven targeted read-only `unittest` checks passed. The initial invocation failed to import `mamrl` without `PYTHONPATH`; the corrected invocation was:

```bash
PYTHONPATH=src PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  tests.test_exp3b.TestEquivariantSemanticExp3B.test_exact_s4_permutation_equivariance_across_all_24_permutations \
  tests.test_exp3b.TestEquivariantSemanticExp3B.test_semantic_suites_strict_symmetry_and_zero_overlap \
  tests.test_v3_pipeline.TestMaskingAndCeilings.test_masking_and_prior_only_strip_feedback_strictly \
  tests.test_exp2b.TestSymmetricGamesExp2B.test_masking_strips_early_feedback_strictly \
  tests.test_exp2a.TestOnlineOpponentsV3.test_snapshot_restore_exactness_and_partial_mask \
  tests.test_multi_barrier.TestMultiBarrierGate0.test_exact_gate0_proves_horizon_dependent_barrier_progression \
  tests.test_multi_barrier.TestMultiBarrierGate0.test_vectorized_env_reproduces_exact_bayes_optimal_at_t64
```

Reproduce the archive measurement with `git archive --format=zip HEAD | wc -c`. The CSV reconciliation and six-horizon summary comparison were ad hoc read-only scripts; they were not saved, so their exact checks are not reproducible from a repository script yet.

**Recommended Lead actions:** qualify the gate-method claim; reconcile all tables by explicit row keys and mapped columns; verify masked-fit provenance. This report is a proposal for Lead review.