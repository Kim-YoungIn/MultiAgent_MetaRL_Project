# TASK-002 Contract Audit Report

**Verdict: FAIL on the masked-feedback contract; full verification remains incomplete.** The masked controls expose turn-4 feedback at the first post-prefix decision. The full test suite also could not complete because this execution environment has no writable temporary directory.

| Check | Result | Evidence |
|---|---|---|
| Masking and retrospective revelation | **FAIL** | `encode_obs_batch` and `encode_sym_obs_batch` allow previous-action and reward feedback when zero-based `step_t >= 4`. At `step_t=4`, that feedback is from turn 4, within the masked prefix. The first post-prefix decision therefore receives previously hidden feedback. `prior_only` continues to hide feedback throughout. DRQN auxiliary targets begin at index 4. |
| Unprivileged observations | **PASS, scoped inspection** | Delayed-response and symmetric observations contain public action/reward fields, not hidden type or payoff matrices. Exp 2A passes `true_theta` on reset only when `policy.is_privileged`. This is a code-path inspection, not a proof covering every possible caller. |
| Exact Gate-0 DP | **PASS, test assertions** | The R0, R1 and R3 exact-rational tests, including independent R3 benchmarks and horizon sensitivity, passed. This did not independently recompute every sealed Gate-0 artifact. |
| Opponent snapshot and restore | **PASS, with qualification** | The snapshot/restore and shared-prefix tests passed. A full restore from a supplied snapshot restores Q-table and RNG state; a masked restore restores selected Q-tables only. `restore(snap=None)` restores initial Q-tables without resetting RNG. |
| Symmetric rule splits | **PASS** | Payoff-matrix overlap: `0` for both games. Competition has 4 train and 4 test rules, with each anchor appearing once per split. Cooperation has 16 train and 8 test rules; each role appears 4 and 2 times respectively in each action slot. |
| CSV/JSON consistency | **PASS, sampled fields** | At four-decimal tolerance, all checked paired-contrast fields matched: Core 1,344; Exp 2A 1,536; Exp 2B 864. Checked Core method fields: 210; Exp 2B matchup fields: 448; zero mismatches. Other columns and tables remain unchecked. |
| Full test suite | **INCOMPLETE** | `Ran 30 tests in 7.008s`: 25 passed and 5 errored because no usable temporary directory was available. The errors do not establish application failures, but runner and resume checks remain unverified. |
| Submission size | **PASS for HEAD** | `git archive --format=zip HEAD | wc -c` returned **653,164 bytes**, below 5 MB. This measures the committed HEAD archive, not uncommitted working-tree changes or a separately assembled final submission. |

**Recommended Lead action:** Correct and test the masked boundary so all feedback generated during turns 1–4 remains hidden, then assess whether affected sealed results require rerunning. Rerun the full suite in an environment with writable temporary storage and complete the remaining CSV/JSON checks before declaring the audit complete.

**Reproduction commands**

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src \
.venv/bin/python -m unittest discover -s tests -v

git archive --format=zip HEAD | wc -c

rg -n 'allow_feedback|start_t =|snapshot|def restore' \
  src/mamrl/agents/{ppo_rl2,drqn_rl2,opponents_v3}.py \
  src/mamrl/study_exp2b.py
```

The archive was streamed to the byte counter because the filesystem was read-only; no temporary archive or report file was written.