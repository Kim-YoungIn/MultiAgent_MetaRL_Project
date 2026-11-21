# Research status, 2026-11-21

Current objective and stage
- Completed a fresh, matched comparison of absolute cumulative return, absolute terminal payoff and gain-only training for exploration in new games.
- Main result, figures and reproduction commands in research/absolute_exploration/REPORT.md.

Verified outcomes
- 24 blocks, 6,144 held-out structured games with 2 resets each and learning Q opponents. Scored one opening and one response after four common random calibration turns.
- Cumulative minus gain-only return +.02476 [.02043,.02874]. Cumulative minus immediate +.00638 [.00308,.00958]. Restored-state reward-visibility interaction +.00003 [−.00161,+.00174]. Prespecified 98.33% block intervals.
- Gain-only had lower absolute terminal payoff but a slightly larger visible-minus-masked gap. Absolute terminal-only produced the strongest terminal reward with high first-move cost.
- Matched history improved cumulative return vs donor history, mainly through first reward. Selection for more informative numerical reward remains unresolved when physical and opponent state are held fixed.
- Independent replay verified 864 samples, all 12,288 held-out decisions, targets, hashes and primary intervals. New exact fingerprint union 36,864; prior-panel overlap zero. Two dev reruns matched all 6 NPZ files byte for byte.
- D-031 full primary completed 25/25 fits paired. Raw recomputation M−P −.01274 [−.01608,−.00954]. Zero-shot tied and M's erased-Q score at k24 was higher, so a lower baseline did not explain that comparison.
- D-031 extensions continue from sealed sources in runs/d031_completion_v2/study/. At 21:25 KST 139/375 fit ledgers completed. Original H2 and similarity reports remain in research/rule_similarity/REPORT.md.

Next action
- Extend the cumulative-return comparison to multiple freely chosen interaction steps using equal total budgets and frozen learner/feedback controls. Do so with a fresh evaluation panel.
