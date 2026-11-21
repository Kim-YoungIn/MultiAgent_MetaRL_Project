# Two-turn planning in unfamiliar structured games

After four shared random interactions, a learned opening that anticipated a second decision earned **+0.00356** two-turn reward over a myopic opening using the same history and frozen neural responder. The 97.5% paired block-bootstrap interval was **[+0.00064, +0.00678]**. The two policies chose different openings on 18.7% of trials. Relative to the myopic policy, the planner sacrificed 0.00351 in expected first-turn reward on average. Both turns were scored in the primary endpoint.

The stronger claim about *choosing a move specifically for the information in its reward* was unresolved. We restored the environment and opponent to their pre-opening state while letting the focal agent retain its opening transcript. In that assay, a uniformly random opening still made the first reward useful: revealing it improved the responder's terminal payoff by 0.01367. But the interaction between opening specialization and reward visibility was only **+0.00063 [−0.00015, +0.00167]**. The visible-specialized and masked-specialized openings' two-turn returns were .04248 and .04283 with visible reward and .02722 and .02819 when masked. Using reward information once obtained and learning to select a more informative action are different empirical achievements.

The protocol used 16 independently trained neural pairs on disjoint task catalogs. Across the 16 blocks the main held-out population comprised 4,096 games with two resets per game; the zero- and eight-prefix diagnostics used additional fresh games. Both networks were supervised on simulator branches. The frozen responder saw matched visible and masked examples. The opener received calibration records only. The rewind removes the opening's physical effects on the world and opponent; its two-turn value pertains to that diagnostic intervention. This study compares planning objectives. The direct meta-training versus ordinary pretraining comparison appears in the [opponent-adaptation study](../opponent_adaptation/REPORT.md). The reason to freeze a responder is related to the coupling between learning exploration and exploitation described in [DREAM](https://proceedings.mlr.press/v139/liu21s.html).

Independent verification reconstructed all saved neural choices, all primary block effects and confidence bounds. It replayed 960 sampled records through the underlying simulator with independent Q updates and verified source, data and checkpoint hashes. The 22,528 distinct serialized rule fingerprints were disjoint from the 3,776 in the checked previous structured panels. Exact uniqueness does not imply strategic novelty. No final evaluation of the proposed reward-stakes extension was run.

From the repository root, use Python with NumPy and PyTorch:

```bash
export PYTHONPATH=05_toy_models:.
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
.venv/bin/python -m unittest research.active_probe.test_data -v
.venv/bin/python -m research.active_probe.run runs/repro_active_probe --split final --workers 4
.venv/bin/python -m research.active_probe.verify runs/repro_active_probe
```

The retained execution is under `runs/active_probe_v1/`, including source snapshots, full branch arrays, fitted weights and `independent_verification.json`. [Frozen protocol](PROTOCOL.md)
