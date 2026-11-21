# Reward-signal selectivity at altered terminal stakes

The fixed visible-specialized opener's genuine-minus-shuffled terminal reward exceeded the masked-specialized opener's by 0.001883 at payoff weight eight, with a 95% paired block-bootstrap interval [0.000800, 0.002937]. Sixteen existing model pairs faced 4,096 fresh games, with 32 within-motif and opponent-action donor permutations per game. The original terminal responder and opener weights were fixed.

At weight eight, visible-specialized and masked-specialized openers achieved genuine terminal payoffs 0.02756 and 0.02601, and shuffled-signal payoffs 0.00962 and 0.00995. Their full weighted returns were 0.21496 and 0.19900. The myopic opener obtained 0.22266. The positive interaction describes sensitivity to this specific shuffled reward intervention and trained decoder. It does not establish superiority to the myopic decision. The weight multiplies a single terminal payoff. At weight one, the corresponding visible, masked and myopic genuine returns were 0.03016, 0.03035 and 0.03142. The two specialized openers disagreed on 7.7% of games at weight one and 20.8% at weight eight. The visible opener paid an average opening cost of 0.00962 against the myopic opener at weight eight.

The saved output includes each game, donor permutation, decisions and the true and shuffled terminal payoffs. Verification checked 16 checkpoint hashes, all donor mappings and saved payoff indexing and repeated the paired bootstrap. The uniform opener earned 0.17864 at weight eight, with genuine and shuffled terminal rewards of 0.02553 and 0.00902. Exact fingerprints did not overlap 32,448 checked fingerprints from the active-probe, structured-transfer, structured-scale, two-turn-oracle and opponent-adaptation panels. An independent simulator and opponent replay checked 64 games, covering each block and motif. A separate NumPy inference audit reproduced all persisted neural decisions within a score tolerance of 1e−6. The sham intervention changes only the scalar reward input of the visible channel. The complete means and audit are saved in `runs/probe_stakes_v1/outcomes.csv` and `prediction_audit.json`. Exact fingerprints identify identical serializations.

Run from the repository root with NumPy and PyTorch available.

```bash
PYTHONPATH=05_toy_models:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.probe_stakes.run
PYTHONPATH=05_toy_models:. OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.probe_stakes.verify
PYTHONPATH=05_toy_models:. OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.probe_stakes.complete
PYTHONPATH=05_toy_models:. OPENBLAS_NUM_THREADS=1 .venv/bin/python -m research.probe_stakes.audit_predictions
```

The simulation writes to runs/probe_stakes_v1 and skips existing blocks. The verification record contains per-block preservation hashes.
