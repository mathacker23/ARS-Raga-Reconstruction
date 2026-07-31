
# Reproducible code for the mathematical raga-reconstruction example

This repository is a **proof-of-concept simulation**, not a reconstruction of a
historical performance. The synthetic generator intentionally uses simple,
explicit transition kernels so that every result can be inspected.

## Reproducibility

- Master seed: `20260731`
- Each raga/order/trial receives a deterministic derived seed.
- Python and NumPy versions should be recorded before submission.
- The simulation writes `results.csv`.

## Run

```bash
python -m pip install -r requirements.txt
python simulate.py
```

## What the experiment tests

For a finite symbolic alphabet A, an order-k Markov model estimates

P(X_t=a | X_{t-k:t-1})

with a symmetric Dirichlet prior. Missing symbols are then reconstructed by
maximum a posteriori dynamic programming. Orders 1, 2, and 3 are compared.

## Important cultural-scientific boundary

The names Yaman, Bhairav, Todi, Marwa, Purvi, and Bhimpalasi identify the
symbolic note inventories used in the simulation. The code does **not** claim
that these inventories constitute complete raga grammars. Raga identity depends
on melodic movement, phraseology, intonation, ornamentation, emphasis, and
performance tradition. Real archival reconstruction therefore requires
expert-annotated recordings and raga-specific constraints.

## GitHub

After creating the repository, replace the placeholder in the manuscript with
the actual public URL, e.g.

`https://github.com/YOUR-USERNAME/ars-raga-reconstruction`

Do not cite a repository URL that has not actually been created.
