
"""Reproducible proof-of-concept experiment for the ARS mathematics.

This script does NOT claim to reproduce historical recordings. It generates
synthetic sequences from explicitly stated transition models, hides notes, and
tests whether a constrained higher-order model can recover them.

Run:
    python simulate.py

A fixed master seed is used, and each trial receives a deterministic child seed.
"""

from pathlib import Path
import csv
import numpy as np

from model import seed_everything, fit_markov, reconstruct_missing, reconstruct_order1
from metrics import mask_sequence, missing_accuracy, mean_std

MASTER_SEED = 20260731
N_TRAIN = 1200
N_TEST = 60
LENGTH = 32
ALPHA = 0.5

# Canonical note inventories used only as symbolic labels. These are NOT complete
# raga grammars: the experiment deliberately avoids pretending that a scale alone
# captures a raga's full melodic identity.
RAGAS = {
    "Yaman": ["S", "R", "G", "M+", "P", "D", "N"],
    "Bhairav": ["S", "r", "G", "M", "P", "d", "N"],
    "Todi": ["S", "r", "g", "M+", "P", "d", "N"],
    "Marwa": ["S", "r", "G", "M+", "P", "D", "N"],
    "Purvi": ["S", "r", "G", "M+", "P", "d", "N"],
    "Bhimpalasi": ["S", "R", "g", "M", "P", "D", "n"],
}

def transition_matrix(A):
    """Create a deliberately transparent synthetic melodic grammar.

    The kernel is based on local interval preference plus a small self-transition
    probability. It is a simulation device, not a claim about performance practice.
    """
    N = len(A)
    T = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            distance = abs(j - i)
            T[i, j] = np.exp(-0.9 * distance)
            if distance == 1:
                T[i, j] *= 2.0
            if j == i:
                T[i, j] *= 0.35
        T[i] /= T[i].sum()
    return T

def sample_sequences(A, T, count, length, rng):
    seqs = []
    for _ in range(count):
        i = int(rng.integers(len(A)))
        seq = [A[i]]
        for _ in range(length - 1):
            i = int(rng.choice(len(A), p=T[i]))
            seq.append(A[i])
        seqs.append(seq)
    return seqs

def run():
    rng = seed_everything(MASTER_SEED)
    rows = []

    for raga_i, (name, A) in enumerate(RAGAS.items()):
        local_rng = seed_everything(MASTER_SEED + 1000 * raga_i)
        T = transition_matrix(A)
        train = sample_sequences(A, T, N_TRAIN, LENGTH, local_rng)
        test = sample_sequences(A, T, N_TEST, LENGTH, local_rng)

        # The same synthetic data are used to compare orders; this is a controlled
        # simulation, not a train/test claim about real archival music.
        for order in [1, 2, 3]:
            _, _, probs = fit_markov(train, A, order=order, alpha=ALPHA)
            for missing_rate in [0.10, 0.30, 0.50]:
                vals = []
                for trial, truth in enumerate(test):
                    trial_rng = seed_everything(
                        MASTER_SEED + 100000*raga_i + 1000*order + 10*int(missing_rate*100) + trial
                    )
                    observed, missing = mask_sequence(truth, missing_rate, trial_rng)
                    rec = reconstruct_missing(observed, A, order, probs)
                    vals.append(missing_accuracy(truth, rec, missing))
                mean, std = mean_std(vals)
                rows.append({
                    "raga": name, "order": order,
                    "missing_rate": missing_rate,
                    "accuracy_mean": mean, "accuracy_std": std,
                    "n_trials": len(vals), "seed": MASTER_SEED
                })

    out = Path("results.csv")
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {out.resolve()}")
    for row in rows[:9]:
        print(row)

if __name__ == "__main__":
    run()
