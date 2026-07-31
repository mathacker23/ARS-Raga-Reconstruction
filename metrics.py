

import numpy as np


def mask_sequence(seq, missing_rate, rng):
    n = len(seq)
    m = int(round(n * missing_rate))
    idx = rng.choice(n, size=m, replace=False)
    out = list(seq)
    for i in idx:
        out[i] = None
    return out, set(idx)


def missing_accuracy(truth, reconstruction, missing_positions):
    if not missing_positions:
        return 1.0
    return float(np.mean([
        truth[i] == reconstruction[i] for i in missing_positions
    ]))


def mean_std(values):
    values = np.asarray(values, dtype=float)
    return float(values.mean()), float(values.std(ddof=1))
