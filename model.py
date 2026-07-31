
"""Core constrained higher-order Markov reconstruction model.

The implementation is deliberately small and transparent: it is intended to make
the mathematics in the accompanying paper directly inspectable and reproducible.
"""

from __future__ import annotations
import math
from collections import Counter, defaultdict
import numpy as np


def seed_everything(seed: int) -> np.random.Generator:
    """Return a dedicated NumPy RNG so every stochastic experiment is reproducible."""
    return np.random.default_rng(seed)


def fit_markov(sequences, alphabet, order=1, alpha=0.5):
    """Fit a Dirichlet-smoothed order-k Markov transition tensor.

    For each context c=(x_{t-k},...,x_{t-1}), the posterior mean is
        (N(c,a)+alpha) / (N(c)+alpha*|A|).
    This is used rather than calling it an unrestricted MLE because sparse
    high-order contexts otherwise receive zero probability.
    """
    A = list(alphabet)
    idx = {a:i for i, a in enumerate(A)}
    counts = defaultdict(Counter)

    for seq in sequences:
        if len(seq) <= order:
            continue
        for t in range(order, len(seq)):
            ctx = tuple(seq[t-order:t])
            counts[ctx][seq[t]] += 1

    probs = {}
    for ctx in counts:
        total = sum(counts[ctx].values())
        probs[ctx] = np.array(
            [(counts[ctx][a] + alpha) / (total + alpha*len(A)) for a in A],
            dtype=float,
        )
    return A, idx, probs


def logp(probs, alphabet, context, nxt, alpha=0.5):
    """Return a smoothed log transition probability for an unseen context too."""
    A = list(alphabet)
    if context in probs:
        p = probs[context][A.index(nxt)]
    else:
        p = 1.0 / len(A)
    return math.log(max(p, 1e-300))


def reconstruct_missing(observed, alphabet, order, probs, allowed=None):
    """Return the MAP sequence consistent with all observed symbols.

    The dynamic program handles missing symbols even in the first k positions.
    For contexts shorter than k, the smoothed model uses a uniform prior.
    """
    n = len(observed)
    A = list(alphabet)
    allowed = allowed or {}

    # State is the recent context. Each state stores the highest-scoring prefix.
    dp = {tuple(): (0.0, [])}

    for t in range(n):
        new = {}
        candidates = [observed[t]] if observed[t] is not None else A

        for ctx, (score, prefix) in dp.items():
            for nxt in candidates:
                if allowed and ctx and nxt not in allowed.get(ctx[-1], A):
                    continue

                if len(ctx) < order:
                    p = 1.0 / len(A)
                else:
                    p = probs.get(ctx, np.ones(len(A))/len(A))[A.index(nxt)]

                score2 = score + math.log(max(p, 1e-300))
                ctx2 = tuple((*ctx, nxt)[-order:])
                if ctx2 not in new or score2 > new[ctx2][0]:
                    new[ctx2] = (score2, prefix + [nxt])
        dp = new

    if not dp:
        raise ValueError("No legal reconstruction exists under the supplied grammar.")

    _, best = max(dp.values(), key=lambda z: z[0])
    return best


def reconstruct_order1(observed, alphabet, probs, allowed=None):
    """Convenience wrapper for the first-order baseline."""
    return reconstruct_missing(observed, alphabet, 1, probs, allowed=allowed)
