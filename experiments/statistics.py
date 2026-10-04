"""Episode-level uncertainty, not window-level pseudo-replication."""

import numpy as np


def wilson(successes, trials):
    if not isinstance(trials, int) or not isinstance(successes, int) or not 0 <= successes <= trials or trials == 0:
        raise ValueError("Need integer counts including failures/timeouts and at least one trial")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return float(max(0, center - radius)), float(min(1, center + radius))


def paired_bootstrap(baseline, proposed, *, seed=0, draws=10000):
    """Input maps paired episode IDs -> binary success, one training seed at a time."""
    if not baseline or baseline.keys() != proposed.keys():
        raise ValueError("Paired methods must contain the exact same episode IDs")
    if not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be positive")
    keys = sorted(baseline)
    if any(v not in (0, 1, False, True) for d in (baseline, proposed) for v in d.values()):
        raise ValueError("Expected binary episode outcomes")
    differences = np.array([int(proposed[k]) - int(baseline[k]) for k in keys], float)
    rng = np.random.default_rng(seed)
    estimates = np.empty(draws)
    for start in range(0, draws, 1000):
        size = min(1000, draws - start)
        estimates[start:start + size] = rng.choice(differences, (size, len(keys))).mean(-1)
    return {"difference": float(differences.mean()),
            "ci95": np.quantile(estimates, [.025, .975]).tolist(),
            "paired_episodes": len(keys), "bootstrap_seed": seed, "draws": draws}
