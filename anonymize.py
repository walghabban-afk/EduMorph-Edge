"""
anonymize.py - Privacy tooling applied before any interaction data are shared
(Appendix B): k-anonymity over quasi-identifiers (k = 5) and the Laplace
mechanism for differentially private release of aggregates (epsilon = 1.2).

numpy-only. Example:
  python anonymize.py   # runs a self-check on synthetic records
"""

import numpy as np

from config import DP_EPSILON, K_ANONYMITY_K


def equivalence_class_sizes(records, quasi_identifiers):
    """records: list of dicts. Returns {qi-tuple: count}."""
    sizes = {}
    for r in records:
        key = tuple(r[q] for q in quasi_identifiers)
        sizes[key] = sizes.get(key, 0) + 1
    return sizes


def is_k_anonymous(records, quasi_identifiers, k=K_ANONYMITY_K):
    return min(equivalence_class_sizes(records, quasi_identifiers).values()) >= k


def enforce_k_anonymity(records, quasi_identifiers, k=K_ANONYMITY_K):
    """Suppress records whose quasi-identifier combination occurs fewer than k
    times (generalize quasi-identifiers beforehand, e.g. age -> age band)."""
    sizes = equivalence_class_sizes(records, quasi_identifiers)
    return [r for r in records if sizes[tuple(r[q] for q in quasi_identifiers)] >= k]


def age_band(age):
    return "13-14" if age <= 14 else ("15-16" if age <= 16 else "17-18")


def laplace_count(count, epsilon=DP_EPSILON, rng=None):
    """Differentially private count (sensitivity 1)."""
    rng = rng or np.random.default_rng()
    return float(count + rng.laplace(0.0, 1.0 / epsilon))


def laplace_mean(values, lower, upper, epsilon=DP_EPSILON, rng=None):
    """Differentially private mean of values clipped to [lower, upper]; the
    budget is split equally between the noisy sum and the noisy count."""
    rng = rng or np.random.default_rng()
    v = np.clip(np.asarray(values, float), lower, upper)
    s = v.sum() + rng.laplace(0.0, (upper - lower) / (epsilon / 2))
    n = max(1.0, len(v) + rng.laplace(0.0, 1.0 / (epsilon / 2)))
    return float(s / n)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    recs = [{"age": age_band(int(a)), "gender": g, "school": f"S{s}"}
            for a, g, s in zip(rng.integers(13, 19, 300), rng.choice(["F", "M"], 300), rng.integers(1, 10, 300))]
    qi = ["age", "gender", "school"]
    kept = enforce_k_anonymity(recs, qi)
    print(f"k-anonymity (k={K_ANONYMITY_K}): kept {len(kept)} of {len(recs)} records; "
          f"check = {is_k_anonymous(kept, qi)}")
    print(f"DP count of 300 (epsilon={DP_EPSILON}): {laplace_count(300, rng=rng):.1f}")
