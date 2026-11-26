"""
utils.py - Shared helpers: seeding, effect sizes and fairness metrics
(Section IV-H). numpy-only except set_seed, which also seeds torch if present.
"""

import random

import numpy as np

from config import RANDOM_SEED


def set_seed(seed=RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def cohens_d(x, y):
    """Cohen's d with pooled SD (Table II reports d = 4.29-4.71)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    return (x.mean() - y.mean()) / sp


def demographic_parity_difference(y_pred, group):
    """max - min positive-prediction rate across groups."""
    y_pred, group = np.asarray(y_pred), np.asarray(group)
    rates = [y_pred[group == g].mean() for g in np.unique(group)]
    return float(max(rates) - min(rates))


def equalized_odds_difference(y_true, y_pred, group):
    """Largest across-group gap in true-positive or false-positive rate."""
    y_true, y_pred, group = map(np.asarray, (y_true, y_pred, group))
    tpr, fpr = [], []
    for g in np.unique(group):
        m = group == g
        pos, neg = y_true[m] == 1, y_true[m] == 0
        if pos.any():
            tpr.append(y_pred[m][pos].mean())
        if neg.any():
            fpr.append(y_pred[m][neg].mean())
    gaps = [max(v) - min(v) for v in (tpr, fpr) if len(v) > 1]
    return float(max(gaps)) if gaps else 0.0
