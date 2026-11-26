import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import anonymize  # noqa: E402
import synthetic_data  # noqa: E402
from utils import demographic_parity_difference, equalized_odds_difference  # noqa: E402


def test_labelling_set_structure_and_split():
    d = synthetic_data.generate()
    assert len(d["label"]) == 14400 and d["face"].shape == (14400, 128)
    s = synthetic_data.student_split()
    assert len(s["train"]) + len(s["val"]) == 163 and len(s["test"]) == 77
    assert not (set(s["train"]) | set(s["val"])) & set(s["test"])


def test_k_anonymity():
    recs = [{"a": i % 3} for i in range(20)] + [{"a": 9}]
    kept = anonymize.enforce_k_anonymity(recs, ["a"], k=5)
    assert anonymize.is_k_anonymous(kept, ["a"], k=5) and len(kept) == 20


def test_fairness_metrics():
    y = np.array([1, 0, 1, 0]); p = np.array([1, 0, 1, 0]); g = np.array([0, 0, 1, 1])
    assert demographic_parity_difference(p, g) == 0.0
    assert equalized_odds_difference(y, p, g) == 0.0
