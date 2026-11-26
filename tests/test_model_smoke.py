"""Forward/backward smoke tests; skipped automatically when torch is absent."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch = pytest.importorskip("torch")

from model import DebiasedLinUCB, DRQNPathOptimizer, MAMLRecommender, MSCTrans, make_state  # noqa: E402


def test_msctrans_forward_backward():
    m = MSCTrans()
    out = m(torch.randn(4, 128), torch.randn(4, 64), torch.randn(4, 768))
    assert out.shape == (4, 3)
    out.sum().backward()


def test_maml_outer_step():
    r = MAMLRecommender(input_dim=8, n_items=5)
    task = (torch.randn(6, 8), torch.randint(0, 5, (6,)), torch.randn(6, 8), torch.randint(0, 5, (6,)))
    assert np.isfinite(r.outer_step([task, task]))


def test_drqn_update():
    opt = DRQNPathOptimizer(n_actions=5)
    s = torch.stack([make_state(torch.zeros(32), 1)])
    for _ in range(70):
        opt.push(s, 0, 1.0, s, False)
    assert np.isfinite(opt.update())


def test_debiased_linucb():
    b = DebiasedLinUCB(in_dim=10)
    x = np.random.randn(32, 10).astype("float32")
    b.train_encoder(x, np.random.randint(0, 7, 32), np.random.rand(32), np.random.randint(0, 2, 32), epochs=2)
    arm = b.select_arm(x[0]); b.update(arm, x[0], 1.0)
    assert 0 <= arm < 7
