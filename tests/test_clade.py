import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "clade"))
import clade  # noqa: E402


def test_partial_regression_matches_ols():
    rng = np.random.default_rng(0)
    n = 500
    W = rng.normal(size=(n, 3))
    R = W @ [0.5, -0.2, 0.1] + rng.normal(size=n)
    Y = 0.7 * R + W @ [1.0, 0.0, -0.5] + rng.normal(size=n)
    f = clade.fit(Y, R[:, None], W)
    X = np.column_stack([np.ones(n), R, W])
    beta = np.linalg.lstsq(X, Y, rcond=None)[0]
    assert np.isclose(f.theta[0], beta[1])          # Frisch-Waugh-Lovell


def test_two_layer_wald_closed_form():
    z1, z2, r = 0.79, 3.45, 0.87
    W, common, contrast = clade.block_wald_two(z1, z2, r)
    V = np.array([[1, r], [r, 1]])
    z = np.array([z1, z2])
    assert np.isclose(W, z @ np.linalg.solve(V, z))
    assert np.isclose(W, common + contrast)


def test_robustness_value():
    assert abs(clade.robustness_value(3.66, 1635) - 0.0865) < 5e-4
    assert abs(clade.robustness_value(3.66, 1635, alpha=0.05) - 0.0411) < 5e-4


def test_spectrum_rank_one():
    C = np.ones((6, 6))
    s = clade.spectrum(C)
    assert np.isclose(s["top_share"], 1.0) and np.isclose(s["r_eff"], 1.0)
