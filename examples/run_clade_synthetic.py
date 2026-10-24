"""Self-contained CLADE example on simulated data with a known answer (CPU, a few seconds).

The simulation mimics the loop analysis: n units, L = 8 layers, covariates W that drive
both the representation and the phenotype (so a naive regression of Y on R_l is
confounded), and layer features that drift smoothly with depth. Only the last layer
has a true effect on Y beyond W.

Run:  python examples/run_clade_synthetic.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "clade"))
import clade  # noqa: E402

n, L = 1600, 8
theta_true = np.zeros(L)
theta_true[-1] = 0.20


def simulate(rng):
    """Return (Y, R, W) for one synthetic data set."""
    # covariates the model also reads (e.g. span, motif scores at both anchors, orientation)
    W = np.column_stack([rng.normal(size=n), rng.normal(size=n), rng.normal(size=n),
                         rng.binomial(1, 0.4, n), rng.binomial(1, 0.15, n)])
    gW = W @ np.array([0.5, 0.3, 0.3, 0.2, -0.1])
    # layer features: a covariate-driven part plus a smooth random walk in depth;
    # the last layer adds a component that is new at that depth
    drift = np.cumsum(rng.normal(scale=0.35, size=(n, L)), axis=1) + rng.normal(size=(n, 1))
    R = 0.8 * gW[:, None] + 0.3 * drift
    R[:, -1] += 0.6 * rng.normal(size=n)
    R = R / R.std(0)
    # heteroskedastic noise
    Y = R @ theta_true + gW + rng.normal(size=n) * (1 + 0.3 * np.abs(W[:, 0]))
    return Y, R, W


Y, R, W = simulate(np.random.default_rng(1))

naive = [np.polyfit(R[:, l], Y, 1)[0] for l in range(L)]
print("naive slopes of Y on R_l (confounded by W):", np.round(naive, 3))
print("true theta:", theta_true)

f = clade.fit(Y, R, W)
print()
print(clade.summary(f))

# repeat the experiment to check bias, standard errors and interval coverage
reps = 200
fits = [clade.fit(*simulate(np.random.default_rng(s))) for s in range(100, 100 + reps)]
est = np.array([g.theta[-1] for g in fits])
ses = np.array([g.se[-1] for g in fits])
cover = np.mean(np.abs(est - theta_true[-1]) <= 1.96 * ses)
print(f"\nfinal layer over {reps} replicates: mean estimate = {est.mean():.3f} (truth 0.200), "
      f"empirical SD = {est.std():.4f}, mean SE = {ses.mean():.4f}, 95% coverage = {cover:.3f}")
