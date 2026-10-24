"""Measured-confounder adjustment of the interchange pairing effect.

For each layer l, the indicator 1[I_l(A -> B) > I_l(A -> B')] is regressed by logistic
regression on standardized anchor covariates. Under no pairing effect beyond the
covariates the adjusted intercept beta_0 is 0, so a positive, significant beta_0 means
the partner advantage persists after adjustment. Also reports the partial Spearman
correlation of the paired difference with a covariate given span.

Covariate sets (columns that must be present in the annotated loop table):
  motif      : scA, scB, span
  occupancy  : any columns starting with ctcf_, rad21_, smc3_  (+ span)
  access     : any columns starting with dnase_, umap_          (+ span)
Columns for the occupancy and accessibility sets are added by
scripts/annotate_anchor_tracks.py from ENCODE narrowPeak/bigWig files.

Usage: python analysis/confounder_adjust.py borzoi_loops_K562_annot.tsv --set motif
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probes"))
from common import path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("table")
ap.add_argument("--set", choices=["motif", "occupancy", "access", "all"], default="motif")
ap.add_argument("--partial", default=None, help="covariate for the partial correlation (default: first of set)")
args = ap.parse_args()

R = pd.read_csv(path(args.table), sep="\t")
NL = len([c for c in R.columns if re.match(r"loop_L\d+$", c)])
prefixes = {"motif": None, "occupancy": ("ctcf_", "rad21_", "smc3_"), "access": ("dnase_", "umap_")}
if args.set == "motif":
    cov = ["scA", "scB", "span"]
elif args.set == "all":
    cov = ["scA", "scB", "span"] + [c for c in R.columns if c.startswith(prefixes["occupancy"] + prefixes["access"])]
else:
    cov = [c for c in R.columns if c.startswith(prefixes[args.set])] + ["span"]
missing = [c for c in cov if c not in R.columns]
if missing or not cov:
    sys.exit(f"missing covariate columns: {missing or 'none found for this set'}")


def logit_fit(y, X, iters=100):
    """Maximum-likelihood logistic regression by Newton-Raphson; returns (beta, se, z, p)."""
    beta = np.zeros(X.shape[1])
    for _ in range(iters):
        mu = 1 / (1 + np.exp(-X @ beta))
        H = X.T @ (X * (mu * (1 - mu))[:, None])
        step = np.linalg.solve(H, X.T @ (y - mu))
        beta += step
        if np.max(np.abs(step)) < 1e-10:
            break
    se = np.sqrt(np.diag(np.linalg.inv(H)))
    z = beta / se
    return beta, se, z, 2 * norm.sf(np.abs(z))


def zscore(x):
    return (x - x.mean()) / (x.std() + 1e-12)


def partial_spearman(d, x, z):
    """Spearman correlation of d and x after removing a linear fit on rank(z)."""
    rd, rx, rz = (pd.Series(v).rank().values for v in (d, x, z))
    Z = np.column_stack([np.ones_like(rz), rz])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    return spearmanr(res(rd), res(rx))


X = np.column_stack([np.ones(len(R))] + [zscore(np.log1p(R[c].clip(lower=0))) if c not in ("scA", "scB", "span")
                                         else zscore(R[c]) for c in cov])
print(f"covariates ({args.set}): {cov}; condition number = {np.linalg.cond(X):.2f}")
pc = args.partial or cov[0]
for L in range(NL):
    d = (R[f"loop_L{L}"] - R[f"mis_L{L}"]).values
    y = (d > 0).astype(int)
    beta, se, z, pv = logit_fit(y, X)
    r, p = partial_spearman(d, R[pc].values, R.span.values)
    print(f"L{L}: raw fraction={y.mean():.3f} beta0={beta[0]:+.3f} z={z[0]:+.2f} "
          f"P={pv[0]:.2e} partial r({pc} | span)={r:+.3f} (P={p:.2f})")
