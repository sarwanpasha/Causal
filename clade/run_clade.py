"""Run CLADE on an annotated loop table.

Y = log loop strength (HiCCUPS observed / expected), R_l = anchor-to-anchor cosine at
block l, W = CTCF sequence grammar (span, motif scores at both anchors, orientation class).

Usage:
  python clade/run_clade.py borzoi_loops_K562_annot.tsv
  python clade/run_clade.py borzoi_loops_GM12878.tsv --covariates span   # reduced panel
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probes"))
import clade  # noqa: E402
from common import path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("table")
ap.add_argument("--feature", default="cos", help="per-layer feature prefix (cos or nd)")
ap.add_argument("--covariates", nargs="*", default=["span", "scA", "scB", "orient"])
args = ap.parse_args()

D = pd.read_csv(path(args.table), sep="\t")
D = D[np.isfinite(D.strength) & (D.strength > 0)].reset_index(drop=True)
L = len([c for c in D.columns if re.match(rf"{args.feature}_L\d+$", c)])
Y = np.log(D.strength.values)
R = D[[f"{args.feature}_L{l}" for l in range(L)]].values
cols = []
for c in args.covariates:
    if c == "orient":
        for level in ("convergent", "divergent"):
            cols.append((D.orient == level).astype(float).values)
    else:
        cols.append(D[c].astype(float).values)
W = np.column_stack(cols)
f = clade.fit(Y, R, W)
print(f"n = {f.n}, layers = {L}, covariates = {args.covariates}")
print(clade.summary(f))
out = pd.DataFrame(dict(layer=[f"L{l}" for l in range(L)], theta=f.theta, se=f.se, z=f.z))
out.to_csv(path(os.path.basename(args.table).replace(".tsv", "_clade.tsv")), sep="\t", index=False)
pd.DataFrame(f.corr, index=out.layer, columns=out.layer).to_csv(
    path(os.path.basename(args.table).replace(".tsv", "_clade_corr.tsv")), sep="\t")
