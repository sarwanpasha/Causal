"""Write a synthetic loop table with the same columns as probes/loop_interchange.py, so
that the analysis scripts can be tried without a GPU or the model weights.

  python examples/make_synthetic_loops.py
  CAUSAL_WORKDIR=work python analysis/interchange_identification.py synthetic_loops.tsv
  CAUSAL_WORKDIR=work python analysis/sensitivity.py synthetic_loops.tsv
  CAUSAL_WORKDIR=work python analysis/analyze_loops.py synthetic_loops.tsv
  CAUSAL_WORKDIR=work python clade/run_clade.py synthetic_loops.tsv

The partner advantage grows with depth by construction; values are arbitrary units.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probes"))
from common import WORKDIR, path  # noqa: E402

rng = np.random.default_rng(0)
n, L = 1500, 8
span = rng.integers(50000, 450000, n)
scA, scB = rng.normal(12, 3, n), rng.normal(12, 3, n)
orient = rng.choice(["convergent", "divergent", "tandem"], n, p=[0.4, 0.15, 0.45])
base = np.linspace(0.002, 0.08, L)                       # influence scale grows with depth
pair = np.linspace(0.0, 0.007, L)                        # pairing-specific effect
cols = dict(chrom=rng.choice([f"chr{i}" for i in range(1, 23)], n), A=rng.integers(1e6, 2e8, n))
cols["B"] = cols["A"] + span
cols["span"] = span
cos = 0.3 + 0.002 * (scA + scB)[:, None] + np.cumsum(rng.normal(0, 0.02, (n, L)), 1)
cos[:, -1] += 0.03 * rng.normal(size=n)
strength = np.exp(0.5 + 0.02 * (scA + scB) - 2e-6 * span + 1.5 * (cos[:, -1] - cos[:, -1].mean())
                  + rng.normal(0, 0.4, n))
cols["strength"] = strength
for l in range(L):
    noise = base[l] * 0.5
    cols[f"loop_L{l}"] = np.abs(base[l] + pair[l] + rng.normal(0, noise, n))
for l in range(L):
    cols[f"mis_L{l}"] = np.abs(base[l] + rng.normal(0, base[l] * 0.5, n))
for l in range(L):
    cols[f"dist_L{l}"] = np.abs(base[l] * 0.95 + rng.normal(0, base[l] * 0.5, n))
for l in range(L):
    cols[f"cos_L{l}"] = cos[:, l]
    cols[f"nd_L{l}"] = 1 - cos[:, l]
cols.update(scA=scA, scB=scB, orient=orient)
os.makedirs(WORKDIR, exist_ok=True)
pd.DataFrame(cols).to_csv(path("synthetic_loops.tsv"), sep="\t", index=False)
print("wrote", path("synthetic_loops.tsv"))
