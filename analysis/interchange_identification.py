"""Per-layer decomposition and calibration of the interchange probe (Table 6 style).

  baseline = mean I(A -> A-d)            distance-only control
  confound = mean I(A -> B') - baseline  anchor-status / 1D confound
  tau      = mean I(A -> B)  - mean I(A -> B')   pairing-specific effect
plus the fraction of loops with I(A -> B) > I(A -> B'), the exact sign test against 1/2,
the one-sided Wilcoxon signed-rank test, and a loop-bootstrap 95% CI for tau.

Usage: python analysis/interchange_identification.py borzoi_loops_K562.tsv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probes"))
from common import path  # noqa: E402


def decompose(df, B=2000, seed=0):
    rng = np.random.default_rng(seed)
    layers = sorted(int(re.match(r"loop_L(\d+)$", c).group(1)) for c in df.columns if re.match(r"loop_L\d+$", c))
    rows = []
    for L in layers:
        sub = df[[f"loop_L{L}", f"mis_L{L}", f"dist_L{L}"]].dropna()
        loop, mis, dist = (sub[c].to_numpy(float) for c in sub.columns)
        n = len(sub)
        d = loop - mis
        k, n_eff = int((d > 0).sum()), int((d != 0).sum())
        boot = np.array([d[rng.integers(0, n, n)].mean() for _ in range(B)])
        rows.append(dict(layer=f"L{L}", n=n, mean_partner=loop.mean(), baseline=dist.mean(),
                         confound=mis.mean() - dist.mean(), tau=d.mean(),
                         tau_lo=np.percentile(boot, 2.5), tau_hi=np.percentile(boot, 97.5),
                         fraction=k / n_eff, sign_p=binomtest(k, n_eff, 0.5, alternative="greater").pvalue,
                         wilcoxon_p=wilcoxon(loop, mis, alternative="greater").pvalue))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "borzoi_loops_K562.tsv"
    R = decompose(pd.read_csv(path(fname), sep="\t"))
    with pd.option_context("display.width", 200, "display.max_columns", 30):
        print(R.to_string(index=False, float_format=lambda v: f"{v:.4g}"))
    R.to_csv(path("interchange_identification.tsv"), sep="\t", index=False)
