"""Rosenbaum sensitivity analysis of the interchange sign test, per layer.

For the paired differences D = I(A -> B) - I(A -> B'):
  Gamma*      exact largest hidden-bias level at which the one-sided sign test stays <= 0.05
  Gamma~      design sensitivity phi / (1 - phi)
  p(Gamma0)   exact one-sided P value under a fixed bias Gamma0 in {1.10, 1.25, 1.50}
  conc        2 * AUC - 1 comparing |D| of positive versus negative differences

Usage: python analysis/sensitivity.py borzoi_loops_K562.tsv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binom, mannwhitneyu

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probes"))
from common import path  # noqa: E402

ALPHA, GMAX, G0_LIST = 0.05, 50.0, [1.10, 1.25, 1.50]


def p_at(k, m, G):
    return binom.sf(k - 1, m, G / (1.0 + G))


def gamma_star(k, m):
    if p_at(k, m, 1.0) > ALPHA:
        return np.nan
    if p_at(k, m, GMAX) <= ALPHA:
        return GMAX
    lo, hi = 1.0, GMAX
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if p_at(k, m, mid) <= ALPHA else (lo, mid)
    return lo


def sensitivity(df):
    layers = sorted(int(re.match(r"loop_L(\d+)$", c).group(1)) for c in df.columns if re.match(r"loop_L\d+$", c))
    rows = []
    for L in layers:
        s = df[[f"loop_L{L}", f"mis_L{L}"]].dropna()
        D = (s[f"loop_L{L}"] - s[f"mis_L{L}"]).to_numpy(float)
        D = D[D != 0]
        m, k = D.size, int((D > 0).sum())
        phi = k / m
        pos, neg = np.abs(D[D > 0]), np.abs(D[D < 0])
        conc = (2 * mannwhitneyu(pos, neg, alternative="greater").statistic / (pos.size * neg.size) - 1
                if pos.size and neg.size else np.nan)
        rec = dict(layer=f"L{L}", m=m, phi=phi, gamma_star=gamma_star(k, m),
                   gamma_tilde=phi / (1 - phi) if 0 < phi < 1 else np.nan, conc=conc)
        for G0 in G0_LIST:
            rec[f"p_G{G0}"] = p_at(k, m, G0)
        rows.append(rec)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    fname = sys.argv[1] if len(sys.argv) > 1 else "borzoi_loops_K562.tsv"
    R = sensitivity(pd.read_csv(path(fname), sep="\t"))
    print(R.to_string(index=False, float_format=lambda v: f"{v:.3g}"))
    R.to_csv(path("sensitivity.tsv"), sep="\t", index=False)
