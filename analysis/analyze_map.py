"""Contact-map reconstruction: ridge regression of log(O/E) on distance, on the similarity
descriptor, and on both, with chromosome-grouped 5-fold cross-validation. Reports the
Spearman correlation of out-of-fold predictions overall and within separation strata.

Usage: python analysis/analyze_map.py borzoi_map.tsv enf_map.tsv
"""
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probes"))
from common import path  # noqa: E402


def go(fname):
    R = pd.read_csv(path(fname), sep="\t")
    R = R[np.isfinite(R.oe) & (R.oe > 0)].reset_index(drop=True)
    NL = len([c for c in R.columns if re.match(r"cos_L\d+$", c)])
    y, logd = np.log(R.oe.values), np.log(R.dist.values)
    X = R[[f"cos_L{L}" for L in range(NL)] + [f"nd_L{L}" for L in range(NL)]].values
    groups = R.chrom.astype(str).values

    def cv(M):
        p = np.zeros(len(y))
        for tr, te in GroupKFold(5).split(M, y, groups):
            p[te] = Ridge().fit(M[tr], y[tr]).predict(M[te])
        return p

    p_d, p_s, p_c = cv(logd.reshape(-1, 1)), cv(X), cv(np.column_stack([logd, X]))
    print(f"\n=== {fname}: contact-map reconstruction (n={len(y)}) ===")
    print(f"  Spearman(pred, log O/E): distance={spearmanr(p_d, y).correlation:.3f} "
          f"similarity={spearmanr(p_s, y).correlation:.3f} combined={spearmanr(p_c, y).correlation:.3f}")
    for lo, hi in [(2e4, 5e4), (5e4, 1e5), (1e5, 2.5e5), (2.5e5, 5e5)]:
        m = ((R.dist >= lo) & (R.dist < hi)).values
        if m.sum() > 50:
            print(f"  {int(lo / 1e3)}-{int(hi / 1e3)} kb: similarity-only Spearman = "
                  f"{spearmanr(p_s[m], y[m]).correlation:.3f} (n={int(m.sum())})")


for f in sys.argv[1:] or ["borzoi_map.tsv", "enf_map.tsv"]:
    if os.path.exists(path(f)):
        go(f)
    else:
        print("missing", path(f))
