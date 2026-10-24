"""ICE versus ISM for long-range enhancer-gene linking, with gene-clustered bootstrap.

Compares AUPRC of the layer-l ICE score and ISM in the 50-100 kb and 100-250 kb strata on
the training benchmark and on the independent held-out K562 set.

Usage: python analysis/regulatory_stats.py --layer 4
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probes"))
from common import path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--layer", type=int, default=4)
ap.add_argument("--boot", type=int, default=2000)
args = ap.parse_args()
rng = np.random.default_rng(0)
ice = f"ICE{args.layer}_frac"


def auprc(sub, col):
    y = sub.Regulated.astype(int).values
    return np.nan if (y.sum() < 3 or y.sum() == len(y)) else average_precision_score(y, sub[col].values)


def compare(R, lo, hi):
    sub = R[(R.dist.abs() >= lo) & (R.dist.abs() < hi)]
    genes = np.unique(sub.ensg.values)
    gidx = {g: np.where(R.ensg.values == g)[0] for g in genes}
    diffs = []
    for _ in range(args.boot):
        rows = np.concatenate([gidx[g] for g in rng.choice(genes, len(genes), replace=True)])
        bs = R.iloc[rows]
        diffs.append(auprc(bs, ice) - auprc(bs, "dR_frac"))
    d = np.array(diffs)
    d = d[~np.isnan(d)]
    return (auprc(sub, ice), auprc(sub, "dR_frac"), d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5),
            (d <= 0).mean(), int(sub.Regulated.sum()))


for name, f in [("train", "iia_results.tsv"), ("heldout", "iia_heldout_results.tsv")]:
    if not os.path.exists(path(f)):
        print("missing", f)
        continue
    R = pd.read_csv(path(f), sep="\t")
    R["dR_frac"] = R.dR / (R.Rb.abs() + 1e-6)
    R[ice] = R[f"ICE{args.layer}"] / (R.Rb.abs() + 1e-6)
    for lo, hi in [(5e4, 1e5), (1e5, 2.6e5)]:
        a, b, m, l, h, p, n = compare(R, lo, hi)
        print(f"{name:7s} [{int(lo // 1e3)},{int(hi // 1e3)}k) pos={n:3d} ICE={a:.3f} ISM={b:.3f} "
              f"diff={m:+.3f} [{l:+.3f},{h:+.3f}] P(diff<=0)={p:.3f}")
