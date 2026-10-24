"""Per-layer summary of the interchange probe: fraction of loops whose true partner
receives more influence than the mispaired control, one-sided Wilcoxon signed-rank
P value, and the association of deep-layer influence with loop strength.

Usage: python analysis/analyze_loops.py borzoi_loops_K562.tsv [borzoi_loops_GM12878.tsv ...]
"""
import os
import re
import sys

import pandas as pd
from scipy.stats import spearmanr, wilcoxon

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probes"))
from common import path  # noqa: E402


def report(fname, max_span=None):
    R = pd.read_csv(path(fname), sep="\t")
    if max_span:
        R = R[R.span < max_span]
    NL = len([c for c in R.columns if re.match(r"loop_L\d+$", c)])
    print(f"\n=== {fname} (n={len(R)}) ===")
    for L in range(NL):
        lp, mi = R[f"loop_L{L}"], R[f"mis_L{L}"]
        p = wilcoxon(lp, mi, alternative="greater").pvalue
        print(f"  L{L}: fraction(partner > mispaired) = {(lp > mi).mean():.3f}  Wilcoxon P = {p:.1e}")
    deep = f"loop_L{NL - 1}"
    if R.strength.notna().sum() > 10:
        rho, p = spearmanr(R[deep], R.strength, nan_policy="omit")
        print(f"  Spearman(deep-layer influence, loop strength) = {rho:.3f} (P = {p:.1e})")
        q = pd.qcut(R.strength.rank(method="first"), 3, labels=["weak", "mid", "strong"])
        print("  median deep-layer influence by strength tertile:",
              {t: round(R[deep][q == t].median(), 4) for t in ["weak", "mid", "strong"]})


for f in sys.argv[1:] or ["borzoi_loops_K562.tsv"]:
    report(f)
if len(sys.argv) == 1:
    print("\nBorzoi restricted to loops shorter than 180 kb (the range Enformer can see):")
    report("borzoi_loops_K562.tsv", max_span=180000)
