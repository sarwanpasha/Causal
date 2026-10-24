"""CTCF motif annotation of loop anchors and the orientation analysis.

Scans +/- 1.5 kb around each anchor with the JASPAR MA0139.1 CTCF profile on both
strands and keeps the strand and log-odds score of the best hit. Loops whose two anchors
both exceed the 60th percentile of best scores are called strong-CTCF loops and are
classified as convergent, divergent or tandem. Also fits a chromosome-grouped ridge
decoder of loop strength from the similarity descriptor.

Usage: python analysis/ctcf_annotate.py borzoi_loops_K562.tsv
Writes <input>_annot.tsv (adds stA, scA, stB, scB, orient), the anchor annotation
used by the CLADE analysis.
"""
import json
import os
import re
import sys
import urllib.request

import numpy as np
import pandas as pd
import pyfaidx
from scipy.stats import mannwhitneyu, spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "probes"))
from common import path  # noqa: E402

fname = sys.argv[1] if len(sys.argv) > 1 else "borzoi_loops_K562.tsv"
R = pd.read_csv(path(fname), sep="\t")
NL = len([c for c in R.columns if re.match(r"loop_L\d+$", c)])


def load_pfm():
    local = path("MA0139.1.jaspar")
    if os.path.exists(local):
        txt = open(local).read()
    else:
        req = urllib.request.Request("https://jaspar.elixir.no/api/v1/matrix/MA0139.1/?format=json",
                                     headers={"User-Agent": "Mozilla/5.0"})
        p = json.loads(urllib.request.urlopen(req, timeout=60).read().decode())["pfm"]
        return np.array([p["A"], p["C"], p["G"], p["T"]], float)
    nums = [list(map(float, re.findall(r"[\d.]+", ln))) for ln in txt.splitlines() if re.search(r"\d", ln)]
    return np.array([r for r in nums if len(r) >= 4][:4], float)


M = load_pfm()
W = M.shape[1]
ppm = (M + 0.25) / (M.sum(0, keepdims=True) + 1)
pwm = np.log2(ppm / 0.25)
genome = pyfaidx.Fasta(path("hg38.fa"))


def onehot(seq):
    o = np.zeros((len(seq), 4))
    b = np.frombuffer(seq.encode(), np.uint8)
    for k, i in zip("ACGT", range(4)):
        o[b == ord(k), i] = 1
    return o


def best_strand(chrom, pos, flank=1500):
    o = onehot(str(genome[chrom][max(0, pos - flank):pos + flank]).upper())
    if len(o) < W + 1:
        return "none", -1e9
    n = len(o) - W + 1
    f, r = np.zeros(n), np.zeros(n)
    rc = o[::-1, ::-1]
    for j in range(W):
        f += o[j:j + n] @ pwm[:, j]
        r += rc[j:j + n] @ pwm[:, j]
    return ("+" if f.max() >= r.max() else "-"), float(max(f.max(), r.max()))


sA = [best_strand(str(x.chrom), int(x.A)) for x in R.itertuples()]
sB = [best_strand(str(x.chrom), int(x.B)) for x in R.itertuples()]
R["stA"], R["scA"] = [t[0] for t in sA], [t[1] for t in sA]
R["stB"], R["scB"] = [t[0] for t in sB], [t[1] for t in sB]
thr = np.nanpercentile(np.concatenate([R.scA, R.scB]), 60)
strong = (R.scA >= thr) & (R.scB >= thr)


def orientation(r):
    if r.stA == "+" and r.stB == "-":
        return "convergent"
    if r.stA == "-" and r.stB == "+":
        return "divergent"
    return "tandem"


R["orient"] = R.apply(orientation, axis=1)
deep = f"loop_L{NL - 1}"
print(f"\nCTCF orientation (strong-CTCF loops n={int(strong.sum())}, threshold={thr:.1f})")
for o in ["convergent", "divergent", "tandem"]:
    m = strong & (R.orient == o)
    print(f"  {o:11s} n={int(m.sum()):4d} median deep-layer influence={R[deep][m].median():.4f}")
conv, div = R[deep][strong & (R.orient == "convergent")], R[deep][strong & (R.orient == "divergent")]
if len(conv) > 5 and len(div) > 5:
    print(f"  convergent > divergent, Mann-Whitney P = {mannwhitneyu(conv, div, alternative='greater').pvalue:.2f}")

feats = [f"cos_L{L}" for L in range(NL)] + [f"nd_L{L}" for L in range(NL)]
G = R.dropna(subset=["strength"]).reset_index(drop=True)
X, ys = G[feats].values, G.strength.values
pred = np.zeros(len(ys))
for tr, te in GroupKFold(5).split(X, ys, G.chrom.values):
    pred[te] = Ridge().fit(X[tr], ys[tr]).predict(X[te])
print(f"\nGraded decoder of loop strength (chromosome-held-out): Spearman = {spearmanr(pred, ys).correlation:.3f}")
out = fname.replace(".tsv", "_annot.tsv")
R.to_csv(path(out), sep="\t", index=False)
print("wrote", path(out))
