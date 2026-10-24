"""Linear decoding probe: loop versus distance-matched non-loop anchor pairs.

Each loop contributes one positive pair and one negative pair: a random intra-chromosomal
pair with identical separation. Features are the per-block similarity descriptor of
Eq. (1) (cosine and normalized distance). An L2-regularized logistic classifier is
evaluated by leave-one-chromosome-out cross-validation.

Usage:
  python probes/loop_decode.py --model borzoi --limit 2000
  python probes/loop_decode.py --model enformer --limit 2500
"""
import argparse

import numpy as np
import pandas as pd
import pyfaidx
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from common import MODEL_SPECS, PairCapture, forward, load_model, one_hot, pair_similarity, path, window_sequence

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["borzoi", "enformer"], default="borzoi")
ap.add_argument("--loops", default="K562_loops_hg38.tsv")
ap.add_argument("--genome", default="hg38.fa")
ap.add_argument("--limit", type=int, default=2000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

spec = MODEL_SPECS[args.model]
SEQLEN, TPOS, TRES, CF = spec["seqlen"], spec["tpos"], spec["tres"], spec["channels_first"]
HALF = SEQLEN // 2
model, NL = load_model(args.model)
genome = pyfaidx.Fasta(path(args.genome))
loops = pd.read_csv(path(args.loops), sep="\t")
if args.model == "enformer":
    loops = loops[loops.span < 180000].reset_index(drop=True)
if args.limit:
    loops = loops.iloc[:args.limit].copy()

rng = np.random.default_rng(args.seed)
chrlen = {c: len(genome[c]) for c in set(loops.chrom.astype(str))}
pairs = []
for _, r in loops.iterrows():
    c, A, B = str(r.chrom), int(r.anchorA), int(r.anchorB)
    d = B - A
    pairs.append((c, A, B, 1, d))
    s = int(rng.integers(HALF, max(HALF + 1, chrlen[c] - HALF - d)))
    pairs.append((c, s, s + d, 0, d))
P = pd.DataFrame(pairs, columns=["chrom", "A", "B", "label", "dist"])

caps = [PairCapture() for _ in range(NL)]
for L in range(NL):
    model.transformer[L].register_forward_hook(caps[L])


@torch.no_grad()
def features(c, A, B):
    seq, s = window_sequence(genome, c, (A + B) // 2, SEQLEN)
    pA, pB = (A - s) // TRES, (B - s) // TRES
    if not (0 <= pA < TPOS and 0 <= pB < TPOS):
        return None
    for cp in caps:
        cp.pA, cp.pB = pA, pB
    forward(model, args.model, one_hot(seq, CF))
    f = []
    for L in range(NL):
        f += list(pair_similarity(caps[L].vA, caps[L].vB))
    return f


X, y, logd, ch = [], [], [], []
for i, r in P.iterrows():
    f = features(str(r.chrom), int(r.A), int(r.B))
    if f is None:
        continue
    X.append(f)
    y.append(int(r.label))
    logd.append([np.log10(max(1, int(r.dist)))])
    ch.append(str(r.chrom))
    if i % 200 == 0:
        print(f"{i}/{len(P)}", flush=True)
X, y, logd, ch = map(np.array, (X, y, logd, ch))


def cv_auc(F):
    aucs = []
    for c in np.unique(ch):
        tr, te = ch != c, ch == c
        if y[te].sum() < 2 or y[te].sum() == te.sum():
            continue
        m = LogisticRegression(max_iter=2000).fit(F[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(F[te])[:, 1]))
    return float(np.mean(aucs))


print(f"\nLoop decoding, {args.model} (leave-one-chromosome-out AUC; n={len(y)}, positives={int(y.sum())})")
print(f"  distance only        : {cv_auc(logd):.3f}")
print(f"  representations only : {cv_auc(X):.3f}")
print(f"  distance + reps      : {cv_auc(np.hstack([logd, X])):.3f}")
