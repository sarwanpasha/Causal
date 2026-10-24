"""Emergent-contact regulatory probe (Borzoi).

For each CRISPRi element-gene pair, the per-block similarity descriptor of Eq. (1) is
computed between the element midpoint and the TSS. A logistic regression of the
regulation label on {descriptor, log distance} is evaluated with gene-grouped 5-fold
cross-validation and reported as AUPRC by distance stratum.

Usage: python probes/emergent_contact.py   (writes contact_features.tsv)
"""
import argparse

import numpy as np
import pandas as pd
import pyfaidx
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupKFold

from common import PairCapture, forward, load_model, one_hot, pair_similarity, path, to_bool, window_sequence

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--ism", default="iia_results.tsv", help="optional ISM results to merge as reference")
args = ap.parse_args()

SEQLEN, TPOS, TRES = 524288, 4096, 128
model, NL = load_model("borzoi")
genome = pyfaidx.Fasta(path("hg38.fa"))
geneset = set(pd.read_csv(path("target_gene_exons.tsv"), sep="\t").ensg.unique())
cr = pd.read_csv(path("EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz"), sep="\t")
cr["ensg"] = cr.measuredGeneEnsemblId.astype(str).str.split(".").str[0]
cr["Regulated"] = to_bool(cr.Regulated)
valid = to_bool(cr.ValidConnection) if "ValidConnection" in cr else pd.Series(True, index=cr.index)
df = cr[valid & (cr.distanceToTSS.abs() <= 250000) & cr.ensg.isin(geneset)].reset_index(drop=True)
if args.limit:
    df = df.iloc[:args.limit]

caps = [PairCapture() for _ in range(NL)]
for L in range(NL):
    model.transformer[L].register_forward_hook(caps[L])


@torch.no_grad()
def contact(c, elem, tss):
    seq, s = window_sequence(genome, c, (elem + tss) // 2, SEQLEN)
    pE, pT = (elem - s) // TRES, (tss - s) // TRES
    if not (0 <= pE < TPOS and 0 <= pT < TPOS):
        return None
    for cp in caps:
        cp.pA, cp.pB = pE, pT
    forward(model, "borzoi", one_hot(seq, True))
    f = []
    for L in range(NL):
        f += list(pair_similarity(caps[L].vA, caps[L].vB))
    return f


rows = []
for i, r in df.iterrows():
    elem = (int(r.chromStart) + int(r.chromEnd)) // 2
    f = contact(str(r.chrTSS), elem, int(r.startTSS))
    if f:
        rows.append([r["name"], r.ensg, int(r.distanceToTSS), bool(r.Regulated)] + f)
    if i % 100 == 0:
        print(f"{i}/{len(df)}", flush=True)
R = pd.DataFrame(rows, columns=["name", "ensg", "dist", "Regulated"] + [f"f{k}" for k in range(2 * NL)])
try:
    ism = pd.read_csv(path(args.ism), sep="\t")
    ism["ISM"] = ism.dR / (ism.Rb.abs() + 1e-6)
    R = R.merge(ism[["name", "ISM"]], on="name", how="left")
except FileNotFoundError:
    R["ISM"] = np.nan
R.to_csv(path("contact_features.tsv"), sep="\t", index=False)

y = R.Regulated.astype(int).values
logd = np.log10(R.dist.abs().clip(1)).values.reshape(-1, 1)
F = R[[f"f{k}" for k in range(2 * NL)]].values
groups = R.ensg.values


def cv_predict(X):
    pr = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        pr[te] = LogisticRegression(max_iter=2000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return pr


p_d, p_f, p_c = cv_predict(logd), cv_predict(F), cv_predict(np.hstack([logd, F]))


def auprc(p, m):
    return average_precision_score(y[m], p[m]) if y[m].sum() >= 3 else float("nan")


print(f"\nEmergent contact vs distance (gene-held-out AUPRC, n={len(R)})")
for lo, hi, nm in [(0, 1e10, "all"), (1e4, 5e4, "10-50kb"), (5e4, 1e5, "50-100kb"), (1e5, 2.6e5, ">100kb")]:
    m = ((R.dist.abs() >= lo) & (R.dist.abs() < hi)).values
    print(f"  {nm:8s} pos={int(y[m].sum()):4d} distance={auprc(p_d, m):.3f} "
          f"contact={auprc(p_f, m):.3f} distance+contact={auprc(p_c, m):.3f}")
