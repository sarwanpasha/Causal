"""ISM and ICE for CRISPRi enhancer-gene pairs with Enformer (K562 CAGE at the TSS).
Pairs are restricted to |distance| <= 90 kb so that element and TSS fall inside
Enformer's 196,608 bp input.

Usage: python probes/regulatory_ism_ice_enformer.py --out enf_iia_results.tsv
"""
import argparse

import numpy as np
import pandas as pd
import pyfaidx
import torch
from sklearn.metrics import average_precision_score

from common import Patch, device, forward, load_model, one_hot, path, shuffle_window, to_bool, window_sequence

ap = argparse.ArgumentParser()
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", default="enf_iia_results.tsv")
args = ap.parse_args()

SEQLEN, TPOS, TRES, OUTBINS = 196608, 1536, 128, 896
HALF = SEQLEN // 2
CROP = (TPOS - OUTBINS) // 2
CAGE = [4828, 5111]        # K562 CAGE tracks in the Enformer human head
DCAP = 90000
TSS_BIN = HALF // TRES - CROP
tss_bins = [TSS_BIN - 1, TSS_BIN, TSS_BIN + 1]
dev = device()
model, NL = load_model("enformer")
cage_t = torch.tensor(CAGE, device=dev)

P = [Patch() for _ in range(NL)]
for L in range(NL):
    model.transformer[L].register_forward_hook(P[L])


def set_all(mode, pos=None):
    for p in P:
        p.mode = mode
        if pos is not None:
            p.pos = pos


@torch.no_grad()
def readout(a):
    return float(forward(model, "enformer", a)[0][tss_bins][:, cage_t].sum().item())


genome = pyfaidx.Fasta(path("hg38.fa"))
geneset = set(pd.read_csv(path("target_gene_exons.tsv"), sep="\t").ensg.unique())
cr = pd.read_csv(path("EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz"), sep="\t")
cr["ensg"] = cr.measuredGeneEnsemblId.astype(str).str.split(".").str[0]
cr["Regulated"] = to_bool(cr.Regulated)
valid = to_bool(cr.ValidConnection) if "ValidConnection" in cr else pd.Series(True, index=cr.index)
df = cr[valid & (cr.distanceToTSS.abs() <= DCAP) & cr.ensg.isin(geneset)].reset_index(drop=True)
if args.limit:
    df = df.iloc[:args.limit]
print("pairs (|dist|<=90kb):", len(df), flush=True)

rng = np.random.default_rng(0)
res, cache = [], {}
for i, row in df.iterrows():
    ce, c, g = int(row.startTSS), str(row.chrTSS), row.ensg
    if (c, ce) not in cache:
        seq, _ = window_sequence(genome, c, ce, SEQLEN)
        a = one_hot(seq, False)
        set_all("off")
        cache[(c, ce)] = (a, readout(a))
    a, Rb = cache[(c, ce)]
    s0 = max(0, int(row.chromStart) - (ce - HALF))
    s1 = min(SEQLEN, int(row.chromEnd) - (ce - HALF))
    if s1 <= s0:
        continue
    pos = list(range(max(0, s0 // TRES), min(TPOS - 1, (s1 - 1) // TRES) + 1))
    set_all("cache", pos)
    Rcf = readout(shuffle_window(a, s0, s1, rng, False))
    set_all("off")
    rec = dict(name=row["name"], ensg=g, dist=int(row.distanceToTSS), EffectSize=float(row.EffectSize),
               Regulated=bool(row.Regulated), Rb=Rb, dR=Rb - Rcf, npos=len(pos))
    for L in range(NL):
        P[L].mode = "patch"
        rec[f"ICE{L}"] = Rb - readout(a)
        P[L].mode = "off"
    res.append(rec)
    if i % 50 == 0:
        print(f"{i}/{len(df)}", flush=True)

R = pd.DataFrame(res)
R.to_csv(path(args.out), sep="\t", index=False)
R["dR_frac"] = R.dR / (R.Rb.abs() + 1e-6)
R["neg_dist"] = -R.dist.abs()


def auprc(mask, col):
    y = R.Regulated[mask].astype(int)
    return average_precision_score(y, R[col][mask]) if 3 <= y.sum() < len(y) else float("nan")


print("\nAUPRC by distance to TSS (Enformer)")
for lo, hi in [(0, 1e4), (1e4, 5e4), (5e4, 9e4)]:
    m = (R.dist.abs() >= lo) & (R.dist.abs() < hi)
    print(f"[{int(lo // 1e3)}k,{int(hi // 1e3)}k) pos={int(R.Regulated[m].sum())} "
          f"ISM={auprc(m, 'dR_frac'):.3f} distance={auprc(m, 'neg_dist'):.3f}")
