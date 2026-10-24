"""In-silico mutagenesis (ISM) and interchange counterfactual effect (ICE) for
CRISPRi enhancer-gene pairs with Borzoi.

For each pair the input window (524 kb) is centred on the target TSS. The readout R(x)
is predicted K562 RNA-seq coverage summed over the gene's exon bins (or, with
--readout cage, K562 CAGE at the 5 central TSS bins).
  ISM:  dR  = R(x) - R(x_shuf(e))              (element shuffled in the input)
  ICE:  ICE_l = R(x) - R(x with block-l activations at the element replaced by those
                       computed from x_shuf(e))

Usage:
  python probes/regulatory_ism_ice.py --benchmark train   --out iia_results.tsv
  python probes/regulatory_ism_ice.py --benchmark heldout --out iia_heldout_results.tsv
  python probes/regulatory_ism_ice.py --readout cage      --out iia_cage_results.tsv
  python probes/regulatory_ism_ice.py --shuffle dinuc --layers 4 --out iia_results_dinuc.tsv
"""
import argparse
import json

import numpy as np
import pandas as pd
import pyfaidx
import torch
from sklearn.metrics import average_precision_score

from common import (Patch, device, dinucleotide_shuffle_window, load_model, one_hot, path,
                    shuffle_window, to_bool, window_sequence)

ap = argparse.ArgumentParser()
ap.add_argument("--benchmark", choices=["train", "heldout"], default="train")
ap.add_argument("--readout", choices=["rna", "cage"], default="rna")
ap.add_argument("--shuffle", choices=["mono", "dinuc"], default="mono")
ap.add_argument("--layers", default="", help="comma-separated blocks to patch (default: all)")
ap.add_argument("--max-dist", type=int, default=250000)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--out", required=True)
args = ap.parse_args()

SEQLEN, BIN, TPOS, TRES = 524288, 32, 4096, 128
HALF = SEQLEN // 2
dev = device()
cfg = json.load(open(path("readout_config.json")))
n_bins = cfg["n_bins"]
PRED_HALF = n_bins * BIN // 2
tdf = pd.read_csv(cfg["targets_file"], sep="\t", index_col=0)
desc = tdf[tdf.columns[-1]].astype(str)
if args.readout == "rna":
    channels = tdf.index[desc.str.strip() == "RNA:K562"].tolist()
else:
    channels = tdf.index[desc.str.contains("CAGE", case=False) & desc.str.contains("K562", case=False)].tolist()
assert channels, "no K562 readout channels found"
ch_t = torch.tensor(channels, device=dev)

model, NL = load_model("borzoi")
layers = list(range(NL)) if not args.layers else [int(x) for x in args.layers.split(",")]
genome = pyfaidx.Fasta(path("hg38.fa"))
ex = pd.read_csv(path("target_gene_exons.tsv"), sep="\t")
exons = {g: list(zip(d.chrom, d.start, d.end)) for g, d in ex.groupby("ensg")}

if args.benchmark == "train":
    cr = pd.read_csv(path("EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz"), sep="\t")
else:
    cr = pd.read_csv(path("EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"), sep="\t")
    cr = cr[cr.CellType == "K562"].copy()
cr["ensg"] = cr.measuredGeneEnsemblId.astype(str).str.split(".").str[0]
cr["Regulated"] = to_bool(cr.Regulated)
valid = to_bool(cr.ValidConnection) if "ValidConnection" in cr else pd.Series(True, index=cr.index)
df = cr[valid & (cr.distanceToTSS.abs() <= args.max_dist) & cr.ensg.isin(exons)].reset_index(drop=True)
if args.limit:
    df = df.iloc[:args.limit]
print("pairs:", len(df), "positives:", int(df.Regulated.sum()), flush=True)

P = [Patch() for _ in range(NL)]
for L in range(NL):
    model.transformer[L].register_forward_hook(P[L])


def set_all(mode, pos=None):
    for p in P:
        p.mode = mode
        if pos is not None:
            p.pos = pos


def gene_bins(ensg, center):
    if args.readout == "cage":
        cb = n_bins // 2
        return list(range(cb - 2, cb + 3))
    lo = center - PRED_HALF
    idx = set()
    for _, st, en in exons[ensg]:
        b0, b1 = max(0, (st - lo) // BIN), min(n_bins - 1, (en - lo) // BIN)
        if b1 >= b0:
            idx.update(range(int(b0), int(b1) + 1))
    if not idx:
        c = n_bins // 2
        idx = set(range(c - 16, c + 16))
    return sorted(idx)


@torch.no_grad()
def readout(a, bins):
    out = model(torch.from_numpy(a)[None].to(dev))[0]
    return float(out[ch_t][:, bins].sum().item())


rng = np.random.default_rng(0)
res, cache = [], {}
for i, row in df.iterrows():
    center, chrom, g = int(row.startTSS), str(row.chrTSS), row.ensg
    key = (chrom, center, g)
    if key not in cache:
        seq, _ = window_sequence(genome, chrom, center, SEQLEN)
        a = one_hot(seq, True)
        bins = gene_bins(g, center)
        set_all("off")
        cache[key] = (a, bins, readout(a, bins))
    a, bins, Rb = cache[key]
    s0 = max(0, int(row.chromStart) - (center - HALF))
    s1 = min(SEQLEN, int(row.chromEnd) - (center - HALF))
    if s1 <= s0:
        continue
    pos = list(range(max(0, s0 // TRES), min(TPOS - 1, (s1 - 1) // TRES) + 1))
    a_s = dinucleotide_shuffle_window(a, s0, s1) if args.shuffle == "dinuc" else shuffle_window(a, s0, s1, rng, True)
    set_all("cache", pos)          # cache counterfactual activations at the element
    Rcf = readout(a_s, bins)       # ... while computing the ISM readout
    set_all("off")
    rec = dict(name=row["name"], gene=row.get("measuredGeneSymbol", ""), ensg=g, dist=int(row.distanceToTSS),
               EffectSize=float(row.EffectSize), Regulated=bool(row.Regulated), Rb=Rb, dR=Rb - Rcf, npos=len(pos))
    for L in layers:
        P[L].mode = "patch"
        rec[f"ICE{L}"] = Rb - readout(a, bins)
        P[L].mode = "off"
    res.append(rec)
    if i % 50 == 0:
        print(f"{i}/{len(df)}", flush=True)

R = pd.DataFrame(res)
R.to_csv(path(args.out), sep="\t", index=False)
R["dR_frac"] = R.dR / (R.Rb.abs() + 1e-6)
for L in layers:
    R[f"ICE{L}_frac"] = R[f"ICE{L}"] / (R.Rb.abs() + 1e-6)


def auprc(mask, col):
    y = R.Regulated[mask].astype(int)
    return average_precision_score(y, R[col][mask]) if 3 <= y.sum() < len(y) else float("nan")


print("\nAUPRC by distance to TSS")
for lo, hi in [(0, 1e4), (1e4, 5e4), (5e4, 1e5), (1e5, 2.6e5)]:
    m = (R.dist.abs() >= lo) & (R.dist.abs() < hi)
    line = f"[{int(lo // 1e3)}k,{int(hi // 1e3)}k) pos={int(R.Regulated[m].sum())} ISM={auprc(m, 'dR_frac'):.3f}"
    line += "".join(f" ICE{L}={auprc(m, f'ICE{L}_frac'):.3f}" for L in layers)
    print(line)
