"""Contact-map reconstruction features.

For every pair of 10 kb bins in the windows listed in hic_pairs.tsv (built by
scripts/prepare_hic_pairs.py), computes the per-block similarity descriptor of Eq. (1)
between the transformer positions at the two bin centres. Sequences come from hg19 to
match the Hi-C assembly.

Usage:
  python probes/contact_map.py --model borzoi   --out borzoi_map.tsv
  python probes/contact_map.py --model enformer --out enf_map.tsv
"""
import argparse

import pandas as pd
import pyfaidx
import torch

from common import MODEL_SPECS, FullCapture, forward, load_model, one_hot, pair_similarity, path, window_sequence

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["borzoi", "enformer"], default="borzoi")
ap.add_argument("--pairs", default="hic_pairs.tsv")
ap.add_argument("--genome", default="hg19.fa")
ap.add_argument("--out", required=True)
args = ap.parse_args()

spec = MODEL_SPECS[args.model]
SEQLEN, TPOS, TRES, CF = spec["seqlen"], spec["tpos"], spec["tres"], spec["channels_first"]
model, NL = load_model(args.model)
genome = pyfaidx.Fasta(path(args.genome))
P = pd.read_csv(path(args.pairs), sep="\t")

caps = [FullCapture() for _ in range(NL)]
for L in range(NL):
    model.transformer[L].register_forward_hook(caps[L])

out = []
for wi, g in P.groupby("win"):
    c = str(g.chrom.iloc[0])
    center = int(g.center.iloc[0])
    seq, s = window_sequence(genome, "chr" + c, center, SEQLEN)
    with torch.no_grad():
        forward(model, args.model, one_hot(seq, CF))
    reps = [caps[L].v for L in range(NL)]
    for _, r in g.iterrows():
        px = (int(r.binX) + 5000 - s) // TRES
        py = (int(r.binY) + 5000 - s) // TRES
        if not (0 <= px < TPOS and 0 <= py < TPOS):
            continue
        f = []
        for L in range(NL):
            f += list(pair_similarity(reps[L][px], reps[L][py]))
        out.append([int(r.win), c, int(r.binX), int(r.binY), int(r.dist), float(r.oe)] + f)
    if int(wi) % 20 == 0:
        print(f"window {wi}", flush=True)

cols = ["win", "chrom", "binX", "binY", "dist", "oe"] + sum([[f"cos_L{L}", f"nd_L{L}"] for L in range(NL)], [])
pd.DataFrame(out, columns=cols).to_csv(path(args.out), sep="\t", index=False)
print("rows", len(out))
