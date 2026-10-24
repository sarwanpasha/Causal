"""Causal interchange probe for chromatin loops (Algorithm 1) plus anchor-pair similarity features.

For every loop (A, B) the script perturbs a 10 kb window around anchor A by a random
permutation of its positions and measures, at every transformer block, the relative change
of the representation at three targets: the true partner B, a mispaired real anchor B'
at matched displacement (0.8d to 1.2d), and the distance-only control A - d.
It also records the unperturbed anchor-to-anchor cosine and normalized distance
(Eq. 1) at every block, which are the representation features used by CLADE.

Usage:
  python probes/loop_interchange.py --model borzoi --loops K562_loops_hg38.tsv --out borzoi_loops_K562.tsv
  python probes/loop_interchange.py --model enformer --loops K562_loops_hg38.tsv --out enf_loops_K562.tsv
"""
import argparse
import bisect
from collections import defaultdict

import numpy as np
import pandas as pd
import pyfaidx
import torch

from common import (MODEL_SPECS, PairCapture, forward, load_model, one_hot, pair_similarity,
                    path, shuffle_window, window_sequence)

ap = argparse.ArgumentParser()
ap.add_argument("--model", choices=["borzoi", "enformer"], default="borzoi")
ap.add_argument("--loops", required=True, help="loop table (chrom, anchorA, anchorB, span, strength)")
ap.add_argument("--out", required=True)
ap.add_argument("--genome", default="hg38.fa")
ap.add_argument("--half-window", type=int, default=5000, help="half width of the perturbed window (bp)")
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()

spec = MODEL_SPECS[args.model]
SEQLEN, TPOS, TRES, CF = spec["seqlen"], spec["tpos"], spec["tres"], spec["channels_first"]
model, NL = load_model(args.model)
genome = pyfaidx.Fasta(path(args.genome))
loops = pd.read_csv(path(args.loops), sep="\t")
if args.model == "enformer":
    # Enformer sees +/- 98 kb; keep loops whose anchors fit in one input window
    loops = loops[loops.span < 180000].reset_index(drop=True)

anchors = defaultdict(list)
for _, r in loops.iterrows():
    anchors[str(r.chrom)] += [int(r.anchorA), int(r.anchorB)]
for c in anchors:
    anchors[c] = sorted(set(anchors[c]))


def mispair(c, A, B, d):
    """A real anchor at displacement in [0.8d, 1.2d] from A that is not A or B."""
    arr = anchors[c]
    t = A + d
    lo = bisect.bisect_left(arr, int(t - 0.2 * d))
    hi = bisect.bisect_right(arr, int(t + 0.2 * d))
    cand = [x for x in arr[lo:hi] if x not in (A, B)]
    return cand[len(cand) // 2] if cand else None


if args.limit:
    loops = loops.iloc[:args.limit]
rng = np.random.default_rng(args.seed)

caps = [PairCapture() for _ in range(NL)]
for L in range(NL):
    model.transformer[L].register_forward_hook(caps[L])


@torch.no_grad()
def reps(a, pA, pB):
    for c in caps:
        c.pA, c.pB = pA, pB
    forward(model, args.model, a)
    return [(caps[L].vA, caps[L].vB) for L in range(NL)]


def measure(c, pa, pb, want_sim=False):
    """Interchange influence I_l(A -> T) at every block; optionally the unperturbed similarity."""
    seq, s = window_sequence(genome, c, (pa + pb) // 2, SEQLEN)
    a = one_hot(seq, CF)
    pA, pB = (pa - s) // TRES, (pb - s) // TRES
    if not (0 <= pA < TPOS and 0 <= pB < TPOS):
        return None
    s0 = max(0, pa - args.half_window - s)
    s1 = min(SEQLEN, pa + args.half_window - s)
    if s1 <= s0:
        return None
    base = reps(a, pA, pB)
    shuf = reps(shuffle_window(a, s0, s1, rng, CF), pA, pB)
    infl = [float(torch.linalg.norm(base[L][1] - shuf[L][1]) / (torch.linalg.norm(base[L][1]) + 1e-6))
            for L in range(NL)]
    if not want_sim:
        return infl, None
    sim = []
    for L in range(NL):
        sim += list(pair_similarity(*base[L]))
    return infl, sim


rows = []
for i, r in loops.iterrows():
    c, A, B = str(r.chrom), int(r.anchorA), int(r.anchorB)
    d = B - A
    st = float(r.strength) if "strength" in r else np.nan
    mp = mispair(c, A, B, d)
    real = measure(c, A, B, want_sim=True)
    mis = measure(c, A, mp) if mp else None
    dc = measure(c, A, A - d)
    if real and mis and dc:
        rows.append([c, A, B, d, st] + real[0] + mis[0] + dc[0] + real[1])
    if i % 100 == 0:
        print(f"{i}/{len(loops)}", flush=True)

cols = (["chrom", "A", "B", "span", "strength"]
        + [f"loop_L{L}" for L in range(NL)] + [f"mis_L{L}" for L in range(NL)]
        + [f"dist_L{L}" for L in range(NL)] + sum([[f"cos_L{L}", f"nd_L{L}"] for L in range(NL)], []))
pd.DataFrame(rows, columns=cols).to_csv(path(args.out), sep="\t", index=False)
print("wrote", path(args.out), len(rows))
