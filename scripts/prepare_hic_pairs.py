"""Sample 120 non-overlapping 500 kb windows centred on K562 loop anchors and extract
KR-normalized observed/expected Hi-C values for all 10 kb bin pairs separated by
20 to 500 kb (hg19 coordinates, matching the .hic file).

Output: hic_pairs.tsv with columns win, chrom, center, binX, binY, dist, oe.
"""
import gzip
import random
import sys

import hicstraw
import numpy as np
import pandas as pd

sys.path.insert(0, "probes")
from common import path  # noqa: E402

RES, HALFW, NWIN = 10000, 250000, 120
hic = hicstraw.HiCFile(path("K562_combined_hg19.hic"))


def zoom(c):
    for norm in ("KR", "VC", "NONE"):
        try:
            return hic.getMatrixZoomData(c, c, "oe", norm, "BP", RES)
        except Exception:
            continue
    return None


centers = []
with gzip.open(path("K562_loops.txt.gz"), "rt") as f:
    f.readline()
    for ln in f:
        p = ln.split("\t")
        try:
            c, a = p[0], (int(p[1]) + int(p[2])) // 2
        except (ValueError, IndexError):
            continue
        if c in [str(i) for i in range(1, 23)]:
            centers.append((c, a))
random.seed(0)
random.shuffle(centers)
sel, used = [], set()
for c, a in centers:
    k = (c, a // 600000)
    if k in used:
        continue
    used.add(k)
    sel.append((c, a))
    if len(sel) >= NWIN:
        break

rows = []
for wi, (c, center) in enumerate(sel):
    s, e = center - HALFW, center + HALFW
    mzd = zoom(c)
    if mzd is None:
        continue
    try:
        recs = mzd.getRecords(s, e, s, e)
    except Exception as ex:
        print("skip", c, center, ex)
        continue
    oe = {(r.binX, r.binY): r.counts for r in recs}
    bins = list(range(s - (s % RES), e, RES))
    for i in range(len(bins)):
        for j in range(i + 1, len(bins)):
            bx, by = bins[i], bins[j]
            d = by - bx
            if d < 20000 or d > 500000:
                continue
            v = oe.get((bx, by), oe.get((by, bx)))
            if v and np.isfinite(v) and v > 0:
                rows.append((wi, c, center, bx, by, d, v))
    if wi % 20 == 0:
        print(wi, len(rows), flush=True)

pd.DataFrame(rows, columns=["win", "chrom", "center", "binX", "binY", "dist", "oe"]).to_csv(
    path("hic_pairs.tsv"), sep="\t", index=False)
print("windows", len(sel), "pairs", len(rows))
