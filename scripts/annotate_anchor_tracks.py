"""Annotate loop anchors with ENCODE occupancy/accessibility peaks and mappability.

For each narrowPeak file the anchor value is the largest signalValue of peaks within
+/- FLANK bp of the anchor (0 if none). For bigWig files (e.g. Umap k100 mappability or
DNase signal) the anchor value is the mean over +/- FLANK bp. For every track two loop
covariates are written: <track>_min and <track>_sum over the two anchors.

Usage:
  python scripts/annotate_anchor_tracks.py borzoi_loops_K562_annot.tsv \
      --peaks ctcf=ENCFF769AUF.bed.gz rad21=ENCFF439DYW.bed.gz smc3=ENCFF289LLT.bed.gz \
              dnase=ENCFF185XRG.bed.gz \
      --regions umap=k100.umap.bed.gz
For --regions files (BED intervals without a signal column, e.g. Umap mappable regions)
the anchor value is the fraction of the +/- FLANK window covered by the intervals.
"""
import argparse
import gzip
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

sys.path.insert(0, "probes")
from common import path  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("table")
ap.add_argument("--peaks", nargs="*", default=[])
ap.add_argument("--bigwig", nargs="*", default=[])
ap.add_argument("--regions", nargs="*", default=[])
ap.add_argument("--flank", type=int, default=1500)
args = ap.parse_args()
R = pd.read_csv(path(args.table), sep="\t")


def read_peaks(fname):
    opener = gzip.open if fname.endswith(".gz") else open
    peaks = defaultdict(list)
    with opener(path(fname), "rt") as f:
        for ln in f:
            p = ln.split("\t")
            peaks[p[0]].append((int(p[1]), int(p[2]), float(p[6])))
    return {c: np.array(sorted(v)) for c, v in peaks.items()}


def peak_value(peaks, chrom, pos):
    arr = peaks.get(chrom)
    if arr is None or len(arr) == 0:
        return 0.0
    lo, hi = pos - args.flank, pos + args.flank
    m = (arr[:, 0] < hi) & (arr[:, 1] > lo)
    return float(arr[m, 2].max()) if m.any() else 0.0


for spec in args.peaks:
    name, fname = spec.split("=", 1)
    pk = read_peaks(fname)
    a = np.array([peak_value(pk, c, p) for c, p in zip(R.chrom, R.A)])
    b = np.array([peak_value(pk, c, p) for c, p in zip(R.chrom, R.B)])
    R[f"{name}_min"], R[f"{name}_sum"] = np.minimum(a, b), a + b

def read_regions(fname):
    opener = gzip.open if fname.endswith(".gz") else open
    reg = defaultdict(list)
    with opener(path(fname), "rt") as f:
        for ln in f:
            if ln.startswith(("track", "#")):
                continue
            p = ln.split("\t")
            reg[p[0]].append((int(p[1]), int(p[2])))
    return {c: np.array(sorted(v)) for c, v in reg.items()}


def covered_fraction(reg, chrom, pos):
    arr = reg.get(chrom)
    if arr is None or len(arr) == 0:
        return 0.0
    lo, hi = pos - args.flank, pos + args.flank
    m = (arr[:, 0] < hi) & (arr[:, 1] > lo)
    cov = np.clip(np.minimum(arr[m, 1], hi) - np.maximum(arr[m, 0], lo), 0, None).sum()
    return float(min(cov, hi - lo) / (hi - lo))


for spec in args.regions:
    name, fname = spec.split("=", 1)
    rg = read_regions(fname)
    a = np.array([covered_fraction(rg, c, p) for c, p in zip(R.chrom, R.A)])
    b = np.array([covered_fraction(rg, c, p) for c, p in zip(R.chrom, R.B)])
    R[f"{name}_min"], R[f"{name}_sum"] = np.minimum(a, b), a + b

if args.bigwig:
    import pyBigWig

    for spec in args.bigwig:
        name, fname = spec.split("=", 1)
        bw = pyBigWig.open(path(fname))

        def mean_at(c, p):
            v = bw.stats(c, max(0, p - args.flank), p + args.flank, type="mean")[0]
            return float(v) if v is not None else 0.0

        a = np.array([mean_at(c, p) for c, p in zip(R.chrom, R.A)])
        b = np.array([mean_at(c, p) for c, p in zip(R.chrom, R.B)])
        R[f"{name}_min"], R[f"{name}_sum"] = np.minimum(a, b), a + b

R.to_csv(path(args.table), sep="\t", index=False)
print("annotated", len(R), "loops with", [c for c in R.columns if c.endswith(("_min", "_sum"))])
