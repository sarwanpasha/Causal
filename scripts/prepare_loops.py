"""Lift HiCCUPS loop calls (Rao et al. 2014, GSE63525, hg19) to hg38 and keep
intra-chromosomal loops spanning 50 to 450 kb. Loop strength is the observed /
expected (donut) enrichment reported by HiCCUPS.

Outputs: K562_loops_hg38.tsv and GM12878_loops_hg38.tsv
(columns chrom, anchorA, anchorB, span, strength).
"""
import gzip
import subprocess
import sys

import pandas as pd

sys.path.insert(0, "probes")
from common import path  # noqa: E402


def lift(gz, tag):
    strength = {}
    with gzip.open(path(gz), "rt") as f:
        header = f.readline().rstrip("\n").split("\t")
        oi, ei = header.index("o"), header.index("e_donut")
        with open(path(f"{tag}_anc.bed"), "w") as out:
            for i, ln in enumerate(f):
                p = ln.rstrip("\n").split("\t")
                try:
                    a = (int(p[1]) + int(p[2])) // 2
                    b = (int(p[4]) + int(p[5])) // 2
                    o, e = float(p[oi]), float(p[ei])
                except (ValueError, IndexError):
                    continue
                strength[i] = o / e if e > 0 else float("nan")
                out.write(f"chr{p[0]}\t{a}\t{a + 1}\t{tag}{i}_A\nchr{p[3]}\t{b}\t{b + 1}\t{tag}{i}_B\n")
    subprocess.run([path("liftOver"), path(f"{tag}_anc.bed"), path("hg19ToHg38.over.chain.gz"),
                    path(f"{tag}_hg38.bed"), path(f"{tag}_unmapped.bed")], check=True)
    pos = {}
    for ln in open(path(f"{tag}_hg38.bed")):
        c, s, _, name = ln.split("\t")[:4]
        pos[name.strip()] = (c, int(s))
    rows = []
    for k in set(n.rsplit("_", 1)[0] for n in pos):
        A, B = pos.get(k + "_A"), pos.get(k + "_B")
        idx = int(k[len(tag):])
        if A and B and A[0] == B[0]:
            d = abs(B[1] - A[1])
            if 50000 < d < 450000:
                rows.append((A[0], min(A[1], B[1]), max(A[1], B[1]), d, strength.get(idx, float("nan"))))
    pd.DataFrame(sorted(rows), columns=["chrom", "anchorA", "anchorB", "span", "strength"]).to_csv(
        path(f"{tag}_loops_hg38.tsv"), sep="\t", index=False)
    print(tag, "loops:", len(rows))


lift("K562_loops.txt.gz", "K562")
lift("GM12878_loops.txt.gz", "GM12878")
