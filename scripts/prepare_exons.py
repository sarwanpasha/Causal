"""Exon models (GENCODE v46) for every target gene in the CRISPRi benchmark and the
held-out cell-type set. Output: target_gene_exons.tsv (chrom, start, end, ensg)."""
import gzip
import sys

import pandas as pd

sys.path.insert(0, "probes")
from common import path  # noqa: E402

train = pd.read_csv(path("EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz"), sep="\t")
heldout = pd.read_csv(path("EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz"), sep="\t")


def genes(d):
    return set(d.measuredGeneEnsemblId.astype(str).str.split(".").str[0])


targets = genes(train) | genes(heldout)
rows = []
with gzip.open(path("gencode.v46.annotation.gtf.gz"), "rt") as f:
    for ln in f:
        if ln[0] == "#":
            continue
        p = ln.split("\t")
        if p[2] != "exon":
            continue
        gid = p[8].split('gene_id "')[1].split('"')[0].split(".")[0]
        if gid in targets:
            rows.append((p[0], int(p[3]), int(p[4]), gid))
ex = pd.DataFrame(rows, columns=["chrom", "start", "end", "ensg"]).drop_duplicates()
ex.to_csv(path("target_gene_exons.tsv"), sep="\t", index=False)
print("genes", ex.ensg.nunique(), "exons", len(ex))
