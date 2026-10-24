#!/usr/bin/env bash
# Download every public input into $CAUSAL_WORKDIR (default ./work).
# Total size is roughly 15 GB, dominated by the two genome FASTA files and the K562 .hic map.
set -euo pipefail
WORK="${CAUSAL_WORKDIR:-$(pwd)/work}"
mkdir -p "$WORK"
cd "$WORK"

fetch() { [ -s "$2" ] || curl -L --fail --retry 3 -o "$2" "$1"; }

# Reference genomes (GRCh38 for loops and CRISPRi pairs; hg19 to match the K562 .hic file)
fetch https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz hg38.fa.gz
fetch https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz hg19.fa.gz
[ -s hg38.fa ] || gunzip -k hg38.fa.gz
[ -s hg19.fa ] || gunzip -k hg19.fa.gz

# GENCODE v46 annotation (exon models of CRISPRi target genes)
fetch https://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_46/gencode.v46.annotation.gtf.gz \
      gencode.v46.annotation.gtf.gz

# ENCODE harmonized CRISPRi enhancer-gene benchmark (stored with git-lfs)
if [ ! -s EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz ]; then
  git clone --depth 1 https://github.com/EngreitzLab/CRISPR_comparison.git CRISPR_comparison
  (cd CRISPR_comparison && git lfs install --local && git lfs pull --include "resources/crispr_data/*GRCh38*")
  cp CRISPR_comparison/resources/crispr_data/EPCrisprBenchmark_ensemble_data_GRCh38.tsv.gz .
  cp CRISPR_comparison/resources/crispr_data/EPCrisprBenchmark_combined_data.heldout_5_cell_types.GRCh38.tsv.gz .
fi

# Borzoi targets table (code only; the weights are fetched by borzoi-pytorch on first use)
[ -d borzoi_repo ] || git clone --depth 1 https://github.com/calico/borzoi.git borzoi_repo

# HiCCUPS loop calls and the K562 in-situ combined Hi-C map (Rao et al. 2014, GEO GSE63525)
GEO=https://ftp.ncbi.nlm.nih.gov/geo/series/GSE63nnn/GSE63525/suppl
fetch "$GEO/GSE63525_K562_HiCCUPS_looplist.txt.gz" K562_loops.txt.gz
fetch "$GEO/GSE63525_GM12878_primary%2Breplicate_HiCCUPS_looplist.txt.gz" GM12878_loops.txt.gz
fetch https://hicfiles.s3.amazonaws.com/hiseq/k562/in-situ/combined.hic K562_combined_hg19.hic

# UCSC liftOver and the hg19 -> hg38 chain
fetch https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver liftOver && chmod +x liftOver
fetch https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz hg19ToHg38.over.chain.gz

# CTCF motif (JASPAR MA0139.1)
fetch "https://jaspar.elixir.no/api/v1/matrix/MA0139.1/?format=jaspar" MA0139.1.jaspar

# ENCODE K562 occupancy and accessibility peaks used for the confounder adjustments
for acc in ENCFF769AUF ENCFF439DYW ENCFF289LLT ENCFF185XRG; do
  [ -s "$acc.bed.gz" ] || curl -L --fail --retry 3 -o "$acc.bed.gz" \
      "https://www.encodeproject.org/files/$acc/@@download/$acc.bed.gz"
done

# Umap k100 single-read mappability (hg38)
fetch https://bismap.hoffmanlab.org/raw/hg38/k100.umap.bed.gz k100.umap.bed.gz

echo "All inputs are in $WORK"
