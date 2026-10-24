#!/usr/bin/env bash
# End-to-end pipeline. Each step writes its outputs into $CAUSAL_WORKDIR.
# GPU steps (marked) need a CUDA GPU with at least 40 GB of memory for Borzoi; on an
# A40 the full pipeline takes roughly two to three days, dominated by the ISM/ICE runs.
set -euo pipefail
export CAUSAL_WORKDIR="${CAUSAL_WORKDIR:-$(pwd)/work}"
cd "$(dirname "$0")/.."

# 0. data and preprocessing (CPU)
bash scripts/download_data.sh
python scripts/prepare_loops.py
python scripts/prepare_exons.py
python scripts/prepare_hic_pairs.py
python scripts/setup_readout.py                                   # GPU (one forward pass)

# 1. structural probes (GPU)
python probes/loop_interchange.py --model borzoi   --loops K562_loops_hg38.tsv    --out borzoi_loops_K562.tsv
python probes/loop_interchange.py --model borzoi   --loops GM12878_loops_hg38.tsv --out borzoi_loops_GM12878.tsv
python probes/loop_interchange.py --model enformer --loops K562_loops_hg38.tsv    --out enf_loops_K562.tsv
python probes/loop_decode.py --model borzoi --limit 2000
python probes/loop_decode.py --model enformer --limit 2500
python probes/contact_map.py --model borzoi   --out borzoi_map.tsv
python probes/contact_map.py --model enformer --out enf_map.tsv

# 2. regulatory probes (GPU)
python probes/regulatory_ism_ice.py --benchmark train   --out iia_results.tsv
python probes/regulatory_ism_ice.py --benchmark heldout --out iia_heldout_results.tsv
python probes/regulatory_ism_ice_enformer.py --out enf_iia_results.tsv
python probes/emergent_contact.py

# 3. analyses (CPU)
python analysis/analyze_loops.py borzoi_loops_K562.tsv borzoi_loops_GM12878.tsv enf_loops_K562.tsv
python analysis/analyze_map.py borzoi_map.tsv enf_map.tsv
python analysis/ctcf_annotate.py borzoi_loops_K562.tsv
python scripts/annotate_anchor_tracks.py borzoi_loops_K562_annot.tsv \
    --peaks ctcf=ENCFF769AUF.bed.gz rad21=ENCFF439DYW.bed.gz smc3=ENCFF289LLT.bed.gz dnase=ENCFF185XRG.bed.gz \
    --regions umap=k100.umap.bed.gz
python analysis/interchange_identification.py borzoi_loops_K562.tsv
python analysis/sensitivity.py borzoi_loops_K562.tsv
for s in motif occupancy access; do
  python analysis/confounder_adjust.py borzoi_loops_K562_annot.tsv --set "$s"
done
python analysis/regulatory_stats.py --layer 4
python clade/run_clade.py borzoi_loops_K562_annot.tsv
python clade/run_clade.py borzoi_loops_GM12878.tsv --covariates span
