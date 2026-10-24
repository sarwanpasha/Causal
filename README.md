# CLADE: deconfounded probing of genomic sequence models

This repository contains the code for probing what frozen sequence-to-expression models
(Borzoi and Enformer) encode about three-dimensional genome folding and enhancer-gene
regulation, and the CLADE estimator for covariate-adjusted, jointly estimated effects of
layer representations on a measured phenotype.

The repository has two parts:

* **CLADE estimator** (`clade/`). Needs only NumPy and SciPy, runs on a laptop, and comes
  with a synthetic example that has a known answer.
* **Probing pipeline** (`probes/`, `analysis/`, `scripts/`). Runs Borzoi and Enformer on
  public data. The forward passes need a CUDA GPU.

## Installation

```bash
git clone https://github.com/sarwanpasha/Causal.git
cd Causal
python -m venv .venv && source .venv/bin/activate

# CLADE, the analysis scripts and the examples (CPU only)
pip install -r requirements-minimal.txt

# full pipeline (GPU): install PyTorch for your CUDA version first
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The model probes were run with Python 3.11, PyTorch 2.x and an NVIDIA A40 (48 GB).

## Quick start: CLADE on synthetic data (about 2 seconds, CPU)

```bash
python examples/run_clade_synthetic.py
```

The simulation has eight layers whose features drift smoothly with depth, covariates that
drive both the features and the phenotype, and a true effect only at the last layer. The
script prints the naive (confounded) slopes, the CLADE per-layer estimates with standard
errors, the spectrum of the cross-layer correlation, the coupled Wald test on the two
deepest layers, the max-t and Bonferroni critical values, the GLS depth trend, and the
omitted-variable robustness value. It then repeats the simulation 200 times and reports
bias, standard-error calibration and 95% interval coverage for the last layer.

To try the analysis scripts on a synthetic loop table with the same columns as the real
probe output:

```bash
export CAUSAL_WORKDIR=$PWD/work
python examples/make_synthetic_loops.py
python analysis/interchange_identification.py synthetic_loops.tsv
python analysis/sensitivity.py synthetic_loops.tsv
python analysis/analyze_loops.py synthetic_loops.tsv
python analysis/confounder_adjust.py synthetic_loops.tsv --set motif
python clade/run_clade.py synthetic_loops.tsv
```

Unit tests: `python -m pytest tests`.

## Using CLADE on your own data

```python
import sys
sys.path.insert(0, "clade")    # path to the clade/ folder of this repository
import clade

# Y: (n,) phenotype; R: (n, L) one feature per layer; W: (n, p) covariates
f = clade.fit(Y, R, W)
print(clade.summary(f, deep_block=[L - 2, L - 1]))
W_stat, df, p = clade.block_wald(f, [L - 2, L - 1])
rv = clade.robustness_value(f.t_homo[-1], f.df)            # Cinelli and Hazlett (2020)
```

`fit` residualizes the phenotype and every layer feature on the covariates (least squares
with an intercept), returns the per-layer partial-regression coefficients, their
heteroskedasticity-robust standard errors, and the joint covariance built from the stacked
influence functions.

## Full pipeline

All inputs and outputs live in one working directory, set with `CAUSAL_WORKDIR`
(default `./work`). `scripts/run_pipeline.sh` runs every step in order; the individual
steps are:

| Step | Script | Output |
|---|---|---|
| Download public data (about 15 GB) | `scripts/download_data.sh` | genomes, loops, Hi-C, CRISPRi benchmark, ENCODE tracks |
| Lift loops to hg38, 50 to 450 kb | `scripts/prepare_loops.py` | `K562_loops_hg38.tsv`, `GM12878_loops_hg38.tsv` |
| Exon models of target genes | `scripts/prepare_exons.py` | `target_gene_exons.tsv` |
| Hi-C O/E bin pairs, 120 windows | `scripts/prepare_hic_pairs.py` | `hic_pairs.tsv` |
| Borzoi output layout | `scripts/setup_readout.py` | `readout_config.json` |
| Causal interchange probe and anchor similarity | `probes/loop_interchange.py` | `borzoi_loops_K562.tsv`, ... |
| Linear decoding probe | `probes/loop_decode.py` | AUC printed |
| Contact-map features | `probes/contact_map.py` | `borzoi_map.tsv`, `enf_map.tsv` |
| ISM and ICE for CRISPRi pairs | `probes/regulatory_ism_ice.py`, `probes/regulatory_ism_ice_enformer.py` | `iia_results.tsv`, ... |
| Emergent-contact regulatory probe | `probes/emergent_contact.py` | `contact_features.tsv` |
| Interchange summary and loop strength | `analysis/analyze_loops.py` | printed |
| Contact-map reconstruction | `analysis/analyze_map.py` | printed |
| CTCF motif annotation and orientation | `analysis/ctcf_annotate.py` | `*_annot.tsv` |
| Anchor occupancy, accessibility, mappability | `scripts/annotate_anchor_tracks.py` | columns added to `*_annot.tsv` |
| Decomposition and calibration of the interchange contrast | `analysis/interchange_identification.py` | `interchange_identification.tsv` |
| Rosenbaum sensitivity per layer | `analysis/sensitivity.py` | `sensitivity.tsv` |
| Measured-confounder adjustment | `analysis/confounder_adjust.py` | printed |
| ICE versus ISM with gene-clustered bootstrap | `analysis/regulatory_stats.py` | printed |
| CLADE on loops | `clade/run_clade.py` | `*_clade.tsv`, `*_clade_corr.tsv` |

`clade/run_clade.py --covariates` selects the adjustment set. The default is the CTCF
grammar panel (span, motif scores at both anchors and orientation class); a loop table
without motif annotation can be run with a reduced set such as `--covariates span`.

## Data sources

* Reference genomes GRCh38 and hg19, UCSC liftOver and the hg19-to-hg38 chain (UCSC Genome Browser)
* HiCCUPS loops and the K562 in-situ Hi-C map, GEO accession GSE63525
* ENCODE harmonized CRISPRi enhancer-gene benchmark (EngreitzLab/CRISPR_comparison)
* GENCODE release 46 annotation
* JASPAR CTCF profile MA0139.1
* ENCODE K562 peaks: CTCF ENCFF769AUF, RAD21 ENCFF439DYW, SMC3 ENCFF289LLT, DNase ENCFF185XRG
* Umap k100 mappability, hg38 (Bismap)
* Model weights: `johahi/borzoi-replicate-0` and `EleutherAI/enformer-official-rough` (Hugging Face)

## License

MIT (see `LICENSE`).
