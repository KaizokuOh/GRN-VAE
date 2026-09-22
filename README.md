
A graph-aware variational autoencoder for reconstructing raw single-cell gene
counts. The model was developed on the dentate-gyrus train/test split used in
the scLDM experiments.

The released pipeline predicts the **1,330 genes retained by signed mouse
TRRUST activation/repression edges**. It does not claim to model every gene in
the source H5AD files.

## Architecture of the VAE 

**-Encoder**

<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/b387d7a7-f71f-486f-aea3-5d2e80a91e95" />

**-Decoder**

<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/37161936-a56d-4002-b56e-1b5a33bc4448" />

**-More details**
- Library-size-normalized `log1p` expression is encoded per gene.
- A shared FiLM-GNN layer passes relation-specific messages over activation,
  repression, and self edges.
- Learned queries pool gene states into `8 x 16` stochastic latent tokens.
- Gene queries cross-attend to the latent tokens and predict raw-count means.
- A softmax produces relative gene abundance, which is multiplied by each
  cell's library size.
- Training uses a Negative Binomial likelihood, KL regularization, and a
  gene-wise Pearson-correlation objective.

Optional architecture switches expose the ongoing ablations without changing
the baseline: learned gene-ID residuals over pretrained embeddings,
relation-wise graph aggregation, and additional decoder refinement blocks.

## Benchmark 
We run a comparison benchmark on the 3 metrics used by scLDM: MSE, NBLL, and gene-wise PCC. 
scLDM predictions were first generated on the full dentate gyrus (~17,000 genes) using that full library size, and then only the 1,300 genes (normalized on the new library size) that we trained on were selected
| Model | Test NB loss ↓ | Test MSE ↓ | Test gene-wise PCC ↑ |
|---|---:|---:|---:|
| scLDM | 0.261572 | 0.878416 | 0.444961 |
| **GRN-VAE + scGPT embeddings** | **0.218760** | **0.607442** | **0.710087** |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[tracking,dev]"
```

## Required inputs

Data and model weights are intentionally not included.

- Training and test H5AD files with gene names at `var/Gene` and raw CSR
  counts at `layers/X_counts`.
- Mouse TRRUST TSV rows in `regulator<TAB>target<TAB>mode` format.
- Optionally, a PyTorch gene-feature artifact containing `gene_names`,
  `embeddings`, and optionally `valid_mask`.

Training and test H5AD files must have the same gene order. The code selects
the signed TRRUST genes and adds one self edge per retained gene.

## Reproduce the selected configuration

```bash
dentate-graph-vae \
  --train-h5ad /path/to/dentategyrus_train.h5ad \
  --test-h5ad /path/to/dentategyrus_test.h5ad \
  --trrust /path/to/trrust_rawdata.mouse.tsv \
  --gene-features /path/to/scgpt_gene_embeddings.pt \
  --output checkpoints/candidate25_scgpt.pt \
  --max-epochs 500 \
  --patience 20 \
  --hidden-dim 96 \
  --latent-dim 16 \
  --latent-tokens 8 \
  --gnn-passes 2 \
  --dropout 0.02 \
  --beta 0.0001 \
  --mse-weight 0 \
  --pcc-weight 0.1 \
  --seed 46 \
  --dataloader-seed 46 \
  --wandb-mode offline \
  --wandb-project mouse-graph-vae-vs-scldm
```

The test file is evaluated only after the best validation checkpoint is
selected. Validation and test reporting is limited to NB loss,
log-normalized MSE, and gene-wise Pearson correlation.

## Optional architecture experiments

```bash
# Distinguish genes that share an imputed pretrained vector.
--gene-id-residual

# Normalize messages within each relation before combining relations.
--relation-aggregation relation_mean

# Add another latent-to-gene decoder refinement block.
--decoder-layers 2
```

## Tests

```bash
pytest
```

## Repository hygiene

`.gitignore` excludes H5AD files, model checkpoints, W&B runs, local virtual
environments, and data directories. Review dataset and pretrained-model
licenses before publishing any external artifacts.
