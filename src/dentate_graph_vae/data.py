"""Load raw dentate-gyrus counts and align them to the mouse TRRUST graph."""

import csv
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse
import torch
from torch.utils.data import Dataset


class SparseCountsDataset(Dataset):
    """Expose rows of a CSR raw-count matrix as dense float tensors."""

    def __init__(self, counts: sparse.csr_matrix) -> None:
        self.counts = counts

    def __len__(self) -> int:
        return self.counts.shape[0]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.counts[index].toarray().ravel().astype(np.float32, copy=False)
        return torch.from_numpy(row), torch.tensor(0)

    def __getitems__(
        self, indices: list[int]
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        rows = self.counts[indices].toarray().astype(np.float32, copy=False)
        return [(torch.from_numpy(row), torch.tensor(0)) for row in rows]


def _decode_strings(values: np.ndarray) -> list[str]:
    return [
        value.decode() if isinstance(value, bytes) else str(value)
        for value in values
    ]


def _read_gene_names(path: Path) -> list[str]:
    with h5py.File(path, "r") as handle:
        if "var/Gene" not in handle:
            raise ValueError(f"{path} does not contain var/Gene")
        return _decode_strings(handle["var/Gene"][:])


def _read_counts(path: Path, source_indices: np.ndarray) -> sparse.csr_matrix:
    with h5py.File(path, "r") as handle:
        if "layers/X_counts" not in handle:
            raise ValueError(f"{path} does not contain layers/X_counts")
        group = handle["layers/X_counts"]
        shape = tuple(int(value) for value in group.attrs["shape"])
        counts = sparse.csr_matrix(
            (group["data"][:], group["indices"][:], group["indptr"][:]),
            shape=shape,
        )[:, source_indices].tocsr()
    if counts.data.size:
        counts.data = np.rint(counts.data).astype(np.float32, copy=False)
    return counts


def build_trrust_graph(
    trrust_path: str | Path, expression_genes: list[str]
) -> tuple[dict, np.ndarray]:
    """Build activation, repression, and self-edge adjacency lists."""

    signed_edges: set[tuple[str, str, str]] = set()
    with Path(trrust_path).open(encoding="utf-8") as handle:
        for line_number, row in enumerate(
            csv.reader(handle, delimiter="\t"), start=1
        ):
            if len(row) < 3:
                raise ValueError(f"invalid TRRUST row {line_number}: {row}")
            regulator, target, mode = (value.strip() for value in row[:3])
            relation = mode.lower()
            if relation in {"activation", "repression"}:
                signed_edges.add((regulator, target, relation))

    available = set(expression_genes)
    matched_edges = sorted(
        edge
        for edge in signed_edges
        if edge[0] in available and edge[1] in available
    )
    retained = {gene for edge in matched_edges for gene in edge[:2]}
    gene_names = [gene for gene in expression_genes if gene in retained]
    if not gene_names:
        raise ValueError("no TRRUST genes matched the expression matrix")

    source_indices = np.asarray(
        [
            index
            for index, gene in enumerate(expression_genes)
            if gene in retained
        ],
        dtype=np.int64,
    )
    gene_to_index = {gene: index for index, gene in enumerate(gene_names)}
    adjacency_lists = []
    for relation in ("activation", "repression"):
        edges = [
            (gene_to_index[source], gene_to_index[target])
            for source, target, edge_relation in matched_edges
            if edge_relation == relation
        ]
        adjacency_lists.append(
            torch.tensor(edges, dtype=torch.long).reshape(-1, 2)
        )
    node_ids = torch.arange(len(gene_names), dtype=torch.long)
    adjacency_lists.append(torch.stack((node_ids, node_ids), dim=1))
    return {
        "adjacency_lists": adjacency_lists,
        "gene_names": gene_names,
        "relation_names": ("activation", "repression", "self"),
        "num_genes": len(gene_names),
    }, source_indices


class DentateGyrusData:
    """Aligned training/test counts and the signed mouse TRRUST graph."""

    def __init__(
        self,
        train_h5ad: str | Path,
        test_h5ad: str | Path,
        trrust_tsv: str | Path,
    ) -> None:
        train_path = Path(train_h5ad)
        test_path = Path(test_h5ad)
        train_genes = _read_gene_names(train_path)
        test_genes = _read_gene_names(test_path)
        if train_genes != test_genes:
            raise ValueError("training and test H5AD gene orders differ")

        self.graph, source_indices = build_trrust_graph(
            trrust_tsv, train_genes
        )
        self.train = SparseCountsDataset(
            _read_counts(train_path, source_indices)
        )
        self.test = SparseCountsDataset(
            _read_counts(test_path, source_indices)
        )


def load_gene_features(
    path: str | Path, gene_names: list[str]
) -> tuple[torch.Tensor, dict]:
    """Align a pretrained embedding artifact to the model's gene order."""

    artifact = torch.load(path, map_location="cpu", weights_only=False)
    source_names = list(artifact["gene_names"])
    if len(source_names) != len(set(source_names)):
        raise ValueError("gene feature artifact contains duplicate names")
    source_index = {gene: index for index, gene in enumerate(source_names)}
    missing = [gene for gene in gene_names if gene not in source_index]
    if missing:
        raise ValueError(f"gene feature artifact lacks required genes: {missing[:10]}")

    order = torch.tensor([source_index[gene] for gene in gene_names])
    features = artifact["embeddings"].float()[order]
    valid = artifact.get(
        "valid_mask", torch.ones(len(source_names), dtype=torch.bool)
    ).bool()[order]
    valid &= torch.isfinite(features).all(dim=1)
    if not valid.any():
        raise ValueError("gene feature artifact has no valid vectors")
    features[~valid] = features[valid].mean(dim=0)
    report = {
        "source": artifact.get("source", "unknown"),
        "dimension": int(features.shape[1]),
        "valid_genes": int(valid.sum()),
        "imputed_genes": int((~valid).sum()),
    }
    return features, report
