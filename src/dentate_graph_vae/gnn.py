"""Relation-aware FiLM message passing for a fixed gene-regulatory graph."""

import torch
from torch import Tensor, nn


class FiLMGNNLayer(nn.Module):
    """Apply target-conditioned FiLM messages for each edge relation."""

    def __init__(
        self,
        hidden_dim: int,
        num_relations: int,
        relation_aggregation: str = "global_mean",
    ) -> None:
        super().__init__()
        if relation_aggregation not in {"global_mean", "relation_mean"}:
            raise ValueError(
                "relation_aggregation must be 'global_mean' or 'relation_mean'"
            )
        self.relation_aggregation = relation_aggregation
        self.messages = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False)
            for _ in range(num_relations)
        )
        self.film = nn.ModuleList(
            nn.Linear(hidden_dim, 2 * hidden_dim)
            for _ in range(num_relations)
        )
        self.normalization = nn.LayerNorm(hidden_dim)
        self.activation = nn.SiLU()

        # Begin with gamma=1 and beta=0, then learn relation-specific modulation.
        for generator in self.film:
            nn.init.zeros_(generator.weight)
            nn.init.zeros_(generator.bias)
            with torch.no_grad():
                generator.bias[:hidden_dim].fill_(1.0)

    def forward(self, nodes: Tensor, adjacency_lists: list[Tensor]) -> Tensor:
        if len(adjacency_lists) != len(self.messages):
            raise ValueError("one adjacency list is required per relation")

        _, num_genes, _ = nodes.shape
        aggregated = torch.zeros_like(nodes)
        denominator = nodes.new_zeros(num_genes)

        for transform, generator, edges in zip(
            self.messages, self.film, adjacency_lists
        ):
            edges = edges.to(device=nodes.device, dtype=torch.long)
            if edges.numel() == 0:
                continue
            source, target = edges[:, 0], edges[:, 1]
            transformed = transform(nodes)
            gamma, beta = generator(nodes).chunk(2, dim=-1)

            relation_sum = torch.zeros_like(nodes)
            relation_sum.index_add_(1, target, transformed[:, source])
            relation_degree = nodes.new_zeros(num_genes)
            relation_degree.index_add_(
                0, target, torch.ones_like(target, dtype=nodes.dtype)
            )

            if self.relation_aggregation == "global_mean":
                aggregated += (
                    gamma * relation_sum
                    + beta * relation_degree.view(1, -1, 1)
                )
                denominator += relation_degree
            else:
                present = (relation_degree > 0).to(nodes.dtype)
                relation_mean = relation_sum / relation_degree.clamp_min(1).view(
                    1, -1, 1
                )
                aggregated += (
                    gamma * relation_mean
                    + beta * present.view(1, -1, 1)
                )
                denominator += present

        update = self.activation(
            aggregated / denominator.clamp_min(1).view(1, -1, 1)
        )
        # The selected architecture normalizes the update before adding it.
        return nodes + self.normalization(update)


class SharedFiLMGNN(nn.Module):
    """Reuse one FiLM-GNN layer for a configurable number of passes."""

    def __init__(
        self,
        hidden_dim: int,
        num_relations: int,
        num_passes: int = 2,
        dropout: float = 0.02,
        relation_aggregation: str = "global_mean",
    ) -> None:
        super().__init__()
        if num_passes < 1:
            raise ValueError("num_passes must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.num_passes = num_passes
        self.layer = FiLMGNNLayer(
            hidden_dim, num_relations, relation_aggregation
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, nodes: Tensor, adjacency_lists: list[Tensor]) -> Tensor:
        for _ in range(self.num_passes):
            nodes = self.dropout(self.layer(nodes, adjacency_lists))
        return nodes
