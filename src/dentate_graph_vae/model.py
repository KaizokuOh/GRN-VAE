"""Graph-token variational autoencoder for raw single-cell gene counts."""

from dataclasses import dataclass

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .gnn import SharedFiLMGNN


@dataclass
class VAEOutput:
    """Outputs required by the count likelihood and latent regularizer."""

    mean_counts: Tensor
    mean: Tensor
    logvar: Tensor
    gene_order: Tensor | None = None


class FeedForward(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, values: Tensor) -> Tensor:
        return values + self.net(self.norm(values))


class DecoderRefinement(nn.Module):
    """One residual gene-query-to-latent cross-attention block."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.query_norm = nn.LayerNorm(hidden_dim)
        self.context_norm = nn.LayerNorm(hidden_dim)
        self.attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = FeedForward(hidden_dim, dropout)

    def forward(self, genes: Tensor, latent: Tensor) -> Tensor:
        context = self.context_norm(latent)
        update, _ = self.attention(
            self.query_norm(genes), context, context, need_weights=False
        )
        return self.ffn(genes + update)


class GraphTokenVAE(nn.Module):
    """Shared FiLM-GNN encoder with token pooling and a count decoder.

    The decoder predicts gene proportions with a softmax and multiplies them
    by the observed library size. Its output is therefore the Negative
    Binomial mean in raw-count space.
    """

    def __init__(
        self,
        num_genes: int,
        num_relations: int,
        hidden_dim: int = 96,
        latent_dim: int = 16,
        num_latent_tokens: int = 8,
        num_gnn_passes: int = 2,
        num_heads: int = 4,
        dropout: float = 0.02,
        gene_features: Tensor | None = None,
        gene_id_residual: bool = False,
        relation_aggregation: str = "global_mean",
        decoder_layers: int = 1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if decoder_layers < 1:
            raise ValueError("decoder_layers must be positive")
        if gene_id_residual and gene_features is None:
            raise ValueError("gene_id_residual requires pretrained gene features")

        self.num_genes = num_genes
        self.num_latent_tokens = num_latent_tokens
        self.maximum_logvar = 6.0

        if gene_features is not None:
            if gene_features.ndim != 2 or gene_features.shape[0] != num_genes:
                raise ValueError(
                    "gene_features must have shape [num_genes, feature_dim]"
                )
            self.register_buffer(
                "external_gene_features", gene_features.detach().float().clone()
            )
            self.external_gene_projection = nn.Linear(
                gene_features.shape[1], hidden_dim
            )
            self.encoder_gene_embedding = (
                nn.Embedding(
                    num_genes,
                    hidden_dim,
                    _weight=torch.zeros(num_genes, hidden_dim),
                )
                if gene_id_residual
                else None
            )
            self.decoder_gene_embedding = (
                nn.Embedding(
                    num_genes,
                    hidden_dim,
                    _weight=torch.zeros(num_genes, hidden_dim),
                )
                if gene_id_residual
                else None
            )
        else:
            self.register_buffer("external_gene_features", None)
            self.external_gene_projection = None
            self.encoder_gene_embedding = nn.Embedding(num_genes, hidden_dim)
            self.decoder_gene_embedding = nn.Embedding(num_genes, hidden_dim)

        self.expression_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU()
        )
        self.gnn = SharedFiLMGNN(
            hidden_dim,
            num_relations,
            num_passes=num_gnn_passes,
            dropout=dropout,
            relation_aggregation=relation_aggregation,
        )

        self.latent_queries = nn.Parameter(
            torch.randn(num_latent_tokens, hidden_dim) / hidden_dim**0.5
        )
        self.pool_norm = nn.LayerNorm(hidden_dim)
        self.pool_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.pool_ffn = FeedForward(hidden_dim, dropout)
        self.to_mean = nn.Linear(hidden_dim, latent_dim)
        self.to_logvar = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.to_logvar.weight)
        nn.init.constant_(self.to_logvar.bias, -4.0)

        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        self.decode_norm = nn.LayerNorm(hidden_dim)
        self.decode_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.decode_ffn = FeedForward(hidden_dim, dropout)
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, 1)
        )
        self.raw_inverse_dispersion = nn.Parameter(
            torch.full((num_genes,), 10.0)
        )
        self.decoder_refinement = nn.ModuleList(
            DecoderRefinement(hidden_dim, num_heads, dropout)
            for _ in range(decoder_layers - 1)
        )

    def gene_identity_features(
        self, gene_order: Tensor | None = None, *, decoder: bool = False
    ) -> Tensor:
        if gene_order is None:
            gene_order = torch.arange(
                self.num_genes, device=self.raw_inverse_dispersion.device
            )
        embedding = (
            self.decoder_gene_embedding if decoder else self.encoder_gene_embedding
        )
        if self.external_gene_features is None:
            return embedding(gene_order)
        features = self.external_gene_projection(
            self.external_gene_features[gene_order]
        )
        if embedding is not None:
            features = features + embedding(gene_order)
        return features

    def encode(
        self,
        expression: Tensor,
        adjacency_lists: list[Tensor],
        gene_order: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size = expression.shape[0]
        nodes = self.expression_encoder(expression.unsqueeze(-1))
        nodes = nodes + self.gene_identity_features(gene_order).unsqueeze(0)
        nodes = self.gnn(nodes, adjacency_lists)

        queries = self.latent_queries.unsqueeze(0).expand(batch_size, -1, -1)
        normalized_nodes = self.pool_norm(nodes)
        pooled, _ = self.pool_attention(
            queries, normalized_nodes, normalized_nodes, need_weights=False
        )
        pooled = self.pool_ffn(queries + pooled)
        raw_logvar = self.to_logvar(pooled)
        logvar = raw_logvar.clamp(-10.0, self.maximum_logvar)
        return self.to_mean(pooled), logvar

    def decode(
        self,
        latent: Tensor,
        library_size: Tensor,
        gene_order: Tensor | None = None,
    ) -> Tensor:
        batch_size = latent.shape[0]
        genes = self.gene_identity_features(
            gene_order, decoder=True
        ).unsqueeze(0).expand(batch_size, -1, -1)
        latent = self.latent_projection(latent)
        normalized_latent = self.decode_norm(latent)
        decoded, _ = self.decode_attention(
            genes, normalized_latent, normalized_latent, need_weights=False
        )
        decoded = self.decode_ffn(genes + decoded)
        for refinement in self.decoder_refinement:
            decoded = refinement(decoded, latent)
        logits = self.output_head(decoded).squeeze(-1)
        return torch.softmax(logits, dim=-1) * library_size

    def forward(
        self,
        expression: Tensor,
        adjacency_lists: list[Tensor],
        library_size: Tensor,
        *,
        sample: bool = True,
        gene_order: Tensor | None = None,
    ) -> VAEOutput:
        if gene_order is not None:
            if gene_order.shape != (self.num_genes,):
                raise ValueError("gene_order must contain one index per gene")
            gene_order = gene_order.to(expression.device, dtype=torch.long)
        mean, logvar = self.encode(expression, adjacency_lists, gene_order)
        latent = mean
        if sample:
            latent = mean + torch.randn_like(mean) * torch.exp(0.5 * logvar)
        return VAEOutput(
            self.decode(latent, library_size, gene_order),
            mean,
            logvar,
            gene_order,
        )

    def normalized_reconstruction(self, output: VAEOutput) -> Tensor:
        totals = output.mean_counts.sum(dim=1, keepdim=True).clamp_min(1e-8)
        return torch.log1p(output.mean_counts / totals * 1e4)

    @staticmethod
    def pearson(prediction: Tensor, target: Tensor, axis: str = "gene") -> Tensor:
        """Differentiable Pearson mean with finite zero-variance gradients."""

        if axis not in {"cell", "gene"}:
            raise ValueError("axis must be 'cell' or 'gene'")
        dimension = 1 if axis == "cell" else 0
        prediction = prediction.float()
        target = target.float()
        prediction = prediction - prediction.mean(dim=dimension, keepdim=True)
        target = target - target.mean(dim=dimension, keepdim=True)
        covariance = (prediction * target).mean(dim=dimension)
        prediction_variance = prediction.square().mean(dim=dimension)
        target_variance = target.square().mean(dim=dimension)
        epsilon = 1e-4
        denominator = (
            prediction_variance.clamp_min(epsilon).sqrt()
            * target_variance.clamp_min(epsilon).sqrt()
        )
        valid = target_variance > epsilon
        if not valid.any():
            return prediction.sum() * 0.0
        return (covariance[valid] / denominator[valid]).mean()

    def losses(
        self,
        counts: Tensor,
        expression: Tensor,
        output: VAEOutput,
        *,
        beta: float = 1e-4,
        mse_weight: float = 0.0,
        pcc_weight: float = 0.1,
    ) -> dict[str, Tensor]:
        raw_dispersion = self.raw_inverse_dispersion
        if output.gene_order is not None:
            raw_dispersion = raw_dispersion[output.gene_order]
        theta = F.softplus(raw_dispersion).clamp_min(1e-4)
        logits = output.mean_counts.clamp_min(1e-8).log() - theta.log()
        distribution = torch.distributions.NegativeBinomial(
            total_count=theta, logits=logits, validate_args=False
        )
        nb_loss = -distribution.log_prob(counts).mean()
        normalized = self.normalized_reconstruction(output)
        mse = F.mse_loss(normalized, expression)
        gene_pcc = self.pearson(normalized, expression, axis="gene")
        kl = -0.5 * (
            1 + output.logvar - output.mean.square() - output.logvar.exp()
        ).mean()
        loss = (
            nb_loss
            + beta * kl
            + mse_weight * mse
            + pcc_weight * (1.0 - gene_pcc)
        )
        return {
            "loss": loss,
            "nb_loss": nb_loss,
            "mse": mse,
            "gene_pcc": gene_pcc,
            "kl": kl,
        }
