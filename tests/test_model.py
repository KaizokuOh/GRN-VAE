import torch

from dentate_graph_vae.model import GraphTokenVAE


def toy_edges(num_genes: int) -> list[torch.Tensor]:
    return [
        torch.tensor([[0, 1], [2, 3], [4, 1]]),
        torch.tensor([[1, 2], [3, 0]]),
        torch.arange(num_genes).repeat(2, 1).T,
    ]


def test_forward_loss_and_backward_are_finite():
    torch.manual_seed(7)
    model = GraphTokenVAE(
        num_genes=5,
        num_relations=3,
        hidden_dim=8,
        latent_dim=4,
        num_latent_tokens=2,
        num_heads=2,
        dropout=0.0,
        gene_features=torch.randn(5, 11),
        gene_id_residual=True,
        relation_aggregation="relation_mean",
        decoder_layers=2,
    )
    counts = torch.poisson(torch.full((6, 5), 2.0))
    expression = torch.log1p(
        counts / counts.sum(dim=1, keepdim=True).clamp_min(1.0) * 1e4
    )
    library_size = counts.sum(dim=1, keepdim=True)
    output = model(expression, toy_edges(5), library_size)
    losses = model.losses(counts, expression, output)
    losses["loss"].backward()

    assert torch.isfinite(output.mean_counts).all()
    assert torch.isfinite(losses["loss"])
    assert all(
        parameter.grad is None or torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
    )


def test_decoder_preserves_library_size():
    model = GraphTokenVAE(
        num_genes=5,
        num_relations=3,
        hidden_dim=8,
        latent_dim=4,
        num_latent_tokens=2,
        num_heads=2,
        dropout=0.0,
    ).eval()
    expression = torch.randn(3, 5)
    library_size = torch.tensor([[20.0], [31.0], [12.0]])
    with torch.no_grad():
        output = model(
            expression, toy_edges(5), library_size, sample=False
        )
    torch.testing.assert_close(
        output.mean_counts.sum(dim=1, keepdim=True), library_size
    )


def test_gene_permutation_equivariance_with_all_optional_modules():
    torch.manual_seed(11)
    num_genes = 5
    model = GraphTokenVAE(
        num_genes=num_genes,
        num_relations=3,
        hidden_dim=8,
        latent_dim=4,
        num_latent_tokens=2,
        num_heads=2,
        dropout=0.0,
        gene_features=torch.randn(num_genes, 7),
        gene_id_residual=True,
        relation_aggregation="relation_mean",
        decoder_layers=2,
    ).eval()
    with torch.no_grad():
        model.encoder_gene_embedding.weight.normal_()
        model.decoder_gene_embedding.weight.normal_()

    permutation = torch.tensor([4, 1, 3, 0, 2])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(num_genes)
    edges = toy_edges(num_genes)
    expression = torch.randn(3, num_genes)
    library_size = torch.tensor([[20.0], [31.0], [12.0]])
    with torch.no_grad():
        expected = model(expression, edges, library_size, sample=False)
        actual = model(
            expression[:, permutation],
            [inverse[relation] for relation in edges],
            library_size,
            sample=False,
            gene_order=permutation,
        )
    torch.testing.assert_close(actual.mean, expected.mean)
    torch.testing.assert_close(
        actual.mean_counts, expected.mean_counts[:, permutation]
    )


def test_gene_residual_starts_at_zero():
    torch.manual_seed(5)
    features = torch.randn(5, 7)
    model = GraphTokenVAE(
        num_genes=5,
        num_relations=3,
        hidden_dim=8,
        latent_dim=4,
        num_latent_tokens=2,
        num_heads=2,
        gene_features=features,
        gene_id_residual=True,
    )
    assert torch.count_nonzero(model.encoder_gene_embedding.weight) == 0
    assert torch.count_nonzero(model.decoder_gene_embedding.weight) == 0
