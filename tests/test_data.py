from pathlib import Path

from dentate_graph_vae.data import build_trrust_graph


def test_trrust_graph_keeps_signed_genes_and_adds_self_edges(tmp_path: Path):
    trrust = tmp_path / "trrust.tsv"
    trrust.write_text(
        "A\tB\tActivation\n"
        "B\tC\tRepression\n"
        "C\tD\tUnknown\n"
        "X\tA\tActivation\n",
        encoding="utf-8",
    )
    graph, indices = build_trrust_graph(trrust, ["A", "B", "C", "D"])

    assert graph["gene_names"] == ["A", "B", "C"]
    assert indices.tolist() == [0, 1, 2]
    assert graph["relation_names"] == ("activation", "repression", "self")
    assert graph["adjacency_lists"][0].tolist() == [[0, 1]]
    assert graph["adjacency_lists"][1].tolist() == [[1, 2]]
    assert graph["adjacency_lists"][2].tolist() == [[0, 0], [1, 1], [2, 2]]
