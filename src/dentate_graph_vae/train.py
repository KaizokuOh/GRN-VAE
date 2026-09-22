"""Train and evaluate the graph-token VAE on dentate-gyrus raw counts."""

import argparse
from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchmetrics.functional import pearson_corrcoef

from .data import DentateGyrusData, load_gene_features
from .model import GraphTokenVAE


def normalize_counts(counts: torch.Tensor, target_sum: float = 1e4) -> torch.Tensor:
    """Library-size normalize each cell and apply log1p."""

    totals = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    return torch.log1p(counts / totals * target_sum)


def atomic_torch_save(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


@torch.no_grad()
def evaluate(
    model: GraphTokenVAE,
    dataset,
    edges: list[torch.Tensor],
    batch_size: int,
    device: torch.device,
    beta: float,
) -> dict[str, float | int]:
    """Evaluate NB loss, log-normalized MSE, and gene-wise PCC."""

    model.eval()
    totals = {"nb_loss": 0.0, "mse": 0.0, "kl": 0.0}
    predictions, targets = [], []
    seen = 0
    for counts, _ in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        counts = counts.to(device=device, dtype=torch.float32)
        expression = normalize_counts(counts)
        output = model(
            expression,
            edges,
            counts.sum(dim=1, keepdim=True),
            sample=False,
        )
        losses = model.losses(
            counts, expression, output, beta=beta, mse_weight=0.0, pcc_weight=0.0
        )
        for name in totals:
            totals[name] += float(losses[name]) * len(counts)
        predictions.append(model.normalized_reconstruction(output).cpu())
        targets.append(expression.cpu())
        seen += len(counts)

    prediction = torch.cat(predictions).double()
    target = torch.cat(targets).double()
    correlations = pearson_corrcoef(prediction, target)
    valid = torch.isfinite(correlations)
    return {
        **{name: value / seen for name, value in totals.items()},
        "gene_wise_pcc": (
            float(correlations[valid].mean()) if valid.any() else float("nan")
        ),
        "valid_pcc_genes": int(valid.sum()),
    }


def model_from_args(
    args: argparse.Namespace,
    num_genes: int,
    num_relations: int,
    gene_features: torch.Tensor | None,
) -> GraphTokenVAE:
    return GraphTokenVAE(
        num_genes=num_genes,
        num_relations=num_relations,
        hidden_dim=args.hidden_dim,
        latent_dim=args.latent_dim,
        num_latent_tokens=args.latent_tokens,
        num_gnn_passes=args.gnn_passes,
        num_heads=args.heads,
        dropout=args.dropout,
        gene_features=gene_features,
        gene_id_residual=args.gene_id_residual,
        relation_aggregation=args.relation_aggregation,
        decoder_layers=args.decoder_layers,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-h5ad", type=Path, required=True)
    parser.add_argument("--test-h5ad", type=Path, required=True)
    parser.add_argument("--trrust", type=Path, required=True)
    parser.add_argument("--gene-features", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--last-checkpoint", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--skip-test", action="store_true")

    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument("--latent-tokens", type=int, default=8)
    parser.add_argument("--gnn-passes", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.02)

    parser.add_argument("--gene-id-residual", action="store_true")
    parser.add_argument(
        "--relation-aggregation",
        choices=("global_mean", "relation_mean"),
        default="global_mean",
    )
    parser.add_argument("--decoder-layers", type=int, default=1)

    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--mse-weight", type=float, default=0.0)
    parser.add_argument("--pcc-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=46)
    parser.add_argument("--dataloader-seed", type=int, default=46)

    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="disabled",
    )
    parser.add_argument("--wandb-project", default="dentate-gyrus-graph-vae")
    parser.add_argument("--wandb-run-name", default="graph-token-vae")
    parser.add_argument("--wandb-run-id")
    parser.add_argument(
        "--wandb-resume", choices=("allow", "must", "never", "auto")
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.max_epochs < 1 or args.patience < 1:
        raise ValueError("max_epochs and patience must be positive")
    last_path = args.last_checkpoint or args.output.with_name(
        f"{args.output.stem}.last{args.output.suffix}"
    )
    if last_path == args.output:
        raise ValueError("output and last-checkpoint paths must differ")

    torch.manual_seed(args.seed)
    datasets = DentateGyrusData(args.train_h5ad, args.test_h5ad, args.trrust)
    graph = datasets.graph
    gene_features = None
    feature_report = None
    if args.gene_features is not None:
        gene_features, feature_report = load_gene_features(
            args.gene_features, graph["gene_names"]
        )
        print(f"gene features: {feature_report}", flush=True)

    split_generator = torch.Generator().manual_seed(args.seed)
    permutation = torch.randperm(len(datasets.train), generator=split_generator)
    validation_size = round(0.1 * len(datasets.train))
    validation_indices = permutation[:validation_size].tolist()
    training_indices = permutation[validation_size:].tolist()
    training = Subset(datasets.train, training_indices)
    validation = Subset(datasets.train, validation_indices)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    edges = [edge.to(device) for edge in graph["adjacency_lists"]]
    model = model_from_args(
        args, graph["num_genes"], len(edges), gene_features
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    loader_generator = torch.Generator().manual_seed(args.dataloader_seed)
    train_loader = DataLoader(
        training,
        batch_size=args.batch_size,
        shuffle=True,
        generator=loader_generator,
        num_workers=0,
    )

    history: list[dict] = []
    best_score = float("inf")
    best_epoch = 0
    best_state = None
    stale = 0
    start_epoch = 1

    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        expected_split = {
            "train": training_indices,
            "validation": validation_indices,
        }
        if checkpoint["split_indices"] != expected_split:
            raise ValueError("resume checkpoint uses a different data split")
        architecture_keys = (
            "hidden_dim",
            "latent_dim",
            "latent_tokens",
            "gnn_passes",
            "heads",
            "dropout",
            "gene_features",
            "gene_id_residual",
            "relation_aggregation",
            "decoder_layers",
        )
        mismatches = {
            key: (checkpoint["config"].get(key), getattr(args, key))
            for key in architecture_keys
            if checkpoint["config"].get(key) != getattr(args, key)
        }
        if mismatches:
            raise ValueError(f"resume configuration mismatch: {mismatches}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        history = list(checkpoint["history"])
        best_score = float(checkpoint["best_score"])
        best_epoch = int(checkpoint["best_epoch"])
        best_state = deepcopy(checkpoint["best_model"])
        stale = int(checkpoint["stale"])
        start_epoch = int(history[-1]["epoch"]) + 1
        torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_states"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_states"])
        loader_generator.set_state(checkpoint["dataloader_rng_state"].cpu())
        print(f"resuming at epoch {start_epoch}", flush=True)

    run = None
    if args.wandb_mode != "disabled":
        import wandb

        run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            mode=args.wandb_mode,
            id=args.wandb_run_id,
            resume=args.wandb_resume,
            config={
                **vars(args),
                "num_genes": graph["num_genes"],
                "relations": graph["relation_names"],
                "gene_feature_report": feature_report,
            },
        )

    with run if run is not None else nullcontext():
        for epoch in range(start_epoch, args.max_epochs + 1):
            model.train()
            train_total = 0.0
            train_seen = 0
            for counts, _ in train_loader:
                counts = counts.to(device=device, dtype=torch.float32)
                expression = normalize_counts(counts)
                optimizer.zero_grad(set_to_none=True)
                output = model(
                    expression,
                    edges,
                    counts.sum(dim=1, keepdim=True),
                    sample=True,
                )
                losses = model.losses(
                    counts,
                    expression,
                    output,
                    beta=args.beta,
                    mse_weight=args.mse_weight,
                    pcc_weight=args.pcc_weight,
                )
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                train_total += float(losses["loss"]) * len(counts)
                train_seen += len(counts)

            metrics = evaluate(
                model,
                validation,
                edges,
                args.batch_size,
                device,
                args.beta,
            )
            score = (
                float(metrics["nb_loss"])
                + args.mse_weight * float(metrics["mse"])
                + args.pcc_weight * (1.0 - float(metrics["gene_wise_pcc"]))
            )
            record = {
                "epoch": epoch,
                "train_loss": train_total / train_seen,
                "validation_nb_loss": metrics["nb_loss"],
                "validation_mse": metrics["mse"],
                "validation_gene_wise_pcc": metrics["gene_wise_pcc"],
                "validation_score": score,
            }
            history.append(record)
            if score < best_score - args.min_delta:
                best_score = score
                best_epoch = epoch
                best_state = deepcopy(model.state_dict())
                stale = 0
            else:
                stale += 1

            payload = {
                "model": model.state_dict(),
                "best_model": best_state,
                "optimizer": optimizer.state_dict(),
                "history": history,
                "best_epoch": best_epoch,
                "best_score": best_score,
                "stale": stale,
                "config": vars(args),
                "gene_names": graph["gene_names"],
                "relation_names": graph["relation_names"],
                "split_indices": {
                    "train": training_indices,
                    "validation": validation_indices,
                },
                "gene_feature_report": feature_report,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_states": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
                "dataloader_rng_state": loader_generator.get_state(),
            }
            atomic_torch_save(payload, last_path)

            if run is not None:
                run.log(
                    {
                        "validation_nb_loss": metrics["nb_loss"],
                        "validation_mse": metrics["mse"],
                        "validation_gene_wise_pcc": metrics["gene_wise_pcc"],
                    },
                    step=epoch,
                )
            print(
                f"epoch {epoch:03d} train={record['train_loss']:.4f} "
                f"val_nb={float(metrics['nb_loss']):.4f} "
                f"val_mse={float(metrics['mse']):.4f} "
                f"val_gene_pcc={float(metrics['gene_wise_pcc']):.4f}",
                flush=True,
            )
            if stale >= args.patience:
                print(f"early stopping at epoch {epoch}; best={best_epoch}")
                break

        if best_state is None:
            raise RuntimeError("no best model state was recorded")
        model.load_state_dict(best_state)
        test_metrics = None
        if not args.skip_test:
            test_metrics = evaluate(
                model,
                datasets.test,
                edges,
                args.batch_size,
                device,
                args.beta,
            )
            if run is not None:
                run.log(
                    {
                        "test_nb_loss": test_metrics["nb_loss"],
                        "test_mse": test_metrics["mse"],
                        "test_gene_wise_pcc": test_metrics["gene_wise_pcc"],
                    }
                )
                best = next(
                    record for record in history if record["epoch"] == best_epoch
                )
                run.summary.update(
                    {
                        "validation_nb_loss": best["validation_nb_loss"],
                        "validation_mse": best["validation_mse"],
                        "validation_gene_wise_pcc": best[
                            "validation_gene_wise_pcc"
                        ],
                        "test_nb_loss": test_metrics["nb_loss"],
                        "test_mse": test_metrics["mse"],
                        "test_gene_wise_pcc": test_metrics["gene_wise_pcc"],
                    }
                )

    atomic_torch_save(
        {
            "model": model.state_dict(),
            "config": vars(args),
            "history": history,
            "best_epoch": best_epoch,
            "best_score": best_score,
            "test_metrics": test_metrics,
            "gene_names": graph["gene_names"],
            "relation_names": graph["relation_names"],
            "split_indices": {
                "train": training_indices,
                "validation": validation_indices,
            },
            "gene_feature_report": feature_report,
            "resume_checkpoint": str(last_path),
        },
        args.output,
    )
    print(f"best epoch: {best_epoch}")
    print(f"test: {test_metrics if test_metrics is not None else 'skipped'}")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
