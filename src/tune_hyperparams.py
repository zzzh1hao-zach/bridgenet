import argparse
import os

import torch
import torch.nn as nn
import torch.optim as optim
from ray import tune
from ray.train import get_checkpoint, get_context
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from dataset import get_data_loaders
from models import FakeArtEfficientNet, ViTBaseline


def train_fn(config):
    """Training function invoked by Ray Tune for each trial."""
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_loader, val_loader, _ = get_data_loaders(
        config["train_path"], config["test_path"], batch_size=config["batch_size"]
    )

    # Build model based on config
    if config["model_type"] == "efficientnet":
        model = FakeArtEfficientNet(pretrained=True).to(device)
    elif config["model_type"] == "vit":
        model = ViTBaseline(img_size=128).to(device)
    else:
        raise ValueError(f"Unknown model: {config['model_type']}")

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )

    # Resume from checkpoint if available
    checkpoint = get_checkpoint()
    if checkpoint:
        checkpoint_path = checkpoint.to_directory()
        model.load_state_dict(
            torch.load(os.path.join(checkpoint_path, "model.pt"), map_location=device)
        )

    for epoch in range(1, config["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for i, (X, y) in enumerate(train_loader):
            if i >= config["max_batches_training"]:
                break
            X, y = X.to(device), y.to(device).unsqueeze(1).float()
            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / min(len(train_loader), config["max_batches_training"])

        model.eval()
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for i, (X, y) in enumerate(val_loader):
                if i >= config["max_batches_val"]:
                    break
                X, y = X.to(device), y.to(device).unsqueeze(1).float()
                outputs = model(X)
                val_loss += loss_fn(outputs, y).item()
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
                all_preds.append(preds)
                all_labels.append(y.cpu().int())

        val_loss /= min(len(val_loader), config["max_batches_val"])
        val_acc = (
            (torch.cat(all_preds) == torch.cat(all_labels)).float().mean().item()
        )

        tune.report(
            {"train_loss": train_loss, "val_loss": val_loss, "val_accuracy": val_acc}
        )

        # Save checkpoint
        checkpoint_dir = get_context().get_trial_dir()
        path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
        torch.save(
            {"model_state": model.state_dict(), "opt_state": optimizer.state_dict()},
            path,
        )


def main(args):
    config = {
        "model_type": args.model,
        "train_path": args.train_dir,
        "test_path": args.test_dir,
        "epochs": args.epochs,
        "max_batches_training": args.max_batches,
        "max_batches_val": args.max_batches_val,
        "batch_size": tune.choice([16, 32, 64]),
        "lr": tune.loguniform(1e-5, 1e-2),
        "weight_decay": tune.loguniform(1e-6, 1e-3),
    }

    scheduler = ASHAScheduler(
        metric="val_loss",
        mode="min",
        max_t=args.epochs,
        grace_period=1,
        reduction_factor=2,
    )

    reporter = CLIReporter(
        metric_columns=["train_loss", "val_loss", "val_accuracy", "training_iteration"]
    )

    result = tune.Tuner(
        train_fn,
        tune_config=tune.TuneConfig(scheduler=scheduler, num_samples=args.num_samples),
        param_space=config,
        run_config=tune.RunConfig(
            name=f"tune_{args.model}",
            progress_reporter=reporter,
            storage_path="~/ray_results",
        ),
    ).fit()

    best = result.get_best_result("val_loss", mode="min")
    print(f"\nBest hyperparameters: {best.config}")
    print(f"Best validation loss: {best.metrics['val_loss']:.4f}")
    print(f"Best validation accuracy: {best.metrics['val_accuracy']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hyperparameter tuning with Ray Tune")
    parser.add_argument("--model", type=str, required=True,
                        choices=["efficientnet", "vit"])
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-batches", type=int, default=500)
    parser.add_argument("--max-batches-val", type=int, default=250)
    parser.add_argument("--num-samples", type=int, default=12)
    args = parser.parse_args()
    main(args)
