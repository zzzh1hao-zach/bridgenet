import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from dataset import get_data_loaders
from models import FakeArtEfficientNet, FakeArtResNet, HybridCNNViT, ViTBaseline


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build model
    if args.model == "resnet":
        model = FakeArtResNet(pretrained=True).to(device)
    elif args.model == "efficientnet":
        model = FakeArtEfficientNet(pretrained=True, dropout_p=args.dropout).to(device)
    elif args.model == "vit":
        model = ViTBaseline(img_size=128).to(device)
    elif args.model == "hybrid":
        model = HybridCNNViT.from_pretrained(
            args.cnn_checkpoint, args.vit_checkpoint, device=device
        )
    else:
        raise ValueError(f"Unknown model: {args.model}")

    train_loader, val_loader, _ = get_data_loaders(
        args.train_dir, args.test_dir, batch_size=args.batch_size
    )

    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    train_losses, val_losses = [], []
    start = time.time()

    for epoch in range(args.epochs):
        # --- Training ---
        model.train()
        total_loss = 0
        progress = tqdm(
            enumerate(train_loader),
            total=min(args.max_batches, len(train_loader)),
            desc=f"Epoch {epoch + 1}/{args.epochs}",
        )

        for i, (X, y) in progress:
            if i >= args.max_batches:
                break
            X, y = X.to(device), y.to(device).unsqueeze(1).float()

            optimizer.zero_grad()
            outputs = model(X)
            loss = loss_fn(outputs, y)
            loss.backward()
            if args.clip_grad:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=f"{total_loss / (i + 1):.4f}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Validation ---
        model.eval()
        val_loss = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for i, (X, y) in enumerate(val_loader):
                if i >= args.max_batches:
                    break
                X, y = X.to(device), y.to(device).unsqueeze(1).float()
                outputs = model(X)
                val_loss += loss_fn(outputs, y).item()
                preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
                all_preds.append(preds)
                all_labels.append(y.cpu().int())

        num_val_batches = min(args.max_batches, len(val_loader))
        avg_train = total_loss / min(args.max_batches, len(train_loader))
        avg_val = val_loss / num_val_batches
        train_losses.append(avg_train)
        val_losses.append(avg_val)

        all_preds = torch.cat(all_preds).view(-1)
        all_labels = torch.cat(all_labels).view(-1)

        print(
            f"  Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}\n"
            f"  Precision: {precision_score(all_labels, all_preds):.4f} | "
            f"Recall: {recall_score(all_labels, all_preds):.4f} | "
            f"F1: {f1_score(all_labels, all_preds):.4f} | "
            f"Accuracy: {accuracy_score(all_labels, all_preds):.4f}"
        )

    elapsed = time.time() - start
    print(f"\nTraining completed in {elapsed:.1f}s")

    # Save model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model saved to {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AI art detection models")
    parser.add_argument("--model", type=str, default="efficientnet",
                        choices=["resnet", "efficientnet", "vit", "hybrid"])
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=4.2e-4)
    parser.add_argument("--weight-decay", type=float, default=1.6e-6)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--max-batches", type=int, default=300,
                        help="Max mini-batches per epoch (for compute-constrained training)")
    parser.add_argument("--clip-grad", action="store_true",
                        help="Enable gradient clipping (recommended for ViT/hybrid)")
    parser.add_argument("--save-path", type=str, default="checkpoints/model.pth")
    parser.add_argument("--cnn-checkpoint", type=str, default=None,
                        help="EfficientNet checkpoint (required for hybrid model)")
    parser.add_argument("--vit-checkpoint", type=str, default=None,
                        help="ViT checkpoint (required for hybrid model)")
    args = parser.parse_args()
    train(args)
