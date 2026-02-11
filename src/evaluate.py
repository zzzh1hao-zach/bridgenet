import argparse

import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from dataset import get_data_loaders
from models import FakeArtEfficientNet, FakeArtResNet, HybridCNNViT, ViTBaseline


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    if args.model == "resnet":
        model = FakeArtResNet(pretrained=False).to(device)
    elif args.model == "efficientnet":
        model = FakeArtEfficientNet(pretrained=False).to(device)
    elif args.model == "vit":
        model = ViTBaseline(img_size=128).to(device)
    elif args.model == "hybrid":
        model = HybridCNNViT.from_pretrained(
            args.cnn_checkpoint, args.vit_checkpoint, device=device
        )
    elif args.model == "ensemble":
        _evaluate_ensemble(args, device)
        return
    else:
        raise ValueError(f"Unknown model: {args.model}")

    if args.checkpoint and args.model != "hybrid":
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    model.eval()
    _, _, test_loader = get_data_loaders(
        args.train_dir, args.test_dir, batch_size=args.batch_size
    )
    _run_eval(model, test_loader, device)


def _evaluate_ensemble(args, device):
    """Evaluate the ensemble model (average of EfficientNet + ViT logits)."""
    eff_model = FakeArtEfficientNet(pretrained=False).to(device)
    eff_model.load_state_dict(torch.load(args.cnn_checkpoint, map_location=device))
    eff_model.eval()

    vit_model = ViTBaseline(img_size=128).to(device)
    vit_model.load_state_dict(torch.load(args.vit_checkpoint, map_location=device))
    vit_model.eval()

    _, _, test_loader = get_data_loaders(
        args.train_dir, args.test_dir, batch_size=args.batch_size
    )

    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Evaluating ensemble"):
            X, y = X.to(device), y.to(device).unsqueeze(1).float()
            outputs = (eff_model(X) + vit_model(X)) / 2
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
            all_preds.append(preds)
            all_labels.append(y.cpu().int())

    all_preds = torch.cat(all_preds).view(-1)
    all_labels = torch.cat(all_labels).view(-1)
    _print_metrics(all_labels, all_preds)


def _run_eval(model, test_loader, device):
    """Run evaluation on test set and print metrics."""
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Evaluating"):
            X, y = X.to(device), y.to(device).unsqueeze(1).float()
            outputs = model(X)
            preds = (torch.sigmoid(outputs) > 0.5).int().cpu()
            all_preds.append(preds)
            all_labels.append(y.cpu().int())

    all_preds = torch.cat(all_preds).view(-1)
    all_labels = torch.cat(all_labels).view(-1)
    _print_metrics(all_labels, all_preds)


def _print_metrics(labels, preds):
    print(f"\nTest Results:")
    print(f"  Accuracy:  {accuracy_score(labels, preds):.4f}")
    print(f"  Precision: {precision_score(labels, preds):.4f}")
    print(f"  Recall:    {recall_score(labels, preds):.4f}")
    print(f"  F1 Score:  {f1_score(labels, preds):.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate AI art detection models")
    parser.add_argument("--model", type=str, required=True,
                        choices=["resnet", "efficientnet", "vit", "hybrid", "ensemble"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--train-dir", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cnn-checkpoint", type=str, default=None,
                        help="EfficientNet checkpoint (for hybrid/ensemble)")
    parser.add_argument("--vit-checkpoint", type=str, default=None,
                        help="ViT checkpoint (for hybrid/ensemble)")
    args = parser.parse_args()
    evaluate(args)
