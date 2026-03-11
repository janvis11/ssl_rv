"""
Fine-tune a linear classifier on a small labeled dataset.

Supports two modes:
  1. Pretrained — loads SimCLR encoder weights, optionally freezes encoder
  2. Scratch   — trains a fresh ResNet-18 from random initialization

Usage:
    # Fine-tune with pretrained SimCLR encoder
    python -m src.finetune --csv_path data/labeled/labels.csv \
        --pretrained_path outputs/checkpoints/simclr_encoder_final.pt --epochs 30

    # Train from scratch (baseline)
    python -m src.finetune --csv_path data/labeled/labels.csv --from_scratch --epochs 30
"""

import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset

from src.augmentations import get_eval_transform, get_train_transform
from src.datasets import LabeledImageDataset, get_label_mapping_from_csv, get_num_classes_from_csv
from src.model import LinearClassifier, ResNet18Encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune / train-from-scratch classifier")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to labeled CSV (image_path,label)")
    parser.add_argument("--pretrained_path", type=str, default=None,
                        help="Path to pretrained SimCLR encoder state-dict")
    parser.add_argument("--from_scratch", action="store_true",
                        help="Train a fresh ResNet-18 from scratch (baseline)")
    parser.add_argument("--freeze_encoder", action="store_true",
                        help="Freeze encoder weights during fine-tuning (linear probe)")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Number of fine-tuning epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--val_split", type=float, default=0.2,
                        help="Fraction of data for validation")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image resize dimension")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints",
                        help="Directory to save best model")
    parser.add_argument("--plots_dir", type=str, default="outputs/plots",
                        help="Directory to save training curves")
    parser.add_argument("--metrics_dir", type=str, default="outputs/metrics",
                        help="Directory to save metrics JSON")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    return parser.parse_args()


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += images.size(0)

    avg_loss = total_loss / max(total, 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


def save_training_curves(history: dict, save_path: str, title_prefix: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history["train_loss"], label="Train", linewidth=2, color="#2563eb")
    axes[0].plot(history["val_loss"], label="Val", linewidth=2, color="#dc2626")
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].set_title(f"{title_prefix} — Loss", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(history["train_acc"], label="Train", linewidth=2, color="#2563eb")
    axes[1].plot(history["val_acc"], label="Val", linewidth=2, color="#dc2626")
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].set_title(f"{title_prefix} — Accuracy", fontsize=13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved: {save_path}")


def main():
    args = parse_args()

    # ── Reproducibility ───────────────────────────────────────────
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    # ── Determine mode ────────────────────────────────────────────
    mode = "scratch" if args.from_scratch else "pretrained"
    print(f"Mode: {mode}")

    # ── Dataset ───────────────────────────────────────────────────
    num_classes = get_num_classes_from_csv(args.csv_path)
    label_mapping = get_label_mapping_from_csv(args.csv_path)
    print(f"Classes ({num_classes}): {list(label_mapping.keys())}")

    train_transform = get_train_transform(args.image_size)
    eval_transform = get_eval_transform(args.image_size)

    full_dataset = LabeledImageDataset(
        csv_path=args.csv_path, transform=train_transform, label_to_index=label_mapping
    )

    # Train / val split
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=args.val_split, random_state=args.seed
    )
    train_subset = Subset(full_dataset, train_idx)

    # Validation uses eval transforms — wrap with a separate dataset instance
    val_dataset = LabeledImageDataset(
        csv_path=args.csv_path, transform=eval_transform, label_to_index=label_mapping
    )
    val_subset = Subset(val_dataset, val_idx)

    train_loader = DataLoader(
        train_subset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_subset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Train: {len(train_idx)} samples | Val: {len(val_idx)} samples")

    # ── Build model ───────────────────────────────────────────────
    encoder = ResNet18Encoder(pretrained=False)

    if mode == "pretrained" and args.pretrained_path:
        state_dict = torch.load(args.pretrained_path, map_location=device, weights_only=True)
        encoder.load_state_dict(state_dict)
        print(f"Loaded pretrained encoder from {args.pretrained_path}")
    else:
        print("Training encoder from scratch (random init)")

    if args.freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False
        print("Encoder weights FROZEN (linear probe)")

    model = LinearClassifier(encoder=encoder, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    # ── Training loop ─────────────────────────────────────────────
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t_loss, t_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        v_loss, v_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(t_loss)
        history["train_acc"].append(t_acc)
        history["val_loss"].append(v_loss)
        history["val_acc"].append(v_acc)

        print(
            f"Epoch [{epoch:>3d}/{args.epochs}]  "
            f"Train Loss: {t_loss:.4f}  Acc: {t_acc:.4f}  |  "
            f"Val Loss: {v_loss:.4f}  Acc: {v_acc:.4f}"
        )

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_path = os.path.join(args.checkpoint_dir, f"best_classifier_{mode}.pt")
            torch.save(model.state_dict(), best_path)

    total_time = time.time() - start_time

    # ── Save outputs ──────────────────────────────────────────────
    curve_path = os.path.join(args.plots_dir, f"finetune_curves_{mode}.png")
    save_training_curves(history, curve_path, title_prefix=f"Fine-tune ({mode})")

    metrics = {
        "mode": mode,
        "best_val_accuracy": round(best_val_acc, 4),
        "final_train_accuracy": round(history["train_acc"][-1], 4),
        "final_val_accuracy": round(history["val_acc"][-1], 4),
        "epochs": args.epochs,
        "total_time_seconds": round(total_time, 2),
        "num_classes": num_classes,
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "freeze_encoder": args.freeze_encoder,
        "learning_rate": args.lr,
    }
    metrics_path = os.path.join(args.metrics_dir, f"finetune_{mode}.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved: {metrics_path}")

    print(f"\n✅ Fine-tuning ({mode}) complete — best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
