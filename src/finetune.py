import argparse
import json
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import get_eval_transform, get_train_transform
from datasets import LabeledImageDataset, get_label_mapping_from_csv
from model import LinearClassifier, ResNet18Encoder


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def build_loaders(
    train_csv: str,
    val_csv: str,
    test_csv: str,
    image_size: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, int]]:
    label_to_index = get_label_mapping_from_csv(train_csv)

    train_dataset = LabeledImageDataset(
        csv_path=train_csv,
        transform=get_train_transform(image_size),
        label_to_index=label_to_index,
    )
    val_dataset = LabeledImageDataset(
        csv_path=val_csv,
        transform=get_eval_transform(image_size),
        label_to_index=label_to_index,
    )
    test_dataset = LabeledImageDataset(
        csv_path=test_csv,
        transform=get_eval_transform(image_size),
        label_to_index=label_to_index,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(train_dataset) >= batch_size,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader, test_loader, label_to_index


def load_encoder_from_pretrained(checkpoint_path: str, device: torch.device) -> ResNet18Encoder:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    encoder = ResNet18Encoder(pretrained=False)

    if isinstance(checkpoint, dict) and "encoder_state_dict" in checkpoint:
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
    elif isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        raise ValueError(
            "Checkpoint contains full model_state_dict but no encoder_state_dict. "
            "Use a SimCLR checkpoint saved by the updated training script."
        )
    else:
        encoder.load_state_dict(checkpoint)

    return encoder


def train_one_epoch(
    model: LinearClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.train()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    progress = tqdm(loader, desc="Fine-tuning", leave=False)
    for images, labels in progress:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.detach().cpu().numpy())
        all_targets.extend(labels.detach().cpu().numpy())
        running_loss += loss.item()

        progress.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = running_loss / max(len(loader), 1)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return avg_loss, acc, macro_f1


def evaluate(
    model: LinearClassifier,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float, float]:
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            logits = model(images)
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            running_loss += loss.item()

    avg_loss = running_loss / max(len(loader), 1)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return avg_loss, acc, macro_f1


def save_classifier_checkpoint(
    path: Path,
    epoch: int,
    model: LinearClassifier,
    optimizer: torch.optim.Optimizer,
    label_to_index: Dict[str, int],
    metrics: dict,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "classifier_state_dict": model.classifier.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "label_to_index": label_to_index,
            "metrics": metrics,
            "args": vars(args),
        },
        path,
    )


def main():
    parser = argparse.ArgumentParser(description="Fine-tune classifier on labeled KITTI frames")
    parser.add_argument("--train_csv", type=str, default="data/labeled/train.csv")
    parser.add_argument("--val_csv", type=str, default="data/labeled/val.csv")
    parser.add_argument("--test_csv", type=str, default="data/labeled/test.csv")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--pretrained_ckpt", type=str, default="outputs/checkpoints/simclr_full_best.pt")
    parser.add_argument("--use_pretrained", action="store_true")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader, label_to_index = build_loaders(
        args.train_csv,
        args.val_csv,
        args.test_csv,
        args.image_size,
        args.batch_size,
        args.num_workers,
    )

    if args.use_pretrained:
        encoder = load_encoder_from_pretrained(args.pretrained_ckpt, device)
        run_name = "pretrained"
    else:
        encoder = ResNet18Encoder(pretrained=False)
        run_name = "scratch"

    model = LinearClassifier(encoder=encoder, num_classes=len(label_to_index), hidden_dim=512, dropout=0.5).to(device)
    
    # Compute class weights
    from collections import Counter
    train_labels = [label for _, label in train_loader.dataset]
    class_counts = Counter(train_labels)
    total_samples = sum(class_counts.values())
    class_weights = [total_samples / (len(class_counts) * count) for count in [class_counts[i] for i in range(len(label_to_index))]]
    class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    if args.use_pretrained:
        # Freeze all encoder layers except the last one (layer4)
        for name, param in model.encoder.named_parameters():
            if 'layer4' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        # Use different learning rates
        encoder_lr = args.lr * 0.1  # Lower LR for encoder
        classifier_lr = args.lr
        optimizer = AdamW([
            {'params': model.encoder.layer4.parameters(), 'lr': encoder_lr},
            {'params': model.classifier.parameters(), 'lr': classifier_lr}
        ])
    else:
        optimizer = AdamW(trainable_params, lr=args.lr)

    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    metrics_dir = Path(args.output_dir) / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    history = {"train": [], "val": []}
    best_val_f1 = -1.0

    print(f"Device: {device}")
    print(f"Mode: {run_name}")
    print(f"Using pretrained: {args.use_pretrained}")

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_f1 = evaluate(
            model, val_loader, criterion, device
        )

        history["train"].append(
            {
                "epoch": epoch,
                "loss": train_loss,
                "accuracy": train_acc,
                "macro_f1": train_f1,
            }
        )
        history["val"].append(
            {
                "epoch": epoch,
                "loss": val_loss,
                "accuracy": val_acc,
                "macro_f1": val_f1,
            }
        )

        print(
            f"Epoch [{epoch}/{args.epochs}] | "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}, val_f1={val_f1:.4f}"
        )

        metrics = {
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_macro_f1": train_f1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_macro_f1": val_f1,
        }

        save_classifier_checkpoint(
            checkpoint_dir / f"classifier_{run_name}_latest.pt",
            epoch,
            model,
            optimizer,
            label_to_index,
            metrics,
            args,
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_classifier_checkpoint(
                checkpoint_dir / f"classifier_{run_name}_best.pt",
                epoch,
                model,
                optimizer,
                label_to_index,
                metrics,
                args,
            )

    test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion, device)

    final_results = {
        "run_name": run_name,
        "test_loss": test_loss,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "best_val_macro_f1": best_val_f1,
        "history": history,
    }

    save_json(final_results, metrics_dir / f"finetune_{run_name}_metrics.json")

    print(
        f"Final test [{run_name}] | "
        f"loss={test_loss:.4f}, accuracy={test_acc:.4f}, macro_f1={test_f1:.4f}"
    )


if __name__ == "__main__":
    main()