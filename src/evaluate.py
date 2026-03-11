"""
Downstream evaluation utilities.

Provides functions to evaluate a classifier on a labeled dataset, extract
embeddings, and compute standard metrics (accuracy, macro F1, confusion matrix).

Usage:
    python -m src.evaluate \
        --checkpoint outputs/checkpoints/best_classifier_pretrained.pt \
        --csv_path data/labeled/labels.csv \
        --mode pretrained
"""

import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from src.augmentations import get_eval_transform
from src.datasets import LabeledImageDataset, get_label_mapping_from_csv, get_num_classes_from_csv
from src.model import LinearClassifier, ResNet18Encoder


# ── Core utilities ────────────────────────────────────────────────


@torch.no_grad()
def extract_embeddings(
    encoder: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract encoder features and labels from a labeled dataloader.

    Returns:
        embeddings: ndarray of shape [N, D]
        labels: ndarray of shape [N]
    """
    encoder.eval()
    all_embeddings: List[torch.Tensor] = []
    all_labels: List[torch.Tensor] = []

    for images, labels in dataloader:
        images = images.to(device)
        features = encoder(images)  # [B, D]
        all_embeddings.append(features.cpu())
        all_labels.append(labels)

    embeddings = torch.cat(all_embeddings, dim=0).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()
    return embeddings, labels


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict:
    """Evaluate a classifier and return a metrics dictionary.

    Returns dict with: accuracy, macro_f1, per_class_report, predictions, true_labels.
    """
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []

    for images, labels in dataloader:
        images = images.to(device)
        logits = model(images)
        preds = logits.argmax(dim=1).cpu().tolist()
        all_preds.extend(preds)
        all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    report = classification_report(all_labels, all_preds, output_dict=True, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    return {
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "predictions": all_preds,
        "true_labels": all_labels,
    }


def save_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    """Plot and save a confusion-matrix heatmap."""
    plt.figure(figsize=(max(6, len(class_names)), max(5, len(class_names) * 0.8)))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted", fontsize=12)
    plt.ylabel("True", fontsize=12)
    plt.title(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved: {save_path}")


def save_metrics(metrics_dict: Dict, path: str) -> None:
    """Serialize a metrics dictionary to JSON."""
    # Filter out non-serializable values
    serializable = {
        k: v for k, v in metrics_dict.items()
        if k not in ("predictions", "true_labels")
    }
    with open(path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"Metrics saved: {path}")


# ── CLI ───────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned classifier")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to saved LinearClassifier state-dict")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to labeled CSV for evaluation")
    parser.add_argument("--mode", type=str, default="pretrained",
                        choices=["pretrained", "scratch"],
                        help="Label for this evaluation run")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--plots_dir", type=str, default="outputs/plots")
    parser.add_argument("--metrics_dir", type=str, default="outputs/metrics")
    return parser.parse_args()


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    # ── Dataset ───────────────────────────────────────────────────
    num_classes = get_num_classes_from_csv(args.csv_path)
    label_mapping = get_label_mapping_from_csv(args.csv_path)
    class_names = [name for name, _ in sorted(label_mapping.items(), key=lambda x: x[1])]

    transform = get_eval_transform(args.image_size)
    dataset = LabeledImageDataset(
        csv_path=args.csv_path, transform=transform, label_to_index=label_mapping
    )
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )
    print(f"Evaluating on {len(dataset)} samples ({num_classes} classes)")

    # ── Model ─────────────────────────────────────────────────────
    encoder = ResNet18Encoder(pretrained=False)
    model = LinearClassifier(encoder=encoder, num_classes=num_classes).to(device)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded checkpoint: {args.checkpoint}")

    # ── Evaluate ──────────────────────────────────────────────────
    results = evaluate_model(model, dataloader, device)
    print(f"\nAccuracy:  {results['accuracy']}")
    print(f"Macro F1:  {results['macro_f1']}")

    # Confusion matrix
    cm = np.array(results["confusion_matrix"])
    cm_path = os.path.join(args.plots_dir, f"confusion_matrix_{args.mode}.png")
    save_confusion_matrix(cm, class_names, cm_path, title=f"Confusion Matrix ({args.mode})")

    # Metrics JSON
    metrics_path = os.path.join(args.metrics_dir, f"evaluation_{args.mode}.json")
    save_metrics(results, metrics_path)

    print(f"\n✅ Evaluation ({args.mode}) complete")


if __name__ == "__main__":
    main()
