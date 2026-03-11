"""
Embedding visualization and clustering quality analysis.

Generates PCA and t-SNE 2D scatter plots colored by label, and computes
silhouette score and Davies–Bouldin index for the encoder representations.

Usage:
    python -m src.visualize_embeddings \
        --checkpoint outputs/checkpoints/simclr_encoder_final.pt \
        --csv_path data/labeled/labels.csv
"""

import argparse
import json
import os
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from torch.utils.data import DataLoader

from src.augmentations import get_eval_transform
from src.datasets import LabeledImageDataset, get_label_mapping_from_csv
from src.evaluate import extract_embeddings
from src.model import ResNet18Encoder


# ── Visualization helpers ─────────────────────────────────────────

PALETTE = [
    "#2563eb", "#dc2626", "#16a34a", "#f59e0b", "#8b5cf6",
    "#ec4899", "#06b6d4", "#84cc16", "#f97316", "#6366f1",
]


def _scatter_plot(
    coords: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    save_path: str,
) -> None:
    """Create a 2D scatter plot colored by label."""
    plt.figure(figsize=(8, 6))

    unique_labels = sorted(set(labels))
    for idx in unique_labels:
        mask = labels == idx
        color = PALETTE[idx % len(PALETTE)]
        name = class_names[idx] if idx < len(class_names) else f"class_{idx}"
        plt.scatter(
            coords[mask, 0], coords[mask, 1],
            c=color, label=name, s=30, alpha=0.7, edgecolors="white", linewidth=0.3,
        )

    plt.xlabel(xlabel, fontsize=11)
    plt.ylabel(ylabel, fontsize=11)
    plt.title(title, fontsize=13)
    plt.legend(fontsize=9, markerscale=1.5, framealpha=0.8)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved: {save_path}")


def visualize_pca(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "PCA Embedding Visualization",
) -> PCA:
    """Reduce to 2D with PCA and save scatter plot. Returns fitted PCA object."""
    pca = PCA(n_components=2)
    coords = pca.fit_transform(embeddings)
    var = pca.explained_variance_ratio_
    _scatter_plot(
        coords, labels, class_names,
        title=f"{title}\n(Var: PC1={var[0]:.1%}, PC2={var[1]:.1%})",
        xlabel="PC 1", ylabel="PC 2",
        save_path=save_path,
    )
    return pca


def visualize_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    save_path: str,
    title: str = "t-SNE Embedding Visualization",
    perplexity: float = 30.0,
    random_state: int = 42,
) -> None:
    """Reduce to 2D with t-SNE and save scatter plot."""
    effective_perplexity = min(perplexity, max(1, len(embeddings) - 1))
    tsne = TSNE(
        n_components=2, perplexity=effective_perplexity,
        random_state=random_state, init="pca", learning_rate="auto",
    )
    coords = tsne.fit_transform(embeddings)
    _scatter_plot(
        coords, labels, class_names,
        title=title,
        xlabel="t-SNE 1", ylabel="t-SNE 2",
        save_path=save_path,
    )


def compute_clustering_metrics(
    embeddings: np.ndarray, labels: np.ndarray,
) -> Dict[str, float]:
    """Compute silhouette score and Davies–Bouldin index."""
    unique_labels = set(labels)
    if len(unique_labels) < 2:
        print("⚠ Need ≥2 classes for clustering metrics — skipping.")
        return {"silhouette_score": None, "davies_bouldin_index": None}

    sil = silhouette_score(embeddings, labels)
    dbi = davies_bouldin_score(embeddings, labels)
    return {
        "silhouette_score": round(float(sil), 4),
        "davies_bouldin_index": round(float(dbi), 4),
    }


# ── CLI ───────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize encoder embeddings")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to encoder state-dict (.pt)")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Labeled CSV for coloring points")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--perplexity", type=float, default=30.0,
                        help="t-SNE perplexity")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional tag appended to output filenames")
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

    # ── Encoder ───────────────────────────────────────────────────
    encoder = ResNet18Encoder(pretrained=False)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    encoder.load_state_dict(state_dict)
    encoder = encoder.to(device)
    print(f"Loaded encoder from {args.checkpoint}")

    # ── Extract embeddings ────────────────────────────────────────
    embeddings, labels = extract_embeddings(encoder, dataloader, device)
    print(f"Extracted embeddings: {embeddings.shape}")

    suffix = f"_{args.tag}" if args.tag else ""

    # ── PCA ────────────────────────────────────────────────────────
    pca_path = os.path.join(args.plots_dir, f"pca_embeddings{suffix}.png")
    visualize_pca(embeddings, labels, class_names, pca_path)

    # ── t-SNE ─────────────────────────────────────────────────────
    tsne_path = os.path.join(args.plots_dir, f"tsne_embeddings{suffix}.png")
    visualize_tsne(embeddings, labels, class_names, tsne_path, perplexity=args.perplexity)

    # ── Clustering metrics ────────────────────────────────────────
    cluster_metrics = compute_clustering_metrics(embeddings, labels)
    print(f"Silhouette Score:     {cluster_metrics['silhouette_score']}")
    print(f"Davies–Bouldin Index: {cluster_metrics['davies_bouldin_index']}")

    metrics_path = os.path.join(args.metrics_dir, f"embedding_quality{suffix}.json")
    with open(metrics_path, "w") as f:
        json.dump(cluster_metrics, f, indent=2)
    print(f"Clustering metrics saved: {metrics_path}")

    print(f"\n✅ Embedding visualization complete")


if __name__ == "__main__":
    main()
