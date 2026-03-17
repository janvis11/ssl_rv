import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import davies_bouldin_score, silhouette_score
from torch.utils.data import DataLoader

from augmentations import get_eval_transform
from datasets import LabeledImageDataset
from model import LinearClassifier, ResNet18Encoder


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def extract_embeddings(model: LinearClassifier, loader: DataLoader, device: torch.device):
    model.eval()
    embeddings = []
    labels = []

    with torch.no_grad():
        for images, batch_labels in loader:
            images = images.to(device, non_blocking=True)
            features = model.encoder(images)
            embeddings.append(features.cpu().numpy())
            labels.append(batch_labels.numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    labels = np.concatenate(labels, axis=0)
    return embeddings, labels


def plot_projection(points, labels, title, save_path, class_names):
    plt.figure(figsize=(8, 6))
    for class_idx, class_name in class_names.items():
        mask = labels == class_idx
        plt.scatter(
            points[mask, 0],
            points[mask, 1],
            label=class_name,
            alpha=0.7,
        )

    plt.title(title)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize encoder embeddings")
    parser.add_argument("--test_csv", type=str, default="data/labeled/test.csv")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--tsne_perplexity", type=float, default=10.0)
    parser.add_argument("--tsne_random_state", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)

    if "label_to_index" not in checkpoint:
        raise ValueError("Classifier checkpoint must contain 'label_to_index'.")

    label_to_index = checkpoint["label_to_index"]
    index_to_label = {index: label for label, index in label_to_index.items()}

    dataset = LabeledImageDataset(
        csv_path=args.test_csv,
        transform=get_eval_transform(args.image_size),
        label_to_index=label_to_index,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    encoder = ResNet18Encoder(pretrained=False)
    model = LinearClassifier(encoder=encoder, num_classes=len(label_to_index)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    embeddings, labels = extract_embeddings(model, loader, device)

    pca = PCA(n_components=2)
    pca_points = pca.fit_transform(embeddings)

    tsne = TSNE(
        n_components=2,
        perplexity=args.tsne_perplexity,
        random_state=args.tsne_random_state,
        init="pca",
        learning_rate="auto",
    )
    tsne_points = tsne.fit_transform(embeddings)

    silhouette = float(silhouette_score(embeddings, labels))
    db_index = float(davies_bouldin_score(embeddings, labels))

    plots_dir = Path(args.output_dir) / "plots"
    metrics_dir = Path(args.output_dir) / "metrics"
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    plot_projection(
        pca_points,
        labels,
        "PCA of Encoder Embeddings",
        plots_dir / "pca_embeddings.png",
        index_to_label,
    )
    plot_projection(
        tsne_points,
        labels,
        "t-SNE of Encoder Embeddings",
        plots_dir / "tsne_embeddings.png",
        index_to_label,
    )

    metrics = {
        "silhouette_score": silhouette,
        "davies_bouldin_index": db_index,
        "num_samples": int(len(labels)),
        "num_classes": int(len(label_to_index)),
    }
    save_json(metrics, metrics_dir / "embedding_metrics.json")

    print(f"Silhouette Score: {silhouette:.4f}")
    print(f"Davies-Bouldin Index: {db_index:.4f}")
    print(f"Saved plots to: {plots_dir}")


if __name__ == "__main__":
    main()