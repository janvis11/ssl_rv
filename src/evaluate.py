import argparse
import json
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from augmentations import get_eval_transform
from datasets import LabeledImageDataset
from model import LinearClassifier, ResNet18Encoder


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained classifier")
    parser.add_argument("--test_csv", type=str, default="data/labeled/test.csv")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="outputs")
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
    model.eval()

    all_preds = []
    all_targets = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

            all_preds.extend(preds.tolist())
            all_targets.extend(labels.numpy().tolist())

    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    cm = confusion_matrix(all_targets, all_preds).tolist()
    report = classification_report(
        all_targets,
        all_preds,
        target_names=[index_to_label[i] for i in range(len(index_to_label))],
        output_dict=True,
        zero_division=0,
    )

    results = {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "classification_report": report,
        "label_mapping": label_to_index,
    }

    output_path = Path(args.output_dir) / "metrics" / "evaluation_results2.json"   #chnage the name of the file to avoid overwriting previous results
    save_json(results, output_path)

    print(f"Accuracy: {acc:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print(f"Saved evaluation results to: {output_path}")


if __name__ == "__main__":
    main()