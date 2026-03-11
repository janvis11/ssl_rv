from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class UnlabeledImageDataset(Dataset):
    """Recursively load unlabeled images from a directory."""

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(
            [
                path
                for path in self.root_dir.rglob("*")
                if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
            ]
        )

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid image files found in: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


class LabeledImageDataset(Dataset):
    """
    CSV format:
        image_path,label
        data/labeled/frame_001.png,road-dominant
    """

    def __init__(
        self,
        csv_path: str,
        transform: Optional[Callable] = None,
        label_to_index: Optional[Dict[str, int]] = None,
    ):
        self.csv_path = Path(csv_path)
        self.transform = transform
        self.data = pd.read_csv(self.csv_path)

        required_columns = {"image_path", "label"}
        if not required_columns.issubset(self.data.columns):
            raise ValueError(
                f"CSV {self.csv_path} must contain columns: {required_columns}"
            )

        self.data["image_path"] = self.data["image_path"].astype(str)
        self.data["label"] = self.data["label"].astype(str)

        labels = sorted(self.data["label"].unique())
        self.label_to_index = (
            label_to_index if label_to_index is not None
            else {label: idx for idx, label in enumerate(labels)}
        )
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Tuple:
        row = self.data.iloc[index]
        image_path = Path(row["image_path"])
        label_name = row["label"]

        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")

        if label_name not in self.label_to_index:
            raise KeyError(f"Unknown label '{label_name}' not found in label mapping.")

        image = Image.open(image_path).convert("RGB")
        label = self.label_to_index[label_name]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_label_mapping_from_csv(csv_path: str) -> Dict[str, int]:
    data = pd.read_csv(csv_path)
    labels = sorted(data["label"].astype(str).unique())
    return {label: idx for idx, label in enumerate(labels)}


def get_num_classes_from_csv(csv_path: str) -> int:
    data = pd.read_csv(csv_path)
    return int(data["label"].nunique())