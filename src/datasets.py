from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


class UnlabeledImageDataset(Dataset):
    """
    Loads images from a folder recursively for self-supervised pretraining.
    """

    def __init__(self, root_dir: str, transform: Optional[Callable] = None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.image_paths = sorted(
            [
                p for p in self.root_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
            ]
        )

        if not self.image_paths:
            raise ValueError(f"No valid images found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image


class LabeledImageDataset(Dataset):
    """
    CSV-based labeled dataset.

    CSV format:
        image_path,label
        data/labeled/img1.png,road
        data/labeled/img2.png,vehicle
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
        if not required_columns.issubset(set(self.data.columns)):
            raise ValueError(f"CSV must contain columns: {required_columns}")

        labels = sorted(self.data["label"].unique())
        self.label_to_index = label_to_index or {label: idx for idx, label in enumerate(labels)}
        self.index_to_label = {idx: label for label, idx in self.label_to_index.items()}

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple:
        row = self.data.iloc[idx]
        image_path = Path(row["image_path"])
        label_name = row["label"]

        image = Image.open(image_path).convert("RGB")
        label = self.label_to_index[label_name]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_num_classes_from_csv(csv_path: str) -> int:
    data = pd.read_csv(csv_path)
    return data["label"].nunique()


def get_label_mapping_from_csv(csv_path: str) -> Dict[str, int]:
    data = pd.read_csv(csv_path)
    labels = sorted(data["label"].unique())
    return {label: idx for idx, label in enumerate(labels)}