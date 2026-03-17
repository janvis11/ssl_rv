import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from augmentations import TransformConfig, get_simclr_transform
from datasets import UnlabeledImageDataset
from losses import NTXentLoss
from model import SimCLR


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_simclr_checkpoint(
    path: Path,
    epoch: int,
    model: SimCLR,
    optimizer: torch.optim.Optimizer,
    loss: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "encoder_state_dict": model.encoder.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss,
            "args": vars(args),
        },
        path,
    )


def save_encoder_only_checkpoint(path: Path, model: SimCLR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), path)


def train_one_epoch(
    model: SimCLR,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: NTXentLoss,
    device: torch.device,
) -> float:
    model.train()
    running_loss = 0.0

    progress = tqdm(loader, desc="SimCLR Training", leave=False)
    for x_i, x_j in progress:
        x_i = x_i.to(device, non_blocking=True)
        x_j = x_j.to(device, non_blocking=True)

        _, z_i = model(x_i)
        _, z_j = model(x_j)

        loss = loss_fn(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        progress.set_postfix(loss=f"{loss.item():.4f}")

    return running_loss / max(len(loader), 1)


def main():
    parser = argparse.ArgumentParser(description="Train SimCLR on unlabeled KITTI frames")
    parser.add_argument("--data_dir", type=str, default="data/unlabeled")
    parser.add_argument("--output_dir", type=str, default="outputs")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--projection_dim", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = get_simclr_transform(TransformConfig(image_size=args.image_size))
    dataset = UnlabeledImageDataset(root_dir=args.data_dir, transform=transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=len(dataset) >= args.batch_size,
    )

    model = SimCLR(projection_dim=args.projection_dim).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    loss_fn = NTXentLoss(temperature=args.temperature)

    checkpoint_dir = Path(args.output_dir) / "checkpoints"
    metrics_dir = Path(args.output_dir) / "metrics"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": []}
    best_loss = float("inf")

    print(f"Device: {device}")
    print(f"Unlabeled samples: {len(dataset)}")

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, loader, optimizer, loss_fn, device)
        history["train_loss"].append({"epoch": epoch, "loss": train_loss})

        print(f"Epoch [{epoch}/{args.epochs}] - train_loss: {train_loss:.4f}")

        save_simclr_checkpoint(
            checkpoint_dir / "simclr_full_latest.pt",
            epoch,
            model,
            optimizer,
            train_loss,
            args,
        )

        if train_loss < best_loss:
            best_loss = train_loss
            save_simclr_checkpoint(
                checkpoint_dir / "simclr_full_best.pt",
                epoch,
                model,
                optimizer,
                train_loss,
                args,
            )
            save_encoder_only_checkpoint(
                checkpoint_dir / "simclr_encoder_best.pt",
                model,
            )

    save_encoder_only_checkpoint(checkpoint_dir / "simclr_encoder_latest.pt", model)
    save_json(history, metrics_dir / "simclr_train_history.json")

    print(f"Training finished. Best loss: {best_loss:.4f}")


if __name__ == "__main__":
    main()