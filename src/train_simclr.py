"""
Self-supervised pretraining with SimCLR on unlabeled images.

Usage:
    python -m src.train_simclr --data_dir data/unlabeled --epochs 100 --batch_size 64
"""

import argparse
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.augmentations import TransformConfig, get_simclr_transform
from src.datasets import UnlabeledImageDataset
from src.losses import NTXentLoss
from src.model import SimCLR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SimCLR self-supervised pretraining")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing unlabeled images")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4,
                        help="Weight decay for Adam optimizer")
    parser.add_argument("--temperature", type=float, default=0.5,
                        help="Temperature for NT-Xent loss")
    parser.add_argument("--image_size", type=int, default=224,
                        help="Image resize dimension")
    parser.add_argument("--projection_dim", type=int, default=128,
                        help="SimCLR projection head output dimension")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="DataLoader workers")
    parser.add_argument("--checkpoint_dir", type=str, default="outputs/checkpoints",
                        help="Directory to save model checkpoints")
    parser.add_argument("--save_every", type=int, default=20,
                        help="Save checkpoint every N epochs")
    parser.add_argument("--plots_dir", type=str, default="outputs/plots",
                        help="Directory to save loss curve plot")
    parser.add_argument("--metrics_dir", type=str, default="outputs/metrics",
                        help="Directory to save metrics JSON")
    return parser.parse_args()


def train_one_epoch(
    model: SimCLR,
    dataloader: DataLoader,
    criterion: NTXentLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    """Run a single training epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        # Each batch item is a tuple (view_i, view_j) from TwoCropsTransform
        view_i, view_j = batch
        view_i = view_i.to(device)
        view_j = view_j.to(device)

        _, z_i = model(view_i)
        _, z_j = model(view_j)

        loss = criterion(z_i, z_j)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


def save_loss_curve(losses: list, save_path: str) -> None:
    """Plot and save the training loss curve."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses) + 1), losses, linewidth=2, color="#2563eb")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("NT-Xent Loss", fontsize=12)
    plt.title("SimCLR Pretraining Loss", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Loss curve saved to {save_path}")


def main():
    args = parse_args()

    # ── Device setup ──────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── Directories ───────────────────────────────────────────────
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.plots_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    # ── Dataset & DataLoader ──────────────────────────────────────
    transform_cfg = TransformConfig(image_size=args.image_size)
    transform = get_simclr_transform(transform_cfg)
    dataset = UnlabeledImageDataset(root_dir=args.data_dir, transform=transform)
    print(f"Loaded {len(dataset)} unlabeled images from {args.data_dir}")

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ── Model, loss, optimizer, scheduler ─────────────────────────
    model = SimCLR(projection_dim=args.projection_dim).to(device)
    criterion = NTXentLoss(temperature=args.temperature)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"SimCLR model parameters: {total_params:,}")

    # ── Training loop ─────────────────────────────────────────────
    epoch_losses = []
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        avg_loss = train_one_epoch(model, dataloader, criterion, optimizer, device)
        scheduler.step()
        epoch_losses.append(avg_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time
        print(
            f"Epoch [{epoch:>3d}/{args.epochs}]  "
            f"Loss: {avg_loss:.4f}  "
            f"LR: {current_lr:.6f}  "
            f"Time: {elapsed:.0f}s"
        )

        # Save periodic checkpoint
        if epoch % args.save_every == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"simclr_epoch{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                    "args": vars(args),
                },
                ckpt_path,
            )
            print(f"  ↳ Checkpoint saved: {ckpt_path}")

    total_time = time.time() - start_time

    # ── Save final encoder checkpoint (for downstream use) ────────
    encoder_path = os.path.join(args.checkpoint_dir, "simclr_encoder_final.pt")
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"Final encoder saved: {encoder_path}")

    # ── Save loss curve ───────────────────────────────────────────
    curve_path = os.path.join(args.plots_dir, "simclr_loss_curve.png")
    save_loss_curve(epoch_losses, curve_path)

    # ── Save training metrics ─────────────────────────────────────
    metrics = {
        "final_loss": epoch_losses[-1],
        "min_loss": min(epoch_losses),
        "min_loss_epoch": int(epoch_losses.index(min(epoch_losses)) + 1),
        "total_epochs": args.epochs,
        "total_time_seconds": round(total_time, 2),
        "num_images": len(dataset),
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "temperature": args.temperature,
        "device": str(device),
        "epoch_losses": epoch_losses,
    }
    metrics_path = os.path.join(args.metrics_dir, "simclr_training.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Training metrics saved: {metrics_path}")

    print(f"\n✅ SimCLR pretraining complete — {args.epochs} epochs in {total_time:.1f}s")


if __name__ == "__main__":
    main()
