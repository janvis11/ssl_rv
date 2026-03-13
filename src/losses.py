import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """Normalized temperature-scaled cross-entropy loss for SimCLR."""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        batch_size = z_i.size(0)

        if batch_size != z_j.size(0):
            raise ValueError("z_i and z_j must have the same batch size.")

        z = torch.cat([z_i, z_j], dim=0)
        z = F.normalize(z, dim=1)

        similarity = torch.matmul(z, z.T) / self.temperature

        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        similarity = similarity.masked_fill(mask, float("-inf"))

        positive_indices = torch.arange(batch_size, device=z.device)
        positive_indices = torch.cat(
            [positive_indices + batch_size, positive_indices],
            dim=0,
        )

        loss = F.cross_entropy(similarity, positive_indices)
        return loss