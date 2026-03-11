import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """
    Normalized temperature-scaled cross entropy loss (SimCLR).
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: Tensor of shape [batch_size, dim]
            z_j: Tensor of shape [batch_size, dim]
        """
        batch_size = z_i.size(0)

        z = torch.cat([z_i, z_j], dim=0)  # [2B, D]
        z = F.normalize(z, dim=1)

        similarity_matrix = torch.matmul(z, z.T) / self.temperature  # [2B, 2B]

        # Remove self-similarity
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        similarity_matrix = similarity_matrix.masked_fill(mask, float("-inf"))

        # Positive pairs:
        # i <-> i+B and i+B <-> i
        positive_indices = torch.arange(batch_size, device=z.device)
        positive_indices = torch.cat([positive_indices + batch_size, positive_indices], dim=0)

        loss = F.cross_entropy(similarity_matrix, positive_indices)
        return loss