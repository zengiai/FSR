import torch
import torch.nn as nn
import torch.nn.functional as F


class Entmax(nn.Module):
    def __init__(self, alpha=1.5, dim=-1):
        super(Entmax, self).__init__()
        self.alpha = alpha
        self.dim = dim

    def forward(self, z):
        return self.entmax(z, self.alpha, self.dim)

    @staticmethod
    def entmax(z, alpha, dim=-1):
        # Entmax transformation
        sorted_z, _ = torch.sort(z, descending=True, dim=dim)
        cumsum = sorted_z.cumsum(dim=dim)

        t = torch.arange(1, z.size(dim) + 1, dtype=torch.float32, device=z.device)
        t = t.view(1, -1)  # Reshape t for broadcasting

        # Find the alpha threshold
        mask = sorted_z - (1 / t) * (cumsum - 1) > 0
        k_max = mask.sum(dim=dim, keepdim=True)

        tau = (cumsum.gather(dim, k_max) - 1) / k_max
        tau[~mask] = 0  # Set tau to 0 where the condition is not met

        # Apply the Entmax transformation
        entmax_result = F.relu(z - tau) ** alpha

        return entmax_result