import torch
import torch.nn as nn
import torch.nn.functional as F


class Entmax(nn.Module):
    def __init__(self, alpha=1.5, dim=-1):
        super(Entmax, self).__init__()
        self.alpha = alpha
        self.dim = dim

    def forward(self, z):
<<<<<<< HEAD
        # Applying the Entmax transformation element-wise
=======
>>>>>>> 4b57aae98d230f0282c345e6775888feda641368
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
<<<<<<< HEAD

        # Set tau to 0 where the condition is not met
        tau = torch.where(mask.any(dim=dim, keepdims=True),
                          tau, torch.tensor(0.0, dtype=torch.float32, device=z.device))
=======
        tau[~mask] = 0  # Set tau to 0 where the condition is not met
>>>>>>> 4b57aae98d230f0282c345e6775888feda641368

        # Apply the Entmax transformation
        entmax_result = F.relu(z - tau) ** alpha

<<<<<<< HEAD
        return entmax_result
=======
        return entmax_result
>>>>>>> 4b57aae98d230f0282c345e6775888feda641368
