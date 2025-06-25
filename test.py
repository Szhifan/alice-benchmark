import torch 
import torch 
import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (NT-Xent) for in-batch negatives.
    Assumes input features are L2-normalized.
    """
    def __init__(self, temperature: float = 0.5, device: torch.device = None):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_i: Tensor of shape (batch_size, feature_dim)
            z_j: Tensor of shape (batch_size, feature_dim)
        Returns:
            scalar NT-Xent loss
        """
        batch_size = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # 2N x D

        # Compute similarity matrix
        sim_matrix = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)  # (2N x 2N)
        sim_matrix = sim_matrix / self.temperature

        # Mask out self-similarities
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(self.device)
        sim_matrix = sim_matrix.masked_fill(mask, -9e15)

        # Positive pairs: i<->i+N
        positives = torch.cat([torch.diag(sim_matrix, batch_size), torch.diag(sim_matrix, -batch_size)])
        # Denominator: sum over all except self
        exp_sim = torch.exp(sim_matrix)
        denom = exp_sim.sum(dim=1)

        # Losses for all positives
        loss_pos = -torch.log(torch.exp(positives) / denom)
        loss = loss_pos.mean()
        return loss

# Example usage
if __name__ == "__main__":
    pass
    # class Encoder(nn.Module):
    #     def __init__(self, in_dim=128, out_dim=64):
    #         super().__init__()
    #         self.net = nn.Sequential(
    #             nn.Linear(in_dim, 256),
    #             nn.ReLU(),
    #             nn.Linear(256, out_dim)
    #         )

    #     def forward(self, x):
    #         x = self.net(x)
    #         return F.normalize(x, dim=1)

    # # Create random batch
    # batch_size, in_dim = 32, 128
    # x_i = torch.randn(batch_size, in_dim)
    # x_j = torch.randn(batch_size, in_dim)

    # encoder = Encoder(in_dim=in_dim, out_dim=64)
    # loss_fn = NTXentLoss(temperature=0.5)

    # z_i = encoder(x_i)
    # z_j = encoder(x_j)

    # loss = loss_fn(z_i, z_j)
    # print(f"NT-Xent Loss: {loss.item():.4f}")
