import torch
import torch.nn as nn

class FiLM(nn.Module):
    """
    FiLM: Feature-wise Linear Modulation
    gamma, beta = MLP(context) -> x' = gamma * x + beta
    """
    def __init__(self, in_channels: int, dim_context: int = 64, hidden: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.fc = nn.Sequential(
            nn.Linear(dim_context, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 2 * in_channels),
        )

    def forward(self, x: torch.Tensor, context_vec: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        context_vec: (B, D)
        """
        b, c, h, w = x.shape
        gb = self.fc(context_vec)  # (B, 2C)
        gamma, beta = torch.chunk(gb, 2, dim=1)  # (B, C), (B, C)
        gamma = gamma.view(b, c, 1, 1)
        beta = beta.view(b, c, 1, 1)
        return gamma * x + beta
