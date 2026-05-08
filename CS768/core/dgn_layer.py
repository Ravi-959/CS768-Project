import torch
import torch.nn as nn
import torch.nn.functional as F

from core.directional_aggregators import bav, bdx


class DGNLayer(nn.Module):

    def __init__(
        self,
        hidden_dim,
        dropout=0.1
    ):

        super().__init__()

        self.linear = nn.Linear(
            hidden_dim * 3,
            hidden_dim
        )

        self.norm = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, Fmat):

        residual = x

        x_mean = x

        x_av = torch.bmm(
            bav(Fmat),
            x
        )

        x_dx = torch.bmm(
            bdx(Fmat),
            x
        )

        x = torch.cat(
            [x_mean, x_av, x_dx],
            dim=-1
        )

        x = self.linear(x)

        x = F.relu(x)

        x = self.dropout(x)

        x = x + residual

        x = self.norm(x)

        return x