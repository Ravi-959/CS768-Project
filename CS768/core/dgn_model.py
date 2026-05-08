import torch
import torch.nn as nn

from core.dgn_layer import DGNLayer


class DGN(nn.Module):

    def __init__(
        self,
        hidden_dim,
        num_layers,
        dropout=0.1
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            28,
            hidden_dim
        )

        self.layers = nn.ModuleList()

        for _ in range(num_layers):

            self.layers.append(
                DGNLayer(
                    hidden_dim,
                    dropout
                )
            )

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(
        self,
        x,
        Fmat,
        mask
    ):

        x = x.squeeze(-1)

        x = self.embedding(x.long())

        for layer in self.layers:
            x = layer(x, Fmat)

        mask = mask.unsqueeze(-1)

        x = x * mask

        graph_embedding = x.sum(dim=1)

        graph_embedding = graph_embedding / mask.sum(dim=1)

        out = self.readout(graph_embedding)

        return out.squeeze(-1)