from torch import nn
from models.drnn import DRNN
import torch

class DRNN_mobility(nn.Module):
    def __init__(self, input_size, hidden_dim, num_layers, output_size, dropout=0.0) -> None:
        super(DRNN_mobility, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
        )
        self.hidden_size = hidden_dim
        self.num_layers = num_layers
        self.linear = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        y, _ = self.gru(x)
        return self.linear(y[:, -1, :])
