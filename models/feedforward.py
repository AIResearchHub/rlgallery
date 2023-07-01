

import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    """
    A simple feed forward network.

    Architecture:
        Sequential(
            Linear(dim, inner_dim)
            GELU()
            Linear(inner_dim, dim)
        )

    Parameters:
        dim (int): The dimension of the input and output
        inner_dim (int): The dimension of the hidden layer
    """

    def __init__(self, in_dim, inner_dim, n_layers):
        super(FeedForward, self).__init__()

        self.in_layer = nn.Linear(in_dim, inner_dim)
        self.ff = nn.ModuleList([
            nn.Linear(inner_dim, inner_dim)
            for _ in range(n_layers)
        ])

    def forward(self, x):
        x = F.relu(self.in_layer(x))

        for layer in self.ff:
            x = F.relu(layer(x))

        return x

