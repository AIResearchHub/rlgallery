

import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, cls, state_size, dim, n_layers):
        super(Critic, self).__init__()

        self.critic = cls(state_size, dim, n_layers)
        self.out = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.critic(x)
        x = self.out(x)

        return x

