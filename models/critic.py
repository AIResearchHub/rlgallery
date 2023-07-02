

import torch.nn as nn


class Critic(nn.Module):

    def __init__(self, cls, state_size, dim, num_value):
        super(Critic, self).__init__()

        self.critic = cls(state_size, dim)
        self.out = nn.Linear(dim, num_value)

    def forward(self, x):
        x = self.critic(x)
        x = self.out(x)

        return x

