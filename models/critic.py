

import torch.nn as nn

from feedforward import FeedForward


class Critic(nn.Module):

    def __init__(self, cls):
        self.critic = cls()

    def forward(self, x):
        return self.ff(x)

