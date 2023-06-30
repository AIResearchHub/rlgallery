

import torch.nn as nn

from feedforward import FeedForward


class Actor(nn.Module):

    """
    Args:
        cls (nn.Module): what model to use (e.g. feed forward, cnn etc...)
    """

    def __init__(self, cls):
        self.actor = cls()

    def forward(self, x):
        return self.ff(x)

