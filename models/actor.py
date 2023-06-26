

import torch.nn as nn

from feedforward import FeedForward


class Actor(nn.Module):

    def __init__(self):
        self.ff = FeedForward()

    def forward(self, x):
        return self.ff(x)

