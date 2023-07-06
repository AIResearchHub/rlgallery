

import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):

    """
    Args:
        cls (nn.Module): what model to use (e.g. feed forward, cnn etc...)
    """

    def __init__(self, cls, state_size, action_size):
        super(Actor, self).__init__()

        self.actor = cls(state_size)
        self.out = nn.Linear(action_size)

    def forward(self, x):
        x = self.actor(x)
        x = self.out(x)

        return F.log_softmax(x, dim=-1)

