

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):

    def __init__(self, channels, out_dim):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(3456, 512)
        self.out = nn.Linear(512, out_dim)

    def forward(self, x):
        # channels last to channels first
        x = x.permute(0, 3, 1, 2)

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)

        x = F.relu(self.fc(x))
        x = self.out(x)

        x = F.log_softmax(x, dim=-1)
        return x

