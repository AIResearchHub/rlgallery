

import torch
import torch.nn as nn

import numpy as np


class IQN(nn.Module):
    """
    Implicit Quantile Network
    """

    def __init__(self,
                 d_model,
                 n_cos=64
                 ):
        super(IQN, self).__init__()

        self.cos_embedding = nn.Linear(n_cos, d_model)
        self.linear = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, 1)

        self.n_cos = n_cos
        self.d_model = d_model

        self.pis = torch.FloatTensor([np.pi*i for i in range(1, n_cos+1)]).view(1, 1, n_cos).cuda()

        self.gelu = nn.GELU()

    def calc_cos(self, batch_size, n_tau=8):
        """
        Calculating the co-sin values depending on the number of tau samples
        """
        assert torch.equal(self.pis,
                           torch.FloatTensor([np.pi*i for i in range(1, self.n_cos+1)]).view(1, 1, self.n_cos).cuda())

        # (batch_size, n_tau, 1)
        taus = torch.rand(batch_size, n_tau).unsqueeze(-1).cuda()
        cos = torch.cos(taus * self.pis)

        assert cos.shape == (batch_size, n_tau, self.n_cos)

        cos = cos.view(batch_size * n_tau, self.n_cos)
        cos = self.gelu(self.cos_embedding(cos))
        cos = cos.view(batch_size, n_tau, self.d_model)

        return cos, taus

    def forward(self, x, n_tau):
        """
        Args:
            x (torch.tensor): (batch_size, 1, d_model)
            n_tau (int): Number of tau samples per input

        Returns:
            x (torch.tensor): (batch_size, n_tau)
            taus (torch.tensor): (batch_size, n_tau)

        """
        batch_size = x.size(0)
        assert x.shape == (batch_size, 1, self.d_model)

        cos, taus = self.calc_cos(batch_size, n_tau=n_tau)

        cos = cos.view(batch_size, n_tau, self.d_model)
        taus = taus.view(batch_size, n_tau)

        x = (x*cos).view(batch_size*n_tau, self.d_model)
        x = self.gelu(self.linear(x))
        x = self.out(x)
        x = x.view(batch_size, n_tau)

        return x, taus

