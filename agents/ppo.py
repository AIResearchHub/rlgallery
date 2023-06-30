

import torch
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Categorical
from torch.optim import Adam

import random

from .agent import Agent


class PPO(Agent):

    def __init__(self, actor, critic, num_action, warmup=100):
        self.num_action = num_action
        self.warmup = warmup

        self.frames = 0

        self.actor = actor
        self.critic = critic

        self.actor_opt = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=self.lr)

        # covariance matrices
        self.cov_var = torch.full(size=(num_action,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

    def get_action(self, obs):
        return random.randrange(0, self.num_action)

    def update(self):
        return 0.

    def train(self):
        if self.frames < self.warmup:
            return 0.

        batch = self.memory.sample()
        loss = self.update(*batch)

        return loss

