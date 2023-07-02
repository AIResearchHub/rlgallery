

import torch
import random
import numpy as np
from collections import deque
from copy import deepcopy


class RecurrentBuffer:

    def __init__(self, buffer_size, bsz, block_len, n_step, gamma):
        self.buffer_size = buffer_size
        self.bsz = bsz
        self.block_len = block_len
        self.n_step = n_step

        self.gamma = np.full(n_step, gamma) ** np.arange(n_step)

        self.obs = deque(maxlen=buffer_size)
        self.actions = deque(maxlen=buffer_size)
        self.rewards = deque(maxlen=buffer_size)

        self.tobs = []
        self.tactions = []
        self.trewards = []

        self.size = 0
        self.ptr = 0

    def reset(self):
        self.obs.append(deepcopy(self.tobs))
        self.actions.append(deepcopy(self.tactions))
        self.rewards.append(deepcopy(self.trewards))

        self.tobs.clear()
        self.tactions.clear()
        self.trewards.clear()

    def add(self, state, action, reward):
        self.tobs.append(state)
        self.tactions.append(action)
        self.trewards.append(reward)

        self.size += 1
        self.size = min(self.size, self.buffer_size)

    def sample(self):

        obs = []
        actions = []
        rewards = []

        for i in range(self.bsz):
            bidx = random.randrange(0, self.size)
            tidx = random.randrange(0, self.lengths[bidx]-self.n_step-self.block_len+1)

            obs.append([self.obs[tidx+t] for t in range(self.block_len + self.n_step)])
            actions.append([self.actions[tidx+t] for t in range(self.block_len)])
            rewards.append([self.rewards[tidx+t:tidx+t+self.n_step] for t in range(self.block_len)])

        obs = torch.tensor(obs)
        actions = torch.tensor(actions).view(self.bsz, self.block_len+self.n_step, 1)
        rewards = torch.tensor(np.sum(np.array(rewards) * self.gamma, axis=2)).view(self.bsz, self.block_len, 1)

        return obs, actions, rewards

