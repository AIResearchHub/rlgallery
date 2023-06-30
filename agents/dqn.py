

import torch
import torch.nn.functional as F

import random
import numpy as np
from copy import deepcopy

from .agent import Agent
from .replaybuffer import ReplayBuffer


class DQN(Agent):

    def __init__(self, model, size, bsz, num_action):
        self.size = size
        self.bsz = bsz
        self.num_action = num_action

        self.model = deepcopy(model)
        self.target_model = deepcopy(model)

        self.memory = ReplayBuffer()

    def get_action(self, obs):
        output = self.model(obs)

        return np.argmax(output)

    def remember(self, *args):
        self.memory.add(*args)

    def update(self, state, action, reward, new_state, done):
        expected = self.model(state)
        target = expected.detach()

        # create
        with torch.no_grad():
            q_value = np.max(self.target_model(new_state), axis=1)

            # reward + 0.99 * max_next_reward
            y = reward + self.gamma * q_value

            # if done replace y with reward
            y = np.where(done, reward, y)

            # replace original q with y for action taken
            target[np.arange(self.batch_size), action] = y

        loss = F.mse(expected, target)
        return loss

    def train(self):
        if self.frames < self.warmup:
            return 0.

        batch = self.memory(bsz=self.bsz)
        loss = self.update(*batch)

        return loss
