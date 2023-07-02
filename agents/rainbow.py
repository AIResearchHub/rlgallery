

import torch
import torch.nn.functional as F
from torch.optim import Adam

import random
from copy import deepcopy

from .agent import Agent
from .replaybuffer import ReplayBuffer


class Rainbow(Agent):
    """
    Upgrades to DQN:
        Dueling Model Architecture
        Prioritized Experience Replay
        Implicit Quantile Network
    """

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.99
    warmup = 100

    bsz = 32
    lr = 1e-4
    gamma = 0.99

    n_step = 4

    def __init__(self, action_size, model):
        self.action_size = action_size

        self.model = deepcopy(model)
        self.target_model = deepcopy(model)

        self.opt = Adam(self.model.parameters(), lr=self.lr)

        self.memory = ReplayBuffer()
        self.frames = 0

    def get_action(self, obs):
        if random.random() < self.epsilon:
            return random.randrange(0, self.action_size)

        output = self.model(torch.tensor(obs).unsqueeze(0))
        output = torch.argmax(output.squeeze())
        return output

    def remember(self, obs, action, reward, next_obs, done):
        self.memory.add(next_obs[-1], action, reward, done)
        self.frames += 1

    def update(self, obs, action, reward, next_obs, done):
        expected = self.model(obs)
        target = expected.detach()

        with torch.no_grad():
            test = self.target_model(next_obs)
            q_value = torch.max(test, axis=1).values

            # reward + 0.99 * max_next_reward
            y = reward + self.gamma * q_value

            # if done replace y with reward
            y = torch.where(done, reward, y)

            # replace original q with y for action taken
            target[torch.arange(self.bsz, dtype=torch.int32), action] = y

        loss = F.huber_loss(expected, target)
        loss.backward()
        self.opt.step()

        return loss.item()

    def train(self):
        if self.frames < self.warmup:
            return 0., 0.

        batch = self.memory.sample(batch_size=self.bsz)
        loss = self.update(*batch)

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss, 0.
