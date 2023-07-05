

import torch
import torch.nn.functional as F
from torch.optim import Adam

import random
import numpy as np
from copy import deepcopy

from .agent import Agent
from .replaybuffer import RecurrentBuffer


class R2D2(Agent):
    """
    Upgrades to Rainbow DQN:
        LSTM memory using burn-in and rollout
    """

    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.99
    warmup = 100

    buffer_size = 10_000
    bsz = 32
    lr = 1e-4
    gamma = 0.95

    n_step = 2
    burnin = 2
    rollout = 3
    block = burnin + rollout

    def __init__(self, action_size, model):
        self.action_size = action_size

        self.model = deepcopy(model)
        self.target_model = deepcopy(model)

        self.opt = Adam(self.model.parameters(), lr=self.lr)

        self.memory = RecurrentBuffer(buffer_size=self.buffer_size,
                                      bsz=self.bsz,
                                      block_len=self.block,
                                      n_step=self.n_step,
                                      gamma=self.gamma)
        self.frames = 0

        self.reset()

    def reset(self):
        self.state = (torch.zeros(1, 512), torch.zeros(1, 512))

    @torch.inference_mode()
    def get_action(self, obs):
        if random.random() < self.epsilon:
            return random.randrange(0, self.action_size)

        output, self.state = self.model(torch.tensor(obs).unsqueeze(0), self.state)
        output = torch.argmax(output.squeeze())

        return output

    def remember(self, obs, action, reward, next_obs, done):
        self.memory.add(obs, action, reward)
        self.frames += 1

        if done:
            self.memory.reset()
            self.reset()

    def update(self, obs, actions, rewards):

        with torch.no_grad():
            state = (torch.zeros(self.bsz, 512),
                     torch.zeros(self.bsz, 512))

            new_states = []
            for t in range(self.burnin+self.n_step):
                new_states.append((state[0].detach(), state[1].detach()))
                _, state = self.target_model.forward(obs[t], state)

            next_q = []
            for t in range(self.burnin+self.n_step, self.block+self.n_step):
                new_states.append((state[0].detach(), state[1].detach()))
                next_q_, state = self.target_model.forward(obs[t], state)
                next_q.append(next_q_)

            next_q = torch.stack(next_q)
            next_q = torch.max(next_q, axis=-1, keepdim=True)[0].to(torch.float32)

            next_q = rewards[self.burnin:] + self.gamma * next_q
            assert next_q.shape == (self.rollout, self.bsz, 1)

        self.model.zero_grad()

        # state = new_states[self.burnin].detach()
        state = (torch.zeros(self.bsz, 512),
                 torch.zeros(self.bsz, 512))
        expected = []
        target = []
        for t in range(self.burnin, self.block):
            expected_, state = self.model(obs[t], state)
            expected.append(expected_)

            target_ = expected_.detach()
            target_[torch.arange(self.bsz), actions[t]] = next_q[t-self.burnin]
            target.append(target_.detach())

        expected = torch.stack(expected)
        target = torch.stack(target)
        loss = F.huber_loss(expected, target)
        loss.backward()
        self.opt.step()

        loss = loss.item()
        return loss, new_states

    def train(self):
        if self.frames < self.warmup:
            return 0., 0.

        batch = self.memory.sample()
        loss = self.update(*batch)

        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, self.epsilon_min)

        return loss, 0.

