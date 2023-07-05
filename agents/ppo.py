

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.optim import Adam

import numpy as np
from copy import deepcopy

from .agent import Agent


class PPO(Agent):
    timesteps_per_batch = 1000
    n_updates = 5
    lr = 1e-4
    gamma = 0.95
    clip = 0.2

    def __init__(self, actor, critic):
        self.frames = 0

        self.actor = actor
        self.critic = critic

        self.actor_opt = Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = Adam(self.critic.parameters(), lr=self.lr)

        # Batch data
        self.logit = None
        self.ready = False

        self.obs = []
        self.actions = []
        self.logits = []
        self.rewards = []
        self.ep_rews = []

    @torch.no_grad()
    def get_action(self, obs):
        obs = torch.tensor(obs).unsqueeze(0)

        logits = self.actor(obs)
        dist = Categorical(logits=logits)

        action = dist.sample()

        self.logit = dist.log_prob(action).detach()

        return action.detach().item()

    def remember(self, obs, action, reward, next_obs, done):
        self.obs.append(obs)
        self.actions.append(action)
        self.logits.append(self.logit)
        self.ep_rews.append(reward)

        self.ready = False
        if done:
            self.rewards.append(deepcopy(self.ep_rews))
            self.ep_rews.clear()
            self.ready = True

        self.frames += 1

    def get_batch(self):
        obs = torch.tensor(np.array(self.obs))
        actions = torch.tensor(self.actions)
        logits = torch.tensor(self.logits)
        rewards = deepcopy(self.rewards)

        self.obs.clear()
        self.actions.clear()
        self.logits.clear()
        self.rewards.clear()

        return obs, actions, logits, rewards

    def evaluate(self, obs, actions):
        """
        Returns approximated value of observation and logits associated with action
        """

        V = self.critic(obs).squeeze()

        logits = self.actor(obs)
        dist = Categorical(logits=logits)

        log_probs = dist.log_prob(actions)
        return V, log_probs

    def compute_rtgs(self, rewards):
        """
        Compute the Reward-To-Go of each timestep in a batch given the rewards
        """
        # The rewards-to-go (rtg) per episode per batch to return.
        # The shape will be (num timesteps per episode)
        rtgs = []

        for ep_rews in reversed(rewards):
            discounted_reward = 0

            for rew in reversed(ep_rews):
                discounted_reward = rew + discounted_reward * self.gamma
                rtgs.insert(0, discounted_reward)

        # Convert the rewards-to-go into a tensor
        rtgs = torch.tensor(rtgs, dtype=torch.float32)
        return rtgs

    def update(self, obs, actions, logits, rewards):
        total_actor_loss = 0.
        total_critic_loss = 0.

        rewards = self.compute_rtgs(rewards)

        # get value of obs and subtract from rewards to get advantage
        V, _ = self.evaluate(obs, actions)
        A_k = rewards - V.detach()

        # normalize advantages
        A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

        for _ in range(self.n_updates):
            V, curr_logits = self.evaluate(obs, actions)

            ratios = torch.exp(curr_logits - logits)

            # Calculate surrogate losses.
            surr1 = ratios * A_k
            surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

            # gradient ascent to minimize (ratios * A_k)
            actor_loss = (-torch.min(surr1, surr2)).mean()

            # get critic to more accurately predict rewards
            critic_loss = F.mse_loss(V, rewards)

            self.actor_opt.zero_grad()
            actor_loss.backward(retain_graph=True)
            self.actor_opt.step()

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()

        return total_actor_loss/self.n_updates, total_critic_loss/self.n_updates

    def train(self):
        if self.frames > self.timesteps_per_batch and self.ready:
            self.frames = 0
            self.ready = False

            batch = self.get_batch()
            return self.update(*batch)

        return 0., 0.

