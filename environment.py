

import gym
from gym.spaces import Discrete

from collections import deque
import numpy as np
import random


def preprocess_frame(frame):
    # to grayscale
    frame = np.mean(frame, axis=2).astype(np.uint8)

    # down sample
    frame = frame[::2, ::2]

    frame = frame.astype("float32")
    frame /= 255.

    return frame


class SimpleEnv:

    def __init__(self, env_name):
        self.env = gym.make(env_name, render_mode="human")
        if type(self.env.action_space) == Discrete:
            self.discrete = True

    @property
    def state_size(self):
        return self.env.observation_space.shape[0]

    @property
    def action_size(self):
        if self.discrete:
            return self.env.action_space.n
        else:
            return self.env.action_space.shape[0]

    def reset(self):
        obs, _ = self.env.reset()
        return obs

    def step(self, action, bound=2):
        if not self.discrete:
            action *= bound

        obs, reward, done, _, _ = self.env.step(action)
        return obs, reward, done

    def render(self):
        self.env.render()


class AtariEnv:

    def __init__(self, env_name, auto_start=True, training=True, no_op_max=20):
        self.env = gym.make(env_name, render_mode="human")
        self.discrete = True
        self.last_lives = 0

        self.auto_start = auto_start
        self.fire = False

        self.training = training
        self.no_op_max = no_op_max

        self.stack = deque(maxlen=4)

    @property
    def state_size(self):
        return self.env.observation_space.shape

    @property
    def action_size(self):
        return self.env.action_space.n

    def reset(self):
        for i in range(4):
            self.stack.append(np.zeros((105, 80)))

        if self.auto_start:
            self.fire = True

        frame, _ = self.env.reset()

        if not self.training:
            for i in range(random.randint(1, self.no_op_max)):
                frame, _, _, _ = self.env.step(1)

        self.stack.append(preprocess_frame(frame))
        return np.array(self.stack, dtype=np.float32)

    def step(self, action):
        if self.fire:
            action = 1
            self.fire = False

        frame, reward, _, terminal, info = self.env.step(action)

        if info["lives"] < self.last_lives:
            life_lost = True
            self.fire = True

        else:
            life_lost = terminal

        self.last_lives = info["lives"]

        self.stack.append(preprocess_frame(frame))
        frame = np.array(self.stack, dtype=np.float32)
        return frame, reward, life_lost

    def render(self):
        """Called at each timestep to render"""
        self.env.render()
