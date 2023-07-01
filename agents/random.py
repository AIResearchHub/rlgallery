

import random

from .agent import Agent


class Random(Agent):

    def __init__(self, action_size, *args, **kwargs):
        self.action_size = action_size

    def get_action(self, obs):
        return random.randrange(0, self.action_size)

    def remember(self, *args, **kwargs):
        pass

    def learn(self):
        pass

    def train(self):
        return 0., 0.

