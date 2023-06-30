

import random

from .agent import Agent


class Random(Agent):

    def __init__(self, **kwargs):
        self.num_action = kwargs.num_action

    def get_action(self, obs):
        return random.randrange(0, self.num_action)

    def remember(self):
        pass

    def learn(self):
        pass

    def train(self):
        return 0.

