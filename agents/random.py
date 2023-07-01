

import random

from .agent import Agent


class Random(Agent):

    def __init__(self, model, action_size):
        self.action_size = action_size

    def get_action(self, obs):
        return random.randrange(0, self.action_size)

    def remember(self):
        pass

    def learn(self):
        pass

    def train(self):
        return 0.

