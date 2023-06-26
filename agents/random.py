

import random


class Random:

    def __init__(self, num_action):
        self.num_action = num_action

    def get_action(self):
        return random.randrange(0, self.num_action)
