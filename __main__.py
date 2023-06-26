

from environment import AtariEnv
from agents import Random


def main(env_name="BreakoutDeterministic-v4",
         total_timesteps=100_000_000
         ):

    env = AtariEnv(env_name)
    agent = Random()

    env.reset()

    done = False
    while not done:
        env.step(agent.get_action())

