

from environment import Env, SimpleEnv, AtariEnv
from models import FeedForward
from agents import Random


"""
Environments:

CartPole-v1 (Discrete)
Pendulum-v1 (Continuous)
BreakoutDeterministic-v4 (Discrete)


"""


def main(env_name="CartPole-v1",
         epochs=100_000_000
         ):

    env = SimpleEnv(env_name)

    model = FeedForward(env.state_size, env.action_size, 64, 2)
    agent = Random(model, env.action_size)

    for epoch in range(epochs):
        done = False
        obs = env.reset()
        total_reward = 0.
        total_loss = 0.

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, _, _ = env.step(action)

            loss = agent.train()

            total_reward += reward
            total_loss += loss

            obs = next_obs
            env.render()


if __name__ == "__main__":
    main()

