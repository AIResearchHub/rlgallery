

from environment import SimpleEnv, AtariEnv
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
    agent = Random(num_action=env.action_size)

    for epoch in range(epochs):
        obs = env.reset()
        done = False
        while not done:
            action = agent.get_action()
            next_obs, reward, done, _, _ = env.step(action)

            obs = next_obs

            env.render()


if __name__ == "__main__":
    main()

