

import time
from datetime import datetime

from environment import Env, SimpleEnv, AtariEnv
from models import FeedForward, Actor, Critic
from agents import Random, PPO, DQN


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

    actor = Actor(FeedForward,
                  dim=64,
                  state_size=env.state_size,
                  action_size=env.action_size,
                  n_layers=2)

    critic = Critic(FeedForward,
                    dim=64,
                    state_size=env.state_size,
                    n_layers=2)

    agent = PPO(actor=actor,
                critic=critic)

    # agent = Random(model=model, action_size=env.action_size)
    dt = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    file = open(f"logs/{dt}", "w")

    start = time.time()
    for epoch in range(epochs):
        done = False
        obs, _ = env.reset()
        total_reward = 0.
        total_loss = 0.

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done, _, _ = env.step(action)

            agent.remember(obs, action, reward, next_obs, done)
            actor_loss, critic_loss = agent.train()

            total_reward += reward
            total_loss += critic_loss

            obs = next_obs

            # env.render()

        print(f"Episode {epoch} \t Reward {total_reward}")
        file.write('{}, {}, {}'.format(time.time()-start, epoch, total_reward))
        file.flush()


if __name__ == "__main__":
    main()

