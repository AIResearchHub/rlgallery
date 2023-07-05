

import time
from datetime import datetime

from environment import SimpleEnv, AtariEnv
from models import Actor, Critic, FeedForward, ConvNet, ConvLSTM
from agents import Random, PPO, DQN, R2D2


"""
Environments:

CartPole-v1 (Discrete)
Pendulum-v1 (Continuous)
BreakoutDeterministic-v4 (Discrete)


"""


def main(env_name="BreakoutDeterministic-v4",
         agent="r2d2",
         epochs=100_000_000
         ):

    # env = SimpleEnv(env_name)
    env = AtariEnv(env_name)

    if agent == "random":
        agent = Random(env.action_size)

    elif agent == "ppo":
        actor = Actor(ConvNet,
                      dim=64,
                      state_size=env.state_size,
                      action_size=env.action_size)
        critic = Critic(ConvNet,
                        dim=64,
                        state_size=env.state_size,
                        num_value=1)
        agent = PPO(actor=actor,
                    critic=critic)

    elif agent == "dqn":
        critic = Critic(ConvNet,
                        dim=64,
                        state_size=env.state_size)
        agent = DQN(action_size=env.action_size,
                    model=critic)

    elif agent == "r2d2":
        critic = Critic(ConvLSTM,
                        dim=64,
                        state_size=env.state_size,
                        num_value=env.action_size)
        agent = R2D2(action_size=env.action_size,
                     model=critic)

    dt = f"{datetime.now().strftime('%Y_%m_%d_%H_%M')}.txt"
    file = open(f"logs/{dt}", "w")

    start = time.time()
    for epoch in range(epochs):
        done = False
        obs = env.reset()
        total_reward = 0.
        total_loss = 0.

        while not done:
            action = agent.get_action(obs)
            next_obs, reward, done = env.step(action)

            agent.remember(obs, action, reward, next_obs, done)
            actor_loss, critic_loss = agent.train()

            total_reward += reward
            total_loss += critic_loss

            obs = next_obs

            # env.render()

        print(f"Episode {epoch} \t Reward {total_reward}")
        file.write('{}, {}, {}\n'.format(time.time()-start, epoch, total_reward))
        file.flush()


if __name__ == "__main__":
    main()

