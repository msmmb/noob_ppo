#!/usr/bin/env python3

import numpy as np
import torch
import gymnasium as gym
import matplotlib.pyplot as plt
from tqdm import tqdm
from ppo import Trainer


if __name__ == '__main__':

    epsilon = 0.2
    gamma = 0.99
    gae_lambda = 0.95
    entropy_coef = 0
    batch_size = 32
    T = 2048
    actor_lr = 3e-4
    critic_lr = 4e-4
    n_epochs = 10

    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode=None)
    trainer = Trainer(epsilon, gamma, gae_lambda, entropy_coef, batch_size, actor_lr, critic_lr, n_epochs)

    n_episodes = 1000
    max_steps = 600
    episode_rewards = []
    avg_rewards = []
    max_reward = -1000

    for episode in (pbar := tqdm(range(n_episodes), desc="avg reward = N/A / max reward = N/A ")):
        state, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0

        for _ in range(max_steps):
            action, prob, value = trainer.agent.choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            trainer.remember(state, action, reward, value, prob, terminated)
            episode_reward += reward

            if len(trainer.memory.states) >= T:
                trainer.learn()
                trainer.memory.clear_memory()

            if terminated or truncated:
                break

            state = new_state

        episode_rewards.append(episode_reward)
        if episode_reward > max_reward: max_reward = episode_reward
        avg_reward = np.mean(episode_rewards[-100:])
        avg_rewards.append(avg_reward)
        pbar.set_description(f"avg reward = {avg_reward:.2f} / max reward = {max_reward:.2f} ")

    env.close()
    torch.save(trainer.agent.actor.state_dict(), "models/actor.pt")
    torch.save(trainer.agent.critic.state_dict(), "models/critic.pt")

    plt.title('Rewards')
    plt.plot(episode_rewards, label="per episode")
    plt.plot(avg_rewards, label="average", color="red")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.savefig("media/training_plot.png")
    plt.show()
        

