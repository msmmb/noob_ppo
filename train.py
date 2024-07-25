#!/usr/bin/env python3

import numpy as np
import gymnasium as gym

from ppo import Agent

if __name__ == '__main__':
    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
    agent = Agent(input_dims=env.observation_space.shape[0], output_dims=env.action_space.shape[0], epsilon=0.2, gamma=0.999, batch_size=32, lr=3e-4, n_epochs=10)

    max_steps = 600
    max_episodes = 300
    episode_rewards = []

    for episode in range(max_episodes):
        state, info = env.reset()
        terminated = truncated = False
        episode_reward = 0

        for step in range(max_steps):
            action, prob, value = agent.choose_action(state)
            new_state, reward, terminated, truncated, info = env.step(action)
            agent.remember(state, action, prob, value, reward, terminated)
            episode_reward += reward

            if len(agent.memory.states) == agent.memory.batch_size:
                agent.learn()
                agent.memory.clear_memory()

            if terminated or truncated:
                break

            state = new_state

        episode_rewards.append(episode_reward)
        avg_reward = np.mean(episode_rewards[-30:])
        print(f"Episode: {episode}, Reward: {episode_reward:.2f}, Average Reward: {avg_reward:.2f}")
        print(agent.actor.log_sigma)

    env.close()

