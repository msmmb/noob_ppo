#!/usr/bin/env python3

import torch
import gymnasium as gym
from ppo import Agent


if __name__ == '__main__':

    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
    agent = Agent()
    agent.actor.load_state_dict(torch.load("models/actor.pt"))
    agent.critic.load_state_dict(torch.load("models/critic.pt"))
    n_episodes = 20

    for episode in range(n_episodes):
        state, _ = env.reset()
        terminated = truncated = False

        while not terminated and not truncated:
            action, prob, value = agent.choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            state = new_state

    env.close()

