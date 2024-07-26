#!/usr/bin/env python3

import numpy as np
import torch
import gymnasium as gym
from ppo import Agent
import matplotlib.pyplot as plt
from tqdm import tqdm


if __name__ == '__main__':

    epsilon = 0.2
    gamma = 0.99
    gae_lambda = 0.8
    entropy_coef = 0
    batch_size = 32
    horizon = 2048
    learning_rate = 3e-4
    n_epochs = 10

    assert horizon > batch_size or horizon % batch_size == 0, "The rollout size should be a multiple of the batch size"

    env = gym.make("BipedalWalker-v3", hardcore=False, render_mode="human")
    input_dims = env.observation_space.shape[0]
    output_dims = env.action_space.shape[0]
    agent = Agent(input_dims, output_dims, epsilon, gamma, gae_lambda, entropy_coef, batch_size, learning_rate, n_epochs)
    agent.actor.load_state_dict(torch.load("models/actor.pt"))
    agent.critic.load_state_dict(torch.load("models/critic.pt"))

    n_episodes = 20
    episode_rewards = []
    avg_rewards = []

    for episode in range(n_episodes):
        state, _ = env.reset()
        terminated = truncated = False
        episode_reward = 0

        while not terminated and not truncated:
            action, prob, value = agent.choose_action(state)
            new_state, reward, terminated, truncated, _ = env.step(action)
            agent.remember(state, action, reward, value, prob, terminated)
            episode_reward += reward
            state = new_state

    env.close()

