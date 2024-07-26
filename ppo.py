import numpy as np
import torch
from torch import nn
import torch.optim as optim
from torch.distributions.normal import Normal


class Buffer:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.states = []
        self.actions = []
        self.rewards = []
        self.values = []
        self.probs = []
        self.dones = []

    def get_steps(self):
        horizon = len(self.states)
        batch_starts = np.arange(0, horizon, self.batch_size)
        indices = np.arange(horizon, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_starts]

        return (np.array(self.states),
                np.array(self.actions),
                np.array(self.rewards),
                np.array(self.values),
                np.array(self.probs),
                np.array(self.dones),
                batches)

    def store_transition(self, state, action, reward, value, prob, done):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.values.append(value)
        self.probs.append(prob)
        self.dones.append(done)

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.rewards[:]
        del self.values[:]
        del self.probs[:]
        del self.dones[:]


class ActorNN(nn.Module):
    def __init__(self, input_dims, output_dims, alpha, hidden_dims=64):
        super(ActorNN, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc_mu = nn.Linear(hidden_dims, output_dims)
        self.log_sigma = nn.Parameter(torch.zeros(output_dims))
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.fc_mu(x)
        std_dev = torch.exp(self.log_sigma)
        distribution = Normal(mean, std_dev)
        return distribution


class CriticNN(nn.Module):
    def __init__(self, input_dims, alpha, hidden_dims=64):
        super(CriticNN, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc3 = nn.Linear(hidden_dims, 1)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value


class Agent:
    def __init__(self, input_dims, output_dims, epsilon, gamma, gae_lambda, entropy_coef, batch_size, lr, n_epochs):
        self.epsilon = epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.actor = ActorNN(self.input_dims, self.output_dims, self.lr)
        self.critic = CriticNN(self.input_dims, self.lr)
        self.memory = Buffer(batch_size=self.batch_size)

    def remember(self, state, action, reward, value, prob, done):
        self.memory.store_transition(state, action, reward, value, prob, done)

    def choose_action(self, observation):
        with torch.no_grad():
            state = torch.tensor(observation, dtype=torch.float32)
            distribution = self.actor(state)
            action = distribution.sample()
            probs = distribution.log_prob(action)
            value = self.critic(state)

        return action.squeeze(0).numpy(), probs, value.item()

    def compute_advantages(self, rewards, values, dones):
        horizon = len(rewards)
        advantages = np.zeros(horizon, dtype=np.float32)
        returns = np.zeros(horizon, dtype=np.float32)

        returns[-1] = rewards[-1]
        for t in reversed(range(horizon - 1)):
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - int(dones[t]))

        for t in reversed(range(horizon - 1)):
            next_value = values[t + 1] 
            delta = rewards[t] + self.gamma * next_value * (1 - int(dones[t])) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * advantages[t + 1] * (1 - int(dones[t]))

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns


    def learn(self):
        for _ in range(self.n_epochs):
            states, actions, rewards, values, probs, dones, batches = self.memory.get_steps()
            horizon = len(rewards)

            advantages, returns = self.compute_advantages(rewards, values, dones)

            advantages = torch.tensor(advantages, dtype=torch.float32)
            values = torch.tensor(values, dtype=torch.float32)
            returns = torch.tensor(returns, dtype=torch.float32)

            for batch in batches:
                states_batch = torch.tensor(states[batch], dtype=torch.float32)
                probs_batch = torch.tensor(probs[batch], dtype=torch.float32)
                actions_batch = torch.tensor(actions[batch], dtype=torch.float32)

                distribution = self.actor(states_batch)
                entropy = distribution.entropy()
                new_probs = distribution.log_prob(actions_batch)
                critic_value = self.critic(states_batch).squeeze(1)

                prob_ratio = torch.exp(new_probs.sum(dim=1) - probs_batch.sum(dim=1))
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = advantages[batch] * torch.clamp(prob_ratio, 1 - self.epsilon, 1 + self.epsilon)

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean() - self.entropy_coef * entropy.mean()
                critic_loss = ((critic_value - returns[batch]) ** 2).mean()

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.actor.optimizer.step()

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.critic.optimizer.step()
