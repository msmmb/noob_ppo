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
        T = len(self.states)
        batch_starts = np.arange(0, T, self.batch_size)
        indices = np.arange(T, dtype=np.int64)
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
    def __init__(self):
        super(ActorNN, self).__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc_mu = nn.Linear(64, 4)
        self.log_sigma = nn.Parameter(torch.zeros(4))

        for layer in [self.fc1, self.fc2, self.fc_mu]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.fc_mu(x)
        std_dev = torch.exp(self.log_sigma)
        distribution = Normal(mean, std_dev)
        return distribution


class CriticNN(nn.Module):
    def __init__(self):
        super(CriticNN, self).__init__()
        self.fc1 = nn.Linear(24, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

        for layer in [self.fc1, self.fc2, self.fc3]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.constant_(layer.bias, 0)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        value = self.fc3(x)
        return value

class Agent:
    def __init__(self):
        self.actor = ActorNN()
        self.critic = CriticNN()

    def choose_action(self, observation):
        with torch.no_grad():
            state = torch.tensor(observation, dtype=torch.float32)
            distribution = self.actor(state)
            action = distribution.sample()
            probs = distribution.log_prob(action)
            value = self.critic(state)

        return action.squeeze(0).numpy(), probs, value.item()


class Trainer:
    def __init__(self, epsilon, gamma, gae_lambda, entropy_coef, batch_size, actor_lr, critic_lr, n_epochs):
        self.agent = Agent()
        self.epsilon = epsilon
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.actor_optimizer = self.optimizer = optim.Adam(self.agent.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = self.optimizer = optim.Adam(self.agent.critic.parameters(), lr=critic_lr)
        self.memory = Buffer(batch_size=self.batch_size)

    def remember(self, state, action, reward, value, prob, done):
        self.memory.store_transition(state, action, reward, value, prob, done)

    def compute_advantages_returns(self, rewards, values, dones):
        T = len(rewards)
        advantages = np.zeros(T, dtype=np.float32)
        returns = np.zeros(T, dtype=np.float32)

        returns[-1] = rewards[-1]
        for t in reversed(range(T - 1)):
            next_value = values[t + 1] 
            delta = rewards[t] + self.gamma * next_value * (1 - int(dones[t])) - values[t]
            advantages[t] = delta + self.gamma * self.gae_lambda * advantages[t + 1] * (1 - int(dones[t]))
            returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - int(dones[t]))

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages, returns

    def learn(self):
        for _ in range(self.n_epochs):
            states, actions, rewards, values, probs, dones, batches = self.memory.get_steps()
            T = len(rewards)

            advantages, returns = self.compute_advantages_returns(rewards, values, dones)

            advantages = torch.tensor(advantages, dtype=torch.float32)
            values = torch.tensor(values, dtype=torch.float32)
            returns = torch.tensor(returns, dtype=torch.float32)

            for batch in batches:
                states_batch = torch.tensor(states[batch], dtype=torch.float32)
                probs_batch = torch.tensor(probs[batch], dtype=torch.float32)
                actions_batch = torch.tensor(actions[batch], dtype=torch.float32)

                distribution = self.agent.actor(states_batch)
                entropy = distribution.entropy()
                new_probs = distribution.log_prob(actions_batch)
                critic_value = self.agent.critic(states_batch).squeeze(1)

                prob_ratio = torch.exp(new_probs.sum(dim=1) - probs_batch.sum(dim=1))
                weighted_probs = advantages[batch] * prob_ratio
                weighted_clipped_probs = advantages[batch] * torch.clamp(prob_ratio, 1 - self.epsilon, 1 + self.epsilon)

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean() - self.entropy_coef * entropy.mean()
                critic_loss = ((critic_value - returns[batch]) ** 2).mean()

                self.actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.actor.parameters(), 0.5)
                self.actor_optimizer.step()

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.agent.critic.parameters(), 0.5)
                self.critic_optimizer.step()
