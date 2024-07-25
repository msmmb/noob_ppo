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
        self.probs = []
        self.vals = []
        self.rewards = []
        self.stops = []

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return (np.array(self.states),
                np.array(self.actions),
                np.array(self.probs),
                np.array(self.vals),
                np.array(self.rewards),
                np.array(self.stops),
                batches)

    def store_memory(self, state, action, probs, vals, rewards, stop):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(rewards)
        self.stops.append(stop)

    def clear_memory(self):
        del self.states[:]
        del self.actions[:]
        del self.probs[:]
        del self.vals[:]
        del self.rewards[:]
        del self.stops[:]


class ActorNN(nn.Module):
    def __init__(self, input_dims, output_dims, alpha, batch_size, hidden_dims=64):
        super(ActorNN, self).__init__()
        self.fc1 = nn.Linear(input_dims, hidden_dims)
        self.fc2 = nn.Linear(hidden_dims, hidden_dims)
        self.fc_mu = nn.Linear(hidden_dims, output_dims)
        # self.fc_sigma = nn.Linear(hidden_dims, output_dims)
        self.log_sigma = nn.Parameter(torch.ones(output_dims))
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        mean = self.fc_mu(x)
        # std_dev = torch.exp(self.fc_sigma(x))
        std_dev = torch.exp(self.log_sigma)
        return mean, std_dev


class CriticNN(nn.Module):
    def __init__(self, input_dims, alpha, batch_size, hidden_dims=64):
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
    def __init__(self, input_dims, output_dims, epsilon, gamma, batch_size, lr, n_epochs):
        self.epsilon = epsilon
        self.gamma = gamma
        self.batch_size = batch_size
        self.lr = lr
        self.n_epochs = n_epochs

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.actor = ActorNN(self.input_dims, self.output_dims, self.lr, self.batch_size)
        self.critic = CriticNN(self.input_dims, self.lr, self.batch_size)
        self.memory = Buffer(self.batch_size)

    def remember(self, state, action, probs, vals, reward, stop):
        self.memory.store_memory(state, action, probs, vals, reward, stop)

    def choose_action(self, observation):
        state = torch.tensor(observation, dtype=torch.float32)

        mean, std_dev = self.actor(state)
        # print(mean, std_dev)
        distribution = Normal(mean, std_dev)
        action = distribution.sample().squeeze(0)
        probs = distribution.log_prob(action).squeeze(0)
        value = self.critic(state).squeeze(0)

        return action.squeeze(0).numpy(), probs.sum().item(), value.item()


    def learn(self):
        for _ in range(self.n_epochs):
            states, actions, old_probs, values, rewards, stops, batches = self.memory.generate_batches()

            advantage = np.zeros(len(rewards), dtype=np.float32)
            returns = np.zeros(len(rewards), dtype=np.float32)

            for t in reversed(range(len(rewards))):
                if t == len(rewards) - 1:
                    returns[t] = rewards[t]
                else:
                    returns[t] = rewards[t] + self.gamma * returns[t + 1] * (1 - int(stops[t]))

            for t in range(len(rewards)):
                advantage[t] = returns[t] - values[t]

            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            advantage = torch.tensor(advantage, dtype=torch.float32)
            values = torch.tensor(values, dtype=torch.float32)
            returns = torch.tensor(returns, dtype=torch.float32)

            for batch in batches:
                batch_states = torch.tensor(states[batch], dtype=torch.float32)
                old_probs_batch = torch.tensor(old_probs[batch], dtype=torch.float32)
                actions_batch = torch.tensor(actions[batch], dtype=torch.float32)

                mean, std_dev = self.actor(batch_states)
                distribution = Normal(mean, std_dev)
                new_probs = distribution.log_prob(actions_batch).sum(1, keepdim=True)
                critic_value = self.critic(batch_states).squeeze(1)

                prob_ratio = torch.exp(new_probs - old_probs_batch)
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = advantage[batch] * torch.clamp(prob_ratio, 1 - self.epsilon, 1 + self.epsilon)

                actor_loss = -torch.min(weighted_probs, weighted_clipped_probs).mean()
                mse_loss = nn.MSELoss()
                critic_loss = mse_loss(critic_value, returns[batch].detach())

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

                # self.actor.optimizer.zero_grad()
                # actor_loss.backward()
                # self.actor.optimizer.step()

                # self.critic.optimizer.zero_grad()
                # critic_loss.backward()
                # self.critic.optimizer.step()

