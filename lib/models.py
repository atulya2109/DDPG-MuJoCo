import torch.nn as nn
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, capacity, obs_dim, action_dim) -> None:
        self.states = np.zeros((capacity, obs_dim))
        self.actions = np.zeros((capacity, action_dim))
        self.rewards = np.zeros((capacity, 1))
        self.next_states = np.zeros((capacity, obs_dim))
        self.dones = np.zeros((capacity, 1))
        self.position = 0
        self.size = 0
        self.capacity = capacity

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.randint(0, self.size, size=batch_size)

        return (
            self.states[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_states[indices],
            self.dones[indices],
        )

    def __len__(self):
        return self.size


class DDPGActor(nn.Module):
    def __init__(self, obs_size, action_size):
        super(DDPGActor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.ReLU(),
            nn.Linear(400, 300),
            nn.ReLU(),
            nn.Linear(300, action_size),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.net(x)


class DDPGCritic(nn.Module):
    def __init__(self, obs_size, act_size):
        super(DDPGCritic, self).__init__()

        self.obs_net = nn.Sequential(
            nn.Linear(obs_size, 400),
            nn.LayerNorm(400),
            nn.ReLU(),
        )

        self.out_net = nn.Sequential(
            nn.Linear(400 + act_size, 300),
            nn.LayerNorm(300),
            nn.ReLU(),
            nn.Linear(300, 1),
        )

    def forward(self, x, a):
        obs = self.obs_net(x)
        return self.out_net(torch.cat([obs, a], dim=1))
