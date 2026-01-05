import torch
import torch.nn as nn
import random

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.net(x)

class Agent:
    def __init__(self, state_dim, grid):
        self.model = DQN(state_dim, grid*grid)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.epsilon = 1.0
        self.grid = grid

    def act(self, state):
        if random.random() < self.epsilon:
            idx = random.randint(0, self.grid*self.grid - 1)
        else:
            with torch.no_grad():
                idx = torch.argmax(self.model(torch.FloatTensor(state))).item()
        return idx % self.grid, idx // self.grid
