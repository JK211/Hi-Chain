import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import gym
from gym import spaces
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from typing import Tuple, Dict, List


class ReplayBuffer:
    """Experience replay buffer."""

    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.uint8)
        )

    def __len__(self):
        return len(self.buffer)


class BDQN(nn.Module):
    """Branching DQN network."""

    def __init__(self, input_dim, cache_size, total_blocks):
        super().__init__()
        self.cache_size = cache_size
        self.total_blocks = total_blocks

        # Shared feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU()
        )

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        # Advantage stream (evict action)
        self.advantage_remove = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, cache_size)
        )

        # Advantage stream (add action)
        self.advantage_add = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, total_blocks)
        )

    def forward(self, x):
        features = self.feature_extractor(x)
        values = self.value_stream(features)
        adv_remove = self.advantage_remove(features)
        q_remove = values + adv_remove - adv_remove.mean(dim=1, keepdim=True)
        adv_add = self.advantage_add(features)
        q_add = values + adv_add - adv_add.mean(dim=1, keepdim=True)
        return q_remove, q_add
