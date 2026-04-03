"""
Deep Q-Network (DQN) agent.

Architecture: fully-connected network with experience replay and target network.
Device policy:
  - TicTacToe → always CPU (network is tiny; GPU transfers dominate)
  - Connect4   → GPU if available, else CPU
"""

import random
import math
import pickle
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from games.tictactoe import TicTacToe

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done", "valid_next"])


class DQNNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super().__init__()
        layers = []
        prev = input_size
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, output_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class DQNAgent:
    name = "DQN"

    def __init__(
        self,
        player=1,
        state_size=9,
        action_size=9,
        hidden_sizes=(128, 128),
        lr=1e-3,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        batch_size=64,
        memory_size=50_000,
        target_update_freq=200,
        game_name="tictactoe",
        device="auto",
    ):
        self.player = player
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.steps = 0

        # Device: TTT always CPU (network too small to benefit from GPU transfer overhead)
        # C4: use MPS (Apple Silicon) > CUDA > CPU in that priority order
        if device == "auto":
            if game_name.lower() != "tictactoe":
                if torch.backends.mps.is_available():
                    self.device = torch.device("mps")
                elif torch.cuda.is_available():
                    self.device = torch.device("cuda")
                else:
                    self.device = torch.device("cpu")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.policy_net = DQNNetwork(state_size, hidden_sizes, action_size).to(self.device)
        self.target_net = DQNNetwork(state_size, hidden_sizes, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = deque(maxlen=memory_size)

    # ------------------------------------------------------------------ #
    # Action selection                                                     #
    # ------------------------------------------------------------------ #

    def choose_move(self, game, training=False):
        valid = game.get_valid_actions()

        if training and random.random() < self.epsilon:
            return random.choice(valid)

        state = self._get_state_tensor(game)
        with torch.no_grad():
            q_values = self.policy_net(state).squeeze().cpu()

        # Mask invalid actions to -inf
        masked = torch.full((self.action_size,), float("-inf"))
        for a in valid:
            masked[a] = q_values[a]

        return int(masked.argmax().item())

    # ------------------------------------------------------------------ #
    # Experience replay                                                    #
    # ------------------------------------------------------------------ #

    def store_transition(self, state_arr, action, reward, next_state_arr, done, valid_next):
        self.memory.append(Transition(state_arr, action, reward, next_state_arr, done, valid_next))

    def train_step(self):
        """Sample a batch and do one gradient update.  Returns loss or None."""
        if len(self.memory) < self.batch_size:
            return None

        batch = random.sample(self.memory, self.batch_size)
        states      = torch.FloatTensor(np.array([t.state      for t in batch])).to(self.device)
        actions     = torch.LongTensor( np.array([t.action     for t in batch])).to(self.device)
        rewards     = torch.FloatTensor(np.array([t.reward     for t in batch])).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in batch])).to(self.device)
        dones       = torch.FloatTensor(np.array([t.done       for t in batch])).to(self.device)

        # Current Q-values
        q_vals = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q-values
        with torch.no_grad():
            next_q = self.target_net(next_states)  # (batch, action_size)
            # Mask invalid next actions
            for i, t in enumerate(batch):
                invalid = [a for a in range(self.action_size) if a not in t.valid_next]
                if invalid:
                    next_q[i, invalid] = float("-inf")
            max_next_q = next_q.max(dim=1).values
            max_next_q[dones.bool()] = 0.0
            targets = rewards + self.gamma * max_next_q

        loss = nn.SmoothL1Loss()(q_vals, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping for stability
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path):
        data = {
            "policy": self.policy_net.state_dict(),
            "target": self.target_net.state_dict(),
            "epsilon": self.epsilon,
            "steps": self.steps,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            ckpt = pickle.load(f)
        self.policy_net.load_state_dict(ckpt["policy"])
        self.target_net.load_state_dict(ckpt["target"])
        self.epsilon = ckpt.get("epsilon", self.epsilon_min)
        self.steps = ckpt.get("steps", 0)

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _get_state_tensor(self, game):
        """
        Build input tensor, normalised to agent's perspective (always sees self as +1).
        """
        arr = game.get_state_array(perspective=self.player)
        return torch.FloatTensor(arr).unsqueeze(0).to(self.device)
