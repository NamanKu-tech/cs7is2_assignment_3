"""
Tabular Q-Learning agent.

State representation: perspective-normalised board tuple
  state_key = tuple(board.flatten() * self.player)
so the same Q-table is used regardless of which side the agent plays.

Q-table: dict[state_key -> dict[action -> float]]
"""

import random
import pickle
from collections import defaultdict


class QLearningAgent:
    name = "Q-Learning"

    def __init__(
        self,
        player=1,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.9995,
    ):
        self.player = player
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        # state -> {action -> Q-value}
        self.q_table = defaultdict(lambda: defaultdict(float))

    # ------------------------------------------------------------------ #
    # Action selection                                                     #
    # ------------------------------------------------------------------ #

    def choose_move(self, game, training=False):
        state = game.get_state_key(perspective=self.player)
        valid = game.get_valid_actions()

        if training and random.random() < self.epsilon:
            return random.choice(valid)

        # Greedy: pick action with highest Q-value among valid moves
        best_val = -float("inf")
        best_action = None
        for action in valid:
            q = self.q_table[state][action]
            if q > best_val:
                best_val = q
                best_action = action

        return best_action if best_action is not None else random.choice(valid)

    # ------------------------------------------------------------------ #
    # Q-update                                                             #
    # ------------------------------------------------------------------ #

    def update(self, state, action, reward, next_state, next_valid, done):
        """
        Standard Q-learning update.
        state / next_state must already be perspective-normalised tuples.
        """
        current_q = self.q_table[state][action]

        if done or not next_valid:
            target = reward
        else:
            future_qs = [self.q_table[next_state][a] for a in next_valid]
            target = reward + self.gamma * max(future_qs)

        self.q_table[state][action] = current_q + self.alpha * (target - current_q)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path):
        data = {
            "q_table": dict(self.q_table),
            "epsilon": self.epsilon,
            "alpha": self.alpha,
            "gamma": self.gamma,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def load(self, path):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: defaultdict(float), data["q_table"])
        self.epsilon = data.get("epsilon", self.epsilon_min)
        self.alpha = data.get("alpha", self.alpha)
        self.gamma = data.get("gamma", self.gamma)

    @property
    def q_table_size(self):
        return len(self.q_table)
