"""
Unified training loop for Q-Learning and DQN agents.

Key correctness feature — DELAYED REWARD:
When the opponent takes a move that ends the game (opponent wins), the agent's
*last* action must receive a negative reward.  Without this, the agent never
learns to avoid moves that hand the opponent a win on their next turn.

The loop structure ensures every agent action eventually receives an update:
  1. Agent moves → store (state, action) temporarily, no update yet.
  2. Opponent moves:
     - If opponent wins → retroactively call update(last_s, last_a, -1, …, done=True)
     - If game draws   → retroactively call update(last_s, last_a,  0, …, done=True)
     - If game ongoing → call update(last_s, last_a, 0, new_state, done=False) and clear buffer
  3. If agent moves end the game → call update immediately with +1 (win) or 0 (draw).
"""

import random
import numpy as np
from tqdm import tqdm

from agents.qlearning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent


class Trainer:
    def __init__(
        self,
        game_factory,
        agent,
        opponent,
        n_episodes=50_000,
        swap_sides=True,
        train_every=1,        # DQN: do a train_step every N episodes
        verbose=True,
        print_every=5000,
    ):
        """
        game_factory : callable returning a fresh game instance (e.g. TicTacToe)
        agent        : QLearningAgent or DQNAgent
        opponent     : any agent with choose_move(game) interface
        swap_sides   : randomly assign agent to P1 or P2 each episode
        train_every  : DQN gradient step frequency (per episode)
        """
        self.game_factory = game_factory
        self.agent = agent
        self.opponent = opponent
        self.n_episodes = n_episodes
        self.swap_sides = swap_sides
        self.train_every = train_every
        self.verbose = verbose
        self.print_every = print_every

        self.is_ql = isinstance(agent, QLearningAgent)
        self.is_dqn = isinstance(agent, DQNAgent)

    def train(self):
        """
        Run all episodes.
        Returns dict with keys:
          win_history  : list[int]  +1=win, 0=draw, -1=loss per episode
          loss_history : list[float] DQN loss per episode (empty for QL)
          q_table_sizes: list[int]  QL states visited (empty for DQN)
        """
        win_history = []
        loss_history = []
        q_size_history = []

        pbar = tqdm(range(self.n_episodes), disable=not self.verbose,
                    desc=f"Training {self.agent.name}", ncols=90)

        for ep in pbar:
            # Optionally swap sides for symmetric learning
            if self.swap_sides:
                agent_player = random.choice([1, -1])
            else:
                agent_player = self.agent.player

            result = self._run_episode(agent_player)
            win_history.append(result["outcome"])

            # DQN gradient update
            if self.is_dqn and (ep + 1) % self.train_every == 0:
                loss = self.agent.train_step()
                if loss is not None:
                    loss_history.append(loss)

            # Epsilon decay (both QL and DQN)
            self.agent.decay_epsilon()

            # Q-table size tracking for QL
            if self.is_ql:
                q_size_history.append(self.agent.q_table_size)

            # Progress update
            if self.verbose and (ep + 1) % self.print_every == 0:
                window = min(self.print_every, len(win_history))
                recent = win_history[-window:]
                wr = sum(1 for x in recent if x == 1) / window
                dr = sum(1 for x in recent if x == 0) / window
                extra = f"ε={self.agent.epsilon:.3f}"
                if self.is_ql:
                    extra += f" | states={self.agent.q_table_size}"
                pbar.set_postfix_str(f"win={wr:.1%} draw={dr:.1%} {extra}")

        return {
            "win_history": win_history,
            "loss_history": loss_history,
            "q_size_history": q_size_history,
        }

    # ------------------------------------------------------------------ #
    # Single episode                                                       #
    # ------------------------------------------------------------------ #

    def _run_episode(self, agent_player):
        game = self.game_factory()
        opponent_player = -agent_player

        # Temporarily assign players for this episode
        self.agent.player = agent_player

        last_agent_state = None   # perspective-normalised state before agent's last move
        last_agent_arr = None     # raw numpy array version (for DQN)
        last_agent_action = None

        outcome = None  # +1 / 0 / -1 from agent's perspective

        while True:
            current = game.current_player

            if current == agent_player:
                # ---- Agent's turn ----
                if self.is_ql:
                    state_key = game.get_state_key(perspective=agent_player)
                    action = self.agent.choose_move(game, training=True)
                else:
                    state_arr = game.get_state_array(perspective=agent_player)
                    action = self.agent.choose_move(game, training=True)

                reward, done = game.make_move(action)

                if done:
                    winner = game.get_winner()
                    final_reward = self._reward(winner, agent_player)
                    outcome = 1 if winner == agent_player else (0 if winner == 0 else -1)

                    if self.is_ql:
                        self.agent.update(
                            state_key, action, final_reward,
                            game.get_state_key(perspective=agent_player), [], True
                        )
                    else:
                        next_arr = game.get_state_array(perspective=agent_player)
                        self.agent.store_transition(state_arr, action, final_reward, next_arr, True, [])
                    break
                else:
                    # Store for delayed update
                    if self.is_ql:
                        last_agent_state = state_key
                    else:
                        last_agent_arr = state_arr
                    last_agent_action = action

            else:
                # ---- Opponent's turn ----
                action = self.opponent.choose_move(game, training=False)
                _, done = game.make_move(action)

                if done:
                    winner = game.get_winner()
                    final_reward = self._reward(winner, agent_player)
                    outcome = 1 if winner == agent_player else (0 if winner == 0 else -1)

                    # Delayed update for agent's last move
                    if last_agent_action is not None:
                        if self.is_ql:
                            self.agent.update(
                                last_agent_state, last_agent_action, final_reward,
                                game.get_state_key(perspective=agent_player), [], True
                            )
                        else:
                            next_arr = game.get_state_array(perspective=agent_player)
                            self.agent.store_transition(
                                last_agent_arr, last_agent_action, final_reward,
                                next_arr, True, []
                            )
                    break
                else:
                    # Game continues — flush pending agent update with reward=0
                    if last_agent_action is not None:
                        next_valid = game.get_valid_actions()
                        if self.is_ql:
                            next_key = game.get_state_key(perspective=agent_player)
                            self.agent.update(
                                last_agent_state, last_agent_action, 0.0,
                                next_key, next_valid, False
                            )
                        else:
                            next_arr = game.get_state_array(perspective=agent_player)
                            self.agent.store_transition(
                                last_agent_arr, last_agent_action, 0.0,
                                next_arr, False, next_valid
                            )
                        last_agent_state = None
                        last_agent_arr = None
                        last_agent_action = None

        return {"outcome": outcome if outcome is not None else 0}

    @staticmethod
    def _reward(winner, agent_player):
        if winner == agent_player:
            return 1.0
        elif winner == 0:
            return 0.0
        else:
            return -1.0
