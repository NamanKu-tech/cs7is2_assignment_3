"""
Evaluation utilities.

evaluate()        — play N games between two agents, return W/D/L stats.
round_robin()     — full round-robin tournament among a dict of agents.
"""

import time
import numpy as np
from tqdm import tqdm


class Evaluator:
    def __init__(self, game_factory, n_games=200, swap_sides=True, verbose=False):
        """
        game_factory : callable returning fresh game
        n_games      : total games to play (split evenly between sides when swap_sides=True)
        swap_sides   : play half games with agent1 as P1, half as P2
        """
        self.game_factory = game_factory
        self.n_games = n_games
        self.swap_sides = swap_sides
        self.verbose = verbose

    def evaluate(self, agent1, agent2):
        """
        Returns dict:
          wins, draws, losses, win_rate, draw_rate, loss_rate,
          avg_game_length, avg_move_time_ms,
          agent1_nodes (list), agent2_nodes (list)
        """
        wins = draws = losses = 0
        total_length = 0
        agent1_times = []
        agent1_nodes = []
        agent2_nodes = []

        games_per_side = self.n_games // 2 if self.swap_sides else self.n_games
        configs = []
        if self.swap_sides:
            configs = [(1, -1)] * games_per_side + [(-1, 1)] * games_per_side
        else:
            configs = [(1, -1)] * self.n_games

        iter_obj = tqdm(configs, desc=f"{agent1.name} vs {agent2.name}",
                        disable=not self.verbose, ncols=80)

        for a1_player, a2_player in iter_obj:
            game = self.game_factory()
            agent1.player = a1_player
            agent2.player = a2_player
            length = 0

            while not game.is_terminal():
                if game.current_player == a1_player:
                    t0 = time.perf_counter()
                    action = agent1.choose_move(game, training=False)
                    agent1_times.append((time.perf_counter() - t0) * 1000)
                    if hasattr(agent1, "nodes_visited"):
                        agent1_nodes.append(agent1.nodes_visited)
                else:
                    action = agent2.choose_move(game, training=False)
                    if hasattr(agent2, "nodes_visited"):
                        agent2_nodes.append(agent2.nodes_visited)

                game.make_move(action)
                length += 1

            winner = game.get_winner()
            total_length += length
            if winner == a1_player:
                wins += 1
            elif winner == 0:
                draws += 1
            else:
                losses += 1

        total = wins + draws + losses
        return {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "win_rate": wins / total,
            "draw_rate": draws / total,
            "loss_rate": losses / total,
            "avg_game_length": total_length / total,
            "avg_move_time_ms": np.mean(agent1_times) if agent1_times else 0.0,
            "avg_nodes": np.mean(agent1_nodes) if agent1_nodes else 0.0,
        }

    def round_robin(self, agents):
        """
        agents: dict[name -> agent]
        Returns:
          matrix : NxN numpy array of win rates (row=agent, col=opponent)
          results: nested dict[name][opponent_name] = stats dict
        """
        names = list(agents.keys())
        n = len(names)
        matrix = np.zeros((n, n))
        results = {name: {} for name in names}

        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
        pbar = tqdm(pairs, desc="Round-robin", ncols=80, disable=not self.verbose)

        for i, j in pbar:
            a1 = agents[names[i]]
            a2 = agents[names[j]]
            pbar.set_postfix_str(f"{names[i]} vs {names[j]}")
            stats = self.evaluate(a1, a2)
            matrix[i, j] = stats["win_rate"]
            results[names[i]][names[j]] = stats

        return matrix, names, results
