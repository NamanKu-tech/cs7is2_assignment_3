"""
Minimax and Minimax-with-Alpha-Beta agents.

MinimaxAgent     — plain minimax, no pruning.
MinimaxABAgent   — minimax with alpha-beta pruning + move ordering.

Both support:
  depth_limit=None  → full-depth search (feasible only for TTT)
  depth_limit=N     → N-ply look-ahead with heuristic evaluation (for C4)

nodes_visited is reset on each choose_move() call and can be inspected
afterward for the scalability analysis.
"""

import math
from games.tictactoe import TicTacToe
from games.connect4 import Connect4


class MinimaxAgent:
    name = "Minimax"

    def __init__(self, player=1, depth_limit=None):
        self.player = player
        self.depth_limit = depth_limit
        self.nodes_visited = 0

    def choose_move(self, game, training=False):  # noqa: ARG002
        self.nodes_visited = 0
        self._cache = {}  # transposition table (cleared per move)
        valid = game.get_valid_actions()
        best_action = valid[0]
        best_val = -math.inf

        for action in valid:
            game.make_move(action)
            val = self._minimax(game, depth=1, maximizing=False)
            game.undo_move(action)
            if val > best_val:
                best_val = val
                best_action = action

        return best_action

    def _minimax(self, game, depth, maximizing):
        self.nodes_visited += 1
        winner = game.get_winner()
        if winner is not None:
            return self._terminal_value(winner)

        if self.depth_limit is not None and depth >= self.depth_limit:
            return self._heuristic(game)

        # Transposition table lookup (only when no depth limit — i.e. TTT full search)
        if self.depth_limit is None:
            key = (game.get_state_key(), maximizing)
            if key in self._cache:
                return self._cache[key]

        valid = game.get_valid_actions()

        if maximizing:
            best = -math.inf
            for action in valid:
                game.make_move(action)
                best = max(best, self._minimax(game, depth + 1, False))
                game.undo_move(action)
        else:
            best = math.inf
            for action in valid:
                game.make_move(action)
                best = min(best, self._minimax(game, depth + 1, True))
                game.undo_move(action)

        if self.depth_limit is None:
            self._cache[key] = best
        return best

    def _terminal_value(self, winner):
        if winner == self.player:
            return 1.0
        elif winner == 0:
            return 0.0
        else:
            return -1.0

    def _heuristic(self, game):
        if isinstance(game, Connect4):
            return game.evaluate_heuristic(self.player) / 200.0  # normalise to ~[-1,1]
        return 0.0


class MinimaxABAgent:
    """Minimax with alpha-beta pruning and move ordering."""

    name = "Minimax+AB"

    def __init__(self, player=1, depth_limit=None):
        self.player = player
        self.depth_limit = depth_limit
        self.nodes_visited = 0

    def choose_move(self, game, training=False):  # noqa: ARG002
        self.nodes_visited = 0
        valid = self._order_moves(game, game.get_valid_actions())
        best_action = valid[0]
        best_val = -math.inf
        alpha = -math.inf
        beta = math.inf

        for action in valid:
            game.make_move(action)
            val = self._alphabeta(game, depth=1, alpha=alpha, beta=beta, maximizing=False)
            game.undo_move(action)
            if val > best_val:
                best_val = val
                best_action = action
            alpha = max(alpha, best_val)

        return best_action

    def _alphabeta(self, game, depth, alpha, beta, maximizing):
        self.nodes_visited += 1
        winner = game.get_winner()
        if winner is not None:
            return self._terminal_value(winner)

        if self.depth_limit is not None and depth >= self.depth_limit:
            return self._heuristic(game)

        valid = self._order_moves(game, game.get_valid_actions())

        if maximizing:
            value = -math.inf
            for action in valid:
                game.make_move(action)
                value = max(value, self._alphabeta(game, depth + 1, alpha, beta, False))
                game.undo_move(action)
                alpha = max(alpha, value)
                if alpha >= beta:
                    break  # beta cut-off
            return value
        else:
            value = math.inf
            for action in valid:
                game.make_move(action)
                value = min(value, self._alphabeta(game, depth + 1, alpha, beta, True))
                game.undo_move(action)
                beta = min(beta, value)
                if beta <= alpha:
                    break  # alpha cut-off
            return value

    def _order_moves(self, game, valid):
        """
        Order moves to improve alpha-beta efficiency:
        - Connect4: centre columns first (3,2,4,1,5,0,6)
        - TicTacToe: centre (4) → corners (0,2,6,8) → edges
        """
        if isinstance(game, Connect4):
            ordered = [c for c in Connect4.CENTER_ORDER if c in valid]
            ordered += [c for c in valid if c not in ordered]
            return ordered
        elif isinstance(game, TicTacToe):
            preference = [4, 0, 2, 6, 8, 1, 3, 5, 7]
            ordered = [a for a in preference if a in valid]
            ordered += [a for a in valid if a not in ordered]
            return ordered
        return valid

    def _terminal_value(self, winner):
        if winner == self.player:
            return 1.0
        elif winner == 0:
            return 0.0
        else:
            return -1.0

    def _heuristic(self, game):
        if isinstance(game, Connect4):
            return game.evaluate_heuristic(self.player) / 200.0
        return 0.0
