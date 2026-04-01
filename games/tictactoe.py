"""
Tic Tac Toe environment.
Board: 3x3 numpy array.  Players: 1 (X) and -1 (O).  Empty: 0.
Actions: integers 0-8 (row-major: action = row*3 + col).
"""

import numpy as np


class TicTacToe:
    ROWS = 3
    COLS = 3
    PLAYER_1 = 1
    PLAYER_2 = -1

    def __init__(self):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = self.PLAYER_1
        self._winner = None

    # ------------------------------------------------------------------ #
    # Core interface                                                        #
    # ------------------------------------------------------------------ #

    def reset(self):
        self.board[:] = 0
        self.current_player = self.PLAYER_1
        self._winner = None
        return self

    def clone(self):
        g = TicTacToe()
        g.board = self.board.copy()
        g.current_player = self.current_player
        g._winner = self._winner
        return g

    def get_valid_actions(self):
        """Return list of valid action indices (0-8)."""
        return [int(i) for i in np.where(self.board.flatten() == 0)[0]]

    def make_move(self, action):
        """Place current player's piece at action (0-8).  Returns (reward, done)."""
        row, col = divmod(action, self.COLS)
        assert self.board[row, col] == 0, f"Cell ({row},{col}) already occupied"
        self.board[row, col] = self.current_player
        winner = self._check_winner()
        if winner is not None:
            self._winner = winner
            done = True
            reward = 1.0 if winner == self.current_player else (0.0 if winner == 0 else -1.0)
        else:
            done = False
            reward = 0.0
        self.current_player *= -1
        return reward, done

    def undo_move(self, action):
        row, col = divmod(action, self.COLS)
        self.board[row, col] = 0
        self.current_player *= -1
        self._winner = None

    def is_terminal(self):
        return self._check_winner() is not None

    def get_winner(self):
        """Returns 1, -1, 0 (draw), or None (ongoing)."""
        return self._check_winner()

    # ------------------------------------------------------------------ #
    # State representations                                                #
    # ------------------------------------------------------------------ #

    def get_state_key(self, perspective=None):
        """
        Hashable tuple for Q-table lookup.
        perspective: if given (1 or -1), normalise board so that player is always +1.
        This halves the effective state space and enables side-invariant learning.
        """
        flat = self.board.flatten()
        if perspective is not None:
            flat = flat * perspective
        return tuple(flat)

    def get_state_array(self, perspective=None):
        """Float32 numpy array (9,) for DQN input."""
        flat = self.board.flatten().astype(np.float32)
        if perspective is not None:
            flat = flat * perspective
        return flat

    # ------------------------------------------------------------------ #
    # Rendering                                                            #
    # ------------------------------------------------------------------ #

    def render(self, show_indices=False):
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = []
        lines.append("  " + " ".join(str(c) for c in range(self.COLS)))
        for r in range(self.ROWS):
            row_str = str(r) + " " + " ".join(symbols[self.board[r, c]] for c in range(self.COLS))
            lines.append(row_str)
        if show_indices:
            lines.append("\nAction indices:")
            for r in range(self.ROWS):
                lines.append("  " + " ".join(str(r * self.COLS + c) for c in range(self.COLS)))
        return "\n".join(lines)

    def print_board(self, show_indices=False):
        print(self.render(show_indices))
        print()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _check_winner(self):
        b = self.board
        for player in (1, -1):
            # rows
            for r in range(self.ROWS):
                if np.all(b[r, :] == player):
                    return player
            # cols
            for c in range(self.COLS):
                if np.all(b[:, c] == player):
                    return player
            # diagonals
            if b[0, 0] == b[1, 1] == b[2, 2] == player:
                return player
            if b[0, 2] == b[1, 1] == b[2, 0] == player:
                return player
        # Draw?
        if not any(self.board[r, c] == 0 for r in range(self.ROWS) for c in range(self.COLS)):
            return 0
        return None

    @property
    def action_space_size(self):
        return self.ROWS * self.COLS

    @property
    def state_size(self):
        return self.ROWS * self.COLS
