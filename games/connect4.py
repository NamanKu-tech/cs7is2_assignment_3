"""
Connect 4 environment.
Board: 6 rows x 7 cols numpy array.  Players: 1 and -1.  Empty: 0.
Actions: column indices 0-6.  Pieces fall to the lowest empty row.
"""

import numpy as np


class Connect4:
    ROWS = 6
    COLS = 7
    WIN_LEN = 4
    PLAYER_1 = 1
    PLAYER_2 = -1

    # Column preference for move ordering (center-first)
    CENTER_ORDER = [3, 2, 4, 1, 5, 0, 6]

    def __init__(self, move_limit=None):
        self.board = np.zeros((self.ROWS, self.COLS), dtype=np.int8)
        self.current_player = self.PLAYER_1
        self._heights = [0] * self.COLS  # pieces dropped per column
        self._winner = None
        self.move_limit = move_limit   # if set, game is forced draw after this many moves
        self._move_count = 0

    # ------------------------------------------------------------------ #
    # Core interface                                                        #
    # ------------------------------------------------------------------ #

    def reset(self):
        self.board[:] = 0
        self.current_player = self.PLAYER_1
        self._heights = [0] * self.COLS
        self._winner = None
        self._move_count = 0
        return self

    def clone(self):
        g = Connect4(move_limit=self.move_limit)
        g.board = self.board.copy()
        g.current_player = self.current_player
        g._heights = self._heights.copy()
        g._winner = self._winner
        g._move_count = self._move_count
        return g

    def get_valid_actions(self):
        """Return list of valid column indices."""
        return [c for c in range(self.COLS) if self._heights[c] < self.ROWS]

    def make_move(self, col):
        """Drop current player's piece in col.  Returns (reward, done)."""
        assert col in self.get_valid_actions(), f"Column {col} is full or invalid"
        row = self.ROWS - 1 - self._heights[col]
        self.board[row, col] = self.current_player
        self._heights[col] += 1
        self._move_count += 1
        # Move limit: forced draw when total moves exceed cap
        if self.move_limit is not None and self._move_count >= self.move_limit:
            self._winner = 0
            self.current_player *= -1
            return 0.0, True
        winner = self._check_winner(row, col)
        if winner is not None:
            self._winner = winner
            done = True
            reward = 1.0 if winner == self.current_player else (0.0 if winner == 0 else -1.0)
        else:
            done = False
            reward = 0.0
        self.current_player *= -1
        return reward, done

    def undo_move(self, col):
        self._heights[col] -= 1
        row = self.ROWS - 1 - self._heights[col]
        self.board[row, col] = 0
        self.current_player *= -1
        self._winner = None
        self._move_count -= 1

    def is_terminal(self):
        return self.get_winner() is not None

    def get_winner(self):
        """Returns 1, -1, 0 (draw), or None (ongoing)."""
        # Check for a winner by scanning the whole board
        for player in (1, -1):
            if self._has_won(player):
                return player
        if all(self._heights[c] >= self.ROWS for c in range(self.COLS)):
            return 0  # draw
        return None

    # ------------------------------------------------------------------ #
    # State representations                                                #
    # ------------------------------------------------------------------ #

    def get_state_key(self, perspective=None):
        flat = self.board.flatten()
        if perspective is not None:
            flat = flat * perspective
        return tuple(flat)

    def get_state_array(self, perspective=None):
        flat = self.board.flatten().astype(np.float32)
        if perspective is not None:
            flat = flat * perspective
        return flat

    # ------------------------------------------------------------------ #
    # Heuristic evaluation (for depth-limited minimax)                     #
    # ------------------------------------------------------------------ #

    def evaluate_heuristic(self, player):
        """
        Board evaluation from `player`'s perspective.
        Scores candidate 4-windows based on piece counts.
        """
        score = 0.0
        opponent = -player

        # Center column bonus
        center_col = self.board[:, self.COLS // 2]
        score += 3.0 * int(np.sum(center_col == player))

        # Score all windows of length 4
        for window in self._get_windows():
            p_count = int(np.sum(window == player))
            o_count = int(np.sum(window == opponent))
            e_count = int(np.sum(window == 0))

            if o_count == 0:
                if p_count == 4:
                    score += 100.0
                elif p_count == 3 and e_count == 1:
                    score += 5.0
                elif p_count == 2 and e_count == 2:
                    score += 2.0
            elif p_count == 0:
                if o_count == 4:
                    score -= 100.0
                elif o_count == 3 and e_count == 1:
                    score -= 4.0
                elif o_count == 2 and e_count == 2:
                    score -= 1.0

        return score

    def _get_windows(self):
        """Yield all length-4 windows (horizontal, vertical, diagonal)."""
        b = self.board
        # Horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                yield b[r, c:c + 4]
        # Vertical
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                yield b[r:r + 4, c]
        # Diagonal down-right
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                yield np.array([b[r + i, c + i] for i in range(4)])
        # Diagonal down-left
        for r in range(self.ROWS - 3):
            for c in range(3, self.COLS):
                yield np.array([b[r + i, c - i] for i in range(4)])

    # ------------------------------------------------------------------ #
    # Rendering                                                            #
    # ------------------------------------------------------------------ #

    def render(self):
        symbols = {0: ".", 1: "X", -1: "O"}
        lines = []
        lines.append("  " + " ".join(str(c) for c in range(self.COLS)))
        lines.append("  " + "-" * (self.COLS * 2 - 1))
        for r in range(self.ROWS):
            row_str = "| " + " ".join(symbols[self.board[r, c]] for c in range(self.COLS)) + " |"
            lines.append(row_str)
        lines.append("  " + "-" * (self.COLS * 2 - 1))
        lines.append("  " + " ".join(str(c) for c in range(self.COLS)))
        return "\n".join(lines)

    def print_board(self):
        print(self.render())
        print()

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _check_winner(self, last_row, last_col):
        """Fast winner check only around the last placed piece."""
        player = self.board[last_row, last_col]
        if player == 0:
            return None

        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            for sign in (1, -1):
                r, c = last_row + sign * dr, last_col + sign * dc
                while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r, c] == player:
                    count += 1
                    r += sign * dr
                    c += sign * dc
            if count >= self.WIN_LEN:
                return player

        if all(self._heights[c] >= self.ROWS for c in range(self.COLS)):
            return 0
        return None

    def _has_won(self, player):
        b = self.board
        # Horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                if b[r, c] == b[r, c+1] == b[r, c+2] == b[r, c+3] == player:
                    return True
        # Vertical
        for r in range(self.ROWS - 3):
            for c in range(self.COLS):
                if b[r, c] == b[r+1, c] == b[r+2, c] == b[r+3, c] == player:
                    return True
        # Diagonals
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                if b[r, c] == b[r+1, c+1] == b[r+2, c+2] == b[r+3, c+3] == player:
                    return True
        for r in range(self.ROWS - 3):
            for c in range(3, self.COLS):
                if b[r, c] == b[r+1, c-1] == b[r+2, c-2] == b[r+3, c-3] == player:
                    return True
        return False

    @property
    def action_space_size(self):
        return self.COLS

    @property
    def state_size(self):
        return self.ROWS * self.COLS
