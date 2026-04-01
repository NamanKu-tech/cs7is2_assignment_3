"""
Default (semi-intelligent) opponent.
Priority: win if possible → block opponent win → prefer centre → random.
Works for both TicTacToe and Connect4.
"""

import random
from games.tictactoe import TicTacToe
from games.connect4 import Connect4


class DefaultAgent:
    name = "Default"

    # TTT preferred squares: centre → corners → edges
    TTT_PREFERENCE = [4, 0, 2, 6, 8, 1, 3, 5, 7]

    def __init__(self, player=1):
        self.player = player

    def choose_move(self, game, training=False):
        valid = game.get_valid_actions()
        me = game.current_player
        opp = -me

        # 1. Win immediately if possible
        win = self._find_winning_move(game, me, valid)
        if win is not None:
            return win

        # 2. Block opponent's immediate win
        block = self._find_winning_move(game, opp, valid)
        if block is not None:
            return block

        # 3. Positional preference
        if isinstance(game, TicTacToe):
            for a in self.TTT_PREFERENCE:
                if a in valid:
                    return a
        elif isinstance(game, Connect4):
            for c in Connect4.CENTER_ORDER:
                if c in valid:
                    return c

        return random.choice(valid)

    def _find_winning_move(self, game, player, valid_actions):
        """Return an action that immediately wins for `player`, or None."""
        saved_player = game.current_player
        for action in valid_actions:
            game.current_player = player
            g = game.clone()
            g.make_move(action)
            winner = g.get_winner()
            if winner == player:
                game.current_player = saved_player
                return action
        game.current_player = saved_player
        return None
