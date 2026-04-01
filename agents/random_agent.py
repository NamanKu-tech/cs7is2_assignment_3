"""Fully random agent — picks uniformly from valid actions."""

import random


class RandomAgent:
    name = "Random"

    def __init__(self, player=1):
        self.player = player

    def choose_move(self, game, training=False):
        return random.choice(game.get_valid_actions())
